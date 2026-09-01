"""
LLM-as-judge evaluator — categorical.

Behavioral metrics (Big Five + NegotiationMetric) são avaliadas por turno como:
  PRESENT | ABSENT | NOT_APPLICABLE  + evidence

Agregação: occurrence_rate = PRESENT / (PRESENT + ABSENT)  (NOT_APPLICABLE ignorado)
Utility (0-1) e Satisfaction (1-7) permanecem separados e contínuos/ordinais.
Agreement = AGREEMENT | NO_AGREEMENT.

Judge retorna JSON {metric, result, evidence}.
"""

import json
import re
import logging
from dataclasses import dataclass, field
from typing import Optional, Union

from ..adapters.base import LLMAdapter
from .big5 import Dimension, BehavioralResult, BehaviorObservation, BehaviorSummary, Big5Profile, BIG5_META
from .negotiation_metrics import NegotiationMetric, NEGOTIATION_META

logger = logging.getLogger(__name__)

AnyMetric = Union[Dimension, NegotiationMetric]
ALL_METRICS_META = {**BIG5_META, **NEGOTIATION_META}

# Outcome agreement (not behavioral)
class AgreementResult(str, __import__("enum").Enum):
    AGREEMENT = "AGREEMENT"
    NO_AGREEMENT = "NO_AGREEMENT"


def resolve_metric(value: str) -> AnyMetric:
    try:
        return Dimension(value)
    except ValueError:
        pass
    try:
        return NegotiationMetric(value)
    except ValueError:
        pass
    raise ValueError(
        f"Métrica desconhecida: '{value}'. "
        f"Big Five válidos: {[d.value for d in Dimension]}. "
        f"Métricas de negociação válidas: {[m.value for m in NegotiationMetric]}."
    )


def is_big5(metric: AnyMetric) -> bool:
    return isinstance(metric, Dimension)


def is_negotiation_metric(metric: AnyMetric) -> bool:
    return isinstance(metric, NegotiationMetric)


# ---------------------------------------------------------------------------
# Prompt templates — categorical
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """You are an expert researcher in behavioral economics, psychology, and negotiation science. Your task is to evaluate a single utterance from a negotiation transcript according to the specific behavioral metric provided.

You will be given:
  - The negotiation context (scenario description)
  - The role of the speaker
  - The metric name and its poles (present vs absent)
  - Behavioral anchors for PRESENT and ABSENT
  - The utterance to evaluate

Rules:
  - Base your judgment ONLY on observable behavior in THIS turn, not on a general impression of the agent.
  - Use NOT_APPLICABLE when there was insufficient opportunity to observe the behavior in this turn (e.g., very short utterance, no relevant content). NOT_APPLICABLE must NOT be treated as ABSENT.
  - Evidence must be a short quote or paraphrase (1 sentence) justifying the classification.

You must respond ONLY with a valid JSON object. No markdown fences.

JSON schema:
{
  "metric": "<metric id>",
  "result": "PRESENT" | "ABSENT" | "NOT_APPLICABLE",
  "evidence": "<short textual evidence>"
}"""

_JUDGE_USER = """## Negotiation Context
{scenario_context}

## Speaker Role
{role}

## Metric: {metric_name} ({high_pole} ↔ {low_pole})

### Behavioral Anchors
PRESENT — {anchor_present}
ABSENT — {anchor_absent}
NOT_APPLICABLE — insufficient opportunity to observe this behavior in this turn (short/irrelevant utterance)

## Utterance to Evaluate (Turn {turn_index})
\"\"\"{utterance}\"\"\"

Evaluate this utterance on the metric "{metric_name}". Respond with JSON only: {{"metric": "{metric_id}", "result": "PRESENT|ABSENT|NOT_APPLICABLE", "evidence": "..."}}."""


# ---------------------------------------------------------------------------
# EvaluatorConfig
# ---------------------------------------------------------------------------

@dataclass
class EvaluatorConfig:
    dimensions: list[AnyMetric] = field(default_factory=list)
    # legacy compat: score_per_turn / invert_neuroticism kept but ignored
    score_per_turn: bool = True
    invert_neuroticism: bool = False

    def __post_init__(self):
        if not self.dimensions:
            self.dimensions = list(Dimension)

    @classmethod
    def from_strings(cls, metric_strings: list[str], **kwargs) -> "EvaluatorConfig":
        resolved = []
        for s in metric_strings:
            try:
                resolved.append(resolve_metric(s))
            except ValueError as e:
                logger.warning("EvaluatorConfig.from_strings: %s — ignorando.", e)
        return cls(dimensions=resolved or list(Dimension), **kwargs)

    @property
    def big5_dimensions(self) -> list[Dimension]:
        return [d for d in self.dimensions if is_big5(d)]

    @property
    def negotiation_metrics(self) -> list[NegotiationMetric]:
        return [d for d in self.dimensions if is_negotiation_metric(d)]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Usa um LLM-juiz para observar utterances de forma categórica.

    Args:
        judge: LLMAdapter do juiz (separado dos agentes negociadores).
        config: EvaluatorConfig — define quais métricas avaliar.
        second_judge: Segundo juiz opcional para cálculo de IRR por turno (agreement).
    """

    def __init__(
        self,
        judge: LLMAdapter,
        config: Optional[EvaluatorConfig] = None,
        second_judge: Optional[LLMAdapter] = None,
    ):
        self.judge = judge
        self.config = config or EvaluatorConfig()
        self.second_judge = second_judge

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def evaluate_turn(
        self,
        utterance: str,
        role: str,
        scenario_context: str,
        turn_index: int,
        dimensions: Optional[list[AnyMetric]] = None,
    ) -> list[BehaviorObservation]:
        metrics = dimensions or self.config.dimensions
        observations = []
        for metric in metrics:
            obs = self._observe_one(utterance, role, scenario_context, turn_index, metric, self.judge)
            if self.second_judge:
                obs2 = self._observe_one(utterance, role, scenario_context, turn_index, metric, self.second_judge)
                obs.confidence = self._irr(obs.result, obs2.result)
            observations.append(obs)
        return observations

    def evaluate_transcript(
        self,
        transcript: list[dict],
        agent_roles: dict[str, str],
        scenario_context: str,
    ) -> dict[str, Big5Profile]:
        """
        Pontua o transcript completo e retorna um Big5Profile por agente.
        Agregação: occurrence_rate = PRESENT / (PRESENT+ABSENT).
        """
        profiles: dict[str, Big5Profile] = {}

        for agent_id, role in agent_roles.items():
            profiles[agent_id] = Big5Profile(
                agent_id=agent_id,
                model_identifier=agent_id,
            )

        for i, turn in enumerate(transcript):
            role = turn.get("role")
            content = turn.get("content", "")
            agent_id = turn.get("agent_id", role)
            if agent_id not in profiles:
                continue
            agent_role = agent_roles.get(agent_id, role)
            turn_obs = self.evaluate_turn(
                utterance=content,
                role=agent_role,
                scenario_context=scenario_context,
                turn_index=i,
            )
            profiles[agent_id].observations.extend(turn_obs)

        # Agrega: counts + occurrence_rate por métrica
        for agent_id, profile in profiles.items():
            for metric in self.config.dimensions:
                obs_for_metric = [o for o in profile.observations if o.dimension == metric]
                present = sum(1 for o in obs_for_metric if o.result == BehavioralResult.PRESENT)
                absent = sum(1 for o in obs_for_metric if o.result == BehavioralResult.ABSENT)
                na = sum(1 for o in obs_for_metric if o.result == BehavioralResult.NOT_APPLICABLE)
                total_applicable = present + absent
                occurrence_rate = (present / total_applicable) if total_applicable > 0 else None
                summary = BehaviorSummary(
                    dimension=metric,
                    present=present,
                    absent=absent,
                    not_applicable=na,
                    occurrence_rate=occurrence_rate,
                    total_applicable=total_applicable,
                )
                profile.summaries[metric] = summary
                # legacy scores kept as occurrence_rate for compat (0-1)
                profile.scores[metric] = occurrence_rate

        return profiles

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _observe_one(
        self,
        utterance: str,
        role: str,
        scenario_context: str,
        turn_index: int,
        metric: AnyMetric,
        judge: LLMAdapter,
    ) -> BehaviorObservation:
        meta = ALL_METRICS_META[metric]
        # anchors are now present/absent (fallback for legacy)
        anchor_present = meta.behavioral_anchors.get("present") or meta.behavioral_anchors.get("enabled") or meta.behavioral_anchors.get(5) or ""
        anchor_absent = meta.behavioral_anchors.get("absent") or meta.behavioral_anchors.get("disabled") or meta.behavioral_anchors.get(1) or ""
        prompt = _JUDGE_USER.format(
            scenario_context=scenario_context,
            role=role,
            metric_name=meta.name,
            high_pole=meta.high_pole,
            low_pole=meta.low_pole,
            anchor_present=anchor_present,
            anchor_absent=anchor_absent,
            turn_index=turn_index,
            utterance=utterance,
            metric_id=metric.value if hasattr(metric, "value") else str(metric),
        )
        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        try:
            raw = judge.complete(messages)
            parsed = self._parse_json(raw)
            result_raw = str(parsed.get("result", "")).strip().upper()
            if result_raw not in ("PRESENT", "ABSENT", "NOT_APPLICABLE"):
                raise ValueError(f"Invalid result '{result_raw}' — expected PRESENT|ABSENT|NOT_APPLICABLE")
            result = BehavioralResult(result_raw)
            evidence = str(parsed.get("evidence", "")).strip()
            if not evidence:
                evidence = "(no evidence provided)"
            return BehaviorObservation(
                dimension=metric,
                result=result,
                evidence=evidence,
                turn_index=turn_index,
                confidence=float(parsed.get("confidence", 1.0)),
            )
        except Exception as e:
            logger.warning("Judge failed for dim=%s turn=%d: %s", metric, turn_index, e)
            return BehaviorObservation(
                dimension=metric,
                result=BehavioralResult.NOT_APPLICABLE,
                evidence=f"[Evaluation failed: {e}]",
                turn_index=turn_index,
                confidence=0.0,
            )

    @staticmethod
    def _parse_json(raw: str) -> dict:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        start = clean.find("{")
        if start == -1:
            raise ValueError(f"No JSON object found in judge response: {raw[:300]}")
        decoder = json.JSONDecoder()
        try:
            obj, _ = decoder.raw_decode(clean[start:])
            return obj
        except json.JSONDecodeError:
            end = clean.rfind("}")
            if end != -1 and end > start:
                return json.loads(clean[start:end + 1])
            raise

    @staticmethod
    def _irr(result1: BehavioralResult, result2: BehavioralResult) -> float:
        """IRR categorical: 1.0 if agree, 0.0 if disagree (NA vs anything = 0.5 if one NA)."""
        if result1 == result2:
            return 1.0
        if BehavioralResult.NOT_APPLICABLE in (result1, result2):
            return 0.5
        return 0.0
