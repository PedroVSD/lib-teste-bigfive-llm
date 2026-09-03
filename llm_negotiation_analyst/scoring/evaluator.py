"""
LLM-as-judge evaluator — categorical, one call per turn.

Behavioral metrics (Big Five + NegotiationMetric) são avaliadas por turno como:
  PRESENT | ABSENT | NOT_APPLICABLE  + evidence

Unidade de avaliação: resposta completa do agente em um turno (não frase).
Para cada turno, o juiz recebe: contexto do cenário + histórico da negociação
+ resposta completa do turno + rubricas de TODAS as métricas aplicáveis.
Em uma única chamada, retorna avaliações para todas as métricas.

Agregação: occurrence_rate = PRESENT / (PRESENT + ABSENT)  (NOT_APPLICABLE ignorado)
Utility (0-1) e Satisfaction (1-7) permanecem separados.
Agreement = AGREEMENT | NO_AGREEMENT.
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
# Prompt templates — batch per turn with history
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_BATCH = """You are an expert researcher in behavioral economics, psychology, and negotiation science. Your task is to evaluate a single TURN response (complete agent utterance, not individual sentences) from a negotiation transcript according to multiple behavioral metrics.

You will be given:
  - The negotiation context (scenario description)
  - The negotiation history (previous turns, in order) — use this to interpret behaviors that depend on history (e.g., anchoring, conditional_concession)
  - The speaker role
  - The complete response for the current turn
  - The list of metrics to evaluate, each with its poles and behavioral anchors for PRESENT and ABSENT

Rules:
  - Evaluate the CURRENT turn response as a single unit, considering the provided history/context.
  - For anchoring, conditional_concession and other history-dependent metrics, base your judgment on the current response IN CONTEXT of the history (e.g., an anchor is judged relative to prior offers).
  - For each metric, return PRESENT (behavior clearly present in this turn), ABSENT (opportunity existed but behavior absent), or NOT_APPLICABLE (insufficient opportunity to observe — e.g., very short or irrelevant utterance). NOT_APPLICABLE must NOT be treated as ABSENT.
  - Evidence must be a short plain text paraphrase (1 sentence, max 120 chars, no double quotes, no newlines) from the CURRENT turn only, justifying the classification. Use "" if NOT_APPLICABLE.
  - Respond ONLY with a valid JSON object. No markdown fences, no preamble. Ensure JSON is valid: double quotes around keys/values, no trailing commas, escape any inner double quotes.

Example:
{"evaluations": {"anchoring": {"result": "PRESENT", "evidence": "proposes R$ 18.000 as initial anchor"}, "rapport": {"result": "ABSENT", "evidence": "no empathetic language"}, "clarity": {"result": "NOT_APPLICABLE", "evidence": ""}}}

JSON schema:
{
  "evaluations": {
    "<metric_id>": {"result": "PRESENT" | "ABSENT" | "NOT_APPLICABLE", "evidence": "<short evidence>"},
    ...
  }
}
You must provide an entry for EVERY metric listed. Do not omit metrics."""

# Legacy single-metric templates kept for backward compat (not used in batch flow)
_JUDGE_SYSTEM = _JUDGE_SYSTEM_BATCH
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


def _build_metrics_block(metrics: list[AnyMetric]) -> str:
    lines = []
    for m in metrics:
        meta = ALL_METRICS_META[m]
        anchor_present = meta.behavioral_anchors.get("present") or meta.behavioral_anchors.get("enabled") or ""
        anchor_absent = meta.behavioral_anchors.get("absent") or meta.behavioral_anchors.get("disabled") or ""
        # Truncate anchors to keep prompt small (model gpt-oss:120b fails with long prompts)
        if len(anchor_present) > 180:
            anchor_present = anchor_present[:177] + "..."
        if len(anchor_absent) > 180:
            anchor_absent = anchor_absent[:177] + "..."
        mid = m.value if hasattr(m, "value") else str(m)
        lines.append(f"### {mid} — {meta.name} ({meta.high_pole} ↔ {meta.low_pole})")
        lines.append(f"PRESENT — {anchor_present}")
        lines.append(f"ABSENT — {anchor_absent}")
        lines.append("")
    return "\n".join(lines)


def _format_history(transcript: list[dict], current_index: int, window: int = 5) -> str:
    """Format negotiation history up to (but not including) current turn. Windowed to avoid token blowup."""
    if not transcript or current_index == 0:
        return "(No prior history — this is the first turn.)"
    start = max(0, current_index - window)
    parts = []
    if start > 0:
        parts.append(f"[... {start} earlier turns omitted ...]")
    for i in range(start, current_index):
        t = transcript[i]
        role = t.get("role", "?")
        content = (t.get("content") or "").strip().replace("\n", " ")
        if len(content) > 300:
            content = content[:297] + "..."
        parts.append(f"Turn {i} — {role}: {content}")
    return "\n".join(parts) if parts else "(No prior history.)"


_JUDGE_USER_BATCH = """## Negotiation Context
{scenario_context}

## Negotiation History (for context, do NOT evaluate these turns — only the current turn)
{history_text}

## Speaker Role (current turn)
{role}

## Current Turn — Complete Response to Evaluate (Turn {turn_index})
\"\"\"{utterance}\"\"\"

## Metrics to Evaluate ({n_metrics} metrics)
{metrics_block}
Evaluate the CURRENT turn response as a single unit, considering the history/context above.
Return JSON only with an entry for every metric:
{{"evaluations": {{"<metric_id>": {{"result": "PRESENT|ABSENT|NOT_APPLICABLE", "evidence": "..."}}, ... }}}}"""


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
    Usa um LLM-juiz para observar respostas completas por turno (uma chamada por turno para todas as métricas).

    Args:
        judge: LLMAdapter do juiz (separado dos agentes negociadores).
        config: EvaluatorConfig — define quais métricas avaliar.
        second_judge: Segundo juiz opcional para cálculo de IRR por turno (agreement).
        history_window: Número de turnos prévios incluídos como contexto (default 8).
    """

    def __init__(
        self,
        judge: LLMAdapter,
        config: Optional[EvaluatorConfig] = None,
        second_judge: Optional[LLMAdapter] = None,
        history_window: int = 8,
    ):
        self.judge = judge
        self.config = config or EvaluatorConfig()
        self.second_judge = second_judge
        self.history_window = history_window

    # ------------------------------------------------------------------
    # API pública — batch per turn
    # ------------------------------------------------------------------

    def evaluate_turn(
        self,
        utterance: str,
        role: str,
        scenario_context: str,
        turn_index: int,
        dimensions: Optional[list[AnyMetric]] = None,
        history: Optional[list[dict]] = None,
        transcript: Optional[list[dict]] = None,
    ) -> list[BehaviorObservation]:
        """
        Avalia a resposta completa de um turno em todas as métricas configuradas com UMA chamada ao juiz.
        `history` ou `transcript` fornecem contexto; se ambos forem None, avalia sem histórico.
        """
        metrics = dimensions or self.config.dimensions
        # Build history_text
        if history is not None:
            # history is list of prior turns dicts
            # Convert to transcript-like for formatting
            fake_transcript = history + [{"role": role, "agent_id": role, "content": utterance}]
            # current_index is len(history)
            history_text = _format_history(fake_transcript, len(history), window=self.history_window)
        elif transcript is not None:
            history_text = _format_history(transcript, turn_index, window=self.history_window)
        else:
            history_text = "(No prior history provided.)"

        observations = self._observe_batch(
            utterance=utterance,
            role=role,
            scenario_context=scenario_context,
            turn_index=turn_index,
            metrics=metrics,
            history_text=history_text,
            judge=self.judge,
        )
        if self.second_judge:
            obs2_list = self._observe_batch(
                utterance=utterance,
                role=role,
                scenario_context=scenario_context,
                turn_index=turn_index,
                metrics=metrics,
                history_text=history_text,
                judge=self.second_judge,
            )
            # Map second judge results by metric
            map2 = {o.dimension: o for o in obs2_list}
            for obs in observations:
                obs2 = map2.get(obs.dimension)
                if obs2:
                    obs.confidence = self._irr(obs.result, obs2.result)
        return observations

    def evaluate_transcript(
        self,
        transcript: list[dict],
        agent_roles: dict[str, str],
        scenario_context: str,
    ) -> dict[str, Big5Profile]:
        """
        Pontua o transcript completo e retorna um Big5Profile por agente.
        Para cada turno, faz UMA chamada ao juiz com todas as métricas + histórico.
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
            # One call per turn for all metrics, with history
            turn_obs = self.evaluate_turn(
                utterance=content,
                role=agent_role,
                scenario_context=scenario_context,
                turn_index=i,
                transcript=transcript,
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
                profile.scores[metric] = occurrence_rate

        return profiles

    # ------------------------------------------------------------------
    # Helpers internos — batch
    # ------------------------------------------------------------------

    def _observe_batch(
        self,
        utterance: str,
        role: str,
        scenario_context: str,
        turn_index: int,
        metrics: list[AnyMetric],
        history_text: str,
        judge: LLMAdapter,
    ) -> list[BehaviorObservation]:
        metrics_block = _build_metrics_block(metrics)
        prompt = _JUDGE_USER_BATCH.format(
            scenario_context=scenario_context,
            role=role,
            history_text=history_text,
            utterance=utterance,
            turn_index=turn_index,
            n_metrics=len(metrics),
            metrics_block=metrics_block,
        )
        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM_BATCH},
            {"role": "user", "content": prompt},
        ]
        try:
            raw = judge.complete(messages)
            parsed = self._parse_json(raw)
            # Extract evaluations dict
            evals = None
            if "evaluations" in parsed and isinstance(parsed["evaluations"], dict):
                evals = parsed["evaluations"]
            else:
                # Support flat format where keys are metric ids directly
                # Detect if parsed has metric ids as top-level keys with {result, evidence}
                possible = {k: v for k, v in parsed.items() if k not in ("evaluations",) and isinstance(v, dict) and "result" in v}
                if possible:
                    evals = possible
                else:
                    # Legacy single-metric format
                    if "result" in parsed and "metric" in parsed:
                        evals = {str(parsed["metric"]): {"result": parsed["result"], "evidence": parsed.get("evidence")}}
                    elif "result" in parsed:
                        # ambiguous — treat as first metric
                        first = metrics[0].value if metrics else "unknown"
                        evals = {first: {"result": parsed["result"], "evidence": parsed.get("evidence")}}

            if evals is None:
                raise ValueError(f"No evaluations found in judge response: {raw[:500]}")

            observations: list[BehaviorObservation] = []
            for metric in metrics:
                mid = metric.value if hasattr(metric, "value") else str(metric)
                # Try exact, then lowercased
                entry = evals.get(mid) or evals.get(mid.lower()) or evals.get(mid.upper())
                # Fallback: case-insensitive search
                if entry is None:
                    for k, v in evals.items():
                        if k.lower() == mid.lower():
                            entry = v
                            break
                if entry is None:
                    logger.warning("Judge response missing metric '%s' turn=%d — marking NOT_APPLICABLE", mid, turn_index)
                    result = BehavioralResult.NOT_APPLICABLE
                    evidence = f"[Missing metric '{mid}' in judge response]"
                    conf = 0.0
                else:
                    if not isinstance(entry, dict):
                        raise ValueError(f"Invalid entry for metric '{mid}': {entry}")
                    result_raw = str(entry.get("result", "")).strip().upper()
                    if result_raw not in ("PRESENT", "ABSENT", "NOT_APPLICABLE"):
                        raise ValueError(f"Invalid result '{result_raw}' for metric '{mid}'")
                    result = BehavioralResult(result_raw)
                    evidence = entry.get("evidence")
                    if evidence is None:
                        evidence = ""
                    evidence = str(evidence).strip()
                    if not evidence and result != BehavioralResult.NOT_APPLICABLE:
                        evidence = "(no evidence provided)"
                    elif not evidence:
                        evidence = ""
                    conf = float(entry.get("confidence", 1.0)) if "confidence" in entry else 1.0
                observations.append(BehaviorObservation(
                    dimension=metric,
                    result=result,
                    evidence=evidence,
                    turn_index=turn_index,
                    confidence=conf,
                ))
            return observations
        except Exception as e:
            logger.warning("Judge batch failed turn=%d: %s — falling back to per-metric", turn_index, e)
            # Fallback: try per-metric single calls (degrades to N*M but ensures data)
            fallback_obs = []
            for m in metrics:
                try:
                    obs = self._observe_one_legacy(utterance, role, scenario_context, turn_index, m, judge)
                    fallback_obs.append(obs)
                except Exception as e2:
                    logger.warning("Fallback single failed for %s turn=%d: %s", m, turn_index, e2)
                    fallback_obs.append(BehaviorObservation(
                        dimension=m,
                        result=BehavioralResult.NOT_APPLICABLE,
                        evidence=f"[Evaluation failed: {e2}]",
                        turn_index=turn_index,
                        confidence=0.0,
                    ))
            if fallback_obs:
                return fallback_obs
            return [
                BehaviorObservation(
                    dimension=m,
                    result=BehavioralResult.NOT_APPLICABLE,
                    evidence=f"[Evaluation failed: {e}]",
                    turn_index=turn_index,
                    confidence=0.0,
                ) for m in metrics
            ]

    def _observe_one_legacy(
        self,
        utterance: str,
        role: str,
        scenario_context: str,
        turn_index: int,
        metric: AnyMetric,
        judge: LLMAdapter,
    ) -> BehaviorObservation:
        """Legacy single-metric judge call (used as fallback)."""
        meta = ALL_METRICS_META[metric]
        anchor_present = meta.behavioral_anchors.get("present") or meta.behavioral_anchors.get("enabled") or ""
        anchor_absent = meta.behavioral_anchors.get("absent") or meta.behavioral_anchors.get("disabled") or ""
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
        # _JUDGE_USER expects anchor_present/absent but legacy template uses anchor_present/absent; keep compat
        # Actually _JUDGE_USER is legacy single metric with anchor_present/absent
        # Build messages with batch system but single metric user
        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM_BATCH},
            {"role": "user", "content": prompt},
        ]
        raw = judge.complete(messages)
        parsed = self._parse_json(raw)
        # Handle both batch and single formats
        if "result" in parsed:
            result_raw = str(parsed.get("result", "")).strip().upper()
            if result_raw not in ("PRESENT", "ABSENT", "NOT_APPLICABLE"):
                raise ValueError(f"Invalid result '{result_raw}'")
            result = BehavioralResult(result_raw)
            evidence = str(parsed.get("evidence", "")).strip() or "(no evidence)"
            return BehaviorObservation(dimension=metric, result=result, evidence=evidence, turn_index=turn_index, confidence=float(parsed.get("confidence", 1.0)))
        elif "evaluations" in parsed and isinstance(parsed["evaluations"], dict):
            mid = metric.value if hasattr(metric, "value") else str(metric)
            entry = parsed["evaluations"].get(mid) or next(iter(parsed["evaluations"].values()), None)
            if isinstance(entry, dict) and "result" in entry:
                result = BehavioralResult(str(entry["result"]).strip().upper())
                evidence = str(entry.get("evidence", "")).strip() or ""
                return BehaviorObservation(dimension=metric, result=result, evidence=evidence, turn_index=turn_index, confidence=float(entry.get("confidence", 1.0)))
        raise ValueError(f"No valid result for metric {metric} in {raw[:300]}")

    # Public alias kept for backward compat
    def _observe_one(
        self,
        utterance: str,
        role: str,
        scenario_context: str,
        turn_index: int,
        metric: AnyMetric,
        judge: LLMAdapter,
    ) -> BehaviorObservation:
        return self._observe_one_legacy(utterance, role, scenario_context, turn_index, metric, judge)

    @staticmethod
    def _parse_json(raw: str) -> dict:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        start = clean.find("{")
        if start == -1:
            raise ValueError(f"No JSON object found in judge response: {raw[:300]}")
        # Try raw_decode first
        decoder = json.JSONDecoder()
        try:
            obj, _ = decoder.raw_decode(clean[start:])
            return obj
        except json.JSONDecodeError as e:
            # Try to repair common issues: trailing commas, single quotes, unescaped newlines in evidence
            candidate = clean[start:]
            # Find last } to trim trailing garbage
            end = candidate.rfind("}")
            if end != -1:
                candidate = candidate[:end+1]
            # Remove trailing commas before } or ]
            candidate = re.sub(r",\s*}", "}", candidate)
            candidate = re.sub(r",\s*]", "]", candidate)
            # Try again
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                # Try replacing single quotes with double quotes if no double quotes present
                if "'" in candidate and '"' not in candidate:
                    try:
                        return json.loads(candidate.replace("'", '"'))
                    except:
                        pass
                # Last attempt: extract evaluations block via regex
                raise e

    @staticmethod
    def _irr(result1: BehavioralResult, result2: BehavioralResult) -> float:
        """IRR categorical: 1.0 if agree, 0.0 if disagree (NA vs anything = 0.5 if one NA)."""
        if result1 == result2:
            return 1.0
        if BehavioralResult.NOT_APPLICABLE in (result1, result2):
            return 0.5
        return 0.0
