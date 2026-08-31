"""
LLM-as-judge evaluator.

Avalia utterances de negociação em duas famílias de métricas:

  Big Five (Dimension)          — traços de personalidade OCEAN
  NegotiationMetric             — táticas, emoções, argumentação e vieses

Ambas as famílias usam a mesma estrutura DimensionMeta e o mesmo
pipeline de avaliação via LLM-as-judge com rubricas estruturadas.

Design decisions:
  - Judge separado dos agentes para evitar viés de auto-avaliação.
  - Cada turno é pontuado independentemente; score final = média.
  - Judge retorna JSON estruturado; fallback em caso de falha.
  - Dual-judge opcional: IRR calculado e armazenado por turno.
"""

import json
import re
import logging
from dataclasses import dataclass, field
from typing import Optional, Union

from ..adapters.base import LLMAdapter
from .big5 import Dimension, DimensionScore, Big5Profile, BIG5_META
from .negotiation_metrics import NegotiationMetric, NEGOTIATION_META

logger = logging.getLogger(__name__)

# Tipo unificado para qualquer métrica avaliável
AnyMetric = Union[Dimension, NegotiationMetric]

# Registro unificado de metadados — lookup único para o evaluator
ALL_METRICS_META = {**BIG5_META, **NEGOTIATION_META}


# ---------------------------------------------------------------------------
# Helpers de resolução de string → enum
# ---------------------------------------------------------------------------

def resolve_metric(value: str) -> AnyMetric:
    """
    Converte uma string (ex: 'agreeableness', 'anchoring') para o enum correto.
    Tenta Big Five primeiro, depois NegotiationMetric.

    Raises:
        ValueError: se a string não corresponde a nenhuma métrica conhecida.
    """
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
# Prompt templates
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """You are an expert researcher in behavioral economics, psychology, and \
negotiation science. Your task is to evaluate a single utterance from a \
negotiation transcript according to the specific metric provided.

You will be given:
  - The negotiation context (scenario description)
  - The role of the speaker (e.g., buyer, seller, employer, candidate)
  - The metric name and its high/low poles
  - Behavioral anchors (either 1/3/5 scale or DISABLED/ENABLED binary)
  - The utterance to evaluate

You must respond ONLY with a valid JSON object. Do not include markdown fences, \
preamble, or explanation outside the JSON.

JSON schema:
{
  "score": <integer 1–5>,
  "justification": "<1–3 sentence explanation referencing specific language from the utterance>",
  "confidence": <float 0.0–1.0>
}"""

_JUDGE_USER = """## Negotiation Context
{scenario_context}

## Speaker Role
{role}

## Metric: {metric_name} ({high_pole} ↔ {low_pole})

### Scoring Anchors
1 — {anchor_1}
3 — {anchor_3}
5 — {anchor_5}

## Utterance to Evaluate (Turn {turn_index})
\"\"\"{utterance}\"\"\"

Evaluate this utterance on the metric "{metric_name}". Respond with JSON only."""

_JUDGE_USER_BINARY = """## Negotiation Context
{scenario_context}

## Speaker Role
{role}

## Metric: {metric_name} ({high_pole} ↔ {low_pole})

### Behavioral States
DISABLED — {anchor_disabled}
ENABLED — {anchor_enabled}

Scoring: score 1-2 = DISABLED (trait absent), 4-5 = ENABLED (trait present). Use 1 for clearly disabled, 5 for clearly enabled.

## Utterance to Evaluate (Turn {turn_index})
\"\"\"{utterance}\"\"\"

Evaluate this utterance on the metric "{metric_name}". Respond with JSON only."""


# ---------------------------------------------------------------------------
# EvaluatorConfig
# ---------------------------------------------------------------------------

@dataclass
class EvaluatorConfig:
    """
    Controla quais métricas o juiz vai avaliar.

    Aceita qualquer combinação de Dimension (Big Five) e NegotiationMetric.

    Exemplos:
        # Só Big Five
        EvaluatorConfig(dimensions=list(Dimension))

        # Só métricas de negociação
        EvaluatorConfig(dimensions=list(NegotiationMetric))

        # Misto
        EvaluatorConfig(dimensions=[
            Dimension.AGREEABLENESS,
            NegotiationMetric.ANCHORING,
            NegotiationMetric.RAPPORT,
        ])

        # A partir de strings (ex: lidas do config.yaml)
        EvaluatorConfig.from_strings(["agreeableness", "anchoring", "rapport"])
    """
    dimensions: list[AnyMetric] = field(default_factory=list)
    score_per_turn: bool = True
    invert_neuroticism: bool = False  # flip N: alto = mais estável

    def __post_init__(self):
        if not self.dimensions:
            # Default: apenas Big Five
            self.dimensions = list(Dimension)

    @classmethod
    def from_strings(cls, metric_strings: list[str], **kwargs) -> "EvaluatorConfig":
        """
        Cria EvaluatorConfig a partir de uma lista de strings.
        Strings inválidas são ignoradas com um warning.

        Uso típico (lendo do config.yaml):
            EvaluatorConfig.from_strings(["agreeableness", "anchoring", "rapport"])
        """
        resolved = []
        for s in metric_strings:
            try:
                resolved.append(resolve_metric(s))
            except ValueError as e:
                logger.warning("EvaluatorConfig.from_strings: %s — ignorando.", e)
        return cls(dimensions=resolved or list(Dimension), **kwargs)

    @property
    def big5_dimensions(self) -> list[Dimension]:
        """Retorna apenas as dimensões Big Five da configuração."""
        return [d for d in self.dimensions if is_big5(d)]

    @property
    def negotiation_metrics(self) -> list[NegotiationMetric]:
        """Retorna apenas as métricas de negociação da configuração."""
        return [d for d in self.dimensions if is_negotiation_metric(d)]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Usa um LLM-juiz para pontuar utterances de negociação.

    Args:
        judge:        LLMAdapter do juiz (separado dos agentes negociadores).
        config:       EvaluatorConfig — define quais métricas avaliar.
        second_judge: Segundo juiz opcional para cálculo de IRR por turno.
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
    ) -> list[DimensionScore]:
        """Pontua um único turno em todas as métricas configuradas."""
        metrics = dimensions or self.config.dimensions
        scores = []
        for metric in metrics:
            score = self._score_one(utterance, role, scenario_context, turn_index, metric, self.judge)
            if self.second_judge:
                score2 = self._score_one(utterance, role, scenario_context, turn_index, metric, self.second_judge)
                score.confidence = self._irr(score.score, score2.score)
            scores.append(score)
        return scores

    def evaluate_transcript(
        self,
        transcript: list[dict],
        agent_roles: dict[str, str],
        scenario_context: str,
    ) -> dict[str, Big5Profile]:
        """
        Pontua o transcript completo e retorna um Big5Profile por agente.

        Nota: 'Big5Profile' aqui armazena tanto scores Big Five quanto
        scores de métricas de negociação — o nome é histórico.
        """
        profiles: dict[str, Big5Profile] = {}

        for agent_id, role in agent_roles.items():
            profiles[agent_id] = Big5Profile(
                agent_id=agent_id,
                model_identifier=agent_id,
            )

        for i, turn in enumerate(transcript):
            role      = turn.get("role")
            content   = turn.get("content", "")
            agent_id  = turn.get("agent_id", role)

            if agent_id not in profiles:
                continue

            agent_role  = agent_roles.get(agent_id, role)
            turn_scores = self.evaluate_turn(
                utterance=content,
                role=agent_role,
                scenario_context=scenario_context,
                turn_index=i,
            )
            profiles[agent_id].per_turn_scores.extend(turn_scores)

        # Agrega: média por métrica por agente
        for agent_id, profile in profiles.items():
            for metric in self.config.dimensions:
                dim_scores = [
                    s.score for s in profile.per_turn_scores
                    if s.dimension == metric
                ]
                if dim_scores:
                    agg = sum(dim_scores) / len(dim_scores)
                    if self.config.invert_neuroticism and metric == Dimension.NEUROTICISM:
                        agg = 6.0 - agg
                    profile.scores[metric] = round(agg, 2)

        return profiles

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _score_one(
        self,
        utterance: str,
        role: str,
        scenario_context: str,
        turn_index: int,
        metric: AnyMetric,
        judge: LLMAdapter,
    ) -> DimensionScore:
        meta = ALL_METRICS_META[metric]
        # Binário (táticas): behavioral_anchors = {"disabled":..., "enabled":...}
        if "enabled" in meta.behavioral_anchors:
            prompt = _JUDGE_USER_BINARY.format(
                scenario_context=scenario_context,
                role=role,
                metric_name=meta.name,
                high_pole=meta.high_pole,
                low_pole=meta.low_pole,
                anchor_disabled=meta.behavioral_anchors["disabled"],
                anchor_enabled=meta.behavioral_anchors["enabled"],
                turn_index=turn_index,
                utterance=utterance,
            )
        else:
            prompt = _JUDGE_USER.format(
                scenario_context=scenario_context,
                role=role,
                metric_name=meta.name,
                high_pole=meta.high_pole,
                low_pole=meta.low_pole,
                anchor_1=meta.behavioral_anchors[1],
                anchor_3=meta.behavioral_anchors[3],
                anchor_5=meta.behavioral_anchors[5],
                turn_index=turn_index,
                utterance=utterance,
            )
        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user",   "content": prompt},
        ]
        try:
            raw    = judge.complete(messages)
            parsed = self._parse_json(raw)
            return DimensionScore(
                dimension=metric,
                score=float(parsed["score"]),
                justification=parsed.get("justification", ""),
                turn_index=turn_index,
                confidence=float(parsed.get("confidence", 1.0)),
            )
        except Exception as e:
            logger.warning("Judge failed for dim=%s turn=%d: %s", metric, turn_index, e)
            return DimensionScore(
                dimension=metric,
                score=3.0,
                justification=f"[Evaluation failed: {e}]",
                turn_index=turn_index,
                confidence=0.0,
            )

    @staticmethod
    def _parse_json(raw: str) -> dict:
        # Remove markdown fences and extra whitespace, then extract first JSON object
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        # Find first '{' — judge sometimes adds preamble/extra data after JSON
        start = clean.find("{")
        if start == -1:
            raise ValueError(f"No JSON object found in judge response: {raw[:300]}")
        # Use raw_decode to tolerate trailing extra data (e.g. two JSONs concatenated)
        decoder = json.JSONDecoder()
        try:
            obj, _ = decoder.raw_decode(clean[start:])
            return obj
        except json.JSONDecodeError:
            # Fallback: slice to last '}' and try
            end = clean.rfind("}")
            if end != -1 and end > start:
                return json.loads(clean[start:end + 1])
            raise

    @staticmethod
    def _irr(score1: float, score2: float) -> float:
        """IRR normalizado: 1.0 = concordância total, 0.0 = máximo desacordo."""
        return max(0.0, 1.0 - abs(score1 - score2) / 4.0)
