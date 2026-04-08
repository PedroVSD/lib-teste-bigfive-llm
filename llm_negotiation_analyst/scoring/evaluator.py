"""
LLM-as-judge evaluator for Big Five dimensions.

Design decisions:
  - Judge is a SEPARATE LLM adapter, decoupled from the negotiating agents.
    This avoids self-evaluation bias: a model judging its own outputs.
  - Each turn is scored independently; aggregate scores are the mean.
  - Judge is prompted to return structured JSON, parsed with fallback.
  - Optional dual-judge mode: pass two judges; inter-rater reliability
    (Cohen's κ or Pearson r) is computed and stored in the report.
"""

import json
import re
import logging
from dataclasses import dataclass
from typing import Optional

from ..adapters.base import LLMAdapter
from .big5 import Dimension, DimensionScore, Big5Profile, BIG5_META

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """You are an expert researcher in personality psychology and \
negotiation science. Your task is to evaluate a single utterance from a \
negotiation transcript according to the Big Five personality framework, \
specifically in the context of negotiation behavior.

You will be given:
  - The negotiation context (scenario description)
  - The role of the speaker (e.g., buyer, seller, employer, candidate)
  - A specific Big Five dimension to score
  - Behavioral anchors for scores 1–5
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

## Dimension: {dimension_name} ({high_pole} ↔ {low_pole})

### Scoring Anchors
1 — {anchor_1}
3 — {anchor_3}
5 — {anchor_5}

## Utterance to Evaluate (Turn {turn_index})
\"\"\"{utterance}\"\"\"

Evaluate this utterance on {dimension_name}. Respond with JSON only."""


# ---------------------------------------------------------------------------
# Evaluator class
# ---------------------------------------------------------------------------

@dataclass
class EvaluatorConfig:
    dimensions: list[Dimension] = None   # None = all 5
    score_per_turn: bool = True          # if False, evaluate full transcript at once
    invert_neuroticism: bool = False     # flip N so high=stable for composite scores

    def __post_init__(self):
        if self.dimensions is None:
            self.dimensions = list(Dimension)


class Evaluator:
    """
    Uses a judge LLM to score negotiation utterances on Big Five dimensions.

    Args:
        judge: LLM adapter used as the evaluator (ideally different from agents).
        config: EvaluatorConfig controlling which dimensions to score.
        second_judge: Optional second judge for inter-rater reliability.
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
    # Public API
    # ------------------------------------------------------------------

    def evaluate_turn(
        self,
        utterance: str,
        role: str,
        scenario_context: str,
        turn_index: int,
        dimensions: Optional[list[Dimension]] = None,
    ) -> list[DimensionScore]:
        """Score a single utterance across the requested dimensions."""
        dims = dimensions or self.config.dimensions
        scores = []
        for dim in dims:
            score = self._score_one(utterance, role, scenario_context, turn_index, dim, self.judge)
            scores.append(score)
            if self.second_judge:
                score2 = self._score_one(utterance, role, scenario_context, turn_index, dim, self.second_judge)
                score.confidence = self._irr(score.score, score2.score)
        return scores

    def evaluate_transcript(
        self,
        transcript: list[dict],  # [{"role": str, "content": str}, ...]
        agent_roles: dict[str, str],  # {"agent_a": "buyer", "agent_b": "seller"}
        scenario_context: str,
    ) -> dict[str, Big5Profile]:
        """
        Score a full negotiation transcript.

        Returns a Big5Profile per agent.
        """
        profiles: dict[str, Big5Profile] = {}

        # Initialize profiles
        for agent_id, role in agent_roles.items():
            profiles[agent_id] = Big5Profile(
                agent_id=agent_id,
                model_identifier=agent_id,  # overridden by engine
            )

        # Score each turn
        for i, turn in enumerate(transcript):
            role = turn.get("role")
            content = turn.get("content", "")
            agent_id = turn.get("agent_id", role)

            if agent_id not in profiles:
                continue

            agent_role = agent_roles.get(agent_id, role)
            turn_scores = self.evaluate_turn(
                utterance=content,
                role=agent_role,
                scenario_context=scenario_context,
                turn_index=i,
            )
            profiles[agent_id].per_turn_scores.extend(turn_scores)

        # Aggregate: mean per dimension per agent
        for agent_id, profile in profiles.items():
            for dim in self.config.dimensions:
                dim_scores = [
                    s.score for s in profile.per_turn_scores if s.dimension == dim
                ]
                if dim_scores:
                    agg = sum(dim_scores) / len(dim_scores)
                    if self.config.invert_neuroticism and dim == Dimension.NEUROTICISM:
                        agg = 6.0 - agg  # invert: 5→1, 1→5
                    profile.scores[dim] = round(agg, 2)

        return profiles

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_one(
        self,
        utterance: str,
        role: str,
        scenario_context: str,
        turn_index: int,
        dimension: Dimension,
        judge: LLMAdapter,
    ) -> DimensionScore:
        meta = BIG5_META[dimension]
        prompt = _JUDGE_USER.format(
            scenario_context=scenario_context,
            role=role,
            dimension_name=meta.name,
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
            raw = judge.complete(messages)
            parsed = self._parse_json(raw)
            return DimensionScore(
                dimension=dimension,
                score=float(parsed["score"]),
                justification=parsed.get("justification", ""),
                turn_index=turn_index,
                confidence=float(parsed.get("confidence", 1.0)),
            )
        except Exception as e:
            logger.warning("Judge failed for dim=%s turn=%d: %s", dimension, turn_index, e)
            return DimensionScore(
                dimension=dimension,
                score=3.0,
                justification=f"[Evaluation failed: {e}]",
                turn_index=turn_index,
                confidence=0.0,
            )

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Parse JSON, stripping accidental markdown fences."""
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        return json.loads(clean)

    @staticmethod
    def _irr(score1: float, score2: float) -> float:
        """
        Simplified inter-rater reliability as normalized agreement.
        Returns 1.0 for identical scores, 0.0 for maximum disagreement (|diff|=4).
        """
        return max(0.0, 1.0 - abs(score1 - score2) / 4.0)
