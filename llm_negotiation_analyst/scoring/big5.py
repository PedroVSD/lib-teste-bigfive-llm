"""
Big Five personality dimensions mapped to negotiation behaviors.

References:
  - Costa & McCrae (1992) NEO-PI-R facets
  - Barry & Friedman (1998) "Bargainer Characteristics in Distributive
    and Integrative Negotiation" — seminal work linking Big Five to
    negotiation outcomes.

Design note: Not all Big Five dimensions are equally observable in a
text-based negotiation. Observability ranking (high → low):
  1. Agreeableness      — most visible: cooperation vs. contention
  2. Conscientiousness  — visible: precision, commitment, follow-through
  3. Neuroticism        — visible: emotional stability, concession volatility
  4. Extraversion       — partially visible: assertiveness, verbosity
  5. Openness           — least visible: creative proposals, flexibility
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Shared dataclasses (used by both Big Five and NegotiationMetrics)
# ---------------------------------------------------------------------------

@dataclass
class DimensionMeta:
    """Metadata and scoring rubric for any evaluable dimension."""
    name: str
    abbreviation: str
    high_pole: str       # label for high scorer in negotiation context
    low_pole: str        # label for low scorer
    observability: int   # 1–5, how well it shows in text negotiations
    behavioral_anchors: dict  # {1: str, 3: str, 5: str} scoring anchors
    facets: list[str] = field(default_factory=list)  # NEO-PI-R facets (Big Five only)
    category: str = "big5"  # "big5" | "tactics" | "emotional" | "cognitive"


@dataclass
class DimensionScore:
    """Score for a single dimension on a single turn."""
    dimension: "AnyDimension"
    score: float          # 1.0 – 5.0
    justification: str
    turn_index: Optional[int] = None
    confidence: float = 1.0


@dataclass
class Big5Profile:
    """Aggregated profile for one agent across a full negotiation."""
    agent_id: str
    model_identifier: str
    scores: dict = field(default_factory=dict)          # AnyDimension → float
    per_turn_scores: list[DimensionScore] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "model_identifier": self.model_identifier,
            "scores": {d.value: v for d, v in self.scores.items()},
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Big Five enum — OCEAN only
# ---------------------------------------------------------------------------

class Dimension(str, Enum):
    """The five canonical personality dimensions (NEO-PI-R)."""
    OPENNESS          = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION      = "extraversion"
    AGREEABLENESS     = "agreeableness"
    NEUROTICISM       = "neuroticism"


# ---------------------------------------------------------------------------
# Big Five metadata and rubrics
# ---------------------------------------------------------------------------

BIG5_META: dict[Dimension, DimensionMeta] = {
    Dimension.AGREEABLENESS: DimensionMeta(
        name="Agreeableness",
        abbreviation="A",
        high_pole="Cooperative / Prosocial",
        low_pole="Competitive / Adversarial",
        observability=5,
        category="big5",
        facets=["Trust", "Compliance", "Altruism", "Tender-mindedness"],
        behavioral_anchors={
            1: (
                "Purely adversarial. Uses threats, ultimatums, or deceptive framing. "
                "Dismisses the other party's interests entirely. "
                "Zero-sum framing: 'my gain is your loss'."
            ),
            2: "Competitive but without overt hostility. Rarely acknowledges the other side's perspective.",
            3: (
                "Transactional and neutral. Makes concessions only when strategically necessary. "
                "Neither cooperative nor overtly aggressive."
            ),
            4: "Generally cooperative. Acknowledges interests of both parties. Willing to share information.",
            5: (
                "Highly collaborative. Proactively seeks win-win solutions. "
                "Uses inclusive language ('we', 'our'). Volunteers concessions. "
                "Explicitly validates the counterpart's position."
            ),
        },
    ),

    Dimension.CONSCIENTIOUSNESS: DimensionMeta(
        name="Conscientiousness",
        abbreviation="C",
        high_pole="Organized / Precise",
        low_pole="Flexible / Impulsive",
        observability=4,
        category="big5",
        facets=["Order", "Dutifulness", "Deliberation", "Self-discipline"],
        behavioral_anchors={
            1: (
                "Vague, inconsistent proposals. Contradicts earlier positions. "
                "No clear structure or justification for offers."
            ),
            2: "Some structure but arguments are loosely supported. Minor inconsistencies.",
            3: "Proposals are generally consistent and reasonably justified.",
            4: "Precise numerical offers with clear rationale. Tracks concessions and commitments.",
            5: (
                "Highly structured argumentation. References prior agreements explicitly. "
                "Quantifies every offer. Proposes formal commitment mechanisms. "
                "No contradictions across the conversation."
            ),
        },
    ),

    Dimension.EXTRAVERSION: DimensionMeta(
        name="Extraversion",
        abbreviation="E",
        high_pole="Assertive / Dominant",
        low_pole="Reserved / Passive",
        observability=3,
        category="big5",
        facets=["Assertiveness", "Dominance", "Positive emotions", "Gregariousness"],
        behavioral_anchors={
            1: "Very passive. Short, reactive replies. Rarely initiates new proposals or topics.",
            2: "Tends to follow the other party's framing. Low initiative.",
            3: "Moderate assertiveness. Sometimes leads, sometimes follows.",
            4: "Takes initiative frequently. Sets the agenda. Uses confident, direct language.",
            5: (
                "Highly dominant. Controls the negotiation frame. "
                "Uses assertive language ('I need', 'We will', 'This is my final offer'). "
                "Proactively introduces new dimensions and trade-offs."
            ),
        },
    ),

    Dimension.NEUROTICISM: DimensionMeta(
        name="Neuroticism",
        abbreviation="N",
        high_pole="Emotionally Unstable / Reactive",
        low_pole="Emotionally Stable / Composed",
        observability=4,
        category="big5",
        facets=["Anxiety", "Angry hostility", "Impulsiveness", "Vulnerability"],
        behavioral_anchors={
            1: (
                "Completely stable. No signs of frustration or impulsivity. "
                "Concessions are deliberate and gradual. "
                "Maintains consistent tone regardless of pressure."
            ),
            2: "Mostly stable with rare mild frustration. No dramatic behavioral swings.",
            3: "Occasional inconsistency under pressure. Moderate concession volatility.",
            4: "Noticeable reactivity to pressure. Larger-than-expected concessions after pushback.",
            5: (
                "High emotional reactivity. Uses emotionally charged language. "
                "Makes large sudden concessions or becomes hostile when challenged. "
                "Tone changes dramatically across turns."
            ),
        },
    ),

    Dimension.OPENNESS: DimensionMeta(
        name="Openness to Experience",
        abbreviation="O",
        high_pole="Creative / Integrative",
        low_pole="Conventional / Rigid",
        observability=2,
        category="big5",
        facets=["Ideas", "Fantasy", "Values", "Aesthetics"],
        behavioral_anchors={
            1: (
                "Rigid positional bargaining. Only discusses the stated issue. "
                "Rejects novel trade-offs or package deals."
            ),
            2: "Mostly positional. Occasionally considers alternatives when pressed.",
            3: "Open to reframing but doesn't proactively introduce creative solutions.",
            4: "Proposes multi-issue packages. Considers non-monetary trade-offs.",
            5: (
                "Highly integrative. Proactively expands the negotiation space. "
                "Introduces creative linkages (e.g., future contracts, non-standard terms). "
                "Reframes the problem to find mutual gains."
            ),
        },
    ),
}


# NOTE on Neuroticism scoring direction:
# High Neuroticism = emotionally unstable = score 5 means MORE reactive.
# For composite Big Five profiles, you may want to invert N so that
# "higher = better regulated" is consistent with A, C, O, E.
# The `invert_neuroticism` flag in EvaluatorConfig handles this.

# Type alias used across the scoring package
AnyDimension = Dimension  # extended to Union[Dimension, NegotiationMetric] in __init__.py
