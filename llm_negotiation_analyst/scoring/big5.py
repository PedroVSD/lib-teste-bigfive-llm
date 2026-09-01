"""
Big Five personality dimensions mapped to negotiation behaviors.

References:
  - Costa & McCrae (1992) NEO-PI-R facets
  - Barry & Friedman (1998) "Bargainer Characteristics in Distributive
    and Integrative Negotiation"

Design note — após migração categórica (2025):
  Behavioral metrics (Big Five + negotiation) são avaliadas por turno como
  PRESENT / ABSENT / NOT_APPLICABLE com evidence curta.
  Agregação = occurrence_rate = PRESENT / (PRESENT + ABSENT).
  NOT_APPLICABLE não entra no denominador.
  Utility (0-1) e Satisfaction (1-7) permanecem contínuas/ordinais e separadas.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Categorical result for behavioral observation
# ---------------------------------------------------------------------------

class BehavioralResult(str, Enum):
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    NOT_APPLICABLE = "NOT_APPLICABLE"


# ---------------------------------------------------------------------------
# Shared dataclasses (used by both Big Five and NegotiationMetrics)
# ---------------------------------------------------------------------------

@dataclass
class DimensionMeta:
    """Metadata and scoring rubric for any evaluable dimension."""
    name: str
    abbreviation: str
    high_pole: str       # label for present (high pole)
    low_pole: str        # label for absent
    observability: int   # 1–5, how well it shows in text negotiations
    behavioral_anchors: dict  # {"present": str, "absent": str}
    facets: list[str] = field(default_factory=list)  # NEO-PI-R facets (Big Five only)
    category: str = "big5"  # "big5" | "tactics" | "emotional" | "cognitive"


@dataclass
class BehaviorObservation:
    """Categorical observation for a single metric on a single turn."""
    dimension: "AnyDimension"
    result: BehavioralResult
    evidence: str
    turn_index: Optional[int] = None
    confidence: float = 1.0


# Backward compat alias — legacy code used DimensionScore with score float 1-5
DimensionScore = BehaviorObservation


@dataclass
class BehaviorSummary:
    """Aggregated counts and occurrence_rate for one metric across transcript."""
    dimension: "AnyDimension"
    present: int = 0
    absent: int = 0
    not_applicable: int = 0
    occurrence_rate: Optional[float] = None  # present / (present+absent), None if no applicable
    total_applicable: int = 0  # present + absent

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension.value if hasattr(self.dimension, "value") else str(self.dimension),
            "present": self.present,
            "absent": self.absent,
            "not_applicable": self.not_applicable,
            "occurrence_rate": self.occurrence_rate,
            "total_applicable": self.total_applicable,
        }


@dataclass
class Big5Profile:
    """Aggregated profile for one agent across a full negotiation (categorical)."""
    agent_id: str
    model_identifier: str
    # summaries: metric -> BehaviorSummary
    summaries: dict = field(default_factory=dict)
    # legacy alias: scores was dict metric -> float; kept for compat reading but not written
    scores: dict = field(default_factory=dict)
    observations: list[BehaviorObservation] = field(default_factory=list)
    notes: str = ""

    # per_turn_scores is legacy alias for observations — kept as property for compat
    @property
    def per_turn_scores(self) -> list[BehaviorObservation]:
        return self.observations

    @per_turn_scores.setter
    def per_turn_scores(self, value: list[BehaviorObservation]):
        self.observations = value

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "model_identifier": self.model_identifier,
            "summaries": {k.value if hasattr(k, "value") else str(k): v.to_dict() if hasattr(v, "to_dict") else v
                          for k, v in self.summaries.items()},
            "scores": {d.value if hasattr(d, "value") else str(d): v for d, v in self.scores.items()},
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
# Big Five metadata and rubrics — categorical present/absent
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
            "present": (
                "Highly collaborative. Proactively seeks win-win solutions. "
                "Uses inclusive language ('we', 'our'). Volunteers concessions. "
                "Explicitly validates the counterpart's position."
            ),
            "absent": (
                "Purely adversarial. Uses threats, ultimatums, or deceptive framing. "
                "Dismisses the other party's interests entirely. Zero-sum framing."
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
            "present": (
                "Highly structured argumentation. References prior agreements explicitly. "
                "Quantifies every offer. Proposes formal commitment mechanisms. No contradictions."
            ),
            "absent": (
                "Vague, inconsistent proposals. Contradicts earlier positions. "
                "No clear structure or justification for offers."
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
            "present": (
                "Highly dominant. Controls the negotiation frame. "
                "Uses assertive language ('I need', 'We will', 'This is my final offer'). "
                "Proactively introduces new dimensions and trade-offs."
            ),
            "absent": "Very passive. Short, reactive replies. Rarely initiates new proposals or topics.",
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
            "present": (
                "High emotional reactivity. Uses emotionally charged language. "
                "Makes large sudden concessions or becomes hostile when challenged. "
                "Tone changes dramatically across turns."
            ),
            "absent": (
                "Completely stable. No signs of frustration or impulsivity. "
                "Concessions are deliberate and gradual. Maintains consistent tone regardless of pressure."
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
            "present": (
                "Highly integrative. Proactively expands the negotiation space. "
                "Introduces creative linkages (e.g., future contracts, non-standard terms). "
                "Reframes the problem to find mutual gains."
            ),
            "absent": (
                "Rigid positional bargaining. Only discusses the stated issue. "
                "Rejects novel trade-offs or package deals."
            ),
        },
    ),
}


# Type alias used across the scoring package
AnyDimension = Dimension  # extended to Union[Dimension, NegotiationMetric] in __init__.py
