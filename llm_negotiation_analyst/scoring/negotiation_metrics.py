"""
negotiation_metrics.py
======================
Negotiation behavioral metrics beyond the Big Five — categorical.

Each metric is evaluated per turn as:
  PRESENT         — observable behavior present in the turn
  ABSENT          — opportunity existed but behavior absent
  NOT_APPLICABLE  — turn without enough opportunity to observe; excluded from the denominator

Aggregation: occurrence_rate = PRESENT / (PRESENT + ABSENT)  → 0–100%
  NOT_APPLICABLE ignored.

Organization:
  TACTICS            anchoring, conditional_concession, value_creation
  EMOTIONAL          rapport, resilience
  ARGUMENTATION      fact_justification, clarity
  BIASES             loss_aversion
"""

from enum import Enum
from .big5 import DimensionMeta


class NegotiationMetric(str, Enum):
    """Negotiation behavioral and tactic metrics — categorical."""

    ANCHORING              = "anchoring"
    CONDITIONAL_CONCESSION = "conditional_concession"
    VALUE_CREATION         = "value_creation"
    RAPPORT                = "rapport"
    RESILIENCE             = "resilience"
    FACT_JUSTIFICATION     = "fact_justification"
    CLARITY                = "clarity"
    LOSS_AVERSION          = "loss_aversion"


NEGOTIATION_META: dict[NegotiationMetric, DimensionMeta] = {

    NegotiationMetric.ANCHORING: DimensionMeta(
        name="Initial Offer Firmness (Anchoring)",
        abbreviation="ANC",
        high_pole="Strong / Inflexible Anchor",
        low_pole="Yields Quickly",
        observability=5,
        category="tactics",
        behavioral_anchors={
            "present": (
                "Makes an extreme offer in their own favor (strong anchoring) "
                "and defends the value with solid arguments before making any concession."
            ),
            "absent": (
                "Makes a weak initial offer, anchoring against themselves, "
                "or immediately yields their value at the first sign of resistance."
            ),
        },
    ),

    NegotiationMetric.CONDITIONAL_CONCESSION: DimensionMeta(
        name="Use of Conditional Concessions",
        abbreviation="CON",
        high_pole="Strict Exchanges (Give-and-Take)",
        low_pole="Unilateral Concession",
        observability=5,
        category="tactics",
        behavioral_anchors={
            "present": (
                "Every concession is strictly tied to an explicit gain: "
                "'If I accept X, you MUST give me Y in return.'"
            ),
            "absent": (
                "Makes concessions unilaterally, lowering their price or yielding "
                "benefits without asking for absolutely anything in return."
            ),
        },
    ),

    NegotiationMetric.VALUE_CREATION: DimensionMeta(
        name="Focus on Value Creation (Win-Win)",
        abbreviation="VAL",
        high_pole="Integrative / Creative",
        low_pole="Distributive / Zero-Sum",
        observability=3,
        category="tactics",
        behavioral_anchors={
            "present": (
                "Proactively adds new variables to the table (bonuses, time off, deadlines) "
                "to create a package that benefits both sides."
            ),
            "absent": (
                "Focuses exclusively on fighting over a single metric (e.g., salary alone), "
                "treating the negotiation as a tug of war."
            ),
        },
    ),

    NegotiationMetric.RAPPORT: DimensionMeta(
        name="Rapport Building (Empathy)",
        abbreviation="RAP",
        high_pole="Highly Empathic / Partner",
        low_pole="Cold / Transactional",
        observability=5,
        category="emotional",
        behavioral_anchors={
            "present": (
                "Actively validates the opponent's emotions, uses a collaborative tone, "
                "and explicitly focuses on building a long-term partnership."
            ),
            "absent": (
                "Cold, robotic, or purely transactional tone. "
                "Ignores the human side and the opponent's needs."
            ),
        },
    ),

    NegotiationMetric.RESILIENCE: DimensionMeta(
        name="Resilience Under Pressure",
        abbreviation="RES",
        high_pole="Calm / Unshakable",
        low_pole="Impulsive / Fearful",
        observability=3,
        category="emotional",
        behavioral_anchors={
            "present": (
                "Completely unshakable in the face of cancellation threats or harsh demands. "
                "Redirects focus to the facts calmly and confidently."
            ),
            "absent": (
                "Instantly yields to ultimatums, shows desperation, "
                "or reacts with disproportionate aggression when pressured."
            ),
        },
    ),

    NegotiationMetric.FACT_JUSTIFICATION: DimensionMeta(
        name="Fact-Based Justification",
        abbreviation="JUS",
        high_pole="Highly Grounded",
        low_pole="Empty Arguments",
        observability=5,
        category="argumentation",
        behavioral_anchors={
            "present": (
                "Supports every offer with solid data: macroeconomic scenario, inflation, "
                "market averages, ROI metrics, or industry benchmarks."
            ),
            "absent": (
                "Makes demands based only on personal desire or subjective needs, "
                "with no market justification or concrete data."
            ),
        },
    ),

    NegotiationMetric.CLARITY: DimensionMeta(
        name="Clarity and Logical Structure",
        abbreviation="CLA",
        high_pole="Structured / Mathematical",
        low_pole="Confusing / Disorganized",
        observability=5,
        category="argumentation",
        behavioral_anchors={
            "present": (
                "Highly structured. Separates proposals by topic, "
                "summarizes values clearly, and presents flawless arithmetic."
            ),
            "absent": (
                "Mixes proposals, presents mathematically conflicting values, "
                "or expresses themselves vaguely and hard to follow."
            ),
        },
    ),

    NegotiationMetric.LOSS_AVERSION: DimensionMeta(
        name="Loss Aversion",
        abbreviation="LSS",
        high_pole="Loss-Reactive",
        low_pole="Focused on Final Gain",
        observability=1,
        category="cognitive_bias",
        behavioral_anchors={
            "present": (
                "Fights desperately against the removal of any item "
                "already considered guaranteed, even when offered double value elsewhere."
            ),
            "absent": (
                "Focuses rationally on the package's total value, not minding "
                "whether a specific benefit was removed as long as it is compensated elsewhere."
            ),
        },
    ),
}


METRICS_BY_CATEGORY: dict[str, list[NegotiationMetric]] = {
    "tactics": [
        NegotiationMetric.ANCHORING,
        NegotiationMetric.CONDITIONAL_CONCESSION,
        NegotiationMetric.VALUE_CREATION,
    ],
    "emotional": [
        NegotiationMetric.RAPPORT,
        NegotiationMetric.RESILIENCE,
    ],
    "argumentation": [
        NegotiationMetric.FACT_JUSTIFICATION,
        NegotiationMetric.CLARITY,
    ],
    "cognitive_bias": [
        NegotiationMetric.LOSS_AVERSION,
    ],
}
