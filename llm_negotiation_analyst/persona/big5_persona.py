"""
persona/big5_persona.py
=======================
Define e gera instruções de personalidade Big Five para injeção no system
prompt de agentes negociadores.

Propósito
---------
Hoje a biblioteca apenas AVALIA traços Big Five após a negociação.
Este módulo permite INDUZIR traços antes dela começar, adicionando uma
seção de persona ao system prompt do agente.

Isso habilita dois tipos de experimento:
  1. Persona fixa   — testar como um modelo se comporta quando instruído
                      a agir como altamente Agreeable vs. altamente Neurotic.
  2. Persona livre  — não injetar persona e observar o traço natural do modelo
                      (o comportamento padrão já existente na biblioteca).

Design
------
- Big5Persona é um dataclass com scores 1–5 por dimensão (None = omitir).
- PersonaPromptBuilder transforma os scores em texto instrucional em inglês.
- O texto gerado é ANEXADO ao system prompt do cenário, não o substitui.
- Scores None são ignorados — você pode especificar só as dimensões que
  interessam ao seu estudo.

Exemplo de output gerado
------------------------
--- Personality Profile ---
You have a very high level of Openness to Experience.
  Behavioral guidance: Propose creative reframings of the negotiation.
  Introduce non-obvious trade-offs and package deals. Be intellectually
  curious about the other party's constraints and explore unconventional
  solutions willingly.

You have a low level of Agreeableness.
  Behavioral guidance: Prioritize your own goals over the relationship.
  Be direct and firm. Do not volunteer concessions. Challenge the other
  party's proposals critically rather than accommodating them.
---------------------------
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Score-to-label mapping
# ---------------------------------------------------------------------------

def _level_label(score: int) -> str:
    """Convert a 1–5 score to a natural language intensity label."""
    return {
        1: "a very low level of",
        2: "a low level of",
        3: "a moderate level of",
        4: "a high level of",
        5: "a very high level of",
    }[score]


# ---------------------------------------------------------------------------
# Per-dimension behavioral guidance by score band
# ---------------------------------------------------------------------------

_GUIDANCE: dict[str, dict[str, str]] = {
    "openness": {
        "high": (
            "open to experience"
            "closed to experience"
        ),
        "neutral": (
            "open to experience"
            "closed to experience"
        ),
        "low": (
            "open to experience"
            "closed to experience"
        ),
    },
    "conscientiousness": {
        "high": (
            "conscientious"
            "unconscientious"
        ),
        "neutral": (
            "conscientious"
            "unconscientious"
        ),
        "low": (
            "conscientious"
            "unconscientious"
        ),
    },
    "extraversion": {
        "high": (
            "extroverted"
            "introverted"
        ),
        "neutral": (
            "extroverted"
            "introverted"
        ),
        "low": (
            "extroverted"
            "introverted"
        ),
    },
    "agreeableness": {
        "high": (
            "agreeable"
            "antagonistic"
        ),
        "neutral": (
            "agreeable"
            "antagonistic"
        ),
        "low": (
            "agreeable"
            "antagonistic"
        ),
    },
    "neuroticism": {
        "high": (
            "neurotic"
            "emotionally stable"
        ),
        "neutral": (
            "neurotic"
            "emotionally stable"
        ),
        "low": (
            "neurotic"
            "emotionally stable"
        ),
    },
}


def _band(score: int) -> str:
    """Map score to guidance band."""
    if score >= 4:
        return "high"
    if score <= 2:
        return "low"
    return "neutral"


# ---------------------------------------------------------------------------
# Dimension display names
# ---------------------------------------------------------------------------

_DIM_NAMES: dict[str, str] = {
    "openness":          "Openness to Experience",
    "conscientiousness": "Conscientiousness",
    "extraversion":      "Extraversion",
    "agreeableness":     "Agreeableness",
    "neuroticism":       "Neuroticism",
}

_DIM_ORDER = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]


# ---------------------------------------------------------------------------
# Big5Persona dataclass
# ---------------------------------------------------------------------------

@dataclass
class Big5Persona:
    """
    Defines a Big Five personality profile for a negotiation agent.

    Each dimension accepts:
      - An integer 1–5  (1 = very low, 5 = very high)
      - None            (dimension is not specified / not injected)

    You can specify any subset of dimensions. Unspecified ones are omitted
    from the generated prompt, leaving the model's default behavior intact
    for those traits.

    Examples:
        # Highly agreeable, low neuroticism — cooperative and stable
        Big5Persona(agreeableness=5, neuroticism=1)

        # Full profile
        Big5Persona(openness=4, conscientiousness=5, extraversion=2,
                    agreeableness=1, neuroticism=4)

        # Only control Openness
        Big5Persona(openness=1)
    """
    openness:          Optional[int] = None
    conscientiousness: Optional[int] = None
    extraversion:      Optional[int] = None
    agreeableness:     Optional[int] = None
    neuroticism:       Optional[int] = None

    # Optional free-text additions appended after the generated instructions.
    # Use this for persona details that don't fit the 1–5 scale, e.g.:
    # extra_instructions="You tend to use sports metaphors when negotiating."
    extra_instructions: Optional[str] = None

    def __post_init__(self):
        for dim in _DIM_ORDER:
            val = getattr(self, dim)
            if val is not None and not (1 <= val <= 5):
                raise ValueError(
                    f"Big5Persona.{dim} must be between 1 and 5, got {val}."
                )

    def to_dict(self) -> dict:
        """Serializable representation for storage/logging."""
        return {
            dim: getattr(self, dim)
            for dim in _DIM_ORDER
            if getattr(self, dim) is not None
        }

    def specified_dimensions(self) -> list[str]:
        """Return list of dimension names that have a score set."""
        return [dim for dim in _DIM_ORDER if getattr(self, dim) is not None]

    @classmethod
    def from_dict(cls, d: dict) -> "Big5Persona":
        """Reconstruct from a dict (e.g., loaded from JSONL)."""
        return cls(**{k: v for k, v in d.items() if k in _DIM_ORDER})


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

class PersonaPromptBuilder:
    """
    Converts a Big5Persona into a natural language instruction block
    suitable for injection into an LLM system prompt.

    Usage:
        builder = PersonaPromptBuilder()
        block = builder.build(persona)
        full_system_prompt = scenario_system_prompt + "\\n\\n" + block
    """

    HEADER = "--- Personality Profile ---"
    FOOTER = "---------------------------"

    def build(self, persona: Big5Persona) -> str:
        """
        Generate the personality instruction block.

        Returns an empty string if no dimensions are specified,
        so it's safe to always call this and concatenate.
        """
        dims = persona.specified_dimensions()
        if not dims and not persona.extra_instructions:
            return ""

        lines = [self.HEADER, ""]

        for dim in dims:
            score = getattr(persona, dim)
            name  = _DIM_NAMES[dim]
            label = _level_label(score)
            band  = _band(score)
            guidance = _GUIDANCE[dim][band]

            lines.append(f"You have {label} {name}.")
            lines.append(f"  Behavioral guidance: {guidance}")
            lines.append("")

        if persona.extra_instructions:
            lines.append("Additional behavioral instructions:")
            lines.append(f"  {persona.extra_instructions}")
            lines.append("")

        lines.append(self.FOOTER)
        return "\n".join(lines)

    def inject(self, system_prompt: str, persona: Big5Persona) -> str:
        """
        Append the persona block to an existing system prompt.

        If the persona has no specified dimensions, returns the original
        system prompt unchanged — no side effects.
        """
        block = self.build(persona)
        if not block:
            return system_prompt
        return f"{system_prompt}\n\n{block}"
