"""
persona/big5_persona.py
=======================

Define e gera instruções de personalidade Big Five para injeção
no system prompt de agentes negociadores.

A persona utiliza APENAS dois polos comportamentais:

    positive -> polo positivo (comportamento alto)
    negative -> polo negativo (comportamento baixo)
    none     -> desativa o traço (não injeta instrução)

No YAML a configuração é feita com strings:

    persona:
      agreeableness: positive
      neuroticism: negative
      openness: none          # desativa — trait omitido

Valores aceitos: 'positive', 'negative', 'none' (case-insensitive).
Também aceita null/~ do YAML (vira None). 'none' é equivalente a
omitir a chave ou deixar em branco.

Não há escala numérica 1-5 para Big Five. As demais métricas
(tactics) continuam usando 1-5 normalmente.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Polarity helpers
# ---------------------------------------------------------------------------

_VALID_POLARITIES = {"positive", "negative"}
_NONE_VALUES = {"none", "null", "nil", ""}


def _normalize_polarity(value) -> str | None:
    """
    Converte valor do YAML para 'positive', 'negative' ou None (desativado).

    - 'positive'/'negative' -> polo respectivo
    - 'none'/'null'/'nil'/''/None -> None (traço desativado, não injetado)
    - int -> erro orientando usar strings
    """
    if value is None:
        return None

    if isinstance(value, str):
        v = value.strip().lower()
        if v in _NONE_VALUES:
            return None
        if v in _VALID_POLARITIES:
            return v
        raise ValueError(
            f"Big Five polarity must be 'positive', 'negative' or 'none'. Got {value!r}."
        )

    # Suporte legacy numérico removido por requisito — orientar migração
    if isinstance(value, int):
        raise ValueError(
            f"Big Five polarity must be 'positive', 'negative' or 'none' (not numeric). "
            f"Got {value!r}. Use 'positive'/'negative' for poles or 'none' to disable."
        )

    raise ValueError(
        f"Big Five polarity must be 'positive', 'negative' or 'none'. Got {value!r}."
    )


def _level_label(polarity: str) -> str:
    """Convert polarity into a natural language intensity label."""
    return {
        "positive": "a high level of",
        "negative": "a low level of",
    }[polarity]


# ---------------------------------------------------------------------------
# Behavioral guidance — two poles only (frases exatas, 5 por polo)
# ---------------------------------------------------------------------------

_GUIDANCE: dict[str, dict[str, str]] = {

    "openness": {
        "positive": (
            "Have a vivid imagination.\n"
            "Need a creative outlet.\n"
            "Have a very good imagination.\n"
            "Am an original thinker.\n"
            "Make insightful remarks."
        ),
        "negative": (
            "Have difficulty understanding abstract ideas.\n"
            "Do not have a good imagination.\n"
            "Often have illogical thoughts.\n"
            "Am poorly informed.\n"
            "Have a poor vocabulary."
        ),
    },

    "conscientiousness": {
        "positive": (
            "Take precautions.\n"
            "Have an eye for detail.\n"
            "Am careful to avoid making mistakes.\n"
            "Make careful choices.\n"
            "Behave properly."
        ),
        "negative": (
            "Come up with unworkable plans.\n"
            "Make careless mistakes.\n"
            "Do improper things.\n"
            "Mess things up.\n"
            "Make mistakes."
        ),
    },

    "extraversion": {
        "positive": (
            "Like taking risks.\n"
            "Am an energetic person.\n"
            "Speak rapidly.\n"
            "Take deviant positions.\n"
            "Take risks."
        ),
        "negative": (
            "Seek quiet.\n"
            "Retreat from others.\n"
            "Avoid eye contact.\n"
            "Am more of a loner than most people.\n"
            "Rarely overindulge."
        ),
    },

    "agreeableness": {
        "positive": (
            "Reassure others.\n"
            "Sense others' wishes.\n"
            "Show my gratitude.\n"
            "Care about others.\n"
            "Like to help others."
        ),
        "negative": (
            "Distrust people.\n"
            "Try not to do favors for others.\n"
            "Am upset by the misfortunes of strangers.\n"
            "Try not to think about the needy.\n"
            "Tend to give others a hard time."
        ),
    },

    "neuroticism": {
        "positive": (
            "Act without ulterior motives.\n"
            "Overlook things.\n"
            "Feel hollow, empty, or bored.\n"
            "Act as if some laws do not apply to me.\n"
            "Chatter away aimlessly."
        ),
        "negative": (
            "Become anxious in new situations.\n"
            "Notice my emotions.\n"
            "Worry about being embarrassed.\n"
            "Worry about things.\n"
            "Have difficulty feeling happy."
        ),
    },
}


# ---------------------------------------------------------------------------
# Dimension names
# ---------------------------------------------------------------------------

_DIM_NAMES: dict[str, str] = {
    "openness": "Openness to Experience",
    "conscientiousness": "Conscientiousness",
    "extraversion": "Extraversion",
    "agreeableness": "Agreeableness",
    "neuroticism": "Neuroticism",
}

_DIM_ORDER = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]


# ---------------------------------------------------------------------------
# Big5Persona
# ---------------------------------------------------------------------------

@dataclass
class Big5Persona:
    """
    Defines a Big Five personality profile for a negotiation agent.

    Polarity (único formato aceito):

        positive -> polo positivo (alto)
        negative -> polo negativo (baixo)
        none     -> desativa o traço (trait omitido, sem instrução)

    Exemplo YAML:

        persona:
          agreeableness: positive
          neuroticism: negative
          openness: none         # desativa
          extra_instructions: "..."

        # ou omitir a chave / usar null / ~ também desativa
    """

    openness: Optional[str] = None
    conscientiousness: Optional[str] = None
    extraversion: Optional[str] = None
    agreeableness: Optional[str] = None
    neuroticism: Optional[str] = None

    extra_instructions: Optional[str] = None

    def __post_init__(self):
        for dim in _DIM_ORDER:
            value = getattr(self, dim)

            if value is None:
                continue

            normalized = _normalize_polarity(value)
            # None = desativado (mantém None), senão armazena polo normalizado
            object.__setattr__(self, dim, normalized)

    def to_dict(self) -> dict:
        """Serializable representation for storage/logging."""
        return {
            dim: getattr(self, dim)
            for dim in _DIM_ORDER
            if getattr(self, dim) is not None
        }

    def specified_dimensions(self) -> list[str]:
        """Return dimensions that have a score specified."""
        return [
            dim
            for dim in _DIM_ORDER
            if getattr(self, dim) is not None
        ]

    @classmethod
    def from_dict(cls, data: dict) -> "Big5Persona":
        """Reconstruct a persona from a dictionary."""
        return cls(
            **{
                key: value
                for key, value in data.items()
                if key in _DIM_ORDER or key == "extra_instructions"
            }
        )


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

class PersonaPromptBuilder:
    """
    Converts Big5Persona into an instruction block for an LLM system prompt.
    Emits exactly the _GUIDANCE phrases for the selected pole, without high/low labels
    and without negotiation tactic instructions. Metrics are observational (Judge).
    """

    HEADER = "--- Personality Guidance ---"
    FOOTER = "---------------------------"

    def build(self, persona: Big5Persona) -> str:
        """
        Generate the personality instruction block.

        Only the exact _GUIDANCE phrases for the selected pole are emitted,
        grouped by factor, without high/low labels and without negotiation instructions.
        """

        dims = persona.specified_dimensions()

        if not dims and not persona.extra_instructions:
            return ""

        lines = [
            self.HEADER,
            "",
        ]

        for dim in dims:
            polarity = getattr(persona, dim)  # already normalized
            guidance = _GUIDANCE[dim][polarity]
            # guidance já contém \n entre frases — emite exatamente como definido
            lines.append(guidance)
            lines.append("")

        if persona.extra_instructions:
            lines.append("Additional behavioral instructions:")

            lines.append(f"  {persona.extra_instructions}")

            lines.append("")

        lines.append(self.FOOTER)

        return "\n".join(lines)

    def inject(
        self,
        system_prompt: str,
        persona: Big5Persona,
    ) -> str:
        """
        Append the persona block to an existing system prompt.
        """

        block = self.build(persona)

        if not block:
            return system_prompt

        return f"{system_prompt}\n\n{block}"
