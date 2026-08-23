"""
persona/tactics_builder.py
==========================

Transforma as métricas de negociação configuradas no YAML em
instruções comportamentais para o system prompt.

A escala utilizada é 1 a 5 (comportamento gradual), mapeada para
as âncoras 1, 3 e 5 dos metadados:

    1-2 -> âncora 1 (polo negativo)
    3   -> âncora 3 (neutro/moderado)
    4-5 -> âncora 5 (polo positivo)
"""

from ..scoring.negotiation_metrics import (
    NegotiationMetric,
    NEGOTIATION_META,
)


class TacticsPromptBuilder:

    HEADER = "--- Negotiation Tactics & Behavioral Guidelines ---"
    FOOTER = "---------------------------------------------------"

    def build(self, tactics_dict: dict) -> str:

        if not tactics_dict:
            return ""

        lines = [
            self.HEADER,
            "You must follow these behavioral guidelines during the negotiation:",
            "",
        ]

        for key, score in tactics_dict.items():

            if score is None:
                continue

            try:
                score = int(score)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Invalid score for tactic '{key}': {score!r}. "
                    "Expected integer 1-5."
                )

            if not 1 <= score <= 5:
                raise ValueError(
                    f"Tactic '{key}' score must be between 1 and 5. Got {score}."
                )

            try:
                metric = NegotiationMetric(key)
            except ValueError:
                # Métricas desconhecidas são ignoradas (compatibilidade)
                continue

            meta = NEGOTIATION_META[metric]

            # Mapeia 1-5 para âncoras 1, 3, 5
            if score <= 2:
                anchor_key = 1
            elif score == 3:
                anchor_key = 3
            else:
                anchor_key = 5

            guidance = meta.behavioral_anchors[anchor_key]

            lines.append(f"[{meta.name}]")
            lines.append(f"Guidance: {guidance}")
            lines.append("")

        lines.append(self.FOOTER)

        return "\n".join(lines)
