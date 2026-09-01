"""
persona/tactics_builder.py
==========================

Transforma métricas de negociação configuradas no YAML em instruções
para o system prompt. Categórico: PRESENT / ABSENT / NOT_APPLICABLE (ou enabled/disabled).

  PRESENT (enabled, true, present)  -> injeta âncora "present" (comportamento ativo)
  ABSENT (disabled, false, absent)  -> não injeta (métrica desativada)
  NOT_APPLICABLE / none             -> não injeta

Legado 1-5 ainda aceito com aviso: 1-2 -> ABSENT (não injeta), 4-5 -> PRESENT.
"""

from ..scoring.negotiation_metrics import (
    NegotiationMetric,
    NEGOTIATION_META,
)

_PRESENT_VALUES = {"present", "enabled", "true", "on", "yes", "1", "active"}
_ABSENT_VALUES = {"absent", "disabled", "false", "off", "0", "no", "none", "inactive", "not_applicable"}


def _is_present(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _PRESENT_VALUES:
            return True
        if v in _ABSENT_VALUES:
            return False
    return None


# compat alias
_is_enabled = _is_present


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

        has_any = False

        for key, raw_val in tactics_dict.items():

            if raw_val is None:
                continue

            present = _is_present(raw_val)
            if present is not None:
                if not present:
                    continue  # absent/disabled = não injeta
                anchor_key = "present"
            else:
                try:
                    score = int(raw_val)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Invalid value for tactic '{key}': {raw_val!r}. "
                        "Expected 'present'/'absent' (or 'enabled'/'disabled') or integer 1-5."
                    )
                if not 1 <= score <= 5:
                    raise ValueError(
                        f"Tactic '{key}' score must be between 1 and 5. Got {score}."
                    )
                if score >= 4:
                    anchor_key = "present"
                else:
                    continue

            try:
                metric = NegotiationMetric(key)
            except ValueError:
                continue

            meta = NEGOTIATION_META[metric]
            guidance = meta.behavioral_anchors.get(anchor_key)
            if guidance is None:
                # fallback legacy keys
                guidance = meta.behavioral_anchors.get("enabled") or meta.behavioral_anchors.get(5) or meta.behavioral_anchors.get(1)
            if guidance is None:
                continue

            lines.append(f"[{meta.name}]")
            lines.append(f"Guidance: {guidance}")
            lines.append("")
            has_any = True

        if not has_any:
            return ""

        lines.append(self.FOOTER)

        return "\n".join(lines)
