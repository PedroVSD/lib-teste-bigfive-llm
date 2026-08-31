"""
persona/tactics_builder.py
==========================

Transforma as métricas de negociação configuradas no YAML em
instruções comportamentais para o system prompt.

Suporta dois formatos (compatíveis):

  1) Numérico 1-5 (legado, gradual):
     1-2 -> âncora 1 (polo negativo)
     3   -> âncora 3 (neutro/moderado)
     4-5 -> âncora 5 (polo positivo)

  2) Booleano enabled/disabled (atual):
     enabled  -> injeta âncora 5 (ativo, polo positivo)
     disabled -> não injeta (métrica desativada)

Exemplo YAML:
  tactics:
    anchoring: enabled
    rapport: disabled
    clarity: enabled
"""

from ..scoring.negotiation_metrics import (
    NegotiationMetric,
    NEGOTIATION_META,
)

_ENABLED_VALUES = {"enabled", "true", "on", "yes", "1", "active"}
_DISABLED_VALUES = {"disabled", "false", "off", "0", "no", "none", "inactive"}


def _is_enabled(value) -> bool | None:
    """Retorna True=enabled, False=disabled, None=não é booleano."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _ENABLED_VALUES:
            return True
        if v in _DISABLED_VALUES:
            return False
    return None


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

            # Primeiro testa enabled/disabled (string ou bool) — binário puro
            enabled = _is_enabled(raw_val)
            if enabled is not None:
                if not enabled:
                    continue  # disabled = não injeta (métrica ausente)
                # enabled = usa âncora "enabled"
                anchor_key = "enabled"
            else:
                # Fallback legado: tenta int 1-5 (compatibilidade)
                try:
                    score = int(raw_val)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Invalid value for tactic '{key}': {raw_val!r}. "
                        "Expected 'enabled'/'disabled' or integer 1-5."
                    )
                if not 1 <= score <= 5:
                    raise ValueError(
                        f"Tactic '{key}' score must be between 1 and 5. Got {score}."
                    )
                # Legado 1-5 mapeado para binário: 1-2→disabled (não injeta), 3→disabled, 4-5→enabled
                if score >= 4:
                    anchor_key = "enabled"
                else:
                    continue  # 1-3 = não injeta no modo binário
                # Para compatibilidade total, se quiser manter granular, descomente:
                # if score <= 2: anchor_key = "disabled"
                # elif score == 3: anchor_key = "disabled"
                # else: anchor_key = "enabled"

            try:
                metric = NegotiationMetric(key)
            except ValueError:
                # Métricas desconhecidas são ignoradas (compatibilidade)
                continue

            meta = NEGOTIATION_META[metric]
            # Binário: apenas "enabled"/"disabled"
            guidance = meta.behavioral_anchors.get(anchor_key)
            if guidance is None:
                # Fallback para chaves numéricas antigas (1/5)
                guidance = meta.behavioral_anchors.get(5 if anchor_key == "enabled" else 1)
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
