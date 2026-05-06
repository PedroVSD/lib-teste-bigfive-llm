"""
persona/tactics_builder.py
==========================
Transforma notas de táticas de negociação (1 a 5) em instruções textuais
para o prompt do agente, reaproveitando os metadados já existentes.
"""

from ..scoring.negotiation_metrics import NegotiationMetric, NEGOTIATION_META

class TacticsPromptBuilder:
    def build(self, tactics_dict: dict) -> str:
        if not tactics_dict:
            return ""

        lines = [
            "--- Negotiation Tactics & Behavioral Guidelines ---",
            "You must strictly follow these behavioral guidelines during the negotiation:",
            ""
        ]

        for key, score in tactics_dict.items():
            if score is None or not (1 <= score <= 5):
                continue

            try:
                # Transforma a chave do YAML na métrica correspondente
                metric = NegotiationMetric(key)
                meta = NEGOTIATION_META[metric]

                # Como as âncoras são 1, 3 e 5, arredondamos a nota para buscar o texto correto
                anchor_key = 1 if score <= 2 else (3 if score == 3 else 5)
                guidance = meta.behavioral_anchors[anchor_key]

                # Adiciona a instrução no prompt
                lines.append(f"[{meta.name} - Target Behavior]")
                lines.append(f"Guidance: {guidance}")
                lines.append("")

            except ValueError:
                # Ignora chaves que não existam no Enum NegotiationMetric
                pass

        lines.append("---------------------------------------------------")
        return "\n".join(lines)
