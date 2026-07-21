"""
scoring/utility.py
==================
Calcula a Utilidade Econômica de cada agente com base no preço final acordado.

Fórmulas
--------
Vendedor:   u_s(p) = (p − p̲_s) / (p̄_s − p̲_s)
Comprador:  u_b(p) = (p̄_b − p) / (p̄_b − p̲_b)

Variáveis:
  p      preço final acordado (extraído do transcript pelo juiz LLM)
  p̄_s   preço alvo do vendedor  (melhor resultado esperado — máximo desejável)
  p̲_s   preço mínimo aceitável do vendedor (piso / BATNA)
  p̄_b   preço máximo aceitável do comprador (teto / BATNA)
  p̲_b   preço alvo do comprador  (melhor resultado esperado — mínimo desejável)

Interpretação do resultado:
  u = 1.0   → obteve exatamente o valor alvo
  u = 0.0   → obteve exatamente o valor mínimo aceitável (piso/teto)
  u > 1.0   → superou o valor alvo (muito bom)
  u < 0.0   → ficou abaixo do piso / acima do teto (inaceitável)
  u = None  → não houve acordo ou não foi possível extrair o preço

Referências
-----------
  Raiffa, H. (1982). The Art and Science of Negotiation.
  Lax, D. A., & Sebenius, J. K. (1986). The Manager as Negotiator.
"""

import json
import re
import logging
from dataclasses import dataclass
from typing import Literal, Optional

from ..adapters.base import LLMAdapter
from ..simulation.engine import NegotiationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parâmetros e resultado
# ---------------------------------------------------------------------------

@dataclass
class RoleUtilityParams:
    """
    Parâmetros de utilidade para um papel na negociação.

    Defina no config.yaml e passe para UtilityCalculator.

    Exemplo (negociação salarial):
        candidate → role_type="buyer"  (quer o MAIOR salário possível)
            p_target = 18000  # alvo do candidato
            p_floor  = 15500  # BATNA (oferta concorrente)

        recruiter → role_type="seller" (quer pagar o MENOR salário possível)
            p_target = 14000  # alvo do recrutador
            p_floor  = 16500  # teto orçamentário

    Nota: "seller" e "buyer" são convenções matemáticas, não literais.
    Em negociações salariais, o candidato age como "comprador" do salário
    (quer maximizar) e o recrutador como "vendedor" (quer minimizar o custo).
    """
    role: str
    role_type: Literal["seller", "buyer"]
    p_target: float   # p̄_s (alvo do vendedor) ou p̲_b (alvo do comprador)
    p_floor: float    # p̲_s (mínimo do vendedor) ou p̄_b (máximo do comprador)
    currency: str = "R$"
    unit: str = ""    # ex: "/mês", "/ano"


@dataclass
class UtilityResult:
    """Resultado do cálculo de utilidade para um agente."""
    role: str
    role_type: str
    agreed_price: Optional[float]
    utility: Optional[float]        # None se não houve acordo
    params: RoleUtilityParams
    settled: bool
    extraction_raw: str = ""        # resposta bruta do juiz ao extrair o preço
    note: str = ""

    @property
    def interpretation(self) -> str:
        if self.utility is None:
            return "Sem acordo — utilidade não calculável."
        if self.utility >= 1.0:
            return f"Superou o valor alvo (u={self.utility:.3f} ≥ 1.0). Excelente resultado."
        if self.utility >= 0.7:
            return f"Próximo ao valor alvo (u={self.utility:.3f}). Bom resultado."
        if self.utility >= 0.3:
            return f"Resultado mediano (u={self.utility:.3f}). Aceitável mas distante do alvo."
        if self.utility >= 0.0:
            return f"Próximo ao piso (u={self.utility:.3f}). Resultado fraco."
        return f"Abaixo do piso aceitável (u={self.utility:.3f}). Acordo desvantajoso."

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "role_type": self.role_type,
            "agreed_price": self.agreed_price,
            "utility": self.utility,
            "settled": self.settled,
            "p_target": self.params.p_target,
            "p_floor": self.params.p_floor,
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# Prompt para extração do preço acordado
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM = """You are a precise data extraction assistant.
Your task is to read a negotiation transcript and extract the final agreed price.

Rules:
- If an agreement was explicitly reached, extract the numeric price value only.
- If no agreement was reached, return null for "price".
- Do NOT include currency symbols in the numeric value.
- Respond ONLY with valid JSON. No markdown, no explanation.

JSON schema:
{
  "settled": <boolean>,
  "price": <number or null>,
  "currency": "<string or null>",
  "justification": "<one sentence explaining where in the transcript you found this>"
}"""

_EXTRACT_USER = """Negotiation transcript:
\"\"\"
{transcript}
\"\"\"

Extract the final agreed price. If no agreement was reached, set settled=false and price=null."""


# ---------------------------------------------------------------------------
# Calculador de utilidade
# ---------------------------------------------------------------------------

class UtilityCalculator:
    """
    Calcula utilidade econômica para cada papel da negociação.

    Fluxo:
      1. Usa um LLM-juiz para extrair o preço final acordado do transcript.
      2. Aplica as fórmulas matemáticas com os parâmetros do cenário.

    Args:
        judge:       LLMAdapter para extração do preço (pode ser o mesmo juiz de scoring).
        role_params: Dict role → RoleUtilityParams com os valores alvo e piso/teto.
    """

    def __init__(
        self,
        judge: LLMAdapter,
        role_params: dict[str, RoleUtilityParams],
    ):
        self.judge = judge
        self.role_params = role_params

    def evaluate(self, result: NegotiationResult) -> dict[str, UtilityResult]:
        """
        Avalia a utilidade de todos os papéis configurados.

        Returns:
            Dict role → UtilityResult
        """
        # 1. Extrai o preço acordado do transcript
        agreed_price, settled, raw = self._extract_price(result)

        # 2. Calcula utilidade para cada papel
        results: dict[str, UtilityResult] = {}
        for role, params in self.role_params.items():
            utility = None
            note = ""
            if settled and agreed_price is not None:
                try:
                    utility = self._calculate(agreed_price, params)
                except ZeroDivisionError:
                    note = "Divisão por zero: p_target == p_floor."
                    logger.warning("Utilidade de '%s': p_target == p_floor.", role)

            results[role] = UtilityResult(
                role=role,
                role_type=params.role_type,
                agreed_price=agreed_price if settled else None,
                utility=utility,
                params=params,
                settled=settled,
                extraction_raw=raw,
                note=note,
            )

        return results

    # ------------------------------------------------------------------
    # Fórmulas matemáticas
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate(p: float, params: RoleUtilityParams) -> float:
        """Aplica a fórmula de utilidade conforme o tipo de papel."""
        if params.role_type == "seller":
            # u_s(p) = (p − p̲_s) / (p̄_s − p̲_s)
            return (p - params.p_floor) / (params.p_target - params.p_floor)
        else:
            # u_b(p) = (p̄_b − p) / (p̄_b − p̲_b)
            return (params.p_floor - p) / (params.p_floor - params.p_target)

    # ------------------------------------------------------------------
    # Extração do preço via LLM
    # ------------------------------------------------------------------

    def _extract_price(
        self, result: NegotiationResult
    ) -> tuple[Optional[float], bool, str]:
        """
        Pede ao juiz para extrair o preço final do transcript.

        Returns:
            (agreed_price, settled, raw_response)
        """
        # Monta versão resumida do transcript (últimos 8 turnos são mais relevantes)
        turns = result.transcript[-8:] if len(result.transcript) > 8 else result.transcript
        transcript_text = "\n\n".join(
            f"[Turn {t.turn_index} | {t.role.upper()}]\n{t.content}"
            for t in turns
        )

        messages = [
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user",   "content": _EXTRACT_USER.format(transcript=transcript_text)},
        ]

        try:
            raw = self.judge.complete(messages)
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            start = clean.find('{')
            end   = clean.rfind('}')
            if start == -1 or end == -1 or end <= start:
                raise ValueError("Nenhum JSON encontrado na resposta.")
            parsed = json.loads(clean[start:end+1])
            settled = bool(parsed.get("settled", False))
            price_raw = parsed.get("price")
            price = float(price_raw) if price_raw is not None else None
            return price, settled, raw
        except Exception as e:
            logger.warning("UtilityCalculator: falha ao extrair preço: %s", e)
            return None, result.settled, ""
