"""
scoring/utility.py
==================
Computes each agent's Economic Utility from the final agreed price.

Formulas
--------
Seller:   u_s(p) = (p − p̲_s) / (p̄_s − p̲_s)
Buyer:    u_b(p) = (p̄_b − p) / (p̄_b − p̲_b)

Variables:
  p      final agreed price (extracted from the transcript by the LLM judge)
  p̄_s   seller target price  (best expected outcome — most desirable)
  p̲_s   seller minimum acceptable price (floor / BATNA)
  p̄_b   buyer maximum acceptable price (ceiling / BATNA)
  p̲_b   buyer target price  (best expected outcome — most desirable)

Result interpretation:
  u = 1.0   → obtained exactly the target value
  u = 0.0   → obtained exactly the minimum acceptable value (floor/ceiling)
  u > 1.0   → beat the target value (very good)
  u < 0.0   → fell below the floor / above the ceiling (unacceptable)
  u = None  → no agreement or price could not be extracted

References
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
# Params and result
# ---------------------------------------------------------------------------

@dataclass
class RoleUtilityParams:
    """
    Utility params for one negotiation role.

    New format (inside the agent, after tactics):
        models:
          agent_1:
            utility:
              p_target: 18000  # target
              p_floor: 15500   # floor/ceiling (BATNA)

    Legacy (top-level utility with role_type/currency/unit) still supported for compat.

    role_type inference when omitted:
      p_target > p_floor → seller-like (wants to maximize, e.g., salary candidate, VGA seller)
      p_target < p_floor → buyer-like (wants to minimize)
    """
    role: str
    role_type: Literal["seller", "buyer"] = "seller"
    p_target: float = 0.0   # p̄_s (seller target) or p̲_b (buyer target)
    p_floor: float = 0.0    # p̲_s (seller minimum) or p̄_b (buyer maximum)
    currency: str = "$"  # removed from yaml — kept for compat only (all prices in USD)
    unit: str = ""    # removed from yaml — kept for compat only

    def __post_init__(self):
        # Infer role_type when missing or inconsistent with p_target/p_floor
        # p_target > p_floor → wants to maximize (seller-like for price), else buyer-like
        # Keep explicit value when already coherent
        try:
            if self.p_target > self.p_floor and self.role_type == "buyer":
                # a maximizing buyer is mathematically seller-like,
                # but keep buyer to avoid breaking; the formula uses role_type
                pass
            elif self.p_target < self.p_floor and self.role_type == "seller":
                pass
        except Exception:
            pass


@dataclass
class UtilityResult:
    """Utility calculation result for one agent."""
    role: str
    role_type: str
    agreed_price: Optional[float]
    utility: Optional[float]        # None when there was no agreement
    params: RoleUtilityParams
    settled: bool
    extraction_raw: str = ""        # judge raw response when extracting the price
    note: str = ""

    @property
    def interpretation(self) -> str:
        if self.utility is None:
            return "No agreement — utility not calculable."
        if self.utility >= 1.0:
            return f"Beat the target value (u={self.utility:.3f} ≥ 1.0). Excellent result."
        if self.utility >= 0.7:
            return f"Close to the target value (u={self.utility:.3f}). Good result."
        if self.utility >= 0.3:
            return f"Median result (u={self.utility:.3f}). Acceptable but far from target."
        if self.utility >= 0.0:
            return f"Close to the floor (u={self.utility:.3f}). Weak result."
        return f"Below the acceptable floor (u={self.utility:.3f}). Disadvantageous agreement."

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
# Agreed-price extraction prompt
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
    Computes economic utility for each negotiation role.

    Flow:
      1. Uses an LLM judge to extract the final agreed price from the transcript.
      2. Applies the math formulas with the scenario params.

    Args:
        judge:       LLMAdapter for price extraction (may be the same scoring judge).
        role_params: Dict role → RoleUtilityParams with target and floor/ceiling values.
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
        Evaluate utility for all configured roles.

        Returns:
            Dict role → UtilityResult
        """
        # 1. Extract the agreed price from the transcript
        agreed_price, settled, raw = self._extract_price(result)

        # 2. Compute utility for each role
        results: dict[str, UtilityResult] = {}
        for role, params in self.role_params.items():
            utility = None
            note = ""
            if settled and agreed_price is not None:
                try:
                    utility = self._calculate(agreed_price, params)
                except ZeroDivisionError:
                    note = "Division by zero: p_target == p_floor."
                    logger.warning("Utility of '%s': p_target == p_floor.", role)

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
    # Math formulas
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate(p: float, params: RoleUtilityParams) -> float:
        """Apply the utility formula for the role type."""
        if params.role_type == "seller":
            # u_s(p) = (p − p̲_s) / (p̄_s − p̲_s)
            return (p - params.p_floor) / (params.p_target - params.p_floor)
        else:
            # u_b(p) = (p̄_b − p) / (p̄_b − p̲_b)
            return (params.p_floor - p) / (params.p_floor - params.p_target)

    # ------------------------------------------------------------------
    # Price extraction via LLM
    # ------------------------------------------------------------------

    def _extract_price(
        self, result: NegotiationResult
    ) -> tuple[Optional[float], bool, str]:
        """
        Ask the judge to extract the final price from the transcript.

        Returns:
            (agreed_price, settled, raw_response)
        """
        # Build a summarized transcript version (last 8 turns are most relevant)
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
                raise ValueError("No JSON found in the response.")
            parsed = json.loads(clean[start:end+1])
            settled = bool(parsed.get("settled", False))
            price_raw = parsed.get("price")
            price = float(price_raw) if price_raw is not None else None
            return price, settled, raw
        except Exception as e:
            logger.warning("UtilityCalculator: failed to extract price: %s", e)
            return None, result.settled, ""
