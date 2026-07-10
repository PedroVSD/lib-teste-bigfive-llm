"""
context/situational.py
======================
Define e injeta contexto macroeconômico e situacional no system prompt
dos agentes negociadores.

Propósito
---------
Permite simular como condições externas (inflação, taxa de juros, crises,
governo, etc.) influenciam o comportamento dos agentes durante a negociação.

O mecanismo é opcional e pode ser desligado por agente ou globalmente,
passando enabled=False no SituationalContext ou simplesmente não fornecendo
um contexto ao SimulationEngine.

Design
------
- SituationalContext é um dataclass com campos opcionais por categoria.
- ContextPromptBuilder transforma os campos em um bloco instrucional.
- O bloco é ANEXADO ao system prompt (após a persona, se houver).
- Tudo que está em SituationalContext é salvo em metadata do NegotiationResult.

Exemplo de output gerado
------------------------
--- Economic & Situational Context ---
The negotiation takes place under the following external conditions.
Both parties are aware of this context and should let it influence
their reasoning, urgency, and risk tolerance.

Macroeconomic environment:
  Inflation is very high (above 10% annually). Purchasing power is eroding
  quickly. Both parties should factor rising costs into any multi-period
  commitments.

Interest rates:
  Interest rates are high. Credit is expensive, making financing-dependent
  deals harder to justify and increasing the cost of delayed agreements.

Political/institutional environment:
  The current government has a market-friendly orientation, favoring
  deregulation and private sector activity.

Active crisis:
  There is an active economic recession. Unemployment is rising and
  consumer confidence is low. Both parties face increased pressure to
  close deals quickly rather than risk prolonged uncertainty.

Custom conditions:
  The company recently went through a round of layoffs, creating internal
  pressure to reduce headcount costs further.
--------------------------------------
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


# ---------------------------------------------------------------------------
# Enums for structured fields
# ---------------------------------------------------------------------------

class InflationLevel(str, Enum):
    VERY_LOW   = "very_low"    # < 2%
    LOW        = "low"         # 2–4%
    MODERATE   = "moderate"    # 4–7%
    HIGH       = "high"        # 7–10%
    VERY_HIGH  = "very_high"   # > 10%


class InterestRateLevel(str, Enum):
    VERY_LOW  = "very_low"
    LOW       = "low"
    MODERATE  = "moderate"
    HIGH      = "high"
    VERY_HIGH = "very_high"


class GovernmentOrientation(str, Enum):
    MARKET_FRIENDLY   = "market_friendly"
    INTERVENTIONIST   = "interventionist"
    TECHNOCRATIC      = "technocratic"
    POPULIST          = "populist"
    TRANSITIONAL      = "transitional"   # e.g., post-election uncertainty
    CONSERVATIVE      = "conservative"
    ANCAP             = "anarcho_capitalist"
    LIBERAL_ON_MARKET = "austrian"


class CrisisType(str, Enum):
    ECONOMIC_RECESSION    = "economic_recession"
    FINANCIAL_CRISIS      = "financial_crisis"
    POLITICAL_INSTABILITY = "political_instability"
    HEALTH_PANDEMIC       = "health_pandemic"
    SUPPLY_CHAIN          = "supply_chain_disruption"
    ENERGY_CRISIS         = "energy_crisis"
    GEOPOLITICAL          = "geopolitical_conflict"
    CURRENCY_CRISIS       = "currency_crisis"


# ---------------------------------------------------------------------------
# Natural language descriptions
# ---------------------------------------------------------------------------

_INFLATION_DESC: dict[InflationLevel, str] = {
    InflationLevel.VERY_LOW: (
        "Inflation is very low (below 2%). Prices are stable and purchasing "
        "power is preserved. Parties can make long-term commitments with "
        "confidence in nominal values."
    ),
    InflationLevel.LOW: (
        "Inflation is low (2–4%). The economic environment is stable. "
        "Price adjustments are mild and predictable."
    ),
    InflationLevel.MODERATE: (
        "Inflation is moderate (4–7%). Parties should consider indexation "
        "clauses in multi-year agreements. Cost increases are noticeable "
        "but manageable."
    ),
    InflationLevel.HIGH: (
        "Inflation is high (7–10%). Purchasing power is eroding meaningfully. "
        "Parties should be cautious about fixed nominal commitments over "
        "long periods. Urgency to close deals increases."
    ),
    InflationLevel.VERY_HIGH: (
        "Inflation is very high (above 10%). Purchasing power is eroding "
        "quickly. Both parties should factor rising costs into any "
        "multi-period commitments. Resistance to nominal price increases "
        "may be lower than usual, but budget constraints are tighter."
    ),
}

_INTEREST_DESC: dict[InterestRateLevel, str] = {
    InterestRateLevel.VERY_LOW: (
        "Interest rates are very low. Credit is cheap and capital is abundant. "
        "Financing-dependent deals are attractive. Parties have low urgency "
        "to avoid delayed agreements."
    ),
    InterestRateLevel.LOW: (
        "Interest rates are low. Financing is accessible and affordable. "
        "Deals with deferred payments or installments are relatively easy to structure."
    ),
    InterestRateLevel.MODERATE: (
        "Interest rates are at a moderate level. Financing costs are "
        "meaningful but not prohibitive. Parties should weigh the cost "
        "of delaying deals against their financing options."
    ),
    InterestRateLevel.HIGH: (
        "Interest rates are high. Credit is expensive, making "
        "financing-dependent deals harder to justify and increasing the "
        "cost of delayed agreements. Cash deals may be preferred."
    ),
    InterestRateLevel.VERY_HIGH: (
        "Interest rates are very high. Credit is very expensive and many "
        "parties may face liquidity constraints. There is strong pressure "
        "to conclude deals quickly and avoid prolonged negotiations. "
        "Deferred payment structures are unattractive."
    ),
}

_GOVERNMENT_DESC: dict[GovernmentOrientation, str] = {
    GovernmentOrientation.MARKET_FRIENDLY: (
        "The current government has a market-friendly orientation, favoring "
        "deregulation and private sector activity. Regulatory risk is low "
        "and contract enforcement is generally reliable."
    ),
    GovernmentOrientation.INTERVENTIONIST: (
        "The current government is interventionist, with active regulation "
        "of prices, wages, and business activities. Regulatory risk is higher. "
        "Parties should consider the possibility of price controls or policy "
        "changes affecting the deal."
    ),
    GovernmentOrientation.TECHNOCRATIC: (
        "The government is technocratic and rules-based. Policy is predictable "
        "and based on technical criteria. Institutional trust is high."
    ),
    GovernmentOrientation.POPULIST: (
        "The government has a populist orientation. Policy can be unpredictable "
        "and may shift based on political pressures. Both parties should "
        "consider political risk in long-term commitments."
    ),
    GovernmentOrientation.TRANSITIONAL: (
        "The government is in a transitional or post-election period. "
        "Policy direction is uncertain. Both parties should treat long-term "
        "regulatory assumptions with caution."
    ),
    GovernmentOrientation.CONSERVATIVE: (
        "The current government is conservative, prioritizing institutional "
        "stability, traditional values, and gradual policy evolution. Regulatory "
        "changes tend to be incremental, and contract enforcement is generally "
        "reliable. Market conditions are stable, though reforms may proceed cautiously."
    ),
    GovernmentOrientation.ANCAP: (
        "The government follows an anarcho-capitalist orientation, with minimal "
        "or absent state intervention in economic activities. Regulation is minimal as possible "
        "limited, and market forces dictate contractual and commercial outcomes. "
        "While flexibility is high, parties should account for potential gaps in "
        "institutional enforcement and legal recourse."
        "He draws inspiration from Murray Rothbard, Hans Hermann Hoppe, David D. Friedman and other economists who follow the same path."
        ),
    GovernmentOrientation.LIBERAL_ON_MARKET: (
        "The government adopts a market-liberal stance influenced by Austrian "
        "economic principles, emphasizing free markets, low regulation, and "
        "sound monetary policy. Regulatory risk is low, and economic policy "
        "favors private sector initiative and capital efficiency. Parties can "
        "generally expect a predictable and business-friendly environment."
        "He draws inspiration from Friedrich Hayek, Mises, Thomas Sowell, and other economists who follow the same path."
        ),
}

_CRISIS_DESC: dict[CrisisType, str] = {
    CrisisType.ECONOMIC_RECESSION: (
        "There is an active economic recession. Unemployment is rising and "
        "consumer confidence is low. Both parties face increased pressure to "
        "close deals quickly rather than risk prolonged uncertainty. "
        "Concessions may be more available from the weaker party."
    ),
    CrisisType.FINANCIAL_CRISIS: (
        "A financial crisis is underway. Credit markets are stressed and "
        "liquidity is scarce. Counterparty risk is elevated. Parties should "
        "prefer simpler deal structures and avoid deferred payment risk."
    ),
    CrisisType.POLITICAL_INSTABILITY: (
        "There is significant political instability. Institutional trust is "
        "low and contract enforcement may be uncertain. Both parties may seek "
        "shorter commitment periods and stronger exit clauses."
    ),
    CrisisType.HEALTH_PANDEMIC: (
        "A health pandemic is affecting economic activity. Supply chains and "
        "demand are disrupted. Remote work is common. Both parties face "
        "elevated uncertainty about future conditions."
    ),
    CrisisType.SUPPLY_CHAIN: (
        "Supply chain disruptions are active. Delivery timelines are unreliable "
        "and input costs are volatile. Parties should factor supply risk into "
        "pricing and delivery commitments."
    ),
    CrisisType.ENERGY_CRISIS: (
        "An energy crisis is underway. Energy costs are elevated and supply "
        "is uncertain. Operating costs for businesses are higher than usual, "
        "increasing pressure to reduce costs elsewhere."
    ),
    CrisisType.GEOPOLITICAL: (
        "A geopolitical conflict is creating international uncertainty. "
        "Trade flows may be disrupted and currency volatility is higher. "
        "Both parties should consider the risk of external shocks to the deal."
    ),
    CrisisType.CURRENCY_CRISIS: (
        "A currency crisis is affecting the economy. Exchange rate volatility "
        "is high. Deals denominated in local currency carry devaluation risk. "
        "Parties may prefer foreign currency or indexed clauses."
    ),
}


# ---------------------------------------------------------------------------
# SituationalContext dataclass
# ---------------------------------------------------------------------------

@dataclass
class SituationalContext:
    """
    Defines the macroeconomic and situational conditions for a negotiation.

    All fields are optional. Only specified fields are injected into the prompt.
    Set enabled=False to disable context injection entirely without removing
    the object from your code — useful for A/B testing with and without context.

    Fields
    ------
    enabled             : Master switch. False = no injection at all.
    inflation           : InflationLevel enum or None.
    interest_rates      : InterestRateLevel enum or None.
    government          : GovernmentOrientation enum or None.
    crises              : List of active CrisisType (can be multiple simultaneous).
    gdp_growth          : Free-text GDP growth description é o PIB (e.g., "GDP contracted 2%").
    unemployment        : Free-text unemployment description(Nível de desemprego).
    custom_conditions   : List of free-text strings for any condition not covered above.
    year                : Optional year string for temporal grounding (e.g., "2025").
    country             : Optional country/region (e.g., "Brazil").

    Example
    -------
        ctx = SituationalContext(
            inflation=InflationLevel.HIGH,
            interest_rates=InterestRateLevel.HIGH,
            government=GovernmentOrientation.INTERVENTIONIST,
            crises=[CrisisType.ECONOMIC_RECESSION],
            country="Brazil",
            year="2025",
            custom_conditions=[
                "The company recently went through a round of layoffs."
            ],
        )
    """
    enabled:            bool                        = True
    inflation:          Optional[InflationLevel]    = None
    interest_rates:     Optional[InterestRateLevel] = None
    government:         Optional[GovernmentOrientation] = None
    crises:             list[CrisisType]            = field(default_factory=list)
    gdp_growth:         Optional[str]               = None
    unemployment:       Optional[str]               = None
    custom_conditions:  list[str]                   = field(default_factory=list)
    year:               Optional[str]               = None
    country:            Optional[str]               = None

    def is_active(self) -> bool:
        """True if enabled AND at least one condition is specified."""
        if not self.enabled:
            return False
        return any([
            self.inflation,
            self.interest_rates,
            self.government,
            self.crises,
            self.gdp_growth,
            self.unemployment,
            self.custom_conditions,
        ])

    def disable(self) -> "SituationalContext":
        """Return a copy of this context with enabled=False."""
        import copy
        c = copy.copy(self)
        c.enabled = False
        return c

    def to_dict(self) -> dict:
        """Serializable representation for storage/logging."""
        return {
            "enabled": self.enabled,
            "inflation": self.inflation.value if self.inflation else None,
            "interest_rates": self.interest_rates.value if self.interest_rates else None,
            "government": self.government.value if self.government else None,
            "crises": [c.value for c in self.crises],
            "gdp_growth": self.gdp_growth,
            "unemployment": self.unemployment,
            "custom_conditions": self.custom_conditions,
            "year": self.year,
            "country": self.country,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SituationalContext":
        return cls(
            enabled=d.get("enabled", True),
            inflation=InflationLevel(d["inflation"]) if d.get("inflation") else None,
            interest_rates=InterestRateLevel(d["interest_rates"]) if d.get("interest_rates") else None,
            government=GovernmentOrientation(d["government"]) if d.get("government") else None,
            crises=[CrisisType(c) for c in d.get("crises", [])],
            gdp_growth=d.get("gdp_growth"),
            unemployment=d.get("unemployment"),
            custom_conditions=d.get("custom_conditions", []),
            year=d.get("year"),
            country=d.get("country"),
        )

    @classmethod
    def disabled(cls) -> "SituationalContext":
        """Convenience: return an explicitly disabled context."""
        return cls(enabled=False)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

class ContextPromptBuilder:
    """
    Converts a SituationalContext into a natural language instruction block
    for injection into an LLM system prompt.

    Returns an empty string if the context is disabled or has no conditions,
    so it is always safe to call inject() unconditionally.
    """

    HEADER = "--- Economic & Situational Context ---"
    FOOTER = "--------------------------------------"

    def build(self, ctx: SituationalContext) -> str:
        if not ctx.is_active():
            return ""

        lines = [self.HEADER, ""]

        # Temporal/geographic grounding
        grounding_parts = []
        if ctx.year:
            grounding_parts.append(f"year {ctx.year}")
        if ctx.country:
            grounding_parts.append(ctx.country)

        if grounding_parts:
            lines.append(
                f"The negotiation takes place in {', '.join(grounding_parts)}."
            )
        else:
            lines.append("The negotiation takes place under the following external conditions.")

        lines.append(
            "Both parties are aware of this context and should let it influence "
            "their reasoning, urgency, and risk tolerance."
        )
        lines.append("")

        if ctx.inflation:
            lines.append("Macroeconomic environment:")
            lines.append(f"  {_INFLATION_DESC[ctx.inflation]}")
            lines.append("")

        if ctx.interest_rates:
            lines.append("Interest rates:")
            lines.append(f"  {_INTEREST_DESC[ctx.interest_rates]}")
            lines.append("")

        if ctx.gdp_growth:
            lines.append("GDP growth:")
            lines.append(f"  {ctx.gdp_growth}")
            lines.append("")

        if ctx.unemployment:
            lines.append("Unemployment:")
            lines.append(f"  {ctx.unemployment}")
            lines.append("")

        if ctx.government:
            lines.append("Political/institutional environment:")
            lines.append(f"  {_GOVERNMENT_DESC[ctx.government]}")
            lines.append("")

        for crisis in ctx.crises:
            lines.append("Active crisis:")
            lines.append(f"  {_CRISIS_DESC[crisis]}")
            lines.append("")

        if ctx.custom_conditions:
            lines.append("Custom conditions:")
            for cond in ctx.custom_conditions:
                lines.append(f"  {cond}")
            lines.append("")

        lines.append(self.FOOTER)
        return "\n".join(lines)

    def inject(self, system_prompt: str, ctx: SituationalContext) -> str:
        """
        Append the context block to an existing system prompt.
        Returns the original prompt unchanged if context is disabled.
        """
        block = self.build(ctx)
        if not block:
            return system_prompt
        return f"{system_prompt}\n\n{block}"


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

class ContextPresets:
    """Ready-made situational contexts for common experimental scenarios."""

    @staticmethod
    def brazil_2015_crisis() -> SituationalContext:
        """Brazil's 2015 economic and political crisis."""
        return SituationalContext(
            inflation=InflationLevel.HIGH,
            interest_rates=InterestRateLevel.VERY_HIGH,
            government=GovernmentOrientation.INTERVENTIONIST,
            crises=[CrisisType.ECONOMIC_RECESSION, CrisisType.POLITICAL_INSTABILITY],
            gdp_growth="GDP contracted approximately 3.5%.",
            unemployment="Unemployment rising sharply, reaching double digits.",
            country="Brazil",
            year="2015",
        )

    @staticmethod
    def post_pandemic_recovery() -> SituationalContext:
        """Post-COVID recovery period with supply chain stress."""
        return SituationalContext(
            inflation=InflationLevel.HIGH,
            interest_rates=InterestRateLevel.MODERATE,
            crises=[CrisisType.SUPPLY_CHAIN],
            gdp_growth="GDP recovering but below pre-pandemic trend.",
            custom_conditions=[
                "Remote work is still common and accepted.",
                "Supply lead times are 2–3x longer than pre-pandemic norms.",
            ],
            year="2022",
        )

    @staticmethod
    def stable_growth() -> SituationalContext:
        """Stable, low-risk macroeconomic environment."""
        return SituationalContext(
            inflation=InflationLevel.LOW,
            interest_rates=InterestRateLevel.LOW,
            government=GovernmentOrientation.TECHNOCRATIC,
            gdp_growth="GDP growing at approximately 3% annually.",
        )

    @staticmethod
    def financial_crisis() -> SituationalContext:
        """Acute financial crisis environment."""
        return SituationalContext(
            inflation=InflationLevel.MODERATE,
            interest_rates=InterestRateLevel.VERY_HIGH,
            crises=[CrisisType.FINANCIAL_CRISIS, CrisisType.ECONOMIC_RECESSION],
            custom_conditions=[
                "Credit markets are frozen. Banks are not lending.",
                "Several large firms have recently gone bankrupt.",
            ],
        )

    @staticmethod
    def geopolitical_tension() -> SituationalContext:
        """High geopolitical uncertainty with energy stress."""
        return SituationalContext(
            inflation=InflationLevel.HIGH,
            interest_rates=InterestRateLevel.HIGH,
            crises=[CrisisType.GEOPOLITICAL, CrisisType.ENERGY_CRISIS],
            custom_conditions=[
                "Trade routes are partially disrupted.",
                "Energy prices are 60% above their 5-year average.",
            ],
        )

    @staticmethod
    def disabled() -> SituationalContext:
        """Explicitly disabled context — no injection."""
        return SituationalContext.disabled()
