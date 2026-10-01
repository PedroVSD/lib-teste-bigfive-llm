"""
Negotiation scenario definitions.

A NegotiationScenario defines:
  - The shared context (given to both agents)
  - Role-specific system prompts (private to each agent)
  - The opening move structure

Scenarios are declarative dataclasses — they contain no logic.
The SimulationEngine consumes them.
"""

from dataclasses import dataclass, field


@dataclass
class NegotiationScenario:
    """
    Defines a negotiation scenario.

    Attributes:
        name:             Short identifier (used in filenames and reports).
        description:      Human-readable summary for reports.
        shared_context:   Text shown to both agents before the negotiation starts.
                          Should be neutral (no private goals here).
        roles:            Dict mapping role_id → system_prompt.
                          System prompt should include the role's private goal,
                          BATNA (Best Alternative To Negotiated Agreement),
                          and any constraints.
        opening_role:     Which role speaks first. The opening agent generates
                          the first message freely from its role/persona/context —
                          there is no fixed opening prompt.
        max_turns:        Maximum number of rounds (each agent speaking once = 1 round).
        settlement_keywords: Optional list of phrases that signal agreement
                             (used by engine to detect early termination).
        metadata:         Arbitrary dict for extra info (domain, difficulty, etc.)
    """
    name: str
    description: str
    shared_context: str
    roles: dict[str, str]                   # {"buyer": "system prompt...", "seller": "..."}
    opening_role: str
    max_turns: int = 8
    settlement_keywords: list[str] = field(default_factory=lambda: [
        "we have a deal", "agreed", "aceito", "fechado", "deal", "acordo"
    ])
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Built-in scenarios
# ---------------------------------------------------------------------------

# 1. Salary negotiation
SALARY_NEGOTIATION = NegotiationScenario(
    name="salary_negotiation",
    description="Negotiation scenario between an experienced technology professional and a company after a job offer. Evaluates salary anchoring, justification, concessions, value creation, and subjective factors.",
    shared_context=(
        "An experienced software engineer has received a job offer from a technology company. "
        "Both sides are interested in reaching an agreement, but neither knows the other side's limit. "
        "Beyond salary, the parties can negotiate bonuses, remote work, vacation, benefits, and working hours. "
        "IMPORTANT: If an agreement is definitively reached by both parties, you MUST "
        "include the exact phrase 'SIMULACAO_CONCLUIDA' at the end of your response."
    ),
    roles={
        "candidate": (
            "You are an experienced software engineer negotiating a new opportunity. You have a stable job "
            "and do not need to accept just any offer. You value compensation, professional growth, "
            "flexibility, recognition, and stability. Do not reveal your minimum threshold without strategic need. "
            "Negotiate assertively and professionally."
        ),
        "recruiter": (
            "You are the hiring manager. There is some budget flexibility, but raises must be justified. "
            "You consider compensation, internal pay equity, employee retention, and the cost of hiring someone else. "
            "You have a maximum limit, but you must not reveal it directly. "
            "Negotiate professionally seeking to close the hire within what is possible."
        ),
    },
    opening_role="recruiter",
    max_turns=8,
    settlement_keywords=[
        "SIMULACAO_CONCLUIDA",
        "ACORDO_FECHADO",
        "[ACORDO_FECHADO]",
    ],
    metadata={"domain": "HR", "currency": "BRL", "difficulty": "medium", "label": "Salary negotiation",
              "aspects": "anchor, utility, subjective valuation, conditional concessions, loss aversion, value creation, rapport"},
)

# 2. Company acquisition
COMPANY_ACQUISITION = NegotiationScenario(
    name="company_acquisition",
    description="Negotiation between the founder of a small technology company and a larger company interested in acquiring it. Explores financial anchors, asymmetric information, risk, utility, and value creation through deal structure.",
    shared_context=(
        "The founder of a small technology company is negotiating its sale with a larger company in the same industry. "
        "The company owns valuable intellectual property and important clients, but there is uncertainty about its future growth. "
        "Beyond price, the parties can negotiate upfront payment, performance-based payments (earn-out), founder retention, "
        "management participation, and intellectual property rights. "
        "IMPORTANT: If an agreement is definitively reached by both parties, you MUST "
        "include the exact phrase 'SIMULACAO_CONCLUIDA' at the end of your response."
    ),
    roles={
        "seller": (
            "You are the company founder. You value receiving a large share of the money immediately and would like to keep "
            "some influence over the company's future. You believe the technological potential is greater than current financial "
            "results show. You have a minimum acceptable value, but you must not reveal it. Negotiate defending your valuation "
            "while showing flexibility on deal structure."
        ),
        "buyer": (
            "You represent a larger company interested in the acquisition. You are concerned about integration, client retention, "
            "and future performance. You have more flexibility on deal structure than on increasing the upfront payment. "
            "You can use performance bonuses, earn-outs, or retention contracts to compose the total value. Negotiate seeking to "
            "reduce risk and justify your valuation."
        ),
    },
    opening_role="seller",
    max_turns=10,
    settlement_keywords=[
        "SIMULACAO_CONCLUIDA",
        "ACORDO_FECHADO",
        "[ACORDO_FECHADO]",
    ],
    metadata={"domain": "M&A", "currency": "BRL", "difficulty": "hard", "label": "Company acquisition",
              "aspects": "anchor, utility, value creation, conditional concessions, loss aversion, risk, asymmetric information"},
)

# 3. Supplier contract
STRATEGIC_SUPPLIER_CONTRACT = NegotiationScenario(
    name="strategic_supplier_contract",
    description="Commercial negotiation between an industrial company and a strategic supplier. Scenario with multiple negotiable variables to test value creation, conditional concessions, and different utility functions.",
    shared_context=(
        "An industrial company needs to negotiate an annual contract with a strategic raw-material supplier. "
        "The supplier initially proposed $240 per unit. The buyer considers that a competitive price would be near "
        "$190. However, price is not the only important element. The parties can negotiate minimum purchase volume, payment terms, "
        "delivery times, quality, contract duration, warranties, and late-delivery penalties. "
        "IMPORTANT: If an agreement is definitively reached by both parties, you MUST "
        "include the exact phrase 'SIMULACAO_CONCLUIDA' at the end of your response."
    ),
    roles={
        "buyer": (
            "You represent the purchasing department. Your main goal is to reduce total acquisition cost, but supply "
            "reliability is extremely important. You prefer to pay a bit more for a reliable supplier "
            "than to risk production interruptions. Alternative suppliers exist, but switching would create relevant "
            "operational costs. Negotiate seeking a better price without sacrificing reliability."
        ),
        "supplier": (
            "You represent the supplier. Your initial proposal is $240 per unit. You want a long-term contract and "
            "predictable demand. You are willing to reduce the price if you receive a larger minimum volume or faster payments. You consider "
            "contractual penalties particularly risky and prefer to avoid them. Negotiate defending your price while offering trade-offs."
        ),
    },
    opening_role="supplier",
    max_turns=10,
    settlement_keywords=[
        "SIMULACAO_CONCLUIDA",
        "ACORDO_FECHADO",
        "[ACORDO_FECHADO]",
    ],
    metadata={"domain": "Supply Chain", "currency": "BRL", "difficulty": "hard", "label": "Supplier contract",
              "aspects": "anchor, utility, value creation, trade-offs, conditional concessions, loss aversion, clarity"},
)

# 4. Property dispute
PROPERTY_BOUNDARY_DISPUTE = NegotiationScenario(
    name="property_boundary_dispute",
    description="Negotiation between two neighboring property owners disputing land boundaries. Reduces pure financial importance and increases perceived fairness, emotions, relationship, and subjective valuation.",
    shared_context=(
        "Two neighboring owners disagree about the boundary between their properties. One of them claims that a recently "
        "built wall occupies approximately 12 square meters of their land. The other believes the wall was built correctly. "
        "A lawsuit would be expensive and could take months, so both sides are interested in finding a private solution. "
        "Alternatives include moving the wall, paying financial compensation, exchanging a small land area, splitting legal "
        "costs, or establishing a permanent land-use agreement. "
        "IMPORTANT: If an agreement is definitively reached by both parties, you MUST "
        "include the exact phrase 'SIMULACAO_CONCLUIDA' at the end of your response."
    ),
    roles={
        "owner_a": (
            "You are Owner A. You believe approximately 12 square meters of your land were occupied by your neighbor's wall. "
            "Your initial demand is $16,000. You strongly value perceived fairness and feel your neighbor acted disrespectfully. However, you would accept "
            "lower compensation if your neighbor acknowledges the problem and accepts a solution you consider fair. "
            "Negotiate seeking fairness and recognition."
        ),
        "owner_b": (
            "You are Owner B. You believe the wall is correctly placed and reject the accusation of having deliberately "
            "occupied your neighbor's land. You do not want to move the wall because that would be expensive and disruptive. You are willing "
            "to discuss financial compensation or other alternatives if they avoid rebuilding the wall and reduce the risk of a prolonged "
            "legal dispute. Negotiate seeking to avoid costs and litigation."
        ),
    },
    opening_role="owner_a",
    max_turns=10,
    settlement_keywords=[
        "SIMULACAO_CONCLUIDA",
        "ACORDO_FECHADO",
        "[ACORDO_FECHADO]",
    ],
    metadata={"domain": "Property", "currency": "BRL", "difficulty": "hard", "label": "Property dispute",
              "aspects": "anchor, utility, subjective valuation, perceived fairness, loss aversion, rapport, resilience"},
)

# 5. Computer part purchase — GPU
VGA_PURCHASE = NegotiationScenario(
    name="vga_purchase",
    description="Negotiation over a graphics card (GPU) purchase between an experienced seller and a young buyer. Evaluates persuasion, price research, anchoring, concessions, and deliberate decision-making.",
    shared_context=(
        "A computer store is negotiating the sale of a highly demanded graphics card (GPU) for gaming and work. "
        "The store has the GPU in stock ready for delivery, with warranty and installment options. "
        "The buyer needs the part to build a PC and use it for work as a developer. Both sides want to close, "
        "but they diverge on price and conditions. Beyond price, the parties can negotiate extended warranty, installments, "
        "upfront discount, bundles, shipping, and delivery time. "
        "IMPORTANT: If an agreement is definitively reached by both parties, you MUST "
        "include the exact phrase 'SIMULACAO_CONCLUIDA' at the end of your response."
    ),
    roles={
        "seller": (
            "You are the store salesperson, with 10 years of experience selling hardware. You know the product deeply "
            "and know how to argue. You value closing the sale with a good margin, but you prefer granting benefits "
            "over lowering the price too much. You have a minimum limit, but you must not reveal it. Be persuasive, professional "
            "and empathetic, but firm in defending value."
        ),
        "buyer": (
            "You are the buyer: a young person building a PC, a newly hired junior developer. You always research "
            "prices and think carefully before deciding. Your budget is limited and you need cost-benefit and installments. "
            "You value fair price, warranty, origin, and installments. Do not reveal your maximum limit without strategy. "
            "Negotiate prudently, asking for data, comparing offers, and proposing conditional trades."
        ),
    },
    opening_role="seller",
    max_turns=8,
    settlement_keywords=[
        "SIMULACAO_CONCLUIDA",
        "ACORDO_FECHADO",
        "[ACORDO_FECHADO]",
    ],
    metadata={"domain": "Retail", "currency": "BRL", "difficulty": "medium", "label": "GPU purchase",
              "aspects": "anchor, price research, conditional concessions, value creation, persuasion, deliberate decision"},
)

# Registry for easy lookup
SCENARIO_REGISTRY: dict[str, NegotiationScenario] = {
    s.name: s for s in [
        SALARY_NEGOTIATION, COMPANY_ACQUISITION, STRATEGIC_SUPPLIER_CONTRACT, PROPERTY_BOUNDARY_DISPUTE, VGA_PURCHASE
    ]
}
