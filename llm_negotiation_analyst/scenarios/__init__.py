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
from typing import Optional


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
        opening_role:     Which role speaks first.
        opening_prompt:   Optional fixed first message (benchmark mode).
                          If None, the opening agent generates it freely.
        max_turns:        Maximum number of turns (each agent speaking once = 1 turn pair).
        settlement_keywords: Optional list of phrases that signal agreement
                             (used by engine to detect early termination).
        metadata:         Arbitrary dict for extra info (domain, difficulty, etc.)
    """
    name: str
    description: str
    shared_context: str
    roles: dict[str, str]                   # {"buyer": "system prompt...", "seller": "..."}
    opening_role: str
    opening_prompt: Optional[str] = None
    max_turns: int = 8
    settlement_keywords: list[str] = field(default_factory=lambda: [
        "we have a deal", "agreed", "aceito", "fechado", "deal", "acordo"
    ])
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Built-in scenarios
# ---------------------------------------------------------------------------
#Abaixo é como controla os cenários dos experimento. É possível criar cada cenário
SALARY_NEGOTIATION = NegotiationScenario(
    name="salary_negotiation",
    description="Job offer salary negotiation between a candidate and an HR recruiter.",
    shared_context=(
        "A technology company is hiring a mid-level data engineer. "
        "The candidate has received a formal job offer and is now negotiating "
        "compensation with the HR recruiter. The discussion covers base salary, "
        "signing bonus, and remote work policy. "
        "Both parties want to reach an agreement."
        "IMPORTANT: If an agreement is definitively reached by both parties, you MUST "
        "include the exact phrase 'SIMULACAO_CONCLUIDA' at the end of your response."
    ),
    roles={
        "candidate": (
            "You are a data engineer candidate negotiating your job offer. "
            "Your target salary is R$18,000/month. The current offer is R$14,000/month. "
            "Your BATNA is a competing offer of R$15,500/month with no signing bonus. "
            "You also want at least 3 days of remote work per week. "
            "Negotiate assertively but professionally. Do not reveal your BATNA unless strategically useful."
        ),
        "recruiter": (
            "You are an HR recruiter for a tech company. "
            "The approved budget for this role is up to R$16,500/month. "
            "You can offer a one-time signing bonus of up to R$5,000. "
            "Remote work policy allows up to 2 days/week for mid-level roles, "
            "with exceptions requiring VP approval. "
            "Your goal is to close the hire within budget while keeping the candidate motivated."
        ),
    },
    opening_role="recruiter",
    opening_prompt=(
        "Thank you for taking the time to speak with us today. "
        "We're very excited about the possibility of you joining our team. "
        "I'd like to discuss the details of our offer and see if we can make this work for both of us."
    ),
    max_turns=8,
    settlement_keywords=[
        "SIMULACAO_CONCLUIDA"
    ],
    metadata={"domain": "HR", "currency": "BRL", "difficulty": "medium"},
)

PROCUREMENT_NEGOTIATION = NegotiationScenario(
    name="procurement_b2b",
    description="B2B procurement negotiation for a software license contract.",
    shared_context=(
        "A mid-sized retail company (buyer) is negotiating a 2-year enterprise software "
        "license with a SaaS vendor (seller). The contract covers pricing, support SLA, "
        "and implementation timeline. Both parties have negotiated before and share "
        "a basic level of trust."
    ),
    roles={
        "buyer": (
            "You are the procurement manager for a retail chain. "
            "Your budget for this contract is R$150,000/year. "
            "The vendor's listed price is R$180,000/year. "
            "You need a 99.5% uptime SLA and implementation within 60 days. "
            "Your BATNA is a competing vendor offering R$130,000/year with a weaker SLA. "
            "Push for the best deal without sacrificing SLA quality."
        ),
        "seller": (
            "You are an enterprise account executive at a SaaS company. "
            "Your floor price for this deal is R$114,000/year (below this you need VP approval). "
            "You can offer 99.9% uptime SLA. Standard implementation is 90 days; "
            "expedited (60 days) costs an additional R$15,000. "
            "Your goal is to close a 2-year contract at or above R$150,000/year. "
            "You can offer free onboarding training as a sweetener."
        ),
    },
    opening_role="buyer",
    max_turns=10,
    settlement_keywords=[
        "[I AM BATMAN]"
    ],
    #settlement_keywords=[],
    metadata={"domain": "B2B SaaS", "currency": "BRL", "difficulty": "hard"},
)

HOSTAGE_CRISIS_DEBRIEF = NegotiationScenario(
    name="crisis_negotiation_training",
    description=(
        "Training scenario: a crisis negotiator attempts to de-escalate a "
        "barricade situation. Designed to stress-test Agreeableness and Neuroticism."
    ),
    shared_context=(
        "A distressed individual has locked themselves in an office building after "
        "a workplace conflict. A trained crisis negotiator is attempting to establish "
        "communication and achieve a peaceful resolution. This is a training simulation."
    ),
    roles={
        "negotiator": (
            "You are a trained police crisis negotiator. "
            "Your primary goal is the safe resolution of the situation without harm. "
            "Use active listening, empathy, and de-escalation techniques. "
            "Do not make promises you cannot keep. Build rapport gradually."
        ),
        "subject": (
            "You are a distressed employee who feels wronged by your company. "
            "You are emotionally volatile but not violent. "
            "You want to be heard and to have your grievances acknowledged. "
            "You are suspicious of authority but respond to genuine empathy."
        ),
    },
    opening_role="negotiator",
    opening_prompt="Hello, my name is Officer Santos. I'm here to listen. Can we talk?",
    max_turns=12,
    settlement_keywords=[
        "SIMULACAO_CONCLUIDA",
        "formalizar o acordo", "iniciar a implementação", "assinar o contrato",
        "confirmo os termos", "parceria firmada","we have a deal"
    ],
    metadata={"domain": "crisis", "difficulty": "hard", "note": "training only"},
)

# Registry for easy lookup
SCENARIO_REGISTRY: dict[str, NegotiationScenario] = {
    s.name: s for s in [
        SALARY_NEGOTIATION, PROCUREMENT_NEGOTIATION, HOSTAGE_CRISIS_DEBRIEF
    ]
}
