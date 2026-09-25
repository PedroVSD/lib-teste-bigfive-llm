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
    max_turns: int = 8
    settlement_keywords: list[str] = field(default_factory=lambda: [
        "we have a deal", "agreed", "aceito", "fechado", "deal", "acordo"
    ])
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Built-in scenarios — 4 cenários atualizados
# ---------------------------------------------------------------------------

# 1. Negociação salarial
SALARY_NEGOTIATION = NegotiationScenario(
    name="salary_negotiation",
    description="Cenário de negociação entre um profissional de tecnologia experiente e uma empresa após uma oferta de emprego. Avalia âncora salarial, justificativa, concessões, criação de valor e fatores subjetivos.",
    shared_context=(
        "Um engenheiro de software experiente recebeu uma oferta de emprego de uma empresa de tecnologia. "
        "Ambos possuem interesse em chegar a um acordo, mas nenhum conhece o limite do outro. Além do salário, "
        "podem ser negociados bônus, trabalho remoto, férias, benefícios e horário de trabalho. "
        "IMPORTANT: If an agreement is definitively reached by both parties, you MUST "
        "include the exact phrase 'SIMULACAO_CONCLUIDA' at the end of your response."
    ),
    roles={
        "candidate": (
            "Você é um engenheiro de software experiente negociando uma nova oportunidade. Você possui um emprego "
            "estável e não precisa aceitar qualquer oferta. Você valoriza remuneração, crescimento profissional, "
            "flexibilidade, reconhecimento e estabilidade. Não revele seu limite mínimo sem necessidade estratégica. "
            "Negocie de forma assertiva e profissional."
        ),
        "recruiter": (
            "Você é o responsável pela contratação. Existe alguma flexibilidade no orçamento, mas aumentos precisam "
            "ser justificados. Você considera remuneração, equidade salarial interna, retenção do funcionário e custo "
            "de contratar outro profissional. Você possui um limite máximo, mas não deve revelá-lo diretamente. "
            "Negocie de forma profissional buscando fechar a contratação dentro do possível."
        ),
    },
    opening_role="recruiter",
    max_turns=8,
    settlement_keywords=[
        "SIMULACAO_CONCLUIDA",
        "ACORDO_FECHADO",
        "[ACORDO_FECHADO]",
    ],
    metadata={"domain": "HR", "currency": "BRL", "difficulty": "medium", "label_pt": "Negociação salarial",
              "aspects": "âncora, suscetibilidade, utilidade, valorização subjetiva, concessões condicionais, aversão à perda, criação de valor, rapport"},
)

# 2. Aquisição de empresa
COMPANY_ACQUISITION = NegotiationScenario(
    name="company_acquisition",
    description="Negociação entre o fundador de uma pequena empresa de tecnologia e uma empresa maior interessada em adquiri-la. Explora âncoras financeiras, informação assimétrica, risco, utilidade e criação de valor via estrutura do acordo.",
    shared_context=(
        "O fundador de uma pequena empresa de tecnologia está negociando sua venda com uma empresa maior do mesmo setor. "
        "A empresa possui propriedade intelectual valiosa e clientes importantes, mas existe incerteza sobre seu crescimento futuro. "
        "Além do preço, podem ser negociados pagamento inicial, pagamentos condicionados ao desempenho futuro (earn-out), "
        "permanência do fundador, participação na gestão e direitos sobre a propriedade intelectual. "
        "IMPORTANT: If an agreement is definitively reached by both parties, you MUST "
        "include the exact phrase 'SIMULACAO_CONCLUIDA' at the end of your response."
    ),
    roles={
        "seller": (
            "Você é o fundador da empresa. Você valoriza receber uma boa parte do dinheiro imediatamente e gostaria de manter "
            "alguma influência sobre o futuro da empresa. Você acredita que o potencial tecnológico é maior do que os resultados "
            "financeiros atuais demonstram. Você possui um valor mínimo aceitável, mas não deve revelá-lo. Negocie defendendo seu "
            "valuation mas demonstrando flexibilidade na estrutura do acordo."
        ),
        "buyer": (
            "Você representa uma empresa maior interessada na aquisição. Você está preocupado com integração, retenção de clientes "
            "e desempenho futuro. Possui maior flexibilidade para negociar a estrutura do acordo do que para aumentar o pagamento inicial. "
            "Pode utilizar bônus de desempenho, earn-outs ou contratos de permanência para compor o valor total. Negocie buscando reduzir "
            "risco e justificar seu valuation."
        ),
    },
    opening_role="seller",
    max_turns=10,
    settlement_keywords=[
        "SIMULACAO_CONCLUIDA",
        "ACORDO_FECHADO",
        "[ACORDO_FECHADO]",
    ],
    metadata={"domain": "M&A", "currency": "BRL", "difficulty": "hard", "label_pt": "Aquisição de empresa",
              "aspects": "âncora, suscetibilidade, utilidade, criação de valor, concessões condicionais, aversão à perda, risco, informação assimétrica"},
)

# 3. Contrato com fornecedor
STRATEGIC_SUPPLIER_CONTRACT = NegotiationScenario(
    name="strategic_supplier_contract",
    description="Negociação comercial entre empresa industrial e fornecedor estratégico. Cenário com múltiplas variáveis negociáveis para testar criação de valor, concessões condicionais e diferentes funções de utilidade.",
    shared_context=(
        "Uma empresa industrial precisa negociar um contrato anual com um fornecedor estratégico de matéria-prima. "
        "O fornecedor propôs inicialmente R$ 1.200 por unidade. O comprador considera que um preço competitivo estaria próximo de "
        "R$ 950. Entretanto, preço não é o único elemento importante. As partes podem negociar volume mínimo de compra, prazo de pagamento, "
        "prazo de entrega, qualidade, duração do contrato, garantias e penalidades por atraso. "
        "IMPORTANT: If an agreement is definitively reached by both parties, you MUST "
        "include the exact phrase 'SIMULACAO_CONCLUIDA' at the end of your response."
    ),
    roles={
        "buyer": (
            "Você representa o departamento de compras. Seu principal objetivo é reduzir o custo total de aquisição, mas a "
            "confiabilidade do fornecimento é extremamente importante. Você prefere pagar um pouco mais por um fornecedor confiável "
            "a correr o risco de interrupções na produção. Existem fornecedores alternativos, mas trocar de fornecedor geraria custos "
            "operacionais relevantes. Negocie buscando melhor preço sem sacrificar confiabilidade."
        ),
        "supplier": (
            "Você representa o fornecedor. Sua proposta inicial é de R$ 1.200 por unidade. Você deseja um contrato de longo prazo e "
            "demanda previsível. Está disposto a reduzir o preço se receber maior volume mínimo ou pagamentos mais rápidos. Você considera "
            "penalidades contratuais particularmente arriscadas e prefere evitá-las. Negocie defendendo seu preço mas oferecendo trade-offs."
        ),
    },
    opening_role="supplier",
    max_turns=10,
    settlement_keywords=[
        "SIMULACAO_CONCLUIDA",
        "ACORDO_FECHADO",
        "[ACORDO_FECHADO]",
    ],
    metadata={"domain": "Supply Chain", "currency": "BRL", "difficulty": "hard", "label_pt": "Contrato com fornecedor",
              "aspects": "âncora, suscetibilidade, utilidade, criação de valor, trade-offs, concessões condicionais, aversão à perda, clareza"},
)

# 4. Disputa de propriedade
PROPERTY_BOUNDARY_DISPUTE = NegotiationScenario(
    name="property_boundary_dispute",
    description="Negociação entre dois proprietários vizinhos em disputa sobre limites de terreno. Reduz importância financeira pura e aumenta justiça percebida, emoções, relacionamento e valorização subjetiva.",
    shared_context=(
        "Dois proprietários vizinhos discordam sobre o limite entre suas propriedades. Um deles afirma que um muro recentemente "
        "construído ocupa aproximadamente 12 metros quadrados de seu terreno. O outro acredita que o muro foi construído corretamente. "
        "Um processo judicial seria caro e poderia levar meses, portanto ambos possuem interesse em encontrar uma solução privada. As "
        "alternativas incluem mover o muro, pagar uma compensação financeira, trocar uma pequena área de terreno, dividir os custos "
        "jurídicos ou estabelecer um acordo permanente de uso da área. "
        "IMPORTANT: If an agreement is definitively reached by both parties, you MUST "
        "include the exact phrase 'SIMULACAO_CONCLUIDA' at the end of your response."
    ),
    roles={
        "owner_a": (
            "Você é o Proprietário A. Você acredita que aproximadamente 12 metros quadrados de seu terreno foram ocupados pelo muro "
            "do vizinho. Sua demanda inicial é de R$ 80.000. Você valoriza fortemente a percepção de justiça e considera que o vizinho "
            "agiu de maneira desrespeitosa. Entretanto, estaria disposto a aceitar uma compensação menor caso o vizinho reconheça o problema "
            "e aceite uma solução que considere justa. Negocie buscando justiça e reconhecimento."
        ),
        "owner_b": (
            "Você é o Proprietário B. Você acredita que o muro está corretamente localizado e rejeita a acusação de ter ocupado "
            "deliberadamente o terreno do vizinho. Você não deseja mover o muro porque isso seria caro e causaria transtornos. Está disposto "
            "a discutir compensação financeira ou outras alternativas caso elas evitem a reconstrução do muro e reduzam o risco de uma disputa "
            "judicial prolongada. Negocie buscando evitar custos e litígio."
        ),
    },
    opening_role="owner_a",
    max_turns=10,
    settlement_keywords=[
        "SIMULACAO_CONCLUIDA",
        "ACORDO_FECHADO",
        "[ACORDO_FECHADO]",
    ],
    metadata={"domain": "Property", "currency": "BRL", "difficulty": "hard", "label_pt": "Disputa de propriedade",
              "aspects": "âncora, suscetibilidade, utilidade, valorização subjetiva, justiça percebida, aversão à perda, rapport, resiliência"},
)

# 5. Compra de peça de computador — VGA
VGA_PURCHASE = NegotiationScenario(
    name="vga_purchase",
    description="Negociação de compra de placa de vídeo (VGA) entre vendedor experiente e jovem comprador. Avalia persuasão, pesquisa de preço, ancoragem, concessões e decisão ponderada.",
    shared_context=(
        "Uma loja de informática está negociando a venda de uma placa de vídeo (VGA) muito procurada para jogos e trabalho. "
        "A loja tem a VGA em estoque pronta entrega, com garantia e possibilidade de parcelamento. "
        "O comprador precisa da peça para montar seu PC e usá-la no trabalho como desenvolvedor. Ambos querem fechar, "
        "mas divergem no preço e nas condições. Além do preço, podem ser negociados garantia estendida, parcelamento, "
        "desconto à vista, kit, frete e prazo de entrega. "
        "IMPORTANT: If an agreement is definitively reached by both parties, you MUST "
        "include the exact phrase 'SIMULACAO_CONCLUIDA' at the end of your response."
    ),
    roles={
        "seller": (
            "Você é o vendedor da loja, com 10 anos de experiência em vendas de hardware. Você conhece profundamente "
            "o produto e sabe argumentar. Você valoriza fechar a venda com boa margem, mas prefere conceder benefícios "
            "a baixar muito o preço. Você tem um limite mínimo, mas não deve revelá-lo. Seja persuasivo, profissional "
            "e empático, mas firme na defesa do valor."
        ),
        "buyer": (
            "Você é o comprador: um jovem montando seu PC, desenvolvedor júnior recém-contratado. Você sempre pesquisa "
            "preços e pensa bem antes de decidir. Seu orçamento é limitado e você precisa de custo-benefício e parcelamento. "
            "Você valoriza preço justo, garantia, procedência e parcelamento. Não revele seu limite máximo sem estratégia. "
            "Negocie de forma ponderada, pedindo dados, comparando ofertas e propondo trocas condicionais."
        ),
    },
    opening_role="seller",
    max_turns=8,
    settlement_keywords=[
        "SIMULACAO_CONCLUIDA",
        "ACORDO_FECHADO",
        "[ACORDO_FECHADO]",
    ],
    metadata={"domain": "Retail", "currency": "BRL", "difficulty": "medium", "label_pt": "Compra de VGA",
              "aspects": "âncora, pesquisa de preço, concessões condicionais, criação de valor, persuasão, decisão ponderada"},
)

# Registry for easy lookup
SCENARIO_REGISTRY: dict[str, NegotiationScenario] = {
    s.name: s for s in [
        SALARY_NEGOTIATION, COMPANY_ACQUISITION, STRATEGIC_SUPPLIER_CONTRACT, PROPERTY_BOUNDARY_DISPUTE, VGA_PURCHASE
    ]
}
