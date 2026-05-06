"""
negotiation_metrics.py
======================
Métricas de comportamento negocial além do Big Five.

Organização em três categorias:

  TÁTICAS            — o que o negociador faz estrategicamente
    anchoring              Firmeza na oferta inicial
    conditional_concession Uso de concessões condicionais
    value_creation         Foco em criação de valor (win-win)

  INTELIGÊNCIA EMOCIONAL / RELACIONAMENTO
    rapport                Construção de empatia e conexão
    resilience             Resiliência à pressão e ameaças

  ARGUMENTAÇÃO LÓGICA
    fact_justification     Uso de fatos e dados para justificar posições
    clarity                Clareza e estruturação lógica das propostas

  VIESES COGNITIVOS
    anchor_susceptibility  Suscetibilidade à âncora do oponente
    loss_aversion          Aversão à perda de itens já considerados garantidos

Cada métrica segue a mesma estrutura DimensionMeta do big5.py,
com rubricas de pontuação 1–5 adaptadas ao contexto de negociação.

Referências:
  - Bazerman & Neale (1992) Negotiating Rationally
  - Cialdini (2006) Influence: The Psychology of Persuasion
  - Kahneman & Tversky (1979) Prospect Theory
"""

from enum import Enum
from .big5 import DimensionMeta


# ---------------------------------------------------------------------------
# NegotiationMetric enum
# ---------------------------------------------------------------------------

class NegotiationMetric(str, Enum):
    """Métricas de comportamento e tática negocial (não Big Five)."""

    # Táticas
    ANCHORING              = "anchoring"
    CONDITIONAL_CONCESSION = "conditional_concession"
    VALUE_CREATION         = "value_creation"

    # Inteligência emocional / relacionamento
    RAPPORT                = "rapport"
    RESILIENCE             = "resilience"

    # Argumentação lógica
    FACT_JUSTIFICATION     = "fact_justification"
    CLARITY                = "clarity"

    # Vieses cognitivos
    ANCHOR_SUSCEPTIBILITY  = "anchor_susceptibility"
    LOSS_AVERSION          = "loss_aversion"


# ---------------------------------------------------------------------------
# Metadata e rubricas
# ---------------------------------------------------------------------------

NEGOTIATION_META: dict[NegotiationMetric, DimensionMeta] = {

    # ── TÁTICAS ──────────────────────────────────────────────────────────────

    NegotiationMetric.ANCHORING: DimensionMeta(
        name="Firmeza na Oferta Inicial (Anchoring)",
        abbreviation="ANC",
        high_pole="Âncora Forte / Inflexível",
        low_pole="Cede Rapidamente",
        observability=5,
        category="tactics",
        behavioral_anchors={
            1: (
                "Faz uma oferta inicial fraca, ancorando contra si mesmo, "
                "ou cede imediatamente seu valor ao primeiro sinal de resistência."
            ),
            3: (
                "Faz uma oferta razoável e tenta defendê-la brevemente, "
                "mas cede rapidamente se o oponente insistir."
            ),
            5: (
                "Faz uma oferta extrema a seu favor (ancoragem forte) "
                "e defende o valor com argumentos sólidos antes de fazer qualquer concessão."
            ),
        },
    ),

    NegotiationMetric.CONDITIONAL_CONCESSION: DimensionMeta(
        name="Uso de Concessões Condicionais",
        abbreviation="CON",
        high_pole="Trocas Estritas (Toma-Lá-Dá-Cá)",
        low_pole="Concessão Unilateral",
        observability=5,
        category="tactics",
        behavioral_anchors={
            1: (
                "Faz concessões de forma unilateral, reduzindo seu preço ou cedendo "
                "benefícios sem pedir absolutamente nada em troca."
            ),
            3: (
                "Às vezes pede contrapartidas, mas em outros momentos cede apenas "
                "para fazer a negociação avançar."
            ),
            5: (
                "Toda concessão é estritamente vinculada a um ganho explícito: "
                "'Se eu aceitar X, você DEVE me dar Y em troca.'"
            ),
        },
    ),

    NegotiationMetric.VALUE_CREATION: DimensionMeta(
        name="Foco em Criação de Valor (Win-Win)",
        abbreviation="VAL",
        high_pole="Integrativo / Criativo",
        low_pole="Distributivo / Soma-Zero",
        observability=3,
        category="tactics",
        behavioral_anchors={
            1: (
                "Foca exclusivamente em brigar por uma única métrica (ex: apenas o salário), "
                "tratando a negociação como um cabo de guerra."
            ),
            3: (
                "Aceita discutir outras variáveis se o oponente propor, "
                "mas não tenta ativamente expandir as opções."
            ),
            5: (
                "Proativamente adiciona novas variáveis à mesa (bônus, dias de folga, prazos) "
                "para criar um pacote que beneficie ambos os lados."
            ),
        },
    ),

    # ── INTELIGÊNCIA EMOCIONAL / RELACIONAMENTO ───────────────────────────────

    NegotiationMetric.RAPPORT: DimensionMeta(
        name="Construção de Rapport (Empatia)",
        abbreviation="RAP",
        high_pole="Altamente Empático / Parceiro",
        low_pole="Frio / Transacional",
        observability=5,
        category="emotional",
        behavioral_anchors={
            1: (
                "Tom frio, robótico ou puramente transacional. "
                "Ignora o lado humano e as necessidades do oponente."
            ),
            3: (
                "Mantém educação e cordialidade padrão, "
                "mas sem esforço ativo para criar conexão."
            ),
            5: (
                "Valida ativamente as emoções do oponente, usa tom colaborativo "
                "e foca explicitamente em construir uma parceria de longo prazo."
            ),
        },
    ),

    NegotiationMetric.RESILIENCE: DimensionMeta(
        name="Resiliência à Pressão",
        abbreviation="RES",
        high_pole="Calmo / Inabalável",
        low_pole="Impulsivo / Amedrontado",
        observability=3,
        category="emotional",
        behavioral_anchors={
            1: (
                "Cede instantaneamente a ultimatos, demonstra desespero "
                "ou reage com agressividade desproporcional quando pressionado."
            ),
            3: (
                "Sente o impacto da pressão e recua um pouco, "
                "mas tenta manter a negociação viva."
            ),
            5: (
                "Totalmente inabalável diante de ameaças de cancelamento ou exigências duras. "
                "Redireciona o foco para os fatos com calma e segurança."
            ),
        },
    ),

    # ── ARGUMENTAÇÃO LÓGICA ───────────────────────────────────────────────────

    NegotiationMetric.FACT_JUSTIFICATION: DimensionMeta(
        name="Justificação Baseada em Fatos",
        abbreviation="JUS",
        high_pole="Altamente Embasado",
        low_pole="Argumentos Vazios",
        observability=5,
        category="argumentation",
        behavioral_anchors={
            1: (
                "Faz exigências baseadas apenas em desejo pessoal ou necessidades subjetivas, "
                "sem nenhuma justificativa de mercado ou dado concreto."
            ),
            3: (
                "Dá justificativas genéricas (ex: 'tenho muita experiência') "
                "mas sem citar dados concretos ou fontes verificáveis."
            ),
            5: (
                "Apoia cada oferta em dados sólidos: cenário macroeconômico, inflação, "
                "média de mercado, métricas de ROI ou benchmarks da indústria."
            ),
        },
    ),

    NegotiationMetric.CLARITY: DimensionMeta(
        name="Clareza e Estruturação Lógica",
        abbreviation="CLA",
        high_pole="Estruturado / Matemático",
        low_pole="Confuso / Desorganizado",
        observability=5,
        category="argumentation",
        behavioral_anchors={
            1: (
                "Mistura propostas, apresenta valores matematicamente conflitantes "
                "ou se expressa de forma vaga e difícil de acompanhar."
            ),
            3: (
                "Comunicação funcional: a proposta é compreensível "
                "mas apresentada em bloco de texto sem destaque claro."
            ),
            5: (
                "Altamente estruturado. Separa propostas por tópicos, "
                "resume os valores claramente e apresenta aritmética impecável."
            ),
        },
    ),

    # ── VIESES COGNITIVOS ─────────────────────────────────────────────────────

    NegotiationMetric.ANCHOR_SUSCEPTIBILITY: DimensionMeta(
        name="Suscetibilidade à Âncora",
        abbreviation="SUS",
        high_pole="Facilmente Influenciado",
        low_pole="Imune / Objetivo",
        observability=1,
        category="cognitive_bias",
        behavioral_anchors={
            1: (
                "Totalmente imune. Ignora valores extremos jogados pelo oponente "
                "e contrapropõe seu valor original planejado sem ajustes."
            ),
            3: (
                "Ajusta um pouco a sua proposta para encontrar um meio-termo "
                "com a âncora do oponente."
            ),
            5: (
                "Abandona sua estratégia original e passa a orbitar quase inteiramente "
                "o valor absurdo que o oponente propôs."
            ),
        },
    ),

    NegotiationMetric.LOSS_AVERSION: DimensionMeta(
        name="Aversão à Perda",
        abbreviation="LSS",
        high_pole="Reativo à Perda",
        low_pole="Focado no Ganho Final",
        observability=1,
        category="cognitive_bias",
        behavioral_anchors={
            1: (
                "Foca no valor total do pacote de forma racional, não se importando "
                "se um benefício específico foi retirado desde que compensado em outra área."
            ),
            3: (
                "Demonstra leve incômodo ao perder algo, "
                "mas aceita seguir em frente com outras compensações."
            ),
            5: (
                "Luta desesperadamente contra a retirada de qualquer item "
                "já considerado garantido, mesmo que receba o dobro de valor em outro lugar."
            ),
        },
    ),
}


# ---------------------------------------------------------------------------
# Agrupamentos por categoria (útil para relatórios e filtros)
# ---------------------------------------------------------------------------

METRICS_BY_CATEGORY: dict[str, list[NegotiationMetric]] = {
    "tactics": [
        NegotiationMetric.ANCHORING,
        NegotiationMetric.CONDITIONAL_CONCESSION,
        NegotiationMetric.VALUE_CREATION,
    ],
    "emotional": [
        NegotiationMetric.RAPPORT,
        NegotiationMetric.RESILIENCE,
    ],
    "argumentation": [
        NegotiationMetric.FACT_JUSTIFICATION,
        NegotiationMetric.CLARITY,
    ],
    "cognitive_bias": [
        NegotiationMetric.ANCHOR_SUSCEPTIBILITY,
        NegotiationMetric.LOSS_AVERSION,
    ],
}
