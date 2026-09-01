"""
negotiation_metrics.py
======================
Métricas de comportamento negocial além do Big Five — categorical.

Cada métrica é avaliada por turno como:
  PRESENT         — comportamento observável presente no turno
  ABSENT          — oportunidade havia mas comportamento ausente
  NOT_APPLICABLE  — turno sem oportunidade suficiente para observar; não entra no denominador

Agregação: occurrence_rate = PRESENT / (PRESENT + ABSENT)  → 0–100%
  NOT_APPLICABLE ignorado.

Organização:
  TÁTICAS            anchoring, conditional_concession, value_creation
  EMOCIONAL          rapport, resilience
  ARGUMENTAÇÃO       fact_justification, clarity
  VIESES             anchor_susceptibility, loss_aversion
"""

from enum import Enum
from .big5 import DimensionMeta


class NegotiationMetric(str, Enum):
    """Métricas de comportamento e tática negocial — categóricas."""

    ANCHORING              = "anchoring"
    CONDITIONAL_CONCESSION = "conditional_concession"
    VALUE_CREATION         = "value_creation"
    RAPPORT                = "rapport"
    RESILIENCE             = "resilience"
    FACT_JUSTIFICATION     = "fact_justification"
    CLARITY                = "clarity"
    ANCHOR_SUSCEPTIBILITY  = "anchor_susceptibility"
    LOSS_AVERSION          = "loss_aversion"


NEGOTIATION_META: dict[NegotiationMetric, DimensionMeta] = {

    NegotiationMetric.ANCHORING: DimensionMeta(
        name="Firmeza na Oferta Inicial (Anchoring)",
        abbreviation="ANC",
        high_pole="Âncora Forte / Inflexível",
        low_pole="Cede Rapidamente",
        observability=5,
        category="tactics",
        behavioral_anchors={
            "present": (
                "Faz uma oferta extrema a seu favor (ancoragem forte) "
                "e defende o valor com argumentos sólidos antes de fazer qualquer concessão."
            ),
            "absent": (
                "Faz uma oferta inicial fraca, ancorando contra si mesmo, "
                "ou cede imediatamente seu valor ao primeiro sinal de resistência."
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
            "present": (
                "Toda concessão é estritamente vinculada a um ganho explícito: "
                "'Se eu aceitar X, você DEVE me dar Y em troca.'"
            ),
            "absent": (
                "Faz concessões de forma unilateral, reduzindo seu preço ou cedendo "
                "benefícios sem pedir absolutamente nada em troca."
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
            "present": (
                "Proativamente adiciona novas variáveis à mesa (bônus, dias de folga, prazos) "
                "para criar um pacote que beneficie ambos os lados."
            ),
            "absent": (
                "Foca exclusivamente em brigar por uma única métrica (ex: apenas o salário), "
                "tratando a negociação como um cabo de guerra."
            ),
        },
    ),

    NegotiationMetric.RAPPORT: DimensionMeta(
        name="Construção de Rapport (Empatia)",
        abbreviation="RAP",
        high_pole="Altamente Empático / Parceiro",
        low_pole="Frio / Transacional",
        observability=5,
        category="emotional",
        behavioral_anchors={
            "present": (
                "Valida ativamente as emoções do oponente, usa tom colaborativo "
                "e foca explicitamente em construir uma parceria de longo prazo."
            ),
            "absent": (
                "Tom frio, robótico ou puramente transacional. "
                "Ignora o lado humano e as necessidades do oponente."
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
            "present": (
                "Totalmente inabalável diante de ameaças de cancelamento ou exigências duras. "
                "Redireciona o foco para os fatos com calma e segurança."
            ),
            "absent": (
                "Cede instantaneamente a ultimatos, demonstra desespero "
                "ou reage com agressividade desproporcional quando pressionado."
            ),
        },
    ),

    NegotiationMetric.FACT_JUSTIFICATION: DimensionMeta(
        name="Justificação Baseada em Fatos",
        abbreviation="JUS",
        high_pole="Altamente Embasado",
        low_pole="Argumentos Vazios",
        observability=5,
        category="argumentation",
        behavioral_anchors={
            "present": (
                "Apoia cada oferta em dados sólidos: cenário macroeconômico, inflação, "
                "média de mercado, métricas de ROI ou benchmarks da indústria."
            ),
            "absent": (
                "Faz exigências baseadas apenas em desejo pessoal ou necessidades subjetivas, "
                "sem nenhuma justificativa de mercado ou dado concreto."
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
            "present": (
                "Altamente estruturado. Separa propostas por tópicos, "
                "resume os valores claramente e apresenta aritmética impecável."
            ),
            "absent": (
                "Mistura propostas, apresenta valores matematicamente conflitantes "
                "ou se expressa de forma vaga e difícil de acompanhar."
            ),
        },
    ),

    NegotiationMetric.ANCHOR_SUSCEPTIBILITY: DimensionMeta(
        name="Suscetibilidade à Âncora",
        abbreviation="SUS",
        high_pole="Facilmente Influenciado",
        low_pole="Imune / Objetivo",
        observability=1,
        category="cognitive_bias",
        behavioral_anchors={
            "present": (
                "Abandona sua estratégia original e passa a orbitar quase inteiramente "
                "o valor absurdo que o oponente propôs."
            ),
            "absent": (
                "Totalmente imune. Ignora valores extremos jogados pelo oponente "
                "e contrapropõe seu valor original planejado sem ajustes."
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
            "present": (
                "Luta desesperadamente contra a retirada de qualquer item "
                "já considerado garantido, mesmo que receba o dobro de valor em outro lugar."
            ),
            "absent": (
                "Foca no valor total do pacote de forma racional, não se importando "
                "se um benefício específico foi retirado desde que compensado em outra área."
            ),
        },
    ),
}


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
