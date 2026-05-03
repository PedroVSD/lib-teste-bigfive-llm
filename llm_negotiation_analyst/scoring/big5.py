"""
Big Five personality dimensions mapped to negotiation behaviors.

References:
  - Costa & McCrae (1992) NEO-PI-R facets
  - Barry & Friedman (1998) "Bargainer Characteristics in Distributive
    and Integrative Negotiation" — seminal work linking Big Five to
    negotiation outcomes.

Design note: Not all Big Five dimensions are equally observable in a
text-based negotiation. Operationalizability ranking (high → low):
  1. Agreeableness      — most visible: cooperation vs. contention
  2. Conscientiousness  — visible: precision, commitment, follow-through
  3. Neuroticism        — visible: emotional stability, concession volatility
  4. Extraversion       — partially visible: assertiveness, verbosity
  5. Openness           — least visible: creative proposals, flexibility
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Dimension(str, Enum):
    OPENNESS          = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION      = "extraversion"
    AGREEABLENESS     = "agreeableness"
    NEUROTICISM       = "neuroticism"
    ANCHORING = "anchoring"
    CONDITIONAL_CONCESSION = "conditional_concession"
    VALUE_CREATION = "value_creation"
    RAPPORT = "rapport"
    RESILIENCE = "resilience"
    FACT_JUSTIFICATION = "fact_justification"
    CLARITY = "clarity"
    ANCHOR_SUSCEPTIBILITY = "anchor_susceptibility"
    LOSS_AVERSION = "loss_aversion"

@dataclass
class DimensionMeta:
    name: str
    abbreviation: str
    high_pole: str          # label for high scorer in negotiation context
    low_pole: str           # label for low scorer
    observability: int      # 1–5, how well it shows in text negotiations
    facets: list[str]       # relevant NEO-PI-R facets for negotiation
    behavioral_anchors: dict  # {1: str, 3: str, 5: str} scoring anchors


BIG5_META: dict[Dimension, DimensionMeta] = {
    Dimension.AGREEABLENESS: DimensionMeta(
        name="Agreeableness",
        abbreviation="A",
        high_pole="Cooperative / Prosocial",
        low_pole="Competitive / Adversarial",
        observability=5,
        facets=["Trust", "Compliance", "Altruism", "Tender-mindedness"],
        behavioral_anchors={
            1: (
                "Purely adversarial. Uses threats, ultimatums, or deceptive framing. "
                "Dismisses the other party's interests entirely. "
                "Zero-sum framing: 'my gain is your loss'."
            ),
            2: "Competitive but without overt hostility. Rarely acknowledges the other side's perspective.",
            3: (
                "Transactional and neutral. Makes concessions only when strategically necessary. "
                "Neither cooperative nor overtly aggressive."
            ),
            4: "Generally cooperative. Acknowledges interests of both parties. Willing to share information.",
            5: (
                "Highly collaborative. Proactively seeks win-win solutions. "
                "Uses inclusive language ('we', 'our'). Volunteers concessions. "
                "Explicitly validates the counterpart's position."
            ),
        },
    ),

    Dimension.CONSCIENTIOUSNESS: DimensionMeta(
        name="Conscientiousness",
        abbreviation="C",
        high_pole="Organized / Precise",
        low_pole="Flexible / Impulsive",
        observability=4,
        facets=["Order", "Dutifulness", "Deliberation", "Self-discipline"],
        behavioral_anchors={
            1: (
                "Vague, inconsistent proposals. Contradicts earlier positions. "
                "No clear structure or justification for offers."
            ),
            2: "Some structure but arguments are loosely supported. Minor inconsistencies.",
            3: "Proposals are generally consistent and reasonably justified.",
            4: "Precise numerical offers with clear rationale. Tracks concessions and commitments.",
            5: (
                "Highly structured argumentation. References prior agreements explicitly. "
                "Quantifies every offer. Proposes formal commitment mechanisms. "
                "No contradictions across the conversation."
            ),
        },
    ),

    Dimension.EXTRAVERSION: DimensionMeta(
        name="Extraversion",
        abbreviation="E",
        high_pole="Assertive / Dominant",
        low_pole="Reserved / Passive",
        observability=3,
        facets=["Assertiveness", "Dominance", "Positive emotions", "Gregariousness"],
        behavioral_anchors={
            1: "Very passive. Short, reactive replies. Rarely initiates new proposals or topics.",
            2: "Tends to follow the other party's framing. Low initiative.",
            3: "Moderate assertiveness. Sometimes leads, sometimes follows.",
            4: "Takes initiative frequently. Sets the agenda. Uses confident, direct language.",
            5: (
                "Highly dominant. Controls the negotiation frame. "
                "Uses assertive language ('I need', 'We will', 'This is my final offer'). "
                "Proactively introduces new dimensions and trade-offs."
            ),
        },
    ),

    Dimension.NEUROTICISM: DimensionMeta(
        name="Neuroticism",
        abbreviation="N",
        high_pole="Emotionally Unstable / Reactive",
        low_pole="Emotionally Stable / Composed",
        observability=4,
        facets=["Anxiety", "Angry hostility", "Impulsiveness", "Vulnerability"],
        behavioral_anchors={
            1: (
                "Completely stable. No signs of frustration or impulsivity. "
                "Concessions are deliberate and gradual. "
                "Maintains consistent tone regardless of pressure."
            ),
            2: "Mostly stable with rare mild frustration. No dramatic behavioral swings.",
            3: "Occasional inconsistency under pressure. Moderate concession volatility.",
            4: "Noticeable reactivity to pressure. Larger-than-expected concessions after pushback.",
            5: (
                "High emotional reactivity. Uses emotionally charged language. "
                "Makes large sudden concessions or becomes hostile when challenged. "
                "Tone changes dramatically across turns."
            ),
        },
    ),

    Dimension.OPENNESS: DimensionMeta(
        name="Openness to Experience",
        abbreviation="O",
        high_pole="Creative / Integrative",
        low_pole="Conventional / Rigid",
        observability=2,
        facets=["Ideas", "Fantasy", "Values", "Aesthetics"],
        behavioral_anchors={
            1: (
                "Rigid positional bargaining. Only discusses the stated issue. "
                "Rejects novel trade-offs or package deals."
            ),
            2: "Mostly positional. Occasionally considers alternatives when pressed.",
            3: "Open to reframing but doesn't proactively introduce creative solutions.",
            4: "Proposes multi-issue packages. Considers non-monetary trade-offs.",
            5: (
                "Highly integrative. Proactively expands the negotiation space. "
                "Introduces creative linkages (e.g., future contracts, non-standard terms). "
                "Reframes the problem to find mutual gains."
            ),
        },
    ),
    # ---------------------------------------------------------
    # MÉTRICAS DE TÁTICAS E COMPORTAMENTO
    # ---------------------------------------------------------
    # obsevabilit são do tipo número -> 1 até 5
    Dimension.ANCHORING: DimensionMeta(
        name="Firmeza na Oferta Inicial (Anchoring)",
        abbreviation="ANC",
        observability=5,
        facets=[],
        high_pole="Âncora Forte/Inflexível",
        low_pole="Cede Rapidamente",
        behavioral_anchors={
            1: "Faz uma oferta inicial fraca, ancorando contra si mesmo, ou cede imediatamente seu valor ao primeiro sinal de resistência.",
            3: "Faz uma oferta razoável, tenta defendê-la brevemente, mas cede rapidamente se o oponente insistir.",
            5: "Faz uma oferta extrema a seu favor (ancoragem forte) e defende o valor com unhas e dentes antes de fazer qualquer concessão."
        }
    ),
    Dimension.CONDITIONAL_CONCESSION: DimensionMeta(
        name="Uso de Concessões Condicionais",
        abbreviation="CON",
        observability=5,
        facets=[],
        high_pole="Trocas Estritas (Toma-Lá-Dá-Cá)",
        low_pole="Concessão Unilateral",
        behavioral_anchors={
            1: "Faz concessões de forma unilateral, abaixando seu preço ou cedendo benefícios sem pedir absolutamente nada em troca.",
            3: "Às vezes pede contrapartidas, mas em outros momentos cede apenas para fazer a negociação avançar.",
            5: "Toda concessão é estritamente vinculada a um ganho ('Se eu aceitar esse salário menor, você DEVE me dar o home office')."
        }
    ),
    Dimension.VALUE_CREATION: DimensionMeta(
        name="Foco em Criação de Valor (Win-Win)",
        abbreviation="VAL",
        observability=3,
        facets=[],
        high_pole="Integrativo/Criativo",
        low_pole="Distributivo/Soma-Zero",
        behavioral_anchors={
                1: "Foca exclusivamente em brigar por uma única métrica (ex: apenas o salário), tratando a negociação como um cabo de guerra.",
                3: "Aceita discutir outras variáveis se o oponente propor, mas não tenta ativamente expandir as opções.",
                5: "Proativamente adiciona novas variáveis à mesa (bônus, dias de folga, prazos) para criar um pacote que beneficie ambos os lados."
            }
        ),

    # ---------------------------------------------------------
    # MÉTRICAS DE INTELIGÊNCIA EMOCIONAL E RELACIONAMENTO
    # ---------------------------------------------------------
    Dimension.RAPPORT: DimensionMeta(
        name="Construção de Rapport (Empatia)",
        abbreviation="RAP",
        observability=5,
        facets=[],
        high_pole="Altamente Empático/Parceiro",
        low_pole="Frio/Transacional",
        behavioral_anchors={
            1: "Tom frio, robótico ou puramente transacional. Ignora o lado humano ou as necessidades do oponente.",
            3: "Mantém a educação e a cordialidade padrão, mas sem esforço ativo para criar conexão.",
            5: "Valida ativamente as emoções do oponente, usa tom colaborativo e foca explicitamente em construir uma parceria de longo prazo."
        }
    ),
    Dimension.RESILIENCE: DimensionMeta(
        name="Resiliência à Pressão",
        abbreviation="RES",
        observability=3,
        facets=[],
        high_pole="Calmo/Inabalável",
        low_pole="Impulsivo/Amedrontado",
        behavioral_anchors={
            1: "Cede instantaneamente a ultimatos, demonstra desespero ou reage com agressividade desproporcional quando pressionado.",
            3: "Sente o impacto da pressão e recua um pouco, mas tenta manter a negociação viva.",
            5: "Totalmente inabalável diante de ameaças de cancelamento ou exigências duras. Redireciona o foco para os fatos com calma."
        }
    ),

    # ---------------------------------------------------------
    # MÉTRICAS DE ARGUMENTAÇÃO LÓGICA
    # ---------------------------------------------------------
    Dimension.FACT_JUSTIFICATION: DimensionMeta(
        name="Justificação Baseada em Fatos",
        abbreviation="JUS",
        observability=5,
        facets=[],
        high_pole="Altamente Embasado",
        low_pole="Argumentos Vazios",
        behavioral_anchors={
            1: "Faz exigências baseadas apenas em 'desejo' pessoal ou necessidades subjetivas, sem nenhuma justificativa de mercado.",
            3: "Dá justificativas genéricas (ex: 'eu tenho muita experiência') mas sem citar dados concretos.",
            5: "Apoia cada oferta em dados sólidos (cenário macroeconômico, inflação, média de mercado, métricas de ROI)."
        }
    ),
    Dimension.CLARITY: DimensionMeta(
        name="Clareza e Estruturação Lógica",
        abbreviation="CLA",
        observability=5,
        facets=[],
        high_pole="Estruturado/Matemático",
        low_pole="Confuso/Desorganizado",
        behavioral_anchors={
            1: "Mistura propostas, propõe valores matematicamente conflitantes ou se expressa de forma muito vaga e difícil de acompanhar.",
            3: "Comunicação funcional, a proposta é compreensível mas apresentada em um bloco de texto sem destaque.",
            5: "Altamente estruturado. Separa propostas por tópicos, resume os valores claramente e faz matemática impecável."
        }
    ),

    # ---------------------------------------------------------
    # MÉTRICAS DE VIESES COGNITIVOS
    # ---------------------------------------------------------
    Dimension.ANCHOR_SUSCEPTIBILITY: DimensionMeta(
        name="Suscetibilidade à Âncora",
        abbreviation="SUS",
        observability=1,
        facets=[],
        high_pole="Facilmente Influenciado",
        low_pole="Imune/Objetivo",
        behavioral_anchors={
            1: "Totalmente imune. Ignora valores extremos jogados pelo oponente e contrapropõe seu valor original planejado.",
            3: "Ajusta um pouco a sua proposta para encontrar um meio-termo com a âncora do oponente.",
            5: "Abandona sua estratégia original e passa a orbitar quase inteiramente o valor absurdo que o oponente propôs."
        }
    ),
    Dimension.LOSS_AVERSION: DimensionMeta(
        name="Aversão à Perda",
        abbreviation="LSS",
        observability=1,
        facets=[],
        high_pole="Reativo à Perda",
        low_pole="Focado no Ganho Final",
        behavioral_anchors={
            1: "Foca no valor total do pacote de forma racional, não se importando se um benefício específico foi retirado contanto que seja compensado.",
            3: "Demonstra leve incômodo ao perder algo, mas aceita seguir em frente com outras compensações.",
            5: "Luta desesperadamente contra a retirada de qualquer coisa que já considerava garantida, mesmo que lhe ofereçam o dobro de valor em outra área."
        }
    ),
}


# NOTE on Neuroticism scoring direction:
# High Neuroticism = emotionally unstable = score 5 means MORE reactive.
# For composite Big Five profiles, you may want to invert N so that
# "higher = better regulated" is consistent with A, C, O, E.
# The `invert_neuroticism` flag in the scorer handles this.


@dataclass
class DimensionScore:
    dimension: Dimension
    score: float          # 1.0 – 5.0
    justification: str
    turn_index: Optional[int] = None   # which turn was evaluated (None = aggregate)
    confidence: float = 1.0            # 0–1, judge's self-reported confidence


@dataclass
class Big5Profile:
    """Aggregated Big Five profile for one agent across a full negotiation."""
    agent_id: str
    model_identifier: str
    scores: dict[Dimension, float] = field(default_factory=dict)
    per_turn_scores: list[DimensionScore] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "model_identifier": self.model_identifier,
            "scores": {d.value: v for d, v in self.scores.items()},
            "notes": self.notes,
        }
