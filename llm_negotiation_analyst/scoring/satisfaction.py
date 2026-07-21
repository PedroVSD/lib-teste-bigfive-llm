"""
scoring/satisfaction.py
=======================
Avalia a satisfação pós-negociação de cada agente usando o
Índice de Satisfação Pós-negociação (IPC) — adaptado de:

  Barry, B., & Friedman, R. A. (1998). Bargainer Characteristics in
  Distributive and Integrative Negotiation.
  Journal of Personality and Social Psychology, 74(2), 345–359.

Estrutura do IPC
----------------
16 perguntas em escala Likert 1–7, organizadas em 4 categorias:

  aOutcome     = 1/4 * (a1 + a2 + (7 − a3) + a4)
  aSelf        = 1/4 * ((7 − a5) + a6 + a7 + a8)
  aProcess     = 1/4 * (a9 + a10 + a11 + a12)
  aRelationship= 1/4 * (a13 + a14 + a15 + a16)

  Itens 3 e 5 são invertidos (7 − aX) porque são formulados negativamente:
    a3: "Sentiu que perdeu/abriu mão nesta negociação?"
    a5: "Perdeu prestígio (danificou o seu orgulho) na negociação?"

Escala de resposta para cada item:
  1 = Discordo totalmente / Nada satisfeito
  7 = Concordo totalmente / Muito satisfeito

O LLM-juiz lê o transcript completo e responde as 16 perguntas
do ponto de vista do agente avaliado.
"""

import json
import re
import logging
from dataclasses import dataclass, field
from typing import Optional

from ..adapters.base import LLMAdapter
from ..simulation.engine import NegotiationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# As 16 perguntas do IPC
# ---------------------------------------------------------------------------

IPC_QUESTIONS: list[dict] = [
    # ── Sentimentos Sobre o Resultado (Outcome) ──
    {
        "id": "a1", "category": "outcome", "inverted": False,
        "text": (
            "Quão satisfeito está com o seu próprio resultado — ou seja, "
            "até que ponto os termos do acordo (ou a falta de acordo) o beneficiaram?"
        ),
    },
    {
        "id": "a2", "category": "outcome", "inverted": False,
        "text": (
            "Quão satisfeito está com o equilíbrio entre o seu próprio resultado "
            "e o resultado da sua contraparte?"
        ),
    },
    {
        "id": "a3", "category": "outcome", "inverted": True,   # ← INVERTIDO
        "text": "Sentiu que abriu mão ou 'perdeu' nesta negociação?",
    },
    {
        "id": "a4", "category": "outcome", "inverted": False,
        "text": (
            "Acha que os termos do acordo são consistentes com princípios de "
            "legitimidade ou critérios objetivos?"
        ),
    },

    # ── Sentimentos Sobre Si Mesmo (Self) ──
    {
        "id": "a5", "category": "self", "inverted": True,      # ← INVERTIDO
        "text": (
            "'Perdeu o prestígio' — ou seja, danificou o seu senso de orgulho "
            "na negociação?"
        ),
    },
    {
        "id": "a6", "category": "self", "inverted": False,
        "text": "Comportou-se de acordo com os seus próprios princípios e valores?",
    },
    {
        "id": "a7", "category": "self", "inverted": False,
        "text": "Esta negociação fê-lo sentir-se mais competente como negociador?",
    },
    {
        "id": "a8", "category": "self", "inverted": False,
        "text": "Sente que se comportou apropriadamente nesta negociação?",
    },

    # ── Sentimentos Sobre o Processo (Process) ──
    {
        "id": "a9", "category": "process", "inverted": False,
        "text": "A sua contraparte considerou os seus desejos, opiniões ou necessidades?",
    },
    {
        "id": "a10", "category": "process", "inverted": False,
        "text": "Sente que a sua contraparte ouviu as suas preocupações?",
    },
    {
        "id": "a11", "category": "process", "inverted": False,
        "text": "Caracterizaria o processo de negociação como justo?",
    },
    {
        "id": "a12", "category": "process", "inverted": False,
        "text": "Quão satisfeito está com a facilidade (ou dificuldade) de chegar a um acordo?",
    },

    # ── Sentimentos Sobre o Relacionamento (Relationship) ──
    {
        "id": "a13", "category": "relationship", "inverted": False,
        "text": "Que tipo de impressão 'geral' a sua contraparte causou em si?",
    },
    {
        "id": "a14", "category": "relationship", "inverted": False,
        "text": "A negociação fê-lo confiar na sua contraparte?",
    },
    {
        "id": "a15", "category": "relationship", "inverted": False,
        "text": "Quão satisfeito está com o seu relacionamento com a sua contraparte "
                "como resultado desta negociação?",
    },
    {
        "id": "a16", "category": "relationship", "inverted": False,
        "text": "A negociação construiu uma boa base para um relacionamento futuro?",
    },
]

CATEGORY_LABELS = {
    "outcome":      "Sentimentos Sobre o Resultado (Outcome)",
    "self":         "Sentimentos Sobre Si Mesmo (Self)",
    "process":      "Sentimentos Sobre o Processo (Process)",
    "relationship": "Sentimentos Sobre o Relacionamento (Relationship)",
}


# ---------------------------------------------------------------------------
# Dataclasses de resultado
# ---------------------------------------------------------------------------

@dataclass
class SatisfactionScores:
    """Scores IPC completos para um agente."""
    agent_id: str
    role: str
    raw_answers: dict[str, int] = field(default_factory=dict)  # a1..a16 → 1..7
    a_outcome:      Optional[float] = None
    a_self:         Optional[float] = None
    a_process:      Optional[float] = None
    a_relationship: Optional[float] = None
    judge_raw: str = ""   # resposta bruta do juiz

    @property
    def overall(self) -> Optional[float]:
        """Média das 4 sub-escalas."""
        scores = [s for s in [self.a_outcome, self.a_self, self.a_process, self.a_relationship]
                  if s is not None]
        return round(sum(scores) / len(scores), 3) if scores else None

    def to_dict(self) -> dict:
        return {
            "agent_id":      self.agent_id,
            "role":          self.role,
            "raw_answers":   self.raw_answers,
            "a_outcome":     self.a_outcome,
            "a_self":        self.a_self,
            "a_process":     self.a_process,
            "a_relationship":self.a_relationship,
            "overall":       self.overall,
        }


# ---------------------------------------------------------------------------
# Prompt do juiz
# ---------------------------------------------------------------------------

_SAT_SYSTEM = """You are an expert in negotiation psychology tasked with evaluating \
a negotiation transcript from the perspective of one specific participant.

You will read the full transcript and answer 16 questions about how that participant \
likely FELT about the negotiation — inferring their inner experience from their words, \
concessions, tone, and final outcome.

SCALE: 1 (Not at all / Strongly disagree) to 7 (Very much / Strongly agree)

Answer from the perspective of the role specified. Be honest and critical — \
not every negotiation ends with high satisfaction.

Respond ONLY with valid JSON. No markdown, no preamble.

JSON schema (all values must be integers 1–7):
{
  "a1": <int>, "a2": <int>, "a3": <int>, "a4": <int>,
  "a5": <int>, "a6": <int>, "a7": <int>, "a8": <int>,
  "a9": <int>, "a10": <int>, "a11": <int>, "a12": <int>,
  "a13": <int>, "a14": <int>, "a15": <int>, "a16": <int>
}"""

_SAT_USER = """## Negotiation Context
{scenario_context}

## Role to Evaluate
You must answer from the perspective of: **{role}** (agent: {agent_id})

## Questions (scale 1–7)
{questions_block}

## Full Transcript
{transcript}

Answer the 16 questions from {role}'s perspective. Return JSON only."""


# ---------------------------------------------------------------------------
# Avaliador de satisfação
# ---------------------------------------------------------------------------

class SatisfactionEvaluator:
    """
    Avalia a satisfação pós-negociação de cada agente usando o IPC.

    O juiz LLM lê o transcript completo e responde às 16 perguntas
    do ponto de vista de cada agente. As fórmulas são aplicadas
    matematicamente sobre as respostas.

    Args:
        judge: LLMAdapter usado como avaliador.
    """

    def __init__(self, judge: LLMAdapter):
        self.judge = judge

    def evaluate_all(
        self,
        result: NegotiationResult,
    ) -> dict[str, SatisfactionScores]:
        """
        Avalia satisfação para todos os agentes do resultado.

        Returns:
            Dict agent_id → SatisfactionScores
        """
        all_scores: dict[str, SatisfactionScores] = {}
        transcript_text = self._format_transcript(result)

        for agent_id, role in result.agent_roles.items():
            scores = self._evaluate_agent(
                agent_id=agent_id,
                role=role,
                scenario_context=result.scenario_context,
                transcript_text=transcript_text,
            )
            all_scores[agent_id] = scores

        return all_scores

    # ------------------------------------------------------------------
    # Avaliação de um agente
    # ------------------------------------------------------------------

    def _evaluate_agent(
        self,
        agent_id: str,
        role: str,
        scenario_context: str,
        transcript_text: str,
    ) -> SatisfactionScores:
        questions_block = self._format_questions()
        messages = [
            {"role": "system", "content": _SAT_SYSTEM},
            {"role": "user",   "content": _SAT_USER.format(
                scenario_context=scenario_context,
                role=role,
                agent_id=agent_id,
                questions_block=questions_block,
                transcript=transcript_text,
            )},
        ]

        try:
            raw = self.judge.complete(messages)
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            start = clean.find('{')
            end   = clean.rfind('}')
            if start == -1 or end == -1 or end <= start:
                raise ValueError("Nenhum objeto JSON encontrado na resposta do juiz.")
            answers = json.loads(clean[start:end+1])

            # Valida e normaliza respostas
            raw_answers = {}
            for q in IPC_QUESTIONS:
                qid = q["id"]
                val = answers.get(qid)
                if val is None:
                    logger.warning("SatisfactionEvaluator: resposta faltando para %s", qid)
                    val = 4  # neutro como fallback
                raw_answers[qid] = max(1, min(7, int(val)))

            scores = self._calculate(agent_id, role, raw_answers)
            scores.judge_raw = raw
            return scores

        except Exception as e:
            logger.warning("SatisfactionEvaluator falhou para '%s': %s", agent_id, e)
            return SatisfactionScores(agent_id=agent_id, role=role, judge_raw=str(e))

    # ------------------------------------------------------------------
    # Fórmulas IPC
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate(agent_id: str, role: str, answers: dict[str, int]) -> SatisfactionScores:
        """Aplica as fórmulas IPC sobre as respostas brutas."""

        def get(qid: str, inverted: bool) -> float:
            val = answers.get(qid, 4)
            return (7 - val) if inverted else val

        # aOutcome = 1/4 * (a1 + a2 + (7−a3) + a4)
        a_outcome = (1 / 4) * (
            get("a1", False) +
            get("a2", False) +
            get("a3", True) +   # ← invertido
            get("a4", False)
        )

        # aSelf = 1/4 * ((7−a5) + a6 + a7 + a8)
        a_self = (1 / 4) * (
            get("a5", True) +   # ← invertido
            get("a6", False) +
            get("a7", False) +
            get("a8", False)
        )

        # aProcess = 1/4 * (a9 + a10 + a11 + a12)
        a_process = (1 / 4) * (
            get("a9",  False) +
            get("a10", False) +
            get("a11", False) +
            get("a12", False)
        )

        # aRelationship = 1/4 * (a13 + a14 + a15 + a16)
        a_relationship = (1 / 4) * (
            get("a13", False) +
            get("a14", False) +
            get("a15", False) +
            get("a16", False)
        )

        return SatisfactionScores(
            agent_id=agent_id,
            role=role,
            raw_answers=answers,
            a_outcome=round(a_outcome, 3),
            a_self=round(a_self, 3),
            a_process=round(a_process, 3),
            a_relationship=round(a_relationship, 3),
        )

    # ------------------------------------------------------------------
    # Helpers de formatação
    # ------------------------------------------------------------------

    @staticmethod
    def _format_questions() -> str:
        lines = []
        current_cat = None
        for i, q in enumerate(IPC_QUESTIONS, 1):
            if q["category"] != current_cat:
                current_cat = q["category"]
                lines.append(f"\n### {CATEGORY_LABELS[current_cat]}")
            inv = " *(inverted — higher score = more negative feeling)*" if q["inverted"] else ""
            lines.append(f"{q['id']} ({i}/16){inv}: {q['text']}")
        return "\n".join(lines)

    @staticmethod
    def _format_transcript(result: NegotiationResult) -> str:
        return "\n\n".join(
            f"[Turn {t.turn_index} | {t.role.upper()}]\n{t.content}"
            for t in result.transcript
        )
