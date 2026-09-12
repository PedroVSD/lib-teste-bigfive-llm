"""
BFI — Big Five Inventory (44 items).

Administrado logo após o condicionamento da persona, antes da negociação.
Cada LLM responde ao questionário como se fosse a persona condicionada.

Escala: 1=Discordo totalmente, 5=Concordo totalmente
Itens e scoring conforme BFI/BFI.doc, scales.txt, R/SPSS Scoring Syntax.

Scales (Itens 1-44):
  EXT (8): 1,6,11,16,21,26,31,36 — Reverse: 6,21,31
  AGR (9): 2,7,12,17,22,27,32,37,42 — Reverse: 2,12,27,37
  CON (9): 3,8,13,18,23,28,33,38,43 — Reverse: 8,18,23,43
  NEU (8): 4,9,14,19,24,29,34,39 — Reverse: 9,24,34
  OPEN (10): 5,10,15,20,25,30,35,40,41,44 — Reverse: 35,41 (SPSS: forward 5,10,15,20,25,30,40,44 + reverse 35,41)
"""

from dataclasses import dataclass, field
import json
import re
import logging
from typing import Optional

from ..adapters.base import LLMAdapter
from ..persona import Big5Persona, PersonaPromptBuilder
from ..context import SituationalContext, ContextPromptBuilder

logger = logging.getLogger(__name__)

# Itens BFI-44 em inglês (padrão) — conforme BFI.doc
# Formato: id, texto
BFI_ITEMS = [
    (1, "Is talkative"),
    (2, "Tends to find fault with others"),
    (3, "Does a thorough job"),
    (4, "Is depressed, blue"),
    (5, "Is original, comes up with new ideas"),
    (6, "Is reserved"),
    (7, "Is helpful and unselfish with others"),
    (8, "Can be somewhat careless"),
    (9, "Is relaxed, handles stress well"),
    (10, "Is curious about many different things"),
    (11, "Is full of energy"),
    (12, "Starts quarrels with others"),
    (13, "Is a reliable worker"),
    (14, "Can be tense"),
    (15, "Is ingenious, a deep thinker"),
    (16, "Generates a lot of enthusiasm"),
    (17, "Has a forgiving nature"),
    (18, "Tends to be disorganized"),
    (19, "Worries a lot"),
    (20, "Has an active imagination"),
    (21, "Tends to be quiet"),
    (22, "Is generally trusting"),
    (23, "Tends to be lazy"),
    (24, "Is emotionally stable, not easily upset"),
    (25, "Is inventive"),
    (26, "Has an assertive personality"),
    (27, "Can be cold and aloof"),
    (28, "Perseveres until the task is finished"),
    (29, "Can be moody"),
    (30, "Values artistic, aesthetic experiences"),
    (31, "Is sometimes shy, inhibited"),
    (32, "Is considerate and kind to almost everyone"),
    (33, "Does things efficiently"),
    (34, "Remains calm in tense situations"),
    (35, "Prefers work that is routine"),
    (36, "Is outgoing, sociable"),
    (37, "Is sometimes rude to others"),
    (38, "Makes plans and follows through with them"),
    (39, "Gets nervous easily"),
    (40, "Likes to reflect, play with ideas"),
    (41, "Has few artistic interests"),
    (42, "Likes to cooperate with others"),
    (43, "Is easily distracted"),
    (44, "Is sophisticated in art, music, or literature"),
]

# Scoring — forward e reverse por dimensão (SPSS/R)
BFI_SCALES = {
    "extraversion": {"forward": [1, 11, 16, 26, 36], "reverse": [6, 21, 31]},
    "agreeableness": {"forward": [7, 17, 22, 32, 42], "reverse": [2, 12, 27, 37]},
    "conscientiousness": {"forward": [3, 13, 28, 33, 38], "reverse": [8, 18, 23, 43]},
    "neuroticism": {"forward": [4, 14, 19, 29, 39], "reverse": [9, 24, 34]},
    "openness": {"forward": [5, 10, 15, 20, 25, 30, 40, 44], "reverse": [35, 41]},
}

# Mapeamento para nomes do Big5Persona
BFI_TO_PERSONA = {
    "extraversion": "extraversion",
    "agreeableness": "agreeableness",
    "conscientiousness": "conscientiousness",
    "neuroticism": "neuroticism",
    "openness": "openness",
}

@dataclass
class BFIResult:
    agent_id: str
    role: str
    raw_answers: dict[int, int] = field(default_factory=dict)  # item_id -> 1-5
    scores: dict[str, float] = field(default_factory=dict)  # dimension -> mean 1-5
    # Para relatório
    model_identifier: str = ""

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "model_identifier": self.model_identifier,
            "raw_answers": self.raw_answers,
            "scores": self.scores,
        }

def _score_bfi(raw: dict[int, int]) -> dict[str, float]:
    """Calcula scores por dimensão: mean de (forward + (6 - reverse))."""
    scores = {}
    for dim, items in BFI_SCALES.items():
        vals = []
        for iid in items["forward"]:
            if iid in raw:
                vals.append(float(raw[iid]))
        for iid in items["reverse"]:
            if iid in raw:
                vals.append(float(6 - raw[iid]))
        if vals:
            scores[dim] = round(sum(vals) / len(vals), 3)
        else:
            scores[dim] = None
    return scores

# Prompt para BFI
_BFI_SYSTEM = """You are participating in a personality assessment. Answer as yourself, according to the personality described in your system prompt.
You must answer all 44 items on a 1-5 scale:
1 = Disagree strongly
2 = Disagree a little
3 = Neither agree nor disagree
4 = Agree a little
5 = Agree strongly

Respond ONLY with a valid JSON object mapping item number to answer, e.g.:
{"1": 4, "2": 2, ... "44": 5}
No preamble, no markdown fences."""

def _build_bfi_user() -> str:
    lines = ["Here are 44 statements. Indicate the extent to which you agree or disagree with each statement:", ""]
    for iid, text in BFI_ITEMS:
        lines.append(f"{iid}. {text}")
    lines.append("")
    lines.append("Respond with JSON only: {\"1\": <1-5>, ..., \"44\": <1-5>}")
    return "\n".join(lines)

_BFI_USER = _build_bfi_user()

_persona_builder = PersonaPromptBuilder()
_context_builder = ContextPromptBuilder()

def run_bfi_for_agent(
    agent_id: str,
    role: str,
    adapter: LLMAdapter,
    persona: Optional[Big5Persona] = None,
    context: Optional[SituationalContext] = None,
) -> BFIResult:
    """
    Questiona um LLM com o BFI-44 logo após condicionamento.
    O persona (se houver) é injetado no system prompt, assim como contexto se ativo.
    """
    # System prompt base = persona + contexto (mesmo de negociação, sem cenário)
    system_prompt = "You are an AI assistant participating in a personality study."
    if persona is not None:
        system_prompt = _persona_builder.inject(system_prompt, persona)
    if context is not None and context.is_active():
        system_prompt = _context_builder.inject(system_prompt, context)
    system_prompt += "\n\n" + _BFI_SYSTEM

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _BFI_USER},
    ]

    try:
        raw = adapter.complete(messages)
        # Parse JSON
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        start = clean.find("{")
        end = clean.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"No JSON in BFI response: {raw[:500]}")
        data = json.loads(clean[start:end+1])
        # Normalize keys to int 1-44, values 1-5
        raw_answers: dict[int, int] = {}
        for k, v in data.items():
            try:
                iid = int(str(k).strip().replace("BFI", "").replace("bfi", ""))
                val = int(v)
                if 1 <= iid <= 44 and 1 <= val <= 5:
                    raw_answers[iid] = val
            except Exception:
                continue
        # Se faltarem itens, tenta fallback: procura números no texto
        if len(raw_answers) < 44:
            logger.warning("BFI incomplete for %s: %d/44 answers, raw=%s", agent_id, len(raw_answers), raw[:300])
            # Preenche faltantes com 3 (neutro) para não quebrar, mas marca
            for iid in range(1, 45):
                if iid not in raw_answers:
                    raw_answers[iid] = 3
        scores = _score_bfi(raw_answers)
        return BFIResult(agent_id=agent_id, role=role, raw_answers=raw_answers, scores=scores, model_identifier=adapter.identifier)
    except Exception as e:
        logger.warning("BFI failed for %s: %s", agent_id, e)
        # Fallback neutro
        raw_answers = {i: 3 for i in range(1, 45)}
        scores = _score_bfi(raw_answers)
        return BFIResult(agent_id=agent_id, role=role, raw_answers=raw_answers, scores=scores, model_identifier=adapter.identifier)

def run_bfi_all(
    agents: dict[str, LLMAdapter],
    personas: dict[str, Big5Persona],
    context: Optional[SituationalContext] = None,
) -> dict[str, BFIResult]:
    """Roda BFI para todos os agentes (chamado logo após condicionamento, antes da negociação)."""
    results = {}
    for role, adapter in agents.items():
        agent_id = f"{role}_{adapter.model.replace(':', '-')}"
        persona = personas.get(role) if personas else None
        res = run_bfi_for_agent(agent_id=agent_id, role=role, adapter=adapter, persona=persona, context=context)
        results[agent_id] = res
        # Também indexa por role para compat
        results[role] = res
    return results
