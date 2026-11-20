"""
BFI — Big Five Inventory (44 items).

Administered right after persona conditioning, before the negotiation.
Each LLM answers the questionnaire as if it were the conditioned persona.

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
import re
import logging
import time
from typing import Optional

from ..adapters.base import LLMAdapter
from ..guards import load_json_value
from ..persona import Big5Persona, PersonaPromptBuilder
from ..context import SituationalContext, ContextPromptBuilder

logger = logging.getLogger(__name__)

# Standard English BFI-44 items — per BFI.doc
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

# Scoring — forward and reverse per dimension (SPSS/R)
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
    # Items that received no valid answer after all attempts
    missing_items: list[int] = field(default_factory=list)
    missing_policy: str = "neutral"
    attempts: int = 1
    # True when the per-agent time budget ran out before the BFI was complete
    timed_out: bool = False
    # For the report
    model_identifier: str = ""

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "model_identifier": self.model_identifier,
            "raw_answers": self.raw_answers,
            "scores": self.scores,
            "missing_items": self.missing_items,
            "missing_policy": self.missing_policy,
            "attempts": self.attempts,
            "timed_out": self.timed_out,
        }

def _score_bfi(raw: dict[int, int]) -> dict[str, float]:
    """Compute scores per dimension: mean of (forward + (6 - reverse))."""
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

Rules:
- Return ONE JSON object whose keys are the item numbers you were asked to answer.
- Every value must be an integer from 1 to 5.
- Do not include explanations, markdown fences, or any other text.

Example shape:
{"1": 4, "2": 2, ...}"""

def _build_bfi_user() -> str:
    lines = ["Here are 44 statements. Indicate the extent to which you agree or disagree with each statement:", ""]
    for iid, text in BFI_ITEMS:
        lines.append(f"{iid}. {text}")
    lines.append("")
    lines.append("Respond with JSON only: {\"1\": <1-5>, ..., \"44\": <1-5>}")
    return "\n".join(lines)

_BFI_USER = _build_bfi_user()

_BFI_MAX_ATTEMPTS = 3

_MISSING_POLICIES = {
    "neutral": "neutral",
    "fill": "neutral",
    "3": "neutral",
    "keep": "keep",
    "retain": "keep",
    "blank": "keep",
    "none": "keep",
    "raise": "raise",
    "error": "raise",
    "strict": "raise",
}


class BFIIncompleteError(RuntimeError):
    """Raised when the BFI is incomplete and ``missing_policy='raise'``."""

    def __init__(self, agent_id: str, missing_items: list[int], attempts: int):
        self.agent_id = agent_id
        self.missing_items = missing_items
        self.attempts = attempts
        super().__init__(
            f"BFI incomplete for '{agent_id}' after {attempts} attempt(s): "
            f"{44 - len(missing_items)}/44 answered, {len(missing_items)} missing. "
            "No values were filled in (missing_policy=raise)."
        )


def normalize_missing_policy(policy) -> str:
    """Map a YAML value to ``neutral``, ``keep`` or ``raise``."""
    if policy is None:
        return "neutral"
    key = str(policy).strip().lower()
    if key not in _MISSING_POLICIES:
        raise ValueError(
            f"Invalid BFI missing policy {policy!r}. "
            "Use 'neutral' (fill 3), 'keep' (leave unanswered) or 'raise' (abort)."
        )
    return _MISSING_POLICIES[key]

# Phrases some models use instead of a bare integer.
_SCALE_WORDS = (
    ("strongly disagree", 1),
    ("disagree strongly", 1),
    ("disagree a little", 2),
    ("neither agree nor disagree", 3),
    ("neither", 3),
    ("neutral", 3),
    ("agree a little", 4),
    ("strongly agree", 5),
    ("agree strongly", 5),
    ("disagree", 2),
    ("agree", 4),
)

_persona_builder = PersonaPromptBuilder()
_context_builder = ContextPromptBuilder()


def _preprocess_json_text(text: str) -> str:
    """Repair the most common non-JSON quirks in BFI answers.

    Handles bare numeric keys (``{1: 4}``), single-quoted numeric keys
    (``{'1': 4}``) and literal newlines inside the object.
    """
    text = re.sub(r"([{,]\s*)(\d{1,2})\s*:", r'\1"\2":', text)
    text = re.sub(r"'(\d{1,2})'\s*:", r'"\1":', text)
    text = re.sub(r":\s*'([1-5])'", r': "\1"', text)
    return text


def _unwrap_answers(value):
    """Unwrap common container shapes ({"answers": {...}}, {"items": [...]})."""
    if isinstance(value, dict):
        for key in ("answers", "responses", "items", "ratings", "result", "data"):
            inner = value.get(key)
            if isinstance(inner, (dict, list)):
                return inner
    return value


def _coerce_item_id(key) -> Optional[int]:
    if isinstance(key, bool):
        return None
    if isinstance(key, int):
        iid = key
    else:
        match = re.search(r"\d{1,2}", str(key))
        if not match:
            return None
        iid = int(match.group())
    return iid if 1 <= iid <= 44 else None


def _coerce_answer(value) -> Optional[int]:
    if isinstance(value, dict):
        for key in ("answer", "score", "value", "response", "rating"):
            if key in value:
                return _coerce_answer(value[key])
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        answer = int(value)
        return answer if 1 <= answer <= 5 else None

    text = str(value).strip()
    if not text:
        return None

    # "4", "4.0", "4 - Agree a little" — avoid matching digits inside 10-44.
    match = re.search(r"(?<!\d)([1-5])(?!\d)", text)
    if match:
        return int(match.group(1))

    lowered = text.lower()
    for phrase, answer in _SCALE_WORDS:
        if phrase in lowered:
            return answer
    return None


def _collect_answers(value) -> dict[int, int]:
    answers: dict[int, int] = {}
    value = _unwrap_answers(value)

    if isinstance(value, dict):
        for key, raw in value.items():
            iid = _coerce_item_id(key)
            answer = _coerce_answer(raw)
            if iid is not None and answer is not None:
                answers[iid] = answer
        return answers

    if isinstance(value, list):
        if value and all(not isinstance(entry, (dict, list)) for entry in value):
            # Plain array in item order: [4, 2, 3, ...]
            for iid, raw in enumerate(value, start=1):
                answer = _coerce_answer(raw)
                if iid <= 44 and answer is not None:
                    answers[iid] = answer
        else:
            # [{"item": 1, "answer": 4}, ...]
            for entry in value:
                if not isinstance(entry, dict):
                    continue
                iid = _coerce_item_id(
                    entry.get("item", entry.get("id", entry.get("number", entry.get("index"))))
                )
                answer = _coerce_answer(
                    entry.get("answer", entry.get("score", entry.get("value", entry.get("rating"))))
                )
                if iid is not None and answer is not None:
                    answers[iid] = answer

    return answers


def _parse_bfi_answers(raw: str) -> dict[int, int]:
    """Parse a BFI reply into {item_id: 1..5}, tolerating messy model output."""
    if not raw:
        return {}
    value, error = load_json_value(_preprocess_json_text(str(raw)))
    if error is not None:
        logger.warning("BFI reply is not valid JSON: %s | raw=%s", error, str(raw)[:300])
        return {}
    return _collect_answers(value)


def _batch_items_prompt(items: list[int]) -> str:
    """First prompt for a subset of items (``bfi.batch_size`` smaller than 44)."""
    statements = dict(BFI_ITEMS)
    lines = [
        "Here are some statements. Indicate the extent to which you agree or "
        "disagree with each statement:",
        "",
    ]
    for iid in items:
        lines.append(f"{iid}. {statements[iid]}")
    lines.append("")
    lines.append(
        "Respond with JSON only, using the item numbers above as keys: "
        '{"<item>": <1-5>, ...}'
    )
    return "\n".join(lines)


def _missing_items_prompt(missing: list[int]) -> str:
    statements = dict(BFI_ITEMS)
    lines = [
        "Your previous answer was incomplete or invalid. Answer ONLY the items below.",
        "Return ONE JSON object mapping each item number to an integer from 1 to 5.",
        "",
    ]
    for iid in missing:
        lines.append(f"{iid}. {statements[iid]}")
    lines.append("")
    lines.append("JSON only, with no explanations or markdown.")
    return "\n".join(lines)


def normalize_timeout(timeout) -> Optional[float]:
    """Validate a per-agent BFI time budget in seconds (``None`` disables it)."""
    if timeout is None or timeout == "":
        return None
    try:
        seconds = float(timeout)
    except (TypeError, ValueError):
        raise ValueError(
            f"Invalid BFI timeout {timeout!r}; expected a number of seconds."
        )
    if seconds <= 0:
        raise ValueError(
            f"Invalid BFI timeout {timeout!r}; it must be greater than zero."
        )
    return seconds


def _is_timeout_error(exc: Exception) -> bool:
    """Best-effort detection of provider/HTTP timeouts across adapters."""
    if isinstance(exc, TimeoutError):
        return True
    if "timeout" in type(exc).__name__.lower():
        return True
    text = str(exc).lower()
    return "timed out" in text or "timeout" in text


def _call_bfi(adapter: LLMAdapter, messages: list[dict]) -> str:
    """Prefer the adapter's native JSON mode, falling back to plain completion.

    A timeout is re-raised instead of falling back: the time budget is already
    spent, so retrying in plain mode would only exceed it further.
    """
    if getattr(adapter, "supports_structured_output", False):
        try:
            raw = adapter.complete_structured(messages)
            if raw and str(raw).strip():
                return raw
            logger.info(
                "BFI structured call returned empty content for %s; "
                "retrying in plain mode.",
                adapter.identifier,
            )
        except Exception as e:
            if _is_timeout_error(e):
                logger.warning(
                    "BFI structured call timed out for %s; not falling back.",
                    adapter.identifier,
                )
                raise
            logger.info(
                "BFI structured call failed for %s (%s); retrying in plain mode.",
                adapter.identifier,
                e,
            )
    return adapter.complete(messages)


def run_bfi_for_agent(
    agent_id: str,
    role: str,
    adapter: LLMAdapter,
    persona: Optional[Big5Persona] = None,
    context: Optional[SituationalContext] = None,
    missing_policy: str = "neutral",
    timeout: Optional[float] = None,
    batch_size: Optional[int] = None,
    max_attempts: int = _BFI_MAX_ATTEMPTS,
) -> BFIResult:
    """
    Quiz an LLM with the BFI-44 right after conditioning.
    The persona (if any) is injected into the system prompt, as is context when active.

    Incomplete or malformed replies are retried with a targeted prompt listing
    only the missing items. After all attempts, ``missing_policy`` decides what
    happens to unanswered items:

    - ``neutral`` (default): fill them with 3 (neutral) and record them.
    - ``keep``: leave them unanswered; scores use only answered items.
    - ``raise``: raise :class:`BFIIncompleteError`.

    ``timeout`` is a total time budget in seconds for this agent's whole BFI
    (all attempts included). The adapter's ``config.timeout`` is temporarily
    lowered to the remaining budget and restored afterwards, so a slow provider
    cannot stall the run indefinitely.

    ``batch_size`` splits the 44 items into smaller requests (e.g. ``11`` makes
    4 requests per agent). Smaller batches are easier for the model to answer
    correctly and retries only resend the failed batch, which avoids re-sending
    all 44 items. Leave it as ``None``/``44`` for a single request per agent.
    ``max_attempts`` caps how many times each batch may be requested.
    """
    policy = normalize_missing_policy(missing_policy)
    budget = normalize_timeout(timeout)
    deadline = None if budget is None else time.monotonic() + budget
    original_timeout = adapter.config.timeout

    chunk = 44 if batch_size is None else int(batch_size)
    if not 1 <= chunk <= 44:
        raise ValueError(
            f"Invalid BFI batch_size {batch_size!r}; expected 1-44."
        )
    if int(max_attempts) < 1:
        raise ValueError(
            f"Invalid BFI max_attempts {max_attempts!r}; expected >= 1."
        )
    attempt_limit = int(max_attempts)
    batches = [list(range(start, min(start + chunk, 45))) for start in range(1, 45, chunk)]
    # Base system prompt = persona + context (same as negotiation, without scenario)
    system_prompt = "You are an AI assistant participating in a personality study."
    if persona is not None:
        system_prompt = _persona_builder.inject(system_prompt, persona)
    if context is not None and context.is_active():
        system_prompt = _context_builder.inject(system_prompt, context)
    system_prompt += "\n\n" + _BFI_SYSTEM
    if chunk < 44:
        system_prompt += (
            "\n\n- The questionnaire is split into parts. Answer only the items "
            "listed in each request, using their original numbers as keys."
        )

    answers: dict[int, int] = {}
    attempts_used = 0
    last_raw = ""
    timed_out = False
    stop = False

    try:
        for batch_number, batch in enumerate(batches, start=1):
            if stop:
                break
            batch_attempt = 0
            while batch_attempt < attempt_limit:
                pending = [iid for iid in batch if iid not in answers]
                if not pending:
                    break

                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        timed_out = True
                        stop = True
                        logger.warning(
                            "BFI timeout for %s after %d call(s); time budget "
                            "of %.1fs exhausted.",
                            agent_id,
                            attempts_used,
                            budget,
                        )
                        break
                    # Keep the adapter call inside what is left of the budget.
                    adapter.config.timeout = remaining

                batch_attempt += 1
                attempts_used += 1

                if len(pending) == 44:
                    user_prompt = _BFI_USER
                elif batch_attempt == 1:
                    user_prompt = _batch_items_prompt(pending)
                else:
                    user_prompt = _missing_items_prompt(pending)

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]

                try:
                    raw = _call_bfi(adapter, messages)
                except Exception as e:
                    logger.warning(
                        "BFI call failed for %s (call %d, batch attempt %d/%d): %s",
                        agent_id,
                        attempts_used,
                        batch_attempt,
                        attempt_limit,
                        e,
                    )
                    if _is_timeout_error(e):
                        timed_out = True
                        stop = True
                        break
                    continue

                last_raw = raw or ""
                answers.update(_parse_bfi_answers(last_raw))
                logger.info(
                    "BFI %s call %d (batch %d/%d): %d/44 answered",
                    agent_id,
                    attempts_used,
                    batch_number,
                    len(batches),
                    len(answers),
                )
    finally:
        if budget is not None:
            adapter.config.timeout = original_timeout

    missing = [iid for iid in range(1, 45) if iid not in answers]
    if missing:
        if policy == "raise":
            logger.error(
                "BFI incomplete for %s after %d attempt(s): %d/44 answered. "
                "Aborting without filling values (missing_policy=raise).",
                agent_id,
                attempts_used,
                len(answers),
            )
            raise BFIIncompleteError(agent_id, missing, attempts_used)

        cause = " (time budget exhausted)" if timed_out else ""
        if policy == "keep":
            logger.warning(
                "BFI incomplete for %s after %d attempt(s)%s: %d/44 answered; "
                "leaving %d item(s) unanswered (missing_policy=keep). Last raw=%s",
                agent_id,
                attempts_used,
                cause,
                len(answers),
                len(missing),
                last_raw[:300],
            )
        else:
            logger.warning(
                "BFI incomplete for %s after %d attempt(s)%s: %d/44 answered; "
                "filling %d item(s) with 3 (neutral). Last raw=%s",
                agent_id,
                attempts_used,
                cause,
                len(answers),
                len(missing),
                last_raw[:300],
            )
            for iid in missing:
                answers[iid] = 3

    scores = _score_bfi(answers)
    return BFIResult(
        agent_id=agent_id,
        role=role,
        raw_answers=answers,
        scores=scores,
        model_identifier=adapter.identifier,
        missing_items=missing,
        missing_policy=policy,
        attempts=attempts_used,
        timed_out=timed_out,
    )

def run_bfi_all(
    agents: dict[str, LLMAdapter],
    personas: dict[str, Big5Persona],
    context: Optional[SituationalContext] = None,
    missing_policy: str = "neutral",
    timeout: Optional[float] = None,
    batch_size: Optional[int] = None,
    max_attempts: int = _BFI_MAX_ATTEMPTS,
) -> dict[str, BFIResult]:
    """Run BFI for all agents (called right after conditioning, before the negotiation)."""
    policy = normalize_missing_policy(missing_policy)
    budget = normalize_timeout(timeout)
    results = {}
    for role, adapter in agents.items():
        agent_id = f"{role}_{adapter.model.replace(':', '-')}"
        persona = personas.get(role) if personas else None
        res = run_bfi_for_agent(
            agent_id=agent_id,
            role=role,
            adapter=adapter,
            persona=persona,
            context=context,
            missing_policy=policy,
            timeout=budget,
            batch_size=batch_size,
            max_attempts=max_attempts,
        )
        results[agent_id] = res
        # Also index by role for compat
        results[role] = res
    return results
