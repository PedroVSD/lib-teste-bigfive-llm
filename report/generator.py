"""
Report generator: produces a structured Markdown report from a negotiation run.

Includes:
  - Scenario summary
  - Full annotated transcript
  - Per-agent Big Five radar summary (text-based)
  - Key behavioral observations
  - Dataset export reminder
"""

from pathlib import Path
from datetime import datetime, timezone

from ..simulation.engine import NegotiationResult
from ..scoring.big5 import Big5Profile, Dimension, BIG5_META


# Unicode block chars for ASCII radar bars
_BAR = "█"
_BAR_EMPTY = "░"
_BAR_WIDTH = 20


def _bar(score: float, width: int = _BAR_WIDTH) -> str:
    """Render a score 1–5 as a text progress bar."""
    filled = int(round((score - 1) / 4 * width))
    return _BAR * filled + _BAR_EMPTY * (width - filled) + f"  {score:.2f}/5"


def generate_report(
    result: NegotiationResult,
    profiles: dict[str, Big5Profile],
    output_path: str | Path | None = None,
) -> str:
    """
    Generate a Markdown report for a negotiation run.

    Args:
        result:      NegotiationResult from the engine.
        profiles:    Dict of Big5Profile per agent from the evaluator.
        output_path: If provided, write the report to this file.

    Returns:
        The report as a Markdown string.
    """
    lines = []
    a = lines.append  # shorthand

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    a(f"# Negotiation Analysis Report")
    a(f"")
    a(f"> **Scenario:** {result.scenario_description}")
    a(f"> **Run ID:** `{result.run_id}`  ")
    a(f"> **Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ")
    a(f"> **Duration:** {result.duration_seconds:.1f}s | **Turns:** {result.total_turns} | "
      f"**Settled:** {'✅ Yes' if result.settled else '❌ No'}")
    a(f"")

    # ------------------------------------------------------------------
    # Agents
    # ------------------------------------------------------------------
    a(f"## Agents")
    a(f"")
    a(f"| Agent ID | Model | Role |")
    a(f"|----------|-------|------|")
    for agent_id, model_id in result.agents.items():
        role = result.agent_roles.get(agent_id, "—")
        a(f"| `{agent_id}` | `{model_id}` | {role} |")
    a(f"")

    # ------------------------------------------------------------------
    # Big Five Profiles
    # ------------------------------------------------------------------
    a(f"## Big Five Personality Profiles")
    a(f"")
    a(
        "_Scores are aggregated means across all scored turns. "
        "Scale: 1 (low pole) → 5 (high pole). "
        "Observability note: Openness and Extraversion are less reliably "
        "captured in text negotiations than Agreeableness, Conscientiousness, "
        "and Neuroticism._"
    )
    a(f"")

    for agent_id, profile in profiles.items():
        role = result.agent_roles.get(agent_id, "")
        model = profile.model_identifier
        a(f"### {agent_id}  _(role: {role})_")
        a(f"**Model:** `{model}`")
        a(f"")
        a(f"```")
        for dim in Dimension:
            meta = BIG5_META[dim]
            score = profile.scores.get(dim, 0.0)
            obs_note = "⚠ low observability" if meta.observability <= 2 else ""
            a(f"{meta.name:<24} {_bar(score)}  {obs_note}")
        a(f"```")
        a(f"")

        # Narrative per dimension
        a(f"#### Behavioral observations")
        a(f"")
        for dim in Dimension:
            meta = BIG5_META[dim]
            score = profile.scores.get(dim)
            if score is None:
                continue
            # Gather justifications from per-turn scores
            justifications = [
                s.justification
                for s in profile.per_turn_scores
                if s.dimension == dim and s.justification and not s.justification.startswith("[Eval")
            ]
            a(f"**{meta.name}** (score: {score:.2f}) — *{meta.high_pole if score >= 3 else meta.low_pole}*")
            if justifications:
                # Show the most informative justification (longest)
                best = max(justifications, key=len)
                a(f"> {best}")
            a(f"")

        if profile.notes:
            a(f"_Notes: {profile.notes}_")
            a(f"")

    # ------------------------------------------------------------------
    # Transcript
    # ------------------------------------------------------------------
    a(f"## Full Transcript")
    a(f"")
    a(f"_The transcript is annotated with per-turn scores where available._")
    a(f"")

    # Build a lookup: (agent_id, turn_index) → scores
    score_lookup: dict[tuple, dict[Dimension, float]] = {}
    for profile in profiles.values():
        for s in profile.per_turn_scores:
            key = (profile.agent_id, s.turn_index)
            if key not in score_lookup:
                score_lookup[key] = {}
            score_lookup[key][s.dimension] = s.score

    for turn in result.transcript:
        a(f"---")
        a(f"**Turn {turn.turn_index} · {turn.role.upper()}** "
          f"(`{turn.agent_id}`)"
          + (f" _{turn.latency_ms:.0f}ms_" if turn.latency_ms else ""))
        a(f"")
        a(turn.content)
        a(f"")

        scores = score_lookup.get((turn.agent_id, turn.turn_index))
        if scores:
            score_str = " | ".join(
                f"{BIG5_META[d].abbreviation}={v:.1f}"
                for d, v in scores.items()
            )
            a(f"<sub>Big5 scores: {score_str}</sub>")
            a(f"")

    a(f"---")
    a(f"")

    # ------------------------------------------------------------------
    # Methodology note
    # ------------------------------------------------------------------
    a(f"## Methodology Notes")
    a(f"")
    a(
        "- **Scoring method:** LLM-as-judge with structured JSON rubrics per Big Five dimension.\n"
        "- **Aggregation:** Mean of per-turn scores. Turns with failed evaluations are excluded.\n"
        "- **Neuroticism direction:** High score (5) = emotionally reactive. "
        "This follows standard NEO-PI-R convention and is **not** inverted in this report.\n"
        "- **Observability:** Openness (O) and Extraversion (E) have low observability in "
        "text negotiations and should be interpreted with caution.\n"
        "- **Replication:** Set `temperature=0` on both agents and the judge for deterministic results. "
        "For local models (Ollama), also fix `seed` in `AdapterConfig.extra`.\n"
        "- **Bias mitigation:** Use a judge model different from the negotiating agents. "
        "For critical studies, use dual judges and report inter-rater reliability (IRR)."
    )
    a(f"")

    # ------------------------------------------------------------------
    # Dataset reference
    # ------------------------------------------------------------------
    a(f"## Dataset Export")
    a(f"")
    a(
        "Structured JSONL exports for downstream analysis are saved alongside this report:\n\n"
        f"- `transcripts/{result.scenario_name}_{result.run_id}.jsonl` — one row per turn\n"
        f"- `scores/{result.scenario_name}_{result.run_id}_scores.jsonl` — one row per agent with full per-turn scores\n\n"
        "Load with pandas:\n"
        "```python\n"
        "import pandas as pd\n"
        f"df_scores = pd.read_json('results/scores/{result.scenario_name}_{result.run_id}_scores.jsonl', lines=True)\n"
        "```"
    )

    report_md = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(report_md, encoding="utf-8")

    return report_md
