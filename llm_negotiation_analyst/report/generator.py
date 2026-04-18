"""
Report generator: produces a structured Markdown report from a negotiation run.

Includes:
  - Scenario summary
  - Induced Big Five personas (if used)
  - Situational context (if used)
  - Per-agent Big Five observed profiles (text bars + narrative)
  - Persona vs. observed comparison (if persona was used)
  - Full annotated transcript
  - Methodology notes
  - Dataset export reference
"""

from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from ..simulation.engine import NegotiationResult
from ..scoring.big5 import Big5Profile, Dimension, BIG5_META

_BAR   = "█"
_EMPTY = "░"
_W     = 20


def _bar(score: float, width: int = _W) -> str:
    if score <= 0:
        return _EMPTY * width + "  —"
    filled = max(0, min(width, int(round((score - 1) / 4 * width))))
    return _BAR * filled + _EMPTY * (width - filled) + f"  {score:.2f}/5"


def _comparison_bar(induced: Optional[int], observed: Optional[float], width: int = _W) -> str:
    if observed is None:
        observed = 0.0
    bar = list(_EMPTY * width)
    if induced is not None:
        idx = max(0, min(width - 1, int(round((induced - 1) / 4 * width))))
        bar[idx] = "▼"
    obs_filled = max(0, min(width, int(round((observed - 1) / 4 * width)))) if observed > 0 else 0
    for i in range(obs_filled):
        if bar[i] == _EMPTY:
            bar[i] = _BAR
    induced_str  = f"induced={induced}"      if induced  is not None else "induced=—"
    observed_str = f"observed={observed:.2f}" if observed > 0          else "observed=—"
    return "".join(bar) + f"  {induced_str} | {observed_str}"


def generate_report(
    result: NegotiationResult,
    profiles: dict[str, Big5Profile],
    output_path: str | Path | None = None,
) -> str:
    lines: list[str] = []
    a = lines.append
    e = lines.extend

    personas_meta: dict        = result.metadata.get("personas", {})
    context_meta:  Optional[dict] = result.metadata.get("context")

    # ── HEADER ────────────────────────────────────────────────────────
    a("# Negotiation Analysis Report")
    a("")
    a(f"> **Scenario:** {result.scenario_description}  ")
    a(f"> **Run ID:** `{result.run_id}`  ")
    a(f"> **Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ")
    a(f"> **Duration:** {result.duration_seconds:.1f}s | "
      f"**Turns:** {result.total_turns} | "
      f"**Settled:** {'✅ Yes' if result.settled else '❌ No'}")
    a("")

    # ── 1. EXPERIMENT SETUP ───────────────────────────────────────────
    e(["## 1. Experiment Setup", ""])

    e(["### 1.1 Scenario", ""])
    a(f"**Name:** `{result.scenario_name}`  ")
    a(f"**Description:** {result.scenario_description}")
    a("")
    a("**Shared context given to both agents:**")
    a("")
    a(f"> {result.scenario_context}")
    a("")

    e(["### 1.2 Agents", ""])
    a("| Agent ID | Model | Role |")
    a("|----------|-------|------|")
    for agent_id, model_id in result.agents.items():
        role = result.agent_roles.get(agent_id, "—")
        a(f"| `{agent_id}` | `{model_id}` | {role} |")
    a("")

    e(["### 1.3 Induced Big Five Personas", ""])
    if personas_meta:
        a("_These personality instructions were injected into each agent's system prompt "
          "before the negotiation. They represent the **intended** disposition — "
          "see Section 3 for observed scores._")
        a("")
        for role, scores in personas_meta.items():
            if not scores:
                continue
            a(f"**{role}**")
            a("")
            a("```")
            for dim_key, score in scores.items():
                meta = BIG5_META[Dimension(dim_key)]
                a(f"{meta.name:<24} {_bar(float(score))}")
            a("```")
            a("")
    else:
        a("_No persona induced. Agents used their default behavioral disposition._")
        a("")

    e(["### 1.4 Situational Context", ""])
    if context_meta and context_meta.get("enabled", True):
        active = {
            k: v for k, v in context_meta.items()
            if k != "enabled" and v not in (None, [], {}, "")
        }
        if active:
            a("_The following conditions were injected into both agents' system prompts._")
            a("")
            a("| Condition | Value |")
            a("|-----------|-------|")
            labels = {
                "inflation":         "Inflation level",
                "interest_rates":    "Interest rates",
                "government":        "Government orientation",
                "crises":            "Active crises",
                "gdp_growth":        "GDP growth",
                "unemployment":      "Unemployment",
                "country":           "Country / region",
                "year":              "Year",
                "custom_conditions": "Custom conditions",
            }
            for field, label in labels.items():
                val = active.get(field)
                if val in (None, [], ""):
                    continue
                if isinstance(val, list):
                    val = ", ".join(str(v) for v in val)
                a(f"| {label} | {val} |")
            a("")
        else:
            a("_Context object present but no conditions were active._")
            a("")
    else:
        a("_No situational context applied._")
        a("")

    # ── 2. NEGOTIATION OUTCOME ────────────────────────────────────────
    e(["## 2. Negotiation Outcome", ""])
    a(f"- **Agreement reached:** {'Yes ✅' if result.settled else 'No ❌'}")
    a(f"- **Total turns:** {result.total_turns}")
    a(f"- **Total duration:** {result.duration_seconds:.1f}s")
    latencies = [t.latency_ms for t in result.transcript if t.latency_ms]
    if latencies:
        a(f"- **Average turn latency:** {sum(latencies)/len(latencies):.0f}ms")
    a("")
    a("**Turns per agent:**")
    a("")
    a("| Agent | Role | Turns |")
    a("|-------|------|-------|")
    for agent_id in result.agents:
        role  = result.agent_roles.get(agent_id, "—")
        count = sum(1 for t in result.transcript if t.agent_id == agent_id)
        a(f"| `{agent_id}` | {role} | {count} |")
    a("")

    # ── 3. OBSERVED BIG FIVE PROFILES ─────────────────────────────────
    e(["## 3. Observed Big Five Profiles", ""])
    a("_Scores = mean of per-turn LLM-as-judge evaluations. "
      "Scale: 1 (low pole) → 5 (high pole). "
      "Turns where the judge failed are excluded. "
      "⚠ = low observability in text negotiations._")
    a("")

    for agent_id, profile in profiles.items():
        role    = result.agent_roles.get(agent_id, "")
        induced = personas_meta.get(role, {})

        a(f"### {agent_id}  _(role: {role})_")
        a(f"**Model:** `{profile.model_identifier}`")
        a("")

        # Observed bars
        a("```")
        for dim in Dimension:
            meta  = BIG5_META[dim]
            score = profile.scores.get(dim, 0.0)
            note  = "⚠ low observability" if meta.observability <= 2 else ""
            a(f"{meta.name:<24} {_bar(score)}  {note}")
        a("```")
        a("")

        # Persona vs. observed comparison
        if induced:
            a("**Induced (▼) vs. Observed (█) comparison:**")
            a("")
            a("```")
            header_bar = f"{'Dimension':<24} {'1────────────3────────────5'}"
            a(header_bar)
            a("-" * len(header_bar))
            for dim in Dimension:
                meta     = BIG5_META[dim]
                ind_val  = induced.get(dim.value)
                obs_val  = profile.scores.get(dim)
                a(f"{meta.name:<24} {_comparison_bar(ind_val, obs_val)}")
            a("```")
            a("")
            a("_▼ = induced score (instructed). █ = observed score (emergent). "
              "Alignment indicates the model followed the persona._")
            a("")

        # Behavioral observations
        a("#### Behavioral Observations")
        a("")
        scored = [d for d in Dimension if profile.scores.get(d) is not None]
        if not scored:
            a("_No dimensions were successfully scored._")
            a("")
        else:
            for dim in scored:
                meta  = BIG5_META[dim]
                score = profile.scores[dim]
                pole  = meta.high_pole if score >= 3 else meta.low_pole
                warn  = " _(⚠ interpret with caution)_" if meta.observability <= 2 else ""

                justifications = [
                    s.justification
                    for s in profile.per_turn_scores
                    if s.dimension == dim
                    and s.justification
                    and not s.justification.startswith("[Eval")
                    and s.confidence > 0
                ]

                a(f"**{meta.name}** — `{score:.2f}/5` — *{pole}*{warn}")
                if justifications:
                    a(f"> {max(justifications, key=len)}")
                else:
                    a("> _No valid justifications recorded._")
                a("")

        if profile.notes:
            a(f"_Notes: {profile.notes}_")
            a("")

    # ── 4. CROSS-AGENT SUMMARY ────────────────────────────────────────
    e(["## 4. Cross-Agent Comparison", ""])
    a("_Side-by-side observed Big Five scores. Useful for spotting behavioral asymmetries._")
    a("")
    agent_ids = list(profiles.keys())
    a("| Dimension |" + "".join(f" {aid} |" for aid in agent_ids))
    a("|-----------|" + "".join("-----------|" for _ in agent_ids))
    for dim in Dimension:
        meta = BIG5_META[dim]
        flag = " ⚠" if meta.observability <= 2 else ""
        row  = f"| {meta.name}{flag} |"
        for aid in agent_ids:
            score = profiles[aid].scores.get(dim)
            row  += f" {'`'+f'{score:.2f}`' if score is not None else '—'} |"
        a(row)
    a("")
    a("_⚠ = low observability._")
    a("")

    # ── 5. FULL TRANSCRIPT ────────────────────────────────────────────
    e(["## 5. Full Transcript", ""])
    a("_Scores of 3.00 with confidence=0 indicate the judge failed for that turn._")
    a("")

    score_lookup: dict[tuple, dict] = {}
    for profile in profiles.values():
        for s in profile.per_turn_scores:
            key = (profile.agent_id, s.turn_index)
            score_lookup.setdefault(key, {})[s.dimension] = s

    for turn in result.transcript:
        lat = f" _{turn.latency_ms:.0f}ms_" if turn.latency_ms else ""
        a("---")
        a(f"**Turn {turn.turn_index} · {turn.role.upper()}** (`{turn.agent_id}`){lat}")
        a("")
        a(turn.content)
        a("")
        dim_scores = score_lookup.get((turn.agent_id, turn.turn_index), {})
        valid = {d: s for d, s in dim_scores.items() if s.confidence > 0}
        if valid:
            parts = " | ".join(
                f"{BIG5_META[d].abbreviation}={s.score:.1f}(conf={s.confidence:.2f})"
                for d, s in valid.items()
            )
            a(f"<sub>📊 {parts}</sub>")
            a("")

    a("---")
    a("")

    # ── 6. METHODOLOGY NOTES ──────────────────────────────────────────
    e(["## 6. Methodology Notes", ""])
    a("- **Scoring:** LLM-as-judge with structured JSON rubrics, one call per dimension per turn.")
    a("- **Aggregation:** Arithmetic mean of valid per-turn scores (confidence > 0).")
    a("- **Neuroticism direction:** Score 5 = emotionally reactive. NEO-PI-R convention, not inverted.")
    a("- **Observability:** O and E are less reliable in text. Treat with caution.")
    a("- **Persona injection:** Behavioral instructions prepended to system prompt. "
      "Models may deviate from induced disposition.")
    a("- **Situational context:** Injected equally into all agents. Effect is not directly measured.")
    a("- **Judge independence:** Judge should differ from negotiating agents to avoid self-evaluation bias.")
    a("- **Reproducibility:** Use `temperature=0` and `seed` (Ollama) for deterministic results.")
    a("- **IRR:** When `second_judge` is used, `confidence` reflects normalized inter-rater reliability per turn. "
      "Values below 0.75 indicate significant disagreement.")
    a("")

    # ── 7. DATASET EXPORT ─────────────────────────────────────────────
    e(["## 7. Dataset Export", ""])
    a("Structured JSONL exports for downstream analysis:")
    a("")
    a(f"- `transcripts/{result.scenario_name}_{result.run_id}.jsonl` — one row per turn")
    a(f"- `scores/{result.scenario_name}_{result.run_id}_scores.jsonl` — one row per agent")
    a(f"- `runs_index.jsonl` — global index of all runs")
    a("")
    a("```python")
    a("import pandas as pd")
    a("")
    a(f"df_scores = pd.read_json(")
    a(f"    'results/scores/{result.scenario_name}_{result.run_id}_scores.jsonl', lines=True")
    a(f")")
    a(f"df_transcript = pd.read_json(")
    a(f"    'results/transcripts/{result.scenario_name}_{result.run_id}.jsonl', lines=True")
    a(f")")
    a("")
    a("# Agreeableness por agente")
    a("df_scores['agreeableness'] = df_scores['scores'].apply(lambda s: s.get('agreeableness'))")
    a("print(df_scores[['agent_id', 'model_identifier', 'agreeableness']])")
    a("```")

    report_md = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(report_md, encoding="utf-8")

    return report_md
