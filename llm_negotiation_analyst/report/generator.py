"""
Report generator: produces a structured Markdown report from a negotiation run.

Includes:
  - Scenario summary
  - Induced Personas (if used)
  - Situational context (if used)
  - Per-agent Observed profiles (text bars + narrative)
  - Persona vs. observed comparison (if persona was used)
  - Full annotated transcript
  - Methodology notes
  - Dataset export reference
"""

from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from ..simulation.engine import NegotiationResult
from ..scoring import Big5Profile, ALL_METRICS_META, resolve_metric

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
    induced_str  = f"induced={induced}"       if induced  is not None else "induced=—"
    observed_str = f"observed={observed:.2f}" if observed > 0         else "observed=—"
    return "".join(bar) + f"  {induced_str} | {observed_str}"


def generate_report(
    result: NegotiationResult,
    profiles: dict[str, Big5Profile],
    output_path: str | Path | None = None,
) -> str:
    lines: list[str] = []
    a = lines.append
    e = lines.extend

    personas_meta: dict           = result.metadata.get("personas", {})
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

    e(["### 1.3 Induced Behavioral Personas", ""])
    if personas_meta:
        a("_These behavioral instructions were injected into each agent's system prompt "
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
                # Evita falhar se o utilizador colocou uma métrica no YAML que não existe
                try:
                    metric = resolve_metric(dim_key)
                    meta = ALL_METRICS_META[metric]
                    a(f"{meta.name:<30} {_bar(float(score))}")
                except ValueError:
                    pass
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

    # ── 3. OBSERVED BEHAVIORAL PROFILES ─────────────────────────────────
    e(["## 3. Observed Behavioral Profiles", ""])
    a("_Scores = mean of per-turn LLM-as-judge evaluations. "
      "Scale: 1 (low pole) → 5 (high pole). "
      "Turns where the judge failed are excluded. "
      "⚠ = low observability in text negotiations._")
    a("")

    # Descobre quais métricas foram ativamente avaliadas nesta simulação
    evaluated_dims = set()
    for profile in profiles.values():
        evaluated_dims.update(profile.scores.keys())

    for agent_id, profile in profiles.items():
        role    = result.agent_roles.get(agent_id, "")
        induced = personas_meta.get(role, {})

        a(f"### {agent_id}  _(role: {role})_")
        a(f"**Model:** `{profile.model_identifier}`")
        a("")

        # Observed bars (Só mostra as que foram efetivamente medidas)
        a("```")
        for dim in sorted(list(evaluated_dims), key=lambda d: d.value):
            meta  = ALL_METRICS_META[dim]
            score = profile.scores.get(dim, 0.0)
            note  = "⚠ low observability" if str(meta.observability) in ("baixa", "low", "1", "2") else ""
            a(f"{meta.name:<30} {_bar(score)}  {note}")
        a("```")
        a("")

        # Persona vs. observed comparison
        if induced:
            a("**Induced (▼) vs. Observed (█) comparison:**")
            a("")
            a("```")
            header_bar = f"{'Dimension':<30} {'1────────────3────────────5'}"
            a(header_bar)
            a("-" * len(header_bar))
            # Mostra apenas as que o utilizador induziu no config ou que foram medidas
            dims_to_compare = [d for d in ALL_METRICS_META.keys() if (d.value in induced or d in evaluated_dims)]
            for dim in sorted(dims_to_compare, key=lambda d: d.value):
                meta     = ALL_METRICS_META[dim]
                ind_val  = induced.get(dim.value)
                obs_val  = profile.scores.get(dim)
                a(f"{meta.name:<30} {_comparison_bar(ind_val, obs_val)}")
            a("```")
            a("")
            a("_▼ = induced score (instructed). █ = observed score (emergent). "
              "Alignment indicates the model followed the persona._")
            a("")

        # Behavioral observations
        a("#### Behavioral Observations")
        a("")
        scored = [d for d in evaluated_dims if profile.scores.get(d) is not None]
        if not scored:
            a("_No dimensions were successfully scored._")
            a("")
        else:
            for dim in sorted(scored, key=lambda d: d.value):
                meta  = ALL_METRICS_META[dim]
                score = profile.scores[dim]
                pole  = meta.high_pole if score >= 3 else meta.low_pole
                warn  = " _(⚠ interpret with caution)_" if str(meta.observability) in ("baixa", "low", "1", "2") else ""

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
    a("_Side-by-side observed scores. Useful for spotting behavioral asymmetries._")
    a("")
    agent_ids = list(profiles.keys())
    a("| Dimension |" + "".join(f" {aid} |" for aid in agent_ids))
    a("|-----------|" + "".join("-----------|" for _ in agent_ids))
    for dim in sorted(list(evaluated_dims), key=lambda d: d.value):
        meta = ALL_METRICS_META[dim]
        flag = " ⚠" if str(meta.observability) in ("baixa", "low", "1", "2") else ""
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
                f"{ALL_METRICS_META[d].abbreviation}={s.score:.1f}(conf={s.confidence:.2f})"
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
    a("- **Observability:** Some psychological metrics are less reliable in text. Treat with caution.")
    a("- **Persona injection:** Behavioral instructions prepended to system prompt. "
      "Models may deviate from induced disposition.")
    a("- **Situational context:** Injected equally into all agents. Effect is not directly measured.")
    a("- **Judge independence:** Judge should differ from negotiating agents to avoid self-evaluation bias.")
    a("- **Reproducibility:** Use `temperature=0` and `seed` (Ollama/LMStudio) for deterministic results.")
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

    report_md = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(report_md, encoding="utf-8")

    return report_md
