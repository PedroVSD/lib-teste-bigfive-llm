"""
Report Generator: Produces a structured Markdown simulation report.

Sections:
  1. Experiment Setup
  3. Behavioral Metrics (categorical)
  4. Agent Comparison (% occurrence_rate)
  5. Economic Utility (continuous 0-1)
  6. Post-negotiation Satisfaction (ordinal 1-7)
  7. Full Transcript
  8. Methodology Notes
  9. Raw Data
"""

from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from ..simulation.engine import NegotiationResult
from ..scoring import Big5Profile, ALL_METRICS_META, resolve_metric
from ..scoring.big5 import BehavioralResult
from ..scoring.report_sections import render_utility_section, render_satisfaction_section


def _detect_settlement_retroactively(result: NegotiationResult) -> bool:
    indicators = [
        "SIMULACAO_CONCLUIDA",
        "ACORDO_FECHADO",
        "[ACORDO_FECHADO]",
        "confirmo os termos",
        "iniciar a implementação",
        "parceria firmada",
        "muito feliz que conseguimos",
        "we have a deal",
        "acordo fechado",
        "aceito os termos",
        "fechado",
        "deal",
    ]
    try:
        extra = result.metadata.get("settlement_keywords", [])
        if extra:
            indicators.extend(extra)
    except Exception:
        pass
    confirmed_roles: set[str] = set()
    for turn in result.transcript:
        if not turn.content:
            continue
        content_lower = turn.content.lower()
        if any(ind.lower() in content_lower for ind in indicators):
            confirmed_roles.add(turn.role)
    required = len(result.agent_roles) if result.agent_roles else 2
    distinct_roles_in_transcript = len({t.role for t in result.transcript})
    if distinct_roles_in_transcript <= 1:
        return len(confirmed_roles) >= 1
    return len(confirmed_roles) >= min(2, required)


def _induced_expected(dim, val) -> Optional[BehavioralResult]:
    """Convert induced (positive/negative/enabled/disabled) to expected PRESENT/ABSENT, respecting polarity."""
    if val is None:
        return None
    from ..scoring.big5 import Dimension as _D
    is_big5 = isinstance(dim, _D)
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("none", "null", "nil", "disabled", "false", "off", "not_applicable"):
            # disabled/none = not induced → no expectation; return None to skip comparison
            if v in ("disabled",):
                return BehavioralResult.ABSENT
            return None
        if is_big5:
            if v == "positive":
                return BehavioralResult.PRESENT  # high pole expected
            if v == "negative":
                return BehavioralResult.ABSENT   # low pole expected → PRESENT absent
            # legacy numeric as string
            try:
                num = float(v)
                # >=3 → positive → PRESENT
                return BehavioralResult.PRESENT if num >= 3.0 else BehavioralResult.ABSENT
            except:
                return None
        else:
            if v in ("enabled", "present", "positive"):
                return BehavioralResult.PRESENT
            if v in ("absent", "negative"):
                return BehavioralResult.ABSENT
            try:
                num = float(v)
                return BehavioralResult.PRESENT if num >= 3.0 else BehavioralResult.ABSENT
            except:
                return None
    if isinstance(val, (int, float)):
        if is_big5:
            return BehavioralResult.PRESENT if float(val) >= 3.0 else BehavioralResult.ABSENT
        else:
            return BehavioralResult.PRESENT if float(val) >= 3.0 else BehavioralResult.ABSENT
    return None


def _format_occurrence(summary) -> str:
    """Format occurrence_rate as 65% (13/20; 5 NA)."""
    if summary is None or summary.occurrence_rate is None:
        if summary and summary.total_applicable == 0:
            return f"— (0 applicable; {summary.not_applicable} NA)"
        return "—"
    pct = round(summary.occurrence_rate * 100)
    return f"**{pct}%** ({summary.present}/{summary.total_applicable}; {summary.not_applicable} NA)"


def _format_occurrence_compact(summary) -> str:
    if summary is None or summary.occurrence_rate is None:
        return "—"
    pct = round(summary.occurrence_rate * 100)
    return f"{pct}%"


def _alignment(expected: Optional[BehavioralResult], summary) -> str:
    if expected is None or summary is None or summary.occurrence_rate is None:
        return "—"
    # expected PRESENT → aligned when occurrence >= 50%; expected ABSENT → aligned when <50%
    occurred = summary.occurrence_rate >= 0.5
    expected_present = expected == BehavioralResult.PRESENT
    if occurred == expected_present:
        return "✅ Compatible"
    else:
        return "❌ Not compatible"


def generate_report(
    result: NegotiationResult,
    profiles: dict[str, Big5Profile],
    output_path: str | Path | None = None,
    utility_results: Optional[dict] = None,
    satisfaction_results: Optional[dict] = None,
    bfi_results: Optional[dict] = None,
) -> str:
    lines: list[str] = []
    a = lines.append
    e = lines.extend

    settled = result.settled or _detect_settlement_retroactively(result)
    personas_meta = result.metadata.get("personas", {})
    context_meta = result.metadata.get("context")
    experiment_name = result.metadata.get("experiment_name")
    display_name = result.metadata.get("experiment_display_name") or result.metadata.get("experiment_title") or result.metadata.get("yaml_name")

    # Agreement categorical
    agreement_label = "AGREEMENT" if settled else "NO_AGREEMENT"
    ended_by = (result.metadata.get("ended_by") if result.metadata else None) or ("agreement" if settled else "turn_limit")
    max_rounds = result.metadata.get("max_rounds") if result.metadata else None

    # ── HEADER ─────────────────────────────────────────────────────
    a("# Negotiation Analysis Report")
    a("")
    if display_name and display_name != experiment_name:
        a(f"> **Experiment:** {display_name}  ")
        if experiment_name:
            a(f"> **File:** `{experiment_name}`  ")
    elif experiment_name:
        a(f"> **Experiment:** `{experiment_name}`  ")
    a(f"> **Scenario:** {result.scenario_description}  ")
    a(f"> **Run ID:** `{result.run_id}`  ")
    a(f"> **Generated at:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ")
    if max_rounds:
        a(f"> **Duration:** {result.duration_seconds:.1f}s | **Turns:** {result.total_turns} ({max_rounds} rounds)")
    else:
        a(f"> **Duration:** {result.duration_seconds:.1f}s | **Turns:** {result.total_turns}")
    a(f"> **Agreement:** `{agreement_label}`")
    if ended_by == "turn_limit" and not settled:
        a(f"> **Ending:** turn limit reached — no agreement (NO_AGREEMENT)")
    a("")

    # ── 1. SETUP ────────────────────────────────
    e(["## 1. Experiment Setup", ""])

    e(["### 1.1 Scenario", ""])
    a(f"**Name:** `{result.scenario_name}`  ")
    a(f"**Description:** {result.scenario_description}")
    a("")
    a("**Shared context sent to both agents:**")
    a("")
    a(f"> {result.scenario_context}")
    a("")

    e(["### 1.2 Agents", ""])
    a("| Agent ID | LLM Model | Scenario Role |")
    a("|-----------|------------|------------------|")
    for agent_id, model_id in result.agents.items():
        role = result.agent_roles.get(agent_id, "—")
        a(f"| `{agent_id}` | `{model_id}` | {role} |")
    a("")

    e(["### 1.3 Induced Personas", ""])
    if personas_meta:
        a("_Personality instructions injected into the System Prompt (Target Behavior)._")
        a("")
        has_any = False
        for role, scores in personas_meta.items():
            if not scores:
                continue
            has_any = True
            a(f"**Role: {role}**")
            a("")
            a("| Dimension | Induced Value (Target) | Expected (categorical) |")
            a("|-----------|-------------------------|----------------------|")
            for dim_key, score in scores.items():
                if isinstance(score, str) and score.lower() in ("none", "false", "off"):
                    continue
                if score is None:
                    continue
                # disabled treated as expected ABSENT but still shown
                if isinstance(score, str) and score.lower() == "disabled":
                    # show as DISABLED
                    try:
                        metric = resolve_metric(dim_key)
                        meta = ALL_METRICS_META[metric]
                        a(f"| {meta.name} | **DISABLED** | `ABSENT` |")
                    except ValueError:
                        a(f"| {dim_key} | **DISABLED** | `ABSENT` |")
                    continue
                try:
                    metric = resolve_metric(dim_key)
                    meta = ALL_METRICS_META[metric]
                    exp = _induced_expected(metric, score)
                    exp_str = exp.value if exp else "—"
                    if isinstance(score, str) and score.lower() in ("positive", "negative", "enabled"):
                        disp = f"**{score.upper()}**"
                    else:
                        disp = f"**{score}**"
                    a(f"| {meta.name} | {disp} | `{exp_str}` |")
                except ValueError:
                    a(f"| {dim_key} | **{score}** | — |")
            a("")
        if not has_any:
            a("_Personas configured but all disabled (none)._")
            a("")
    else:
        a("_No induced persona. Models acted with default behavior._")
        a("")

    e(["### 1.4 Situational Context (Macro)", ""])
    if context_meta and context_meta.get("enabled", True):
        active = {k: v for k, v in context_meta.items() if k != "enabled" and v not in (None, [], {}, "")}
        if active:
            a("| External Condition | Configured Value |")
            a("|-------------------|-------------------|")
            for field, val in active.items():
                if isinstance(val, list):
                    val = ", ".join(str(v) for v in val)
                a(f"| {field.replace('_', ' ').capitalize()} | {val} |")
            a("")
        else:
            a("_No special macroeconomic condition activated._")
            a("")
    else:
        a("_Situational context disabled for this run._")
        a("")

    a("")

    # ── 3. BEHAVIORAL METRICS ──────────────────────────
    e(["## 3. Behavioral Metrics (Categorical Observation)", ""])
    a("_Each per-turn metric is classified as `PRESENT` / `ABSENT` / `NOT_APPLICABLE` with textual evidence. `NOT_APPLICABLE` does not enter the denominator._")
    a("")
    a("**Aggregation:** `occurrence_rate = PRESENT / (PRESENT + ABSENT)` — occurrence percentage over applicable turns.")
    a("")

    # collect evaluated dimensions
    evaluated_dims = set()
    for profile in profiles.values():
        evaluated_dims.update(profile.summaries.keys())
        # fallback legacy
        if not profile.summaries and profile.scores:
            evaluated_dims.update(profile.scores.keys())

    from ..scoring.big5 import Dimension as _Big5Dimension
    from ..scoring.negotiation_metrics import NegotiationMetric as _NegotiationMetric
    _BIG5_ORDER = [_Big5Dimension.OPENNESS, _Big5Dimension.CONSCIENTIOUSNESS,
                   _Big5Dimension.EXTRAVERSION, _Big5Dimension.AGREEABLENESS, _Big5Dimension.NEUROTICISM]
    def _dim_sort_key(d):
        if isinstance(d, _Big5Dimension):
            try:
                return (0, _BIG5_ORDER.index(d))
            except ValueError:
                return (0, 99)
        return (1, d.value)

    for agent_id, profile in profiles.items():
        role = result.agent_roles.get(agent_id, "")
        induced = personas_meta.get(role, {})

        a(f"### Agent: {agent_id} ({role})")
        a("")

        a("| Evaluated Dimension | Induced | Expected | Observed (occurrence_rate) | Alignment |")
        a("|-------------------|----------|----------|-------------------------------|-------------|")

        # dimensions to show: all evaluated + all induced
        induced_metrics = set()
        for k in induced.keys():
            try:
                induced_metrics.add(resolve_metric(k))
            except:
                pass
        dims_to_show = sorted(list(evaluated_dims | induced_metrics), key=_dim_sort_key)

        for dim in dims_to_show:
            meta = ALL_METRICS_META[dim]
            ind_val = induced.get(dim.value)
            # summary may live in summaries (new) or scores (legacy)
            summary = profile.summaries.get(dim)
            if summary is None and dim in profile.scores:
                # legacy: scores[metric] is float occurrence_rate
                occ = profile.scores.get(dim)
                # cannot reconstruct counts: not available
                summary = None
                obs_str = f"{round(occ*100)}%" if occ is not None else "—"
                ind_exp = _induced_expected(dim, ind_val)
                ind_str = ind_val.upper() if isinstance(ind_val, str) else (str(ind_val) if ind_val is not None else "—")
                exp_str = ind_exp.value if ind_exp else "—"
                status = _alignment(ind_exp, type("S", (), {"occurrence_rate": occ, "present": 0, "total_applicable": 0, "not_applicable": 0})() if occ is not None else None)
                a(f"| {meta.name} | {ind_str} | {exp_str} | {obs_str} | {status} |")
                continue
            ind_str = ind_val.upper() if isinstance(ind_val, str) else (str(ind_val) if ind_val is not None else "—")
            ind_exp = _induced_expected(dim, ind_val)
            exp_str = ind_exp.value if ind_exp else "—"
            obs_str = _format_occurrence(summary)
            status = _alignment(ind_exp, summary)
            a(f"| {meta.name} | {ind_str} | `{exp_str}` | {obs_str} | {status} |")
        a("")

        # Per-dimension observations — readable
        a("#### Evidence per dimension (samples per turn)")
        a("")
        a("_Each line below is one turn's categorical observation with short textual evidence. `occurrence_rate` already summarized in the table above._")
        a("")
        obs_by_dim = {}
        source = profile.observations
        for o in source:
            obs_by_dim.setdefault(o.dimension, []).append(o)
        if not obs_by_dim:
            a("_No observations recorded._")
            a("")
        else:
            for dim in sorted(obs_by_dim.keys(), key=_dim_sort_key):
                meta = ALL_METRICS_META[dim]
                summary = profile.summaries.get(dim)
                occ = _format_occurrence(summary) if summary else "—"
                warn = " _(⚠ low observability)_" if meta.observability <= 2 else ""
                a(f"**{meta.name}** (`{meta.abbreviation}`) — {occ} — *{meta.high_pole} ↔ {meta.low_pole}*{warn}")
                a("")
                # Per-dimension table — shows all applicable observations, NOT_APPLICABLE collapsed
                a("| Turn | Result | Conf. | Evidence |")
                a("|-------|-----------|-------|----------|")
                shown = 0
                for o in sorted(obs_by_dim[dim], key=lambda x: x.turn_index):
                    if o.result == BehavioralResult.NOT_APPLICABLE:
                        continue
                    ev = o.evidence.replace("\n", " ").replace("|", "\\|").strip()
                    if len(ev) > 180:
                        ev = ev[:177] + "..."
                    # escape pipes
                    a(f"| T{o.turn_index} | **{o.result.value}** | {o.confidence:.2f} | {ev} |")
                    shown += 1
                if shown == 0:
                    # NA only — show 1 example
                    for o in sorted(obs_by_dim[dim], key=lambda x: x.turn_index)[:1]:
                        ev = o.evidence.replace("\n", " ").replace("|", "\\|").strip()
                        a(f"| T{o.turn_index} | **{o.result.value}** | {o.confidence:.2f} | {ev} |")
                # NA summary when present
                na_count = sum(1 for o in obs_by_dim[dim] if o.result == BehavioralResult.NOT_APPLICABLE)
                if na_count:
                    a(f"| — | *{na_count}× NOT_APPLICABLE* | — | _Turns without enough opportunity (ignored in %)_ |")
                a("")

        # ── BFI-44 — right after each agent's last table
        if bfi_results:
            # Lookup by agent_id or role
            bfi = bfi_results.get(agent_id)
            if not bfi:
                # try by role
                bfi = bfi_results.get(role)
            if bfi:
                # bfi may be a BFIResult or a dict
                scores = getattr(bfi, "scores", None)
                if scores is None and isinstance(bfi, dict):
                    scores = bfi.get("scores") or bfi.get("Scores")
                raw = getattr(bfi, "raw_answers", None)
                if raw is None and isinstance(bfi, dict):
                    raw = bfi.get("raw_answers") or bfi.get("rawAnswers")
                if scores:
                    a(f"#### BFI-44 — {agent_id} ({role})")
                    a("")
                    a("_Big Five Inventory questionnaire applied right after persona conditioning, before the negotiation. Scale 1=Strongly disagree — 5=Strongly agree. Scores are per-dimension means (reversed items: 6 - answer)._")
                    a("")
                    a("| BFI Dimension | Score (1-5) | Interpretation | Items |")
                    a("|--------------|-------------|---------------|-------|")
                    bfi_labels = {
                        "extraversion": "Extraversion",
                        "agreeableness": "Agreeableness",
                        "conscientiousness": "Conscientiousness",
                        "neuroticism": "Neuroticism",
                        "openness": "Openness",
                    }
                    for dim_key in ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]:
                        sc = scores.get(dim_key)
                        if sc is None:
                            sc_str = "—"
                            interp = "—"
                        else:
                            sc_str = f"{sc:.2f}"
                            if sc >= 3.5:
                                interp = "High"
                            elif sc <= 2.5:
                                interp = "Low"
                            else:
                                interp = "Medium"
                        # Items per dimension for reference
                        from ..scoring.bfi import BFI_SCALES
                        scale = BFI_SCALES.get(dim_key, {})
                        itens = f"F:{','.join(str(x) for x in scale.get('forward', []))} R:{','.join(str(x) for x in scale.get('reverse', []))}"
                        a(f"| {bfi_labels.get(dim_key, dim_key)} | {sc_str} | {interp} | {itens} |")
                    a("")
                    if raw and isinstance(raw, dict):
                        # Show summarized raw answers (1-44)
                        # Format as a compact line
                        raw_str = ", ".join(f"{k}:{v}" for k, v in sorted(raw.items())[:10])
                        if len(raw) > 10:
                            raw_str += f" ... (+{len(raw)-10} items)"
                        a(f"_Raw answers (sample): {raw_str}_")
                        a("")
                    a(f"_BFI applied via `{getattr(bfi, 'model_identifier', agent_id)}` right after `Big5Persona` and `SituationalContext`, before `Turn 0`._")
                    a("")

    # ── 4. AGENT COMPARISON ──────────────────────────────────
    e(["## 4. Agent Comparison (Behavioral)", ""])
    a("_Occurrence percentage (PRESENT/(PRESENT+ABSENT)) — NOT_APPLICABLE ignored._")
    a("")
    agent_ids = list(profiles.keys())
    a("| Dimension |" + "".join(f" {aid} |" for aid in agent_ids))
    a("|-----------|" + "".join("-----------|" for _ in agent_ids))
    for dim in sorted(list(evaluated_dims), key=_dim_sort_key):
        meta = ALL_METRICS_META[dim]
        row = f"| {meta.name} |"
        for aid in agent_ids:
            summary = profiles[aid].summaries.get(dim)
            if summary is None:
                # legacy fallback
                occ = profiles[aid].scores.get(dim)
                row += f" {round(occ*100)}% |" if occ is not None else " — |"
            else:
                row += f" {_format_occurrence_compact(summary)} |"
        a(row)
    a("")

    # ── 5. UTILITY ────────────────────────────────────────
    e(["## 5. Utility (Continuous 0–1)", ""])
    if utility_results:
        e(render_utility_section(utility_results))
    else:
        a("_No utility calculation configured for this run._")
        a("")

    # ── 6. SATISFACTION ───────────────────────────
    e(["## 6. Subjective / Perceptual Evaluation (Ordinal 1–7)", ""])
    if satisfaction_results:
        e(render_satisfaction_section(satisfaction_results))
    else:
        a("_No subjective evaluation collected (PSI 1–7). Subjectivity stays separate from behavioral metrics._")
        a("")

    # ── 7. TRANSCRIPT ───────────────────────────────────────
    e(["## 7. Full Negotiation Transcript", ""])

    # Lookup turn → that turn's observations
    score_lookup: dict[tuple, dict] = {}
    for profile in profiles.values():
        source = profile.observations
        for o in source:
            key = (profile.agent_id, o.turn_index)
            score_lookup.setdefault(key, {})[o.dimension] = o

    for turn in result.transcript:
        a("---")
        a(f"**Turn {turn.turn_index} · {turn.role.upper()}**  `({turn.agent_id})`")
        a("")
        # speech quote
        a(f"> {turn.content}")
        a("")
        dim_obs = score_lookup.get((turn.agent_id, turn.turn_index), {})
        # split PRESENT / ABSENT / NA
        present = {d: o for d, o in dim_obs.items() if o.result == BehavioralResult.PRESENT}
        absent = {d: o for d, o in dim_obs.items() if o.result == BehavioralResult.ABSENT}
        na = {d: o for d, o in dim_obs.items() if o.result == BehavioralResult.NOT_APPLICABLE}
        if present or absent:
            a("**This turn's observations** _(PRESENT/ABSENT; NOT_APPLICABLE omitted)_:")
            a("")
            for d, o in sorted(present.items(), key=lambda x: ALL_METRICS_META[x[0]].abbreviation):
                meta = ALL_METRICS_META[d]
                ev = o.evidence.replace("\n", " ").strip()
                a(f"- `{meta.abbreviation}` **{meta.name}** — **PRESENT** (conf. {o.confidence:.2f}) — _{ev}_")
            for d, o in sorted(absent.items(), key=lambda x: ALL_METRICS_META[x[0]].abbreviation):
                meta = ALL_METRICS_META[d]
                ev = o.evidence.replace("\n", " ").strip()
                a(f"- `{meta.abbreviation}` **{meta.name}** — **ABSENT** (conf. {o.confidence:.2f}) — _{ev}_")
            if na:
                a(f"- _+ {len(na)}× NOT_APPLICABLE (no opportunity on this turn, ignored in %)_")
            a("")
        elif na:
            a(f"_All {len(na)} metrics NOT_APPLICABLE on this turn (no opportunity)_")
            a("")

    # ── 8. NOTES ───────────────────────────────────────
    e(["## 8. Methodology Notes", ""])
    a("- **Behavioral Metrics:** categorical LLM-as-judge. Per turn: `PRESENT` (behavior present), `ABSENT` (opportunity existed but absent), `NOT_APPLICABLE` (no opportunity; ignored). Each label includes short textual `evidence` based only on that turn's observable behavior.")
    a("- **Aggregation:** `occurrence_rate = PRESENT / (PRESENT + ABSENT)` — percentage over applicable turns. `NOT_APPLICABLE` does not enter the denominator. E.g., `65% (13/20; 5 NA)` = 13 PRESENT in 20 applicable, 5 NA ignored.")
    a("- **Big Five polarity:** induced `positive` → expected `PRESENT` (high pole); `negative` → expected `ABSENT` (high pole absent = low pole). Alignment: `PRESENT≥50%` compatible with `positive`, `ABSENT≥50%` compatible with `negative`. For tactics `enabled→PRESENT`, `disabled→ABSENT`.")
    a("- **Negotiation Outcomes:** categorical `agreement` `AGREEMENT | NO_AGREEMENT` (requires confirmation from both roles). Continuous `final_price` is not binarized.")
    a("- **Utility:** continuous 0–1 per role (`(p - p_floor)/(p_target - p_floor)`). May be <0 or >1. `joint_utility` is the Nash product (Luce & Raiffa Eq.4).")
    a("- **Subjective/Perceptual:** ordinal `1–7` scale (PSI: outcome, self, process, relationship) — separate from behavioral metrics.")
    a("- **Judge independence:** judge separated from negotiating agents to avoid self-evaluation bias.")
    a("- **IRR:** when `second_judge` is used, `confidence` = per-turn categorical agreement rate (1.0 agree, 0.0 disagree, 0.5 when one is NA).")
    a("- **Reproducibility:** use `temperature=0` and `seed` for deterministic results.")
    a("")

    # ── 9. RAW DATA ───────────────────────────────────────
    e(["## 9. Raw Data", ""])
    a("This run's detailed data was saved to:")
    if display_name:
        safe_display = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in str(display_name)).strip()
        exp_prefix = f"{safe_display}_{experiment_name}_{result.scenario_name}" if experiment_name else f"{safe_display}_{result.scenario_name}"
    else:
        exp_prefix = f"{experiment_name}_{result.scenario_name}" if experiment_name else result.scenario_name
    a(f"- Transcript: `results/transcripts/{exp_prefix}_{result.run_id}.jsonl`")
    a(f"- Scores: `results/scores/{exp_prefix}_{result.run_id}_scores.jsonl` (categorical summaries + observations)")
    if display_name and display_name != experiment_name:
        a(f"- Experiment (display): `{display_name}`")
    if experiment_name:
        a(f"- Experiment (file): `{experiment_name}`")
    a("")

    report_md = "\n".join(lines)
    if output_path:
        Path(output_path).write_text(report_md, encoding="utf-8")

    return report_md
