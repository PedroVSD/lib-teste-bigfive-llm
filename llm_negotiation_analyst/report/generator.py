"""
Gerador de Relatórios: Produz relatório Markdown estruturado de simulação.

Seções:
  1. Configuração do Experimento
  2. Resultado da Negociação (Outcomes)
  3. Métricas Comportamentais (Behavioral Metrics — categorical)
  4. Comparação Entre Agentes (% occurrence_rate)
  5. Utilidade Econômica (Utility — contínua 0-1)
  6. Satisfação Pós-negociação (Satisfaction — ordinal 1-7)
  7. Transcrição Completa
  8. Notas de Metodologia
  9. Dados Brutos
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
    """Converte induzido (positive/negative/enabled/disabled) para PRESENT/ABSENT esperado, respeitando polaridade."""
    if val is None:
        return None
    from ..scoring.big5 import Dimension as _D
    is_big5 = isinstance(dim, _D)
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("none", "null", "nil", "disabled", "false", "off", "not_applicable"):
            # disabled/none = não induzido → sem expectativa; para comparabilidade tratamos como ABSENT esperado? Retorna None para não comparar
            if v in ("disabled",):
                return BehavioralResult.ABSENT
            return None
        if is_big5:
            if v == "positive":
                return BehavioralResult.PRESENT  # high pole esperada
            if v == "negative":
                return BehavioralResult.ABSENT   # low pole esperada → PRESENT ausente
            # legacy numérico como string
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
    """Formata occurrence_rate como 65% (13/20; 5 NA)."""
    if summary is None or summary.occurrence_rate is None:
        if summary and summary.total_applicable == 0:
            return f"— (0 aplicáveis; {summary.not_applicable} NA)"
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
    # expected PRESENT → alinhado se occurrence >= 50%; expected ABSENT → alinhado se <50%
    occurred = summary.occurrence_rate >= 0.5
    expected_present = expected == BehavioralResult.PRESENT
    if occurred == expected_present:
        return "✅ Compatível"
    else:
        return "❌ Não compatível"


def generate_report(
    result: NegotiationResult,
    profiles: dict[str, Big5Profile],
    output_path: str | Path | None = None,
    utility_results: Optional[dict] = None,
    satisfaction_results: Optional[dict] = None,
) -> str:
    lines: list[str] = []
    a = lines.append
    e = lines.extend

    settled = result.settled or _detect_settlement_retroactively(result)
    personas_meta = result.metadata.get("personas", {})
    context_meta = result.metadata.get("context")
    experiment_name = result.metadata.get("experiment_name")

    # Agreement categorical
    agreement_label = "AGREEMENT" if settled else "NO_AGREEMENT"

    # ── CABEÇALHO ─────────────────────────────────────────────────────
    a("# Relatório de Análise de Negociação")
    a("")
    if experiment_name:
        a(f"> **Experimento:** `{experiment_name}`  ")
    a(f"> **Cenário:** {result.scenario_description}  ")
    a(f"> **ID da Execução:** `{result.run_id}`  ")
    a(f"> **Gerado em:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ")
    a(f"> **Duração:** {result.duration_seconds:.1f}s | **Turnos:** {result.total_turns}")
    a(f"> **Acordo:** `{agreement_label}`")
    a("")

    # ── 1. CONFIGURAÇÃO ────────────────────────────────
    e(["## 1. Configuração do Experimento (Setup)", ""])

    e(["### 1.1 Cenário", ""])
    a(f"**Nome:** `{result.scenario_name}`  ")
    a(f"**Descrição:** {result.scenario_description}")
    a("")
    a("**Contexto compartilhado enviado a ambos os agentes:**")
    a("")
    a(f"> {result.scenario_context}")
    a("")

    e(["### 1.2 Agentes", ""])
    a("| Agente ID | Modelo LLM | Papel no Cenário |")
    a("|-----------|------------|------------------|")
    for agent_id, model_id in result.agents.items():
        role = result.agent_roles.get(agent_id, "—")
        a(f"| `{agent_id}` | `{model_id}` | {role} |")
    a("")

    e(["### 1.3 Personas Induzidas", ""])
    if personas_meta:
        a("_Instruções de personalidade injetadas no System Prompt (Comportamento Alvo)._")
        a("")
        has_any = False
        for role, scores in personas_meta.items():
            if not scores:
                continue
            has_any = True
            a(f"**Papel: {role}**")
            a("| Dimensão | Valor Induzido (Target) | Esperado (categórico) |")
            a("|-----------|-------------------------|----------------------|")
            for dim_key, score in scores.items():
                if isinstance(score, str) and score.lower() in ("none", "false", "off"):
                    continue
                if score is None:
                    continue
                # disabled tratado como ABSENT esperado mas ainda exibe
                if isinstance(score, str) and score.lower() == "disabled":
                    # exibe como DISABLED
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
            a("_Personas configuradas mas todas desativadas (none)._")
            a("")
    else:
        a("_Nenhuma persona induzida. Modelos agiram com comportamento padrão._")
        a("")

    e(["### 1.4 Contexto Situacional (Macro)", ""])
    if context_meta and context_meta.get("enabled", True):
        active = {k: v for k, v in context_meta.items() if k != "enabled" and v not in (None, [], {}, "")}
        if active:
            a("| Condição Externa | Valor Configurado |")
            a("|-------------------|-------------------|")
            for field, val in active.items():
                if isinstance(val, list):
                    val = ", ".join(str(v) for v in val)
                a(f"| {field.replace('_', ' ').capitalize()} | {val} |")
            a("")
        else:
            a("_Nenhuma condição macroeconômica especial ativada._")
            a("")
    else:
        a("_Contexto situacional desativado para esta execução._")
        a("")

    # ── 2. RESULTADO DA NEGOCIAÇÃO (Outcomes) ────────────────────────
    e(["## 2. Negotiation Outcomes", ""])
    a(f"- **Agreement:** `{agreement_label}`")
    a(f"- **Total de turnos:** {result.total_turns}")
    a(f"- **Duração total:** {result.duration_seconds:.1f}s")
    latencias = [t.latency_ms for t in result.transcript if t.latency_ms]
    if latencias:
        a(f"- **Latência média por turno:** {sum(latencias)/len(latencias):.0f}ms")
    a("")
    a("_Outcomes são categóricos (AGREEMENT/NO_AGREEMENT) ou contínuos (utility, preço). Comportamento é separado na seção 3._")
    a("")

    # ── 3. BEHAVIORAL METRICS ──────────────────────────
    e(["## 3. Behavioral Metrics (Observação Categórica)", ""])
    a("_Cada métrica por turno é classificada como `PRESENT` / `ABSENT` / `NOT_APPLICABLE` com evidência textual. `NOT_APPLICABLE` não entra no denominador._")
    a("")
    a("**Agregação:** `occurrence_rate = PRESENT / (PRESENT + ABSENT)` — percentual de ocorrência nos turnos aplicáveis.")
    a("")

    # coletar dimensões avaliadas
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

        a(f"### Agente: {agent_id} ({role})")
        a("")

        a("| Dimensão Avaliada | Induzido | Esperado | Observado (occurrence_rate) | Alinhamento |")
        a("|-------------------|----------|----------|-------------------------------|-------------|")

        # dimensões a mostrar: todas avaliadas + todas induzidas
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
            # summary pode estar em summaries (novo) ou scores (legado)
            summary = profile.summaries.get(dim)
            if summary is None and dim in profile.scores:
                # legacy: scores[metric] is float occurrence_rate
                occ = profile.scores.get(dim)
                # tentar reconstruir counts: não disponível
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

        # Observações por dimensão — legível
        a("#### Evidências por dimensão (amostras por turno)")
        a("")
        a("_Cada linha abaixo é uma observação categórica de um turno com evidência textual curta. `occurrence_rate` já resumida na tabela acima._")
        a("")
        obs_by_dim = {}
        source = profile.observations
        for o in source:
            obs_by_dim.setdefault(o.dimension, []).append(o)
        if not obs_by_dim:
            a("_Nenhuma observação registrada._")
            a("")
        else:
            for dim in sorted(obs_by_dim.keys(), key=_dim_sort_key):
                meta = ALL_METRICS_META[dim]
                summary = profile.summaries.get(dim)
                occ = _format_occurrence(summary) if summary else "—"
                warn = " _(⚠ baixa observabilidade)_" if meta.observability <= 2 else ""
                a(f"**{meta.name}** (`{meta.abbreviation}`) — {occ} — *{meta.high_pole} ↔ {meta.low_pole}*{warn}")
                a("")
                # Tabela por dimensão — mostra todas as observações aplicáveis, NOT_APPLICABLE colapsado
                a("| Turno | Resultado | Conf. | Evidence |")
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
                    # só NA — mostrar 1 exemplo
                    for o in sorted(obs_by_dim[dim], key=lambda x: x.turn_index)[:1]:
                        ev = o.evidence.replace("\n", " ").replace("|", "\\|").strip()
                        a(f"| T{o.turn_index} | **{o.result.value}** | {o.confidence:.2f} | {ev} |")
                # resumo NA se houver
                na_count = sum(1 for o in obs_by_dim[dim] if o.result == BehavioralResult.NOT_APPLICABLE)
                if na_count:
                    a(f"| — | *{na_count}× NOT_APPLICABLE* | — | _Turnos sem oportunidade suficiente (ignorados no %)_ |")
                a("")

    # ── 4. COMPARAÇÃO ENTRE AGENTES ──────────────────────────────────
    e(["## 4. Comparação Entre Agentes (Behavioral)", ""])
    a("_Percentual de ocorrência (PRESENT/(PRESENT+ABSENT)) — NOT_APPLICABLE ignorado._")
    a("")
    agent_ids = list(profiles.keys())
    a("| Dimensão |" + "".join(f" {aid} |" for aid in agent_ids))
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
    e(["## 5. Utility (Contínua 0–1)", ""])
    if utility_results:
        e(render_utility_section(utility_results))
    else:
        a("_Nenhum cálculo de utilidade configurado para esta execução._")
        a("")

    # ── 6. SATISFAÇÃO ───────────────────────────
    e(["## 6. Subjective / Perceptual Evaluation (Ordinal 1–7)", ""])
    if satisfaction_results:
        e(render_satisfaction_section(satisfaction_results))
    else:
        a("_Nenhuma avaliação subjetiva coletada (IPC 1–7). Subjetividade permanece separada das métricas comportamentais._")
        a("")

    # ── 7. TRANSCRIÇÃO ───────────────────────────────────────
    e(["## 7. Transcrição Completa da Negociação", ""])

    # Lookup turno → observações daquele turno
    score_lookup: dict[tuple, dict] = {}
    for profile in profiles.values():
        source = profile.observations
        for o in source:
            key = (profile.agent_id, o.turn_index)
            score_lookup.setdefault(key, {})[o.dimension] = o

    for turn in result.transcript:
        a("---")
        a(f"**Turno {turn.turn_index} · {turn.role.upper()}**  `({turn.agent_id})`")
        a("")
        # citação da fala
        a(f"> {turn.content}")
        a("")
        dim_obs = score_lookup.get((turn.agent_id, turn.turn_index), {})
        # separar PRESENT / ABSENT / NA
        present = {d: o for d, o in dim_obs.items() if o.result == BehavioralResult.PRESENT}
        absent = {d: o for d, o in dim_obs.items() if o.result == BehavioralResult.ABSENT}
        na = {d: o for d, o in dim_obs.items() if o.result == BehavioralResult.NOT_APPLICABLE}
        if present or absent:
            a("**Observações deste turno** _(PRESENT/ABSENT; NOT_APPLICABLE omitido)_:")
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
                a(f"- _+ {len(na)}× NOT_APPLICABLE (sem oportunidade neste turno, ignorado no %)_")
            a("")
        elif na:
            a(f"_Todas as {len(na)} métricas NOT_APPLICABLE neste turno (sem oportunidade)_")
            a("")

    # ── 8. NOTAS ───────────────────────────────────────
    e(["## 8. Notas de Metodologia", ""])
    a("- **Behavioral Metrics:** LLM-as-judge categórico. Por turno: `PRESENT` (comportamento presente), `ABSENT` (oportunidade havia mas ausente), `NOT_APPLICABLE` (sem oportunidade; ignorado). Cada rótulo inclui `evidence` textual curta baseada apenas no comportamento observável daquele turno.")
    a("- **Agregação:** `occurrence_rate = PRESENT / (PRESENT + ABSENT)` — percentual nos turnos aplicáveis. `NOT_APPLICABLE` não entra no denominador. Ex: `65% (13/20; 5 NA)` = 13 PRESENT em 20 aplicáveis, 5 NA ignorados.")
    a("- **Polaridade Big Five:** Induzido `positive` → esperado `PRESENT` (polo alto); `negative` → esperado `ABSENT` (polo alto ausente = polo baixo). Alinhamento: `PRESENT≥50%` compatível com `positive`, `ABSENT≥50%` compatível com `negative`. Para táticas `enabled→PRESENT`, `disabled→ABSENT`.")
    a("- **Negotiation Outcomes:** `agreement` categórico `AGREEMENT | NO_AGREEMENT` (exige confirmação de ambos os papéis). `final_price`, `surplus`, `reservation_distance` contínuos não são binarizados.")
    a("- **Utility:** Contínua 0–1 por papel (`(p - p_floor)/(p_target - p_floor)`). Pode ser <0 ou >1. `joint_utility` = soma.")
    a("- **Subjective/Perceptual:** Escala ordinal `1–7` (IPC: fairness, satisfaction, perception, relationship) — separada das métricas comportamentais.")
    a("- **Judge independence:** Juiz separado dos agentes negociadores para evitar viés de auto-avaliação.")
    a("- **IRR:** Quando `second_judge` é usado, `confidence` = taxa de acordo categórico por turno (1.0 acordo, 0.0 desacordo, 0.5 se um NA).")
    a("- **Reproducibility:** Use `temperature=0` e `seed` para resultados determinísticos.")
    a("")

    # ── 9. DADOS BRUTOS ───────────────────────────────────────
    e(["## 9. Dados Brutos", ""])
    a("Os dados detalhados desta execução foram salvos em:")
    exp_prefix = f"{experiment_name}_{result.scenario_name}" if experiment_name else result.scenario_name
    a(f"- Transcrição: `results/transcripts/{exp_prefix}_{result.run_id}.jsonl`")
    a(f"- Scores: `results/scores/{exp_prefix}_{result.run_id}_scores.jsonl` (summaries + observations categóricos)")
    if experiment_name:
        a(f"- Experimento: `{experiment_name}`")
    a("")

    report_md = "\n".join(lines)
    if output_path:
        Path(output_path).write_text(report_md, encoding="utf-8")

    return report_md
