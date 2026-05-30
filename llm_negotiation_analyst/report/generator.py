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

"""
Gerador de Relatórios: Produz um relatório Markdown estruturado de uma simulação.
"""

from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from ..simulation.engine import NegotiationResult
from ..scoring import Big5Profile, ALL_METRICS_META, resolve_metric


def _detect_settlement_retroactively(result: NegotiationResult) -> bool:
    """
    Verifica o transcript em busca de linguagem de acordo,
    para corrigir casos onde o engine não detectou a keyword.
    """
    indicators = [
        "confirmo os termos",
        "iniciar a implementação", "parceria firmada",
        "muito feliz que conseguimos",
        "SIMULACAO_CONCLUIDA",
    ]
    # Verifica os últimos 4 turnos — acordo geralmente aparece no final
    last_turns = result.transcript[-4:] if len(result.transcript) >= 4 else result.transcript
    for turn in last_turns:
        content_lower = turn.content.lower()
        if any(ind.lower() in content_lower for ind in indicators):
            return True
    return False


def generate_report(
    result: NegotiationResult,
    profiles: dict[str, Big5Profile],
    output_path: str | Path | None = None,
) -> str:
    lines: list[str] = []
    a = lines.append
    e = lines.extend
    settled = result.settled or _detect_settlement_retroactively(result)

    personas_meta: dict           = result.metadata.get("personas", {})
    context_meta:  Optional[dict] = result.metadata.get("context")

    # ── CABEÇALHO ─────────────────────────────────────────────────────
    a("# Relatório de Análise de Negociação")
    a("")
    a(f"> **Cenário:** {result.scenario_description}  ")
    a(f"> **ID da Execução:** `{result.run_id}`  ")
    a(f"> **Gerado em:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ")
    a(f"> **Duração:** {result.duration_seconds:.1f}s | "
      f"**Turnos:** {result.total_turns} | ")
    a(f"> **Acordo Fechado:** {'✅ Sim' if settled else '❌ Não'}")
    a("")
    # Detecção retroativa de acordo pelo transcript



    # ── 1. CONFIGURAÇÃO DO EXPERIMENTO (SETUP) ────────────────────────
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
        for role, scores in personas_meta.items():
            if not scores:
                continue
            a(f"**Papel: {role}**")
            a("| Dimensão | Valor Induzido (Target) |")
            a("|-----------|-------------------------|")
            for dim_key, score in scores.items():
                try:
                    metric = resolve_metric(dim_key)
                    meta = ALL_METRICS_META[metric]
                    a(f"| {meta.name} | **{score}**/5 |")
                except ValueError:
                    pass
            a("")
    else:
        a("_Nenhuma persona específica foi induzida. Os modelos agiram conforme seu comportamento padrão._")
        a("")

    e(["### 1.4 Contexto Situacional (Macro)", ""])
    if context_meta and context_meta.get("enabled", True):
        active = {k: v for k, v in context_meta.items() if k != "enabled" and v not in (None, [], {}, "")}
        if active:
            a("| Condição Externa | Valor Configurado |")
            a("|-------------------|-------------------|")
            for field, val in active.items():
                if isinstance(val, list): val = ", ".join(str(v) for v in val)
                a(f"| {field.replace('_', ' ').capitalize()} | {val} |")
            a("")
        else:
            a("_Nenhuma condição macroeconômica especial foi ativada._")
            a("")
    else:
        a("_Contexto situacional desativado para esta execução._")
        a("")

    # ── 2. RESULTADO DA NEGOCIAÇÃO ────────────────────────────────────
    e(["## 2. Resultado da Negociação", ""])
    a(f"- **Acordo alcançado:** {'Sim ✅' if result.settled else 'Não ❌'}")
    a(f"- **Total de turnos:** {result.total_turns}")
    a(f"- **Duração total:** {result.duration_seconds:.1f}s")

    latencias = [t.latency_ms for t in result.transcript if t.latency_ms]
    if latencias:
        a(f"- **Latência média por turno:** {sum(latencias)/len(latencias):.0f}ms")
    a("")

    # ── 3. PERFIS COMPORTAMENTAIS OBSERVADOS ──────────────────────────
    e(["## 3. Perfis Comportamentais Observados", ""])
    a("_Comparação entre o comportamento esperado (Induzido) e o comportamento real medido pelo Juiz._")
    a("")

    evaluated_dims = set()
    for profile in profiles.values():
        evaluated_dims.update(profile.scores.keys())

    for agent_id, profile in profiles.items():
        role = result.agent_roles.get(agent_id, "")
        induced = personas_meta.get(role, {})

        a(f"### Agente: {agent_id}")
        a("")

        # --- PARTE 1: A TABELA DE ALINHAMENTO ---
        a("| Dimensão Avaliada | Induzido (Alvo) | Observado (Real) | Alinhamento |")
        a("|-------------------|-----------------|------------------|-------------|")

        dims_to_show = sorted(list(set(list(evaluated_dims) + [resolve_metric(d) for d in induced.keys()])))

        for dim in dims_to_show:
            meta = ALL_METRICS_META[dim]
            ind_val = induced.get(dim.value)
            obs_val = profile.scores.get(dim)

            ind_str = f"{ind_val}" if ind_val else "—"
            obs_str = f"**{obs_val:.2f}**" if obs_val else "—"

            status = "—"
            if ind_val and obs_val:
                diff = abs(obs_val - ind_val)
                if diff <= 0.6: status = "✅ Ótimo"
                elif diff <= 1.5: status = "⚠️ Desvio"
                else: status = "❌ Falhou"

            a(f"| {meta.name} | {ind_str} | {obs_str} | {status} |")
        a("")

        # --- PARTE 2: OBSERVAÇÕES COMPORTAMENTAIS (JUSTIFICATIVAS) ---
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

    # ── 4. COMPARAÇÃO LADO A LADO ─────────────────────────────────────
    e(["## 4. Comparação Entre Agentes", ""])
    a("_Visão comparativa dos scores observados para identificar assimetrias._")
    a("")

    agent_ids = list(profiles.keys())
    # Monta o cabeçalho da tabela com os nomes dos agentes
    header = "| Dimensão |" + "".join(f" {aid} |" for aid in agent_ids)
    separator = "|-----------|" + "".join("-----------|" for _ in agent_ids)
    a(header)
    a(separator)

    for dim in sorted(list(evaluated_dims), key=lambda d: d.value):
        meta = ALL_METRICS_META[dim]
        row  = f"| {meta.name} |"
        for aid in agent_ids:
            score = profiles[aid].scores.get(dim)
            row += f" {f'`{score:.2f}`' if score is not None else '—'} |"
        a(row)
    a("")

    # ── 5. TRANSCRIÇÃO ANOTADA ────────────────────────────────────────
    e(["## 5. Transcrição Completa da Negociação", ""])

    score_lookup: dict[tuple, dict] = {}
    for profile in profiles.values():
        for s in profile.per_turn_scores:
            key = (profile.agent_id, s.turn_index)
            score_lookup.setdefault(key, {})[s.dimension] = s

    for turn in result.transcript:
        a("---")
        a(f"**Turno {turn.turn_index} · {turn.role.upper()}**")
        a("")
        a(turn.content)
        a("")
        dim_scores = score_lookup.get((turn.agent_id, turn.turn_index), {})
        valid = {d: s for d, s in dim_scores.items() if s.confidence > 0}
        if valid:
            parts = " | ".join(f"{ALL_METRICS_META[d].abbreviation}: {s.score:.1f}" for d, s in valid.items())
            a(f"*(Métricas detectadas: {parts})*")
            a("")

    # ── 6. NOTAS DE METODOLOGIA ───────────────────────────────────────
    e(["## 6. Notas de Metodologia", ""])
    a("- **Scoring:** LLM-as-judge with structured JSON rubrics, one call per dimension per turn.")
    a("- **Aggregation:** Arithmetic mean of valid per-turn scores (confidence > 0).")
    a("- **Observability:** Some psychological metrics are less reliable in text. Treat with caution.")
    a("- **Persona injection:** Behavioral instructions prepended to system prompt. Models may deviate from induced disposition.")
    a("- **Situational context:** Injected equally into all agents. Effect is not directly measured.")
    a("- **Judge independence:** Judge should differ from negotiating agents to avoid self-evaluation bias.")
    a("- **Reproducibility:** Use `temperature=0` and `seed` (Ollama/LMStudio) for deterministic results.")
    a("- **IRR:** When `second_judge` is used, `confidence` reflects normalized inter-rater reliability per turn. Values below 0.75 indicate significant disagreement.")
    a("- **Indução vs Observação:** O valor 'Induzido' é o alvo definido no YAML; o 'Observado' é a performance real detectada pelo juiz.")
    a("- **Alinhamento:** ✅ Ótimo (desvio < 0.6), ⚠️ Desvio (até 1.5), ❌ Falhou (desvio > 1.5).")
    a("")

    # ── 7. EXPORTAÇÃO ─────────────────────────────────────────────────
    e(["## 7. Dados Brutos", ""])
    a(f"Os dados detalhados desta execução foram salvos em:")
    a(f"- Transcrição: `results/transcripts/{result.scenario_name}_{result.run_id}.jsonl`")
    a(f"- Scores: `results/scores/{result.scenario_name}_{result.run_id}_scores.jsonl`")

    report_md = "\n".join(lines)
    if output_path:
        Path(output_path).write_text(report_md, encoding="utf-8")

    return report_md
