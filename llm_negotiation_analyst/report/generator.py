"""
Gerador de Relatórios: Produz um relatório Markdown estruturado de uma simulação.

Seções:
  1. Configuração do Experimento
  2. Resultado da Negociação
  3. Perfis Comportamentais Observados
  4. Comparação Entre Agentes
  5. Utilidade Econômica          ← novo
  6. Satisfação Pós-negociação    ← novo
  7. Transcrição Completa
  8. Notas de Metodologia
  9. Dados Brutos
"""

from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from ..simulation.engine import NegotiationResult
from ..scoring import Big5Profile, ALL_METRICS_META, resolve_metric
from ..scoring.report_sections import render_utility_section, render_satisfaction_section


# ─────────────────────────────────────────────────────────────────────────────
# Detecção retroativa de acordo
# ─────────────────────────────────────────────────────────────────────────────

def _detect_settlement_retroactively(result: NegotiationResult) -> bool:
    """
    Verifica os últimos turnos em busca de linguagem de acordo,
    para corrigir casos onde o engine não detectou a keyword.
    """
    indicators = [
        "confirmo os termos",
        "iniciar a implementação",
        "parceria firmada",
        "muito feliz que conseguimos",
        "SIMULACAO_CONCLUIDA",
    ]
    last_turns = result.transcript[-4:] if len(result.transcript) >= 4 else result.transcript
    for turn in last_turns:
        content_lower = turn.content.lower()
        if any(ind.lower() in content_lower for ind in indicators):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Gerador principal
# ─────────────────────────────────────────────────────────────────────────────

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

    settled       = result.settled or _detect_settlement_retroactively(result)
    personas_meta = result.metadata.get("personas", {})
    context_meta  = result.metadata.get("context")

    # ── CABEÇALHO ─────────────────────────────────────────────────────
    a("# Relatório de Análise de Negociação")
    a("")
    a(f"> **Cenário:** {result.scenario_description}  ")
    a(f"> **ID da Execução:** `{result.run_id}`  ")
    a(f"> **Gerado em:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ")
    a(f"> **Duração:** {result.duration_seconds:.1f}s | **Turnos:** {result.total_turns}")
    a(f"> **Acordo Fechado:** {'✅ Sim' if settled else '❌ Não'}")
    a("")

    # ── 1. CONFIGURAÇÃO DO EXPERIMENTO ────────────────────────────────
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
                    meta   = ALL_METRICS_META[metric]
                    a(f"| {meta.name} | **{score}**/5 |")
                except ValueError:
                    pass
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

    # ── 2. RESULTADO DA NEGOCIAÇÃO ────────────────────────────────────
    e(["## 2. Resultado da Negociação", ""])
    a(f"- **Acordo alcançado:** {'Sim ✅' if settled else 'Não ❌'}")
    a(f"- **Total de turnos:** {result.total_turns}")
    a(f"- **Duração total:** {result.duration_seconds:.1f}s")
    latencias = [t.latency_ms for t in result.transcript if t.latency_ms]
    if latencias:
        a(f"- **Latência média por turno:** {sum(latencias)/len(latencias):.0f}ms")
    a("")

    # ── 3. PERFIS COMPORTAMENTAIS OBSERVADOS ──────────────────────────
    e(["## 3. Perfis Comportamentais Observados", ""])
    a("_Comparação entre comportamento induzido e comportamento real medido pelo Juiz._")
    a("")

    evaluated_dims = set()
    for profile in profiles.values():
        evaluated_dims.update(profile.scores.keys())

    for agent_id, profile in profiles.items():
        role    = result.agent_roles.get(agent_id, "")
        induced = personas_meta.get(role, {})

        a(f"### Agente: {agent_id}")
        a("")

        # Tabela de alinhamento
        a("| Dimensão Avaliada | Induzido (Alvo) | Observado (Real) | Alinhamento |")
        a("|-------------------|-----------------|------------------|-------------|")

        dims_to_show = sorted(
            list(set(list(evaluated_dims) + [resolve_metric(d) for d in induced.keys()]))
        )

        for dim in dims_to_show:
            meta    = ALL_METRICS_META[dim]
            ind_val = induced.get(dim.value)
            obs_val = profile.scores.get(dim)
            ind_str = f"{ind_val}" if ind_val else "—"
            obs_str = f"**{obs_val:.2f}**" if obs_val else "—"
            status  = "—"
            if ind_val and obs_val:
                diff = abs(obs_val - ind_val)
                if diff <= 0.6:   status = "✅ Ótimo"
                elif diff <= 1.5: status = "⚠️ Desvio"
                else:             status = "❌ Falhou"
            a(f"| {meta.name} | {ind_str} | {obs_str} | {status} |")
        a("")

        # Observações comportamentais
        a("#### Observações Comportamentais")
        a("")
        scored = [d for d in evaluated_dims if profile.scores.get(d) is not None]
        if not scored:
            a("_Nenhuma dimensão pontuada com sucesso._")
            a("")
        else:
            for dim in sorted(scored, key=lambda d: d.value):
                meta  = ALL_METRICS_META[dim]
                score = profile.scores[dim]
                pole  = meta.high_pole if score >= 3 else meta.low_pole
                warn  = " _(⚠ baixa observabilidade)_" if meta.observability <= 2 else ""
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
                    a("> _Sem justificativas válidas registradas._")
                a("")

    # ── 4. COMPARAÇÃO ENTRE AGENTES ───────────────────────────────────
    e(["## 4. Comparação Entre Agentes", ""])
    a("_Scores observados lado a lado para identificar assimetrias._")
    a("")
    agent_ids = list(profiles.keys())
    a("| Dimensão |" + "".join(f" {aid} |" for aid in agent_ids))
    a("|-----------|" + "".join("-----------|" for _ in agent_ids))
    for dim in sorted(list(evaluated_dims), key=lambda d: d.value):
        meta = ALL_METRICS_META[dim]
        row  = f"| {meta.name} |"
        for aid in agent_ids:
            score = profiles[aid].scores.get(dim)
            row  += f" {f'`{score:.2f}`' if score is not None else '—'} |"
        a(row)
    a("")

    # ── 5. UTILIDADE ECONÔMICA ────────────────────────────────────────
    if utility_results:
        e(render_utility_section(utility_results))

    # ── 6. SATISFAÇÃO PÓS-NEGOCIAÇÃO (IPC) ───────────────────────────
    if satisfaction_results:
        e(render_satisfaction_section(satisfaction_results))

    # ── 7. TRANSCRIÇÃO COMPLETA ───────────────────────────────────────
    e(["## 7. Transcrição Completa da Negociação", ""])

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
            parts = " | ".join(
                f"{ALL_METRICS_META[d].abbreviation}: {s.score:.1f}"
                for d, s in valid.items()
            )
            a(f"*(Métricas detectadas: {parts})*")
            a("")

    # ── 8. NOTAS DE METODOLOGIA ───────────────────────────────────────
    e(["## 8. Notas de Metodologia", ""])
    a("- **Scoring:** LLM-as-judge com rubricas JSON estruturadas, uma chamada por dimensão por turno.")
    a("- **Aggregation:** Média aritmética dos scores válidos por turno (confidence > 0).")
    a("- **Persona injection:** Instruções comportamentais prepended ao system prompt. Modelos podem desviar da disposição induzida.")
    a("- **Situational context:** Injetado igualmente em todos os agentes. Efeito não medido diretamente.")
    a("- **Judge independence:** Juiz separado dos agentes negociadores para evitar viés de auto-avaliação.")
    a("- **Reproducibility:** Use `temperature=0` e `seed` (Ollama/LMStudio) para resultados determinísticos.")
    a("- **IRR:** Quando `second_judge` é usado, `confidence` reflete IRR normalizado por turno. Valores abaixo de 0.75 indicam desacordo significativo.")
    a("- **Alinhamento:** ✅ Ótimo (desvio ≤ 0.6), ⚠️ Desvio (≤ 1.5), ❌ Falhou (> 1.5).")
    a("- **Utilidade:** Escala 0–1 onde 0 = obteve o mínimo aceitável e 1 = obteve o valor alvo. Pode ser negativa (abaixo do piso) ou > 1 (superou o alvo).")
    a("- **IPC:** Índice de Satisfação Pós-negociação. Escala 1–7. Itens a3 e a5 invertidos (7 − valor) por formulação negativa. Referência: Barry & Friedman (1998).")
    a("")

    # ── 9. DADOS BRUTOS ───────────────────────────────────────────────
    e(["## 9. Dados Brutos", ""])
    a("Os dados detalhados desta execução foram salvos em:")
    a(f"- Transcrição: `results/transcripts/{result.scenario_name}_{result.run_id}.jsonl`")
    a(f"- Scores: `results/scores/{result.scenario_name}_{result.run_id}_scores.jsonl`")
    a("")

    report_md = "\n".join(lines)
    if output_path:
        Path(output_path).write_text(report_md, encoding="utf-8")

    return report_md
