"""
Seções de relatório para Utilidade e Satisfação.
Importar e chamar no generator.py.
"""

from .utility import UtilityResult
from .satisfaction import SatisfactionScores, CATEGORY_LABELS, IPC_QUESTIONS


def render_utility_section(utility_results: dict[str, UtilityResult]) -> list[str]:
    """
    Gera a seção Markdown de Utilidade Econômica para o relatório.
    Retorna lista de linhas para append no generator.
    """
    if not utility_results:
        return []

    lines = []
    a = lines.append

    a("## Utilidade Econômica")
    a("")
    a(
        "_A utilidade mede o quão bem cada agente se saiu em relação aos seus valores privados "
        "(alvo e piso/teto). Escala: 0.0 = obteve o mínimo aceitável; 1.0 = obteve o valor alvo; "
        "> 1.0 = superou o alvo; < 0.0 = ficou abaixo do piso._"
    )
    a("")

    # Verifica se algum resultado tem preço extraído
    any_settled = any(r.settled for r in utility_results.values())
    if not any_settled:
        a("_Não foi detectado acordo com preço definido. Utilidade não calculável._")
        a("")
        return lines

    # Tabela comparativa
    a("| Papel | Tipo | Preço Acordado | Preço Alvo | Piso / Teto | Utilidade | Interpretação |")
    a("|-------|------|----------------|------------|-------------|-----------|---------------|")

    for role, res in utility_results.items():
        price_str  = f"{res.params.currency}{res.agreed_price:,.2f}{res.params.unit}" if res.agreed_price else "—"
        target_str = f"{res.params.currency}{res.params.p_target:,.2f}{res.params.unit}"
        floor_str  = f"{res.params.currency}{res.params.p_floor:,.2f}{res.params.unit}"
        util_str   = f"**{res.utility:.3f}**" if res.utility is not None else "—"
        interp     = res.interpretation.split(".")[0]  # primeira frase apenas
        a(f"| {role} | {res.role_type} | {price_str} | {target_str} | {floor_str} | {util_str} | {interp} |")

    a("")

    # Detalhe por papel
    for role, res in utility_results.items():
        if res.utility is not None:
            a(f"**{role}** — {res.interpretation}")
            if res.note:
                a(f"> ⚠️ {res.note}")
    a("")

    return lines


def render_satisfaction_section(satisfaction_results: dict[str, SatisfactionScores]) -> list[str]:
    """
    Gera a seção Markdown de Satisfação (IPC) para o relatório.
    """
    if not satisfaction_results:
        return []

    lines = []
    a = lines.append

    a("## Satisfação Pós-negociação (IPC)")
    a("")
    a(
        "_Índice de Satisfação Pós-negociação baseado em Barry & Friedman (1998). "
        "16 questões em escala 1–7, organizadas em 4 sub-escalas. "
        "Itens 3 (a3) e 5 (a5) são invertidos pois indicam sentimentos negativos._"
    )
    a("")

    # ── Tabela de sub-escalas por agente ──
    agent_ids = list(satisfaction_results.keys())
    a("### Scores por Sub-escala")
    a("")
    header = "| Sub-escala |" + "".join(f" {aid} |" for aid in agent_ids)
    sep    = "|------------|" + "".join("------------|" for _ in agent_ids)
    a(header)
    a(sep)

    subscales = [
        ("Resultado (aOutcome)",        "a_outcome"),
        ("Si Mesmo (aSelf)",            "a_self"),
        ("Processo (aProcess)",         "a_process"),
        ("Relacionamento (aRelationship)", "a_relationship"),
        ("**Geral (média)**",           "overall"),
    ]

    for label, attr in subscales:
        row = f"| {label} |"
        for aid in agent_ids:
            scores = satisfaction_results[aid]
            val = getattr(scores, attr)
            row += f" `{val:.3f}` |" if val is not None else " — |"
        a(row)
    a("")

    # ── Respostas brutas por agente ──
    a("### Respostas Brutas (1–7 por questão)")
    a("")
    a(
        "_Os valores abaixo são as respostas originais do juiz antes da inversão dos itens 3 e 5. "
        "A inversão é aplicada automaticamente nas fórmulas acima._"
    )
    a("")

    # Monta tabela com todas as questões
    q_header = "| ID | Categoria | Questão (resumo) | Invertido |" + "".join(f" {aid} |" for aid in agent_ids)
    q_sep    = "|----|-----------|-----------------|-----------|" + "".join("-----------|" for _ in agent_ids)
    a(q_header)
    a(q_sep)

    for q in IPC_QUESTIONS:
        qid      = q["id"]
        cat      = CATEGORY_LABELS[q["category"]].split("(")[0].strip()
        inv_mark = "✅" if q["inverted"] else ""
        # Resumo da pergunta (primeiras 60 chars)
        resumo = q["text"][:60].rstrip() + ("…" if len(q["text"]) > 60 else "")
        row = f"| {qid} | {cat} | {resumo} | {inv_mark} |"
        for aid in agent_ids:
            answers = satisfaction_results[aid].raw_answers
            val = answers.get(qid, "—")
            row += f" {val} |"
        a(row)

    a("")
    a(
        "_Referência: Barry, B., & Friedman, R. A. (1998). Bargainer Characteristics "
        "in Distributive and Integrative Negotiation. "
        "Journal of Personality and Social Psychology, 74(2), 345–359._"
    )
    a("")

    return lines
