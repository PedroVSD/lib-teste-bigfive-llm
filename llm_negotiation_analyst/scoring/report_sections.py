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
    # Joint Utility (Nash, Luce & Raiffa Eq.4) — dentro do bloco Utility, logo abaixo de candidate/recruiter
    try:
        if len(utility_results) >= 2 and any_settled:
            # p_s = seller reservation (min), p_b = buyer reservation (max)
            p_s = None
            p_b = None
            p = None
            for role, res in utility_results.items():
                rt = getattr(getattr(res, "params", None), "role_type", "")
                pf = getattr(getattr(res, "params", None), "p_floor", None)
                if rt == "seller" and pf is not None:
                    p_s = pf
                elif rt == "buyer" and pf is not None:
                    p_b = pf
                if res.agreed_price is not None:
                    p = res.agreed_price
            if p_s is None or p_b is None:
                floors = [getattr(getattr(r, "params", None), "p_floor", None) for r in utility_results.values()]
                floors = [f for f in floors if f is not None]
                if len(floors) >= 2:
                    p_s, p_b = min(floors), max(floors)
            joint = None
            if p is not None and p_s is not None and p_b is not None and p_b != p_s:
                joint = (p - p_s) * (p_b - p) / ((p_b - p_s) ** 2)
            if joint is not None:
                a(f"**Joint Utility:** `{joint:.3f}`")
                a("")
    except Exception:
        pass
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
