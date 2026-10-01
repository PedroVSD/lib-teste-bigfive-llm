"""
Report sections for Utility and Satisfaction.
Import and call from generator.py.
"""

from .utility import UtilityResult
from .satisfaction import SatisfactionScores, CATEGORY_LABELS, IPC_QUESTIONS


def render_utility_section(utility_results: dict[str, UtilityResult]) -> list[str]:
    """
    Generate the Economic Utility Markdown section for the report.
    Returns a list of lines to append in the generator.
    """
    if not utility_results:
        return []

    lines = []
    a = lines.append

    a("## Economic Utility")
    a("")
    a(
        "_Utility measures how well each agent did relative to their private "
        "values (target and floor/ceiling). Scale: 0.0 = obtained the minimum acceptable; 1.0 = obtained the target value; "
        "> 1.0 = beat the target; < 0.0 = fell below the floor._"
    )
    a("")

    # Check whether any result has an extracted price
    any_settled = any(r.settled for r in utility_results.values())
    if not any_settled:
        a("_No agreement with a defined price detected. Utility not calculable._")
        a("")
        return lines

    # Comparison table
    a("| Role | Type | Agreed Price | Target Price | Floor / Ceiling | Utility | Interpretation |")
    a("|-------|------|----------------|------------|-------------|-----------|---------------|")

    for role, res in utility_results.items():
        price_str  = f"{res.params.currency}{res.agreed_price:,.2f}{res.params.unit}" if res.agreed_price else "—"
        target_str = f"{res.params.currency}{res.params.p_target:,.2f}{res.params.unit}"
        floor_str  = f"{res.params.currency}{res.params.p_floor:,.2f}{res.params.unit}"
        util_str   = f"**{res.utility:.3f}**" if res.utility is not None else "—"
        interp     = res.interpretation.split(".")[0]  # first sentence only
        a(f"| {role} | {res.role_type} | {price_str} | {target_str} | {floor_str} | {util_str} | {interp} |")

    a("")

    # Per-role detail
    for role, res in utility_results.items():
        if res.utility is not None:
            a(f"**{role}** — {res.interpretation}")
            if res.note:
                a(f"> ⚠️ {res.note}")
    # Joint Utility (Nash, Luce & Raiffa Eq.4) — inside the Utility block, right below candidate/recruiter
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
    Generate the Satisfaction (PSI) Markdown section for the report.
    """
    if not satisfaction_results:
        return []

    lines = []
    a = lines.append

    a("## Post-negotiation Satisfaction (PSI)")
    a("")
    a(
        "_Post-negotiation Satisfaction Index based on Barry & Friedman (1998). "
        "16 questions on a 1–7 scale, organized into 4 subscales. "
        "Items 3 (a3) and 5 (a5) are reversed as they indicate negative feelings._"
    )
    a("")

    # ── Subscale table per agent ──
    agent_ids = list(satisfaction_results.keys())
    a("### Scores per Subscale")
    a("")
    header = "| Subscale |" + "".join(f" {aid} |" for aid in agent_ids)
    sep    = "|------------|" + "".join("------------|" for _ in agent_ids)
    a(header)
    a(sep)

    subscales = [
        ("Outcome (aOutcome)",        "a_outcome"),
        ("Self (aSelf)",            "a_self"),
        ("Process (aProcess)",         "a_process"),
        ("Relationship (aRelationship)", "a_relationship"),
        ("**Overall (mean)**",           "overall"),
    ]

    for label, attr in subscales:
        row = f"| {label} |"
        for aid in agent_ids:
            scores = satisfaction_results[aid]
            val = getattr(scores, attr)
            row += f" `{val:.3f}` |" if val is not None else " — |"
        a(row)
    a("")

    # ── Raw answers per agent ──
    a("### Raw Answers (1–7 per question)")
    a("")
    a(
        "_Values below are the judge's original answers before reversing items 3 and 5. "
        "Reversal is applied automatically in the formulas above._"
    )
    a("")

    # Build table with all questions
    q_header = "| ID | Category | Question (summary) | Reversed |" + "".join(f" {aid} |" for aid in agent_ids)
    q_sep    = "|----|-----------|-----------------|-----------|" + "".join("-----------|" for _ in agent_ids)
    a(q_header)
    a(q_sep)

    for q in IPC_QUESTIONS:
        qid      = q["id"]
        cat      = CATEGORY_LABELS[q["category"]].split("(")[0].strip()
        inv_mark = "✅" if q["inverted"] else ""
        # Question summary (first 60 chars)
        resumo = q["text"][:60].rstrip() + ("…" if len(q["text"]) > 60 else "")
        row = f"| {qid} | {cat} | {resumo} | {inv_mark} |"
        for aid in agent_ids:
            answers = satisfaction_results[aid].raw_answers
            val = answers.get(qid, "—")
            row += f" {val} |"
        a(row)

    a("")
    a(
        "_Reference: Barry, B., & Friedman, R. A. (1998). Bargainer Characteristics "
        "in Distributive and Integrative Negotiation. "
        "Journal of Personality and Social Psychology, 74(2), 345–359._"
    )
    a("")

    return lines
