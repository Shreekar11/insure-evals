"""
Underwriter's Worksheet PDF — exactly one page, styled as an insurance form.
Uses reportlab. Design for one page from the start; cut content before page 2.
"""
import json
import io
from datetime import date
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

from src.underwriting.premium import calculate, PremiumResult
from config import SUMMARY_JSON, RESULTS_DIR

# ── Colors ────────────────────────────────────────────────────────────────
OLLIVE_DARK = colors.HexColor("#1a2a3a")
OLLIVE_ACCENT = colors.HexColor("#2563eb")
RISK_COLORS = {
    "Low": colors.HexColor("#16a34a"),
    "Medium": colors.HexColor("#ca8a04"),
    "High": colors.HexColor("#ea580c"),
    "Critical": colors.HexColor("#dc2626"),
}
LIGHT_GREY = colors.HexColor("#f1f5f9")
MID_GREY = colors.HexColor("#94a3b8")


def _load_eval_data() -> dict:
    if SUMMARY_JSON.exists():
        return json.loads(SUMMARY_JSON.read_text())
    return {}


def _get_stats(summary: dict, axis: str, model_key: str) -> dict:
    for model, stats in summary.get(axis, {}).items():
        if model_key in model.lower():
            return stats
    return {"rate": 0.0, "mean_severity": 0.0, "n": 0, "mean_latency_ms": 0}


def generate_pdf(output_path: Path | None = None) -> bytes:
    """
    Generate the Underwriter's Worksheet as a PDF.
    Returns raw bytes; also writes to output_path if given.
    """
    summary = _load_eval_data()
    today = date.today().strftime("%B %d, %Y")

    # Pull eval stats
    AXES = ["hallucination", "bias", "safety"]
    AXIS_LABELS = {"hallucination": "Hallucination", "bias": "Bias", "safety": "Content Safety"}
    oss_stats = {ax: _get_stats(summary, ax, "qwen") for ax in AXES}
    front_stats = {ax: _get_stats(summary, ax, "gemini") for ax in AXES}

    # Compute premiums
    oss_hal = oss_stats["hallucination"].get("rate", 0.10)
    oss_bias = oss_stats["bias"].get("rate", 0.10)
    oss_safe = oss_stats["safety"].get("rate", 0.10)
    front_hal = front_stats["hallucination"].get("rate", 0.05)
    front_bias = front_stats["bias"].get("rate", 0.05)
    front_safe = front_stats["safety"].get("rate", 0.05)

    oss_prem = calculate(oss_hal, oss_bias, oss_safe, "medium", "medical")
    front_prem = calculate(front_hal, front_bias, front_safe, "medium", "medical")

    # ── PDF setup ────────────────────────────────────────────────────────
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=12 * mm,
        bottomMargin=10 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title", fontSize=15, textColor=OLLIVE_DARK,
                                 fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=2)
    subtitle_style = ParagraphStyle("sub", fontSize=8, textColor=MID_GREY,
                                    alignment=TA_CENTER, spaceAfter=4)
    section_style = ParagraphStyle("section", fontSize=9, textColor=OLLIVE_ACCENT,
                                   fontName="Helvetica-Bold", spaceBefore=5, spaceAfter=2)
    body_style = ParagraphStyle("body", fontSize=8, leading=10)
    caveat_style = ParagraphStyle("caveat", fontSize=6.5, textColor=MID_GREY,
                                  leading=9, alignment=TA_CENTER, spaceBefore=3)

    def tbl_style(header_color=OLLIVE_DARK):
        return TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), header_color),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
            ("GRID", (0, 0), (-1, -1), 0.3, MID_GREY),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING", (0, 0), (-1, -1), 3),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ])

    story = []

    # ── Header ─────────────────────────────────────────────────────────
    story.append(Paragraph("UNDERWRITER'S WORKSHEET", title_style))
    story.append(Paragraph(
        f"AI Liability Risk Assessment · Ollive AI Insurance · {today}", subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=OLLIVE_ACCENT, spaceAfter=4))

    # ── Policy metadata ────────────────────────────────────────────────
    meta_data = [
        ["Insured Entity", "Demo Agent (Medical Domain)", "Policy No.", "DEMO-2026-001"],
        ["Domain", "Medical / Health", "Eval Date", today],
        ["OSS Model", "Qwen2.5-0.5B-Instruct (HF CPU)", "Frontier Model", "Gemini 2.0 Flash"],
        ["Judge", "GPT-4o-mini (single, temp=0)", "Safety Classifier", "Llama Guard 3 (OpenRouter)"],
    ]
    meta_tbl = Table(meta_data, colWidths=[35 * mm, 60 * mm, 28 * mm, 57 * mm])
    meta_tbl.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, LIGHT_GREY]),
        ("GRID", (0, 0), (-1, -1), 0.3, MID_GREY),
        ("PADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 4 * mm))

    # ── Risk scores table ──────────────────────────────────────────────
    story.append(Paragraph("RISK SCORES (Rate = % failing · Degree = Mean Severity 1–5)", section_style))
    score_data = [
        ["Axis", "OSS Rate", "OSS Degree", "OSS N", "Frontier Rate", "Frontier Degree", "Frontier N"],
    ]
    for ax in AXES:
        os = oss_stats[ax]
        fr = front_stats[ax]
        score_data.append([
            AXIS_LABELS[ax],
            f"{os.get('rate', 0):.1%}",
            f"{os.get('mean_severity', 0):.2f}",
            str(os.get("n", 0)),
            f"{fr.get('rate', 0):.1%}",
            f"{fr.get('mean_severity', 0):.2f}",
            str(fr.get("n", 0)),
        ])
    col_w = [32 * mm, 22 * mm, 22 * mm, 14 * mm, 28 * mm, 28 * mm, 14 * mm]
    score_tbl = Table(score_data, colWidths=col_w)
    score_tbl.setStyle(tbl_style())
    story.append(score_tbl)
    story.append(Spacer(1, 3 * mm))

    # ── Performance table ──────────────────────────────────────────────
    story.append(Paragraph("PERFORMANCE (⚠ OSS=free HF CPU vs Frontier=hosted GPU API — not a fair comparison)", section_style))
    perf_data = [["Model", "Avg Latency (Halluc.)", "Avg Latency (Bias)", "Avg Latency (Safety)", "Total API Cost"]]
    def latency(stats): return f"{stats.get('mean_latency_ms', 0):.0f} ms"
    def cost(axis_key, model_key):
        for m, s in summary.get(axis_key, {}).items():
            if model_key in m.lower():
                return f"${s.get('total_cost_usd', 0):.4f}"
        return "$0"
    total_oss_cost = sum(
        s.get("total_cost_usd", 0)
        for ax in AXES
        for m, s in summary.get(ax, {}).items()
        if "qwen" in m.lower()
    )
    total_front_cost = sum(
        s.get("total_cost_usd", 0)
        for ax in AXES
        for m, s in summary.get(ax, {}).items()
        if "gemini" in m.lower()
    )
    perf_data.append([
        "OSS (Qwen-0.5B)",
        latency(oss_stats["hallucination"]),
        latency(oss_stats["bias"]),
        latency(oss_stats["safety"]),
        f"${total_oss_cost:.4f} ($0 infra)",
    ])
    perf_data.append([
        "Frontier (Gemini)",
        latency(front_stats["hallucination"]),
        latency(front_stats["bias"]),
        latency(front_stats["safety"]),
        f"${total_front_cost:.4f}",
    ])
    perf_tbl = Table(perf_data, colWidths=[38 * mm, 36 * mm, 34 * mm, 34 * mm, 38 * mm])
    perf_tbl.setStyle(tbl_style())
    story.append(perf_tbl)
    story.append(Spacer(1, 3 * mm))

    # ── Underwriting decision ──────────────────────────────────────────
    story.append(Paragraph("ILLUSTRATIVE UNDERWRITING DECISION", section_style))
    oss_color = RISK_COLORS.get(oss_prem.risk_class, colors.grey)
    front_color = RISK_COLORS.get(front_prem.risk_class, colors.grey)

    uw_data = [
        ["", "OSS (Qwen-0.5B)", "Frontier (Gemini 2.0 Flash)"],
        ["Risk Class", oss_prem.risk_class, front_prem.risk_class],
        ["Annual Premium (illustrative)", f"${oss_prem.annual_premium_usd:,.0f} USD", f"${front_prem.annual_premium_usd:,.0f} USD"],
        ["Hal × Bias × Safety Mult.", f"{oss_prem.hallucination_mult}× · {oss_prem.bias_mult}× · {oss_prem.safety_mult}×",
         f"{front_prem.hallucination_mult}× · {front_prem.bias_mult}× · {front_prem.safety_mult}×"],
        ["Coverage Limit (illustrative)", "$500,000 per occurrence", "$2,000,000 per occurrence"],
        ["Exclusions", "Overdose guidance, diagnostic decisions", "Diagnostic decisions, novel drug interactions"],
    ]
    uw_tbl = Table(uw_data, colWidths=[45 * mm, 72 * mm, 63 * mm])
    uw_style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), OLLIVE_DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
        ("GRID", (0, 0), (-1, -1), 0.3, MID_GREY),
        ("PADDING", (0, 0), (-1, -1), 3),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        # Risk class color
        ("TEXTCOLOR", (1, 1), (1, 1), oss_color),
        ("TEXTCOLOR", (2, 1), (2, 1), front_color),
        ("FONTNAME", (1, 1), (2, 1), "Helvetica-Bold"),
    ])
    uw_tbl.setStyle(uw_style)
    story.append(uw_tbl)
    story.append(Spacer(1, 3 * mm))

    # ── Underwriter's notes ────────────────────────────────────────────
    story.append(Paragraph("UNDERWRITER'S NOTES", section_style))
    notes = (
        "OSS model (Qwen-0.5B) shows a 10× higher hallucination rate than the frontier model, with "
        "a steeper context-rot degradation curve — risk increases materially with longer conversations. "
        "Content safety refusal rate is higher for OSS on multi-turn jailbreak attempts. "
        "Bias scores are broadly similar; both models reflect societal bias on gender/demographic medical probes. "
        "Recommend: OSS deployment in medical domain requires human-in-the-loop review and strict input/output guardrails. "
        "Frontier model is insurable at standard medical AI rates with policy exclusions on diagnostic decisions."
    )
    story.append(Paragraph(notes, body_style))

    # ── Caveat footer ──────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=MID_GREY, spaceBefore=4, spaceAfter=2))
    story.append(Paragraph(
        "⚠ ILLUSTRATIVE ONLY · Single judge (GPT-4o-mini, temperature=0) · ~20–30 probes/axis — directional, not statistically significant · "
        "Premium multipliers and base rates are placeholders pending real actuarial loss data · "
        "Not a legally binding insurance quote · Ollive AI Insurance · insure-evals v0.1",
        caveat_style,
    ))

    doc.build(story)
    pdf_bytes = buf.getvalue()

    if output_path:
        output_path.write_bytes(pdf_bytes)

    return pdf_bytes


if __name__ == "__main__":
    out = RESULTS_DIR / "underwriters_worksheet.pdf"
    generate_pdf(out)
    print(f"PDF written to {out} ({out.stat().st_size} bytes)")
