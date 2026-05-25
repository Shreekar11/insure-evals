"""
Eval Dashboard tab — rate+degree charts (OSS vs Frontier) + context-rot curve + cost/latency table.
Reads from cached results/summary.json — no live API calls.
"""
import json
import gradio as gr
from pathlib import Path

from src.viz.charts import (
    rate_degree_chart,
    context_rot_chart,
    latency_cost_table,
    maker_checker_chart,
    maker_checker_stats,
)
from config import SUMMARY_JSON, RESULTS_DIR

GOLD_RESULTS = RESULTS_DIR / "gold_set_results.json"
CONTROL_RESULTS = RESULTS_DIR / "control_rejudge_results.json"


def _load_summary() -> dict:
    if SUMMARY_JSON.exists():
        return json.loads(SUMMARY_JSON.read_text())
    return {}


def _load_maker_checker() -> tuple[list, list]:
    gold = json.loads(GOLD_RESULTS.read_text()) if GOLD_RESULTS.exists() else []
    control = json.loads(CONTROL_RESULTS.read_text()) if CONTROL_RESULTS.exists() else []
    return gold, control


def build_dashboard_tab():
    summary = _load_summary()
    has_data = bool(summary)

    with gr.Column():
        gr.Markdown(
            "## AI Risk Evaluation Dashboard\n"
            "Comparative scores for OSS (Qwen2.5-0.5B) vs Frontier (Gemini 2.0 Flash).\n\n"
            "> **Methodology note:** temperature=0, ~20–30 probes per axis. Rates are "
            "**directional, not statistically significant**. Hallucination scored against 5 authored "
            "medical reference docs via **maker-checker** (GPT-4o-mini maker + Claude-3.5-Haiku blind "
            "checker + consensus — see section below). Bias scored via gold labels — no LLM judge. "
            "Safety scored via single-judge rubric + Llama Guard 4."
        )

        if not has_data:
            gr.Markdown(
                "⚠️ No eval results yet. Run `python scripts/run_eval.py` to generate results."
            )
            return

        gr.Markdown("### Rate (% failing) & Degree (mean severity 1–5)")
        rate_fig = rate_degree_chart(summary)
        gr.Plot(value=rate_fig)

        # ── Maker-Checker v2 comparison ──────────────────────────────────────
        gold, control = _load_maker_checker()
        if gold and control:
            s = maker_checker_stats(gold, control)
            base, mc = s["baseline"], s["maker_checker"]
            prec_delta = (mc["precision"] - base["precision"]) * 100
            rec_delta = (mc["recall"] - base["recall"]) * 100
            gr.Markdown(
                "### 🔬 Maker-Checker v2 — Hallucination Judge Upgrade\n"
                "The v1 single judge (GPT-4o-mini) scores groundedness against **5 reference docs only**. "
                "When the model answers correctly from knowledge *outside* those docs, the single judge "
                "flags a **false positive**. v2 adds a **blind second checker** (Claude-3.5-Haiku) that "
                "judges pure factual truth — docs-blind and maker-blind — and a deterministic asymmetric "
                "consensus that can *remove* a false alarm but never add one against the docs.\n\n"
                f"- **Gold set (pre-registered, N={base['tp']+base['fp']+base['tn']+base['fn']}):** "
                f"precision **{base['precision']:.0%} → {mc['precision']:.0%}** ({prec_delta:+.0f} pp), "
                f"recall held at **{mc['recall']:.0%}** ({rec_delta:+.0f} pp) — false alarms removed, "
                f"real hallucinations (recall guards) still caught.\n"
                f"- **Control set (N={s['control_total']} cached traces, no model re-run):** "
                f"**{s['control_removed']} false-positive flags removed**, {s['control_unchanged']} unchanged "
                f"— the fix is surgical, not indiscriminate.\n"
                "- **Proof case `hal_006`** (warfarin + ibuprofen): maker flagged sev-4 on a "
                "*medically correct* answer → blind checker confirmed `true` → consensus **overturned** the flag."
            )
            mc_fig = maker_checker_chart(gold, control)
            gr.Plot(value=mc_fig)
            gr.Markdown(
                "> **Caveat (mandatory):** N=6 gold + 34 control. **Directional, not statistically "
                "significant** — this demonstrates and corrects a documented failure mode; it is not a "
                "powered study. No kappa, no confidence intervals. Omission ≠ hallucination: the checker "
                "judges factual correctness only. Gold set is a constructed demonstration set, not "
                "production-representative."
            )

        rot_fig = context_rot_chart(summary)
        if rot_fig:
            gr.Markdown(
                "### Context-Rot: Memory-Only Recall Failure\n"
                "10 anchor sessions re-asked at turns 5, 10, and 20 with **no RAG** — model must answer from memory alone.\n\n"
                "**What the data shows:** OSS (Qwen-0.5B) fails **90%** of re-asks across *all* turns — "
                "including turn 5, before the 10-turn buffer eviction even occurs. "
                "The curve is flat, not rising: this is a **recall floor**, not decay over time. "
                "The 0.5B model cannot reliably retain a safety-critical fact it was given just 4 turns ago. "
                "Frontier (Gemini) holds at **30%** flat — those failures reflect knowledge gaps on 3 specific "
                "numeric anchors, not memory degradation.\n\n"
                "> **The insurable signal is the 60 percentage-point gap** — OSS vs Frontier — "
                "persistent across every turn. "
                "Single judge (GPT-4o-mini). N=10 sessions per turn. Directional, not statistically significant."
            )
            gr.Plot(value=rot_fig)

        gr.Markdown(
            "### Cost & Latency\n"
            "> ⚠️ **Caveat:** OSS runs in-process on a free HF CPU; Frontier runs on hosted GPU API. "
            "This comparison reflects free-tier CPU infra as much as model capability — "
            "not a fair model-to-model latency comparison."
        )
        table_rows = latency_cost_table(summary)
        if table_rows:
            gr.Dataframe(
                value=table_rows,
                headers=["Axis", "Model", "N", "Rate", "Mean Severity", "Avg Latency", "Total Cost"],
                datatype=["str", "str", "number", "str", "str", "str", "str"],
                interactive=False,
            )
