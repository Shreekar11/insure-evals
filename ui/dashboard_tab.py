"""
Eval Dashboard tab — rate+degree charts (OSS vs Frontier) + context-rot curve + cost/latency table.
Reads from cached results/summary.json — no live API calls.
"""
import json
import gradio as gr
from pathlib import Path

from src.viz.charts import rate_degree_chart, context_rot_chart, latency_cost_table
from config import SUMMARY_JSON


def _load_summary() -> dict:
    if SUMMARY_JSON.exists():
        return json.loads(SUMMARY_JSON.read_text())
    return {}


def build_dashboard_tab():
    summary = _load_summary()
    has_data = bool(summary)

    with gr.Column():
        gr.Markdown(
            "## AI Risk Evaluation Dashboard\n"
            "Comparative scores for OSS (Qwen2.5-0.5B) vs Frontier (Gemini 2.0 Flash).\n\n"
            "> **Methodology note:** Single judge (GPT-4o-mini, temperature=0). "
            "~20–30 probes per axis. Rates are **directional, not statistically significant**. "
            "Hallucination scored against 5 authored medical reference docs (ground truth). "
            "Bias scored via gold labels — no LLM judge. Safety scored via judge rubric + Llama Guard 3."
        )

        if not has_data:
            gr.Markdown(
                "⚠️ No eval results yet. Run `python scripts/run_eval.py` to generate results."
            )
            return

        gr.Markdown("### Rate (% failing) & Degree (mean severity 1–5)")
        rate_fig = rate_degree_chart(summary)
        gr.Plot(value=rate_fig)

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
