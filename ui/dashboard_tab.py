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
                "### Context-Rot Curve\n"
                "Hallucination rate vs conversation turn — measuring real memory degradation across 10 anchor sessions.\n\n"
                "**Methodology:** Each session establishes an anchor fact at turn 1 (with RAG). "
                "Distractor turns fill the conversation buffer. At turns 5, 10, and 20 the anchor is re-asked "
                "**with no RAG injection** — the model must answer from memory alone. "
                "After turn 10, the anchor is evicted from the rolling 10-turn buffer (memory rot). "
                "A re-ask at turn 20 forces pure recall from an empty buffer slot → the 0.5B model hallucinates; "
                "Gemini's stronger retention holds.\n\n"
                "> Single judge (GPT-4o-mini). N=10 sessions per turn bucket. "
                "Directional signal, not statistically significant. "
                "OSS > Frontier at turn 20 is the insurable risk gradient."
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
