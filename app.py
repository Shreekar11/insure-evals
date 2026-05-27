"""
insure-evals — AI Risk Evaluation Harness for AI Liability Insurance
"I didn't build an eval harness. I built v0.1 of an AI underwriting engine."

HF Spaces entrypoint. Assembles all tabs from ui/.
Reads OPENROUTER_API_KEY and GEMINI_API_KEY from env / HF Space Secrets.
"""
import os
import sys

# Allow imports from repo root on HF Spaces
sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from ui.chat_tab import build_chat_tab
from ui.dashboard_tab import build_dashboard_tab
from ui.traces_tab import build_traces_tab
from ui.premium_tab import build_premium_tab


def build_app() -> gr.Blocks:
    with gr.Blocks(title="insure-evals — AI Risk Evaluation") as demo:
        gr.Markdown(
            "# 🔬 insure-evals\n"
            "**AI Risk Evaluation Harness** | AI liability insurance for agents.\n\n"
            "Measures **hallucination rate**, **bias rate**, and **content safety rate** "
            "(each as rate % + mean severity 1–5) across an OSS model (Qwen2.5-0.5B) "
            "and a frontier model (Gemini 2.0 Flash). "
            "Scores feed an illustrative insurance underwriting formula."
        )

        with gr.Tabs():
            with gr.Tab("💬 Chat"):
                build_chat_tab()

            with gr.Tab("📊 Eval Dashboard"):
                build_dashboard_tab()

            with gr.Tab("💰 Premium Calculator"):
                build_premium_tab()

            with gr.Tab("🔍 Traces"):
                build_traces_tab()

            with gr.Tab("ℹ️ About"):
                gr.Markdown(
                    "## About insure-evals\n\n"
                    "### What this is\n"
                    "A prototype AI underwriting measurement layer. "
                    "Two chat assistants (OSS + frontier) in a medical domain are scored "
                    "on hallucination, bias, and content safety — the same axes an insurer "
                    "would use to price a liability policy for an AI agent.\n\n"
                    "### Evaluation methodology\n"
                    "| Axis | Method |\n|---|---|\n"
                    "| Hallucination | 17 probes against 5 authored medical reference docs; "
                    "**maker** (GPT-4o-mini, groundedness) + **blind checker** (Claude-3.5-Haiku, "
                    "factual truth) + deterministic consensus — removes false alarms without "
                    "overriding the docs |\n"
                    "| Bias | 11 probes: BBQ-style gold-label accuracy (no judge) + "
                    "custom medical bias probes + factual-bias answer key |\n"
                    "| Content Safety | 17 probes: jailbreak + over-refusal; "
                    "GPT-4o-mini rubric judge + Llama Guard 3 as independent second signal |\n\n"
                    "### Honesty caveats\n"
                    "- **Hallucination uses maker-checker** (GPT-4o-mini + Claude-3.5-Haiku, temperature=0); "
                    "bias/safety still single-judge. 2-judge consensus, not a full ensemble — no Cohen's kappa, no confidence intervals\n"
                    "- **Small N** (~20–30 probes/axis) — rates are directional, not statistically significant\n"
                    "- **Premium is illustrative** — multipliers are placeholders pending real actuarial data\n"
                    "- **Latency caveat** — OSS runs on free HF CPU, frontier on hosted GPU API; not a fair comparison\n\n"
                    "### Stack\n"
                    "OSS: `Qwen/Qwen2.5-0.5B-Instruct` via HuggingFace `transformers` (in-process, CPU) | "
                    "Frontier: `google/gemini-2.0-flash-001` via OpenRouter | "
                    "Maker judge: `openai/gpt-4o-mini` via OpenRouter | "
                    "Blind checker: `anthropic/claude-3.5-haiku` via OpenRouter | "
                    "Safety classifier: `meta-llama/llama-guard-4-12b` via OpenRouter | "
                    "UI: Gradio on HF Spaces\n\n"
                    "### Source\n"
                    "GitHub: [Shreekar11/insure-evals](https://github.com/Shreekar11/insure-evals)"
                )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(),
    )
