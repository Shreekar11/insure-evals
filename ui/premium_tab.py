"""
Premium Calculator tab — illustrative annual premium from eval scores.
"""
import json
import gradio as gr

from src.underwriting.premium import calculate
from config import SUMMARY_JSON


def _get_rates_from_summary(model_key: str) -> tuple[float, float, float]:
    """Extract hal/bias/safety rates from summary.json for a given model substring."""
    if not SUMMARY_JSON.exists():
        return 0.10, 0.10, 0.10
    summary = json.loads(SUMMARY_JSON.read_text())
    rates = {}
    for axis in ("hallucination", "bias", "safety"):
        for model, stats in summary.get(axis, {}).items():
            if model_key.lower() in model.lower():
                rates[axis] = stats.get("rate", 0.0)
    return (
        rates.get("hallucination", 0.10),
        rates.get("bias", 0.10),
        rates.get("safety", 0.10),
    )


def compute_premium(model_choice: str, volume: str) -> str:
    key = "qwen" if "oss" in model_choice.lower() else "gemini"
    hal, bias, safety = _get_rates_from_summary(key)
    result = calculate(hal, bias, safety, deployment_volume=volume, domain="medical")

    sensitivity_line = ""
    saves = result.sensitivity.get("cut_hallucination_30pct_saves_usd", 0)
    if saves > 0:
        sensitivity_line = f"\n💡 **Sensitivity:** Cutting hallucination rate 30% → saves **${saves:,.0f}/yr**"

    return (
        f"### Underwriting Quote — {model_choice}\n\n"
        f"| Item | Value |\n|---|---|\n"
        f"| Annual Premium (illustrative) | **${result.annual_premium_usd:,.0f} USD/yr** |\n"
        f"| Risk Class | **{result.risk_class}** |\n"
        f"| Hallucination rate | {hal:.1%} × {result.hallucination_mult}× |\n"
        f"| Bias rate | {bias:.1%} × {result.bias_mult}× |\n"
        f"| Content safety rate | {safety:.1%} × {result.safety_mult}× |\n"
        f"| Deployment volume | {volume} × {result.volume_mult}× |\n"
        f"| Domain (medical) | {result.domain_mult}× |\n"
        f"{sensitivity_line}\n\n"
        f"> {result.caveat}"
    )


def build_premium_tab():
    with gr.Column():
        gr.Markdown(
            "## Illustrative Premium Calculator\n"
            "Translates eval scores into an insurance underwriting quote. "
            "This is **v0.1 of Ollive's underwriting engine** — the formula structure is "
            "the deliverable; the constants are placeholders.\n\n"
            "> ⚠️ **All premium figures are illustrative.** Multipliers and base rates are "
            "placeholders pending real actuarial loss data. Not a legally binding quote."
        )

        with gr.Row():
            model_choice = gr.Radio(
                choices=["OSS (Qwen2.5-0.5B)", "Frontier (Gemini 2.0 Flash)"],
                value="OSS (Qwen2.5-0.5B)",
                label="Agent to underwrite",
            )
            volume = gr.Radio(
                choices=["low", "medium", "high", "enterprise"],
                value="medium",
                label="Deployment volume",
            )

        quote_btn = gr.Button("Generate Quote", variant="primary")
        quote_output = gr.Markdown(value="*Click 'Generate Quote' to compute.*")

        quote_btn.click(
            compute_premium,
            inputs=[model_choice, volume],
            outputs=[quote_output],
        )

        gr.Markdown(
            "### Formula Structure\n"
            "```\n"
            "premium = base_rate\n"
            "        × hallucination_multiplier(rate)\n"
            "        × bias_multiplier(rate)\n"
            "        × safety_multiplier(rate)\n"
            "        × deployment_volume_multiplier\n"
            "        × domain_risk_multiplier\n"
            "```\n"
            "Each multiplier maps the failure rate to a risk loading "
            "(e.g. <5% rate → 1.0×, >50% rate → 4.0×). "
            "Real actuarial pricing requires loss history, claim severity data, "
            "and reinsurance modelling — none of which are available at this prototype stage."
        )
