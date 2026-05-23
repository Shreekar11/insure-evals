"""
Chart builders for the dashboard tab.
Returns Matplotlib figures (rendered by Gradio as gr.Plot).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


OSS_COLOR = "#e05c5c"       # red-ish
FRONTIER_COLOR = "#4a90d9"  # blue-ish
AXES_ORDER = ["hallucination", "bias", "safety"]
AXIS_LABELS = {"hallucination": "Hallucination", "bias": "Bias", "safety": "Content Safety"}


def _model_short(name: str) -> str:
    if "qwen" in name.lower() or "oss" in name.lower():
        return "OSS (Qwen-0.5B)"
    return "Frontier (Gemini)"


def rate_degree_chart(summary: dict) -> plt.Figure:
    """
    Side-by-side bar chart: Rate (%) and Degree (mean severity 1-5) per axis, both models.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Risk Evaluation: OSS vs Frontier", fontsize=13, fontweight="bold", y=1.01)

    x = np.arange(len(AXES_ORDER))
    bar_w = 0.35

    for col, (metric, ylabel, scale) in enumerate([
        ("rate", "Failure Rate (%)", 100),
        ("mean_severity", "Mean Severity (1–5)", 1),
    ]):
        ax = axes[col]
        oss_vals, front_vals = [], []
        for axis in AXES_ORDER:
            axis_data = summary.get(axis, {})
            oss_v = front_v = 0.0
            for model, stats in axis_data.items():
                if "qwen" in model.lower() or "oss" in model.lower():
                    oss_v = stats.get(metric, 0) * scale
                else:
                    front_v = stats.get(metric, 0) * scale
            oss_vals.append(oss_v)
            front_vals.append(front_v)

        bars1 = ax.bar(x - bar_w / 2, oss_vals, bar_w, label="OSS (Qwen-0.5B)", color=OSS_COLOR, alpha=0.85)
        bars2 = ax.bar(x + bar_w / 2, front_vals, bar_w, label="Frontier (Gemini)", color=FRONTIER_COLOR, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([AXIS_LABELS[a] for a in AXES_ORDER])
        ax.set_ylabel(ylabel)
        if metric == "rate":
            ax.set_ylim(0, 110)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        else:
            ax.set_ylim(0, 5.5)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar in bars1:
            h = bar.get_height()
            if h > 0:
                label = f"{h:.1f}%" if metric == "rate" else f"{h:.2f}"
                ax.annotate(label, xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        for bar in bars2:
            h = bar.get_height()
            if h > 0:
                label = f"{h:.1f}%" if metric == "rate" else f"{h:.2f}"
                ax.annotate(label, xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    return fig


def context_rot_chart(summary: dict) -> plt.Figure | None:
    """
    Context-rot curve: hallucination rate vs conversation turn, per model.
    """
    rot = summary.get("context_rot", {})
    if not rot:
        return None

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_title("Context-Rot Curve: Hallucination Rate vs Conversation Length",
                 fontsize=11, fontweight="bold")

    for model, points in rot.items():
        turns = [p["turn"] for p in sorted(points, key=lambda p: p["turn"])]
        rates = [p["rate"] * 100 for p in sorted(points, key=lambda p: p["turn"])]
        color = OSS_COLOR if ("qwen" in model.lower() or "oss" in model.lower()) else FRONTIER_COLOR
        label = _model_short(model)
        ax.plot(turns, rates, "o-", color=color, label=label, linewidth=2, markersize=6)

    ax.set_xlabel("Conversation Turn")
    ax.set_ylabel("Hallucination Rate (%)")
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    note = "Note: N is small per turn — directional, not statistically significant."
    ax.annotate(note, xy=(0.5, -0.18), xycoords="axes fraction", ha="center", fontsize=8, color="gray")
    plt.tight_layout()
    return fig


def latency_cost_table(summary: dict) -> list[list]:
    """Returns rows for a gr.Dataframe."""
    rows = []
    for axis in AXES_ORDER:
        for model, stats in summary.get(axis, {}).items():
            short = _model_short(model)
            rows.append([
                AXIS_LABELS[axis],
                short,
                stats.get("n", 0),
                f"{stats.get('rate', 0):.1%}",
                f"{stats.get('mean_severity', 0):.2f}",
                f"{stats.get('mean_latency_ms', 0):.0f} ms",
                f"${stats.get('total_cost_usd', 0):.5f}",
            ])
    return rows
