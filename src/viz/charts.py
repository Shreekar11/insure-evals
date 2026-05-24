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
    Context-rot curve: memory-only recall failure rate vs conversation turn, per model.
    Uses categorical x-axis (equal spacing) so turn 20 doesn't dwarf turns 5/10.
    Marks the buffer-eviction boundary and annotates the OSS/Frontier gap.
    """
    rot = summary.get("context_rot", {})
    if not rot:
        return None

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_title("Context-Rot: Memory-Only Recall Failure Rate vs Conversation Turn",
                 fontsize=11, fontweight="bold")

    all_turns: list[int] = []
    plot_data: list[tuple] = []
    for model, points in rot.items():
        pts = sorted(points, key=lambda p: p["turn"])
        turns = [p["turn"] for p in pts]
        rates = [p["rate"] * 100 for p in pts]
        ns = [p["n"] for p in pts]
        color = OSS_COLOR if ("qwen" in model.lower() or "oss" in model.lower()) else FRONTIER_COLOR
        label = _model_short(model)
        plot_data.append((turns, rates, ns, color, label))
        all_turns = turns  # same for all models

    # Categorical positions for equal spacing
    positions = list(range(len(all_turns)))
    tick_labels = [f"Turn {t}\n(N={plot_data[0][2][i]})" for i, t in enumerate(all_turns)]

    for turns, rates, ns, color, label in plot_data:
        ax.plot(positions, rates, "o-", color=color, label=label,
                linewidth=2.5, markersize=8, zorder=3)
        for i, (pos, rate) in enumerate(zip(positions, rates)):
            ax.annotate(f"{rate:.0f}%", xy=(pos, rate),
                        xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=9, color=color, fontweight="bold")

    # Eviction boundary: buffer evicts anchor after turn 10 (between positions 1 and 2)
    if len(positions) >= 3:
        evict_x = (positions[1] + positions[2]) / 2
        ax.axvline(evict_x, color="gray", linestyle="--", linewidth=1, alpha=0.6, zorder=1)
        ax.text(evict_x + 0.04, 105, "← buffer eviction", fontsize=8,
                color="gray", va="top")

    # Gap annotation between the two models at the last turn
    if len(plot_data) == 2:
        rates_a = plot_data[0][1]
        rates_b = plot_data[1][1]
        gap = abs(rates_a[-1] - rates_b[-1])
        if gap > 5:
            top = max(rates_a[-1], rates_b[-1])
            bot = min(rates_a[-1], rates_b[-1])
            mid = (top + bot) / 2
            ax.annotate("", xy=(positions[-1] + 0.18, top),
                        xytext=(positions[-1] + 0.18, bot),
                        arrowprops=dict(arrowstyle="<->", color="#aaa", lw=1.2))
            ax.text(positions[-1] + 0.24, mid, f"{gap:.0f}pp\ngap",
                    fontsize=8, color="#aaa", va="center")

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Recall Failure Rate (%)")
    ax.set_ylim(0, 120)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlim(-0.4, len(positions) - 0.4)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    note = ("Memory-only re-ask (no RAG). N=10 sessions/turn. "
            "Directional signal — not statistically significant.")
    ax.annotate(note, xy=(0.5, -0.22), xycoords="axes fraction",
                ha="center", fontsize=8, color="gray")
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
