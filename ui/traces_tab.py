"""
Traces tab — drill into eval probes. Harmful jailbreak responses are REDACTED.
"""
import json
import gradio as gr

from src.eval.store import fetch_all, fetch_from_json
from config import TRACES_DB


def _load_traces() -> list[dict]:
    # Prefer SQLite (local dev); fall back to JSON export (HF Spaces)
    if TRACES_DB.exists():
        return fetch_all()
    return fetch_from_json()


def _sanitize_response(row: dict) -> str:
    """Never expose raw harmful jailbreak content in the public UI."""
    is_jailbreak = row.get("extra", {}).get("type") == "jailbreak" if isinstance(row.get("extra"), dict) else False
    if is_jailbreak and row.get("flagged"):
        return "[REDACTED — flagged unsafe by judge/guardrail]"
    # Also redact if explicitly marked
    response = row.get("response", "")
    if response.startswith("[REDACTED"):
        return response
    return response


def _format_rows(traces: list[dict]) -> list[list]:
    rows = []
    for t in traces:
        extra = t.get("extra")
        if isinstance(extra, str):
            try:
                extra = json.loads(extra)
            except Exception:
                extra = {}
        probe_type = extra.get("type", "") if extra else ""
        verdict = extra.get("verdict", "") if extra else ""

        rows.append([
            t["id"],
            t["axis"],
            t["model"][:30],
            t["probe_id"],
            t["prompt"][:120] + ("…" if len(t["prompt"]) > 120 else ""),
            _sanitize_response({**t, "extra": extra})[:200] + "…",
            "✓ FLAGGED" if t["flagged"] else "ok",
            t["severity"],
            f"{t['latency_ms']:.0f}ms",
            f"${t['cost_usd']:.5f}",
            t.get("judge_reasoning", "")[:100],
        ])
    return rows


def build_traces_tab():
    with gr.Column():
        gr.Markdown(
            "## Eval Traces\n"
            "Drill into individual probe results. "
            "Raw responses for flagged jailbreak probes are redacted — "
            "only metadata and scores are shown for safety-classified harmful content.\n\n"
            "**Export:** Download the raw SQLite DB to analyse results locally."
        )

        refresh_btn = gr.Button("Refresh traces", variant="secondary")
        trace_count = gr.Markdown("Loading…")
        traces_df = gr.Dataframe(
            headers=["ID", "Axis", "Model", "Probe", "Prompt", "Response", "Result", "Severity", "Latency", "Cost", "Reasoning"],
            datatype=["number", "str", "str", "str", "str", "str", "str", "number", "str", "str", "str"],
            interactive=False,
            wrap=True,
        )

        def refresh():
            traces = _load_traces()
            rows = _format_rows(traces)
            count_md = f"**{len(traces)} traces** loaded from `results/traces.db`"
            return count_md, rows

        refresh_btn.click(refresh, outputs=[trace_count, traces_df])

        # Auto-load on mount
        traces = _load_traces()
        rows = _format_rows(traces)
        trace_count.value = f"**{len(traces)} traces** loaded from `results/traces.db`"
        traces_df.value = rows
