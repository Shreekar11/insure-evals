"""
Offline eval runner — populates results/traces.db + results/summary.json.

Usage:
  python scripts/run_eval.py                        # all axes
  python scripts/run_eval.py --axis hallucination   # one axis
  python scripts/run_eval.py --axis bias safety     # multiple axes
  python scripts/run_eval.py --axis context_rot     # context-rot only (opt-in)

Caveats (per SPEC):
  - Single judge (GPT-4o-mini) — no ensemble, no confidence intervals
  - ~20-30 probes/axis; rates are directional, not statistically significant
  - Temperature=0 + pinned model versions for reproducibility
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.eval.store import init_db, summary
from src.eval.runner import run_axis
from src.eval.scorers import hallucination, bias, safety
from config import SUMMARY_JSON

AXIS_SCORERS = {
    "hallucination": hallucination.score_probe,
    "bias": bias.score_probe,
    "safety": safety.score_probe,
}

_STANDARD_AXES = ["hallucination", "bias", "safety"]
_ALL_AXES = _STANDARD_AXES + ["context_rot"]


def _run_context_rot(agents, delay_s: float = 1.5):
    import json as _json
    from pathlib import Path
    from src.eval.conversation_runner import run_drift_session
    from src.eval.store import insert

    root = Path(__file__).parent.parent
    sessions_path = root / "data" / "probes" / "context_rot.json"
    distractors_path = root / "data" / "probes" / "distractors.json"

    sessions = _json.loads(sessions_path.read_text())
    distractors = _json.loads(distractors_path.read_text())

    print(f"\n=== Running axis: context_rot ({len(sessions)} sessions × {len(agents)} agents) ===")
    print("NOTE: Each session runs up to 20 turns. This is slow — use --oss-only for quick tests.")

    for session in sessions:
        for agent in agents:
            print(f"  Session {session['id']} | Agent {agent.name} ...")
            results = run_drift_session(session, agent, distractors, delay_s=delay_s)
            for r in results:
                insert(r)
            n_flagged = sum(1 for r in results if r.flagged)
            print(f"    → {len(results)} re-asks scored, {n_flagged} flagged")


def main():
    parser = argparse.ArgumentParser(description="Run insure-evals probe suite")
    parser.add_argument(
        "--axis",
        nargs="*",
        choices=_ALL_AXES,
        default=_STANDARD_AXES,  # context_rot is opt-in, not default
        help="Which axes to run (default: hallucination bias safety; add context_rot explicitly)",
    )
    parser.add_argument(
        "--oss-only", action="store_true", help="Run OSS agent only (skip frontier)"
    )
    parser.add_argument(
        "--frontier-only", action="store_true", help="Run frontier agent only (skip OSS)"
    )
    parser.add_argument(
        "--delay", type=float, default=1.5, help="Seconds between API calls (default 1.5)"
    )
    args = parser.parse_args()

    print("Initialising database...")
    init_db()

    print("Loading agents...")
    agents = []
    if not args.frontier_only:
        from src.agents.oss_agent import OSSAgent
        oss = OSSAgent()
        agents.append(oss)
        print(f"  OSS agent: {oss.name}")

    if not args.oss_only:
        from src.agents.frontier_agent import FrontierAgent
        frontier = FrontierAgent()
        agents.append(frontier)
        print(f"  Frontier agent: {frontier.name}")

    if not agents:
        print("No agents to run.")
        return

    # Save original axes before filtering out context_rot
    original_axes = list(args.axis)

    if "context_rot" in args.axis:
        _run_context_rot(agents, delay_s=args.delay)
        args.axis = [a for a in args.axis if a != "context_rot"]

    for axis in args.axis:
        print(f"\n=== Running axis: {axis} ===")
        score_fn = AXIS_SCORERS[axis]
        run_axis(axis, agents, score_fn, verbose=True, delay_s=args.delay)

    print("\nComputing summary...")
    summ = summary()
    SUMMARY_JSON.write_text(json.dumps(summ, indent=2))
    print(f"Summary written to {SUMMARY_JSON}")

    # Print rate+degree table (standard axes only — context_rot has its own output)
    print("\n=== Results (rate = % flagged, degree = mean severity) ===")
    print(f"{'Axis':<15} {'Model':<35} {'N':>4} {'Rate':>7} {'Degree':>7} {'ms':>8} {'$':>8}")
    print("-" * 80)
    for axis in ("hallucination", "bias", "safety"):
        for model, stats in summ.get(axis, {}).items():
            short_model = model[:34]
            print(
                f"{axis:<15} {short_model:<35} {stats['n']:>4} "
                f"{stats['rate']:>7.1%} {stats['mean_severity']:>7.2f} "
                f"{stats['mean_latency_ms']:>8.0f} {stats['total_cost_usd']:>8.5f}"
            )
    print(f"\nSingle-judge (GPT-4o-mini) | N is small — directional, not statistically significant")

    if "context_rot" in original_axes:
        rot = summ.get("context_rot", {})
        if rot:
            print("\n=== Context-Rot Curve ===")
            for model, turns in rot.items():
                print(f"  {model[:40]}:")
                for t in sorted(turns, key=lambda x: x["turn"]):
                    print(f"    Turn {t['turn']:>3}: rate={t['rate']:.1%} (N={t['n']})")


if __name__ == "__main__":
    main()
