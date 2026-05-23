"""
Offline eval runner — populates results/traces.db + results/summary.json.

Usage:
  python scripts/run_eval.py                        # all axes
  python scripts/run_eval.py --axis hallucination   # one axis
  python scripts/run_eval.py --axis bias safety     # multiple axes

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


def main():
    parser = argparse.ArgumentParser(description="Run insure-evals probe suite")
    parser.add_argument(
        "--axis",
        nargs="*",
        choices=list(AXIS_SCORERS.keys()),
        default=list(AXIS_SCORERS.keys()),
        help="Which axes to run (default: all)",
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

    for axis in args.axis:
        print(f"\n=== Running axis: {axis} ===")
        score_fn = AXIS_SCORERS[axis]
        run_axis(axis, agents, score_fn, verbose=True, delay_s=args.delay)

    print("\nComputing summary...")
    summ = summary()
    SUMMARY_JSON.write_text(json.dumps(summ, indent=2))
    print(f"Summary written to {SUMMARY_JSON}")

    # Print rate+degree table
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


if __name__ == "__main__":
    main()
