"""
Probe runner — iterates probes × both agents, scores, persists to SQLite.
"""
import json
import time
from pathlib import Path
from typing import Callable

from src.eval.schema import ProbeResult
from src.eval.store import insert, init_db
from config import PROBES_DIR


def load_probes(axis: str) -> list[dict]:
    path = PROBES_DIR / f"{axis}.json"
    if not path.exists():
        raise FileNotFoundError(f"Probe file not found: {path}")
    with open(path) as f:
        return json.load(f)


def run_axis(
    axis: str,
    agents: list,
    score_fn: Callable[[dict, object], ProbeResult],
    verbose: bool = True,
    delay_s: float = 1.0,
) -> list[ProbeResult]:
    """
    Run all probes for one axis against all agents.
    delay_s: sleep between API calls to avoid rate limits.
    """
    probes = load_probes(axis)
    results: list[ProbeResult] = []

    for probe in probes:
        for agent in agents:
            if verbose:
                print(f"  [{axis}] {probe['id']} | {agent.name}...", end=" ", flush=True)
            try:
                result = score_fn(probe, agent)
                insert(result)
                results.append(result)
                flag_str = "FLAGGED" if result.flagged else "ok"
                if verbose:
                    print(f"{flag_str} sev={result.severity} {result.latency_ms:.0f}ms")
            except Exception as e:
                if verbose:
                    print(f"ERROR: {e}")
            time.sleep(delay_s)

    return results
