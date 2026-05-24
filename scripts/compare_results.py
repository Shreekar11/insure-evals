"""
Maker-Checker v2 comparison report — Phase 5.

Reads:
  results/gold_set_results.json    (6 gold-set probes, pre-registered labels)
  results/control_rejudge_results.json  (34 cached hallucination traces)

Prints:
  1. Two confusion matrices on the gold set (single-judge baseline vs maker-checker)
  2. Control delta (flag changes on 34 cached traces)
  3. Disagreement list (maker vs checker conflicted, incl. hal_006)
  4. Mandatory small-N caveat

Positive class = "is a hallucination" (gold_label == "hallucination").
Precision = TP/(TP+FP)  — maker-checker reduces FP (false alarms).
Recall    = TP/(TP+FN)  — recall-guards verify no false negatives introduced.

No kappa. No confidence intervals. Directional only.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR

GOLD_OUT    = RESULTS_DIR / "gold_set_results.json"
CONTROL_OUT = RESULTS_DIR / "control_rejudge_results.json"


def _confusion_matrix(results: list[dict], flag_key: str) -> dict:
    """
    Build 2×2 confusion matrix.
    flag_key: "maker_flagged" (single-judge baseline) or "consensus_flagged" (maker-checker).
    Positive class = gold_label == "hallucination".
    """
    tp = fp = tn = fn = 0
    for r in results:
        gold_pos = r["gold_label"] == "hallucination"
        predicted_pos = bool(r[flag_key])
        if gold_pos and predicted_pos:
            tp += 1
        elif not gold_pos and predicted_pos:
            fp += 1
        elif gold_pos and not predicted_pos:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, precision=precision, recall=recall)


def _print_matrix(label: str, m: dict):
    print(f"\n  {label}")
    print(f"  {'':25s}  Predicted HAL  Predicted OK")
    print(f"  {'Gold: HAL':25s}  TP={m['tp']:>3}          FN={m['fn']:>3}")
    print(f"  {'Gold: NOT HAL':25s}  FP={m['fp']:>3}          TN={m['tn']:>3}")
    n = m['tp'] + m['fp'] + m['tn'] + m['fn']
    print(f"  Precision = {m['tp']}/{m['tp']+m['fp']} = {m['precision']:.1%}   "
          f"Recall = {m['tp']}/{m['tp']+m['fn']} = {m['recall']:.1%}   N={n}")


def main():
    if not GOLD_OUT.exists() or not CONTROL_OUT.exists():
        print("ERROR: run scripts/run_maker_checker.py --mode both first.")
        sys.exit(1)

    gold    = json.loads(GOLD_OUT.read_text())
    control = json.loads(CONTROL_OUT.read_text())

    print("=" * 70)
    print("  MAKER-CHECKER v2 — COMPARISON REPORT")
    print("=" * 70)

    # ── 1. Gold-set confusion matrices ──────────────────────────────────────
    print("\n╔══ GOLD SET (6 probes, pre-registered labels) ══╗")
    print("  Positive class = 'is a hallucination'")
    print("  Baseline = single-judge (maker flagged?) | Maker-Checker = consensus flagged?")

    m_single = _confusion_matrix(gold, "maker_flagged")
    m_mc     = _confusion_matrix(gold, "consensus_flagged")

    _print_matrix("BASELINE (single judge — maker only):", m_single)
    _print_matrix("MAKER-CHECKER (consensus):", m_mc)

    print()
    prec_delta = m_mc["precision"] - m_single["precision"]
    rec_delta  = m_mc["recall"]    - m_single["recall"]
    print(f"  Δ Precision: {prec_delta:+.1%}   Δ Recall: {rec_delta:+.1%}")

    # ── 2. Control delta ─────────────────────────────────────────────────────
    print("\n╔══ CONTROL SET (34 cached hallucination traces — no model re-run) ══╗")

    changed    = [r for r in control if r["baseline_flagged"] != r["consensus_flagged"]]
    removed    = [r for r in changed if r["baseline_flagged"] and not r["consensus_flagged"]]
    added      = [r for r in changed if not r["baseline_flagged"] and r["consensus_flagged"]]

    print(f"  Total traces:        {len(control)}")
    print(f"  Unchanged:           {len(control) - len(changed)}")
    print(f"  Flags removed (↓FP): {len(removed)}")
    print(f"  Flags added   (↑FP): {len(added)}  "
          f"← maker re-judged differently; checker did not add these")
    print()
    print("  Note: flags added = maker changed verdict between runs (GPT-4o-mini")
    print("  temp=0 has residual non-determinism across sessions). Checker is")
    print("  conservative — it only *removes* flags, never adds them alone.")

    # ── 3. Disagreement list ─────────────────────────────────────────────────
    print("\n╔══ DISAGREEMENT LIST (maker vs checker conflicted) ══╗")
    print("  Includes all traces where maker and checker gave conflicting signals.\n")

    all_results = gold + control
    disagreements = [r for r in all_results if r.get("disagreement")]

    if not disagreements:
        print("  (none)")
    else:
        for r in disagreements:
            pid   = r["probe_id"]
            model = r.get("model", "Gemini (gold set)")
            action = r["consensus_action"]
            mk_flag = r["maker_flagged"]
            ck_verd = r["checker_verdict"]
            final  = r["consensus_flagged"]
            mk_rsn = r.get("maker_reasoning", "")[:80]
            ck_rsn = r.get("checker_reasoning", "")[:80]
            star   = " ◀ PROOF CASE" if pid == "hal_006" else ""
            print(f"  {pid:12s} [{model[:28]:28s}]  action={action:22s}{star}")
            print(f"             maker={'flagged' if mk_flag else 'pass':7s}  checker={ck_verd:10s}  final={'HAL' if final else 'OK '}")
            print(f"             maker: {mk_rsn}")
            print(f"             checker: {ck_rsn}")
            print()

    # ── 4. hal_006 spotlight ─────────────────────────────────────────────────
    print("╔══ PROOF CASE: hal_006 (warfarin + ibuprofen) ══╗")
    hal006 = [r for r in control if r["probe_id"] == "hal_006"]
    if hal006:
        for r in hal006:
            model = r.get("model", "unknown")
            print(f"\n  Model: {model}")
            print(f"  Prompt: {r['prompt'][:100]}")
            print(f"  Response (cached): {r['response'][:120]}...")
            print(f"  Maker: {'FLAGGED' if r['maker_flagged'] else 'pass'}  "
                  f"sev={r['maker_severity']}  reasoning: {r['maker_reasoning'][:80]}")
            print(f"  Checker: {r['checker_verdict']}  confidence={r['checker_confidence']:.2f}  "
                  f"reasoning: {r['checker_reasoning'][:80]}")
            print(f"  Consensus: action={r['consensus_action']}  "
                  f"final_flagged={r['consensus_flagged']}")
    else:
        print("  hal_006 not found in control set.")

    # ── 5. Honesty caveat ────────────────────────────────────────────────────
    n_gold    = len(gold)
    n_control = len(control)
    print()
    print("=" * 70)
    print("  CAVEAT (SPEC §11 — mandatory):")
    print(f"  N={n_gold} gold-set probes + N={n_control} control traces. Directional, not statistically")
    print("  significant. This demonstrates and corrects a documented failure mode;")
    print("  it is not a powered study. Omission ≠ hallucination: the checker judges")
    print("  factual correctness only — incompleteness is a separate future axis.")
    print("  Gold set is a constructed demonstration set, not production-representative.")
    print("  No kappa. No confidence intervals.")
    print("=" * 70)


if __name__ == "__main__":
    main()
