"""
Deterministic asymmetric consensus for the maker-checker hallucination layer.

Implements the §4 table from Maker-Checker v2 - SPEC.md exactly.

Maker ∈ {grounded, ungrounded} × Checker ∈ {true, false, uncertain}

| Maker      | Checker   | Meaning                            | Action              |
|------------|-----------|------------------------------------|---------------------|
| grounded   | true      | fine                               | pass (not flagged)  |
| grounded   | false     | in docs but checker says false     | keep_grounded_log   |
| grounded   | uncertain | —                                  | pass (not flagged)  |
| ungrounded | true      | ungrounded-but-true → THE FIX      | overturn (sev → 0)  |
| ungrounded | false     | real hallucination / recall guard  | flag_confirmed      |
| ungrounded | uncertain | can't confirm true                 | keep_flag_uncertain |

One-directional override: checker may REMOVE a flag (ungrounded+true) but NEVER add one
against the authoritative docs (grounded+false → keep maker's verdict + log for review).
"uncertain" never overturns: missing positive confidence = stay with maker.
"""


def resolve(maker_result: dict, checker_result: dict) -> dict:
    """
    Apply the §4 consensus table deterministically.

    Args:
        maker_result:  output of judge_groundedness — must have 'flagged' (bool).
                       May also have 'severity', 'confidence', 'reasoning'.
        checker_result: output of check_factual — must have 'verdict' (str).
                        Values: "true" | "false" | "uncertain".

    Returns:
        {
            flagged:      bool   — final verdict
            severity:     int    — 0 if overturned, else maker's severity
            action:       str    — one of the five action labels below
            disagreement: bool   — True when maker and checker give conflicting signals
        }

    Actions:
        pass              — both agree: not a hallucination
        overturn          — ungrounded-but-true: checker removes the flag (THE FIX)
        flag_confirmed    — ungrounded-and-false: consensus keeps flag (recall guard)
        keep_flag_uncertain — ungrounded but checker can't confirm truth: keep flag
        keep_grounded_log — grounded but checker says false: keep maker (no doc override) + log
    """
    maker_flagged: bool = bool(maker_result.get("flagged", False))
    maker_severity: int = int(maker_result.get("severity", 1))
    checker_verdict: str = str(checker_result.get("verdict", "uncertain")).lower()

    if checker_verdict not in ("true", "false", "uncertain"):
        checker_verdict = "uncertain"

    if not maker_flagged:
        # Maker says grounded — checker can log but cannot add a flag against the docs
        if checker_verdict == "false":
            # Cell: grounded + false — log for review; keep maker's not-flagged verdict
            return {
                "flagged": False,
                "severity": 0,
                "action": "keep_grounded_log",
                "disagreement": True,
            }
        # grounded + true | uncertain — everything agrees
        return {
            "flagged": False,
            "severity": 0,
            "action": "pass",
            "disagreement": False,
        }

    # Maker flagged (ungrounded)
    if checker_verdict == "true":
        # Ungrounded-but-true → THE FIX: overturn the flag
        return {
            "flagged": False,
            "severity": 0,
            "action": "overturn",
            "disagreement": True,
        }

    if checker_verdict == "false":
        # Ungrounded-and-false → real hallucination (recall guard case)
        return {
            "flagged": True,
            "severity": maker_severity,
            "action": "flag_confirmed",
            "disagreement": False,
        }

    # checker_verdict == "uncertain" — can't confirm truth; keep maker's flag
    return {
        "flagged": True,
        "severity": maker_severity,
        "action": "keep_flag_uncertain",
        "disagreement": False,
    }
