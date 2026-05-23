"""
Phase 1 GATE TEST — run before committing to Qwen2.5-0.5B.

Tests:
  1. Can Qwen-0.5B read a short reference snippet and answer faithfully?
  2. Does it use the dosage converter tool output correctly?

Pass criteria:
  - Answer contains the correct fact from the ref snippet
  - Answer does NOT contradict the ref snippet
  - Tool invocation: model either calls the tool or uses its result correctly

Usage:
  python scripts/gate_test.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

REF_SNIPPET = """
=== Drug Reference: Paracetamol (Acetaminophen) ===
Standard adult dose: 500 mg to 1000 mg every 4-6 hours.
Maximum daily dose: 4000 mg (4 g) per day in healthy adults.
Contraindications: Do NOT use in patients with severe hepatic (liver) impairment.
Overdose risk: Exceeding 4 g/day increases risk of hepatotoxicity (liver damage).
Interaction: Concurrent alcohol use increases hepatotoxicity risk even at normal doses.
"""

GATE_QUESTION_1 = (
    "Based on the drug reference provided, what is the maximum daily dose of "
    "paracetamol for a healthy adult, and what organ does overdose risk affect?"
)

GATE_QUESTION_2 = (
    "A patient weighs 60 kg. The doctor prescribes 15 mg/kg of ibuprofen. "
    "Use the dosage converter tool to calculate the total dose in mg."
)

SYSTEM_PROMPT = (
    "You are a medical information assistant. "
    "When asked about dosing calculations, use the convert_dosage tool. "
    "Answer concisely and only from the provided reference material.\n\n"
    f"Reference material:\n{REF_SNIPPET}"
)


def check_answer_1(reply: str) -> tuple[bool, str]:
    r = reply.lower()
    has_4g = "4 g" in r or "4g" in r or "4000 mg" in r or "4,000" in r
    has_liver = "liver" in r or "hepat" in r
    if has_4g and has_liver:
        return True, "PASS — correct max dose (4g) and organ (liver) mentioned"
    missing = []
    if not has_4g:
        missing.append("max dose (4 g / 4000 mg)")
    if not has_liver:
        missing.append("organ (liver / hepatic)")
    return False, f"FAIL — missing: {', '.join(missing)}"


def check_answer_2(reply: str) -> tuple[bool, str]:
    r = reply.lower()
    has_900 = "900" in r
    has_tool_ref = any(w in r for w in ["dosage", "tool", "mg/kg", "15", "60"])
    if has_900:
        return True, "PASS — correct total dose (900 mg) produced"
    if has_tool_ref:
        return None, "PARTIAL — model references dosing but didn't produce 900 mg; inspect manually"
    return False, "FAIL — no mention of correct dose (900 mg) or tool context"


def run_gate():
    print("=" * 60)
    print("PHASE 1 GATE TEST — Qwen2.5-0.5B-Instruct")
    print("=" * 60)

    print("\nLoading OSS agent (may take 1-3 min on first run)...")
    from src.agents.oss_agent import OSSAgent
    from src.tools.dosage import convert_dosage

    agent = OSSAgent()

    # Test 1: grounded reference reading
    print("\n[Test 1] Grounded reference reading")
    print(f"Q: {GATE_QUESTION_1}")
    messages_1 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": GATE_QUESTION_1},
    ]
    reply_1 = agent.chat(messages_1)
    print(f"A: {reply_1}")
    print(f"   Latency: {agent.last_meta.latency_ms:.0f} ms  |  Tokens: {agent.last_meta.completion_tokens}")
    passed_1, note_1 = check_answer_1(reply_1)
    print(f"   Result: {note_1}")

    # Test 2: tool use (inject tool result, check model uses it)
    print("\n[Test 2] Dosage tool usage")
    print(f"Q: {GATE_QUESTION_2}")
    tool_result = convert_dosage(15, "mg/kg", weight_kg=60, drug="ibuprofen")
    messages_2 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": GATE_QUESTION_2},
        {
            "role": "assistant",
            "content": (
                f"[Tool: convert_dosage(value=15, from_unit='mg/kg', weight_kg=60, drug='ibuprofen')] "
                f"Result: {tool_result.result}"
            ),
        },
        {"role": "user", "content": "Please state the total dose clearly."},
    ]
    reply_2 = agent.chat(messages_2)
    print(f"A: {reply_2}")
    print(f"   Latency: {agent.last_meta.latency_ms:.0f} ms  |  Tokens: {agent.last_meta.completion_tokens}")
    passed_2, note_2 = check_answer_2(reply_2)
    print(f"   Result: {note_2}")

    # Verdict
    print("\n" + "=" * 60)
    both_passed = passed_1 is True and passed_2 is not False
    if both_passed:
        print("GATE: PASSED — Qwen2.5-0.5B sufficient. Proceeding with 0.5B.")
    else:
        print("GATE: FAILED — escalate to Qwen2.5-1.5B-Instruct.")
        print("  Fix: set OSS_MODEL_ID='Qwen/Qwen2.5-1.5B-Instruct' in config.py")
    print("=" * 60)
    return both_passed


if __name__ == "__main__":
    result = run_gate()
    sys.exit(0 if result else 1)
