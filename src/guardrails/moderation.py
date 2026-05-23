"""
Llama Guard 3 (meta-llama/llama-guard-3-8b) via OpenRouter.
Used as both runtime guardrail (in/out moderation) and the safety eval's independent second signal.
"""
from dataclasses import dataclass
from openai import OpenAI
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, SAFETY_CLASSIFIER

# Hazard categories Llama Guard 3 reports
HAZARD_LABELS = {
    "S1": "Violent Crimes",
    "S2": "Non-Violent Crimes",
    "S3": "Sex-Related Crimes",
    "S4": "Child Sexual Exploitation",
    "S5": "Defamation",
    "S6": "Specialized Advice",   # includes medical misinformation
    "S7": "Privacy",
    "S8": "Intellectual Property",
    "S9": "Indiscriminate Weapons",
    "S10": "Hate",
    "S11": "Suicide & Self-Harm",
    "S12": "Sexual Content",
    "S13": "Elections",
    "S14": "Code Interpreter Abuse",
}


@dataclass
class ModerationResult:
    safe: bool
    label: str          # "safe" or the hazard code e.g. "S6"
    category: str       # human-readable category
    raw_response: str


def _parse_llama_guard_response(text: str) -> tuple[bool, str, str]:
    """Parse Llama Guard 3 output: 'safe' or 'unsafe\nS<N>'."""
    text = text.strip().lower()
    if text.startswith("safe"):
        return True, "safe", "safe"
    # Extract hazard code
    lines = text.splitlines()
    code = ""
    for line in lines:
        line = line.strip().upper()
        if line.startswith("S") and line[1:].isdigit():
            code = line
            break
    category = HAZARD_LABELS.get(code, "Unknown hazard")
    return False, code or "unsafe", category


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    return _client


def moderate(user_message: str, assistant_response: str = "") -> ModerationResult:
    """
    Run Llama Guard 3 on the conversation turn.
    Pass assistant_response to check output; omit to check input only.
    """
    messages = [{"role": "user", "content": user_message}]
    if assistant_response:
        messages.append({"role": "assistant", "content": assistant_response})

    try:
        resp = _get_client().chat.completions.create(
            model=SAFETY_CLASSIFIER,
            messages=messages,
            max_tokens=20,
            temperature=0,
        )
        raw = resp.choices[0].message.content or "safe"
    except Exception as e:
        # Fail open — don't block on classifier error, but flag it
        return ModerationResult(safe=True, label="error", category=str(e), raw_response=str(e))

    safe, label, category = _parse_llama_guard_response(raw)
    return ModerationResult(safe=safe, label=label, category=category, raw_response=raw)


BLOCKED_REPLY = (
    "[Response blocked by safety guardrail. "
    "This content was flagged as potentially harmful and cannot be displayed.]"
)
