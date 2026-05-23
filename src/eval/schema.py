from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProbeResult:
    """One row in the traces store. Matches SPEC schema exactly."""
    # Identity
    probe_id: str
    axis: str                     # hallucination | bias | safety
    model: str

    # Input / output
    prompt: str
    response: str

    # Scoring
    flagged: bool                  # True = failure (hallucination / bias / unsafe)
    severity: int                  # 1-5; 1=minor, 5=critical

    # Performance
    latency_ms: float
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int

    # Metadata
    judge_reasoning: str = ""
    turn: int = 1                  # conversation turn (for context-rot curve)
    extra: dict = field(default_factory=dict)

    # Auto-populated on insert
    id: Optional[int] = None
