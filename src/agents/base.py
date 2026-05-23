from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TurnMeta:
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""


class Agent(ABC):
    """Shared interface for OSS and frontier chat agents."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def chat(self, messages: list[dict]) -> str:
        """Accept an OpenAI-style message list, return the assistant reply."""
        ...

    # Populated after each chat() call
    last_meta: TurnMeta = field(default_factory=TurnMeta)
