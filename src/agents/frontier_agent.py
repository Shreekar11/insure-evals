import time
from dataclasses import dataclass, field

from openai import OpenAI

from src.agents.base import Agent, TurnMeta
from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    FRONTIER_MODEL,
    EVAL_TEMPERATURE,
    CHAT_MAX_TOKENS,
)

# Pricing per million tokens (approximate; OpenRouter may vary)
_INPUT_COST_PER_M = 0.10   # Gemini 2.0 Flash approximate
_OUTPUT_COST_PER_M = 0.40


def _cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens * _INPUT_COST_PER_M / 1_000_000
        + completion_tokens * _OUTPUT_COST_PER_M / 1_000_000
    )


@dataclass
class FrontierAgent(Agent):
    """
    Gemini 2.0 Flash via OpenRouter (OpenAI-compatible client).
    Temperature is configurable — use EVAL_TEMPERATURE=0 for eval runs.
    """
    temperature: float = EVAL_TEMPERATURE
    max_tokens: int = CHAT_MAX_TOKENS
    last_meta: TurnMeta = field(default_factory=TurnMeta)
    _client: OpenAI = field(default=None, repr=False)

    def __post_init__(self):
        self._client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )

    @property
    def name(self) -> str:
        return "Gemini 2.0 Flash (Frontier)"

    def chat(self, messages: list[dict]) -> str:
        t0 = time.perf_counter()
        response = self._client.chat.completions.create(
            model=FRONTIER_MODEL,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        reply = response.choices[0].message.content or ""
        usage = response.usage
        prompt_tok = usage.prompt_tokens if usage else 0
        completion_tok = usage.completion_tokens if usage else 0

        self.last_meta = TurnMeta(
            latency_ms=elapsed_ms,
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
            cost_usd=_cost(prompt_tok, completion_tok),
            model=FRONTIER_MODEL,
        )
        return reply
