from config import MEMORY_MAX_TURNS


class ConversationMemory:
    """Rolling message buffer — keeps the last N user+assistant turn pairs."""

    def __init__(self, system_prompt: str = "", max_turns: int = MEMORY_MAX_TURNS):
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self._history: list[dict] = []

    def add(self, role: str, content: str):
        self._history.append({"role": role, "content": content})
        # Keep only the last max_turns pairs (2 messages per turn)
        if len(self._history) > self.max_turns * 2:
            self._history = self._history[-(self.max_turns * 2):]

    def messages(self) -> list[dict]:
        """Return the full message list with system prompt prepended."""
        if self.system_prompt:
            return [{"role": "system", "content": self.system_prompt}] + self._history
        return list(self._history)

    def clear(self):
        self._history = []

    def __len__(self) -> int:
        return len(self._history)
