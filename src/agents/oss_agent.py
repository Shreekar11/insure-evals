import time
from dataclasses import dataclass, field

from src.agents.base import Agent, TurnMeta
from config import OSS_MODEL_ID, EVAL_TEMPERATURE, CHAT_MAX_TOKENS


@dataclass
class OSSAgent(Agent):
    """
    Qwen2.5-0.5B-Instruct running in-process via HuggingFace transformers.
    Model is loaded once on first use (lazy) to avoid blocking Gradio startup.
    """
    _model_id: str = OSS_MODEL_ID
    _tokenizer: object = field(default=None, repr=False)
    _model: object = field(default=None, repr=False)
    last_meta: TurnMeta = field(default_factory=TurnMeta)

    @property
    def name(self) -> str:
        return "Qwen2.5-0.5B (OSS)"

    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            torch_dtype=torch.float32,   # CPU-safe
            device_map="cpu",
            trust_remote_code=True,
        )
        self._model.eval()

    def chat(self, messages: list[dict]) -> str:
        self._load()
        import torch

        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(text, return_tensors="pt")
        input_len = inputs["input_ids"].shape[-1]

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=CHAT_MAX_TOKENS,
                do_sample=False,           # temperature=0 equivalent for eval
                temperature=None,
                top_p=None,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        new_ids = output_ids[0][input_len:]
        reply = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        completion_tokens = len(new_ids)
        self.last_meta = TurnMeta(
            latency_ms=elapsed_ms,
            prompt_tokens=input_len,
            completion_tokens=completion_tokens,
            cost_usd=0.0,    # in-process — no API cost
            model=self._model_id,
        )
        return reply
