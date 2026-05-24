import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
REFS_DIR = DATA_DIR / "refs"
PROBES_DIR = DATA_DIR / "probes"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
TRACES_DB = RESULTS_DIR / "traces.db"
SUMMARY_JSON = RESULTS_DIR / "summary.json"

# ── API keys (never hardcode — loaded from env / HF Space Secrets) ─────────
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
# Accept both GOOGLE_API_KEY and GEMINI_API_KEY (user may set either)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ── Model IDs ─────────────────────────────────────────────────────────────
OSS_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"          # transformers in-process
FRONTIER_MODEL = "google/gemini-2.0-flash-001"         # via OpenRouter
JUDGE_MODEL = "openai/gpt-4o-mini"                     # via OpenRouter
SAFETY_CLASSIFIER = "meta-llama/llama-guard-4-12b"     # via OpenRouter (guard-3-8b broken at Cloudflare provider, code 8001)

# ── Eval settings ─────────────────────────────────────────────────────────
EVAL_TEMPERATURE = 0          # deterministic — pinned for reproducibility
EVAL_MAX_TOKENS = 512
CHAT_MAX_TOKENS = 1024
MEMORY_MAX_TURNS = 10         # rolling buffer: keep last N turns
RAG_TOP_K = 3                 # top-k chunks returned by RAG tool

# ── Underwriting (illustrative — all constants are placeholders) ───────────
BASE_PREMIUM_USD = 10_000     # placeholder: pending real loss data
DOMAIN_RISK_MEDICAL = 1.5     # placeholder: medical domain risk multiplier
