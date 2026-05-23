"""
Simple keyword-based RAG over the 5 medical reference docs.
No embedding model needed — TF-IDF style scoring keeps this CPU-zero-cost.
"""
import re
import math
from pathlib import Path
from collections import Counter
from config import REFS_DIR, RAG_TOP_K


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class RefDoc:
    def __init__(self, path: Path):
        self.path = path
        self.name = path.stem
        self.text = path.read_text(encoding="utf-8")
        # Split into ~200-word chunks with 50-word overlap
        words = self.text.split()
        self.chunks: list[str] = []
        step = 150
        size = 200
        for i in range(0, max(1, len(words) - size + step), step):
            self.chunks.append(" ".join(words[i : i + size]))


class RAGRetriever:
    """
    Keyword (BM25-lite) retriever over the medical reference docs.
    Exposes retrieve() as the RAG tool the agents call.
    """

    def __init__(self, refs_dir: Path = REFS_DIR):
        self._docs = [RefDoc(p) for p in sorted(refs_dir.glob("*.md"))]
        if not self._docs:
            raise FileNotFoundError(f"No .md files found in {refs_dir}")
        self._all_chunks: list[tuple[str, str]] = []  # (doc_name, chunk)
        for doc in self._docs:
            for chunk in doc.chunks:
                self._all_chunks.append((doc.name, chunk))
        self._build_idf()

    def _build_idf(self):
        N = len(self._all_chunks)
        df: Counter = Counter()
        for _, chunk in self._all_chunks:
            for tok in set(_tokenize(chunk)):
                df[tok] += 1
        self._idf = {tok: math.log((N + 1) / (cnt + 1)) + 1 for tok, cnt in df.items()}

    def _score(self, query_tokens: list[str], chunk: str) -> float:
        chunk_tokens = _tokenize(chunk)
        freq = Counter(chunk_tokens)
        doc_len = len(chunk_tokens)
        avg_len = 150
        k1, b = 1.5, 0.75
        score = 0.0
        for tok in query_tokens:
            if tok not in self._idf:
                continue
            tf = freq.get(tok, 0)
            bm25 = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_len))
            score += self._idf[tok] * bm25
        return score

    def retrieve(self, query: str, top_k: int = RAG_TOP_K) -> list[dict]:
        """Return top_k most relevant chunks with their source doc names."""
        query_tokens = _tokenize(query)
        scored = [
            (self._score(query_tokens, chunk), doc_name, chunk)
            for doc_name, chunk in self._all_chunks
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"doc": doc_name, "chunk": chunk, "score": round(score, 3)}
            for score, doc_name, chunk in scored[:top_k]
        ]

    def format_context(self, query: str, top_k: int = RAG_TOP_K) -> str:
        """Formatted context block ready to inject into a system/user message."""
        hits = self.retrieve(query, top_k)
        if not hits:
            return ""
        parts = ["[Retrieved reference material]"]
        for h in hits:
            parts.append(f"\n--- {h['doc']} ---\n{h['chunk']}")
        return "\n".join(parts)

    @property
    def doc_names(self) -> list[str]:
        return [d.name for d in self._docs]


# Module-level singleton — lazy-loaded on first use
_retriever: RAGRetriever | None = None


def get_retriever() -> RAGRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever
