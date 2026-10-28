"""Retrieval over the fraud-typology / regulatory corpus (TF-IDF; swap for dense embeddings if desired)."""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .config import CFG, Config

# Map detected anomaly types to typology ids so retrieval is anchored even for terse queries.
TYPE_QUERY = {
    "fan_out": "fan-out dispersal many recipients high out-degree splitting peel chain structuring layering",
    "fund_cycling": "cyclical fund movement round-tripping layering wash trading cycle reciprocal transfers",
    "community_bridging": "community bridging cross-cluster intermediary rapid movement between wallets mixer",
}


@dataclass
class Chunk:
    cid: str
    source: str
    text: str


def _read_pdf(p: Path) -> str:
    try:
        from pypdf import PdfReader
        return "\n".join(pg.extract_text() or "" for pg in PdfReader(str(p)).pages)
    except Exception:  # noqa: BLE001
        try:
            return subprocess.run(["pdftotext", "-layout", str(p), "-"], capture_output=True, text=True, check=True).stdout
        except Exception:  # noqa: BLE001
            return ""


def load_chunks(cfg: Config = CFG, max_chars: int = 900) -> list[Chunk]:
    chunks: list[Chunk] = []
    for p in sorted(cfg.knowledge_dir.glob("*")):
        if p.suffix.lower() not in {".md", ".txt", ".pdf"}:
            continue
        text = _read_pdf(p) if p.suffix.lower() == ".pdf" else p.read_text(encoding="utf-8", errors="ignore")
        if p.suffix.lower() == ".md" and re.search(r"^## \[", text, re.M):
            for m in re.finditer(r"^## \[([^\]]+)\]\s*(.*?)\n(.*?)(?=^## \[|\Z)", text, re.M | re.S):
                chunks.append(Chunk(m.group(1), p.name, f"{m.group(2)}. {' '.join(m.group(3).split())}"))
            continue
        buf, n = "", 0
        for para in re.split(r"\n\s*\n", text):
            para = " ".join(para.split())
            if not para:
                continue
            if len(buf) + len(para) > max_chars and buf:
                chunks.append(Chunk(f"{p.stem}#{n}", p.name, buf)); n += 1; buf = ""
            buf = f"{buf} {para}".strip()
        if buf:
            chunks.append(Chunk(f"{p.stem}#{n}", p.name, buf))
    return chunks


class Retriever:
    def __init__(self, cfg: Config = CFG):
        self.chunks = load_chunks(cfg)
        if not self.chunks:
            raise RuntimeError(f"knowledge base empty: add files to {cfg.knowledge_dir}")
        self.vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        self.M = self.vec.fit_transform([c.text for c in self.chunks])

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        sims = linear_kernel(self.vec.transform([query]), self.M).ravel()
        order = sims.argsort()[::-1][:k]
        return [(self.chunks[i], float(sims[i])) for i in order if sims[i] > 0]

    def for_types(self, types: list[str], k: int = 4) -> list[tuple[Chunk, float]]:
        q = " ".join(TYPE_QUERY[t] for t in types) or "suspicious transaction red flag indicator"
        return self.search(q, k)
