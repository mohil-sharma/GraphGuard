"""Explanation-quality evaluation (grounded vs ungrounded) for the rubric in the review document.

Automatic proxies (fast, reproducible):
  citation_rate      share of sentences carrying a [ID] that exists in the retrieved passages
  bad_citation_rate  citations pointing to ids that were NOT retrieved (hallucinated sources)
  evidence_accuracy  share of numeric evidence values (degrees, cycle size...) mentioned that match the evidence JSON
Also writes a CSV for manual 1-5 rating of accuracy / grounding / actionability (the qualitative rubric).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .service import GraphGuard


def _score(res: dict) -> dict:
    text = res["explanation"]
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    ids = set(re.findall(r"\[([A-Za-z0-9_\-#]+)\]", text))
    valid = {c["id"] for c in res["citations"]}
    cited_sents = sum(bool(re.search(r"\[[A-Za-z0-9_\-#]+\]", s)) for s in sents)
    ev = res["evidence"]["facts"]
    true_vals = {str(v) for k, v in ev.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", text)) - {str(res["tx_id"]), "0"}
    nums = {n for n in nums if len(n) < 6}
    return {"citation_rate": cited_sents / max(len(sents), 1),
            "bad_citation_rate": len(ids - valid) / max(len(ids), 1) if ids else 0.0,
            "evidence_accuracy": len(nums & (true_vals | {f'{res["risk"]:.2f}'})) / max(len(nums), 1)}


def run(n: int = 30, out_csv: str = "artifacts/explanation_eval.csv") -> pd.DataFrame:
    gg = GraphGuard()
    rows = []
    for f in gg.flagged(limit=n):
        i = gg.g.index_of(f["tx_id"])
        from .anomaly import evidence_for
        ev = evidence_for(gg.g, i, gg.risk[i], gg.flags, gg.cfg)
        for grounded in (True, False):
            r = gg.explainer.explain(ev, grounded=grounded)
            rows.append({"tx_id": f["tx_id"], "grounded": grounded, "patterns": ",".join(ev.types), **_score(r),
                         "explanation": r["explanation"], "manual_accuracy_1to5": "", "manual_grounding_1to5": "",
                         "manual_actionability_1to5": ""})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(df.groupby("grounded")[["citation_rate", "bad_citation_rate", "evidence_accuracy"]].mean().round(3))
    return df
