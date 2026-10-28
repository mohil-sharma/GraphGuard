"""Query layer shared by the REST API and the dashboard."""
from __future__ import annotations

import json
from functools import cached_property

import numpy as np
import pandas as pd

from . import anomaly, graph
from .config import CFG, Config
from .explain import Explainer


class GraphGuard:
    def __init__(self, cfg: Config = CFG, arch: str = "graphsage"):
        self.cfg, self.arch = cfg, arch
        self.g = graph.build(cfg)
        sc = pd.read_parquet(cfg.analytics / "risk_scores.parquet")
        assert (sc.txId.values == self.g.tx_ids).all(), "risk scores out of sync with graph - rerun `train`"
        self.risk = sc[f"risk_{arch}"].values
        self.flags = anomaly.classify(self.g, cfg)

    @cached_property
    def explainer(self) -> Explainer:
        return Explainer(self.cfg)

    def metrics(self) -> dict:
        p = self.cfg.artifacts / "metrics.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def flagged(self, limit: int = 50, min_type: str | None = None, time_step: int | None = None) -> list[dict]:
        m = self.risk >= self.cfg.risk_threshold
        if time_step is not None:
            m &= self.g.time_step == time_step
        if min_type:
            m &= self.flags[:, ("fan_out", "fund_cycling", "community_bridging").index(min_type)]
        idx = np.where(m)[0]
        idx = idx[np.argsort(-self.risk[idx])][:limit]
        names = ("fan_out", "fund_cycling", "community_bridging")
        return [{"tx_id": int(self.g.tx_ids[i]), "risk": float(self.risk[i]), "time_step": int(self.g.time_step[i]),
                 "patterns": [n for n, f in zip(names, self.flags[i]) if f], "known_label": int(self.g.label[i])} for i in idx]

    def wallet(self, tx_id: int, explain: bool = True, grounded: bool = True) -> dict:
        i = self.g.index_of(tx_id)
        ev = anomaly.evidence_for(self.g, i, self.risk[i], self.flags, self.cfg)
        out = {"tx_id": tx_id, "risk": float(self.risk[i]), "flagged": bool(self.risk[i] >= self.cfg.risk_threshold),
               "patterns": ev.types, "primary_pattern": ev.primary, "evidence": ev.facts,
               "subgraph": {"nodes": ev.subgraph_nodes, "edges": ev.subgraph_edges}}
        if explain:
            out["explanation"] = self.explainer.explain(ev, grounded=grounded)
        return out
