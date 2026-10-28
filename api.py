"""REST API:  uvicorn graphguard.api:app --port 8000"""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query

from .service import GraphGuard

app = FastAPI(title="GraphGuard", version="0.1.0",
              description="GNN anomaly detection on crypto transaction graphs with RAG-grounded LLM explanations")


@lru_cache(maxsize=1)
def svc() -> GraphGuard:
    return GraphGuard()


@app.get("/health")
def health():
    s = svc()
    return {"status": "ok", "nodes": s.g.n, "edges": int(len(s.g.src)), "arch": s.arch, "llm_provider": s.cfg.llm_provider}


@app.get("/metrics")
def metrics():
    return svc().metrics()


@app.get("/flagged")
def flagged(limit: int = Query(50, le=500), pattern: str | None = Query(None, pattern="^(fan_out|fund_cycling|community_bridging)$"),
            time_step: int | None = None):
    return svc().flagged(limit, pattern, time_step)


@app.get("/wallet/{tx_id}")
def wallet(tx_id: int, explain: bool = True, grounded: bool = True):
    try:
        return svc().wallet(tx_id, explain, grounded)
    except KeyError:
        raise HTTPException(404, f"txId {tx_id} not in graph")
