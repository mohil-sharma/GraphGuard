"""Categorise flagged nodes into fan-out / rapid fund cycling / community-bridging (+ wash-trading style clusters)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import CFG, Config
from .graph import TxGraph

FAN_OUT, CYCLING, BRIDGING = "fan_out", "fund_cycling", "community_bridging"


@dataclass
class Evidence:
    tx_id: int
    risk: float
    types: list[str]
    primary: str | None
    facts: dict = field(default_factory=dict)
    subgraph_nodes: list[int] = field(default_factory=list)
    subgraph_edges: list[tuple[int, int]] = field(default_factory=list)


def classify(g: TxGraph, cfg: Config = CFG) -> np.ndarray:
    """(N,3) boolean matrix: fan-out, cycling, bridging."""
    m = g.meta
    fan = ((m.out_deg >= cfg.fanout_min_out_degree) & (m.fanout_ratio >= cfg.fanout_ratio_min)).values
    cyc = (m.in_cycle | (m.reciprocal > 0)).values
    bri = ((m.bridge_ratio >= cfg.bridge_ratio_min) & (m.cross_deg >= 2)).values
    return np.column_stack([fan, cyc, bri])


def evidence_for(g: TxGraph, i: int, risk: float, flags: np.ndarray, cfg: Config = CFG, hops: int = 1,
                 max_nodes: int = 40) -> Evidence:
    m = g.meta.iloc[i]
    types = [t for t, f in zip((FAN_OUT, CYCLING, BRIDGING), flags[i]) if f]
    # priority: cycling (strongest laundering signal) > fan-out > bridging
    primary = next((t for t in (CYCLING, FAN_OUT, BRIDGING) if t in types), None)
    nodes = g.neighbours(i, hops)
    if len(nodes) > max_nodes:  # keep the highest-degree neighbours
        deg = g.meta.iloc[nodes][["in_deg", "out_deg"]].sum(1).values
        nodes = np.unique(np.concatenate([[i], nodes[np.argsort(-deg)[:max_nodes - 1]]]))
    inset = np.isin(g.src, nodes) & np.isin(g.dst, nodes)
    facts = {
        "time_step": int(m.time_step), "in_degree": int(m.in_deg), "out_degree": int(m.out_deg),
        "fanout_ratio": round(float(m.fanout_ratio), 2), "reciprocal_edges": int(m.reciprocal),
        "on_directed_cycle": bool(m.in_cycle), "cycle_component_size": int(m.scc_size) if m.in_cycle else 0,
        "velocity_vs_step_mean": round(float(m.velocity), 2), "cross_community_neighbours": int(m.cross_deg),
        "bridge_ratio": round(float(m.bridge_ratio), 2), "community_id": int(m.community),
        "subgraph_size": int(len(nodes)),
        "neighbour_illicit_labels": int((g.label[nodes] == 1).sum() - (g.label[i] == 1)),
    }
    return Evidence(int(g.tx_ids[i]), float(risk), types, primary, facts,
                    [int(t) for t in g.tx_ids[nodes]],
                    [(int(g.tx_ids[a]), int(g.tx_ids[b])) for a, b in zip(g.src[inset], g.dst[inset])])
