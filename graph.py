"""Transaction-graph construction from the processed / analytics zones."""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .config import CFG, Config
from .etl import FEAT_COLS

ENG_COLS = ["in_deg", "out_deg", "fanout_ratio", "reciprocal", "velocity", "recency", "nbr_step_mean"]


@dataclass
class TxGraph:
    tx_ids: np.ndarray          # (N,) original txId
    time_step: np.ndarray       # (N,)
    label: np.ndarray           # (N,) -1 unknown / 0 licit / 1 illicit
    X: np.ndarray               # (N, F) standardised features
    feat_names: list[str]
    src: np.ndarray             # directed edges as node indices
    dst: np.ndarray
    A_dir: sp.csr_matrix
    in_cycle: np.ndarray        # bool: node lies on a directed cycle
    scc_size: np.ndarray
    community: np.ndarray
    bridge_ratio: np.ndarray    # share of neighbours living in a different community
    meta: pd.DataFrame          # per-node raw structural stats (unscaled) for explanations

    @property
    def n(self) -> int:
        return len(self.tx_ids)

    def index_of(self, tx_id: int) -> int:
        i = np.searchsorted(self._sorted_ids, tx_id)
        if i >= self.n or self._sorted_ids[i] != tx_id:
            raise KeyError(tx_id)
        return int(self._order[i])

    def __post_init__(self):
        self._order = np.argsort(self.tx_ids)
        self._sorted_ids = self.tx_ids[self._order]

    def neighbours(self, i: int, hops: int = 1) -> np.ndarray:
        A = (self.A_dir + self.A_dir.T).tocsr()
        seen, frontier = {i}, {i}
        for _ in range(hops):
            nxt: set[int] = set()
            for u in frontier:
                nxt.update(A.indices[A.indptr[u]:A.indptr[u + 1]].tolist())
            frontier = nxt - seen
            seen |= nxt
        return np.array(sorted(seen))


def build(cfg: Config = CFG, seed: int = 7) -> TxGraph:
    nodes = pd.read_parquet(cfg.processed / "nodes.parquet")
    edges = pd.read_parquet(cfg.processed / "edges.parquet")
    gf = pd.read_parquet(cfg.analytics / "graph_features.parquet")
    nodes = nodes.merge(gf, on="txId", how="left").sort_values("txId").reset_index(drop=True)
    idx = pd.Series(np.arange(len(nodes)), index=nodes.txId.values)
    src = idx.loc[edges.src.values].values
    dst = idx.loc[edges.dst.values].values
    n = len(nodes)

    A = sp.csr_matrix((np.ones(len(src), dtype="float32"), (src, dst)), shape=(n, n))
    n_scc, comp = connected_components(A, directed=True, connection="strong")
    scc_size = np.bincount(comp)[comp]
    in_cycle = scc_size >= 2

    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(zip(src.tolist(), dst.tolist()))
    comms = nx.community.louvain_communities(G, seed=seed)
    community = np.zeros(n, dtype=int)
    for c, members in enumerate(comms):
        community[list(members)] = c
    cross = (community[src] != community[dst]).astype("float32")
    und_deg = np.bincount(src, minlength=n) + np.bincount(dst, minlength=n)
    cross_deg = np.bincount(src, weights=cross, minlength=n) + np.bincount(dst, weights=cross, minlength=n)
    bridge_ratio = np.divide(cross_deg, und_deg, out=np.zeros(n), where=und_deg > 0)

    meta = nodes[["txId", "time_step", "label"] + ENG_COLS].copy()
    meta["in_cycle"] = in_cycle
    meta["scc_size"] = scc_size
    meta["community"] = community
    meta["bridge_ratio"] = bridge_ratio
    meta["cross_deg"] = cross_deg.astype(int)

    eng = np.column_stack([
        np.log1p(nodes.in_deg), np.log1p(nodes.out_deg), np.log1p(nodes.fanout_ratio), np.log1p(nodes.reciprocal),
        nodes.velocity, nodes.recency, nodes.nbr_step_mean, in_cycle.astype(float), bridge_ratio, np.log1p(cross_deg),
    ])
    eng_names = ["log_in", "log_out", "log_fanout", "log_recip", "velocity", "recency", "nbr_step",
                 "in_cycle", "bridge_ratio", "log_cross"]
    X = np.column_stack([nodes[FEAT_COLS].values, eng]).astype("float32")
    tr = nodes.time_step.values <= cfg.train_max_step
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6      # scaler fitted on the training window only
    X = (X - mu) / sd
    return TxGraph(nodes.txId.values, nodes.time_step.values, nodes.label.values, X, FEAT_COLS + eng_names,
                   src, dst, A, in_cycle, scc_size, community, bridge_ratio, meta)
