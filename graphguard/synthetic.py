"""Synthetic data in the *exact* Elliptic CSV layout, with planted fan-out / cycling / bridging.

Used for smoke tests and demos when the real Elliptic dataset is not present.
Real files expected in lake/raw/elliptic/:
    elliptic_txs_features.csv  (no header: txId, time_step, 165 features)
    elliptic_txs_edgelist.csv  (header: txId1,txId2)
    elliptic_txs_classes.csv   (header: txId,class  -> '1' illicit, '2' licit, 'unknown')
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

N_FEAT = 165


def generate(out_dir: Path, n_nodes: int = 20000, n_steps: int = 49, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = np.arange(100000, 100000 + n_nodes)
    step = np.sort(rng.integers(1, n_steps + 1, n_nodes))
    X = rng.normal(size=(n_nodes, N_FEAT)).astype("float32")
    label = np.full(n_nodes, 2, dtype=int)  # 2 = licit
    edges: list[tuple[int, int]] = []

    # background: sparse, mostly forward-in-time chains within a time step
    by_step = {s: np.where(step == s)[0] for s in range(1, n_steps + 1)}
    for s, idx in by_step.items():
        if len(idx) < 3:
            continue
        for _ in range(int(len(idx) * 1.1)):
            a, b = rng.choice(idx, 2, replace=False)
            edges.append((a, b) if a < b else (b, a))

    def plant_fanout(idx):
        hub = rng.choice(idx)
        for t in rng.choice(idx, min(len(idx) - 1, rng.integers(10, 25)), replace=False):
            if t != hub:
                edges.append((hub, t))
        return [hub]

    def plant_cycle(idx):
        ring = rng.choice(idx, rng.integers(3, 6), replace=False)
        for i in range(len(ring)):
            edges.append((ring[i], ring[(i + 1) % len(ring)]))
        return list(ring)

    def plant_bridge(idx):
        node = rng.choice(idx)
        far = rng.choice(np.arange(n_nodes), 6, replace=False)  # unrelated nodes -> other communities
        for f in far:
            edges.append((node, f) if rng.random() < 0.5 else (f, node))
        return [node]

    planters = [plant_fanout, plant_cycle, plant_bridge]
    for s, idx in by_step.items():
        if len(idx) < 30:
            continue
        for _ in range(rng.integers(3, 8)):
            for i in planters[rng.integers(len(planters))](idx):
                label[i] = 1

    # illicit nodes carry a (noisy) feature signal so that the classifier has something to learn
    ill = label == 1
    X[ill, :12] += rng.normal(1.2, 0.8, size=(ill.sum(), 12)).astype("float32")

    # ~77% unlabeled like the real dataset
    cls = np.where(rng.random(n_nodes) < 0.77, "unknown", label.astype(str))

    feats = pd.DataFrame(X)
    feats.insert(0, "time_step", step)
    feats.insert(0, "txId", ids)
    feats.to_csv(out_dir / "elliptic_txs_features.csv", header=False, index=False)
    pd.DataFrame({"txId1": ids[[a for a, _ in edges]], "txId2": ids[[b for _, b in edges]]}).drop_duplicates().to_csv(
        out_dir / "elliptic_txs_edgelist.csv", index=False
    )
    pd.DataFrame({"txId": ids, "class": cls}).to_csv(out_dir / "elliptic_txs_classes.csv", index=False)
    return out_dir
