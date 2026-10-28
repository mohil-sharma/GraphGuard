"""Train / evaluate GCN, GraphSAGE and a Random-Forest baseline; write risk scores to the analytics zone."""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

from . import graph as G
from . import models as M
from .config import CFG, Config

log = logging.getLogger("graphguard.train")


def _prf(y, pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[1], zero_division=0)
    return {"precision": float(p[0]), "recall": float(r[0]), "f1": float(f[0])}


def fit_gnn(arch: str, g: G.TxGraph, cfg: Config):
    torch.manual_seed(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, norm = M.make(arch, g.X.shape[1], cfg.hidden, cfg.layers, cfg.dropout)
    model.to(dev)
    adj = M.to_torch(norm(g.A_dir), dev)
    x = torch.tensor(g.X, device=dev)
    y = torch.tensor(np.maximum(g.label, 0).astype("int64"), device=dev)
    train = torch.tensor((g.label >= 0) & (g.time_step <= cfg.train_max_step), device=dev)
    n_ill = int((y[train] == 1).sum())
    w = torch.tensor([1.0, (train.sum().item() - n_ill) / max(n_ill, 1)], device=dev).clamp(max=30.0)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for ep in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(x, adj)[train], y[train], weight=w)
        loss.backward()
        opt.step()
        if ep % 25 == 0 or ep == cfg.epochs - 1:
            log.info("%s epoch %3d loss %.4f", arch, ep, loss.item())
    model.eval()
    with torch.no_grad():
        prob = F.softmax(model(x, adj), 1)[:, 1].cpu().numpy()
        emb = model.embed(x, adj).cpu().numpy()
    return model, prob, emb


def run(cfg: Config = CFG, archs=("graphsage", "gcn")) -> dict:
    cfg.ensure()
    g = G.build(cfg)
    lab = g.label >= 0
    test = lab & (g.time_step > cfg.train_max_step)
    train = lab & (g.time_step <= cfg.train_max_step)
    if test.sum() == 0 or (g.label[test] == 1).sum() == 0:
        log.warning("no labelled illicit nodes in the test window - metrics will be zero")
    metrics: dict = {"n_nodes": g.n, "n_edges": int(len(g.src)), "n_labelled": int(lab.sum()),
                     "illicit_share_labelled": float((g.label[lab] == 1).mean()) if lab.any() else 0.0,
                     "split": f"train steps 1-{cfg.train_max_step}, test steps {cfg.train_max_step + 1}-{cfg.n_steps}",
                     "models": {}}
    scores = pd.DataFrame({"txId": g.tx_ids})

    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight="balanced", random_state=cfg.seed)
    rf.fit(g.X[train][:, :165], g.label[train])  # raw Elliptic features only, as in Weber et al.
    p_rf = rf.predict_proba(g.X[:, :165])[:, 1]
    metrics["models"]["random_forest_baseline"] = {**_prf(g.label[test], p_rf[test] >= 0.5), "seconds": time.time() - t0}
    scores["risk_random_forest"] = p_rf

    for arch in archs:
        t0 = time.time()
        model, prob, emb = fit_gnn(arch, g, cfg)
        metrics["models"][arch] = {**_prf(g.label[test], prob[test] >= cfg.risk_threshold), "seconds": time.time() - t0}
        scores[f"risk_{arch}"] = prob
        np.save(cfg.artifacts / f"embeddings_{arch}.npy", emb.astype("float16"))
        torch.save(model.state_dict(), cfg.artifacts / f"{arch}.pt")

    scores["time_step"] = g.time_step
    scores["label"] = g.label
    scores.to_parquet(cfg.analytics / "risk_scores.parquet", index=False)
    (cfg.artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("metrics: %s", json.dumps(metrics["models"], indent=1))
    return metrics
