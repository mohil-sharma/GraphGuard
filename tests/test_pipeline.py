"""End-to-end smoke test on tiny synthetic data; also checks Spark/pandas ETL parity."""
import numpy as np
import pandas as pd
import pytest

from graphguard import anomaly, etl, graph, rag, synthetic, train
from graphguard.config import Config


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("lake")
    c = Config(lake=root / "lake", artifacts=root / "art")
    c.epochs = 30
    c.ensure()
    synthetic.generate(c.raw / "elliptic", n_nodes=4000)
    return c


def test_etl_engines_agree(cfg):
    etl.run_pandas(cfg)
    a = pd.read_parquet(cfg.analytics / "graph_features.parquet").sort_values("txId").reset_index(drop=True)
    try:
        etl.run_spark(cfg)
    except Exception as e:  # Spark/Java unavailable
        pytest.skip(f"spark unavailable: {type(e).__name__}")
    b = pd.read_parquet(cfg.analytics / "graph_features.parquet").sort_values("txId").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False, atol=1e-6)


def test_graph_train_and_categorise(cfg):
    etl.run_pandas(cfg)
    g = graph.build(cfg)
    assert g.X.shape[0] == 4000 and np.isfinite(g.X).all()
    flags = anomaly.classify(g, cfg)
    assert flags[:, 1].any(), "planted cycles must be detected"
    m = train.run(cfg)
    assert set(m["models"]) == {"random_forest_baseline", "graphsage", "gcn"}


def test_rag_retrieves_cycle_typology(cfg):
    r = rag.Retriever(cfg.__class__(knowledge_dir=cfg.knowledge_dir))
    assert any(c.cid == "TYP-CYCLE" for c, _ in r.for_types(["fund_cycling"], 3))
