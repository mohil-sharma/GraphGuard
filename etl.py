"""Distributed preprocessing: raw zone -> processed zone -> analytics zone.

Engine "spark"  : PySpark implementation (the big-data path from the review document).
Engine "pandas" : functionally identical single-machine fallback (used if Spark/Java is unavailable).

Outputs
    processed/nodes.parquet     txId, time_step, label(-1 unknown / 0 licit / 1 illicit), f0..f164
    processed/edges.parquet     src, dst   (validated, de-duplicated)
    analytics/graph_features.parquet
                                txId, in_deg, out_deg, fanout_ratio, reciprocal, velocity, recency, nbr_step_mean
    analytics/dq_report.json    schema-validation / data-quality counters
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from .config import CFG, Config

log = logging.getLogger("graphguard.etl")
N_FEAT = 165
FEAT_COLS = [f"f{i}" for i in range(N_FEAT)]


def _raw_dir(cfg: Config) -> Path:
    return cfg.raw / "elliptic"


def _check_raw(cfg: Config) -> Path:
    d = _raw_dir(cfg)
    missing = [f for f in ("elliptic_txs_features.csv", "elliptic_txs_edgelist.csv", "elliptic_txs_classes.csv") if not (d / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {d}. Download the Elliptic dataset (see README) or run `python -m graphguard.cli demo-data`."
        )
    return d


# --------------------------------------------------------------------------- pandas engine
def run_pandas(cfg: Config = CFG) -> dict:
    d = _check_raw(cfg)
    feats = pd.read_csv(d / "elliptic_txs_features.csv", header=None)
    feats.columns = ["txId", "time_step"] + FEAT_COLS
    cls = pd.read_csv(d / "elliptic_txs_classes.csv")
    edges = pd.read_csv(d / "elliptic_txs_edgelist.csv")
    edges.columns = ["src", "dst"]

    n_raw_edges = len(edges)
    edges = edges.drop_duplicates()
    valid = set(feats.txId)
    edges = edges[edges.src.isin(valid) & edges.dst.isin(valid) & (edges.src != edges.dst)]

    cls["label"] = cls["class"].astype(str).map({"1": 1, "2": 0}).fillna(-1).astype(int)
    nodes = feats.merge(cls[["txId", "label"]], on="txId", how="left").fillna({"label": -1})
    nodes["label"] = nodes["label"].astype(int)

    step = nodes.set_index("txId")["time_step"]
    out_deg = edges.groupby("src").size().rename("out_deg")
    in_deg = edges.groupby("dst").size().rename("in_deg")
    g = nodes[["txId", "time_step"]].set_index("txId").join([in_deg, out_deg]).fillna(0)
    g["fanout_ratio"] = g.out_deg / (g.in_deg + 1)
    rev = edges.rename(columns={"src": "dst", "dst": "src"})
    recip = edges.merge(rev, on=["src", "dst"]).groupby("src").size().rename("reciprocal")
    g = g.join(recip).fillna({"reciprocal": 0})
    g["deg"] = g.in_deg + g.out_deg
    g["velocity"] = g.deg / (g.groupby("time_step").deg.transform("mean") + 1e-9)
    g["recency"] = g.time_step.max() - g.time_step
    nbr = pd.concat([edges.assign(n=edges.dst.map(step)).rename(columns={"src": "txId"})[["txId", "n"]],
                     edges.assign(n=edges.src.map(step)).rename(columns={"dst": "txId"})[["txId", "n"]]])
    g = g.join(nbr.groupby("txId").n.mean().rename("nbr_step_mean"))
    g["nbr_step_mean"] = g.nbr_step_mean.fillna(g.time_step)
    g = g.drop(columns=["deg", "time_step"]).reset_index()

    return _write(cfg, nodes, edges, g, {"engine": "pandas", "raw_edges": n_raw_edges, "kept_edges": len(edges),
                                          "nodes": len(nodes), "dropped_edges": n_raw_edges - len(edges)})


# --------------------------------------------------------------------------- spark engine

def _ensure_java() -> None:
    """Spark needs Java 8/11/17/21. Prefer a project-local JDK in .tools/ if it has a binary for this OS."""
    from .config import ROOT
    exe = "java.exe" if sys.platform.startswith("win") else "java"
    local = [p for p in sorted((ROOT / ".tools").glob("jdk-17*")) if (p / "bin" / exe).exists()]
    if local:
        os.environ["JAVA_HOME"] = str(local[-1])
        os.environ["PATH"] = str(local[-1] / "bin") + os.pathsep + os.environ.get("PATH", "")
    elif not Path(os.environ.get("JAVA_HOME", "/nonexistent")).exists():
        os.environ.pop("JAVA_HOME", None)  # stale JAVA_HOME breaks Spark's launcher

def run_spark(cfg: Config = CFG) -> dict:
    _ensure_java()
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    from pyspark.sql import SparkSession, Window
    from pyspark.sql import functions as F
    from pyspark.sql import types as T

    d = _check_raw(cfg)
    spark = (SparkSession.builder.master("local[*]").appName("GraphGuard-ETL")
             .config("spark.sql.shuffle.partitions", "16").config("spark.ui.enabled", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    schema = T.StructType([T.StructField("txId", T.LongType()), T.StructField("time_step", T.IntegerType())] +
                          [T.StructField(c, T.FloatType()) for c in FEAT_COLS])
    feats = spark.read.csv(str(d / "elliptic_txs_features.csv"), header=False, schema=schema)
    cls = (spark.read.csv(str(d / "elliptic_txs_classes.csv"), header=True)
           .select(F.col("txId").cast("long").alias("txId"),
                   F.when(F.col("class") == "1", 1).when(F.col("class") == "2", 0).otherwise(-1).alias("label")))
    raw_edges = (spark.read.csv(str(d / "elliptic_txs_edgelist.csv"), header=True)
                 .select(F.col("txId1").cast("long").alias("src"), F.col("txId2").cast("long").alias("dst")))
    n_raw = raw_edges.count()

    ids = feats.select("txId")
    edges = (raw_edges.dropDuplicates().filter("src != dst")
             .join(ids.withColumnRenamed("txId", "src"), "src")
             .join(ids.withColumnRenamed("txId", "dst"), "dst").select("src", "dst").cache())
    nodes = feats.join(cls, "txId", "left").fillna({"label": -1}).cache()

    out_deg = edges.groupBy("src").agg(F.count("*").alias("out_deg")).withColumnRenamed("src", "txId")
    in_deg = edges.groupBy("dst").agg(F.count("*").alias("in_deg")).withColumnRenamed("dst", "txId")
    rev = edges.select(F.col("dst").alias("src"), F.col("src").alias("dst"))
    recip = (edges.join(rev, ["src", "dst"]).groupBy("src").agg(F.count("*").alias("reciprocal"))
             .withColumnRenamed("src", "txId"))
    step = nodes.select("txId", "time_step")
    nbr = (edges.join(step.withColumnRenamed("txId", "dst").withColumnRenamed("time_step", "n"), "dst")
           .select(F.col("src").alias("txId"), "n")
           .unionByName(edges.join(step.withColumnRenamed("txId", "src").withColumnRenamed("time_step", "n"), "src")
                        .select(F.col("dst").alias("txId"), "n"))
           .groupBy("txId").agg(F.avg("n").alias("nbr_step_mean")))

    g = (step.join(in_deg, "txId", "left").join(out_deg, "txId", "left").join(recip, "txId", "left")
         .join(nbr, "txId", "left")
         .fillna({"in_deg": 0, "out_deg": 0, "reciprocal": 0}))
    w = Window.partitionBy("time_step")
    g = (g.withColumn("deg", F.col("in_deg") + F.col("out_deg"))
         .withColumn("fanout_ratio", F.col("out_deg") / (F.col("in_deg") + 1))
         .withColumn("velocity", F.col("deg") / (F.avg("deg").over(w) + 1e-9))
         .withColumn("recency", F.max("time_step").over(Window.partitionBy()) - F.col("time_step"))
         .withColumn("nbr_step_mean", F.coalesce("nbr_step_mean", F.col("time_step").cast("double")))
         .select("txId", "in_deg", "out_deg", "fanout_ratio", "reciprocal", "velocity", "recency", "nbr_step_mean"))

    dq = {"engine": "spark", "raw_edges": n_raw, "kept_edges": edges.count(), "nodes": nodes.count()}
    dq["dropped_edges"] = dq["raw_edges"] - dq["kept_edges"]
    # Spark -> parquet zones. Parquet writes on Windows need winutils; fall back to a driver-side write.
    try:
        cfg.processed.mkdir(parents=True, exist_ok=True)
        cfg.analytics.mkdir(parents=True, exist_ok=True)
        nodes.write.mode("overwrite").parquet(str(cfg.processed / "nodes_spark"))
        edges.write.mode("overwrite").parquet(str(cfg.processed / "edges_spark"))
        g.write.mode("overwrite").parquet(str(cfg.analytics / "graph_features_spark"))
        dq["spark_parquet_write"] = True
    except Exception as e:  # noqa: BLE001
        log.warning("native Spark parquet write failed (%s); using driver-side write", type(e).__name__)
        dq["spark_parquet_write"] = False
    out = _write(cfg, nodes.toPandas(), edges.toPandas(), g.toPandas(), dq)
    spark.stop()
    return out


def _write(cfg: Config, nodes: pd.DataFrame, edges: pd.DataFrame, g: pd.DataFrame, dq: dict) -> dict:
    cfg.ensure()
    nodes.to_parquet(cfg.processed / "nodes.parquet", index=False)
    edges.to_parquet(cfg.processed / "edges.parquet", index=False)
    g.to_parquet(cfg.analytics / "graph_features.parquet", index=False)
    (cfg.analytics / "dq_report.json").write_text(json.dumps(dq, indent=2, default=int))
    log.info("ETL done: %s", dq)
    return dq


def run(engine: str = "spark", cfg: Config = CFG) -> dict:
    if engine == "spark":
        try:
            return run_spark(cfg)
        except FileNotFoundError:
            raise
        except Exception as e:  # noqa: BLE001
            log.warning("Spark engine failed (%s: %s) -> falling back to pandas", type(e).__name__, str(e)[:200])
    return run_pandas(cfg)
