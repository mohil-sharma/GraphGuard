"""python -m graphguard.cli <command>   (see README)"""
from __future__ import annotations

import argparse
import json
import logging

from .config import CFG


def main(argv=None):
    ap = argparse.ArgumentParser(prog="graphguard")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo-data", help="generate synthetic Elliptic-format data into the raw zone")
    d.add_argument("--nodes", type=int, default=20000)
    e = sub.add_parser("etl", help="raw -> processed/analytics zones")
    e.add_argument("--engine", choices=["spark", "pandas"], default="spark")
    t = sub.add_parser("train", help="train GCN / GraphSAGE / RF baseline")
    t.add_argument("--epochs", type=int)
    sub.add_parser("explain-eval", help="grounded vs ungrounded explanation evaluation")
    x = sub.add_parser("explain", help="explain one transaction")
    x.add_argument("tx_id", type=int)
    x.add_argument("--ungrounded", action="store_true")
    a = sub.add_parser("all", help="etl + train (+ explanation eval)")
    a.add_argument("--engine", choices=["spark", "pandas"], default="spark")
    sub.add_parser("flagged", help="list top flagged transactions")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    CFG.ensure()

    if args.cmd == "demo-data":
        from .synthetic import generate
        print("wrote", generate(CFG.raw / "elliptic", n_nodes=args.nodes))
    elif args.cmd in ("etl", "all"):
        from . import etl
        print(json.dumps(etl.run(args.engine), indent=2))
    if args.cmd in ("train", "all"):
        from . import train
        if getattr(args, "epochs", None):
            CFG.epochs = args.epochs
        print(json.dumps(train.run()["models"], indent=2))
    if args.cmd in ("explain-eval", "all"):
        from . import eval_explanations
        eval_explanations.run()
    if args.cmd == "flagged":
        from .service import GraphGuard
        for r in GraphGuard().flagged(20):
            print(r)
    if args.cmd == "explain":
        from .service import GraphGuard
        print(json.dumps(GraphGuard().wallet(args.tx_id, grounded=not args.ungrounded), indent=2, default=str))


if __name__ == "__main__":
    main()
