"""Command line: ``clv download`` and ``clv run``."""

from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="clv", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download", help="download and cache the UCI Online Retail data")
    dl.add_argument("--data-home", default="data")
    run = sub.add_parser("run", help="full analysis -> reports/metrics.json + figures")
    run.add_argument("--data-home", default="data")
    run.add_argument("--out", default="reports")
    args = parser.parse_args(argv)

    if args.command == "download":
        from clv.data import load_online_retail

        df = load_online_retail(args.data_home)
        print(f"{len(df):,} invoice lines cached in {args.data_home}/")
    else:
        from clv.pipeline import run as run_pipeline

        m = run_pipeline(args.data_home, args.out)
        main_split = m["validation"]["main"]
        print(json.dumps({
            "holdout_transactions": main_split["holdout_transactions"]["bgnbd"],
            "top_decile_revenue_capture": main_split["top_decile_revenue_capture"],
            "runtime_seconds": m["runtime_seconds"],
        }, indent=2))


if __name__ == "__main__":
    main()
