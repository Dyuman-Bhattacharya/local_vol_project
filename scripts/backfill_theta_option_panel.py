from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from market_data.yahoo_loader import fetch_underlying_history


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Backfill a Theta-based SPY option panel using historical 15:45 snapshots.")
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--start_date", required=True)
    ap.add_argument("--end_date", required=True)
    ap.add_argument("--snapshot_time", default="15:45")
    ap.add_argument("--timezone", default="America/New_York")
    ap.add_argument("--archive_root", default="data/archive")
    ap.add_argument("--panel_out", default="data/processed/spy_options_panel.parquet")
    ap.add_argument("--panel_manifest", default="data/processed/spy_options_panel_manifest.json")
    ap.add_argument("--min_days", type=int, default=1)
    ap.add_argument("--max_days", type=int, default=365)
    ap.add_argument("--max_expiries", type=int, default=None)
    ap.add_argument("--risk_free_rate", type=float, default=None)
    ap.add_argument("--dividend_yield", type=float, default=None)
    ap.add_argument("--skip_existing", action="store_true")
    return ap.parse_args()


def market_days(ticker: str, start_date: str, end_date: str) -> list[str]:
    hist = fetch_underlying_history(
        ticker,
        start=start_date,
        end=(pd.Timestamp(end_date) + pd.Timedelta(days=1)).date().isoformat(),
        interval="1d",
    )
    days = pd.to_datetime(hist["date"], errors="coerce").dropna().dt.date.astype(str).tolist()
    return [d for d in days if start_date <= d <= end_date]


def main() -> None:
    args = parse_args()
    py = sys.executable
    dates = market_days(args.ticker, args.start_date, args.end_date)
    if not dates:
        raise RuntimeError(f"No market days found between {args.start_date} and {args.end_date}")

    for d in dates:
        archive_path = PROJECT_ROOT / args.archive_root / args.ticker.upper() / d / "option_chain_snapshot.parquet"
        if args.skip_existing and archive_path.exists():
            print(f"Skipping existing snapshot for {d}", flush=True)
            continue
        cmd = [
            py,
            str(PROJECT_ROOT / "scripts" / "archive_option_snapshot.py"),
            "--provider",
            "theta",
            "--ticker",
            args.ticker,
            "--date",
            d,
            "--archive_root",
            str(PROJECT_ROOT / args.archive_root),
            "--panel_out",
            str(PROJECT_ROOT / args.panel_out),
            "--panel_manifest",
            str(PROJECT_ROOT / args.panel_manifest),
            "--timezone",
            args.timezone,
            "--snapshot_time",
            args.snapshot_time,
            "--min_days",
            str(args.min_days),
            "--max_days",
            str(args.max_days),
        ]
        if args.max_expiries is not None:
            cmd += ["--max_expiries", str(args.max_expiries)]
        if args.risk_free_rate is not None:
            cmd += ["--risk_free_rate", str(args.risk_free_rate)]
        if args.dividend_yield is not None:
            cmd += ["--dividend_yield", str(args.dividend_yield)]
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


if __name__ == "__main__":
    main()
