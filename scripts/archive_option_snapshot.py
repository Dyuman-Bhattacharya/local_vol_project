from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from market_data.loaders import load_any
from market_data.panel_store import append_panel_rows, write_panel_manifest
from market_data.theta_loader import fetch_theta_option_chain_snapshot
from market_data.yahoo_loader import fetch_option_chain_snapshot


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Archive a full daily option-chain snapshot and optionally append it to a dated panel.")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--provider", choices=["yfinance", "theta"], default="yfinance")
    ap.add_argument("--date", required=True, help="Snapshot label date YYYY-MM-DD")
    ap.add_argument("--archive_root", default="data/archive")
    ap.add_argument("--panel_out", default=None, help="Optional CSV/parquet panel file to append the snapshot into")
    ap.add_argument("--panel_manifest", default=None, help="Optional JSON manifest path to refresh after append")
    ap.add_argument("--timezone", default="America/New_York")
    ap.add_argument("--snapshot_time", default="15:45", help="Local clock time label for the archived snapshot")
    ap.add_argument("--input_file", default=None, help="Optional local CSV/parquet snapshot to archive instead of fetching live")
    ap.add_argument("--min_days", type=int, default=1)
    ap.add_argument("--max_days", type=int, default=730)
    ap.add_argument("--max_expiries", type=int, default=None)
    ap.add_argument("--risk_free_rate", type=float, default=None)
    ap.add_argument("--dividend_yield", type=float, default=None)
    return ap.parse_args()


def _fetch_both_types(args: argparse.Namespace) -> pd.DataFrame:
    if args.input_file:
        snapshot = load_any(args.input_file)
        if "provider" not in snapshot.columns:
            snapshot["provider"] = "file_snapshot_archive"
        snapshot["collection_date"] = str(args.date)
        snapshot["snapshot_time_local"] = str(args.snapshot_time)
        snapshot["timezone"] = str(args.timezone)
        return snapshot.sort_values(["date", "maturity_date", "option_type", "strike"]).reset_index(drop=True)

    if args.provider == "theta":
        out = fetch_theta_option_chain_snapshot(
            args.ticker,
            valuation_date=args.date,
            snapshot_time=str(args.snapshot_time),
            timezone=str(args.timezone),
            min_days=int(args.min_days),
            max_days=int(args.max_days),
            max_expiries=args.max_expiries,
            risk_free_rate=args.risk_free_rate,
            dividend_yield=args.dividend_yield,
        )
        out["provider"] = "theta_snapshot_archive"
    else:
        frames = []
        for option_type in ["call", "put"]:
            df = fetch_option_chain_snapshot(
                args.ticker,
                valuation_date=args.date,
                option_type=option_type,
                min_days=int(args.min_days),
                max_days=int(args.max_days),
                max_expiries=args.max_expiries,
                risk_free_rate=args.risk_free_rate,
                dividend_yield=args.dividend_yield,
            )
            frames.append(df)
        out = pd.concat(frames, ignore_index=True)
        out["provider"] = "yfinance_snapshot_archive"
    out["collection_date"] = str(args.date)
    out["snapshot_time_local"] = str(args.snapshot_time)
    out["timezone"] = str(args.timezone)
    return out.sort_values(["date", "maturity_date", "option_type", "strike"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    snapshot = _fetch_both_types(args)

    date_dir = Path(args.archive_root) / args.ticker.upper() / str(args.date)
    date_dir.mkdir(parents=True, exist_ok=True)

    snapshot_csv = date_dir / "option_chain_snapshot.csv"
    snapshot_parquet = date_dir / "option_chain_snapshot.parquet"
    snapshot.to_csv(snapshot_csv, index=False)
    snapshot.to_parquet(snapshot_parquet, index=False)

    metadata = {
        "ticker": args.ticker.upper(),
        "date": str(args.date),
        "rows": int(len(snapshot)),
        "timezone": str(args.timezone),
        "snapshot_time_local": str(args.snapshot_time),
        "input_file": str(args.input_file) if args.input_file else None,
        "provider": str(args.provider if args.input_file is None else "file"),
        "min_days": int(args.min_days),
        "max_days": int(args.max_days),
        "max_expiries": args.max_expiries,
        "risk_free_rate": args.risk_free_rate,
        "dividend_yield": args.dividend_yield,
        "resolved_risk_free_rate": float(pd.to_numeric(snapshot.get("risk_free_rate"), errors="coerce").dropna().median())
        if "risk_free_rate" in snapshot.columns and pd.to_numeric(snapshot.get("risk_free_rate"), errors="coerce").notna().any()
        else None,
        "resolved_dividend_yield": float(pd.to_numeric(snapshot.get("dividend_yield"), errors="coerce").dropna().median())
        if "dividend_yield" in snapshot.columns and pd.to_numeric(snapshot.get("dividend_yield"), errors="coerce").notna().any()
        else None,
        "risk_free_rate_source": str(snapshot["risk_free_rate_source"].mode().iloc[0]) if "risk_free_rate_source" in snapshot.columns and not snapshot["risk_free_rate_source"].dropna().empty else None,
        "dividend_yield_source": str(snapshot["dividend_yield_source"].mode().iloc[0]) if "dividend_yield_source" in snapshot.columns and not snapshot["dividend_yield_source"].dropna().empty else None,
        "snapshot_csv": str(snapshot_csv),
        "snapshot_parquet": str(snapshot_parquet),
    }
    (date_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    if args.panel_out:
        panel = append_panel_rows(Path(args.panel_out), snapshot)
        metadata["panel_out"] = str(args.panel_out)
        metadata["panel_rows_after_append"] = int(len(panel))
        if args.panel_manifest:
            manifest = write_panel_manifest(
                panel,
                args.panel_manifest,
                panel_path=args.panel_out,
                ticker=args.ticker,
                provider=(f"{args.provider}_snapshot_archive" if args.input_file is None else "file_snapshot_archive"),
                snapshot_time_local=str(args.snapshot_time),
                timezone=str(args.timezone),
                archive_root=args.archive_root,
                extra={
                    "latest_collection_date": str(args.date),
                    "collector_mode": "live" if args.input_file is None else "file_replay",
                },
            )
            metadata["panel_manifest"] = manifest

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
