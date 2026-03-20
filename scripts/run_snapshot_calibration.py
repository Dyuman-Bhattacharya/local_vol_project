from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from market_data.carry import apply_carry_to_snapshot_frame
from market_data.loaders import load_any
from market_data.yahoo_loader import fetch_option_chain_snapshot
from pipeline.calibration import (
    SnapshotCalibrationConfig,
    calibrate_option_snapshot,
    save_calibration_artifacts,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Calibrate an interpolated total-variance / Dupire local-vol surface from an option snapshot.")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--date", required=True, help="valuation date label YYYY-MM-DD")
    ap.add_argument("--out", required=True)
    ap.add_argument("--input_file", default=None, help="Optional local CSV/parquet snapshot; bypasses live fetch when provided")
    ap.add_argument("--option_type", choices=["call", "put"], default="call")
    ap.add_argument("--min_days", type=int, default=7)
    ap.add_argument("--max_days", type=int, default=730)
    ap.add_argument("--max_expiries", type=int, default=None)
    ap.add_argument("--risk_free_rate", type=float, default=None)
    ap.add_argument("--dividend_yield", type=float, default=None)
    ap.add_argument("--min_volume", type=float, default=25.0)
    ap.add_argument("--min_open_interest", type=float, default=200.0)
    ap.add_argument("--moneyness_min", type=float, default=0.90)
    ap.add_argument("--moneyness_max", type=float, default=1.10)
    ap.add_argument("--max_relative_spread", type=float, default=0.05)
    ap.add_argument("--iv_min", type=float, default=0.03)
    ap.add_argument("--iv_max", type=float, default=1.50)
    ap.add_argument("--min_points_per_slice", type=int, default=10)
    ap.add_argument("--n_strikes", type=int, default=45)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.input_file is not None:
        df_raw = load_any(args.input_file)
        df_raw = apply_carry_to_snapshot_frame(
            df_raw,
            ticker=str(args.ticker),
            valuation_date=str(args.date),
            risk_free_rate=args.risk_free_rate,
            dividend_yield=args.dividend_yield,
        )
    else:
        df_raw = fetch_option_chain_snapshot(
            args.ticker,
            valuation_date=args.date,
            option_type=str(args.option_type),
            min_days=int(args.min_days),
            max_days=int(args.max_days),
            max_expiries=args.max_expiries,
            risk_free_rate=args.risk_free_rate,
            dividend_yield=args.dividend_yield,
        )

    artifacts = calibrate_option_snapshot(
        df_raw,
        ticker=str(args.ticker),
        valuation_date=str(args.date),
        cfg=SnapshotCalibrationConfig(
            option_type=str(args.option_type),
            min_volume=float(args.min_volume),
            min_open_interest=float(args.min_open_interest),
            moneyness_min=float(args.moneyness_min),
            moneyness_max=float(args.moneyness_max),
            max_relative_spread=float(args.max_relative_spread),
            iv_min=float(args.iv_min),
            iv_max=float(args.iv_max),
            min_points_per_slice=int(args.min_points_per_slice),
            n_strikes=int(args.n_strikes),
        ),
    )
    save_calibration_artifacts(artifacts, out)

    print(f"Saved calibration artifacts to {out}")
    print("Summary:")
    for key, value in artifacts.summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
