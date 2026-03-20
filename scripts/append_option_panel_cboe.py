from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from market_data.cboe_eod import CboeEODConfig, normalize_cboe_option_eod_summary
from market_data.panel_store import append_panel_rows, write_panel_manifest


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Append one new Cboe Option EOD Summary file into an existing dated panel.")
    ap.add_argument("--input", required=True, help="New Cboe EOD CSV/ZIP file")
    ap.add_argument("--panel", required=True, help="Existing panel CSV/parquet to append into")
    ap.add_argument("--snapshot", choices=["1545", "eod"], default="1545")
    ap.add_argument("--underlying", default="SPY")
    ap.add_argument("--risk_free_rate", type=float, default=None)
    ap.add_argument("--dividend_yield", type=float, default=None)
    ap.add_argument("--panel_manifest", default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    panel_path = Path(args.panel)

    raw = pd.read_csv(input_path, compression="infer")
    new_rows = normalize_cboe_option_eod_summary(
        raw,
        cfg=CboeEODConfig(
            snapshot=str(args.snapshot),
            risk_free_rate=args.risk_free_rate,
            dividend_yield=args.dividend_yield,
            underlying_filter=[args.underlying],
        ),
    )

    combined = append_panel_rows(panel_path, new_rows)
    if args.panel_manifest:
        write_panel_manifest(
            combined,
            args.panel_manifest,
            panel_path=panel_path,
            ticker=args.underlying,
            provider="cboe_option_eod_summary",
            snapshot_time_local="15:45" if args.snapshot == "1545" else "16:00",
            timezone="America/New_York",
            extra={"latest_append_file": str(input_path), "snapshot_kind": args.snapshot},
        )

    print(f"Panel rows: {len(combined)}")
    print(f"Appended from: {input_path}")
    print(f"Panel file: {panel_path}")


if __name__ == "__main__":
    main()
