from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from market_data.cboe_eod import CboeEODConfig, load_cboe_option_eod_summary
from market_data.panel_store import write_panel, write_panel_manifest


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a standardized dated option panel from Cboe Option EOD Summary files.")
    ap.add_argument("--input", nargs="+", required=True, help="One or more Cboe EOD CSV/ZIP files")
    ap.add_argument("--out", required=True, help="Output CSV or parquet path")
    ap.add_argument("--snapshot", choices=["1545", "eod"], default="1545")
    ap.add_argument("--underlying", default="SPY", help="Underlying symbol to keep, e.g. SPY")
    ap.add_argument("--risk_free_rate", type=float, default=None)
    ap.add_argument("--dividend_yield", type=float, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    panel = load_cboe_option_eod_summary(
        args.input,
        cfg=CboeEODConfig(
            snapshot=str(args.snapshot),
            risk_free_rate=args.risk_free_rate,
            dividend_yield=args.dividend_yield,
            underlying_filter=[args.underlying],
        ),
    )

    write_panel(panel, out_path)

    summary = {
        "rows": int(len(panel)),
        "dates": int(panel["date"].nunique()),
        "contracts": int(panel["contract_symbol"].nunique()),
        "underlyings": sorted(panel["underlying"].dropna().unique().tolist()),
        "snapshot_kind": args.snapshot,
        "source_files": [str(Path(p)) for p in args.input],
        "output_file": str(out_path),
    }
    (out_path.parent / f"{out_path.stem}_summary.json").write_text(json.dumps(summary, indent=2))
    write_panel_manifest(
        panel,
        out_path.parent / f"{out_path.stem}_manifest.json",
        panel_path=out_path,
        ticker=args.underlying,
        provider="cboe_option_eod_summary",
        snapshot_time_local="15:45" if args.snapshot == "1545" else "16:00",
        timezone="America/New_York",
        extra={"source_files": [str(Path(p)) for p in args.input], "snapshot_kind": args.snapshot},
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
