#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_cmd(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run a one-off snapshot workflow: calibration, optional hedging backtest, optional report."
    )
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--date", required=True, help="valuation date label YYYY-MM-DD")
    ap.add_argument("--output_root", default="output/daily")
    ap.add_argument("--input_file", default=None, help="Optional local option snapshot CSV/parquet")
    ap.add_argument("--option_type", choices=["call", "put"], default="call")
    ap.add_argument("--risk_free_rate", type=float, default=None)
    ap.add_argument("--dividend_yield", type=float, default=None)

    ap.add_argument("--run_backtest", action="store_true")
    ap.add_argument("--backtest_mode", choices=["legacy_single_surface", "daily_market_panel"], default="legacy_single_surface")
    ap.add_argument("--panel_file", default=None, help="Historical option panel CSV/parquet for the robust daily market-panel hedge study")
    ap.add_argument("--run_pricing_validation", action="store_true")
    ap.add_argument("--history_csv", default=None)
    ap.add_argument("--history_start", default=None)
    ap.add_argument("--history_end", default=None)
    ap.add_argument("--K_mode", choices=["atm", "fixed_strike", "fixed_moneyness"], default="atm")
    ap.add_argument("--K_fixed", type=float, default=None)
    ap.add_argument("--K_mult", type=float, default=None)
    ap.add_argument("--T", type=float, default=0.2356164383561644)
    ap.add_argument("--step_days", type=int, default=1)
    ap.add_argument("--min_paths", type=int, default=5)
    ap.add_argument("--spot_band_pct", type=float, default=None)
    ap.add_argument("--bs_sigma", type=float, default=0.20)
    ap.add_argument("--tx_bps", type=float, default=10.0)
    ap.add_argument("--r", type=float, default=0.0)
    ap.add_argument("--q", type=float, default=0.0)

    ap.add_argument("--generate_report", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = PROJECT_ROOT / args.output_root / args.date
    output_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable

    calib_cmd = [
        py,
        str(PROJECT_ROOT / "scripts" / "run_snapshot_calibration.py"),
        "--ticker",
        args.ticker,
        "--date",
        args.date,
        "--out",
        str(output_dir),
        "--option_type",
        args.option_type,
    ]
    if args.input_file is not None:
        calib_cmd += ["--input_file", args.input_file]
    if args.risk_free_rate is not None:
        calib_cmd += ["--risk_free_rate", str(args.risk_free_rate)]
    if args.dividend_yield is not None:
        calib_cmd += ["--dividend_yield", str(args.dividend_yield)]
    run_cmd(calib_cmd)

    if args.run_backtest:
        if args.backtest_mode == "daily_market_panel":
            if args.panel_file is None:
                raise RuntimeError("--panel_file is required when --backtest_mode=daily_market_panel")
            backtest_cmd = [
                py,
                str(PROJECT_ROOT / "scripts" / "run_backtest_market_panel.py"),
                "--ticker",
                args.ticker,
                "--panel_file",
                args.panel_file,
                "--out",
                str(output_dir),
                "--option_type",
                args.option_type,
                "--strike_mode",
                args.K_mode,
                "--T",
                str(args.T),
                "--entry_step_days",
                str(args.step_days),
                "--tx_bps",
                str(args.tx_bps),
            ]
            if args.history_csv is not None:
                backtest_cmd += ["--history_csv", args.history_csv]
            if args.K_fixed is not None:
                backtest_cmd += ["--K_fixed", str(args.K_fixed)]
            if args.K_mult is not None:
                backtest_cmd += ["--K_mult", str(args.K_mult)]
        else:
            backtest_cmd = [
                py,
                str(PROJECT_ROOT / "scripts" / "run_backtest_history_windows.py"),
                "--ticker",
                args.ticker,
                "--calib_dir",
                str(output_dir),
                "--option_type",
                args.option_type,
                "--K_mode",
                args.K_mode,
                "--T",
                str(args.T),
                "--step_days",
                str(args.step_days),
                "--min_paths",
                str(args.min_paths),
                "--bs_sigma",
                str(args.bs_sigma),
                "--tx_bps",
                str(args.tx_bps),
                "--r",
                str(args.r),
                "--q",
                str(args.q),
            ]
            if args.history_csv is not None:
                backtest_cmd += ["--history_csv", args.history_csv]
            if args.history_start is not None:
                backtest_cmd += ["--history_start", args.history_start]
            if args.history_end is not None:
                backtest_cmd += ["--history_end", args.history_end]
            if args.K_fixed is not None:
                backtest_cmd += ["--K_fixed", str(args.K_fixed)]
            if args.K_mult is not None:
                backtest_cmd += ["--K_mult", str(args.K_mult)]
            if args.spot_band_pct is not None:
                backtest_cmd += ["--spot_band_pct", str(args.spot_band_pct)]
        run_cmd(backtest_cmd)

    if args.run_pricing_validation:
        pricing_cmd = [
            py,
            str(PROJECT_ROOT / "scripts" / "run_pricing_validation.py"),
            "--calib_dir",
            str(output_dir),
        ]
        run_cmd(pricing_cmd)

    if args.generate_report:
        report_cmd = [
            py,
            str(PROJECT_ROOT / "scripts" / "generate_report.py"),
            "--output-dir",
            str(output_dir),
        ]
        run_cmd(report_cmd)


if __name__ == "__main__":
    main()
