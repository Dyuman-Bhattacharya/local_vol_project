from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from hedging.market_panel_backtest import (
    build_daily_calibration_cache,
    build_history_from_panel,
    load_underlying_history_csv,
    run_daily_market_panel_backtest,
    select_listed_contracts,
    standardize_option_panel,
)
from hedging.metrics import summarize_market_panel_backtest
from hedging.transaction_costs import TransactionCostModel
from market_data.loaders import load_any
from pipeline.calibration import SnapshotCalibrationConfig
from pricing.pde_solver import PDEConfig


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run a daily recalibrated historical hedge study on a listed option panel.")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--panel_file", required=True, help="CSV/parquet option panel with one snapshot per date")
    ap.add_argument("--history_csv", default=None, help="Optional underlying history CSV with date and close/S columns")
    ap.add_argument("--out", required=True)
    ap.add_argument("--option_type", choices=["call", "put"], default="call")
    ap.add_argument("--strike_mode", choices=["atm", "fixed_strike", "fixed_moneyness"], default="atm")
    ap.add_argument("--K_fixed", type=float, default=None)
    ap.add_argument("--K_mult", type=float, default=None)
    ap.add_argument("--T", type=float, required=True, help="Target option maturity in years")
    ap.add_argument("--entry_step_days", type=int, default=5)
    ap.add_argument("--max_contracts", type=int, default=None)
    ap.add_argument("--contract_moneyness_band", type=float, default=0.02)
    ap.add_argument("--contract_max_relative_spread", type=float, default=0.05)
    ap.add_argument("--contract_min_volume", type=float, default=25.0)
    ap.add_argument("--contract_min_open_interest", type=float, default=200.0)
    ap.add_argument("--tx_bps", type=float, default=10.0)
    ap.add_argument("--pde_space", type=int, default=120)
    ap.add_argument("--pde_time", type=int, default=100)
    ap.add_argument("--min_volume", type=float, default=25.0)
    ap.add_argument("--min_open_interest", type=float, default=200.0)
    ap.add_argument("--moneyness_min", type=float, default=0.90)
    ap.add_argument("--moneyness_max", type=float, default=1.10)
    ap.add_argument("--max_relative_spread", type=float, default=0.05)
    ap.add_argument("--iv_min", type=float, default=0.03)
    ap.add_argument("--iv_max", type=float, default=1.50)
    ap.add_argument("--min_points_per_slice", type=int, default=10)
    ap.add_argument("--n_strikes", type=int, default=45)
    ap.add_argument("--min_panel_dates", type=int, default=2)
    ap.add_argument("--save_daily_calibrations", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    panel_raw = load_any(args.panel_file)
    panel = standardize_option_panel(panel_raw)
    if panel.empty:
        raise RuntimeError("Option panel is empty after standardization")
    unique_dates = int(panel["date"].nunique())
    if unique_dates < int(args.min_panel_dates):
        summary = {
            "ticker": args.ticker,
            "panel_file": str(args.panel_file),
            "n_panel_dates": unique_dates,
            "status": "insufficient_panel_history",
            "required_min_panel_dates": int(args.min_panel_dates),
            "message": "Not enough daily snapshots yet to run the canonical short-dated hedge study.",
        }
        (out / "daily_market_backtest_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return

    history = load_underlying_history_csv(args.history_csv) if args.history_csv else build_history_from_panel(panel)

    contracts = select_listed_contracts(
        panel,
        option_type=str(args.option_type),
        target_T=float(args.T),
        strike_mode=str(args.strike_mode),
        K_fixed=args.K_fixed,
        K_mult=args.K_mult,
        entry_step_days=int(args.entry_step_days),
        max_contracts=args.max_contracts,
        contract_moneyness_band=float(args.contract_moneyness_band),
        contract_max_relative_spread=float(args.contract_max_relative_spread),
        contract_min_volume=float(args.contract_min_volume),
        contract_min_open_interest=float(args.contract_min_open_interest),
    )
    if not contracts:
        raise RuntimeError("No listed contracts selected from the supplied panel")

    required_dates = sorted({c.entry_date for c in contracts} | {d for c in contracts for d in panel["date"][(panel["date"] >= c.entry_date) & (panel["date"] < c.maturity_date)].tolist()})

    snapshot_cfg = SnapshotCalibrationConfig(
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
    )
    calibration_cache = build_daily_calibration_cache(
        panel,
        ticker=str(args.ticker),
        snapshot_cfg=snapshot_cfg,
        required_dates=required_dates,
        cache_dir=(out / "daily_calibrations") if args.save_daily_calibrations else None,
    )

    tx_model = TransactionCostModel(kind="proportional", rate=float(args.tx_bps) / 10000.0)
    pde_cfg = PDEConfig(n_space=int(args.pde_space), n_time=int(args.pde_time))

    contract_df = pd.DataFrame([c.__dict__ for c in contracts])
    contract_df.to_csv(out / "selected_contracts.csv", index=False)

    results = {}
    for model in ["BS", "LocalVol"]:
        df = run_daily_market_panel_backtest(
            contracts=contracts,
            panel=panel,
            history=history,
            calibration_cache=calibration_cache,
            model=model,
            tx_costs=tx_model,
            pde_cfg=pde_cfg,
        )
        df.to_csv(out / f"daily_market_backtest_{model}.csv", index=False)
        results[model] = summarize_market_panel_backtest(df)

    summary = {
        "ticker": args.ticker,
        "panel_file": str(args.panel_file),
        "history_csv": str(args.history_csv) if args.history_csv else None,
        "n_panel_dates": unique_dates,
        "panel_date_start": str(pd.to_datetime(panel["date"]).min().date()),
        "panel_date_end": str(pd.to_datetime(panel["date"]).max().date()),
        "n_contracts": int(len(contracts)),
        "status": "ok",
        "models": results,
        "settings": {
            "option_type": args.option_type,
            "strike_mode": args.strike_mode,
            "K_fixed": args.K_fixed,
            "K_mult": args.K_mult,
            "T": float(args.T),
            "entry_step_days": int(args.entry_step_days),
            "contract_moneyness_band": float(args.contract_moneyness_band),
            "contract_max_relative_spread": float(args.contract_max_relative_spread),
            "contract_min_volume": float(args.contract_min_volume),
            "contract_min_open_interest": float(args.contract_min_open_interest),
            "tx_bps": float(args.tx_bps),
            "pde_space": int(args.pde_space),
            "pde_time": int(args.pde_time),
            "min_panel_dates": int(args.min_panel_dates),
        },
    }
    (out / "daily_market_backtest_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Saved daily market-panel backtest outputs to {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
