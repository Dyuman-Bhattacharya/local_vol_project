# scripts/run_backtest_history_windows.py
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

# Add src/ to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from market_data.yahoo_loader import fetch_underlying_history
from hedging.delta_hedger import (
    DeltaHedger,
    BSDeltaModel,
    LocalVolPDEDeltaModelStrikeSurface,
)
from hedging.transaction_costs import TransactionCostModel
from hedging.backtest import run_delta_hedge_backtest
from hedging.metrics import summarize_backtest
from local_volatility.surface import LocalVolSurface

from precompute_lv_grids import build_lv_grids

OptionType = Literal["call", "put"]
KMode = Literal["atm", "fixed_strike", "fixed_moneyness"]


def load_history_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date")
    close_col = cols.get("close", cols.get("s"))
    if date_col is None or close_col is None:
        raise RuntimeError(f"History CSV must contain date and close/S columns: {path}")
    out = df[[date_col, close_col]].copy()
    out.columns = ["date", "S"]
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce").dt.tz_convert(None)
    out["S"] = pd.to_numeric(out["S"], errors="coerce")
    out = out.dropna(subset=["date", "S"]).sort_values("date").reset_index(drop=True)
    if out.empty:
        raise RuntimeError(f"History CSV contains no valid rows: {path}")
    return out


def build_windows(df_prices: pd.DataFrame, T: float, step_days: int) -> list[pd.DataFrame]:
    df_prices = df_prices.sort_values("date").reset_index(drop=True).copy()
    dates = df_prices["date"].to_numpy()

    paths = []
    for start_idx in range(0, len(df_prices), step_days):
        start_date = dates[start_idx]
        end_date = start_date + np.timedelta64(int(round(T * 365)), "D")

        mask = (df_prices["date"] >= start_date) & (df_prices["date"] <= end_date)
        sub = df_prices.loc[mask, ["date", "S"]].copy()
        if len(sub) < 10:
            continue

        t0 = sub["date"].iloc[0]
        sub["t"] = (sub["date"] - t0).dt.days / 365.0
        sub = sub[sub["t"] <= T].copy()
        if sub["t"].iloc[-1] < 0.8 * T:
            continue

        paths.append(sub[["t", "S"]].reset_index(drop=True))
    return paths


def filter_paths_by_initial_spot(
    paths: list[pd.DataFrame],
    *,
    spot_ref: float,
    spot_band_pct: float,
) -> list[pd.DataFrame]:
    low = float(spot_ref) * (1.0 - float(spot_band_pct))
    high = float(spot_ref) * (1.0 + float(spot_band_pct))
    out = []
    for p in paths:
        s0 = float(p["S"].iloc[0])
        if low <= s0 <= high:
            out.append(p)
    return out


def _band_suffix(spot_band_pct: Optional[float]) -> str:
    if spot_band_pct is None:
        return ""
    pct = 100.0 * float(spot_band_pct)
    text = f"{pct:.2f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"_band{text}pct"


def _load_manifest(calib_dir: Path) -> Optional[dict]:
    p = calib_dir / "lv_grid_manifest.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _grids_match(
    *,
    calib_dir: Path,
    manifest: dict,
    T: float,
    option_type: str,
    r: float,
    q: float,
    K_grid: np.ndarray,
    S_low: float,
    S_high: float,
    n_space: int,
    n_time: int,
) -> bool:
    try:
        if int(manifest.get("version", 0)) != 2:
            return False
        if float(manifest["T"]) != float(T):
            return False
        if str(manifest["option_type"]) != str(option_type):
            return False
        if abs(float(manifest["r"]) - float(r)) > 1e-12:
            return False
        if abs(float(manifest["q"]) - float(q)) > 1e-12:
            return False
        if abs(float(manifest["S_low"]) - float(S_low)) > 1e-12:
            return False
        if abs(float(manifest["S_high"]) - float(S_high)) > 1e-12:
            return False
        if int(manifest["n_space"]) != int(n_space):
            return False
        if int(manifest["n_time"]) != int(n_time):
            return False
        if int(manifest["n_strikes"]) != int(K_grid.size):
            return False

        saved_K = np.load(calib_dir / manifest["K_file"])
        return np.array_equal(saved_K, K_grid)
    except Exception:
        return False


def ensure_lv_grids(
    *,
    calib_dir: Path,
    T: float,
    option_type: OptionType,
    r: float,
    q: float,
    K_grid: np.ndarray,
    S_low: float,
    S_high: float,
    n_space: int = 120,
    n_time: int = 100,
) -> dict:
    calib_dir = Path(calib_dir)
    manifest = _load_manifest(calib_dir)

    if manifest is not None and _grids_match(
        calib_dir=calib_dir,
        manifest=manifest,
        T=T,
        option_type=option_type,
        r=r,
        q=q,
        K_grid=K_grid,
        S_low=S_low,
        S_high=S_high,
        n_space=n_space,
        n_time=n_time,
    ):
        return manifest

    built = build_lv_grids(
        calib_dir=calib_dir,
        T=float(T),
        option_type=str(option_type),
        K_grid=K_grid,
        S_low=float(S_low),
        S_high=float(S_high),
        n_space=int(n_space),
        n_time=int(n_time),
        r=float(r),
        q=float(q),
    )
    return _load_manifest(calib_dir) or {}


def derive_path_strikes(
    *,
    paths: list[pd.DataFrame],
    K_mode: KMode,
    K_fixed: Optional[float],
    K_mult: Optional[float],
) -> np.ndarray:
    strikes = []
    for p in paths:
        S0 = float(p["S"].iloc[0])
        if K_mode == "atm":
            strikes.append(S0)
        elif K_mode == "fixed_moneyness":
            if K_mult is None:
                raise ValueError("K_mult required for fixed_moneyness")
            strikes.append(float(K_mult) * S0)
        else:
            if K_fixed is None:
                raise ValueError("K_fixed required for fixed_strike")
            strikes.append(float(K_fixed))
    return np.asarray(strikes, dtype=float)


def validate_lv_domain_coverage(lv_surface: LocalVolSurface, paths: list[pd.DataFrame], strikes: np.ndarray) -> None:
    all_spots = np.concatenate([p["S"].to_numpy(dtype=float) for p in paths])
    low = float(min(np.min(all_spots), np.min(strikes)))
    high = float(max(np.max(all_spots), np.max(strikes)))
    lv_low = float(lv_surface.S_grid[0])
    lv_high = float(lv_surface.S_grid[-1])

    if low < lv_low or high > lv_high:
        raise RuntimeError(
            "Requested backtest domain falls outside the calibrated local-vol surface. "
            f"Need spots/strikes in [{low:.6f}, {high:.6f}], but calibration only covers "
            f"[{lv_low:.6f}, {lv_high:.6f}]. Recalibrate on a compatible spot regime."
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--calib_dir", required=True)
    ap.add_argument("--history_start", required=False)
    ap.add_argument("--history_end", required=False)
    ap.add_argument("--history_csv", default=None, help="Optional local history CSV with date and close/S columns")

    ap.add_argument("--option_type", choices=["call", "put"], default="call")
    ap.add_argument("--K_mode", choices=["atm", "fixed_strike", "fixed_moneyness"], default="atm")
    ap.add_argument("--K_fixed", type=float, default=None, help="Absolute strike for fixed_strike")
    ap.add_argument("--K_mult", type=float, default=None, help="Strike multiplier for fixed_moneyness: K = K_mult * S0")
    ap.add_argument("--T", type=float, required=True, help="years, e.g. 0.25")
    ap.add_argument("--step_days", type=int, default=7, help="window start step in days")
    ap.add_argument("--min_paths", type=int, default=5, help="Minimum number of paths required after filtering")
    ap.add_argument("--bs_sigma", type=float, default=0.20)
    ap.add_argument("--tx_bps", type=float, default=10.0)
    ap.add_argument("--grid_space", type=int, default=120)
    ap.add_argument("--grid_time", type=int, default=100)
    ap.add_argument("--r", type=float, default=0.0)
    ap.add_argument("--q", type=float, default=0.0)
    ap.add_argument(
        "--spot_band_pct",
        type=float,
        default=None,
        help="Optional initial-spot compatibility band around the calibration spot, as a fraction (e.g. 0.05 for +/-5%)",
    )
    args = ap.parse_args()

    calib = Path(args.calib_dir)
    lv_pkl = calib / "local_vol_surface.pkl"
    if not lv_pkl.exists():
        raise RuntimeError(
            f"Missing {lv_pkl}. "
            "Run Notebooks 1-3 (or scripts/run_snapshot_calibration.py) first."
        )

    if args.K_mode == "fixed_strike" and args.K_fixed is None:
        raise ValueError("--K_fixed is required for --K_mode fixed_strike")
    if args.K_mode == "fixed_moneyness" and args.K_mult is None:
        raise ValueError("--K_mult is required for --K_mode fixed_moneyness")

    with open(lv_pkl, "rb") as f:
        lv_surface = pickle.load(f)

    if not isinstance(lv_surface, LocalVolSurface):
        raise RuntimeError(f"{lv_pkl} does not contain a LocalVolSurface")

    if args.history_csv is not None:
        df_prices = load_history_csv(args.history_csv)
        if args.history_start is not None:
            df_prices = df_prices[df_prices["date"] >= pd.to_datetime(args.history_start)]
        if args.history_end is not None:
            df_prices = df_prices[df_prices["date"] <= pd.to_datetime(args.history_end)]
    else:
        if args.history_start is None or args.history_end is None:
            raise ValueError("--history_start and --history_end are required unless --history_csv is provided")
        df_prices = fetch_underlying_history(args.ticker, start=args.history_start, end=args.history_end)

    paths = build_windows(df_prices, T=float(args.T), step_days=int(args.step_days))
    if args.spot_band_pct is not None:
        call_grid_npz = calib / "call_price_grid.npz"
        if not call_grid_npz.exists():
            raise RuntimeError(
                f"--spot_band_pct requires calibration spot metadata, but {call_grid_npz} is missing."
            )
        calib_npz = np.load(call_grid_npz)
        calib_spot = float(calib_npz["S0"])
        paths_before = len(paths)
        paths = filter_paths_by_initial_spot(
            paths,
            spot_ref=calib_spot,
            spot_band_pct=float(args.spot_band_pct),
        )
        print(
            f"Filtered paths by initial-spot band +/-{100.0 * float(args.spot_band_pct):.2f}% "
            f"around calibration spot {calib_spot:.6f}: kept {len(paths)}/{paths_before}",
            flush=True,
        )

    if len(paths) < int(args.min_paths):
        raise RuntimeError(
            f"Too few paths built ({len(paths)}). Expand date range, reduce step_days, "
            f"or relax --spot_band_pct / --min_paths."
        )

    strikes = derive_path_strikes(
        paths=paths,
        K_mode=str(args.K_mode),
        K_fixed=float(args.K_fixed) if args.K_fixed is not None else None,
        K_mult=float(args.K_mult) if args.K_mult is not None else None,
    )
    validate_lv_domain_coverage(lv_surface, paths, strikes)

    K_grid = np.unique(np.round(strikes, 10))
    S_low = float(lv_surface.S_grid[0])
    S_high = float(lv_surface.S_grid[-1])

    manifest = ensure_lv_grids(
        calib_dir=calib,
        T=float(args.T),
        option_type=str(args.option_type),
        r=float(args.r),
        q=float(args.q),
        K_grid=K_grid,
        S_low=S_low,
        S_high=S_high,
        n_space=int(args.grid_space),
        n_time=int(args.grid_time),
    )

    t_grid = np.load(calib / manifest["t_file"])
    S_grid = np.load(calib / manifest["S_file"])
    K_grid = np.load(calib / manifest["K_file"])
    V_grid = np.load(calib / manifest["V_file"])
    D_grid = np.load(calib / manifest["D_file"])

    tx = TransactionCostModel(kind="proportional", rate=float(args.tx_bps) / 1e4)

    for model_name in ["BS", "LocalVol"]:
        per_path_dfs = []
        n_paths = len(paths)
        print(f"\nRunning {model_name} hedging on {n_paths} paths", flush=True)

        for i, p in enumerate(paths, start=1):
            print(f"[{model_name}] Path {i}/{n_paths}", end="\r", flush=True)

            S0 = float(p["S"].iloc[0])
            if args.K_mode == "atm":
                K = S0
            elif args.K_mode == "fixed_moneyness":
                K = float(args.K_mult) * S0
            else:
                K = float(args.K_fixed)

            if model_name == "BS":
                model = BSDeltaModel(
                    S0_ref=S0,
                    K=K,
                    T=float(args.T),
                    r=float(args.r),
                    q=float(args.q),
                    sigma=float(args.bs_sigma),
                    option_type=str(args.option_type),
                )
            else:
                model = LocalVolPDEDeltaModelStrikeSurface(
                    K=K,
                    T=float(args.T),
                    r=float(args.r),
                    q=float(args.q),
                    option_type=str(args.option_type),
                    K_grid=K_grid,
                    S_grid=S_grid,
                    t_grid=t_grid,
                    V_grid=V_grid,
                    D_grid=D_grid,
                )

            hedger = DeltaHedger(
                model=model,
                K=K,
                T=float(args.T),
                r=float(args.r),
                option_type=str(args.option_type),
                tx_costs=tx,
            )

            res = run_delta_hedge_backtest(hedger=hedger, price_paths=[p])
            df_one = res.per_trade.copy()
            df_one["path_id"] = i - 1
            df_one["model"] = model_name
            df_one["K"] = K
            df_one["S0"] = S0
            df_one["option_type"] = str(args.option_type)
            df_one["K_mode"] = str(args.K_mode)
            df_one["K_mult"] = float(args.K_mult) if args.K_mult is not None else np.nan
            per_path_dfs.append(df_one)

        print()

        all_trades = pd.concat(per_path_dfs, ignore_index=True)
        summary = summarize_backtest(all_trades)
        print(f"\n=== {model_name} summary over {len(paths)} paths ===", flush=True)
        print(summary)

        band_suffix = _band_suffix(args.spot_band_pct)
        out_csv = calib / (
            f"backtest_{model_name}_{args.option_type}_{args.K_mode}"
            f"_T{args.T}_step{args.step_days}{band_suffix}.csv"
        )
        all_trades.to_csv(out_csv, index=False)
        print(f"Saved per-trade CSV: {out_csv}", flush=True)


if __name__ == "__main__":
    main()
