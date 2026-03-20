#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from pricing.pde_solver import price_european_pde_local_vol, PDEConfig


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run pricing validation for a calibrated local-vol surface.")
    ap.add_argument("--calib_dir", required=True)
    ap.add_argument("--market_csv", default=None, help="Optional cleaned options CSV; defaults to calib_dir/cleaned_options_data.csv")
    ap.add_argument("--max_options", type=int, default=150)
    ap.add_argument("--n_space", type=int, default=220)
    ap.add_argument("--n_time", type=int, default=120)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    calib_dir = Path(args.calib_dir)
    fig_dir = calib_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    lv_surface_path = calib_dir / "local_vol_surface.pkl"
    market_path = Path(args.market_csv) if args.market_csv is not None else calib_dir / "cleaned_options_data.csv"

    if not lv_surface_path.exists():
        raise FileNotFoundError(f"Missing {lv_surface_path}")
    if not market_path.exists():
        raise FileNotFoundError(f"Missing {market_path}")

    with open(lv_surface_path, "rb") as f:
        lv_surface = pickle.load(f)

    df_market = pd.read_csv(market_path)
    df_market = df_market[df_market["option_type"].astype(str).str.lower() == "call"].copy()
    df_market["moneyness"] = df_market["strike"] / df_market["underlying_price"]
    if "bid" in df_market.columns and "ask" in df_market.columns:
        df_market["inside_bid_ask_input"] = df_market["mid"].between(df_market["bid"], df_market["ask"], inclusive="both")

    mask = (
        df_market["mid"].notna()
        & (df_market["time_to_expiry"] > 0.0)
        & (df_market["time_to_expiry"] <= lv_surface.t_grid.max())
        & (df_market["strike"] >= lv_surface.S_grid.min())
        & (df_market["strike"] <= lv_surface.S_grid.max())
        & (df_market["underlying_price"] >= lv_surface.S_grid.min())
        & (df_market["underlying_price"] <= lv_surface.S_grid.max())
    )
    df_eval = df_market.loc[mask].copy()
    df_eval["trusted_domain"] = lv_surface.is_trusted(
        df_eval["strike"].to_numpy(dtype=float),
        df_eval["time_to_expiry"].to_numpy(dtype=float),
    )
    df_eval["supported_domain"] = lv_surface.is_supported(
        df_eval["strike"].to_numpy(dtype=float),
        df_eval["time_to_expiry"].to_numpy(dtype=float),
    )
    df_eval = df_eval[df_eval["trusted_domain"]].copy()
    if df_eval.empty:
        raise RuntimeError("No market options overlap the trusted region of the calibrated local-vol domain")

    if len(df_eval) > int(args.max_options):
        df_eval = df_eval.sample(int(args.max_options), random_state=7).sort_values(["time_to_expiry", "strike"]).reset_index(drop=True)
    else:
        df_eval = df_eval.sort_values(["time_to_expiry", "strike"]).reset_index(drop=True)

    pde_cfg = PDEConfig(n_space=int(args.n_space), n_time=int(args.n_time))
    prices = []
    for _, row in df_eval.iterrows():
        prices.append(
            price_european_pde_local_vol(
                S0=float(row["underlying_price"]),
                K=float(row["strike"]),
                T=float(row["time_to_expiry"]),
                r=float(row["risk_free_rate"]),
                q=float(row["dividend_yield"]),
                option_type="call",
                lv_surface=lv_surface,
                cfg=pde_cfg,
            )
        )

    df_eval["model_price"] = prices
    df_eval["market_price"] = df_eval["mid"]
    df_eval["pricing_error"] = df_eval["model_price"] - df_eval["market_price"]
    df_eval["abs_error"] = df_eval["pricing_error"].abs()
    if "bid" in df_eval.columns and "ask" in df_eval.columns:
        df_eval["model_inside_bid_ask"] = df_eval["model_price"].between(df_eval["bid"], df_eval["ask"], inclusive="both")
        spread = np.maximum((df_eval["ask"] - df_eval["bid"]).to_numpy(dtype=float), 1e-8)
        clipped = np.clip(df_eval["model_price"].to_numpy(dtype=float), df_eval["bid"].to_numpy(dtype=float), df_eval["ask"].to_numpy(dtype=float))
        df_eval["normalized_band_violation"] = np.abs(df_eval["model_price"].to_numpy(dtype=float) - clipped) / spread
    df_eval.to_csv(calib_dir / "repricing_validation.csv", index=False)
    summary = {
        "n_eval": int(len(df_eval)),
        "trusted_domain_only": True,
        "mean_abs_error": float(df_eval["abs_error"].mean()),
        "rmse": float(np.sqrt(np.mean(df_eval["pricing_error"].to_numpy(dtype=float) ** 2))),
    }
    if "model_inside_bid_ask" in df_eval.columns:
        summary["model_inside_bid_ask_fraction"] = float(df_eval["model_inside_bid_ask"].mean())
        summary["mean_normalized_band_violation"] = float(df_eval["normalized_band_violation"].mean())
    (calib_dir / "repricing_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(df_eval["market_price"], df_eval["model_price"], s=18, alpha=0.55)
    mn = min(df_eval["market_price"].min(), df_eval["model_price"].min())
    mx = max(df_eval["market_price"].max(), df_eval["model_price"].max())
    ax1.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1.2)
    ax1.set_xlabel("Market Mid Price")
    ax1.set_ylabel("LV PDE Price")
    ax1.set_title("Local Vol Repricing: Model vs Market")
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(df_eval["pricing_error"], bins=35, alpha=0.85)
    ax2.set_xlabel("LV Price - Market Mid")
    ax2.set_ylabel("Count")
    ax2.set_title("Repricing Error Distribution")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "05_repricing_errors.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(df_eval["moneyness"], df_eval["pricing_error"], s=18, alpha=0.55)
    axes[0].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("Moneyness K/S")
    axes[0].set_ylabel("LV Price - Market Mid")
    axes[0].set_title("Repricing Error vs Moneyness")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(df_eval["time_to_expiry"], df_eval["pricing_error"], s=18, alpha=0.55)
    axes[1].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("Maturity T (years)")
    axes[1].set_ylabel("LV Price - Market Mid")
    axes[1].set_title("Repricing Error vs Maturity")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "05_repricing_by_moneyness.png", dpi=150, bbox_inches="tight")
    plt.savefig(fig_dir / "05_repricing_by_maturity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved repricing validation to {calib_dir / 'repricing_validation.csv'}")
    print(f"Mean abs repricing error: {df_eval['abs_error'].mean():.6g}")


if __name__ == "__main__":
    main()
