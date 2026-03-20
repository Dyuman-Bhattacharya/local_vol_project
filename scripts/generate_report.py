#!/usr/bin/env python3
"""
scripts/generate_report.py

Load existing artifacts produced by the notebooks / scripts and generate:
- Publication-quality figures saved to: output/figures/
- A single markdown report saved to: output/report.md

Hard rules:
- NO recalibration here.
- DO NOT assume files exist: discover and check.
- Use matplotlib only (no seaborn).
"""

from __future__ import annotations

import argparse
import sys
import math
import json
import textwrap
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # non-interactive
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------

REQUIRED_FIGS: Dict[str, List[str]] = {
    "Data / Notebook 1": [
        "01_data_distribution.png",
        "01_iv_surface_market.png",
        "01_outliers.png",
    ],
    "IV Surface / Notebook 2": [
        "02_surface_fit_quality.png",
        "02_arbitrage_checks.png",
        "02_call_price_grid.png",
        "02_density_validation.png",
    ],
    "Local Vol / Notebook 3-4": [
        "03_local_vol_surface.png",
        "03_local_vol_slices.png",
        "04_lv_diagnostics.png",
        "04_regularization_impact.png",
    ],
    "Pricing Validation / Notebook 4": [
        "05_repricing_errors.png",
        "05_repricing_by_moneyness.png",
        "05_repricing_by_maturity.png",
    ],
    "Hedging Backtest / Notebook 5": [
        "05_hedging_pnl_distribution.png",
        "05_error_ecdf.png",
        "05_qq_plot.png",
        "05_transaction_costs.png",
        "05_regime_comparison.png",
        "05_error_decomposition.png",
    ],
}


FIGURE_CAPTIONS: Dict[str, str] = {
    "01_data_distribution.png": "Distribution of cleaned option quotes across strikes, maturities, and liquidity filters.",
    "01_iv_surface_market.png": "Market-implied volatility structure before parametric fitting.",
    "01_outliers.png": "Diagnostic view of quotes removed or flagged during cleaning.",
    "02_surface_fit_quality.png": "Comparison between market-implied volatilities and the interpolated total-variance surface.",
    "02_arbitrage_checks.png": "Static-arbitrage diagnostics on the fitted call-price grid.",
    "02_call_price_grid.png": "Projected call-price grid used as the canonical arbitrage-free surface.",
    "02_density_validation.png": "Risk-neutral density diagnostics implied by the final call grid.",
    "03_local_vol_surface.png": "Regularized Dupire local-volatility surface used in pricing and hedging.",
    "03_local_vol_slices.png": "Representative local-volatility slices across maturities and strikes.",
    "04_lv_diagnostics.png": "Diagnostics showing where local-vol regularization is active.",
    "04_regularization_impact.png": "Before-versus-after comparison of raw and cleaned local volatility.",
    "05_repricing_errors.png": "Absolute repricing error distribution for the local-vol PDE model.",
    "05_repricing_by_moneyness.png": "Repricing errors grouped by moneyness.",
    "05_repricing_by_maturity.png": "Repricing errors grouped by maturity.",
    "05_hedging_pnl_distribution.png": "Distribution of the primary hedge-study metric for Black-Scholes and local volatility.",
    "05_error_ecdf.png": "Empirical cumulative distribution of the absolute primary hedge-study metric.",
    "05_qq_plot.png": "Quantile comparison of Black-Scholes and local-vol hedge-study outcomes.",
    "05_transaction_costs.png": "Transaction-cost comparison across hedge models.",
    "05_regime_comparison.png": "Comparison across maturity buckets or contract groupings.",
    "05_error_decomposition.png": "Mean-versus-dispersion view of the hedge-study metric.",
}


@dataclass
class Paths:
    root: Path
    output_dir: Path
    figures_dir: Path
    report_path: Path


class MissingArtifactError(RuntimeError):
    pass


def project_root_from_this_file() -> Path:
    # scripts/generate_report.py -> project root is parent of scripts/
    return Path(__file__).resolve().parents[1]


def pick_output_dir(root: Path) -> Path:
    """
    Prefer existing output/; else existing outputs/; else create output/.
    """
    cand1 = root / "output"
    cand2 = root / "outputs"
    if cand1.exists() and cand1.is_dir():
        return cand1
    if cand2.exists() and cand2.is_dir():
        return cand2
    # default create output/
    cand1.mkdir(parents=True, exist_ok=True)
    return cand1


def ensure_dirs(p: Paths) -> None:
    p.output_dir.mkdir(parents=True, exist_ok=True)
    p.figures_dir.mkdir(parents=True, exist_ok=True)


def savefig(fig: plt.Figure, path: Path, dpi: int = 175) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def log(msg: str) -> None:
    safe = msg.encode("ascii", errors="replace").decode("ascii")
    print(safe, flush=True)


def list_existing(output_dir: Path) -> List[Path]:
    return sorted([p for p in output_dir.rglob("*") if p.is_file()])


def find_first(paths: Sequence[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def require(path: Optional[Path], how_to_fix: str) -> Path:
    if path is None or not path.exists():
        raise MissingArtifactError(
            f"Missing required artifact.\n"
            f"Expected: {path}\n"
            f"Fix: {how_to_fix}"
        )
    return path


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Failed reading CSV: {path}\n{e}") from e


def safe_read_json(path: Path) -> Dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed reading JSON: {path}\n{e}") from e


def _fmt(x: object, digits: int = 6) -> str:
    try:
        val = float(x)
    except Exception:
        return str(x)
    if np.isnan(val):
        return "nan"
    if abs(val) >= 1000 or (abs(val) > 0 and abs(val) < 1e-3):
        return f"{val:.{digits}e}"
    return f"{val:.{digits}g}"


def report_title_from_artifacts(output_dir: Path) -> str:
    summary_path = output_dir / "calibration_summary.json"
    if summary_path.exists():
        try:
            summary = safe_read_json(summary_path)
            final_counts = summary.get("arbitrage_counts", {})
            raw_counts = summary.get("arbitrage_counts_raw", {})
            if int(final_counts.get("n_fail", 1)) == 0:
                raw_fail = int(raw_counts.get("n_fail", 0))
                if raw_fail > 0:
                    return "Arbitrage-Free Output Grid (Projected from Interpolated Total-Variance Surface) -> Dupire Local Vol -> Pricing -> Hedging Report"
                return "Arbitrage-Free IV Surface -> Dupire Local Vol -> Pricing -> Hedging Report"
        except Exception:
            pass
    return "IV Surface -> Dupire Local Vol -> Pricing -> Hedging Report"


def safe_load_npz(path: Path) -> Dict[str, np.ndarray]:
    try:
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}
    except Exception as e:
        raise RuntimeError(f"Failed loading NPZ: {path}\n{e}") from e


def column_any(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def finite(a: np.ndarray) -> np.ndarray:
    return a[np.isfinite(a)]


# -----------------------------
# Black–Scholes implied vol (for fit-quality plots)
# -----------------------------

import math

def _norm_cdf(x):
    """
    Standard normal CDF.
    Works for scalars or numpy arrays.
    """
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))


def bs_call_price(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0:
        return max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    fwd = S * math.exp((r - q) * T)
    vol_sqrt = sigma * math.sqrt(T)
    d1 = (math.log(fwd / K) + 0.5 * sigma * sigma * T) / vol_sqrt
    d2 = d1 - vol_sqrt
    return S * math.exp(-q * T) * float(_norm_cdf(np.array(d1))) - K * math.exp(-r * T) * float(_norm_cdf(np.array(d2)))


def implied_vol_call_brent(
    price: float, S: float, K: float, T: float, r: float, q: float,
    lo: float = 1e-6, hi: float = 5.0, tol: float = 1e-8, max_iter: int = 200
) -> float:
    """
    Simple robust implied vol inverter for plotting (no dependency on project src).
    Uses bisection (monotone), not Brent, to keep this file self-contained.
    """
    if T <= 0:
        return np.nan
    intrinsic = max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    upper = S * math.exp(-q * T)
    if not (intrinsic <= price <= upper + 1e-8):
        return np.nan

    def f(sig: float) -> float:
        return bs_call_price(S, K, T, r, q, sig) - price

    flo = f(lo)
    fhi = f(hi)
    if np.isnan(flo) or np.isnan(fhi):
        return np.nan
    # ensure bracket (it should be monotone increasing in sigma)
    if flo > 0:
        return lo
    if fhi < 0:
        return np.nan

    a, b = lo, hi
    fa, fb = flo, fhi
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(fm) < tol or (b - a) < tol:
            return m
        if fm > 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


# -----------------------------
# Plot builders
# -----------------------------

def plot_heatmap_counts(
    df: pd.DataFrame, x: str, y: str, x_bins: int, y_bins: int,
    title: str, xlabel: str, ylabel: str
) -> plt.Figure:
    xvals = df[x].to_numpy(dtype=float)
    yvals = df[y].to_numpy(dtype=float)
    mask = np.isfinite(xvals) & np.isfinite(yvals)
    xvals = xvals[mask]
    yvals = yvals[mask]

    H, xedges, yedges = np.histogram2d(xvals, yvals, bins=[x_bins, y_bins])
    fig = plt.figure(figsize=(9.5, 6.5))
    ax = fig.add_subplot(111)
    im = ax.imshow(
        H.T,
        origin="lower",
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("count")
    return fig


def plot_surface_from_grid(Z: np.ndarray, x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str, zlabel: str) -> plt.Figure:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(14, 6))
    # left: heatmap
    ax1 = fig.add_subplot(1, 2, 1)
    im = ax1.imshow(Z, origin="lower", aspect="auto", extent=[x.min(), x.max(), y.min(), y.max()])
    ax1.set_title(title + " (heatmap)")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    cb = fig.colorbar(im, ax=ax1)
    cb.set_label(zlabel)

    # right: 3D
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    X, Y = np.meshgrid(x, y)
    # if Z is (nY, nX) we match mesh
    ax2.plot_surface(X, Y, Z, rstride=2, cstride=2, linewidth=0.2, antialiased=True, alpha=0.95)
    ax2.set_title(title + " (3D)")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax2.set_zlabel(zlabel)
    return fig


def ecdf(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = finite(a)
    a = np.sort(a)
    if a.size == 0:
        return a, a
    y = np.arange(1, a.size + 1) / a.size
    return a, y


def qq_plot(a: np.ndarray, b: np.ndarray, title: str, xlabel: str, ylabel: str) -> plt.Figure:
    a = finite(a)
    b = finite(b)
    n = min(a.size, b.size)
    if n == 0:
        fig = plt.figure(figsize=(7.5, 6))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No finite data for QQ plot.", ha="center", va="center")
        ax.set_axis_off()
        return fig
    qa = np.quantile(a, np.linspace(0.01, 0.99, 99))
    qb = np.quantile(b, np.linspace(0.01, 0.99, 99))
    fig = plt.figure(figsize=(7.5, 6))
    ax = fig.add_subplot(111)
    ax.scatter(qa, qb, s=18, alpha=0.9)
    mn = min(qa.min(), qb.min())
    mx = max(qa.max(), qb.max())
    ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return fig


# -----------------------------
# Section generators
# -----------------------------

def generate_data_figures(paths: Paths) -> Dict[str, Path]:
    """
    Uses: output/cleaned_options_data.csv
    """
    out: Dict[str, Path] = {}

    cleaned_csv = find_first([
        paths.output_dir / "cleaned_options_data.csv",
        paths.output_dir / "cleaned_options_data.parquet",
    ])

    if cleaned_csv is None or not cleaned_csv.exists():
        raise MissingArtifactError(
            "Missing cleaned options data.\n"
            f"Looked for: {paths.output_dir/'cleaned_options_data.csv'}\n"
            "Fix: Run notebooks/01_data_exploration.ipynb"
        )

    if cleaned_csv.suffix.lower() == ".csv":
        df = safe_read_csv(cleaned_csv)
    else:
        df = pd.read_parquet(cleaned_csv)

    # Identify common columns
    col_T = column_any(df, ["T", "time_to_expiry", "ttm", "tau"])
    col_K = column_any(df, ["K", "strike"])
    col_S0 = column_any(df, ["S0", "spot", "underlying_price", "underlying"])
    col_iv = column_any(df, ["iv", "implied_vol", "implied_volatility", "sigma_imp"])
    col_mid = column_any(df, ["mid", "mid_price", "price"])
    col_bid = column_any(df, ["bid"])
    col_ask = column_any(df, ["ask"])

    if col_T is None or col_K is None:
        raise MissingArtifactError(
            "cleaned_options_data.csv is missing required columns for plotting.\n"
            f"Need at least: maturity(T/ttm) and strike(K).\n"
            f"Columns present: {list(df.columns)}"
        )

    # moneyness if possible
    if col_S0 is not None:
        df["_moneyness"] = df[col_K].astype(float) / df[col_S0].astype(float)
        x_field = "_moneyness"
        x_label = "moneyness K/S"
    else:
        x_field = col_K
        x_label = "strike K"

    # 01_data_distribution.png — maturity vs strike/moneyness heatmap
    fig = plot_heatmap_counts(
        df=df, x=col_T, y=x_field, x_bins=40, y_bins=40,
        title="Data distribution: maturity vs strike/moneyness",
        xlabel="maturity T (years)",
        ylabel=x_label,
    )
    p = paths.figures_dir / "01_data_distribution.png"
    savefig(fig, p)
    out["01_data_distribution.png"] = p

    # 01_iv_surface_market.png — raw market IV surface plot (or equivalent)
    # If IV exists: heatmap of IV by (T, moneyness). Else: use mid-price heatmap as diagnostic.
    if col_iv is not None:
        df_iv = df[[col_T, x_field, col_iv]].copy()
        df_iv = df_iv[np.isfinite(df_iv[col_iv].astype(float))]
        fig = plot_heatmap_counts(
            df=df_iv, x=col_T, y=x_field, x_bins=45, y_bins=45,
            title="Raw market surface diagnostic (IV)",
            xlabel="maturity T (years)",
            ylabel=x_label,
        )
        ax = fig.axes[0]
        ax.set_title("Raw market implied vol diagnostic (counts heatmap)")
        # Add a second plot overlay? Keep simple, robust.
    else:
        if col_mid is None:
            # fall back to counts distribution again, but label it
            fig = plot_heatmap_counts(
                df=df, x=col_T, y=x_field, x_bins=45, y_bins=45,
                title="Raw market surface diagnostic (no IV column found)",
                xlabel="maturity T (years)",
                ylabel=x_label,
            )
        else:
            # create binned average mid price
            Tvals = df[col_T].astype(float).to_numpy()
            Xvals = df[x_field].astype(float).to_numpy()
            Pvals = df[col_mid].astype(float).to_numpy()
            mask = np.isfinite(Tvals) & np.isfinite(Xvals) & np.isfinite(Pvals)
            Tvals, Xvals, Pvals = Tvals[mask], Xvals[mask], Pvals[mask]

            Tb = np.linspace(Tvals.min(), Tvals.max(), 50)
            Xb = np.linspace(Xvals.min(), Xvals.max(), 50)
            Ti = np.clip(np.digitize(Tvals, Tb) - 1, 0, len(Tb) - 2)
            Xi = np.clip(np.digitize(Xvals, Xb) - 1, 0, len(Xb) - 2)

            grid = np.full((len(Tb) - 1, len(Xb) - 1), np.nan)
            cnt = np.zeros_like(grid, dtype=int)
            for t_i, x_i, pv in zip(Ti, Xi, Pvals):
                if np.isnan(grid[t_i, x_i]):
                    grid[t_i, x_i] = pv
                else:
                    grid[t_i, x_i] += pv
                cnt[t_i, x_i] += 1
            with np.errstate(invalid="ignore"):
                grid = grid / np.maximum(cnt, 1)

            fig = plt.figure(figsize=(9.5, 6.5))
            ax = fig.add_subplot(111)
            im = ax.imshow(
                grid.T, origin="lower", aspect="auto",
                extent=[Tb[0], Tb[-1], Xb[0], Xb[-1]],
            )
            ax.set_title("Raw market surface diagnostic (avg mid price in bins)")
            ax.set_xlabel("maturity T (years)")
            ax.set_ylabel(x_label)
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("avg mid price")
            ax.grid(False)

    p = paths.figures_dir / "01_iv_surface_market.png"
    savefig(fig, p)
    out["01_iv_surface_market.png"] = p

    # 01_outliers.png — outlier / filtering visualization (or equivalent)
    # Use bid-ask spread / relative spread vs mid when possible.
    fig = plt.figure(figsize=(9.5, 6.5))
    ax = fig.add_subplot(111)

    if col_bid is not None and col_ask is not None and col_mid is not None:
        bid = df[col_bid].astype(float).to_numpy()
        ask = df[col_ask].astype(float).to_numpy()
        mid = df[col_mid].astype(float).to_numpy()
        mask = np.isfinite(bid) & np.isfinite(ask) & np.isfinite(mid) & (mid > 0)
        rel_spread = (ask[mask] - bid[mask]) / mid[mask]
        ax.hist(rel_spread[np.isfinite(rel_spread)], bins=60, alpha=0.9)
        ax.set_title("Outlier / filtering diagnostic: relative bid-ask spread")
        ax.set_xlabel("(ask - bid) / mid")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.3)
        # mark heuristic thresholds
        q95 = np.nanquantile(rel_spread, 0.95) if rel_spread.size else np.nan
        if np.isfinite(q95):
            ax.axvline(q95, linestyle="--", linewidth=1.2, label=f"95th pct = {q95:.3g}")
            ax.legend()
    else:
        # fallback: price vs moneyness scatter, with extreme points highlighted
        if col_mid is None:
            ax.text(0.5, 0.5, "No bid/ask/mid columns available for outlier diagnostics.", ha="center", va="center")
            ax.set_axis_off()
        else:
            x = df[x_field].astype(float).to_numpy()
            y = df[col_mid].astype(float).to_numpy()
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            if y.size:
                q1, q99 = np.quantile(y, [0.01, 0.99])
                is_out = (y < q1) | (y > q99)
                ax.scatter(x[~is_out], y[~is_out], s=10, alpha=0.35, label="inliers")
                ax.scatter(x[is_out], y[is_out], s=12, alpha=0.9, label="1% tails")
                ax.set_title("Outlier / filtering diagnostic: mid price vs moneyness (tails highlighted)")
                ax.set_xlabel(x_label)
                ax.set_ylabel("mid price")
                ax.grid(True, alpha=0.3)
                ax.legend()

    p = paths.figures_dir / "01_outliers.png"
    savefig(fig, p)
    out["01_outliers.png"] = p

    return out


def bilinear_interp_2d(x_grid: np.ndarray, y_grid: np.ndarray, Z: np.ndarray, xq: np.ndarray, yq: np.ndarray) -> np.ndarray:
    """
    Z shape: (len(y_grid), len(x_grid)) where y varies first.
    """
    xg = np.asarray(x_grid, float)
    yg = np.asarray(y_grid, float)
    Z = np.asarray(Z, float)

    xq = np.asarray(xq, float)
    yq = np.asarray(yq, float)

    # clamp to grid bounds
    xq_c = np.clip(xq, xg.min(), xg.max())
    yq_c = np.clip(yq, yg.min(), yg.max())

    ix = np.searchsorted(xg, xq_c, side="right") - 1
    iy = np.searchsorted(yg, yq_c, side="right") - 1
    ix = np.clip(ix, 0, len(xg) - 2)
    iy = np.clip(iy, 0, len(yg) - 2)

    x1 = xg[ix]
    x2 = xg[ix + 1]
    y1 = yg[iy]
    y2 = yg[iy + 1]

    # weights
    wx = np.where(x2 > x1, (xq_c - x1) / (x2 - x1), 0.0)
    wy = np.where(y2 > y1, (yq_c - y1) / (y2 - y1), 0.0)

    z11 = Z[iy, ix]
    z21 = Z[iy, ix + 1]
    z12 = Z[iy + 1, ix]
    z22 = Z[iy + 1, ix + 1]

    z = (1 - wx) * (1 - wy) * z11 + wx * (1 - wy) * z21 + (1 - wx) * wy * z12 + wx * wy * z22
    return z


def generate_iv_surface_figures(paths: Paths) -> Dict[str, Path]:
    """
    Uses:
    - output/call_price_grid.npz      (required)
    - output/density.npz              (optional but expected)
    - output/surface_slice_summary.csv (optional)
    - output/cleaned_options_data.csv (optional for fit-quality by maturity)
    """
    out: Dict[str, Path] = {}

    call_grid = find_first([paths.output_dir / "call_price_grid.npz"])
    require(call_grid, "Run notebooks/02_iv_surface_fitting.ipynb")

    data = safe_load_npz(call_grid)
    C = data.get("C")
    K = data.get("K")
    T = data.get("T")
    S0 = float(np.atleast_1d(data.get("S0"))[0]) if "S0" in data else np.nan
    r = float(np.atleast_1d(data.get("r"))[0]) if "r" in data else 0.0
    q = float(np.atleast_1d(data.get("q"))[0]) if "q" in data else 0.0

    if C is None or K is None or T is None:
        raise MissingArtifactError(
            f"{call_grid} is missing one of required arrays: C, K, T.\n"
            "Fix: Re-run notebooks/02_iv_surface_fitting.ipynb"
        )

    # Ensure shapes: notebook saved C as (nT, nK)
    C = np.asarray(C, float)
    K = np.asarray(K, float)
    T = np.asarray(T, float)

    # 02_call_price_grid.png — fitted call price surface visualization
    fig = plot_surface_from_grid(
        Z=C, x=K, y=T,
        title="Fitted call-price surface from interpolated total variance",
        xlabel="strike K", ylabel="maturity T (years)", zlabel="call price C",
    )
    p = paths.figures_dir / "02_call_price_grid.png"
    savefig(fig, p)
    out["02_call_price_grid.png"] = p

    # 02_density_validation.png — Breeden–Litzenberger density plots
    density_npz = find_first([paths.output_dir / "density.npz"])
    if density_npz is not None and density_npz.exists():
        den = safe_load_npz(density_npz)
        dens = np.asarray(den.get("density"), float) if den.get("density") is not None else None
        mass = np.asarray(den.get("mass"), float) if den.get("mass") is not None else None
        Kd = np.asarray(den.get("K"), float) if den.get("K") is not None else K
        Td = np.asarray(den.get("T"), float) if den.get("T") is not None else T

        fig = plt.figure(figsize=(12.5, 7))
        ax = fig.add_subplot(111)
        if dens is None:
            ax.text(0.5, 0.5, "density.npz present but missing 'density' array.", ha="center", va="center")
            ax.set_axis_off()
        else:
            # Plot a few maturity slices
            nT = dens.shape[0]
            pick = np.unique(np.clip(np.round(np.linspace(0, nT - 1, 5)).astype(int), 0, nT - 1))
            for idx in pick:
                ax.plot(Kd, dens[idx, :], linewidth=1.6, label=f"T={Td[idx]:.3f}")
            ax.set_title("Risk-neutral density via Breeden–Litzenberger (selected maturities)")
            ax.set_xlabel("strike K")
            ax.set_ylabel("density p(K,T)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            if mass is not None and np.size(mass) == np.size(Td):
                ax2 = ax.twinx()
                ax2.plot(Td, mass, linestyle="--", linewidth=1.4)
                ax2.set_ylabel("density mass ∫p dK (diagnostic)")
        p = paths.figures_dir / "02_density_validation.png"
        savefig(fig, p)
        out["02_density_validation.png"] = p
    else:
        # Generate a minimal diagnostic from convexity of C (if no density artifact)
        # p ~ exp(rT) * d2C/dK2
        dK = np.gradient(K)
        d2 = np.zeros_like(C)
        for i in range(C.shape[0]):
            # second derivative in K using np.gradient twice
            d1 = np.gradient(C[i, :], K)
            d2[i, :] = np.gradient(d1, K)
        dens_approx = np.exp(r * T)[:, None] * d2
        fig = plt.figure(figsize=(12.5, 7))
        ax = fig.add_subplot(111)
        pick = np.unique(np.clip(np.round(np.linspace(0, C.shape[0] - 1, 5)).astype(int), 0, C.shape[0] - 1))
        for idx in pick:
            ax.plot(K, dens_approx[idx, :], linewidth=1.6, label=f"T={T[idx]:.3f}")
        ax.set_title("Approx. density diagnostic (from d²C/dK² on fitted grid)")
        ax.set_xlabel("strike K")
        ax.set_ylabel("approx density (unnormalized)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        p = paths.figures_dir / "02_density_validation.png"
        savefig(fig, p)
        out["02_density_validation.png"] = p

    # 02_arbitrage_checks.png — butterfly/calendar violation counts or heatmaps
    # Compute simple finite-diff checks on fitted C(K,T).
    # - monotone in K: dC/dK <= 0
    # - convex in K: d2C/dK2 >= 0
    # - monotone in T: dC/dT >= 0
    dC_dK = np.gradient(C, K, axis=1)
    d2C_dK2 = np.gradient(dC_dK, K, axis=1)
    dC_dT = np.gradient(C, T, axis=0)

    vio_monK = (dC_dK > 1e-10)
    vio_convK = (d2C_dK2 < -1e-10)
    vio_monT = (dC_dT < -1e-10)

    # show counts over T and K
    fig = plt.figure(figsize=(14, 4.8))
    ax1 = fig.add_subplot(1, 3, 1)
    im1 = ax1.imshow(vio_monK.astype(float), origin="lower", aspect="auto",
                     extent=[K.min(), K.max(), T.min(), T.max()])
    ax1.set_title("Violation: monotonicity in K (dC/dK > 0)")
    ax1.set_xlabel("K")
    ax1.set_ylabel("T")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03)

    ax2 = fig.add_subplot(1, 3, 2)
    im2 = ax2.imshow(vio_convK.astype(float), origin="lower", aspect="auto",
                     extent=[K.min(), K.max(), T.min(), T.max()])
    ax2.set_title("Violation: convexity in K (d²C/dK² < 0)")
    ax2.set_xlabel("K")
    ax2.set_ylabel("T")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03)

    ax3 = fig.add_subplot(1, 3, 3)
    im3 = ax3.imshow(vio_monT.astype(float), origin="lower", aspect="auto",
                     extent=[K.min(), K.max(), T.min(), T.max()])
    ax3.set_title("Violation: monotonicity in T (dC/dT < 0)")
    ax3.set_xlabel("K")
    ax3.set_ylabel("T")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.03)

    fig.suptitle("Static arbitrage checks on fitted call-price grid", y=1.02)
    p = paths.figures_dir / "02_arbitrage_checks.png"
    savefig(fig, p)
    out["02_arbitrage_checks.png"] = p

    # 02_surface_fit_quality.png — fitted vs market IV by maturity
    # We do: market call price -> implied vol, model grid price -> implied vol, plot per maturity.
    cleaned_csv = find_first([paths.output_dir / "cleaned_options_data.csv"])
    if cleaned_csv is not None and cleaned_csv.exists():
        df = safe_read_csv(cleaned_csv)
        col_T = column_any(df, ["T", "time_to_expiry", "ttm", "tau"])
        col_K = column_any(df, ["K", "strike"])
        col_mid = column_any(df, ["mid", "mid_price", "price"])
        col_S0 = column_any(df, ["S0", "spot", "underlying_price", "underlying"])

        if col_T and col_K and col_mid:
            # pick call options only if option_type exists
            col_type = column_any(df, ["option_type", "type"])
            if col_type:
                df = df[df[col_type].astype(str).str.lower().str.contains("call")].copy()

            # Determine spot per row, else fall back to grid's S0
            if col_S0 is None:
                df["_S0_used"] = S0
            else:
                df["_S0_used"] = df[col_S0].astype(float)

            df["_T"] = df[col_T].astype(float)
            df["_K"] = df[col_K].astype(float)
            df["_mid"] = df[col_mid].astype(float)

            # Keep finite and within grid bounds (so model interpolation works)
            mask = np.isfinite(df["_T"]) & np.isfinite(df["_K"]) & np.isfinite(df["_mid"]) & np.isfinite(df["_S0_used"])
            df = df[mask].copy()
            if len(df) > 0:
                # sample a manageable set for plotting
                df = df.sample(n=min(len(df), 4000), random_state=0)

                # Interpolate model prices on (K,T)
                model_price = bilinear_interp_2d(K, T, C, df["_K"].to_numpy(), df["_T"].to_numpy())
                df["_model_price"] = model_price

                # implied vols
                iv_mkt = []
                iv_mod = []
                for price_mkt, price_mod, s0v, kv, tv in zip(df["_mid"], df["_model_price"], df["_S0_used"], df["_K"], df["_T"]):
                    iv_mkt.append(implied_vol_call_brent(price_mkt, float(s0v), float(kv), float(tv), r, q))
                    iv_mod.append(implied_vol_call_brent(price_mod, float(s0v), float(kv), float(tv), r, q))
                df["_iv_mkt"] = np.array(iv_mkt, float)
                df["_iv_mod"] = np.array(iv_mod, float)

                df = df[np.isfinite(df["_iv_mkt"]) & np.isfinite(df["_iv_mod"])].copy()
                if len(df) > 0:
                    # Bin by maturity (quantiles)
                    df["_T_bin"] = pd.qcut(df["_T"], q=min(6, df["_T"].nunique()), duplicates="drop")
                    fig = plt.figure(figsize=(12.5, 7))
                    ax = fig.add_subplot(111)
                    for b, g in df.groupby("_T_bin", observed=False):
                        ax.scatter(g["_iv_mkt"], g["_iv_mod"], s=10, alpha=0.5, label=str(b))
                    mn = min(df["_iv_mkt"].min(), df["_iv_mod"].min())
                    mx = max(df["_iv_mkt"].max(), df["_iv_mod"].max())
                    ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1.2)
                    ax.set_title("Interpolated-surface fit quality: implied vol (model) vs implied vol (market), colored by maturity bin")
                    ax.set_xlabel("market implied vol")
                    ax.set_ylabel("model implied vol (from fitted call grid)")
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=8, frameon=False, ncols=2)
                    p = paths.figures_dir / "02_surface_fit_quality.png"
                    savefig(fig, p)
                    out["02_surface_fit_quality.png"] = p
                else:
                    # fallback figure
                    fig = plt.figure(figsize=(9, 5))
                    ax = fig.add_subplot(111)
                    ax.text(0.5, 0.5, "Fit-quality plot unavailable (implied vol inversion produced no finite points).", ha="center", va="center")
                    ax.set_axis_off()
                    p = paths.figures_dir / "02_surface_fit_quality.png"
                    savefig(fig, p)
                    out["02_surface_fit_quality.png"] = p
            else:
                fig = plt.figure(figsize=(9, 5))
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, "Fit-quality plot unavailable (no usable market rows).", ha="center", va="center")
                ax.set_axis_off()
                p = paths.figures_dir / "02_surface_fit_quality.png"
                savefig(fig, p)
                out["02_surface_fit_quality.png"] = p
        else:
            fig = plt.figure(figsize=(9, 5))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Fit-quality plot unavailable (missing T/K/mid columns in cleaned data).", ha="center", va="center")
            ax.set_axis_off()
            p = paths.figures_dir / "02_surface_fit_quality.png"
            savefig(fig, p)
            out["02_surface_fit_quality.png"] = p
    else:
        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "Fit-quality plot unavailable (cleaned_options_data.csv not found).", ha="center", va="center")
        ax.set_axis_off()
        p = paths.figures_dir / "02_surface_fit_quality.png"
        savefig(fig, p)
        out["02_surface_fit_quality.png"] = p

    return out


def generate_local_vol_figures(paths: Paths) -> Dict[str, Path]:
    """
    Uses:
    - output/local_vol_surface.npz (preferred) or output/local_vol_grid.npz
    - output/local_vol_surface.pkl (optional)
    """
    out: Dict[str, Path] = {}

    lv_npz = find_first([
        paths.output_dir / "local_vol_surface.npz",
        paths.output_dir / "local_vol_grid.npz",
    ])
    require(lv_npz, "Run notebooks/03_local_vol_extraction.ipynb and/or notebooks/04_pricing_validation.ipynb")

    data = safe_load_npz(lv_npz)
    sigma = data.get("sigma_loc") if data.get("sigma_loc") is not None else data.get("sigma")
    K = data.get("K")
    T = data.get("T")
    raw_sigma = data.get("raw_sigma_loc")
    domain_mask = data.get("domain_mask")

    if sigma is None:
        raise MissingArtifactError(f"{lv_npz} missing sigma_loc. Re-run notebook 03/04.")
    sigma = np.asarray(sigma, float)

    # if K/T missing in local_vol_grid.npz, try infer from local_vol_surface.npz
    if K is None or T is None:
        alt = find_first([paths.output_dir / "local_vol_surface.npz"])
        if alt and alt.exists():
            d2 = safe_load_npz(alt)
            K = d2.get("K")
            T = d2.get("T")
            if raw_sigma is None:
                raw_sigma = d2.get("raw_sigma_loc")
            if domain_mask is None:
                domain_mask = d2.get("domain_mask")

    if K is None or T is None:
        raise MissingArtifactError(
            f"{lv_npz} missing K/T grids.\n"
            "Fix: Ensure notebook 04 saved local_vol_surface.npz with K and T grids."
        )

    K = np.asarray(K, float)
    T = np.asarray(T, float)

    # 03_local_vol_surface.png — heatmap + 3D (2-panel)
    fig = plot_surface_from_grid(
        Z=sigma, x=K, y=T,
        title="Dupire local volatility surface",
        xlabel="strike K", ylabel="maturity T (years)", zlabel="σ_loc",
    )
    p = paths.figures_dir / "03_local_vol_surface.png"
    savefig(fig, p)
    out["03_local_vol_surface.png"] = p

    # 03_local_vol_slices.png — fixed-T and fixed-K slices
    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    # fixed T near median
    t_idx = int(np.argmin(np.abs(T - np.median(T))))
    ax1.plot(K, sigma[t_idx, :], linewidth=2.0, label=f"T={T[t_idx]:.3f}")
    ax1.set_title("Local vol slice: σ_loc(K | fixed T)")
    ax1.set_xlabel("K")
    ax1.set_ylabel("σ_loc")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    k_idx = int(np.argmin(np.abs(K - np.median(K))))
    ax2.plot(T, sigma[:, k_idx], linewidth=2.0, label=f"K={K[k_idx]:.2f}")
    ax2.set_title("Local vol slice: σ_loc(T | fixed K)")
    ax2.set_xlabel("T")
    ax2.set_ylabel("σ_loc")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    p = paths.figures_dir / "03_local_vol_slices.png"
    savefig(fig, p)
    out["03_local_vol_slices.png"] = p

    # 04_lv_diagnostics.png — gradient / trusted-domain diagnostics summary
    dT = np.gradient(T)
    dK = np.gradient(K)
    grad_T = np.gradient(sigma, T, axis=0)
    grad_K = np.gradient(sigma, K, axis=1)
    grad_mag = np.sqrt(grad_T**2 + grad_K**2)

    sig_f = finite(sigma.ravel())
    g_f = finite(grad_mag.ravel())

    fig = plt.figure(figsize=(14, 5.2))
    ax1 = fig.add_subplot(1, 2, 1)
    im = ax1.imshow(grad_mag, origin="lower", aspect="auto", extent=[K.min(), K.max(), T.min(), T.max()])
    ax1.set_title("Gradient magnitude |∇σ_loc| (diagnostic)")
    ax1.set_xlabel("K")
    ax1.set_ylabel("T")
    cb = fig.colorbar(im, ax=ax1)
    cb.set_label("|∇σ_loc|")

    ax2 = fig.add_subplot(1, 2, 2)
    if domain_mask is not None:
        domain_mask = np.asarray(domain_mask, dtype=float)
        im2 = ax2.imshow(domain_mask, origin="lower", aspect="auto", extent=[K.min(), K.max(), T.min(), T.max()], vmin=0.0, vmax=2.0, cmap="viridis")
        ax2.set_title("Trusted-Domain Classification")
        ax2.set_xlabel("K")
        ax2.set_ylabel("T")
        cb2 = fig.colorbar(im2, ax=ax2)
        cb2.set_ticks([0.0, 1.0, 2.0])
        cb2.set_ticklabels(["extrapolation", "weak", "trusted"])
    else:
        ax2.hist(sig_f, bins=60, alpha=0.85, label="σ_loc")
        ax2b = ax2.twinx()
        ax2b.hist(g_f, bins=60, alpha=0.35, label="|∇σ_loc|")
        ax2.set_title("Distribution of σ_loc and |∇σ_loc|")
        ax2.set_xlabel("value")
        ax2.set_ylabel("count (σ_loc)")
        ax2b.set_ylabel("count (|∇σ_loc|)")
        ax2.grid(True, alpha=0.25)

    p = paths.figures_dir / "04_lv_diagnostics.png"
    savefig(fig, p)
    out["04_lv_diagnostics.png"] = p

    # 04_regularization_impact.png — before/after regularization comparison
    fig = plt.figure(figsize=(14, 5.2))
    ax1 = fig.add_subplot(1, 2, 1)
    im1 = ax1.imshow(sigma, origin="lower", aspect="auto", extent=[K.min(), K.max(), T.min(), T.max()])
    ax1.set_title("Regularized σ_loc")
    ax1.set_xlabel("K")
    ax1.set_ylabel("T")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03)

    ax2 = fig.add_subplot(1, 2, 2)
    if raw_sigma is not None:
        raw_sigma = np.asarray(raw_sigma, float)
        im2 = ax2.imshow(raw_sigma, origin="lower", aspect="auto", extent=[K.min(), K.max(), T.min(), T.max()])
        ax2.set_title("Raw (pre-regularization) σ_loc")
        ax2.set_xlabel("K")
        ax2.set_ylabel("T")
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03)
    else:
        ax2.text(0.5, 0.5, "raw_sigma_loc not present in NPZ.\n(Regularization impact shown only for regularized surface.)",
                 ha="center", va="center")
        ax2.set_axis_off()

    fig.suptitle("Regularization impact (raw vs regularized) — if raw available", y=1.02)
    p = paths.figures_dir / "04_regularization_impact.png"
    savefig(fig, p)
    out["04_regularization_impact.png"] = p

    return out


def generate_pricing_validation_figures(paths: Paths) -> Dict[str, Path]:
    """
    No recalibration allowed, so we validate repricing by comparing:
    - market mid prices in cleaned_options_data.csv
    vs
    - model prices from fitted call grid call_price_grid.npz via bilinear interpolation.

    Produces:
    - 05_repricing_errors.png (scatter + residual hist)
    - 05_repricing_by_moneyness.png
    - 05_repricing_by_maturity.png
    """
    out: Dict[str, Path] = {}

    cleaned_csv = find_first([paths.output_dir / "cleaned_options_data.csv"])
    call_grid = find_first([paths.output_dir / "call_price_grid.npz"])

    require(cleaned_csv, "Run notebooks/01_data_exploration.ipynb")
    require(call_grid, "Run notebooks/02_iv_surface_fitting.ipynb")

    df = safe_read_csv(cleaned_csv)
    grid = safe_load_npz(call_grid)

    C = np.asarray(grid["C"], float)
    K = np.asarray(grid["K"], float)
    T = np.asarray(grid["T"], float)

    col_T = column_any(df, ["T", "time_to_expiry", "ttm", "tau"])
    col_K = column_any(df, ["K", "strike"])
    col_mid = column_any(df, ["mid", "mid_price", "price"])
    col_S0 = column_any(df, ["S0", "spot", "underlying_price", "underlying"])

    if not (col_T and col_K and col_mid):
        raise MissingArtifactError(
            "Cannot build repricing validation: missing T/K/mid columns in cleaned_options_data.csv.\n"
            f"Columns: {list(df.columns)}"
        )

    # calls only if possible
    col_type = column_any(df, ["option_type", "type"])
    if col_type:
        df = df[df[col_type].astype(str).str.lower().str.contains("call")].copy()

    df["_T"] = df[col_T].astype(float)
    df["_K"] = df[col_K].astype(float)
    df["_mkt"] = df[col_mid].astype(float)

    if col_S0:
        df["_S0"] = df[col_S0].astype(float)
    else:
        df["_S0"] = float(np.atleast_1d(grid.get("S0", np.nan))[0])

    # keep within grid bounds and finite
    mask = np.isfinite(df["_T"]) & np.isfinite(df["_K"]) & np.isfinite(df["_mkt"]) & np.isfinite(df["_S0"])
    df = df[mask].copy()
    df = df[(df["_T"] >= T.min()) & (df["_T"] <= T.max()) & (df["_K"] >= K.min()) & (df["_K"] <= K.max())].copy()
    if len(df) == 0:
        raise MissingArtifactError(
            "No market options overlap the fitted call grid bounds.\n"
            "Fix: Ensure notebooks used consistent strike/maturity ranges."
        )

    # sample for plots
    df = df.sample(n=min(len(df), 6000), random_state=1)

    df["_model"] = bilinear_interp_2d(K, T, C, df["_K"].to_numpy(), df["_T"].to_numpy())
    df["_err"] = df["_model"] - df["_mkt"]
    df["_abs_err"] = np.abs(df["_err"])

    # moneyness if available
    df["_moneyness"] = df["_K"] / df["_S0"]

    # 05_repricing_errors.png — scatter + residual hist
    fig = plt.figure(figsize=(14, 5.2))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(df["_mkt"], df["_model"], s=10, alpha=0.45)
    mn = min(df["_mkt"].min(), df["_model"].min())
    mx = max(df["_mkt"].max(), df["_model"].max())
    ax1.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1.2)
    ax1.set_title("Repricing: model vs market (calls)")
    ax1.set_xlabel("market mid price")
    ax1.set_ylabel("model price (from fitted call grid)")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(df["_err"].to_numpy(), bins=60, alpha=0.9)
    ax2.set_title("Residual histogram (model - market)")
    ax2.set_xlabel("price error")
    ax2.set_ylabel("count")
    ax2.grid(True, alpha=0.3)

    p = paths.figures_dir / "05_repricing_errors.png"
    savefig(fig, p)
    out["05_repricing_errors.png"] = p

    # 05_repricing_by_moneyness.png — error vs moneyness
    fig = plt.figure(figsize=(9.5, 6))
    ax = fig.add_subplot(111)
    ax.scatter(df["_moneyness"], df["_err"], s=10, alpha=0.45)
    ax.axhline(0.0, linestyle="--", linewidth=1.2)
    ax.set_title("Repricing error vs moneyness")
    ax.set_xlabel("moneyness K/S")
    ax.set_ylabel("model - market")
    ax.grid(True, alpha=0.3)
    p = paths.figures_dir / "05_repricing_by_moneyness.png"
    savefig(fig, p)
    out["05_repricing_by_moneyness.png"] = p

    # 05_repricing_by_maturity.png — error vs maturity
    fig = plt.figure(figsize=(9.5, 6))
    ax = fig.add_subplot(111)
    ax.scatter(df["_T"], df["_err"], s=10, alpha=0.45)
    ax.axhline(0.0, linestyle="--", linewidth=1.2)
    ax.set_title("Repricing error vs maturity")
    ax.set_xlabel("maturity T (years)")
    ax.set_ylabel("model - market")
    ax.grid(True, alpha=0.3)
    p = paths.figures_dir / "05_repricing_by_maturity.png"
    savefig(fig, p)
    out["05_repricing_by_maturity.png"] = p

    return out


def discover_backtest_csvs(output_dir: Path) -> Tuple[List[Path], List[Path], str]:
    daily_bs = sorted(output_dir.glob("daily_market_backtest_BS*.csv"))
    daily_lv = sorted(output_dir.glob("daily_market_backtest_LocalVol*.csv"))
    if daily_bs and daily_lv:
        return [max(daily_bs, key=lambda p: p.stat().st_mtime)], [max(daily_lv, key=lambda p: p.stat().st_mtime)], "daily_market_panel"

    bs = sorted(output_dir.glob("backtest_BS_*.csv"))
    lv = sorted(output_dir.glob("backtest_LocalVol_*.csv"))
    # allow alternative naming conventions
    if not bs:
        bs = sorted(output_dir.glob("backtest_BlackScholes_*.csv")) + sorted(output_dir.glob("backtest_black_scholes_*.csv"))
    if not lv:
        lv = sorted(output_dir.glob("backtest_LV_*.csv")) + sorted(output_dir.glob("backtest_local_vol_*.csv"))

    if not bs or not lv:
        return bs, lv, "legacy_single_surface"

    def suffix_map(paths: List[Path], prefix: str) -> Dict[str, Path]:
        out: Dict[str, Path] = {}
        for p in paths:
            name = p.name
            if not name.startswith(prefix):
                continue
            out[name[len(prefix) :]] = p
        return out

    bs_map = suffix_map(bs, "backtest_BS_")
    lv_map = suffix_map(lv, "backtest_LocalVol_")
    common = sorted(set(bs_map) & set(lv_map))

    if common:
        def pair_mtime(suffix: str) -> float:
            return max(bs_map[suffix].stat().st_mtime, lv_map[suffix].stat().st_mtime)

        best = max(common, key=pair_mtime)
        return [bs_map[best]], [lv_map[best]], "legacy_single_surface"

    return [max(bs, key=lambda p: p.stat().st_mtime)], [max(lv, key=lambda p: p.stat().st_mtime)], "legacy_single_surface"


def choose_metric_column(df: pd.DataFrame) -> str:
    c = column_any(df, [
        "market_pnl_net",
        "replication_error_net",
        # preferred explicit names (Notebook 5)
        "hedge_error_net",
        "hedge_error_gross",

        # generic fallbacks
        "hedging_error",
        "net_hedging_error",
        "replication_error",
        "pnl",
        "net_pnl",
        "net_p&l",
        "error",
    ])
    if c is None:
        raise MissingArtifactError(
            "Backtest CSV missing a recognizable hedging error / pnl column.\n"
            f"Columns: {list(df.columns)}"
        )
    return c


def choose_cost_column(df: pd.DataFrame) -> Optional[str]:
    return column_any(df, ["total_tx_cost", "transaction_cost", "transaction_costs", "tcost", "cost", "costs", "fees"])


def choose_maturity_column(df: pd.DataFrame) -> Optional[str]:
    return column_any(df, ["target_time_to_expiry", "observed_time_to_expiry", "T", "maturity", "ttm", "time_to_expiry", "tau"])


def choose_moneyness_column(df: pd.DataFrame) -> Optional[str]:
    return column_any(df, ["moneyness", "K_over_S", "k_over_s", "K/S", "k_s", "strike_over_spot"])


def choose_pricing_error_column(df: pd.DataFrame) -> Optional[str]:
    return column_any(df, ["entry_pricing_error", "pricing_error_at_entry", "pricing_error"])


def choose_replication_column(df: pd.DataFrame) -> Optional[str]:
    return column_any(df, ["replication_error_net", "hedge_error_net", "hedging_error", "replication_error"])


def generate_hedging_figures(paths: Paths) -> Dict[str, Path]:
    """
    Uses:
    - output/daily_market_backtest_BS*.csv and output/daily_market_backtest_LocalVol*.csv
      or the legacy single-surface backtest CSVs.

    Produces the 6 required hedging figures.
    """
    out: Dict[str, Path] = {}

    bs_csvs, lv_csvs, backtest_mode = discover_backtest_csvs(paths.output_dir)
    if not bs_csvs or not lv_csvs:
        raise MissingArtifactError(
            "Missing hedging backtest outputs.\n"
            f"Looked for: {paths.output_dir}/daily_market_backtest_BS*.csv and {paths.output_dir}/daily_market_backtest_LocalVol*.csv,\n"
            f"or legacy {paths.output_dir}/backtest_BS_*.csv and {paths.output_dir}/backtest_LocalVol_*.csv\n"
            "Fix: Run notebooks/05_hedging_backtest.ipynb (or the backtest script it calls)."
        )

    # concatenate all panels/files
    bs_df = pd.concat([safe_read_csv(p) for p in bs_csvs], ignore_index=True)
    lv_df = pd.concat([safe_read_csv(p) for p in lv_csvs], ignore_index=True)

    bs_err_col = choose_metric_column(bs_df)
    lv_err_col = choose_metric_column(lv_df)

    bs_err_f = finite(bs_df[bs_err_col].astype(float).to_numpy())
    lv_err_f = finite(lv_df[lv_err_col].astype(float).to_numpy())
    metric_name = "net market PnL" if backtest_mode == "daily_market_panel" else "hedging error"

    # 05_hedging_pnl_distribution.png — BS vs LV net metric histograms
    fig = plt.figure(figsize=(12.5, 6.5))
    ax = fig.add_subplot(111)
    ax.hist(bs_err_f, bins=80, alpha=0.55, label="Black-Scholes")
    ax.hist(lv_err_f, bins=80, alpha=0.55, label="Local Vol")
    ax.set_title(f"Distribution of {metric_name}: BS vs Local Vol")
    ax.set_xlabel(bs_err_col)
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    ax.legend()
    p = paths.figures_dir / "05_hedging_pnl_distribution.png"
    savefig(fig, p)
    out["05_hedging_pnl_distribution.png"] = p

    # 05_error_ecdf.png — ECDF of |metric|
    fig = plt.figure(figsize=(9.5, 6))
    ax = fig.add_subplot(111)
    x1, y1 = ecdf(np.abs(bs_err_f))
    x2, y2 = ecdf(np.abs(lv_err_f))
    ax.plot(x1, y1, linewidth=2.0, label="BS |metric|")
    ax.plot(x2, y2, linewidth=2.0, label="LV |metric|")
    ax.set_title(f"ECDF of |{metric_name}|")
    ax.set_xlabel("|metric|")
    ax.set_ylabel("ECDF")
    ax.grid(True, alpha=0.3)
    ax.legend()
    p = paths.figures_dir / "05_error_ecdf.png"
    savefig(fig, p)
    out["05_error_ecdf.png"] = p

    # 05_qq_plot.png — QQ plot of BS vs LV metrics
    fig = qq_plot(bs_err_f, lv_err_f, f"QQ plot: BS {metric_name} vs LV {metric_name}", "BS quantiles", "LV quantiles")
    p = paths.figures_dir / "05_qq_plot.png"
    savefig(fig, p)
    out["05_qq_plot.png"] = p

    # 05_transaction_costs.png — transaction cost comparison
    bs_cost_col = choose_cost_column(bs_df)
    lv_cost_col = choose_cost_column(lv_df)

    fig = plt.figure(figsize=(9.5, 6))
    ax = fig.add_subplot(111)
    if bs_cost_col and lv_cost_col:
        bs_cost = finite(bs_df[bs_cost_col].astype(float).to_numpy())
        lv_cost = finite(lv_df[lv_cost_col].astype(float).to_numpy())
        ax.hist(bs_cost, bins=70, alpha=0.55, label="BS costs")
        ax.hist(lv_cost, bins=70, alpha=0.55, label="LV costs")
        ax.set_title("Transaction cost comparison (distribution)")
        ax.set_xlabel("transaction cost")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No transaction cost columns found in backtest CSVs.", ha="center", va="center")
        ax.set_axis_off()
    p = paths.figures_dir / "05_transaction_costs.png"
    savefig(fig, p)
    out["05_transaction_costs.png"] = p

    # 05_regime_comparison.png — performance by maturity bucket
    bs_T_col = choose_maturity_column(bs_df)
    lv_T_col = choose_maturity_column(lv_df)

    fig = plt.figure(figsize=(12.5, 6))
    ax = fig.add_subplot(111)

    if bs_T_col and lv_T_col:
        bs_tmp = bs_df[[bs_T_col, bs_err_col]].copy()
        lv_tmp = lv_df[[lv_T_col, lv_err_col]].copy()
        bs_tmp.columns = ["T", "err"]
        lv_tmp.columns = ["T", "err"]

        # maturity buckets
        bs_tmp = bs_tmp[np.isfinite(bs_tmp["T"].astype(float))]
        lv_tmp = lv_tmp[np.isfinite(lv_tmp["T"].astype(float))]
        q = min(5, max(2, bs_tmp["T"].nunique()))
        bs_tmp["T_bucket"] = pd.qcut(bs_tmp["T"].astype(float), q=q, duplicates="drop")
        lv_tmp["T_bucket"] = pd.qcut(lv_tmp["T"].astype(float), q=q, duplicates="drop")

        bs_grp = bs_tmp.groupby("T_bucket", observed=False)["err"].apply(lambda s: np.nanmean(np.abs(s.astype(float))))
        lv_grp = lv_tmp.groupby("T_bucket", observed=False)["err"].apply(lambda s: np.nanmean(np.abs(s.astype(float))))

        # align index
        buckets = [str(b) for b in bs_grp.index]
        x = np.arange(len(buckets))
        ax.plot(x, bs_grp.to_numpy(), marker="o", linewidth=2.0, label="BS mean |metric|")
        ax.plot(x, lv_grp.reindex(bs_grp.index).to_numpy(), marker="o", linewidth=2.0, label="LV mean |metric|")
        ax.set_xticks(x)
        ax.set_xticklabels(buckets, rotation=25, ha="right")
        ax.set_title("Regime comparison by maturity bucket")
        ax.set_xlabel("maturity bucket")
        ax.set_ylabel("mean |metric|")
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Regime comparison unavailable (no maturity column found).", ha="center", va="center")
        ax.set_axis_off()

    p = paths.figures_dir / "05_regime_comparison.png"
    savefig(fig, p)
    out["05_regime_comparison.png"] = p

    # 05_error_decomposition.png — mean vs dispersion components
    bs_mean = float(np.nanmean(bs_err_f)) if bs_err_f.size else np.nan
    bs_std = float(np.nanstd(bs_err_f)) if bs_err_f.size else np.nan
    lv_mean = float(np.nanmean(lv_err_f)) if lv_err_f.size else np.nan
    lv_std = float(np.nanstd(lv_err_f)) if lv_err_f.size else np.nan

    fig = plt.figure(figsize=(9.5, 6))
    ax = fig.add_subplot(111)
    labels = ["BS", "LocalVol"]
    means = [bs_mean, lv_mean]
    stds = [bs_std, lv_std]
    x = np.arange(len(labels))
    ax.bar(x - 0.15, means, width=0.3, label="mean metric")
    ax.bar(x + 0.15, stds, width=0.3, label="dispersion (std metric)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Metric decomposition: mean vs dispersion")
    ax.set_ylabel("value")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    p = paths.figures_dir / "05_error_decomposition.png"
    savefig(fig, p)
    out["05_error_decomposition.png"] = p

    return out


# -----------------------------
# Report writer
# -----------------------------

def write_report(paths: Paths, created: Dict[str, Path], summary: Dict[str, str]) -> None:
    """
    Write output/report.md with relative links to figures.
    """
    rel = lambda p: p.relative_to(paths.output_dir).as_posix()
    title = report_title_from_artifacts(paths.output_dir)
    calib_summary = safe_read_json(paths.output_dir / "calibration_summary.json") if (paths.output_dir / "calibration_summary.json").exists() else {}
    arb_diag = safe_read_json(paths.output_dir / "arbitrage_diagnostics.json") if (paths.output_dir / "arbitrage_diagnostics.json").exists() else {}
    daily_backtest_summary = safe_read_json(paths.output_dir / "daily_market_backtest_summary.json") if (paths.output_dir / "daily_market_backtest_summary.json").exists() else {}
    panel_manifest = {}
    if daily_backtest_summary.get("panel_file"):
        panel_file = Path(daily_backtest_summary["panel_file"])
        manifest_guess = panel_file.parent / f"{panel_file.stem}_manifest.json"
        if manifest_guess.exists():
            panel_manifest = safe_read_json(manifest_guess)
    repricing_path = paths.output_dir / "repricing_validation.csv"
    repricing = safe_read_csv(repricing_path) if repricing_path.exists() else None
    repricing_summary = safe_read_json(paths.output_dir / "repricing_validation_summary.json") if (paths.output_dir / "repricing_validation_summary.json").exists() else {}

    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("This report is generated by `scripts/generate_report.py` from precomputed artifacts in the output directory.")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    if calib_summary:
        raw_fail = int(calib_summary.get("arbitrage_counts_raw", {}).get("n_fail", 0))
        final_fail = int(calib_summary.get("arbitrage_counts", {}).get("n_fail", 0))
        proj_rmse = calib_summary.get("projection_rmse_adjustment")
        cap_frac = 100.0 * float(calib_summary.get("local_vol_cap_fraction", 0.0))
        fit_band = 100.0 * float(calib_summary.get("fit_in_bid_ask_weighted_fraction", 0.0))
        trusted_frac = 100.0 * float(calib_summary.get("trusted_domain_fraction", 0.0))
        lines.append(f"- Calibration starts from an interpolated total-variance surface with `{raw_fail}` static-arbitrage violations and exports a final surface with `{final_fail}` violations.")
        lines.append(f"- The no-arbitrage projection adjusts the raw call grid by RMSE `{_fmt(proj_rmse)}`.")
        lines.append(f"- The weighted midpoint fit lands inside the market spread for `{_fmt(fit_band, 4)}%` of weighted calibration quotes.")
        lines.append(f"- The local-vol surface is regularized conservatively; `{_fmt(cap_frac, 4)}%` of grid cells sit at the configured volatility cap.")
        lines.append(f"- `{_fmt(trusted_frac, 4)}%` of the local-vol grid is classified as trusted interior domain for pricing claims.")
    if summary:
        bs_abs = summary.get("BS mean |market pnl|") or summary.get("BS mean |error|")
        lv_abs = summary.get("LV mean |market pnl|") or summary.get("LV mean |error|")
        if bs_abs is not None and lv_abs is not None:
            if daily_backtest_summary:
                lines.append(f"- On the canonical daily-recalibrated market-panel hedge study, mean absolute net market PnL is `{bs_abs}` for Black-Scholes and `{lv_abs}` for local volatility.")
            else:
                lines.append(f"- On the canonical historical hedge panel, mean absolute hedge error is `{bs_abs}` for Black-Scholes and `{lv_abs}` for local volatility.")
    lines.append("- The repository should be read as a full pipeline: data cleaning -> IV fitting -> arbitrage repair -> Dupire local volatility -> pricing -> hedging.")
    if daily_backtest_summary:
        lines.append("- The canonical hedge study now uses fixed listed contracts, common market entry premiums, and per-date recalibration rather than a single frozen surface.")
    lines.append("")

    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- Figures directory: `{paths.figures_dir.relative_to(paths.output_dir).as_posix()}/`")
    lines.append(f"- Report file: `{paths.report_path.relative_to(paths.output_dir).as_posix()}`")
    if (paths.output_dir / "arbitrage_free_iv_surface.pkl").exists():
        lines.append("- Canonical arbitrage-free IV surface artifact: `arbitrage_free_iv_surface.pkl`")
    if (paths.output_dir / "local_vol_surface.pkl").exists():
        lines.append("- Canonical local-vol surface artifact: `local_vol_surface.pkl`")
    if daily_backtest_summary:
        lines.append("- Canonical historical hedge-study summary: `daily_market_backtest_summary.json`")
    lines.append("")

    if panel_manifest:
        lines.append("## Daily Data Collection")
        lines.append("")
        lines.append("The canonical real-data workflow in this repository is now based on one full SPY option-chain snapshot per trading day, collected at a fixed pre-close time and appended into a dated panel.")
        lines.append("The canonical calibration and hedge study intentionally focus on the liquid short-dated near-ATM SPY core. The repo now uses Theta for option-chain collection and retains only a small Yahoo fallback for SPY spot/carry inputs.")
        lines.append("")
        lines.append("| Collection field | Value |")
        lines.append("| --- | --- |")
        lines.append(f"| Underlying | {panel_manifest.get('ticker', 'SPY')} |")
        lines.append(f"| Provider | {panel_manifest.get('provider', 'daily snapshot archive')} |")
        lines.append(f"| Snapshot time | {panel_manifest.get('snapshot_time_local', '15:45')} {panel_manifest.get('timezone', 'America/New_York')} |")
        lines.append(f"| Panel dates | {panel_manifest.get('n_dates', 0)} |")
        lines.append(f"| Panel date range | {panel_manifest.get('date_start', 'n/a')} to {panel_manifest.get('date_end', 'n/a')} |")
        lines.append(f"| Panel rows | {panel_manifest.get('rows', 0)} |")
        lines.append("")
        lines.append("The raw archive is kept separately from the research filters. Each daily snapshot is first archived, then standardized, then filtered during calibration and contract selection. That separation keeps the data collection step reproducible and makes it possible to revisit filtering choices later without losing raw observations.")
        lines.append("")

    if summary:
        lines.append("## Key numeric summaries (from loaded artifacts)")
        lines.append("")
        for k, v in summary.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    if calib_summary:
        lines.append("## Calibration Quality")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | ---: |")
        lines.append(f"| Spot | {_fmt(calib_summary.get('spot'))} |")
        lines.append(f"| Clean quotes after filters | {_fmt(calib_summary.get('n_after_filters'))} |")
        lines.append(f"| Maturity slices | {_fmt(calib_summary.get('n_slices'))} |")
        lines.append(f"| Raw static-arbitrage fails | {_fmt(calib_summary.get('arbitrage_counts_raw', {}).get('n_fail'))} |")
        lines.append(f"| Final static-arbitrage fails | {_fmt(calib_summary.get('arbitrage_counts', {}).get('n_fail'))} |")
        lines.append(f"| Weighted fit inside bid/ask | {_fmt(100.0 * float(calib_summary.get('fit_in_bid_ask_weighted_fraction', 0.0)), 4)}% |")
        lines.append(f"| Density negative count | {_fmt(calib_summary.get('density_negative'))} |")
        lines.append(f"| Density mass min | {_fmt(calib_summary.get('density_mass_min'))} |")
        lines.append(f"| Density mass max | {_fmt(calib_summary.get('density_mass_max'))} |")
        lines.append(f"| Local-vol min | {_fmt(calib_summary.get('local_vol_min'))} |")
        lines.append(f"| Local-vol max | {_fmt(calib_summary.get('local_vol_max'))} |")
        lines.append(f"| Local-vol cap fraction | {_fmt(100.0 * float(calib_summary.get('local_vol_cap_fraction', 0.0)), 4)}% |")
        lines.append(f"| Trusted local-vol domain fraction | {_fmt(100.0 * float(calib_summary.get('trusted_domain_fraction', 0.0)), 4)}% |")
        lines.append("")
        lines.append("The most important point is that the exported surface is checked after projection, not merely before it. The interpolated total-variance surface is treated as an input to the repair stage rather than as the final object of record.")
        lines.append("")

    if arb_diag:
        raw_pairs = [p for p in arb_diag.get("calendar_pairs_raw", []) if int(p.get("n_negative_adjacent", 0)) > 0]
        if raw_pairs:
            raw_pairs = sorted(raw_pairs, key=lambda d: (d.get("n_negative_adjacent", 0), -d.get("worst_adjacent_diff", 0.0)), reverse=True)
            lines.append("## Raw Calendar-Arbitrage Hotspots")
            lines.append("")
            lines.append("The table below shows the worst adjacent maturity pairs before the projection step. It explains where the static-arbitrage repair had to intervene.")
            lines.append("")
            lines.append("| Shorter T | Longer T | Violating strikes | Worst adjacent diff | Worst strike |")
            lines.append("| ---: | ---: | ---: | ---: | ---: |")
            for row in raw_pairs[:8]:
                lines.append(
                    f"| {_fmt(row.get('T_left'))} | {_fmt(row.get('T_right'))} | "
                    f"{_fmt(row.get('n_negative_adjacent'))} | {_fmt(row.get('worst_adjacent_diff'))} | {_fmt(row.get('worst_strike'))} |"
                )
            lines.append("")

    if repricing is not None and not repricing.empty:
        err_col = column_any(repricing, ["abs_error", "error_abs", "pricing_error_abs"])
        if err_col is None and {"model_price", "market_price"}.issubset(set(repricing.columns)):
            repricing["_abs_error"] = np.abs(repricing["model_price"] - repricing["market_price"])
            err_col = "_abs_error"
        if err_col is not None:
            errs = finite(pd.to_numeric(repricing[err_col], errors="coerce").to_numpy())
            if errs.size:
                lines.append("## Pricing Validation")
                lines.append("")
                lines.append("Pricing validation measures how well the local-vol surface can reproduce listed vanilla prices inside the trusted region of the calibrated domain.")
                lines.append("")
                if repricing_summary:
                    lines.append(f"- Trusted-domain quotes used: `{_fmt(repricing_summary.get('n_eval'))}`")
                lines.append(f"- Mean absolute repricing error: `{_fmt(np.mean(errs))}`")
                lines.append(f"- Median absolute repricing error: `{_fmt(np.median(errs))}`")
                lines.append(f"- 95th percentile absolute repricing error: `{_fmt(np.quantile(errs, 0.95))}`")
                if "model_inside_bid_ask_fraction" in repricing_summary:
                    lines.append(f"- Model inside bid/ask: `{_fmt(100.0 * float(repricing_summary.get('model_inside_bid_ask_fraction', 0.0)), 4)}%`")
                if "mean_normalized_band_violation" in repricing_summary:
                    lines.append(f"- Mean normalized band violation: `{_fmt(repricing_summary.get('mean_normalized_band_violation'))}`")
                lines.append("")

    if daily_backtest_summary and daily_backtest_summary.get("status") == "ok":
        lines.append("## Historical Hedge Study")
        lines.append("")
        lines.append("The canonical hedge evaluation is performed on a dated option panel. For each trade date the repository recalibrates the arbitrage-free surface from that day's listed chain, derives that day's local-vol surface, and then computes BS and local-vol deltas for the same fixed listed contract.")
        lines.append("")
        lines.append("Both models use the same observed market premium at trade entry. The report therefore separates entry pricing bias from replication quality instead of hiding both inside a single PnL number.")
        lines.append("")
        lines.append(f"- Fixed listed contracts in study: `{daily_backtest_summary.get('n_contracts', 0)}`")
        lines.append(f"- Panel source: `{daily_backtest_summary.get('panel_file', '')}`")
        lines.append("")
        lines.append("| Metric | BS | LocalVol |")
        lines.append("| --- | ---: | ---: |")
        bs_stats = daily_backtest_summary.get("models", {}).get("BS", {})
        lv_stats = daily_backtest_summary.get("models", {}).get("LocalVol", {})
        lines.append(f"| Mean entry pricing error | {_fmt(bs_stats.get('pricing_error_mean'))} | {_fmt(lv_stats.get('pricing_error_mean'))} |")
        lines.append(f"| Mean abs entry pricing error | {_fmt(abs(bs_stats.get('pricing_error_mean', np.nan)))} | {_fmt(abs(lv_stats.get('pricing_error_mean', np.nan)))} |")
        lines.append(f"| Mean replication error (net) | {_fmt(bs_stats.get('replication_net_mean'))} | {_fmt(lv_stats.get('replication_net_mean'))} |")
        lines.append(f"| Mean abs market PnL | {_fmt(bs_stats.get('market_pnl_net_mean'))} | {_fmt(lv_stats.get('market_pnl_net_mean'))} |")
        lines.append(f"| Mean total transaction cost | {_fmt(bs_stats.get('mean_total_tx_cost'))} | {_fmt(lv_stats.get('mean_total_tx_cost'))} |")
        lines.append("")
    elif daily_backtest_summary and daily_backtest_summary.get("status") == "insufficient_panel_history":
        lines.append("## Historical Hedge Study")
        lines.append("")
        lines.append("The canonical hedge-study architecture is active, but the dated option panel is still being built. The report therefore stops at the data-collection stage until enough 15:45 ET snapshots exist to support a short-dated listed-contract study.")
        lines.append("")
        lines.append(f"- Snapshot dates collected so far: `{daily_backtest_summary.get('n_panel_dates', 0)}`")
        lines.append(f"- Minimum required before running the canonical short-dated study: `{daily_backtest_summary.get('required_min_panel_dates', 0)}`")
        lines.append("")

    lines.append("## Hedging Interpretation")
    lines.append("")
    if daily_backtest_summary and daily_backtest_summary.get("status") == "ok":
        lines.append("Because the canonical study uses common market premiums and fixed listed contracts, differences between models should now be read much more cleanly. Entry pricing bias, replication error, transaction costs, and net trade PnL are no longer conflated.")
    elif daily_backtest_summary and daily_backtest_summary.get("status") == "insufficient_panel_history":
        lines.append("At this stage the main objective is to accumulate a clean dated panel. Once the panel has enough daily snapshots, the report will switch automatically from collection status to the full short-dated hedge-study comparison.")
    else:
        lines.append("The historical hedge comparison is a downstream test of the entire chain, not just the local-vol formula. A poor hedging result can come from calibration noise, surface regularization, contract selection, regime mismatch, or the limitations of local volatility itself.")
    lines.append("")
    if daily_backtest_summary and daily_backtest_summary.get("status") == "ok":
        lines.append("In this repository the daily hedge study includes explicit transaction costs, per-date surface recalibration, and fixed-contract selection from listed chains. That makes the comparison materially more robust than a single-surface replay against underlying history alone.")
    elif daily_backtest_summary and daily_backtest_summary.get("status") == "insufficient_panel_history":
        lines.append("The canonical real-data study is intentionally short-dated at first. That choice lets the project produce a meaningful listed-contract hedge panel after only a modest number of collected trading days, instead of pretending a long-horizon study is already supported.")
    else:
        lines.append("In this repository the backtest includes explicit transaction costs and uses precomputed PDE price and delta grids. That makes the comparison more realistic than a frictionless textbook experiment.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- The exported surface is arbitrage-free on the calibrated grid and under the interpolation rules used here, but all diagnostics still depend on the quality of the input quote snapshots.")
    lines.append("- Density mass is reported on a finite strike range, so values below one indicate truncation as well as possible fit weakness.")
    lines.append("- Local volatility is a deterministic volatility model. It can be numerically coherent and still underperform Black-Scholes on a chosen hedge panel.")
    lines.append("")

    lines.append("## Figures")
    lines.append("")

    # Keep order by REQUIRED_FIGS
    for section, figs in REQUIRED_FIGS.items():
        lines.append(f"### {section}")
        lines.append("")
        for fn in figs:
            if fn in created:
                lines.append(f"- `{fn}`")
                caption = FIGURE_CAPTIONS.get(fn)
                if caption:
                    lines.append(f"  {caption}")
                lines.append(f"  ![]({rel(created[fn])})")
            else:
                lines.append(f"- `{fn}` *(not generated)*")
        lines.append("")

    lines.append("## Reading Guide")
    lines.append("")
    lines.append("If you are new to this repository, read the notebooks in order. Notebook 1 establishes data quality, Notebook 2 constructs the arbitrage-free surface, Notebook 3 derives local volatility, Notebook 4 checks pricing accuracy, and Notebook 5 evaluates hedging.")
    lines.append("")
    lines.append("This order mirrors the actual dependency structure of the code. Each stage assumes the previous one is already trustworthy.")
    lines.append("")

    paths.report_path.parent.mkdir(parents=True, exist_ok=True)
    paths.report_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate report figures and markdown from existing artifacts.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory (default: auto-detect output/ or outputs/).")
    args = parser.parse_args(argv)

    root = project_root_from_this_file()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else pick_output_dir(root)

    figures_dir = output_dir / "figures"
    report_path = output_dir / "report.md"

    paths = Paths(root=root, output_dir=output_dir, figures_dir=figures_dir, report_path=report_path)
    ensure_dirs(paths)

    log(f"[generate_report] project root: {paths.root}")
    log(f"[generate_report] output dir:   {paths.output_dir}")
    log(f"[generate_report] figures dir:  {paths.figures_dir}")
    log(f"[generate_report] report:       {paths.report_path}")

    created: Dict[str, Path] = {}
    summary: Dict[str, str] = {}

    # Generate sections
    try:
        created.update(generate_data_figures(paths))
        log("✓ Data figures generated")
    except MissingArtifactError as e:
        log(f"✗ Data figures skipped: {e}")

    try:
        created.update(generate_iv_surface_figures(paths))
        log("✓ IV surface figures generated")
    except MissingArtifactError as e:
        log(f"✗ IV surface figures skipped: {e}")

    try:
        created.update(generate_local_vol_figures(paths))
        log("✓ Local vol figures generated")
    except MissingArtifactError as e:
        log(f"✗ Local vol figures skipped: {e}")

    try:
        created.update(generate_pricing_validation_figures(paths))
        log("✓ Pricing validation figures generated")
    except MissingArtifactError as e:
        log(f"✗ Pricing validation figures skipped: {e}")

    daily_summary_path = paths.output_dir / "daily_market_backtest_summary.json"
    daily_summary = safe_read_json(daily_summary_path) if daily_summary_path.exists() else {}
    if daily_summary.get("status") == "insufficient_panel_history":
        summary["Historical hedge study"] = "waiting for enough 15:45 ET snapshots"
        summary["Collected panel dates"] = str(int(daily_summary.get("n_panel_dates", 0)))
        summary["Required panel dates"] = str(int(daily_summary.get("required_min_panel_dates", 0)))
        log("Hedging figures skipped: panel is still accumulating daily snapshots.")
    else:
        try:
            created.update(generate_hedging_figures(paths))
            log("✓ Hedging figures generated")
            calib_summary_path = paths.output_dir / "calibration_summary.json"
            if calib_summary_path.exists():
                calib_summary = safe_read_json(calib_summary_path)
                raw_counts = calib_summary.get("arbitrage_counts_raw")
                final_counts = calib_summary.get("arbitrage_counts")
                if raw_counts is not None:
                    summary["Raw static-arbitrage fails"] = str(int(raw_counts.get("n_fail", 0)))
                if final_counts is not None:
                    summary["Final static-arbitrage fails"] = str(int(final_counts.get("n_fail", 0)))
                if "projection_rmse_adjustment" in calib_summary:
                    summary["Projection RMSE adj"] = f"{float(calib_summary['projection_rmse_adjustment']):.6g}"
                if "local_vol_cap_fraction" in calib_summary:
                    summary["LV cap fraction"] = f"{100.0 * float(calib_summary['local_vol_cap_fraction']):.3g}%"
            bs_csvs, lv_csvs, backtest_mode = discover_backtest_csvs(paths.output_dir)
            bs_df = pd.concat([safe_read_csv(p) for p in bs_csvs], ignore_index=True)
            lv_df = pd.concat([safe_read_csv(p) for p in lv_csvs], ignore_index=True)
            bs_err_col = choose_metric_column(bs_df)
            lv_err_col = choose_metric_column(lv_df)
            bs_err = finite(bs_df[bs_err_col].astype(float).to_numpy())
            lv_err = finite(lv_df[lv_err_col].astype(float).to_numpy())
            metric_label = "market pnl" if backtest_mode == "daily_market_panel" else "error"
            summary[f"BS mean / std ({metric_label})"] = f"{np.nanmean(bs_err):.6g} / {np.nanstd(bs_err):.6g}"
            summary[f"LV mean / std ({metric_label})"] = f"{np.nanmean(lv_err):.6g} / {np.nanstd(lv_err):.6g}"
            summary[f"BS mean |{metric_label}|"] = f"{np.nanmean(np.abs(bs_err)):.6g}"
            summary[f"LV mean |{metric_label}|"] = f"{np.nanmean(np.abs(lv_err)):.6g}"

            if daily_summary_path.exists():
                daily_summary = safe_read_json(daily_summary_path)
                summary["Historical hedge study"] = "daily recalibration on listed option panel"
                summary["Number of fixed listed contracts"] = str(int(daily_summary.get("n_contracts", 0)))
                for model_key in ["BS", "LocalVol"]:
                    model_stats = daily_summary.get("models", {}).get(model_key, {})
                    if "pricing_error_mean" in model_stats:
                        summary[f"{model_key} mean entry pricing error"] = _fmt(model_stats["pricing_error_mean"])
                    if "replication_net_mean" in model_stats:
                        summary[f"{model_key} mean replication error"] = _fmt(model_stats["replication_net_mean"])
                    if "mean_abs_mtm_error" in model_stats:
                        summary[f"{model_key} mean abs MTM error"] = _fmt(model_stats["mean_abs_mtm_error"])
        except MissingArtifactError as e:
            log("")
            log("HEDGING ARTIFACTS MISSING — cannot complete required report.")
            log(str(e))
            log("")
            log("Expected you to run: notebooks/05_hedging_backtest.ipynb (or its script).")
            return 2

    # Verify required figs exist (as files), even if some sections were skipped
    missing_required: List[str] = []
    for _, figs in REQUIRED_FIGS.items():
        for fn in figs:
            if daily_summary.get("status") == "insufficient_panel_history" and fn.startswith("05_"):
                continue
            if fn not in created or not created[fn].exists():
                missing_required.append(fn)

    # Write report regardless; list missing in report if any.
    write_report(paths, created, summary)

    # Final console summary
    log("")
    log("========================================")
    log("REPORT GENERATED")
    log(f"- {paths.report_path}")
    log(f"- figures in {paths.figures_dir}")
    log("")
    log("Created figures:")
    for fn in sorted(created.keys()):
        log(f"  - {created[fn]}")
    if missing_required:
        log("")
        log("WARNING: Some required figure filenames were not generated:")
        for fn in missing_required:
            log(f"  - {fn}")
        log("See console messages above for which artifacts were missing.")
    log("========================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
