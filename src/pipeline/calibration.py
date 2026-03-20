from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from arbitrage.density import breeden_litzenberger_density
from arbitrage.projection import (
    project_call_surface_static_arbitrage,
    summarize_calendar_violations,
)
from arbitrage.static_checks import check_static_arbitrage_calls
from implied_volatility.black_scholes import (
    bs_price,
    no_arbitrage_bounds_call,
    put_call_parity_call_from_put,
)
from implied_volatility.iv_solver import IVSolverConfig, implied_vol
from implied_volatility.surface import (
    ArbitrageFreeIVSurface,
    call_price_grid_from_iv_surface,
)
from local_volatility.dupire import DupireConfig, dupire_local_vol_from_call_grid
from local_volatility.regularization import (
    LocalVolRegularizationConfig,
    regularize_local_vol,
)
from local_volatility.surface import LocalVolSurface
from market_data.transforms import TransformConfig, add_derived_columns
from market_data.validators import ValidationConfig, validate_and_clean
from surface_fitting.spline import SplineSurface, fit_spline_surface
from utils.interpolation import Interp1D


OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class SnapshotCalibrationConfig:
    option_type: OptionType = "call"
    min_volume: float = 25.0
    min_open_interest: float = 200.0
    moneyness_min: float = 0.90
    moneyness_max: float = 1.10
    max_relative_spread: float = 0.05
    iv_min: float = 0.03
    iv_max: float = 1.50
    min_points_per_slice: int = 10
    min_maturity_years: float = 7.0 / 365.0
    n_strikes: int = 45
    arbitrage_tol: float = 1e-10
    dupire_floor_density: float = 1e-8
    dupire_cap_sigma: float = 5.0
    trusted_x_quantile_low: float = 0.20
    trusted_x_quantile_high: float = 0.80
    trusted_density_mass_min: float = 0.85
    trusted_edge_time_rows: int = 2
    regularization: LocalVolRegularizationConfig = LocalVolRegularizationConfig(
        repair_short_end=True,
        short_end_min_coverage=0.50,
        short_end_valid_vol_min=0.02,
        short_end_valid_vol_max=1.00,
        short_end_anchor_blend=0.20,
        smooth=True,
        gaussian_sigma_T=0.4,
        gaussian_sigma_K=0.6,
        preserve_T0=True,
        min_vol=0.05,
        max_vol=2.0,
    )


@dataclass(frozen=True)
class SnapshotCalibrationArtifacts:
    ticker: str
    valuation_date: str
    raw_input: pd.DataFrame
    cleaned_quotes: pd.DataFrame
    validation_report: dict[str, Any]
    variance_surface: SplineSurface
    surface_slice_summary: pd.DataFrame
    arbitrage_free_iv_surface: ArbitrageFreeIVSurface
    local_vol_surface: LocalVolSurface
    K_grid: np.ndarray
    T_grid: np.ndarray
    call_grid_raw: np.ndarray
    call_grid: np.ndarray
    iv_grid: np.ndarray
    density_grid: np.ndarray
    density_mass: np.ndarray
    sigma_loc_raw: np.ndarray
    sigma_loc: np.ndarray
    diagnostics: dict[str, Any]
    summary: dict[str, Any]


def _coerce_snapshot_frame(df_raw: pd.DataFrame, *, ticker: str, valuation_date: str) -> pd.DataFrame:
    out = df_raw.copy()
    if "date" in out.columns:
        parsed = pd.to_datetime(out["date"], errors="coerce")
        out["date"] = parsed.fillna(pd.Timestamp(valuation_date))
    else:
        out["date"] = pd.Timestamp(valuation_date)

    if "underlying" in out.columns:
        out["underlying"] = out["underlying"].fillna(ticker)
    else:
        out["underlying"] = ticker
    return out


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    if values.size == 0 or weights.size != values.size:
        raise ValueError("weighted quantile requires aligned non-empty arrays")
    order = np.argsort(values)
    values = values[order]
    weights = np.maximum(weights[order], 0.0)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        return float(np.quantile(values, q))
    cdf = np.cumsum(weights) / total
    idx = int(np.searchsorted(cdf, float(np.clip(q, 0.0, 1.0)), side="left"))
    idx = min(max(idx, 0), values.size - 1)
    return float(values[idx])


def _compute_quote_weights(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rel_spread = pd.to_numeric(out["relative_spread"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    rel_spread = rel_spread.fillna(rel_spread.median() if rel_spread.notna().any() else 0.05)
    rel_spread = np.clip(rel_spread.to_numpy(dtype=float), 1e-4, 1.0)

    volume = pd.to_numeric(out.get("volume"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    open_interest = pd.to_numeric(out.get("open_interest"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    liquidity = 1.0 + 0.50 * np.log1p(np.maximum(volume, 0.0)) + 0.35 * np.log1p(np.maximum(open_interest, 0.0))
    spread_score = 1.0 / np.power(rel_spread, 1.15)
    otm_bonus = np.where(out["is_otm_source"].to_numpy(dtype=bool), 1.0, 0.85)

    raw_weight = otm_bonus * liquidity * spread_score
    median = float(np.nanmedian(raw_weight[np.isfinite(raw_weight)])) if np.any(np.isfinite(raw_weight)) else 1.0
    median = max(median, 1e-8)
    out["quote_weight"] = raw_weight / median
    out["quote_weight"] = np.clip(out["quote_weight"], 0.05, 50.0)
    return out


def _surface_call_prices_at_quotes(
    variance_surface: SplineSurface,
    df: pd.DataFrame,
    *,
    r: float,
    q: float,
) -> np.ndarray:
    x = pd.to_numeric(df["log_moneyness"], errors="coerce").to_numpy(dtype=float)
    T = pd.to_numeric(df["time_to_expiry"], errors="coerce").to_numpy(dtype=float)
    S = pd.to_numeric(df["underlying_price"], errors="coerce").to_numpy(dtype=float)
    K = pd.to_numeric(df["strike"], errors="coerce").to_numpy(dtype=float)

    model_iv = np.asarray(variance_surface.iv(x, T), dtype=float)
    prices = np.empty_like(model_iv, dtype=float)
    for i, (Si, Ki, Ti, sigi) in enumerate(zip(S, K, T, model_iv)):
        prices[i] = float(
            bs_price(
                S=np.array([Si], dtype=float),
                K=np.array([Ki], dtype=float),
                T=np.array([Ti], dtype=float),
                r=float(r),
                q=float(q),
                sigma=np.array([max(float(sigi), 1e-8)], dtype=float),
                option_type="call",
            )[0]
        )
    return prices


def _fit_snapshot_total_variance_surface(
    df: pd.DataFrame,
    cfg: SnapshotCalibrationConfig,
    *,
    value_col: str = "w",
    weight_col: str = "quote_weight",
) -> tuple[SplineSurface, pd.DataFrame, list[float]]:
    grouped = (
        df.groupby("time_to_expiry")
        .filter(lambda g: len(g) >= int(cfg.min_points_per_slice))
        .groupby("time_to_expiry")
    )

    T_slices: list[float] = []
    x_slices: list[np.ndarray] = []
    w_slices: list[np.ndarray] = []
    rows: list[dict[str, float]] = []

    for T, group in grouped:
        if float(T) < float(cfg.min_maturity_years):
            continue
        group_sorted = group.sort_values("log_moneyness")
        if len(group_sorted) < int(cfg.min_points_per_slice):
            continue

        x = group_sorted["log_moneyness"].to_numpy(dtype=float)
        w = group_sorted[value_col].to_numpy(dtype=float)
        weights = (
            group_sorted[weight_col].to_numpy(dtype=float)
            if weight_col in group_sorted.columns
            else np.ones(len(group_sorted), dtype=float)
        )
        T_slices.append(float(T))
        x_slices.append(x)
        w_slices.append(w)
        rows.append(
            {
                "T": float(T),
                "n_points": int(len(group_sorted)),
                "x_min": float(np.min(x)),
                "x_max": float(np.max(x)),
                "x_q15": _weighted_quantile(x, weights, 0.15),
                "x_q85": _weighted_quantile(x, weights, 0.85),
                "w_min": float(np.min(w)),
                "w_max": float(np.max(w)),
                "quote_weight_sum": float(np.sum(weights)),
            }
        )

    if len(T_slices) < 3:
        raise RuntimeError(
            "Too few maturities with enough strikes to build the interpolated total-variance surface"
        )

    x_mins = [float(np.min(x)) for x in x_slices]
    x_maxs = [float(np.max(x)) for x in x_slices]
    x_lo = max(x_mins)
    x_hi = min(x_maxs)
    if not np.isfinite(x_lo) or not np.isfinite(x_hi) or x_hi <= x_lo:
        x_lo = max(float(np.quantile(x, 0.05)) for x in x_slices)
        x_hi = min(float(np.quantile(x, 0.95)) for x in x_slices)
    if not np.isfinite(x_lo) or not np.isfinite(x_hi) or x_hi <= x_lo:
        x_all = np.concatenate(x_slices)
        x_lo = float(np.min(x_all))
        x_hi = float(np.max(x_all))
    x_grid = np.linspace(x_lo, x_hi, 61)

    variance_surface = fit_spline_surface(
        x_slices=x_slices,
        w_slices=w_slices,
        T_slices=T_slices,
        weights_slices=[
            df[df["time_to_expiry"] == T].sort_values("log_moneyness")[weight_col].to_numpy(dtype=float)
            if weight_col in df.columns
            else np.ones(len(x), dtype=float)
            for T, x in zip(T_slices, x_slices)
        ],
        x_grid=x_grid,
        T_grid=np.sort(np.asarray(T_slices, dtype=float)),
        kind_x="pchip",
        kind_T="pchip",
        method_2d="linear",
    )
    slice_summary_df = pd.DataFrame(rows).sort_values("T").reset_index(drop=True)
    return variance_surface, slice_summary_df, list(np.asarray(variance_surface.T_grid, dtype=float))


def _prepare_synthetic_call_quotes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    numeric_cols = [
        "underlying_price",
        "strike",
        "time_to_expiry",
        "risk_free_rate",
        "dividend_yield",
        "bid",
        "ask",
        "mid",
        "volume",
        "open_interest",
        "forward",
        "log_moneyness",
        "relative_spread",
        "moneyness",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["option_type"] = out["option_type"].astype(str).str.lower()
    out = out[out["option_type"].isin(["call", "put"])].copy()
    if out.empty:
        raise RuntimeError("No call/put quotes available for synthetic call-surface construction")

    S = out["underlying_price"].to_numpy(dtype=float)
    K = out["strike"].to_numpy(dtype=float)
    T = out["time_to_expiry"].to_numpy(dtype=float)
    r = out["risk_free_rate"].fillna(0.0).to_numpy(dtype=float)
    q = out["dividend_yield"].fillna(0.0).to_numpy(dtype=float)

    lower, upper = no_arbitrage_bounds_call(S, K, T, r=r, q=q)
    forward = out["forward"].to_numpy(dtype=float)
    is_call = out["option_type"].eq("call").to_numpy()
    is_put = out["option_type"].eq("put").to_numpy()

    bid = out["bid"].to_numpy(dtype=float)
    ask = out["ask"].to_numpy(dtype=float)
    mid = out["mid"].to_numpy(dtype=float)

    surface_bid = bid.copy()
    surface_ask = ask.copy()
    surface_mid = mid.copy()

    if np.any(is_put):
        surface_bid[is_put] = put_call_parity_call_from_put(
            bid[is_put], S[is_put], K[is_put], T[is_put], r=r[is_put], q=q[is_put]
        )
        surface_ask[is_put] = put_call_parity_call_from_put(
            ask[is_put], S[is_put], K[is_put], T[is_put], r=r[is_put], q=q[is_put]
        )
        surface_mid[is_put] = put_call_parity_call_from_put(
            mid[is_put], S[is_put], K[is_put], T[is_put], r=r[is_put], q=q[is_put]
        )

    surface_bid = np.clip(surface_bid, lower, upper)
    surface_ask = np.clip(surface_ask, lower, upper)
    surface_mid = np.clip(surface_mid, lower, upper)
    surface_bid = np.minimum(surface_bid, surface_ask)
    surface_ask = np.maximum(surface_bid, surface_ask)
    surface_mid = np.clip(surface_mid, surface_bid, surface_ask)

    out["surface_bid"] = surface_bid
    out["surface_ask"] = surface_ask
    out["surface_mid"] = surface_mid
    out["surface_option_type"] = "call"
    out["surface_quote_source"] = np.where(is_put, "put_parity", "call_direct")
    out["is_otm_source"] = np.where(is_call, K >= forward, K <= forward)
    out["quote_volume"] = out.get("volume", 0.0)
    out["quote_open_interest"] = out.get("open_interest", 0.0)
    out["quote_relative_spread"] = pd.to_numeric(out.get("relative_spread"), errors="coerce").fillna(np.inf)
    out["source_rank"] = np.where(out["is_otm_source"], 0, 1)

    grouped = []
    for (_, _), group in out.groupby(["time_to_expiry", "strike"], sort=True):
        best = (
            group.sort_values(
                ["source_rank", "quote_relative_spread", "quote_open_interest", "quote_volume"],
                ascending=[True, True, False, False],
            )
            .iloc[0]
            .copy()
        )
        grouped.append(best)

    merged = pd.DataFrame(grouped).reset_index(drop=True)
    merged["option_type"] = "call"
    merged["bid"] = merged["surface_bid"]
    merged["ask"] = merged["surface_ask"]
    merged["mid"] = merged["surface_mid"]

    merge_report = {
        "n_input_quotes": int(len(out)),
        "n_output_quotes": int(len(merged)),
        "source_counts": merged["surface_quote_source"].value_counts(dropna=False).to_dict(),
        "otm_source_fraction": float(np.mean(merged["is_otm_source"].to_numpy(dtype=float))) if len(merged) else 0.0,
    }
    return merged, merge_report


def _bid_ask_fit_diagnostics(
    variance_surface: SplineSurface,
    df: pd.DataFrame,
    *,
    r: float,
    q: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = df.copy()
    model_prices = _surface_call_prices_at_quotes(variance_surface, work, r=r, q=q)
    work["surface_model_price"] = model_prices
    bid = pd.to_numeric(work["bid"], errors="coerce").to_numpy(dtype=float)
    ask = pd.to_numeric(work["ask"], errors="coerce").to_numpy(dtype=float)
    spread = np.maximum(ask - bid, 1e-8)
    clipped = np.clip(model_prices, bid, ask)
    violation = np.abs(model_prices - clipped)
    in_band = violation <= 1e-10
    weight = pd.to_numeric(work["quote_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    work["fit_residual_mid"] = work["surface_model_price"] - pd.to_numeric(work["mid"], errors="coerce")
    work["fit_inside_bid_ask"] = in_band
    work["normalized_band_violation"] = violation / spread
    diagnostics = {
        "approach": "diagnostic_only_mid_fit",
        "iterations": 1,
        "history": [
            {
                "iteration": 0.0,
                "in_band_fraction": float(np.mean(in_band)) if len(in_band) else 0.0,
                "weighted_in_band_fraction": float(np.sum(weight[in_band]) / np.sum(weight)) if np.sum(weight) > 0 else 0.0,
                "mean_normalized_violation": float(np.mean(violation / spread)) if len(spread) else 0.0,
            }
        ],
        "final_in_band_fraction": float(np.mean(in_band)) if len(in_band) else 0.0,
        "final_weighted_in_band_fraction": float(np.sum(weight[in_band]) / np.sum(weight)) if np.sum(weight) > 0 else 0.0,
    }
    return work, diagnostics


def _build_trusted_domain_mask(
    df: pd.DataFrame,
    *,
    K_grid: np.ndarray,
    T_grid: np.ndarray,
    S0: float,
    r: float,
    q: float,
    density_mass: np.ndarray,
    sigma_loc: np.ndarray,
    cfg: SnapshotCalibrationConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    grouped = df.groupby("time_to_expiry")
    row_summary = []
    for T, group in grouped:
        x = group["log_moneyness"].to_numpy(dtype=float)
        w = group["quote_weight"].to_numpy(dtype=float) if "quote_weight" in group.columns else np.ones(len(group), dtype=float)
        row_summary.append(
            {
                "T": float(T),
                "support_lo": float(np.min(x)),
                "support_hi": float(np.max(x)),
                "trusted_lo": _weighted_quantile(x, w, float(cfg.trusted_x_quantile_low)),
                "trusted_hi": _weighted_quantile(x, w, float(cfg.trusted_x_quantile_high)),
            }
        )
    row_df = pd.DataFrame(row_summary).sort_values("T").reset_index(drop=True)
    if row_df.empty:
        mask = np.zeros((T_grid.size, K_grid.size), dtype=np.int8)
        return mask, {"trusted_fraction": 0.0, "supported_fraction": 0.0}

    lo_support_fn = Interp1D(row_df["T"].to_numpy(dtype=float), row_df["support_lo"].to_numpy(dtype=float), kind="linear", fill_value="extrapolate")
    hi_support_fn = Interp1D(row_df["T"].to_numpy(dtype=float), row_df["support_hi"].to_numpy(dtype=float), kind="linear", fill_value="extrapolate")
    lo_trusted_fn = Interp1D(row_df["T"].to_numpy(dtype=float), row_df["trusted_lo"].to_numpy(dtype=float), kind="linear", fill_value="extrapolate")
    hi_trusted_fn = Interp1D(row_df["T"].to_numpy(dtype=float), row_df["trusted_hi"].to_numpy(dtype=float), kind="linear", fill_value="extrapolate")

    domain_mask = np.zeros((T_grid.size, K_grid.size), dtype=np.int8)
    cap_level = float(cfg.regularization.max_vol) if cfg.regularization.max_vol is not None else None

    for j, Tj in enumerate(T_grid):
        F = float(S0 * np.exp((r - q) * Tj))
        x_row = np.log(np.asarray(K_grid, dtype=float) / F)
        support_lo = float(lo_support_fn(np.array([Tj], dtype=float))[0])
        support_hi = float(hi_support_fn(np.array([Tj], dtype=float))[0])
        trusted_lo = float(lo_trusted_fn(np.array([Tj], dtype=float))[0])
        trusted_hi = float(hi_trusted_fn(np.array([Tj], dtype=float))[0])

        supported = (x_row >= support_lo) & (x_row <= support_hi)
        trusted = (x_row >= trusted_lo) & (x_row <= trusted_hi)
        domain_mask[j, supported] = 1
        domain_mask[j, trusted] = 2

        if density_mass[j] < float(cfg.trusted_density_mass_min):
            domain_mask[j, :] = np.minimum(domain_mask[j, :], 1)

        if int(cfg.trusted_edge_time_rows) > 0 and (
            j < int(cfg.trusted_edge_time_rows) or j >= (len(T_grid) - int(cfg.trusted_edge_time_rows))
        ):
            domain_mask[j, :] = np.minimum(domain_mask[j, :], 1)

    if cap_level is not None:
        cap_mask = np.isclose(sigma_loc, cap_level, atol=1e-6)
        domain_mask[cap_mask] = np.minimum(domain_mask[cap_mask], 1)

    summary = {
        "trusted_fraction": float(np.mean(domain_mask == 2)),
        "supported_fraction": float(np.mean(domain_mask >= 1)),
        "trusted_cells": int(np.sum(domain_mask == 2)),
        "supported_cells": int(np.sum(domain_mask >= 1)),
        "extrapolation_cells": int(np.sum(domain_mask == 0)),
        "row_boundaries": row_df.to_dict(orient="records"),
    }
    return domain_mask, summary


def calibrate_option_snapshot(
    df_raw: pd.DataFrame,
    *,
    ticker: str,
    valuation_date: str,
    cfg: SnapshotCalibrationConfig = SnapshotCalibrationConfig(),
) -> SnapshotCalibrationArtifacts:
    df_raw = _coerce_snapshot_frame(df_raw, ticker=ticker, valuation_date=valuation_date)

    df_clean, validation_report = validate_and_clean(
        df_raw,
        cfg=ValidationConfig(
            min_volume=float(cfg.min_volume),
            min_open_interest=float(cfg.min_open_interest),
            enforce_bid_ask=True,
            allow_negative_mid=False,
        ),
    )
    df = add_derived_columns(df_clean, cfg=TransformConfig())

    if df.empty:
        raise RuntimeError(f"No valid option data left after validation for {ticker} on {valuation_date}")

    df["bid_ask_spread"] = df["ask"] - df["bid"]
    df["relative_spread"] = df["bid_ask_spread"] / df["mid"]
    df["moneyness"] = df["strike"] / df["underlying_price"]
    df["days_to_expiry"] = df["time_to_expiry"] * 365.0

    df = df[
        (df["moneyness"] >= float(cfg.moneyness_min))
        & (df["moneyness"] <= float(cfg.moneyness_max))
        & (df["relative_spread"] <= float(cfg.max_relative_spread))
    ].copy()

    if df.empty:
        raise RuntimeError("No data left after option-type / moneyness / spread filters")

    df, merge_report = _prepare_synthetic_call_quotes(df)
    df = _compute_quote_weights(df)

    r = float(pd.to_numeric(df["risk_free_rate"], errors="coerce").fillna(0.0).median())
    q = float(pd.to_numeric(df["dividend_yield"], errors="coerce").fillna(0.0).median())
    S0 = float(pd.to_numeric(df["underlying_price"], errors="coerce").iloc[0])

    iv_cfg = IVSolverConfig(method="hybrid", max_iter=100, tol=1e-10)
    df["iv"] = implied_vol(
        prices=df["mid"].to_numpy(dtype=float),
        S=df["underlying_price"].to_numpy(dtype=float),
        K=df["strike"].to_numpy(dtype=float),
        T=df["time_to_expiry"].to_numpy(dtype=float),
        r=r,
        q=q,
        option_type="call",
        cfg=iv_cfg,
    )
    df = df[
        np.isfinite(df["iv"])
        & (df["iv"] >= float(cfg.iv_min))
        & (df["iv"] <= float(cfg.iv_max))
    ].copy()
    df["w"] = (df["iv"] ** 2) * df["time_to_expiry"]

    variance_surface, slice_summary_df, T_slices = _fit_snapshot_total_variance_surface(
        df,
        cfg,
        value_col="w",
        weight_col="quote_weight",
    )
    df, band_fit_diagnostics = _bid_ask_fit_diagnostics(variance_surface, df, r=r, q=q)

    K_grid = np.linspace(float(df["strike"].min()), float(df["strike"].max()), int(cfg.n_strikes))
    T_grid = np.sort(np.asarray(T_slices, dtype=float))

    C_grid_raw = call_price_grid_from_iv_surface(
        variance_surface,
        S0=S0,
        K_grid=K_grid,
        T_grid=T_grid,
        r=r,
        q=q,
        coord="x_T",
    )

    arb_raw = check_static_arbitrage_calls(
        C_grid_raw,
        K_grid,
        T_grid,
        S0=S0,
        r=r,
        q=q,
        tol=float(cfg.arbitrage_tol),
    )
    projection = project_call_surface_static_arbitrage(
        C_grid_raw,
        K_grid,
        T_grid,
        S0=S0,
        r=r,
        q=q,
        tol=float(cfg.arbitrage_tol),
    )
    C_grid = projection.C_projected
    arb = check_static_arbitrage_calls(
        C_grid,
        K_grid,
        T_grid,
        S0=S0,
        r=r,
        q=q,
        tol=float(cfg.arbitrage_tol),
    )

    af_iv_surface = ArbitrageFreeIVSurface(
        K_grid=K_grid,
        T_grid=T_grid,
        C_grid=C_grid,
        S0=S0,
        r=r,
        q=q,
        iv_solver_cfg=iv_cfg,
    )
    af_iv_grid = af_iv_surface.to_iv_surface().iv_grid
    density = breeden_litzenberger_density(C_grid, K_grid, T_grid, r=r, tol=float(cfg.arbitrage_tol))

    sigma_loc_raw = dupire_local_vol_from_call_grid(
        C_grid,
        K_grid,
        T_grid,
        r=r,
        q=q,
        cfg=DupireConfig(
            floor_density=float(cfg.dupire_floor_density),
            cap_sigma=float(cfg.dupire_cap_sigma),
        ),
    )
    sigma_loc = regularize_local_vol(sigma_loc_raw, cfg=cfg.regularization)
    domain_mask, domain_summary = _build_trusted_domain_mask(
        df,
        K_grid=K_grid,
        T_grid=T_grid,
        S0=S0,
        r=r,
        q=q,
        density_mass=density.mass,
        sigma_loc=sigma_loc,
        cfg=cfg,
    )
    lv_surface = LocalVolSurface.from_dupire_grid(sigma_loc, K_grid, T_grid, domain_mask=domain_mask)

    diagnostics = {
        "calendar_pairs_raw": summarize_calendar_violations(C_grid_raw, K_grid, T_grid),
        "calendar_pairs_final": summarize_calendar_violations(C_grid, K_grid, T_grid),
        "projection": {
            "iterations": int(projection.iterations),
            "rmse_adjustment": float(projection.rmse_adjustment),
            "max_abs_adjustment": float(projection.max_abs_adjustment),
            "counts_before": projection.counts_before,
            "counts_after": projection.counts_after,
            "history": projection.history,
        },
        "surface_quote_merge": merge_report,
        "bid_ask_fit": band_fit_diagnostics,
        "trusted_domain": domain_summary,
        "surface_model": {
            "kind": "interpolated_total_variance",
            "kind_x": "pchip",
            "kind_T": "pchip",
            "method_2d": "linear",
        },
        "arbitrage_free_iv_surface": {
            "artifact_npz": "arbitrage_free_iv_surface.npz",
            "artifact_pkl": "arbitrage_free_iv_surface.pkl",
        },
    }

    cap_level = float(cfg.regularization.max_vol) if cfg.regularization.max_vol is not None else None
    if cap_level is None:
        cap_fraction = 0.0
    else:
        cap_fraction = float(np.mean(np.isclose(sigma_loc, cap_level)))

    summary = {
        "ticker": ticker,
        "date": valuation_date,
        "surface_model": "interpolated_total_variance",
        "n_raw": int(len(df_raw)),
        "n_after_validation": int(validation_report["n_out"]),
        "n_after_filters": int(len(df)),
        "n_slices": int(len(T_slices)),
        "surface_source_counts": merge_report["source_counts"],
        "surface_otm_source_fraction": merge_report["otm_source_fraction"],
        "fit_in_bid_ask_fraction": band_fit_diagnostics["final_in_band_fraction"],
        "fit_in_bid_ask_weighted_fraction": band_fit_diagnostics["final_weighted_in_band_fraction"],
        "spot": S0,
        "r": r,
        "q": q,
        "arbitrage_counts_raw": arb_raw.counts,
        "arbitrage_counts": arb.counts,
        "projection_iterations": int(projection.iterations),
        "projection_rmse_adjustment": float(projection.rmse_adjustment),
        "projection_max_abs_adjustment": float(projection.max_abs_adjustment),
        "arbitrage_free_iv_surface_artifact": True,
        "density_negative": int(np.sum(density.density < -1e-10)),
        "density_mass_min": float(np.min(density.mass)),
        "density_mass_max": float(np.max(density.mass)),
        "local_vol_min": float(np.nanmin(sigma_loc)),
        "local_vol_max": float(np.nanmax(sigma_loc)),
        "local_vol_cap_fraction": cap_fraction,
        "trusted_domain_fraction": float(domain_summary["trusted_fraction"]),
        "supported_domain_fraction": float(domain_summary["supported_fraction"]),
    }

    return SnapshotCalibrationArtifacts(
        ticker=ticker,
        valuation_date=valuation_date,
        raw_input=df_raw,
        cleaned_quotes=df,
        validation_report=validation_report,
        variance_surface=variance_surface,
        surface_slice_summary=slice_summary_df,
        arbitrage_free_iv_surface=af_iv_surface,
        local_vol_surface=lv_surface,
        K_grid=K_grid,
        T_grid=T_grid,
        call_grid_raw=C_grid_raw,
        call_grid=C_grid,
        iv_grid=af_iv_grid,
        density_grid=density.density,
        density_mass=density.mass,
        sigma_loc_raw=sigma_loc_raw,
        sigma_loc=sigma_loc,
        diagnostics=diagnostics,
        summary=summary,
    )


def save_calibration_artifacts(artifacts: SnapshotCalibrationArtifacts, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    artifacts.cleaned_quotes.to_csv(out / "cleaned_options_data.csv", index=False)
    artifacts.cleaned_quotes.to_parquet(out / "chain.parquet")
    artifacts.surface_slice_summary.to_csv(out / "surface_slice_summary.csv", index=False)

    np.savez(
        out / "call_price_grid.npz",
        C=artifacts.call_grid,
        C_raw=artifacts.call_grid_raw,
        K=artifacts.K_grid,
        T=artifacts.T_grid,
        S0=artifacts.summary["spot"],
        r=artifacts.summary["r"],
        q=artifacts.summary["q"],
    )
    np.savez(
        out / "arbitrage_free_iv_surface.npz",
        iv=artifacts.iv_grid,
        C=artifacts.call_grid,
        K=artifacts.K_grid,
        T=artifacts.T_grid,
        S0=artifacts.summary["spot"],
        r=artifacts.summary["r"],
        q=artifacts.summary["q"],
    )
    with open(out / "arbitrage_free_iv_surface.pkl", "wb") as f:
        pickle.dump(artifacts.arbitrage_free_iv_surface, f)

    np.savez(
        out / "density.npz",
        density=artifacts.density_grid,
        mass=artifacts.density_mass,
        K=artifacts.K_grid,
        T=artifacts.T_grid,
    )
    np.savez(
        out / "local_vol_grid.npz",
        sigma_loc=artifacts.sigma_loc,
        sigma_loc_raw=artifacts.sigma_loc_raw,
        K_grid=artifacts.K_grid,
        T_grid=artifacts.T_grid,
        domain_mask=artifacts.local_vol_surface.domain_mask,
        S0=artifacts.summary["spot"],
        r=artifacts.summary["r"],
        q=artifacts.summary["q"],
    )
    np.savez(
        out / "local_vol_surface.npz",
        sigma_loc=artifacts.sigma_loc,
        raw_sigma_loc=artifacts.sigma_loc_raw,
        K=artifacts.K_grid,
        T=artifacts.T_grid,
        domain_mask=artifacts.local_vol_surface.domain_mask,
        S0=artifacts.summary["spot"],
        r=artifacts.summary["r"],
        q=artifacts.summary["q"],
    )
    with open(out / "local_vol_surface.pkl", "wb") as f:
        pickle.dump(artifacts.local_vol_surface, f)

    (out / "arbitrage_diagnostics.json").write_text(json.dumps(artifacts.diagnostics, indent=2), encoding="utf-8")
    (out / "calibration_summary.json").write_text(json.dumps(artifacts.summary, indent=2), encoding="utf-8")
