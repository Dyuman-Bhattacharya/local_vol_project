from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

from hedging.transaction_costs import TransactionCostModel
from implied_volatility.black_scholes import bs_delta, bs_price
from market_data.transforms import TransformConfig, add_derived_columns
from pipeline.calibration import (
    SnapshotCalibrationArtifacts,
    SnapshotCalibrationConfig,
    calibrate_option_snapshot,
    save_calibration_artifacts,
)
from pricing.pde_solver import PDEConfig, solve_european_pde_local_vol_surface


OptionType = Literal["call", "put"]
StrikeMode = Literal["atm", "fixed_strike", "fixed_moneyness"]
ModelKind = Literal["BS", "LocalVol"]


@dataclass(frozen=True)
class ListedContract:
    trade_id: int
    entry_date: pd.Timestamp
    maturity_date: pd.Timestamp
    strike: float
    option_type: OptionType
    entry_spot: float
    market_premium: float
    target_time_to_expiry: float
    observed_time_to_expiry: float
    quote_volume: float
    quote_open_interest: float
    quote_relative_spread: float
    contract_symbol: Optional[str] = None


@dataclass(frozen=True)
class ModelQuote:
    price: float
    delta: float
    sigma_or_marker: float


def year_fraction(start: pd.Timestamp, end: pd.Timestamp) -> float:
    return max((pd.Timestamp(end) - pd.Timestamp(start)).days / 365.0, 0.0)


def load_underlying_history_csv(path: str | Path) -> pd.DataFrame:
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
    out = out.dropna(subset=["date", "S"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if out.empty:
        raise RuntimeError(f"History CSV contains no valid rows: {path}")
    return out


def build_history_from_panel(panel: pd.DataFrame) -> pd.DataFrame:
    hist = (
        panel.groupby("date", as_index=False)["underlying_price"]
        .median()
        .rename(columns={"underlying_price": "S"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    if hist.empty:
        raise RuntimeError("Option panel does not contain usable underlying prices")
    return hist


def get_spot_on_or_before(history: pd.DataFrame, date: pd.Timestamp) -> float:
    sub = history.loc[history["date"] <= pd.Timestamp(date), "S"]
    if sub.empty:
        raise RuntimeError(f"No underlying history available on or before {pd.Timestamp(date).date()}")
    return float(sub.iloc[-1])


def standardize_option_panel(df: pd.DataFrame) -> pd.DataFrame:
    out = add_derived_columns(df.copy(), cfg=TransformConfig())
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
    out["maturity_date"] = pd.to_datetime(out["maturity_date"], errors="coerce").dt.tz_localize(None)
    out["relative_spread"] = (out["ask"] - out["bid"]) / out["mid"]
    out = out.dropna(subset=["date", "maturity_date", "strike", "mid", "underlying_price", "time_to_expiry"])
    out = out[np.isfinite(out["mid"]) & (out["mid"] > 0.0)].copy()
    out["option_type"] = out["option_type"].astype(str).str.lower()
    return out.sort_values(["date", "maturity_date", "strike"]).reset_index(drop=True)


def _quote_liquidity_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["quote_relative_spread"] = pd.to_numeric(out.get("relative_spread"), errors="coerce").fillna(np.inf)
    out["quote_open_interest"] = pd.to_numeric(out.get("open_interest"), errors="coerce").fillna(0.0)
    out["quote_volume"] = pd.to_numeric(out.get("volume"), errors="coerce").fillna(0.0)
    return out


def _choose_best_quote(quotes: pd.DataFrame) -> pd.Series:
    keyed = _quote_liquidity_key(quotes)
    return (
        keyed.sort_values(
            ["quote_relative_spread", "quote_open_interest", "quote_volume"],
            ascending=[True, False, False],
        )
        .iloc[0]
    )


def _target_strike_for_snapshot(
    snapshot: pd.DataFrame,
    *,
    strike_mode: StrikeMode,
    K_fixed: Optional[float],
    K_mult: Optional[float],
) -> float:
    S0 = float(pd.to_numeric(snapshot["underlying_price"], errors="coerce").median())
    if strike_mode == "atm":
        return S0
    if strike_mode == "fixed_moneyness":
        if K_mult is None:
            raise ValueError("K_mult required for fixed_moneyness")
        return float(K_mult) * S0
    if K_fixed is None:
        raise ValueError("K_fixed required for fixed_strike")
    return float(K_fixed)


def select_listed_contracts(
    panel: pd.DataFrame,
    *,
    option_type: OptionType,
    target_T: float,
    strike_mode: StrikeMode = "atm",
    K_fixed: Optional[float] = None,
    K_mult: Optional[float] = None,
    entry_step_days: int = 5,
    max_contracts: Optional[int] = None,
    contract_moneyness_band: float = 0.02,
    contract_max_relative_spread: float = 0.05,
    contract_min_volume: float = 25.0,
    contract_min_open_interest: float = 200.0,
) -> list[ListedContract]:
    panel = standardize_option_panel(panel)
    dates = list(pd.Index(panel["date"].dropna().sort_values().unique()))
    contracts: list[ListedContract] = []

    for idx, entry_date in enumerate(dates[:: max(int(entry_step_days), 1)]):
        snapshot = panel[(panel["date"] == entry_date) & (panel["option_type"] == option_type)].copy()
        if snapshot.empty:
            continue

        expiry_choice = (
            snapshot.groupby("maturity_date", as_index=False)["time_to_expiry"]
            .median()
            .rename(columns={"time_to_expiry": "T"})
        )
        expiry_choice = expiry_choice[expiry_choice["T"] > 0.0].copy()
        if expiry_choice.empty:
            continue
        expiry_choice["target_gap"] = np.abs(expiry_choice["T"] - float(target_T))
        expiry = pd.Timestamp(expiry_choice.sort_values(["target_gap", "T"]).iloc[0]["maturity_date"])

        expiry_slice = snapshot[snapshot["maturity_date"] == expiry].copy()
        if expiry_slice.empty:
            continue

        expiry_slice = _quote_liquidity_key(expiry_slice)
        expiry_slice["moneyness"] = pd.to_numeric(expiry_slice["strike"], errors="coerce") / pd.to_numeric(
            expiry_slice["underlying_price"], errors="coerce"
        )
        expiry_slice = expiry_slice[
            (expiry_slice["quote_relative_spread"] <= float(contract_max_relative_spread))
            & (expiry_slice["quote_open_interest"] >= float(contract_min_open_interest))
            & (expiry_slice["quote_volume"] >= float(contract_min_volume))
        ].copy()
        if expiry_slice.empty:
            continue

        target_strike = _target_strike_for_snapshot(
            expiry_slice,
            strike_mode=strike_mode,
            K_fixed=K_fixed,
            K_mult=K_mult,
        )
        if strike_mode == "atm":
            S0 = float(pd.to_numeric(expiry_slice["underlying_price"], errors="coerce").median())
            lo = 1.0 - float(contract_moneyness_band)
            hi = 1.0 + float(contract_moneyness_band)
            expiry_slice = expiry_slice[(expiry_slice["moneyness"] >= lo) & (expiry_slice["moneyness"] <= hi)].copy()
            if expiry_slice.empty:
                continue
        expiry_slice["strike_distance"] = np.abs(pd.to_numeric(expiry_slice["strike"], errors="coerce") - target_strike)
        best = _choose_best_quote(expiry_slice.sort_values("strike_distance"))

        contracts.append(
            ListedContract(
                trade_id=len(contracts),
                entry_date=pd.Timestamp(entry_date),
                maturity_date=pd.Timestamp(expiry),
                strike=float(best["strike"]),
                option_type=str(option_type),
                entry_spot=float(best["underlying_price"]),
                market_premium=float(best["mid"]),
                target_time_to_expiry=float(target_T),
                observed_time_to_expiry=float(best["time_to_expiry"]),
                quote_volume=float(pd.to_numeric(best.get("volume"), errors="coerce") if pd.notna(best.get("volume")) else 0.0),
                quote_open_interest=float(pd.to_numeric(best.get("open_interest"), errors="coerce") if pd.notna(best.get("open_interest")) else 0.0),
                quote_relative_spread=float(pd.to_numeric(best.get("relative_spread"), errors="coerce") if pd.notna(best.get("relative_spread")) else np.nan),
                contract_symbol=str(best["contract_symbol"]) if "contract_symbol" in best.index and pd.notna(best["contract_symbol"]) else None,
            )
        )
        if max_contracts is not None and len(contracts) >= int(max_contracts):
            break

    return contracts


def collect_contract_marks(panel: pd.DataFrame, contract: ListedContract) -> pd.DataFrame:
    panel = standardize_option_panel(panel)
    marks = panel[
        (panel["option_type"] == contract.option_type)
        & (panel["maturity_date"] == contract.maturity_date)
        & np.isclose(panel["strike"].to_numpy(dtype=float), float(contract.strike), atol=1e-10)
        & (panel["date"] >= contract.entry_date)
        & (panel["date"] <= contract.maturity_date)
    ].copy()
    return marks.sort_values("date").reset_index(drop=True)


def build_daily_calibration_cache(
    panel: pd.DataFrame,
    *,
    ticker: str,
    snapshot_cfg: SnapshotCalibrationConfig,
    required_dates: list[pd.Timestamp],
    cache_dir: str | Path | None = None,
) -> dict[pd.Timestamp, SnapshotCalibrationArtifacts]:
    panel = standardize_option_panel(panel)
    cache: dict[pd.Timestamp, SnapshotCalibrationArtifacts] = {}
    daily_groups = {pd.Timestamp(d): g.copy() for d, g in panel.groupby("date")}

    for date in sorted({pd.Timestamp(d) for d in required_dates}):
        if date not in daily_groups:
            raise RuntimeError(f"Missing option snapshot for calibration date {date.date()}")
        artifacts = calibrate_option_snapshot(
            daily_groups[date],
            ticker=ticker,
            valuation_date=str(date.date()),
            cfg=snapshot_cfg,
        )
        cache[date] = artifacts
        if cache_dir is not None:
            save_calibration_artifacts(artifacts, Path(cache_dir) / date.strftime("%Y-%m-%d"))
    return cache


def _bs_quote(
    artifacts: SnapshotCalibrationArtifacts,
    *,
    S: float,
    K: float,
    tau: float,
    option_type: OptionType,
) -> ModelQuote:
    if tau <= 0.0:
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        delta = 1.0 if (option_type == "call" and S > K) else 0.0
        if option_type == "put":
            delta = -1.0 if S < K else 0.0
        return ModelQuote(price=float(intrinsic), delta=float(delta), sigma_or_marker=0.0)

    sigma = float(np.asarray(artifacts.arbitrage_free_iv_surface.iv(np.array([K]), np.array([tau]))[0], dtype=float))
    price = float(
        bs_price(
            S=np.array([S]),
            K=np.array([K]),
            T=np.array([tau]),
            r=float(artifacts.summary["r"]),
            q=float(artifacts.summary["q"]),
            sigma=np.array([sigma]),
            option_type=option_type,
        )[0]
    )
    delta = float(
        bs_delta(
            np.array([S]),
            np.array([K]),
            np.array([tau]),
            float(artifacts.summary["r"]),
            float(artifacts.summary["q"]),
            np.array([sigma]),
            option_type=option_type,
        )[0]
    )
    return ModelQuote(price=price, delta=delta, sigma_or_marker=sigma)


def _lv_quote(
    artifacts: SnapshotCalibrationArtifacts,
    *,
    S: float,
    K: float,
    tau: float,
    option_type: OptionType,
    pde_cfg: PDEConfig,
) -> ModelQuote:
    if tau <= 0.0:
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        delta = 1.0 if (option_type == "call" and S > K) else 0.0
        if option_type == "put":
            delta = -1.0 if S < K else 0.0
        return ModelQuote(price=float(intrinsic), delta=float(delta), sigma_or_marker=np.nan)

    lv_surface = artifacts.local_vol_surface
    lv_low = float(lv_surface.S_grid[0])
    lv_high = float(lv_surface.S_grid[-1])
    if S < lv_low or S > lv_high or K < lv_low or K > lv_high:
        raise RuntimeError(
            "Requested contract lies outside the calibrated daily local-vol domain. "
            f"Need S={S:.6f}, K={K:.6f} in [{lv_low:.6f}, {lv_high:.6f}]"
        )

    lo = max(lv_low, min(S, K) * 0.60)
    hi = min(lv_high, max(S, K) * 1.40)
    if not hi > lo:
        lo = lv_low
        hi = lv_high

    t_grid, S_grid, V_grid = solve_european_pde_local_vol_surface(
        S_ref=float(S),
        K=float(K),
        T=float(tau),
        r=float(artifacts.summary["r"]),
        q=float(artifacts.summary["q"]),
        option_type=option_type,
        lv_surface=lv_surface,
        cfg=pde_cfg,
        t0=0.0,
        S_min=float(lo),
        S_max=float(hi),
    )
    D_grid = np.gradient(V_grid, S_grid, axis=1, edge_order=2)
    price = float(np.interp(S, S_grid, V_grid[0, :]))
    delta = float(np.interp(S, S_grid, D_grid[0, :]))
    return ModelQuote(price=price, delta=delta, sigma_or_marker=np.nan)


def _model_quote(
    model: ModelKind,
    artifacts: SnapshotCalibrationArtifacts,
    *,
    S: float,
    K: float,
    tau: float,
    option_type: OptionType,
    pde_cfg: PDEConfig,
) -> ModelQuote:
    if model == "BS":
        return _bs_quote(artifacts, S=S, K=K, tau=tau, option_type=option_type)
    if model == "LocalVol":
        return _lv_quote(artifacts, S=S, K=K, tau=tau, option_type=option_type, pde_cfg=pde_cfg)
    raise ValueError(f"Unknown model: {model}")


def _option_payoff(ST: float, K: float, option_type: OptionType) -> float:
    if option_type == "call":
        return float(max(ST - K, 0.0))
    return float(max(K - ST, 0.0))


def run_daily_market_panel_backtest(
    *,
    contracts: list[ListedContract],
    panel: pd.DataFrame,
    history: pd.DataFrame,
    calibration_cache: dict[pd.Timestamp, SnapshotCalibrationArtifacts],
    model: ModelKind,
    tx_costs: TransactionCostModel,
    pde_cfg: PDEConfig = PDEConfig(n_space=120, n_time=100),
) -> pd.DataFrame:
    panel = standardize_option_panel(panel)
    history = history.sort_values("date").reset_index(drop=True)
    panel_dates = list(pd.Index(panel["date"].dropna().sort_values().unique()))

    rows: list[dict[str, Any]] = []
    for contract in contracts:
        schedule = [d for d in panel_dates if contract.entry_date <= d < contract.maturity_date]
        if not schedule or schedule[0] != contract.entry_date:
            raise RuntimeError(
                f"Contract {contract.trade_id} has no calibration schedule beginning on entry date {contract.entry_date.date()}"
            )

        marks = collect_contract_marks(panel, contract)
        mark_by_date = {pd.Timestamp(d): g for d, g in marks.groupby("date")}

        S0 = get_spot_on_or_before(history, contract.entry_date)
        entry_tau = year_fraction(contract.entry_date, contract.maturity_date)
        entry_artifacts = calibration_cache[pd.Timestamp(contract.entry_date)]
        entry_quote = _model_quote(
            model,
            entry_artifacts,
            S=S0,
            K=float(contract.strike),
            tau=entry_tau,
            option_type=contract.option_type,
            pde_cfg=pde_cfg,
        )

        shares = float(entry_quote.delta)
        tc0 = float(tx_costs.cost(d_shares=shares, S=S0))
        total_tc = tc0
        cash_model = float(entry_quote.price) - shares * S0 - tc0
        cash_market = float(contract.market_premium) - shares * S0 - tc0
        mtm_abs_errors: list[float] = []

        for prev_date, cur_date in zip(schedule[:-1], schedule[1:]):
            dt = year_fraction(prev_date, cur_date)
            prev_r = float(calibration_cache[pd.Timestamp(prev_date)].summary["r"])
            cash_model *= float(np.exp(prev_r * dt))
            cash_market *= float(np.exp(prev_r * dt))

            S_cur = get_spot_on_or_before(history, cur_date)
            tau_cur = year_fraction(cur_date, contract.maturity_date)
            cur_quote = _model_quote(
                model,
                calibration_cache[pd.Timestamp(cur_date)],
                S=S_cur,
                K=float(contract.strike),
                tau=tau_cur,
                option_type=contract.option_type,
                pde_cfg=pde_cfg,
            )
            d_shares = float(cur_quote.delta - shares)
            tc = float(tx_costs.cost(d_shares=d_shares, S=S_cur))
            cash_model -= d_shares * S_cur + tc
            cash_market -= d_shares * S_cur + tc
            shares = float(cur_quote.delta)
            total_tc += tc

            mark_rows = mark_by_date.get(pd.Timestamp(cur_date))
            if mark_rows is not None and not mark_rows.empty:
                best_mark = _choose_best_quote(mark_rows)
                mtm_abs_errors.append(abs(float(cur_quote.price) - float(best_mark["mid"])))

        expiry_spot = get_spot_on_or_before(history, contract.maturity_date)
        last_date = schedule[-1]
        dt_last = year_fraction(last_date, contract.maturity_date)
        last_r = float(calibration_cache[pd.Timestamp(last_date)].summary["r"])
        cash_model *= float(np.exp(last_r * dt_last))
        cash_market *= float(np.exp(last_r * dt_last))

        terminal_portfolio_model = shares * expiry_spot + cash_model
        terminal_portfolio_market = shares * expiry_spot + cash_market
        payoff = _option_payoff(expiry_spot, float(contract.strike), contract.option_type)

        replication_error_net = float(terminal_portfolio_model - payoff)
        replication_error_gross = float(replication_error_net + total_tc)
        market_pnl_net = float(terminal_portfolio_market - payoff)
        market_pnl_gross = float(market_pnl_net + total_tc)
        entry_pricing_error = float(entry_quote.price - contract.market_premium)

        rows.append(
            {
                "trade_id": int(contract.trade_id),
                "model": model,
                "entry_date": contract.entry_date.date().isoformat(),
                "maturity_date": contract.maturity_date.date().isoformat(),
                "entry_spot": float(S0),
                "terminal_spot": float(expiry_spot),
                "strike": float(contract.strike),
                "option_type": contract.option_type,
                "target_time_to_expiry": float(contract.target_time_to_expiry),
                "observed_time_to_expiry": float(contract.observed_time_to_expiry),
                "market_premium": float(contract.market_premium),
                "model_entry_price": float(entry_quote.price),
                "entry_pricing_error": entry_pricing_error,
                "entry_model_delta": float(entry_quote.delta),
                "terminal_payoff": float(payoff),
                "terminal_portfolio_model": float(terminal_portfolio_model),
                "terminal_portfolio_market": float(terminal_portfolio_market),
                "replication_error_net": replication_error_net,
                "replication_error_gross": replication_error_gross,
                "market_pnl_net": market_pnl_net,
                "market_pnl_gross": market_pnl_gross,
                "total_tx_cost": float(total_tc),
                "n_rebalances": int(max(len(schedule) - 1, 0)),
                "n_mark_obs": int(len(mtm_abs_errors)),
                "mean_abs_mtm_error": float(np.mean(mtm_abs_errors)) if mtm_abs_errors else float("nan"),
                "max_abs_mtm_error": float(np.max(mtm_abs_errors)) if mtm_abs_errors else float("nan"),
                "quote_volume": float(contract.quote_volume),
                "quote_open_interest": float(contract.quote_open_interest),
                "quote_relative_spread": float(contract.quote_relative_spread),
                "contract_symbol": contract.contract_symbol,
            }
        )

    return pd.DataFrame(rows)
