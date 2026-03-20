# src/market_data/loaders.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

from market_data.carry import resolve_carry_estimate


class MarketDataLoadError(RuntimeError):
    """Raised when market data cannot be loaded or parsed into the expected schema."""


@dataclass(frozen=True)
class LoadConfig:
    """
    Loader configuration.

    Notes
    -----
    - Use `date_col` for the quote/observation date.
    - Use `maturity_col` for the option expiration date.
    - If your source already supplies `T` (time_to_expiry_years), set `t_col`.
    """

    date_col: str = "date"
    maturity_col: str = "maturity_date"
    t_col: str = "time_to_expiry"

    underlying_col: str = "underlying_price"
    strike_col: str = "strike"
    option_type_col: str = "option_type"

    bid_col: str = "bid"
    ask_col: str = "ask"
    mid_col: str = "mid"

    r_col: str = "risk_free_rate"
    q_col: str = "dividend_yield"

    volume_col: str = "volume"
    open_interest_col: str = "open_interest"
    symbol_col: str = "underlying"


def load_csv(path: Union[str, Path], *, cfg: LoadConfig = LoadConfig(), **read_csv_kwargs) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise MarketDataLoadError(f"CSV file not found: {path}")
    return standardize_columns(pd.read_csv(path, **read_csv_kwargs), cfg=cfg)


def load_parquet(path: Union[str, Path], *, cfg: LoadConfig = LoadConfig(), **read_parquet_kwargs) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise MarketDataLoadError(f"Parquet file not found: {path}")
    return standardize_columns(pd.read_parquet(path, **read_parquet_kwargs), cfg=cfg)


def load_any(path: Union[str, Path], *, cfg: LoadConfig = LoadConfig(), **kwargs) -> pd.DataFrame:
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".csv":
        return load_csv(path, cfg=cfg, **kwargs)
    if suf in (".parquet", ".pq"):
        return load_parquet(path, cfg=cfg, **kwargs)
    raise MarketDataLoadError(f"Unsupported file extension: {suf}")


def standardize_columns(df: pd.DataFrame, *, cfg: LoadConfig = LoadConfig()) -> pd.DataFrame:
    """
    Enforce a standardized schema used throughout the project.
    """

    if not isinstance(df, pd.DataFrame):
        raise MarketDataLoadError("Input must be a pandas.DataFrame")

    out = df.copy()

    def _ensure(col: str) -> None:
        if col not in out.columns:
            out[col] = np.nan

    for col in [
        cfg.date_col,
        cfg.maturity_col,
        cfg.t_col,
        cfg.underlying_col,
        cfg.strike_col,
        cfg.option_type_col,
        cfg.bid_col,
        cfg.ask_col,
        cfg.mid_col,
        cfg.r_col,
        cfg.q_col,
        cfg.volume_col,
        cfg.open_interest_col,
        cfg.symbol_col,
    ]:
        _ensure(col)

    rename_map = {
        cfg.date_col: "date",
        cfg.maturity_col: "maturity_date",
        cfg.t_col: "time_to_expiry",
        cfg.underlying_col: "underlying_price",
        cfg.strike_col: "strike",
        cfg.option_type_col: "option_type",
        cfg.bid_col: "bid",
        cfg.ask_col: "ask",
        cfg.mid_col: "mid",
        cfg.r_col: "risk_free_rate",
        cfg.q_col: "dividend_yield",
        cfg.volume_col: "volume",
        cfg.open_interest_col: "open_interest",
        cfg.symbol_col: "underlying",
    }
    out = out.rename(columns=rename_map)

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["maturity_date"] = pd.to_datetime(out["maturity_date"], errors="coerce")

    out["option_type"] = out["option_type"].astype(str).str.lower().str.strip()
    out.loc[out["option_type"].isin(["c", "call", "calls"]), "option_type"] = "call"
    out.loc[out["option_type"].isin(["p", "put", "puts"]), "option_type"] = "put"

    for c in [
        "time_to_expiry",
        "underlying_price",
        "strike",
        "bid",
        "ask",
        "mid",
        "risk_free_rate",
        "dividend_yield",
        "volume",
        "open_interest",
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    canonical_cols = [
        "date",
        "underlying",
        "underlying_price",
        "maturity_date",
        "time_to_expiry",
        "strike",
        "option_type",
        "bid",
        "ask",
        "mid",
        "volume",
        "open_interest",
        "risk_free_rate",
        "dividend_yield",
    ]
    extra = [c for c in out.columns if c not in canonical_cols]
    return out[canonical_cols + extra]


def load_yfinance_option_chain_snapshot(
    ticker: str,
    *,
    as_of_date: str | None = None,
    option_type: str = "call",
    min_days: int = 7,
    max_days: int = 365,
    max_expiries: int | None = None,
    risk_free_rate: float | None = None,
    dividend_yield: float | None = None,
) -> pd.DataFrame:
    """
    Pull the latest option-chain snapshot via yfinance and return a standardized DataFrame.
    """

    if yf is None:
        raise MarketDataLoadError("yfinance is not installed. Add it to poetry dependencies.")

    tkr = yf.Ticker(ticker)
    try:
        expiries = list(tkr.options)
    except Exception as e:
        raise MarketDataLoadError(f"yfinance could not fetch option expiries for {ticker}: {e}")

    if not expiries:
        raise MarketDataLoadError(f"No option expiries found for {ticker}. Try a different ticker (e.g., SPY).")

    try:
        spot = float(getattr(tkr, "fast_info", {}).get("last_price", np.nan))
    except Exception:
        spot = np.nan
    if not np.isfinite(spot):
        hist = tkr.history(period="5d")
        if hist is None or hist.empty:
            raise MarketDataLoadError("Could not fetch underlying price history to determine spot.")
        spot = float(hist["Close"].iloc[-1])

    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    row_date = pd.to_datetime(as_of_date).tz_localize(None).normalize() if as_of_date is not None else today
    carry = resolve_carry_estimate(
        ticker=ticker,
        spot=float(spot),
        as_of_date=row_date,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
    )

    expiry_dates = pd.to_datetime(expiries, errors="coerce")
    expiry_dates = expiry_dates[~expiry_dates.isna()].sort_values()
    days_to = (expiry_dates - today).days
    expiry_dates = expiry_dates[(days_to >= min_days) & (days_to <= max_days)]
    if max_expiries is not None:
        expiry_dates = expiry_dates[:max_expiries]
    if len(expiry_dates) == 0:
        raise MarketDataLoadError(f"No expiries in [{min_days},{max_days}] days for {ticker}.")

    rows = []
    for exp in expiry_dates.strftime("%Y-%m-%d").tolist():
        try:
            chain = tkr.option_chain(exp)
        except Exception:
            continue

        opt_df = chain.calls if option_type == "call" else chain.puts
        if opt_df is None or opt_df.empty:
            continue

        opt_df = opt_df.copy()
        opt_df["maturity_date"] = pd.to_datetime(exp)
        opt_df["date"] = row_date
        opt_df["underlying"] = ticker
        opt_df["underlying_price"] = float(spot)
        opt_df["option_type"] = option_type
        opt_df["risk_free_rate"] = float(carry.risk_free_rate)
        opt_df["dividend_yield"] = float(carry.dividend_yield)
        opt_df["risk_free_rate_source"] = str(carry.risk_free_rate_source)
        opt_df["dividend_yield_source"] = str(carry.dividend_yield_source)

        if "mid" not in opt_df.columns:
            opt_df["mid"] = np.nan

        bid = pd.to_numeric(opt_df.get("bid", np.nan), errors="coerce")
        ask = pd.to_numeric(opt_df.get("ask", np.nan), errors="coerce")
        last = pd.to_numeric(opt_df.get("lastPrice", np.nan), errors="coerce")
        have_ba = bid.notna() & ask.notna() & (bid >= 0) & (ask >= 0)
        opt_df.loc[have_ba, "mid"] = 0.5 * (bid[have_ba] + ask[have_ba])
        still = opt_df["mid"].isna()
        opt_df.loc[still & last.notna() & (last >= 0), "mid"] = last[still & last.notna() & (last >= 0)]

        opt_df = opt_df.rename(
            columns={
                "contractSymbol": "contract_symbol",
                "lastTradeDate": "last_trade_date",
                "impliedVolatility": "vendor_implied_volatility",
                "strike": "strike",
                "volume": "volume",
                "openInterest": "open_interest",
                "bid": "bid",
                "ask": "ask",
            }
        )

        T_years = (pd.to_datetime(exp) - row_date).days / 365.0
        opt_df["time_to_expiry"] = float(max(T_years, 0.0))

        keep = [
            "date",
            "underlying",
            "underlying_price",
            "maturity_date",
            "time_to_expiry",
            "strike",
            "option_type",
            "bid",
            "ask",
            "mid",
            "volume",
            "open_interest",
            "risk_free_rate",
            "dividend_yield",
            "contract_symbol",
            "last_trade_date",
            "vendor_implied_volatility",
            "risk_free_rate_source",
            "dividend_yield_source",
        ]
        for c in keep:
            if c not in opt_df.columns:
                opt_df[c] = np.nan
        rows.append(opt_df[keep])

    if not rows:
        raise MarketDataLoadError(f"Could not load any option chain rows for {ticker} (yfinance returned empty).")

    return standardize_columns(pd.concat(rows, ignore_index=True), cfg=LoadConfig())
