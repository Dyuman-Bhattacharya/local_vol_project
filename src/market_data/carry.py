from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None


@dataclass(frozen=True)
class CarryEstimate:
    risk_free_rate: float
    dividend_yield: float
    risk_free_rate_source: str
    dividend_yield_source: str


def apply_carry_to_snapshot_frame(
    df: pd.DataFrame,
    *,
    ticker: str,
    valuation_date: str | pd.Timestamp,
    risk_free_rate: Optional[float] = None,
    dividend_yield: Optional[float] = None,
) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    spot_series = pd.to_numeric(out.get("underlying_price"), errors="coerce").dropna()
    if spot_series.empty:
        raise ValueError("Snapshot frame does not contain a usable underlying_price column for carry estimation")

    carry = resolve_carry_estimate(
        ticker=ticker,
        spot=float(spot_series.median()),
        as_of_date=valuation_date,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
    )
    out["risk_free_rate"] = float(carry.risk_free_rate)
    out["dividend_yield"] = float(carry.dividend_yield)
    out["risk_free_rate_source"] = str(carry.risk_free_rate_source)
    out["dividend_yield_source"] = str(carry.dividend_yield_source)
    return out


def normalize_rate_quote(value: float | int | None) -> float:
    if value is None:
        return np.nan
    out = float(value)
    if not np.isfinite(out):
        return np.nan
    if abs(out) > 1.0:
        out /= 100.0
    return float(out)


def _history_close_on_or_before(ticker: str, *, as_of_date: pd.Timestamp, lookback_days: int = 10) -> float:
    if yf is None:  # pragma: no cover
        return np.nan
    tk = yf.Ticker(ticker)
    start = (pd.Timestamp(as_of_date) - pd.Timedelta(days=int(lookback_days))).strftime("%Y-%m-%d")
    end = (pd.Timestamp(as_of_date) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    hist = tk.history(start=start, end=end, auto_adjust=False)
    if hist is None or hist.empty or "Close" not in hist.columns:
        return np.nan
    close = pd.to_numeric(hist["Close"], errors="coerce").dropna()
    if close.empty:
        return np.nan
    return float(close.iloc[-1])


def estimate_risk_free_rate(
    *,
    as_of_date: str | pd.Timestamp,
    manual_value: Optional[float] = None,
) -> tuple[float, str]:
    if manual_value is not None:
        return float(manual_value), "manual"

    as_of_ts = pd.Timestamp(as_of_date).tz_localize(None).normalize()
    proxies = [
        ("^IRX", "yfinance_^IRX_13w"),
        ("^FVX", "yfinance_^FVX_5y"),
        ("^TNX", "yfinance_^TNX_10y"),
    ]
    for ticker, source in proxies:
        try:
            close = _history_close_on_or_before(ticker, as_of_date=as_of_ts)
        except Exception:
            close = np.nan
        rate = normalize_rate_quote(close)
        if np.isfinite(rate):
            return float(rate), source
    return 0.0, "fallback_zero"


def estimate_dividend_yield(
    *,
    ticker: str,
    spot: float,
    as_of_date: str | pd.Timestamp,
    manual_value: Optional[float] = None,
) -> tuple[float, str]:
    if manual_value is not None:
        return float(manual_value), "manual"

    if yf is None:  # pragma: no cover
        return 0.0, "fallback_zero"

    as_of_ts = pd.Timestamp(as_of_date).tz_localize(None).normalize()
    tk = yf.Ticker(ticker)

    try:
        divs = tk.dividends
        if divs is not None and not divs.empty:
            idx = pd.to_datetime(divs.index, errors="coerce")
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_localize(None)
            values = pd.Series(pd.to_numeric(divs.to_numpy(), errors="coerce"), index=idx).dropna()
            recent = values[(values.index <= as_of_ts) & (values.index > as_of_ts - pd.DateOffset(years=1))]
            if not recent.empty and float(spot) > 0.0:
                q = float(recent.sum()) / float(spot)
                if np.isfinite(q) and q >= 0.0:
                    return q, "trailing_12m_dividends"
    except Exception:
        pass

    try:
        info = tk.info or {}
    except Exception:
        info = {}
    q = normalize_rate_quote(info.get("dividendYield"))
    if np.isfinite(q) and q >= 0.0:
        return float(q), "ticker_info_dividendYield"
    return 0.0, "fallback_zero"


def resolve_carry_estimate(
    *,
    ticker: str,
    spot: float,
    as_of_date: str | pd.Timestamp,
    risk_free_rate: Optional[float] = None,
    dividend_yield: Optional[float] = None,
) -> CarryEstimate:
    r, r_source = estimate_risk_free_rate(as_of_date=as_of_date, manual_value=risk_free_rate)
    q, q_source = estimate_dividend_yield(
        ticker=ticker,
        spot=float(spot),
        as_of_date=as_of_date,
        manual_value=dividend_yield,
    )
    return CarryEstimate(
        risk_free_rate=float(r),
        dividend_yield=float(q),
        risk_free_rate_source=str(r_source),
        dividend_yield_source=str(q_source),
    )
