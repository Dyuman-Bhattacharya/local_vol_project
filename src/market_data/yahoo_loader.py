from __future__ import annotations

import pandas as pd
import yfinance as yf

from market_data.carry import resolve_carry_estimate
from market_data.loaders import load_yfinance_option_chain_snapshot


class YahooDataError(RuntimeError):
    pass


def fetch_underlying_history(
    ticker: str,
    *,
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch historical underlying prices from Yahoo Finance via yfinance.

    Returns DataFrame with columns:
      ['date', 'S']
    """

    tk = yf.Ticker(ticker)
    hist = tk.history(start=start, end=end, interval=interval)

    if hist.empty:
        raise YahooDataError(f"No price history returned for {ticker}")

    date_col = "Date" if "Date" in hist.reset_index().columns else hist.reset_index().columns[0]
    df = hist.reset_index()[[date_col, "Close"]]
    df.columns = ["date", "S"]
    df["date"] = pd.to_datetime(df["date"])
    df["S"] = df["S"].astype(float)
    return df


def fetch_option_chain_snapshot(
    ticker: str,
    *,
    valuation_date: str | None = None,
    option_type: str = "call",
    min_days: int = 7,
    max_days: int = 365,
    max_expiries: int | None = None,
    risk_free_rate: float | None = None,
    dividend_yield: float | None = None,
) -> pd.DataFrame:
    """
    Canonical Yahoo Finance option-chain fetch wrapper.

    This delegates to market_data.loaders.load_yfinance_option_chain_snapshot so the
    notebooks and scripts share one implementation and one set of assumptions.
    """

    try:
        df = load_yfinance_option_chain_snapshot(
            ticker=ticker,
            as_of_date=valuation_date,
            option_type=option_type,
            min_days=min_days,
            max_days=max_days,
            max_expiries=max_expiries,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
        )
        if df.empty:
            return df
        spot = float(pd.to_numeric(df["underlying_price"], errors="coerce").dropna().median())
        as_of = pd.to_datetime(valuation_date or pd.Timestamp.utcnow().date()).tz_localize(None).normalize()
        carry = resolve_carry_estimate(
            ticker=ticker,
            spot=spot,
            as_of_date=as_of,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
        )
        df["risk_free_rate"] = float(carry.risk_free_rate)
        df["dividend_yield"] = float(carry.dividend_yield)
        df["risk_free_rate_source"] = str(carry.risk_free_rate_source)
        df["dividend_yield_source"] = str(carry.dividend_yield_source)
        return df
    except Exception as e:
        raise YahooDataError(str(e)) from e
