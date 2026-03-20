from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from market_data.carry import resolve_carry_estimate


class ThetaDataError(RuntimeError):
    pass


THETA_BASE_URL = "http://127.0.0.1:25503/v3"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ThetaSnapshotConfig:
    ticker: str
    valuation_date: str
    snapshot_time: str = "15:45"
    timezone: str = "America/New_York"
    min_days: int = 1
    max_days: int = 730
    max_expiries: int | None = None
    risk_free_rate: float | None = None
    dividend_yield: float | None = None


def _theta_get(path: str, *, params: dict[str, Any], timeout: int = 180) -> dict[str, Any]:
    url = f"{THETA_BASE_URL}{path}"
    q = {k: v for k, v in params.items() if v is not None}
    resp = requests.get(url, params=q, timeout=timeout)
    if resp.status_code != 200:
        raise ThetaDataError(f"Theta request failed {resp.status_code} for {path}: {resp.text[:400]}")
    payload = resp.json()
    if not isinstance(payload, dict) or "response" not in payload:
        raise ThetaDataError(f"Unexpected Theta response for {path}")
    return payload


def theta_terminal_ready(*, symbol: str = "SPY", timeout: int = 10) -> bool:
    try:
        _theta_get(
            "/option/list/expirations",
            params={"symbol": symbol.upper(), "format": "json"},
            timeout=timeout,
        )
        return True
    except Exception:
        return False


def ensure_theta_terminal_running(*, symbol: str = "SPY", timeout_sec: int = 45) -> None:
    if theta_terminal_ready(symbol=symbol, timeout=5):
        return
    launcher = PROJECT_ROOT / "scripts" / "start_theta_terminal.cmd"
    if not launcher.exists():
        raise ThetaDataError(f"Theta launcher not found: {launcher}")
    subprocess.Popen(
        ["cmd", "/c", str(launcher)],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + float(timeout_sec)
    while time.time() < deadline:
        if theta_terminal_ready(symbol=symbol, timeout=5):
            return
        time.sleep(2.0)
    raise ThetaDataError("Theta Terminal did not become ready in time")


def _build_occ_symbol(symbol: str, expiration: str, right: str, strike: float) -> str:
    exp = pd.Timestamp(expiration)
    yymmdd = exp.strftime("%y%m%d")
    cp = "C" if str(right).upper().startswith("C") else "P"
    strike_mils = int(round(float(strike) * 1000.0))
    return f"{symbol.upper()}{yymmdd}{cp}{strike_mils:08d}"


def _records_from_theta_payload(payload: dict[str, Any], *, field_prefix: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in payload.get("response", []):
        contract = item.get("contract", {}) or {}
        data = item.get("data", []) or []
        obs = data[0] if data else {}
        row = {
            "underlying": str(contract.get("symbol", "")).upper(),
            "maturity_date": pd.to_datetime(contract.get("expiration"), errors="coerce"),
            "strike": pd.to_numeric(contract.get("strike"), errors="coerce"),
            "option_type": "call" if str(contract.get("right", "")).upper().startswith("C") else "put",
            "contract_symbol": _build_occ_symbol(
                str(contract.get("symbol", "")).upper(),
                str(contract.get("expiration")),
                str(contract.get("right", "")),
                float(contract.get("strike")),
            )
            if contract.get("symbol") and contract.get("expiration") and contract.get("right") and contract.get("strike") is not None
            else None,
        }
        for k, v in obs.items():
            row[f"{field_prefix}{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def _coalesce_numeric(df: pd.DataFrame, left: str, right: str, out: str) -> None:
    if left in df.columns and right in df.columns:
        df[out] = pd.to_numeric(df[left], errors="coerce").combine_first(pd.to_numeric(df[right], errors="coerce"))
    elif left in df.columns:
        df[out] = pd.to_numeric(df[left], errors="coerce")
    elif right in df.columns:
        df[out] = pd.to_numeric(df[right], errors="coerce")
    else:
        df[out] = np.nan


def _fetch_underlying_spot_yfinance(ticker: str, *, valuation_date: str, snapshot_time: str, timezone: str) -> float:
    date = pd.Timestamp(valuation_date).date()
    start = (pd.Timestamp(date) - pd.Timedelta(days=7)).date().isoformat()
    end = (pd.Timestamp(date) + pd.Timedelta(days=1)).date().isoformat()
    hist = yf.Ticker(ticker).history(start=start, end=end, interval="5m", auto_adjust=False)
    if hist is None or hist.empty:
        day = yf.Ticker(ticker).history(start=valuation_date, end=(pd.Timestamp(date) + pd.Timedelta(days=1)).date().isoformat(), interval="1d", auto_adjust=False)
        if day is None or day.empty:
            raise ThetaDataError(f"Could not fetch underlying spot history for {ticker}")
        return float(pd.to_numeric(day["Close"], errors="coerce").dropna().iloc[-1])

    idx = hist.index
    if getattr(idx, "tz", None) is None:
        hist.index = hist.index.tz_localize(timezone)
    else:
        hist.index = hist.index.tz_convert(timezone)

    target_dt = datetime.strptime(f"{valuation_date} {snapshot_time}", "%Y-%m-%d %H:%M").replace(tzinfo=ZoneInfo(timezone))
    day_slice = hist[(hist.index.date == target_dt.date()) & (hist.index <= target_dt)]
    if day_slice.empty:
        day_slice = hist[hist.index.date == target_dt.date()]
    if day_slice.empty:
        raise ThetaDataError(f"No intraday SPY bars available near {valuation_date} {snapshot_time} for {ticker}")
    return float(pd.to_numeric(day_slice["Close"], errors="coerce").dropna().iloc[-1])


def _fetch_quote_frame(cfg: ThetaSnapshotConfig, *, live: bool) -> pd.DataFrame:
    if live:
        payload = _theta_get(
            "/option/snapshot/quote",
            params={"symbol": cfg.ticker.upper(), "expiration": "*", "format": "json"},
        )
        return _records_from_theta_payload(payload, field_prefix="quote_")
    payload = _theta_get(
        "/option/at_time/quote",
        params={
            "symbol": cfg.ticker.upper(),
            "expiration": "*",
            "start_date": pd.Timestamp(cfg.valuation_date).strftime("%Y%m%d"),
            "end_date": pd.Timestamp(cfg.valuation_date).strftime("%Y%m%d"),
            "time_of_day": f"{cfg.snapshot_time}:00.000",
            "format": "json",
        },
    )
    return _records_from_theta_payload(payload, field_prefix="quote_")


def _fetch_open_interest_frame(cfg: ThetaSnapshotConfig, *, live: bool) -> pd.DataFrame:
    if live:
        payload = _theta_get(
            "/option/snapshot/open_interest",
            params={"symbol": cfg.ticker.upper(), "expiration": "*", "format": "json"},
        )
        return _records_from_theta_payload(payload, field_prefix="oi_")
    payload = _theta_get(
        "/option/history/open_interest",
        params={
            "symbol": cfg.ticker.upper(),
            "expiration": "*",
            "start_date": pd.Timestamp(cfg.valuation_date).strftime("%Y%m%d"),
            "end_date": pd.Timestamp(cfg.valuation_date).strftime("%Y%m%d"),
            "format": "json",
        },
    )
    return _records_from_theta_payload(payload, field_prefix="oi_")


def _fetch_volume_frame(cfg: ThetaSnapshotConfig, *, live: bool) -> pd.DataFrame:
    if live:
        payload = _theta_get(
            "/option/snapshot/ohlc",
            params={"symbol": cfg.ticker.upper(), "expiration": "*", "format": "json"},
        )
        return _records_from_theta_payload(payload, field_prefix="ohlc_")
    payload = _theta_get(
        "/option/history/eod",
        params={
            "symbol": cfg.ticker.upper(),
            "expiration": "*",
            "start_date": pd.Timestamp(cfg.valuation_date).strftime("%Y%m%d"),
            "end_date": pd.Timestamp(cfg.valuation_date).strftime("%Y%m%d"),
            "format": "json",
        },
    )
    return _records_from_theta_payload(payload, field_prefix="eod_")


def _merge_theta_frames(quote_df: pd.DataFrame, oi_df: pd.DataFrame, vol_df: pd.DataFrame) -> pd.DataFrame:
    keys = ["underlying", "maturity_date", "strike", "option_type", "contract_symbol"]
    out = quote_df.merge(oi_df, on=keys, how="left").merge(vol_df, on=keys, how="left")
    _coalesce_numeric(out, "quote_bid", None, "bid")
    _coalesce_numeric(out, "quote_ask", None, "ask")
    out["mid"] = 0.5 * (pd.to_numeric(out["bid"], errors="coerce") + pd.to_numeric(out["ask"], errors="coerce"))
    _coalesce_numeric(out, "eod_volume", "ohlc_volume", "volume")
    _coalesce_numeric(out, "oi_open_interest", None, "open_interest")
    eod_last_trade = out["eod_last_trade"] if "eod_last_trade" in out.columns else pd.Series(index=out.index, dtype="object")
    ohlc_timestamp = out["ohlc_timestamp"] if "ohlc_timestamp" in out.columns else pd.Series(index=out.index, dtype="object")
    out["last_trade_date"] = pd.to_datetime(
        eod_last_trade.combine_first(ohlc_timestamp),
        errors="coerce",
        utc=True,
    )
    out["quote_timestamp"] = pd.to_datetime(out.get("quote_timestamp"), errors="coerce", utc=True)
    return out


def fetch_theta_option_chain_snapshot(
    ticker: str,
    *,
    valuation_date: str,
    snapshot_time: str = "15:45",
    timezone: str = "America/New_York",
    min_days: int = 1,
    max_days: int = 730,
    max_expiries: int | None = None,
    risk_free_rate: float | None = None,
    dividend_yield: float | None = None,
) -> pd.DataFrame:
    ensure_theta_terminal_running(symbol=ticker)
    cfg = ThetaSnapshotConfig(
        ticker=ticker,
        valuation_date=valuation_date,
        snapshot_time=snapshot_time,
        timezone=timezone,
        min_days=min_days,
        max_days=max_days,
        max_expiries=max_expiries,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
    )
    today_local = datetime.now(ZoneInfo(timezone)).date().isoformat()
    live = str(valuation_date) >= today_local

    quote_df = _fetch_quote_frame(cfg, live=live)
    oi_df = _fetch_open_interest_frame(cfg, live=live)
    vol_df = _fetch_volume_frame(cfg, live=live)
    df = _merge_theta_frames(quote_df, oi_df, vol_df)

    if df.empty:
        raise ThetaDataError(f"Theta returned no option rows for {ticker} on {valuation_date}")

    spot = _fetch_underlying_spot_yfinance(ticker, valuation_date=valuation_date, snapshot_time=snapshot_time, timezone=timezone)
    as_of = pd.Timestamp(valuation_date).tz_localize(None).normalize()
    carry = resolve_carry_estimate(
        ticker=ticker,
        spot=float(spot),
        as_of_date=as_of,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
    )

    df["date"] = as_of
    df["underlying_price"] = float(spot)
    df["maturity_date"] = pd.to_datetime(df["maturity_date"], errors="coerce").dt.tz_localize(None)
    df["time_to_expiry"] = (df["maturity_date"] - as_of).dt.days / 365.0
    df["risk_free_rate"] = float(carry.risk_free_rate)
    df["dividend_yield"] = float(carry.dividend_yield)
    df["risk_free_rate_source"] = str(carry.risk_free_rate_source)
    df["dividend_yield_source"] = str(carry.dividend_yield_source)
    df["provider"] = "theta_option_chain"
    df["vendor_implied_volatility"] = np.nan

    df = df[(df["time_to_expiry"] * 365.0 >= float(min_days)) & (df["time_to_expiry"] * 365.0 <= float(max_days))].copy()
    if df.empty:
        raise ThetaDataError(f"No Theta rows remained after maturity filtering for {ticker} on {valuation_date}")

    expiries = pd.Series(sorted(df["maturity_date"].dropna().unique()))
    if max_expiries is not None and int(max_expiries) > 0 and len(expiries) > int(max_expiries):
        keep = set(expiries.iloc[: int(max_expiries)])
        df = df[df["maturity_date"].isin(keep)].copy()

    cols = [
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
        "quote_timestamp",
        "quote_bid_size",
        "quote_ask_size",
        "provider",
    ]
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
    return df[cols].sort_values(["date", "maturity_date", "option_type", "strike"]).reset_index(drop=True)
