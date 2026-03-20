from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional

import numpy as np
import pandas as pd


SnapshotKind = Literal["1545", "eod"]


class CboeEODImportError(RuntimeError):
    """Raised when a Cboe Option EOD Summary file cannot be parsed."""


@dataclass(frozen=True)
class CboeEODConfig:
    snapshot: SnapshotKind = "1545"
    risk_free_rate: Optional[float] = None
    dividend_yield: Optional[float] = None
    underlying_filter: Optional[list[str]] = None
    keep_calc_columns: bool = True


def _normalize_name(name: str) -> str:
    chars = []
    for ch in str(name).strip().lower():
        if ch.isalnum():
            chars.append(ch)
        else:
            chars.append("_")
    out = "".join(chars)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _find_column(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    normalized = {_normalize_name(c): c for c in df.columns}
    for cand in candidates:
        col = normalized.get(_normalize_name(cand))
        if col is not None:
            return col
    return None


def _read_one(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise CboeEODImportError(f"File not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".zip", ".gz"}:
        return pd.read_csv(path, compression="infer")
    raise CboeEODImportError(f"Unsupported Cboe EOD file type: {path.suffix}")


def _format_occ_contract_symbol(root: str, expiry: pd.Timestamp, option_type: str, strike: float) -> str:
    root_clean = str(root).upper().strip()
    yy = expiry.strftime("%y")
    mm = expiry.strftime("%m")
    dd = expiry.strftime("%d")
    cp = "C" if str(option_type).lower().startswith("c") else "P"
    strike_int = int(round(float(strike) * 1000.0))
    return f"{root_clean:<6}{yy}{mm}{dd}{cp}{strike_int:08d}"


def normalize_cboe_option_eod_summary(
    df_raw: pd.DataFrame,
    *,
    cfg: CboeEODConfig = CboeEODConfig(),
) -> pd.DataFrame:
    if not isinstance(df_raw, pd.DataFrame):
        raise CboeEODImportError("Input must be a pandas DataFrame")

    quote_date_col = _find_column(df_raw, "Quote Date")
    expiry_col = _find_column(df_raw, "Expiration")
    underlying_col = _find_column(df_raw, "Underlying Symbol")
    root_col = _find_column(df_raw, "Root")
    strike_col = _find_column(df_raw, "Strike")
    option_type_col = _find_column(df_raw, "Option Type")
    vol_col = _find_column(df_raw, "Trade Volume")
    oi_col = _find_column(df_raw, "Open Interest")
    open_col = _find_column(df_raw, "Open")
    high_col = _find_column(df_raw, "High")
    low_col = _find_column(df_raw, "Low")
    close_col = _find_column(df_raw, "Close")
    vwap_col = _find_column(df_raw, "VWAP")

    if quote_date_col is None or expiry_col is None or underlying_col is None or strike_col is None or option_type_col is None:
        raise CboeEODImportError(
            "Cboe EOD file is missing required columns. "
            f"Columns present: {list(df_raw.columns)}"
        )

    if cfg.snapshot == "1545":
        bid_col = _find_column(df_raw, "Bid 1545")
        ask_col = _find_column(df_raw, "Ask 1545")
        bid_size_col = _find_column(df_raw, "Bid Size 1545")
        ask_size_col = _find_column(df_raw, "Ask Size 1545")
        und_bid_col = _find_column(df_raw, "Underlying Bid 1545")
        und_ask_col = _find_column(df_raw, "Underlying Ask 1545")
        und_active_col = _find_column(df_raw, "Active Underlying Price 1545")
        iv_col = _find_column(df_raw, "Implied Volatility 1545")
        delta_col = _find_column(df_raw, "Delta 1545")
        gamma_col = _find_column(df_raw, "Gamma 1545")
        theta_col = _find_column(df_raw, "Theta 1545")
        vega_col = _find_column(df_raw, "Vega 1545")
        rho_col = _find_column(df_raw, "Rho 1545")
        quote_time = pd.Timedelta(hours=15, minutes=45)
    else:
        bid_col = _find_column(df_raw, "Bid EOD")
        ask_col = _find_column(df_raw, "Ask EOD")
        bid_size_col = _find_column(df_raw, "Bid Size EOD")
        ask_size_col = _find_column(df_raw, "Ask Size EOD")
        und_bid_col = _find_column(df_raw, "Underlying Bid EOD")
        und_ask_col = _find_column(df_raw, "Underlying Ask EOD")
        und_active_col = None
        iv_col = delta_col = gamma_col = theta_col = vega_col = rho_col = None
        quote_time = pd.Timedelta(hours=16)

    if bid_col is None or ask_col is None:
        raise CboeEODImportError(
            f"Cboe EOD file does not contain {cfg.snapshot} bid/ask columns. Columns present: {list(df_raw.columns)}"
        )

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df_raw[quote_date_col], errors="coerce").dt.normalize()
    out["quote_timestamp"] = out["date"] + quote_time
    out["maturity_date"] = pd.to_datetime(df_raw[expiry_col], errors="coerce").dt.normalize()
    out["underlying"] = df_raw[underlying_col].astype(str).str.upper().str.strip()
    out["root"] = (
        df_raw[root_col].astype(str).str.upper().str.strip()
        if root_col is not None
        else out["underlying"]
    )
    out["strike"] = pd.to_numeric(df_raw[strike_col], errors="coerce")
    out["option_type"] = df_raw[option_type_col].astype(str).str.lower().str.strip()
    out.loc[out["option_type"].isin(["c", "call"]), "option_type"] = "call"
    out.loc[out["option_type"].isin(["p", "put"]), "option_type"] = "put"
    out["bid"] = pd.to_numeric(df_raw[bid_col], errors="coerce")
    out["ask"] = pd.to_numeric(df_raw[ask_col], errors="coerce")

    mid = 0.5 * (out["bid"] + out["ask"])
    if close_col is not None:
        close_px = pd.to_numeric(df_raw[close_col], errors="coerce")
        mid = mid.where(np.isfinite(mid), close_px)
    if vwap_col is not None:
        vwap = pd.to_numeric(df_raw[vwap_col], errors="coerce")
        mid = mid.where(np.isfinite(mid), vwap)
    out["mid"] = mid

    und_bid = pd.to_numeric(df_raw[und_bid_col], errors="coerce") if und_bid_col else pd.Series(np.nan, index=df_raw.index)
    und_ask = pd.to_numeric(df_raw[und_ask_col], errors="coerce") if und_ask_col else pd.Series(np.nan, index=df_raw.index)
    und_active = pd.to_numeric(df_raw[und_active_col], errors="coerce") if und_active_col else pd.Series(np.nan, index=df_raw.index)
    out["underlying_price"] = und_active
    out["underlying_price"] = out["underlying_price"].where(np.isfinite(out["underlying_price"]), 0.5 * (und_bid + und_ask))

    out["volume"] = pd.to_numeric(df_raw[vol_col], errors="coerce") if vol_col else 0.0
    out["open_interest"] = pd.to_numeric(df_raw[oi_col], errors="coerce") if oi_col else np.nan
    out["risk_free_rate"] = float(cfg.risk_free_rate) if cfg.risk_free_rate is not None else np.nan
    out["dividend_yield"] = float(cfg.dividend_yield) if cfg.dividend_yield is not None else np.nan
    out["time_to_expiry"] = (out["maturity_date"] - out["date"]).dt.days.astype(float) / 365.0
    out["contract_symbol"] = [
        _format_occ_contract_symbol(root, expiry, option_type, strike)
        if pd.notna(expiry) and pd.notna(strike)
        else np.nan
        for root, expiry, option_type, strike in zip(
            out["root"],
            out["maturity_date"],
            out["option_type"],
            out["strike"],
        )
    ]
    out["provider"] = "cboe_option_eod_summary"
    out["snapshot_kind"] = cfg.snapshot

    if bid_size_col is not None:
        out["bid_size"] = pd.to_numeric(df_raw[bid_size_col], errors="coerce")
    if ask_size_col is not None:
        out["ask_size"] = pd.to_numeric(df_raw[ask_size_col], errors="coerce")
    if open_col is not None:
        out["open"] = pd.to_numeric(df_raw[open_col], errors="coerce")
    if high_col is not None:
        out["high"] = pd.to_numeric(df_raw[high_col], errors="coerce")
    if low_col is not None:
        out["low"] = pd.to_numeric(df_raw[low_col], errors="coerce")
    if close_col is not None:
        out["close"] = pd.to_numeric(df_raw[close_col], errors="coerce")
    if vwap_col is not None:
        out["vwap"] = pd.to_numeric(df_raw[vwap_col], errors="coerce")

    if cfg.keep_calc_columns:
        for dest, src in [
            ("implied_volatility_1545", iv_col),
            ("delta_1545", delta_col),
            ("gamma_1545", gamma_col),
            ("theta_1545", theta_col),
            ("vega_1545", vega_col),
            ("rho_1545", rho_col),
        ]:
            if src is not None:
                out[dest] = pd.to_numeric(df_raw[src], errors="coerce")

    out = out.dropna(subset=["date", "maturity_date", "underlying", "strike", "option_type"])

    if cfg.underlying_filter:
        allowed = {str(x).upper().strip() for x in cfg.underlying_filter}
        out = out[out["underlying"].isin(allowed)].copy()

    out = out.sort_values(["date", "underlying", "maturity_date", "option_type", "strike"]).reset_index(drop=True)
    return out


def load_cboe_option_eod_summary(
    paths: Iterable[str | Path],
    *,
    cfg: CboeEODConfig = CboeEODConfig(),
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frames.append(normalize_cboe_option_eod_summary(_read_one(path), cfg=cfg))
    if not frames:
        raise CboeEODImportError("No Cboe EOD files provided")
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["date", "contract_symbol"]).sort_values(["date", "contract_symbol"]).reset_index(drop=True)
    return out
