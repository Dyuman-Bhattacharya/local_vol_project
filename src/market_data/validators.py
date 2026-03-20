# src/market_data/validators.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


class MarketDataValidationError(RuntimeError):
    """Raised when validation fails in a fatal way (e.g., missing required columns)."""


@dataclass(frozen=True)
class ValidationConfig:
    """
    Validation rules and filters.

    Units/Conventions:
    - Rates r, q are continuously compounded annualized (decimal, e.g. 0.05).
    - time_to_expiry is in years.
    """
    require_columns: Tuple[str, ...] = (
        "date",
        "underlying_price",
        "strike",
        "option_type",
        # Either time_to_expiry OR maturity_date must be present for later transforms
    )

    # Basic sanity bounds
    min_spot: float = 1e-8
    min_strike: float = 1e-8
    max_moneyness_multiple: float = 10.0  # require 0 < K < 10*S by default
    min_time_to_expiry: float = 0.0  # allow 0 (but later IV will return NaN)
    max_time_to_expiry: float = 100.0

    # Market quote sanity
    allow_negative_mid: bool = False
    enforce_bid_ask: bool = True  # require bid <= ask when both present
    compute_mid_if_missing: bool = True

    # Optional liquidity filters (set to None to disable)
    min_volume: Optional[float] = None
    min_open_interest: Optional[float] = None


def validate_and_clean(
    df: pd.DataFrame,
    *,
    cfg: ValidationConfig = ValidationConfig(),
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Validate and clean standardized option-chain data.

    Returns
    -------
    clean_df : pd.DataFrame
        Subset of rows passing validation + with computed mid where needed.
    report : dict
        Counts and masks to explain what was dropped and why.
    """
    if not isinstance(df, pd.DataFrame):
        raise MarketDataValidationError("Input must be a pandas.DataFrame")

    missing = [c for c in cfg.require_columns if c not in df.columns]
    if missing:
        raise MarketDataValidationError(f"Missing required columns: {missing}")

    out = df.copy()

    # Ensure option_type is only call/put (drop others)
    mask_type = out["option_type"].isin(["call", "put"])

    # Spot/strike positivity
    mask_spot = out["underlying_price"].astype(float) > cfg.min_spot
    mask_strike = out["strike"].astype(float) > cfg.min_strike

    # Moneyness sanity: K < max_multiple * S (rough outlier kill)
    # Only apply when both are finite.
    S = out["underlying_price"].astype(float)
    K = out["strike"].astype(float)
    mask_moneyness = ~(np.isfinite(S) & np.isfinite(K)) | (K < cfg.max_moneyness_multiple * S)

    # Time-to-expiry sanity (if present)
    if "time_to_expiry" in out.columns:
        T = out["time_to_expiry"].astype(float)
        mask_T = T.between(cfg.min_time_to_expiry, cfg.max_time_to_expiry, inclusive="both") | T.isna()
    else:
        mask_T = pd.Series(True, index=out.index)

    # Date sanity (allow NaT to be fixed later, but prefer real dates)
    mask_date = out["date"].notna() if "date" in out.columns else pd.Series(True, index=out.index)

    # Bid/ask/mid sanity
    bid = out["bid"].astype(float) if "bid" in out.columns else pd.Series(np.nan, index=out.index)
    ask = out["ask"].astype(float) if "ask" in out.columns else pd.Series(np.nan, index=out.index)
    mid = out["mid"].astype(float) if "mid" in out.columns else pd.Series(np.nan, index=out.index)

    # If mid missing, compute from bid/ask if allowed and both available
    if cfg.compute_mid_if_missing:
        mid_missing = mid.isna()
        have_ba = bid.notna() & ask.notna()
        out.loc[mid_missing & have_ba, "mid"] = 0.5 * (bid[mid_missing & have_ba] + ask[mid_missing & have_ba])

    # Recompute mid after possible fill
    mid = out["mid"].astype(float)

    # Price positivity
    if cfg.allow_negative_mid:
        mask_mid = mid.notna()
    else:
        mask_mid = mid.notna() & (mid >= 0.0)

    # Bid <= ask check (only when both present)
    if cfg.enforce_bid_ask:
        mask_ba = ~(bid.notna() & ask.notna()) | (bid <= ask)
    else:
        mask_ba = pd.Series(True, index=out.index)

    # Liquidity filters
    if cfg.min_volume is not None and "volume" in out.columns:
        mask_vol = out["volume"].fillna(0.0) >= cfg.min_volume
    else:
        mask_vol = pd.Series(True, index=out.index)

    if cfg.min_open_interest is not None and "open_interest" in out.columns:
        mask_oi = out["open_interest"].fillna(0.0) >= cfg.min_open_interest
    else:
        mask_oi = pd.Series(True, index=out.index)

    # Combine
    mask_all = (
        mask_type
        & mask_spot
        & mask_strike
        & mask_moneyness
        & mask_T
        & mask_date
        & mask_mid
        & mask_ba
        & mask_vol
        & mask_oi
    )

    clean = out.loc[mask_all].copy()

    report = {
        "n_in": int(len(df)),
        "n_out": int(len(clean)),
        "dropped": int(len(df) - len(clean)),
        "drop_counts": {
            "bad_option_type": int((~mask_type).sum()),
            "bad_spot": int((~mask_spot).sum()),
            "bad_strike": int((~mask_strike).sum()),
            "bad_moneyness": int((~mask_moneyness).sum()),
            "bad_time_to_expiry": int((~mask_T).sum()),
            "missing_or_bad_date": int((~mask_date).sum()),
            "bad_mid": int((~mask_mid).sum()),
            "bad_bid_ask": int((~mask_ba).sum()),
            "below_min_volume": int((~mask_vol).sum()),
            "below_min_open_interest": int((~mask_oi).sum()),
        },
        "mask_pass": mask_all,
    }

    return clean, report
