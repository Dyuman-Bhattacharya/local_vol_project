from __future__ import annotations

import pandas as pd

from market_data.carry import (
    apply_carry_to_snapshot_frame,
    normalize_rate_quote,
    resolve_carry_estimate,
)


def test_normalize_rate_quote_percent_to_decimal() -> None:
    assert normalize_rate_quote(4.25) == 0.0425
    assert normalize_rate_quote(0.0425) == 0.0425


def test_resolve_carry_estimate_prefers_manual_values() -> None:
    est = resolve_carry_estimate(
        ticker="SPY",
        spot=500.0,
        as_of_date="2026-03-18",
        risk_free_rate=0.031,
        dividend_yield=0.012,
    )
    assert est.risk_free_rate == 0.031
    assert est.dividend_yield == 0.012
    assert est.risk_free_rate_source == "manual"
    assert est.dividend_yield_source == "manual"


def test_apply_carry_to_snapshot_frame_overwrites_stale_embedded_values() -> None:
    df = pd.DataFrame(
        {
            "underlying_price": [500.0, 500.0],
            "risk_free_rate": [0.04, 0.04],
            "dividend_yield": [0.0, 0.0],
        }
    )
    out = apply_carry_to_snapshot_frame(
        df,
        ticker="SPY",
        valuation_date="2026-03-18",
        risk_free_rate=0.031,
        dividend_yield=0.012,
    )
    assert (out["risk_free_rate"] == 0.031).all()
    assert (out["dividend_yield"] == 0.012).all()
    assert (out["risk_free_rate_source"] == "manual").all()
    assert (out["dividend_yield_source"] == "manual").all()
