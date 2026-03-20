from __future__ import annotations

import pandas as pd

from market_data.cboe_eod import CboeEODConfig, normalize_cboe_option_eod_summary


def test_normalize_cboe_eod_summary_1545_snapshot() -> None:
    raw = pd.DataFrame(
        [
            {
                "Quote Date": "2026-03-16",
                "Underlying Symbol": "SPY",
                "Root": "SPY",
                "Expiration": "2026-03-20",
                "Strike": 560.0,
                "Option Type": "C",
                "Bid 1545": 5.10,
                "Ask 1545": 5.30,
                "Underlying Bid 1545": 558.9,
                "Underlying Ask 1545": 559.1,
                "Active Underlying Price 1545": 559.0,
                "Trade Volume": 1000,
                "Open Interest": 5000,
                "Implied Volatility 1545": 0.18,
            }
        ]
    )

    out = normalize_cboe_option_eod_summary(raw, cfg=CboeEODConfig(snapshot="1545", risk_free_rate=0.04, dividend_yield=0.01))
    row = out.iloc[0]

    assert str(row["date"].date()) == "2026-03-16"
    assert str(row["maturity_date"].date()) == "2026-03-20"
    assert row["underlying"] == "SPY"
    assert row["option_type"] == "call"
    assert abs(row["mid"] - 5.20) < 1e-12
    assert row["underlying_price"] == 559.0
    assert row["risk_free_rate"] == 0.04
    assert row["dividend_yield"] == 0.01
    assert row["contract_symbol"] == "SPY   260320C00560000"


def test_normalize_cboe_eod_summary_eod_fallback_underlying_mid() -> None:
    raw = pd.DataFrame(
        [
            {
                "Quote Date": "2026-03-16",
                "Underlying Symbol": "SPY",
                "Expiration": "2026-03-27",
                "Strike": 540.0,
                "Option Type": "P",
                "Bid EOD": 4.00,
                "Ask EOD": 4.20,
                "Underlying Bid EOD": 550.0,
                "Underlying Ask EOD": 550.4,
                "Trade Volume": 10,
                "Open Interest": 25,
            }
        ]
    )

    out = normalize_cboe_option_eod_summary(raw, cfg=CboeEODConfig(snapshot="eod"))
    row = out.iloc[0]

    assert row["option_type"] == "put"
    assert abs(row["mid"] - 4.10) < 1e-12
    assert abs(row["underlying_price"] - 550.2) < 1e-12
    assert row["snapshot_kind"] == "eod"
