import pandas as pd
import numpy as np

from market_data.transforms import (
    add_derived_columns,
    compute_time_to_expiry_years,
    TransformConfig,
)


def _df_dates():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "maturity_date": pd.to_datetime(["2025-01-01"]),
            "underlying_price": [100.0],
            "strike": [110.0],
            "risk_free_rate": [0.05],
            "dividend_yield": [0.02],
        }
    )


def test_compute_time_to_expiry_act365():
    df = _df_dates()
    T = compute_time_to_expiry_years(
        df["date"], df["maturity_date"], day_count="ACT/365"
    )
    delta_days = (pd.Timestamp("2025-01-01") - pd.Timestamp("2024-01-01")).days
    expected = delta_days / 365.0
    assert np.isclose(T.iloc[0], expected, atol=1e-12)


def test_add_forward_and_log_moneyness():
    df = _df_dates()
    out = add_derived_columns(df)

    assert "forward" in out.columns
    assert "log_moneyness" in out.columns

    T = out.loc[0, "time_to_expiry"]
    F = 100.0 * np.exp((0.05 - 0.02) * T)
    assert np.isclose(out.loc[0, "forward"], F)
    assert np.isclose(out.loc[0, "log_moneyness"], np.log(110.0 / F))


def test_preserve_existing_time_to_expiry():
    df = _df_dates()
    df["time_to_expiry"] = 0.5

    cfg = TransformConfig(compute_time_to_expiry_if_missing=True)
    out = add_derived_columns(df, cfg=cfg)

    assert np.isclose(out.loc[0, "time_to_expiry"], 0.5)
