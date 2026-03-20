import numpy as np
import pandas as pd

from implied_volatility.black_scholes import put_call_parity_call_from_put
from pipeline.calibration import _prepare_synthetic_call_quotes


def test_prepare_synthetic_call_quotes_prefers_otm_wings_and_preserves_parity():
    S = 100.0
    T = 30.0 / 365.0
    r = 0.03
    q = 0.01
    forward = S * np.exp((r - q) * T)

    df = pd.DataFrame(
        [
            {
                "option_type": "call",
                "underlying_price": S,
                "strike": 90.0,
                "time_to_expiry": T,
                "risk_free_rate": r,
                "dividend_yield": q,
                "bid": 11.8,
                "ask": 12.2,
                "mid": 12.0,
                "volume": 100,
                "open_interest": 200,
                "forward": forward,
                "relative_spread": 0.02,
                "moneyness": 0.90,
                "log_moneyness": np.log(90.0 / forward),
            },
            {
                "option_type": "put",
                "underlying_price": S,
                "strike": 90.0,
                "time_to_expiry": T,
                "risk_free_rate": r,
                "dividend_yield": q,
                "bid": 0.8,
                "ask": 1.2,
                "mid": 1.0,
                "volume": 300,
                "open_interest": 500,
                "forward": forward,
                "relative_spread": 0.01,
                "moneyness": 0.90,
                "log_moneyness": np.log(90.0 / forward),
            },
            {
                "option_type": "call",
                "underlying_price": S,
                "strike": 110.0,
                "time_to_expiry": T,
                "risk_free_rate": r,
                "dividend_yield": q,
                "bid": 1.4,
                "ask": 1.8,
                "mid": 1.6,
                "volume": 350,
                "open_interest": 600,
                "forward": forward,
                "relative_spread": 0.01,
                "moneyness": 1.10,
                "log_moneyness": np.log(110.0 / forward),
            },
            {
                "option_type": "put",
                "underlying_price": S,
                "strike": 110.0,
                "time_to_expiry": T,
                "risk_free_rate": r,
                "dividend_yield": q,
                "bid": 11.7,
                "ask": 12.1,
                "mid": 11.9,
                "volume": 120,
                "open_interest": 180,
                "forward": forward,
                "relative_spread": 0.02,
                "moneyness": 1.10,
                "log_moneyness": np.log(110.0 / forward),
            },
        ]
    )

    merged, report = _prepare_synthetic_call_quotes(df)
    merged = merged.sort_values("strike").reset_index(drop=True)

    assert len(merged) == 2
    assert report["source_counts"] == {"put_parity": 1, "call_direct": 1}
    assert report["otm_source_fraction"] == 1.0

    left = merged.iloc[0]
    right = merged.iloc[1]

    assert left["option_type"] == "call"
    assert left["surface_quote_source"] == "put_parity"
    np.testing.assert_allclose(
        left["mid"],
        put_call_parity_call_from_put(1.0, S, 90.0, T, r=r, q=q),
        rtol=0.0,
        atol=1e-12,
    )

    assert right["option_type"] == "call"
    assert right["surface_quote_source"] == "call_direct"
    np.testing.assert_allclose(right["mid"], 1.6, rtol=0.0, atol=1e-12)
