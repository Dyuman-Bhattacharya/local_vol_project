import pandas as pd

from market_data.theta_loader import _build_occ_symbol, _merge_theta_frames


def test_build_occ_symbol_matches_occ_style_padding():
    sym = _build_occ_symbol("SPY", "2026-03-19", "CALL", 530.0)
    assert sym == "SPY260319C00530000"


def test_merge_theta_frames_coalesces_quote_oi_and_volume_fields():
    keys = {
        "underlying": "SPY",
        "maturity_date": pd.Timestamp("2026-03-24"),
        "strike": 662.0,
        "option_type": "call",
        "contract_symbol": "SPY260324C00662000",
    }
    quote_df = pd.DataFrame(
        [
            {
                **keys,
                "quote_bid": 12.1,
                "quote_ask": 12.3,
                "quote_timestamp": "2026-03-17T15:45:00.000",
            }
        ]
    )
    oi_df = pd.DataFrame([{**keys, "oi_open_interest": 1234}])
    vol_df = pd.DataFrame(
        [
            {
                **keys,
                "eod_volume": 456,
                "eod_last_trade": "2026-03-17T15:59:59.000",
            }
        ]
    )

    out = _merge_theta_frames(quote_df, oi_df, vol_df)
    row = out.iloc[0]
    assert row["bid"] == 12.1
    assert row["ask"] == 12.3
    assert row["mid"] == 12.2
    assert row["open_interest"] == 1234
    assert row["volume"] == 456
    assert pd.Timestamp(row["last_trade_date"]).tz is not None
