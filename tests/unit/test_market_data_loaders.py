import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from market_data.loaders import (
    standardize_columns,
    load_csv,
    LoadConfig,
)


def _minimal_raw_df():
    return pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "maturity_date": ["2024-06-01"],
            "underlying_price": [100.0],
            "strike": [100.0],
            "option_type": ["C"],
            "bid": [4.9],
            "ask": [5.1],
            "risk_free_rate": [0.02],
            "dividend_yield": [0.01],
        }
    )


def test_standardize_columns_basic_schema():
    df = _minimal_raw_df()
    out = standardize_columns(df)

    required = {
        "date",
        "maturity_date",
        "time_to_expiry",
        "underlying_price",
        "strike",
        "option_type",
        "bid",
        "ask",
        "mid",
        "risk_free_rate",
        "dividend_yield",
    }
    assert required.issubset(out.columns)

    assert out.loc[0, "option_type"] == "call"
    assert np.isnan(out.loc[0, "mid"])

def test_standardize_preserves_extra_columns():
    df = _minimal_raw_df()
    df["extra_col"] = 123
    out = standardize_columns(df)
    assert "extra_col" in out.columns


def test_load_csv_roundtrip(tmp_path: Path):
    df = _minimal_raw_df()
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)

    out = load_csv(p)
    assert len(out) == 1
    assert out.loc[0, "strike"] == 100.0
