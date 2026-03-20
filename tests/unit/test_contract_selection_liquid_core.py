import pandas as pd

from hedging.market_panel_backtest import select_listed_contracts


def test_select_listed_contracts_prefers_liquid_near_atm_quotes():
    entry_date = pd.Timestamp("2026-03-17")
    maturity_date = pd.Timestamp("2026-03-25")
    rows = [
        {
            "date": entry_date,
            "maturity_date": maturity_date,
            "time_to_expiry": 8.0 / 365.0,
            "underlying_price": 100.0,
            "risk_free_rate": 0.04,
            "dividend_yield": 0.01,
            "strike": 100.0,
            "option_type": "call",
            "mid": 2.0,
            "bid": 1.95,
            "ask": 2.05,
            "volume": 80,
            "open_interest": 500,
            "contract_symbol": "GOOD",
        },
        {
            "date": entry_date,
            "maturity_date": maturity_date,
            "time_to_expiry": 8.0 / 365.0,
            "underlying_price": 100.0,
            "risk_free_rate": 0.04,
            "dividend_yield": 0.01,
            "strike": 100.0,
            "option_type": "call",
            "mid": 2.0,
            "bid": 1.70,
            "ask": 2.30,
            "volume": 5,
            "open_interest": 20,
            "contract_symbol": "BAD",
        },
    ]
    panel = pd.DataFrame(rows)

    contracts = select_listed_contracts(
        panel,
        option_type="call",
        target_T=8.0 / 365.0,
        strike_mode="atm",
        entry_step_days=1,
        max_contracts=1,
        contract_moneyness_band=0.02,
        contract_max_relative_spread=0.05,
        contract_min_volume=25.0,
        contract_min_open_interest=200.0,
    )

    assert len(contracts) == 1
    assert contracts[0].contract_symbol == "GOOD"
