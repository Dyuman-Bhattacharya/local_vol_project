from __future__ import annotations

import numpy as np

from hedging.market_panel_backtest import (
    build_daily_calibration_cache,
    run_daily_market_panel_backtest,
    select_listed_contracts,
)
from hedging.metrics import summarize_market_panel_backtest
from hedging.transaction_costs import TransactionCostModel
from pipeline.calibration import SnapshotCalibrationConfig
from pricing.pde_solver import PDEConfig
from tests.fixtures.synthetic_data import BSDailyPanelConfig, generate_bs_daily_panel


def _required_dates(panel, contract):
    return [
        d
        for d in panel["date"].drop_duplicates().sort_values().tolist()
        if contract.entry_date <= d < contract.maturity_date
    ]


def test_select_listed_contracts_uses_fixed_listed_contracts() -> None:
    panel, _ = generate_bs_daily_panel(BSDailyPanelConfig())
    contracts = select_listed_contracts(
        panel,
        option_type="call",
        target_T=8.0 / 365.0,
        strike_mode="atm",
        entry_step_days=20,
        max_contracts=1,
    )

    assert len(contracts) == 1
    contract = contracts[0]
    entry_snapshot = panel[(panel["date"] == contract.entry_date) & (panel["maturity_date"] == contract.maturity_date)]
    listed_strikes = set(np.round(entry_snapshot["strike"].to_numpy(dtype=float), 4))
    assert round(contract.strike, 4) in listed_strikes
    assert contract.market_premium > 0.0


def test_daily_market_panel_backtest_separates_pricing_and_replication() -> None:
    panel, history = generate_bs_daily_panel(BSDailyPanelConfig())
    contracts = select_listed_contracts(
        panel,
        option_type="call",
        target_T=8.0 / 365.0,
        strike_mode="atm",
        entry_step_days=20,
        max_contracts=1,
    )
    contract = contracts[0]
    cache = build_daily_calibration_cache(
        panel,
        ticker="SYNTH",
        snapshot_cfg=SnapshotCalibrationConfig(
            option_type="call",
            min_volume=0.0,
            min_open_interest=0.0,
            moneyness_min=0.80,
            moneyness_max=1.20,
            max_relative_spread=0.20,
            iv_min=0.01,
            iv_max=1.0,
            min_points_per_slice=5,
            n_strikes=35,
        ),
        required_dates=_required_dates(panel, contract),
    )

    tx = TransactionCostModel(kind="proportional", rate=0.0)
    pde_cfg = PDEConfig(n_space=80, n_time=60)
    bs = run_daily_market_panel_backtest(
        contracts=contracts,
        panel=panel,
        history=history,
        calibration_cache=cache,
        model="BS",
        tx_costs=tx,
        pde_cfg=pde_cfg,
    )
    lv = run_daily_market_panel_backtest(
        contracts=contracts,
        panel=panel,
        history=history,
        calibration_cache=cache,
        model="LocalVol",
        tx_costs=tx,
        pde_cfg=pde_cfg,
    )

    assert float(bs["market_premium"].iloc[0]) == float(lv["market_premium"].iloc[0])
    assert float(bs["strike"].iloc[0]) == float(lv["strike"].iloc[0])
    assert abs(float(bs["entry_pricing_error"].iloc[0])) < 0.10
    assert abs(float(bs["replication_error_net"].iloc[0])) < 0.75

    diff = float(lv["market_pnl_net"].iloc[0] - lv["replication_error_net"].iloc[0])
    expected = float(lv["market_premium"].iloc[0] - lv["model_entry_price"].iloc[0])
    assert abs(diff - expected) < 1e-4

    summary = summarize_market_panel_backtest(lv)
    assert "pricing_error_mean" in summary
    assert "replication_net_mean" in summary
    assert "market_pnl_net_mean" in summary
