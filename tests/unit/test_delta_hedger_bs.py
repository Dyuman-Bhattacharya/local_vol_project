import numpy as np
import pandas as pd

from hedging.delta_hedger import DeltaHedger, BSDeltaModel
from hedging.transaction_costs import TransactionCostModel


def make_gbm_path(S0, r, sigma, T, n_steps, seed=0):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)
    S = np.empty(n_steps + 1)
    S[0] = S0
    for i in range(n_steps):
        Z = rng.standard_normal()
        S[i + 1] = S[i] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return pd.DataFrame({"t": times, "S": S})


def test_bs_delta_hedger_runs_and_error_is_finite():
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.02
    q = 0.0
    sigma = 0.2

    path = make_gbm_path(S0, r, sigma, T, n_steps=252, seed=42)

    model = BSDeltaModel(
        S0_ref=S0,
        K=K,
        T=T,
        r=r,
        q=q,
        sigma=sigma,
        option_type="call",
    )

    hedger = DeltaHedger(
        model=model,
        K=K,
        T=T,
        r=r,
        option_type="call",
        tx_costs=TransactionCostModel(kind="none"),
    )

    out = hedger.run_path(path["t"].to_numpy(), path["S"].to_numpy())

    assert np.isfinite(out["hedge_error"])
    # Should not be absurdly large relative to spot
    assert abs(out["hedge_error"]) < 0.2 * S0
