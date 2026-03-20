import numpy as np
import pandas as pd

from hedging.delta_hedger import DeltaHedger, BSDeltaModel
from hedging.transaction_costs import TransactionCostModel


def make_gbm_path(S0, r, sigma, T, n_steps, seed):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)
    S = np.empty(n_steps + 1)
    S[0] = S0
    for i in range(n_steps):
        Z = rng.standard_normal()
        S[i + 1] = S[i] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return pd.DataFrame({"t": times, "S": S})


def test_transaction_costs_increase_hedging_error_variance():
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.01
    q = 0.0
    sigma = 0.2

    model = BSDeltaModel(
        S0_ref=S0,
        K=K,
        T=T,
        r=r,
        q=q,
        sigma=sigma,
        option_type="call",
    )

    hedger_free = DeltaHedger(
        model=model,
        K=K,
        T=T,
        r=r,
        option_type="call",
        tx_costs=TransactionCostModel(kind="none"),
    )

    hedger_costly = DeltaHedger(
        model=model,
        K=K,
        T=T,
        r=r,
        option_type="call",
        tx_costs=TransactionCostModel(kind="proportional", rate=0.001),
    )

    seeds = range(30)

    errs_free = []
    errs_costly = []

    for seed in seeds:
        path = make_gbm_path(S0, r, sigma, T, n_steps=252, seed=seed)
        out_free = hedger_free.run_path(path["t"].to_numpy(), path["S"].to_numpy())
        out_costly = hedger_costly.run_path(path["t"].to_numpy(), path["S"].to_numpy())

        errs_free.append(out_free["hedge_error"])
        errs_costly.append(out_costly["hedge_error"])

    var_free = np.var(errs_free, ddof=1)
    var_costly = np.var(errs_costly, ddof=1)

    assert var_costly > var_free
