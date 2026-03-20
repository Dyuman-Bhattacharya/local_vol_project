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


def run_errors(n_steps, seeds):
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.01
    q = 0.0
    sigma = 0.25

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

    errs = []
    for seed in seeds:
        path = make_gbm_path(S0, r, sigma, T, n_steps, seed)
        out = hedger.run_path(path["t"].to_numpy(), path["S"].to_numpy())
        errs.append(out["hedge_error"])
    return np.array(errs)


def test_hedging_error_variance_decreases_with_rebalancing():
    seeds = range(20)

    # coarse vs fine
    errs_coarse = run_errors(n_steps=52, seeds=seeds)   # weekly
    errs_fine = run_errors(n_steps=252, seeds=seeds)    # daily

    var_coarse = np.var(errs_coarse, ddof=1)
    var_fine = np.var(errs_fine, ddof=1)

    assert var_fine < var_coarse
