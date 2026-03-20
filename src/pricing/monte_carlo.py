# src/pricing/monte_carlo.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

from local_volatility.surface import LocalVolSurface
from implied_volatility.black_scholes import bs_price


OptionType = Literal["call", "put"]


class MonteCarloError(RuntimeError):
    """Raised when MC simulation cannot be run due to invalid inputs."""


@dataclass(frozen=True)
class MCConfig:
    n_paths: int = 100_000
    n_steps: int = 252
    seed: Optional[int] = 123
    variance_reduction: Tuple[Literal["antithetic", "control_variate"], ...] = ("antithetic",)

    # Control variate settings (BS with constant vol)
    control_variate_sigma: Optional[float] = None  # if None, no control variate


def _payoff(ST: np.ndarray, K: float, option_type: OptionType) -> np.ndarray:
    if option_type == "call":
        return np.maximum(ST - K, 0.0)
    if option_type == "put":
        return np.maximum(K - ST, 0.0)
    raise ValueError("option_type must be 'call' or 'put'")


def price_european_mc_local_vol(
    *,
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: OptionType,
    lv_surface: LocalVolSurface,
    cfg: MCConfig = MCConfig(),
) -> Tuple[float, float]:
    """
    Monte Carlo pricing under local volatility using log-Euler scheme:

      S_{t+dt} = S_t * exp( (r-q - 0.5 sigma^2) dt + sigma sqrt(dt) Z )

    Returns
    -------
    price, stderr
    """
    if S0 <= 0 or K <= 0:
        raise MonteCarloError("S0 and K must be positive")
    if T <= 0:
        payoff0 = max(S0 - K, 0.0) if option_type == "call" else max(K - S0, 0.0)
        return float(payoff0), 0.0

    n_paths = int(cfg.n_paths)
    n_steps = int(cfg.n_steps)
    if n_paths < 1000 or n_steps < 10:
        raise MonteCarloError("n_paths/n_steps too small")

    rng = np.random.default_rng(cfg.seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    antithetic = "antithetic" in cfg.variance_reduction
    use_cv = ("control_variate" in cfg.variance_reduction) and (cfg.control_variate_sigma is not None)

    # Effective path count for antithetic
    n_base = n_paths // 2 if antithetic else n_paths
    if n_base < 1:
        raise MonteCarloError("Too few paths for chosen variance reduction")

    # Simulate
    S = np.full(n_base, S0, dtype=float)
    if antithetic:
        S_a = np.full(n_base, S0, dtype=float)

    for k in range(n_steps):
        t = k * dt
        sig = lv_surface.sigma(S, t)
        sig2 = sig * sig
        drift = (r - q - 0.5 * sig2) * dt

        Z = rng.standard_normal(n_base)
        incr = sig * sqrt_dt * Z
        S = S * np.exp(drift + incr)

        if antithetic:
            incr_a = sig * sqrt_dt * (-Z)
            S_a = S_a * np.exp(drift + incr_a)

    if antithetic:
        ST = np.concatenate([S, S_a], axis=0)
    else:
        ST = S

    disc = np.exp(-r * T)
    payoff_vals = disc * _payoff(ST, K, option_type)

    if use_cv:
        sigma_cv = float(cfg.control_variate_sigma)
        # Control variate: BS price with constant vol sigma_cv under same (r,q)
        bs = bs_price(S0, K, T, r, q, sigma_cv, option_type)
        # Simulate same payoff under BS with sigma_cv along the *same terminal ST* is not correct.
        # Instead use terminal payoff discounted as control, with known expectation = BS price.
        # For that we need ST under BS. We didn't simulate BS paths. So do an additional cheap BS draw.
        # This keeps implementation honest and simple.
        rng2 = np.random.default_rng((cfg.seed or 0) + 9991)
        Z2 = rng2.standard_normal(ST.size)
        ST_bs = S0 * np.exp((r - q - 0.5 * sigma_cv**2) * T + sigma_cv * np.sqrt(T) * Z2)
        cv_payoff = disc * _payoff(ST_bs, K, option_type)

        # Optimal beta = Cov(X,Y)/Var(Y)
        X = payoff_vals
        Y = cv_payoff
        cov = np.cov(X, Y, ddof=1)
        beta = cov[0, 1] / (cov[1, 1] + 1e-18)
        adj = X - beta * (Y - bs)
        price = float(np.mean(adj))
        stderr = float(np.std(adj, ddof=1) / np.sqrt(adj.size))
        return price, stderr

    price = float(np.mean(payoff_vals))
    stderr = float(np.std(payoff_vals, ddof=1) / np.sqrt(payoff_vals.size))
    return price, stderr
