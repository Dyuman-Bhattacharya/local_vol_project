# src/pricing/greeks.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

from local_volatility.surface import LocalVolSurface
from .pde_solver import price_european_pde_local_vol, PDEConfig


OptionType = Literal["call", "put"]


class GreeksError(RuntimeError):
    """Raised when Greek computation fails."""


@dataclass(frozen=True)
class GreeksConfig:
    bump_rel: float = 1e-3  # relative bump for finite differences
    bump_abs_min: float = 1e-4  # minimum absolute bump
    pde_cfg: PDEConfig = PDEConfig()


def delta_gamma_local_vol_pde(
    *,
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: OptionType,
    lv_surface: LocalVolSurface,
    cfg: GreeksConfig = GreeksConfig(),
    t0: float = 0.0,
) -> Tuple[float, float]:
    """
    Compute delta and gamma by bump-and-reprice with the PDE pricer.
    """

    if cfg is None:
        cfg = GreeksConfig()

    if S0 <= 0:
        raise GreeksError("S0 must be positive")
    
    # If pricing puts via parity, compute greeks via parity too:
    # P = C - S e^{-qT} + K e^{-rT}
    # => Δ_P = Δ_C - e^{-qT}, Γ_P = Γ_C
    if option_type == "put":
        delta_c, gamma_c = delta_gamma_local_vol_pde(
            S0=S0,
            K=K,
            T=T,
            r=r,
            q=q,
            option_type="call",
            lv_surface=lv_surface,
            cfg=cfg,
            t0=t0,
        )
        delta_p = float(delta_c - np.exp(-q * T))
        return delta_p, float(gamma_c)


    h = max(cfg.bump_abs_min, cfg.bump_rel * S0)

    p0 = price_european_pde_local_vol(
        S0=S0, K=K, T=T, r=r, q=q, option_type=option_type, lv_surface=lv_surface, cfg=cfg.pde_cfg, t0=t0,
    )
    p_up = price_european_pde_local_vol(
        S0=S0 + h, K=K, T=T, r=r, q=q, option_type=option_type, lv_surface=lv_surface, cfg=cfg.pde_cfg, t0=t0,
    )
    p_dn = price_european_pde_local_vol(
        S0=max(S0 - h, 1e-12), K=K, T=T, r=r, q=q, option_type=option_type, lv_surface=lv_surface, cfg=cfg.pde_cfg, t0=t0
    )

    delta = (p_up - p_dn) / (2.0 * h)
    gamma = (p_up - 2.0 * p0 + p_dn) / (h * h)

    # Numerical stabilization: gamma should be non-negative for vanilla options.
    # Clip tiny negative values caused by cancellation.
    if gamma < 0.0 and abs(gamma) < 1e-8:
        gamma = 1e-12

    return float(delta), float(gamma)



def theta_local_vol_pde(
    *,
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: OptionType,
    lv_surface: LocalVolSurface,
    dt: float = 1.0 / 365.0,
    cfg: PDEConfig = PDEConfig(),
    t0: float = 0.0,
) -> float:
    """
    Theta ≈ (V(T - dt) - V(T)) / dt  (calendar-time theta; negative typically)
    """
    if T <= dt:
        return 0.0

    p_T = price_european_pde_local_vol(
        S0=S0, K=K, T=T, r=r, q=q, option_type=option_type, lv_surface=lv_surface, cfg=cfg, t0=t0,
    )
    p_Tm = price_european_pde_local_vol(
        S0=S0, K=K, T=T - dt, r=r, q=q, option_type=option_type, lv_surface=lv_surface, cfg=cfg, t0=t0,
    )
    theta = (p_Tm - p_T) / dt
    return float(theta)