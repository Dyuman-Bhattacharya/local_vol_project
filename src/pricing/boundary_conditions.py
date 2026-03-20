# src/pricing/boundary_conditions.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class BoundaryConditions:
    """
    Boundary conditions for European options in S-space.

    We implement standard asymptotics for large/small S:
      Call:
        u(0,t) = 0
        u(S->inf,t) ~ S*e^{-q(T-t)} - K*e^{-r(T-t)}
      Put:
        u(0,t) ~ K*e^{-r(T-t)}
        u(S->inf,t) = 0
    """
    option_type: OptionType


def payoff(S: np.ndarray, K: float, option_type: OptionType) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    if option_type == "call":
        return np.maximum(S - K, 0.0)
    if option_type == "put":
        return np.maximum(K - S, 0.0)
    raise ValueError("option_type must be 'call' or 'put'")


def left_boundary_value(
    tau: float,
    K: float,
    r: float,
    q: float,
    option_type: OptionType,
) -> float:
    """
    Boundary at S=0, expressed in remaining time tau = T - t.
    """
    if option_type == "call":
        return 0.0
    if option_type == "put":
        return float(K * np.exp(-r * tau))
    raise ValueError("option_type must be 'call' or 'put'")


def right_boundary_value(
    S_max: float,
    tau: float,
    K: float,
    r: float,
    q: float,
    option_type: OptionType,
) -> float:
    """
    Boundary at large S = S_max, expressed in remaining time tau = T - t.
    """
    if option_type == "call":
        return float(S_max * np.exp(-q * tau) - K * np.exp(-r * tau))
    if option_type == "put":
        return 0.0
    raise ValueError("option_type must be 'call' or 'put'")
