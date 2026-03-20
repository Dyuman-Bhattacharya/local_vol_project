# src/hedging/transaction_costs.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


class TransactionCostError(RuntimeError):
    """Raised when transaction cost evaluation fails."""


@dataclass(frozen=True)
class TransactionCostModel:
    """
    Transaction cost model for trading delta shares.

    Supported:
      - proportional: cost = rate * |d_shares| * S
      - fixed: cost = fixed_cost per trade if |d_shares|>0
      - bid_ask: approximate by paying half-spread on notional: 0.5*spread*|d_shares|
                (where spread is in price units; user supplies spread as a fraction or absolute)
    """
    kind: Literal["none", "proportional", "fixed", "bid_ask"] = "none"

    # proportional
    rate: float = 0.0  # e.g. 0.001 for 10 bps

    # fixed
    fixed_cost: float = 0.0

    # bid-ask
    spread: float = 0.0
    spread_is_fraction: bool = True  # if True, spread * S is absolute spread

    def cost(self, d_shares: float, S: float) -> float:
        d_shares = float(d_shares)
        S = float(S)
        if S <= 0.0:
            raise TransactionCostError("Spot S must be positive")

        if self.kind == "none":
            return 0.0

        if self.kind == "proportional":
            return float(self.rate * abs(d_shares) * S)

        if self.kind == "fixed":
            return float(self.fixed_cost if abs(d_shares) > 0.0 else 0.0)

        if self.kind == "bid_ask":
            if self.spread_is_fraction:
                abs_spread = self.spread * S
            else:
                abs_spread = self.spread
            # pay half-spread on traded notional
            return float(0.5 * abs_spread * abs(d_shares))

        raise TransactionCostError(f"Unknown cost kind: {self.kind}")
