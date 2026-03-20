# src/local_volatility/regularization.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from utils.platform_compat import apply_windows_platform_fastpath

apply_windows_platform_fastpath()
from scipy.ndimage import gaussian_filter

from utils.numerical import require_strictly_increasing


class RegularizationError(RuntimeError):
    """Raised when regularization cannot be applied."""


@dataclass(frozen=True)
class LocalVolRegularizationConfig:
    """
    Regularization strategy for local vol surfaces computed on a (T,K) grid.

    Methods implemented:
      - density_floor: floor sigma^2 denominator effect already handled upstream in DupireConfig
      - clip: enforce [min_vol, max_vol]
      - smooth: gaussian smoothing in (T,K) index space (not metric-correct but effective)

    gaussian_sigma_T / gaussian_sigma_K are in *grid index units*, not year/strike units.
    """
    min_vol: float = 1e-4
    max_vol: float = 5.0
    repair_short_end: bool = False
    short_end_min_coverage: float = 0.50
    short_end_valid_vol_min: float = 0.02
    short_end_valid_vol_max: float = 1.00
    short_end_anchor_blend: float = 0.20
    smooth: bool = True
    gaussian_sigma_T: float = 1.0
    gaussian_sigma_K: float = 1.0
    preserve_T0: bool = False  # if True, don't smooth the first maturity row


def clip_local_vol(sigma_loc: np.ndarray, *, min_vol: float, max_vol: float) -> np.ndarray:
    sigma_loc = np.asarray(sigma_loc, dtype=float)
    return np.clip(sigma_loc, min_vol, max_vol)


def repair_short_end_wings(
    sigma_loc: np.ndarray,
    *,
    min_vol: float,
    max_vol: float,
    min_coverage: float,
    valid_vol_min: float,
    valid_vol_max: float,
    anchor_blend: float,
) -> np.ndarray:
    """
    Repair very short maturities whose Dupire output is dominated by floor/cap artifacts.

    Strategy:
      - find the first maturity row with enough interior "reasonable" values;
      - blend earlier rows toward that anchor row;
      - flatten the unstable far wings to the nearest interior anchor value.

    This keeps the short end usable for PDE queries without leaving the surface full
    of hard floor/cap blocks.
    """
    s = np.asarray(sigma_loc, dtype=float).copy()
    s = np.clip(s, min_vol, max_vol)
    if s.ndim != 2:
        raise RegularizationError("sigma_loc must be 2D (nT,nK)")

    reasonable = (sigma_loc > valid_vol_min) & (sigma_loc < valid_vol_max)
    coverage = np.mean(reasonable, axis=1)
    stable_rows = np.where(coverage >= float(min_coverage))[0]
    if stable_rows.size == 0:
        return s

    anchor = int(stable_rows[0])
    if anchor <= 0:
        return s

    anchor_good = np.where(reasonable[anchor])[0]
    if anchor_good.size == 0:
        return s

    left = int(anchor_good[0])
    right = int(anchor_good[-1])
    blend0 = float(np.clip(anchor_blend, 0.0, 1.0))

    for j in range(anchor):
        w = (j + 1) / (anchor + 1)
        blend = blend0 + (1.0 - blend0) * w
        s[j, :] = blend * s[anchor, :] + (1.0 - blend) * s[j, :]
        s[j, :left] = s[j, left]
        s[j, right + 1 :] = s[j, right]

    return np.clip(s, min_vol, max_vol)


def smooth_local_vol_gaussian(
    sigma_loc: np.ndarray,
    *,
    sigma_T: float,
    sigma_K: float,
    preserve_first_row: bool = False,
) -> np.ndarray:
    """
    Gaussian smoothing over the grid indices.
    """
    s = np.asarray(sigma_loc, dtype=float)
    if s.ndim != 2:
        raise RegularizationError("sigma_loc must be 2D (nT,nK)")

    if preserve_first_row:
        first = s[0:1, :].copy()
        rest = gaussian_filter(s[1:, :], sigma=(sigma_T, sigma_K), mode="nearest")
        out = np.vstack([first, rest])
        return out

    return gaussian_filter(s, sigma=(sigma_T, sigma_K), mode="nearest")


def regularize_local_vol(
    sigma_loc: np.ndarray,
    *,
    cfg: LocalVolRegularizationConfig = LocalVolRegularizationConfig(),
) -> np.ndarray:
    """
    Apply regularization pipeline: smoothing (optional) then clipping.
    """
    s = np.asarray(sigma_loc, dtype=float)

    if cfg.repair_short_end:
        s = repair_short_end_wings(
            s,
            min_vol=cfg.min_vol,
            max_vol=cfg.max_vol,
            min_coverage=cfg.short_end_min_coverage,
            valid_vol_min=cfg.short_end_valid_vol_min,
            valid_vol_max=cfg.short_end_valid_vol_max,
            anchor_blend=cfg.short_end_anchor_blend,
        )

    if cfg.smooth:
        s = smooth_local_vol_gaussian(
            s,
            sigma_T=cfg.gaussian_sigma_T,
            sigma_K=cfg.gaussian_sigma_K,
            preserve_first_row=cfg.preserve_T0,
        )

    s = clip_local_vol(s, min_vol=cfg.min_vol, max_vol=cfg.max_vol)
    return s
