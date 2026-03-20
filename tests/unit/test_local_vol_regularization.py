import numpy as np

from local_volatility.regularization import (
    regularize_local_vol,
    LocalVolRegularizationConfig,
    repair_short_end_wings,
)


def test_regularization_clips_and_smooths():
    # Artificially ugly local vol surface
    sigma = np.array(
        [
            [0.01, 0.5, 10.0],
            [0.02, 0.6, 9.0],
            [0.03, 0.7, 8.0],
        ],
        dtype=float,
    )

    cfg = LocalVolRegularizationConfig(
        min_vol=0.1,
        max_vol=2.0,
        smooth=True,
        gaussian_sigma_T=1.0,
        gaussian_sigma_K=1.0,
        preserve_T0=False,
    )

    out = regularize_local_vol(sigma, cfg=cfg)

    # Shape preserved
    assert out.shape == sigma.shape

    # Clipping enforced
    assert np.all(out >= cfg.min_vol)
    assert np.all(out <= cfg.max_vol)

    # Smoothing should reduce extreme gradients
    assert np.std(out) < np.std(sigma)


def test_short_end_repair_reduces_floor_and_cap_blocks():
    sigma = np.array(
        [
            [0.0, 0.0, 0.0, 5.0, 5.0, 5.0],
            [0.0, 0.0, 0.10, 0.20, 0.25, 5.0],
            [0.12, 0.14, 0.18, 0.20, 0.22, 0.24],
        ],
        dtype=float,
    )

    out = repair_short_end_wings(
        sigma,
        min_vol=0.05,
        max_vol=2.0,
        min_coverage=0.50,
        valid_vol_min=0.02,
        valid_vol_max=1.00,
        anchor_blend=0.20,
    )

    assert out.shape == sigma.shape
    assert np.all(out >= 0.05)
    assert np.all(out <= 2.0)
    assert np.mean(out <= 0.05 + 1e-12) < np.mean(np.clip(sigma, 0.05, 2.0) <= 0.05 + 1e-12)
    assert np.mean(out >= 2.0 - 1e-12) < np.mean(np.clip(sigma, 0.05, 2.0) >= 2.0 - 1e-12)
