import numpy as np

from local_volatility.surface import LocalVolSurface


def test_local_vol_surface_domain_state_and_trust_helpers():
    S_grid = np.array([90.0, 100.0, 110.0], dtype=float)
    T_grid = np.array([0.1, 0.2], dtype=float)
    sigma_grid = np.full((T_grid.size, S_grid.size), 0.2, dtype=float)
    domain_mask = np.array(
        [
            [0, 1, 2],
            [0, 2, 2],
        ],
        dtype=np.int8,
    )

    lv = LocalVolSurface.from_dupire_grid(
        sigma_grid,
        S_grid,
        T_grid,
        domain_mask=domain_mask,
    )

    states = lv.domain_state(
        np.array([89.0, 100.0, 111.0, 109.9]),
        np.array([0.1, 0.1, 0.2, 0.21]),
    )
    assert states.tolist() == [0, 1, 0, 0]

    assert lv.is_trusted(np.array([110.0]), np.array([0.1])).tolist() == [True]
    assert lv.is_trusted(np.array([100.0]), np.array([0.1])).tolist() == [False]
    assert lv.is_supported(np.array([100.0]), np.array([0.1])).tolist() == [True]
