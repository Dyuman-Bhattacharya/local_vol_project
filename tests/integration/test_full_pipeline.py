# tests/integration/test_full_pipeline.py
from __future__ import annotations

import numpy as np

from tests.fixtures.synthetic_data import BSSyntheticChainConfig, generate_bs_chain

from market_data.validators import validate_and_clean, ValidationConfig
from market_data.transforms import add_derived_columns, TransformConfig

from implied_volatility.iv_solver import implied_vol, IVSolverConfig
from surface_fitting.spline import fit_spline_surface
from implied_volatility.surface import call_price_grid_from_iv_surface

from arbitrage.projection import project_call_surface_static_arbitrage
from arbitrage.static_checks import check_static_arbitrage_calls
from local_volatility.dupire import dupire_local_vol_from_call_grid, DupireConfig
from local_volatility.regularization import regularize_local_vol, LocalVolRegularizationConfig
from local_volatility.surface import LocalVolSurface

from pricing.pde_solver import price_european_pde_local_vol, PDEConfig
from implied_volatility.black_scholes import bs_price


def test_full_pipeline_black_scholes_fixture():
    # --- Fixture: arbitrage-free BS chain
    cfg = BSSyntheticChainConfig(S0=100.0, r=0.02, q=0.0, sigma=0.25)
    df_raw = generate_bs_chain(cfg)

    # --- Stage 1: validate + transforms
    df_clean, report = validate_and_clean(df_raw, cfg=ValidationConfig())
    df = add_derived_columns(df_clean, cfg=TransformConfig(day_count="ACT/365"))

    # --- Stage 2: IV inversion (on calls)
    # (We generated mid prices from BS, so IV should be ~constant)
    iv_cfg = IVSolverConfig(method="hybrid", max_iter=100, tol=1e-12)
    df["iv"] = implied_vol(
        prices=df["mid"].to_numpy(),
        S=df["underlying_price"].to_numpy(),
        K=df["strike"].to_numpy(),
        T=df["time_to_expiry"].to_numpy(),
        r=float(cfg.r),
        q=float(cfg.q),
        option_type="call",
        cfg=iv_cfg,
    )

    assert np.isfinite(df["iv"]).all()

    # --- Stage 3: build slices in x-space, interpolate total variance directly
    T_list = np.sort(df["time_to_expiry"].unique())
    x_slices = []
    w_slices = []

    for T in T_list:
        sub = df[df["time_to_expiry"] == T].copy()
        # x = log(K/F)
        x = sub["log_moneyness"].to_numpy(dtype=float)
        iv = sub["iv"].to_numpy(dtype=float)
        w = (iv * iv) * float(T)
        # sort by x
        order = np.argsort(x)
        x_slices.append(x[order])
        w_slices.append(w[order])

    variance_surface = fit_spline_surface(
        x_slices=x_slices,
        w_slices=w_slices,
        T_slices=T_list,
        kind_x="pchip",
        kind_T="pchip",
        method_2d="linear",
    )

    # --- Stage 3.5: generate a call grid on (T,K), then project it to no-arbitrage
    K_grid = np.linspace(cfg.K_min, cfg.K_max, cfg.nK)
    T_grid = np.linspace(T_list.min(), T_list.max(), 25)

    C_raw = call_price_grid_from_iv_surface(
        variance_surface,
        S0=cfg.S0,
        K_grid=K_grid,
        T_grid=T_grid,
        r=cfg.r,
        q=cfg.q,
        coord="x_T",
    )
    C_fit = project_call_surface_static_arbitrage(
        C_raw,
        K_grid,
        T_grid,
        S0=cfg.S0,
        r=cfg.r,
        q=cfg.q,
        tol=1e-8,
    ).C_projected

    arb_rep = check_static_arbitrage_calls(
        C_fit, K_grid, T_grid,
        S0=cfg.S0, r=cfg.r, q=cfg.q, tol=1e-8
    )
    assert arb_rep.counts["n_fail"] == 0

    # --- Stage 4: Dupire local vol extraction + mild regularization
    dup_cfg = DupireConfig(floor_density=1e-10, cap_sigma=2.0)
    sig_loc = dupire_local_vol_from_call_grid(C_fit, K_grid, T_grid, r=cfg.r, q=cfg.q, cfg=dup_cfg)

    sig_loc = regularize_local_vol(
        sig_loc,
        cfg=LocalVolRegularizationConfig(
            smooth=False,  # keep this off for a strict test
            min_vol=1e-6,
            max_vol=3.0,
        ),
    )

    # check near-ATM local vol roughly matches true sigma (not in wings)
    k_atm = int(np.argmin(np.abs(K_grid - cfg.S0)))
    atm_slice = sig_loc[:, k_atm]
    assert np.all(np.isfinite(atm_slice))
    assert np.allclose(atm_slice, cfg.sigma, rtol=0.10, atol=0.01)

    lv_surface = LocalVolSurface.from_dupire_grid(sig_loc, K_grid, T_grid)

    # --- Stage 5: PDE pricing validation (vanilla repricing)
    # For BS, local vol should reduce to constant vol, so PDE call price ≈ BS call price
    K_test = 100.0
    T_test = 1.0

    p_pde = price_european_pde_local_vol(
        S0=cfg.S0,
        K=K_test,
        T=T_test,
        r=cfg.r,
        q=cfg.q,
        option_type="call",
        lv_surface=lv_surface,
        cfg=PDEConfig(n_space=350, n_time=200),
    )

    p_bs = float(bs_price(cfg.S0, K_test, T_test, cfg.r, cfg.q, cfg.sigma, "call"))

    assert abs(p_pde - p_bs) / max(p_bs, 1e-8) < 0.03
