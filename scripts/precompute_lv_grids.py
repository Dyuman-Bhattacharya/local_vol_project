# scripts/precompute_lv_grids.py
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from local_volatility.surface import LocalVolSurface
from pricing.pde_solver import PDEConfig, solve_european_pde_local_vol_surface

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class LVGridManifest:
    """
    Absolute-spot local-vol grid metadata.

    The precompute layer stores PDE values on explicit calendar-time, strike,
    and absolute-spot axes:

      V_grid[j, k, i] = V(t_j, K_k, S_i)
      D_grid[j, k, i] = dV/dS(t_j, K_k, S_i)

    This is the spot-consistent replacement for the old reference-strike /
    moneyness shortcut.
    """

    version: int
    option_type: OptionType
    T: float
    r: float
    q: float
    S_low: float
    S_high: float
    n_space: int
    n_time: int
    n_strikes: int
    S_file: str
    t_file: str
    K_file: str
    V_file: str
    D_file: str


def _write_manifest(calib_dir: Path, manifest: LVGridManifest) -> None:
    path = calib_dir / "lv_grid_manifest.json"
    path.write_text(json.dumps(asdict(manifest), indent=2))


def _load_manifest(calib_dir: Path) -> Optional[dict]:
    path = calib_dir / "lv_grid_manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def build_lv_grids(
    *,
    calib_dir: Path,
    T: float,
    option_type: OptionType,
    K_grid: np.ndarray,
    S_low: float,
    S_high: float,
    n_space: int,
    n_time: int,
    r: float = 0.0,
    q: float = 0.0,
) -> LVGridManifest:
    """
    Precompute LV PDE price and delta grids on explicit (t, K, S) axes.
    """

    calib_dir = Path(calib_dir)
    calib_dir.mkdir(parents=True, exist_ok=True)

    lv_pkl = calib_dir / "local_vol_surface.pkl"
    if not lv_pkl.exists():
        raise RuntimeError(
            f"Missing {lv_pkl}. "
            "Run Notebooks 1-3 to generate the Dupire local-vol surface."
        )

    with open(lv_pkl, "rb") as f:
        lv_surface = pickle.load(f)

    if not isinstance(lv_surface, LocalVolSurface):
        raise RuntimeError(f"{lv_pkl} does not contain a LocalVolSurface")

    K_grid = np.asarray(K_grid, dtype=float)
    if K_grid.ndim != 1 or K_grid.size == 0:
        raise ValueError("K_grid must be a non-empty 1D array")
    if not np.all(np.diff(K_grid) > 0.0):
        raise ValueError("K_grid must be strictly increasing")
    if np.any(K_grid <= 0.0):
        raise ValueError("K_grid must be strictly positive")
    if S_low <= 0.0 or S_high <= S_low:
        raise ValueError("Require 0 < S_low < S_high")

    lv_low = float(lv_surface.S_grid[0])
    lv_high = float(lv_surface.S_grid[-1])
    if S_low < lv_low or S_high > lv_high:
        raise ValueError(
            "Requested spot grid falls outside the calibrated local-vol domain. "
            f"Requested [{S_low:.6f}, {S_high:.6f}], calibrated [{lv_low:.6f}, {lv_high:.6f}]"
        )
    if float(np.min(K_grid)) < lv_low or float(np.max(K_grid)) > lv_high:
        raise ValueError(
            "Requested strike grid falls outside the calibrated local-vol domain. "
            f"Strike range [{np.min(K_grid):.6f}, {np.max(K_grid):.6f}], "
            f"calibrated [{lv_low:.6f}, {lv_high:.6f}]"
        )

    pde_cfg = PDEConfig(n_space=int(n_space), n_time=int(n_time))

    print(">>> Building spot-consistent LV grids (this may take several minutes)...", flush=True)
    print(f"    strikes: {K_grid.size}, spot range: [{S_low:.4f}, {S_high:.4f}]", flush=True)

    t_grid = None
    S_grid = None
    V_grid = None
    D_grid = None

    for k_idx, K in enumerate(K_grid):
        print(f"LV strike surface progress: {k_idx + 1}/{K_grid.size} (K={K:.6f})", flush=True)

        t_surface, S_surface, V_surface = solve_european_pde_local_vol_surface(
            S_ref=float(K),
            K=float(K),
            T=float(T),
            r=float(r),
            q=float(q),
            option_type=str(option_type),
            lv_surface=lv_surface,
            cfg=pde_cfg,
            t0=0.0,
            S_min=float(S_low),
            S_max=float(S_high),
        )

        D_surface = np.gradient(V_surface, S_surface, axis=1, edge_order=2)

        if t_grid is None:
            t_grid = t_surface
            S_grid = S_surface
            V_grid = np.empty((t_grid.size, K_grid.size, S_grid.size), dtype=float)
            D_grid = np.empty_like(V_grid)
        else:
            if not np.allclose(t_surface, t_grid):
                raise RuntimeError("Inconsistent t_grid returned across strike surfaces")
            if not np.allclose(S_surface, S_grid):
                raise RuntimeError("Inconsistent S_grid returned across strike surfaces")

        V_grid[:, k_idx, :] = V_surface
        D_grid[:, k_idx, :] = D_surface

    np.save(calib_dir / "t_grid.npy", t_grid)
    np.save(calib_dir / "S_grid.npy", S_grid)
    np.save(calib_dir / "K_grid.npy", K_grid)
    np.save(calib_dir / "V_grid.npy", V_grid)
    np.save(calib_dir / "D_grid.npy", D_grid)

    manifest = LVGridManifest(
        version=2,
        option_type=str(option_type),
        T=float(T),
        r=float(r),
        q=float(q),
        S_low=float(S_low),
        S_high=float(S_high),
        n_space=int(n_space),
        n_time=int(n_time),
        n_strikes=int(K_grid.size),
        S_file="S_grid.npy",
        t_file="t_grid.npy",
        K_file="K_grid.npy",
        V_file="V_grid.npy",
        D_file="D_grid.npy",
    )
    _write_manifest(calib_dir, manifest)
    return manifest


def main():
    ap = argparse.ArgumentParser(description="Precompute LV PDE price+delta grids on explicit (t, K, S) axes.")
    ap.add_argument("--calib_dir", required=True)
    ap.add_argument("--T", type=float, required=True, help="Option maturity in years (e.g. 0.25)")
    ap.add_argument("--option_type", choices=["call", "put"], default="call")
    ap.add_argument("--K_min", type=float, required=True)
    ap.add_argument("--K_max", type=float, required=True)
    ap.add_argument("--n_strikes", type=int, default=41)
    ap.add_argument("--S_low", type=float, required=True)
    ap.add_argument("--S_high", type=float, required=True)
    ap.add_argument("--n_space", type=int, default=120)
    ap.add_argument("--n_time", type=int, default=100)
    ap.add_argument("--r", type=float, default=0.0)
    ap.add_argument("--q", type=float, default=0.0)
    args = ap.parse_args()

    K_grid = np.linspace(float(args.K_min), float(args.K_max), int(args.n_strikes))

    manifest = build_lv_grids(
        calib_dir=Path(args.calib_dir),
        T=float(args.T),
        option_type=str(args.option_type),
        K_grid=K_grid,
        S_low=float(args.S_low),
        S_high=float(args.S_high),
        n_space=int(args.n_space),
        n_time=int(args.n_time),
        r=float(args.r),
        q=float(args.q),
    )

    print("Precompute complete.")
    print("Saved:", Path(args.calib_dir))
    print("Manifest:", (Path(args.calib_dir) / "lv_grid_manifest.json").as_posix())
    print("n_strikes:", manifest.n_strikes, "spot range:", (manifest.S_low, manifest.S_high))


if __name__ == "__main__":
    main()
