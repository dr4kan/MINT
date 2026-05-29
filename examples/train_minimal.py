# Copyright (c) 2026 Davide Pagano
# University of Brescia (Italy)
# davide.pagano@unibs.it
#
# Part of MINT — Muon Implicit Neural Tomography.
# Distributed under the MIT License (see LICENSE in the project root).

"""Minimal end-to-end example of MINT.

Generates a tiny synthetic dataset of muons traversing an axis-aligned
box with a single dense cubic inclusion in its centre, trains MINT for
a few hundred iterations, and prints the reconstructed lambda along a
diagonal cut.

The synthetic data are built *inline*: no MINT module is involved in
generating them, only in fitting them.  This file is therefore a
working self-contained smoke test.
"""

from __future__ import annotations

import math

import torch

from mint import (
    LambdaField,
    MuonDataset,
    TrainConfig,
    VolumeBox,
    evaluate_grid,
    theta0_squared,
    train,
)

# -----------------------------------------------------------------------------
# 1. A toy ground-truth scene: a 10x10x10 cm air box with a 4x4x4 cm
#    iron-like inclusion at the centre.
# -----------------------------------------------------------------------------
BOX_HALF = 5.0          # cm
CUBE_HALF = 2.0         # cm
LAM_AIR  = 3.29e-5      # cm^-1
LAM_CUBE = 0.57         # cm^-1 (~ iron)


def lambda_truth(pts: torch.Tensor) -> torch.Tensor:
    """Ground-truth lambda at ``pts`` (cm)."""
    inside_cube = (pts.abs() <= CUBE_HALF).all(dim=-1)
    return torch.where(inside_cube,
                       torch.tensor(LAM_CUBE),
                       torch.tensor(LAM_AIR))


def integrate_truth(entry: torch.Tensor, exit_: torch.Tensor, n_quad: int = 256) -> torch.Tensor:
    """Closed-form ground-truth T for each ray (midpoint quadrature)."""
    t = (torch.arange(n_quad, dtype=entry.dtype) + 0.5) / n_quad   # (Q,)
    pts = entry.unsqueeze(1) + t.view(1, n_quad, 1) * (exit_ - entry).unsqueeze(1)
    lam = lambda_truth(pts.reshape(-1, 3)).reshape(pts.shape[0], n_quad)
    L = torch.linalg.norm(exit_ - entry, dim=-1)
    return lam.mean(dim=1) * L


# -----------------------------------------------------------------------------
# 2. Synthetic muon dataset (downgoing rays, p = 3 GeV/c, no measurement noise).
# -----------------------------------------------------------------------------
def synthesise_dataset(n: int = 20_000, seed: int = 0) -> MuonDataset:
    g = torch.Generator().manual_seed(seed)
    # entry on the top face (z = +BOX_HALF), exit on the bottom face (z = -BOX_HALF)
    x = (torch.rand(n, generator=g) * 2 - 1) * BOX_HALF
    y = (torch.rand(n, generator=g) * 2 - 1) * BOX_HALF
    entry = torch.stack([x, y, torch.full_like(x, +BOX_HALF)], dim=-1)
    # straight, vertical rays (uz < 0 means downward; but our convention is
    # uz > 0 for the angular projections, so we flip the z axis to keep things
    # simple in this toy example).
    entry_dir = torch.tensor([[0.0, 0.0, -1.0]]).expand(n, 3).contiguous()
    exit_ = entry + entry_dir * 2 * BOX_HALF
    exit_dir = entry_dir.clone()      # straight (we add MCS noise to obs below)
    p = torch.full((n,), 3000.0)      # 3 GeV/c

    # Build ground-truth T, then draw the four observables from the analytic
    # MCS Gaussian.  We use the analytic Cholesky of each 2x2 block directly.
    T = integrate_truth(entry, exit_)
    L = torch.linalg.norm(exit_ - entry, dim=-1)
    th2 = theta0_squared(T, p)
    var_theta = th2
    var_pos   = L * L * th2 / 3.0
    cov_tp    = L * th2 / 2.0
    sa = torch.sqrt(var_theta.clamp_min(1e-30))
    sb = torch.sqrt((var_pos - cov_tp * cov_tp / var_theta).clamp_min(1e-30))
    z = torch.randn(n, 4, generator=g)
    dtx = sa * z[:, 0]
    dx  = (cov_tp / sa) * z[:, 0] + sb * z[:, 1]
    dty = sa * z[:, 2]
    dy  = (cov_tp / sa) * z[:, 2] + sb * z[:, 3]

    # MuonDataset infers (dtx, dty, dx, dy) from the entry/exit directions and
    # points.  Here we cheat slightly and shift the exit point + direction so
    # that the inferred observation matches our analytic draw.
    # angular: flip the entry direction by atan2 so the inferred dtx/dty match
    tx_in = entry_dir[:, 0] / entry_dir[:, 2].clamp_min(1e-6)
    ty_in = entry_dir[:, 1] / entry_dir[:, 2].clamp_min(1e-6)
    # we are in the convention uz_in = -1, so atan2(ux, uz) for our convention
    # below uses the same uz_out sign.
    tx_out = torch.tan(torch.atan(tx_in) + dtx)
    ty_out = torch.tan(torch.atan(ty_in) + dty)
    norm = torch.sqrt(1.0 + tx_out**2 + ty_out**2)
    exit_dir = torch.stack([tx_out / norm, ty_out / norm,
                            -1.0 / norm], dim=-1)
    exit_ = exit_.clone()
    exit_[:, 0] += dx
    exit_[:, 1] += dy

    # MINT's MuonDataset expects forward-going tracks (uz_in > 0); for this
    # toy demo we flip the convention so uz becomes positive by simply
    # negating z everywhere.
    entry[:, 2] *= -1
    exit_[:, 2] *= -1
    entry_dir = entry_dir.clone(); entry_dir[:, 2] *= -1
    exit_dir  = exit_dir.clone();  exit_dir[:, 2]  *= -1

    return MuonDataset(entry, entry_dir, exit_, exit_dir, p)


# -----------------------------------------------------------------------------
# 3. Train MINT on the synthetic data.
# -----------------------------------------------------------------------------
def main() -> int:
    dataset = synthesise_dataset(n=20_000, seed=0)
    print(f"synthetic dataset: {len(dataset)} tracks")

    box = VolumeBox(
        lo=torch.tensor([-BOX_HALF, -BOX_HALF, -BOX_HALF], dtype=torch.float32),
        hi=torch.tensor([+BOX_HALF, +BOX_HALF, +BOX_HALF], dtype=torch.float32),
    )
    field = LambdaField(
        box=box,
        encoder="hash",
        n_levels=12, log2_hashmap_size=16,
        base_resolution=8, per_level_scale=1.5,
        hidden_dim=64,
        lambda_air=LAM_AIR,
    )
    print(f"LambdaField: {sum(p.numel() for p in field.parameters()):,} parameters")

    cfg = TrainConfig(
        n_iters=500, batch_size=2048, n_samples=48,
        lr=2e-2, lr_warmup_iters=50, lr_min_ratio=0.1,
        w_tv=1e-3, w_bg=1e-3,
        n_reg_points=1024,
        log_every=50, device="cpu", seed=0,
    )
    train(field, dataset, cfg)

    # diagonal cut through the cube centre
    print("\nReconstructed lambda along the x-axis (y = z = 0):")
    cut = torch.stack([
        torch.linspace(-BOX_HALF, +BOX_HALF, 11),
        torch.zeros(11),
        torch.zeros(11),
    ], dim=-1)
    field.eval()
    with torch.no_grad():
        lam = field(cut)
    for x, v in zip(cut[:, 0].tolist(), lam.tolist()):
        print(f"  x = {x:+5.2f} cm  ->  lambda = {v:.4f} cm^-1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
