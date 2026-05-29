#!/usr/bin/env python3
# Copyright (c) 2026 Davide Pagano
# University of Brescia (Italy)
# davide.pagano@unibs.it
#
# Part of MINT — Muon Implicit Neural Tomography.
# Distributed under the MIT License (see LICENSE in the project root).

"""Evaluate a trained MINT model on a regular 3-D grid.

Reads a checkpoint produced by :command:`train_mint.py`, rebuilds the
:class:`mint.LambdaField`, and writes the evaluated grid to a PyTorch
``.pt`` file containing:

    * ``"lambda"``  — ``(nx, ny, nz)`` tensor of ``lambda`` values (cm^-1);
    * ``"axes"``    — dict with the three axis vectors (cm);
    * ``"box_lo"``  / ``"box_hi"``  — reconstruction-volume corners.

Example
-------

.. code-block:: bash

    python -m scripts.eval_mint \\
        --model    model.pt \\
        --grid     128 128 128 \\
        --out      reconstruction.pt
"""

from __future__ import annotations

import argparse

import torch

from mint import LambdaField, VolumeBox, evaluate_grid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="checkpoint from train_mint.py")
    ap.add_argument("--out",   default="mint_grid.pt",
                    help="output .pt file with the evaluated grid")
    ap.add_argument("--grid",  type=int, nargs=3, default=(128, 128, 128),
                    metavar=("NX", "NY", "NZ"),
                    help="grid resolution per axis")
    ap.add_argument("--chunk", type=int, default=65536,
                    help="forward-pass chunk size (points per batch)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = ap.parse_args()

    ckpt = torch.load(args.model, weights_only=False, map_location="cpu")
    cfg = ckpt.get("config", {})

    box = VolumeBox(
        lo=ckpt["box_lo"].clone().float(),
        hi=ckpt["box_hi"].clone().float(),
    )

    from mint.physics import LAMBDA_AIR
    lambda_init = cfg.get("lambda_init") or LAMBDA_AIR
    field_model = LambdaField(
        box=box,
        encoder="hash",
        hidden_dim=cfg.get("hidden_dim", 128),
        n_hidden_layers=cfg.get("n_hidden_layers", 1),
        n_levels=cfg.get("n_levels", 16),
        n_features_per_level=cfg.get("n_features_per_level", 2),
        log2_hashmap_size=cfg.get("log2_hashmap_size", 19),
        base_resolution=cfg.get("base_resolution", 16),
        per_level_scale=cfg.get("per_level_scale", 1.5),
        lambda_air=lambda_init,
    )
    field_model.load_state_dict(ckpt["state_dict"])
    field_model.to(args.device)
    field_model.box = field_model.box.to(args.device)

    nx, ny, nz = args.grid
    grid = evaluate_grid(field_model, nx=nx, ny=ny, nz=nz, chunk=args.chunk)
    grid = grid.cpu()

    xs = torch.linspace(box.lo[0].item(), box.hi[0].item(), nx)
    ys = torch.linspace(box.lo[1].item(), box.hi[1].item(), ny)
    zs = torch.linspace(box.lo[2].item(), box.hi[2].item(), nz)

    torch.save(
        {
            "lambda": grid,
            "axes":   {"x": xs, "y": ys, "z": zs},
            "box_lo": box.lo.cpu(),
            "box_hi": box.hi.cpu(),
        },
        args.out,
    )
    print(f"evaluated lambda on a {nx}x{ny}x{nz} grid")
    print(f"  shape:  {tuple(grid.shape)}")
    print(f"  range:  [{grid.min().item():.3e}, {grid.max().item():.3e}] cm^-1")
    print(f"  saved:  {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
