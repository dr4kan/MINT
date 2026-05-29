#!/usr/bin/env python3
"""Command-line driver for training MINT on a saved muon dataset.

The input dataset must be a PyTorch ``.pt`` file produced by
:meth:`mint.MuonDataset.save` (or by any other code that writes the
same dictionary layout — see the README for the expected schema).

The output checkpoint contains the trained model parameters, the
training configuration, the per-iteration loss history, and the
reconstruction-volume bounds (so :command:`eval_mint.py` can rebuild
the field without any external metadata).

Example
-------

.. code-block:: bash

    python -m scripts.train_mint \\
        --dataset  mydata.pt \\
        --box-lo  -50 -50 -50 \\
        --box-hi   50  50  50 \\
        --device   cuda \\
        --n-iters  4000 \\
        --out      model.pt
"""

from __future__ import annotations

import argparse
import time

import torch

from mint import (
    LambdaField,
    MuonDataset,
    TrainConfig,
    VolumeBox,
    train,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # I/O
    ap.add_argument("--dataset", required=True, help="path to .pt muon dataset")
    ap.add_argument("--out", default="mint_model.pt", help="output checkpoint path")
    # reconstruction volume (cm)
    ap.add_argument("--box-lo", type=float, nargs=3, required=True,
                    metavar=("X", "Y", "Z"),
                    help="lower corner of the reconstruction box (cm)")
    ap.add_argument("--box-hi", type=float, nargs=3, required=True,
                    metavar=("X", "Y", "Z"),
                    help="upper corner of the reconstruction box (cm)")
    # optimisation
    ap.add_argument("--n-iters",     type=int,   default=4000)
    ap.add_argument("--batch-size",  type=int,   default=8192)
    ap.add_argument("--n-samples",   type=int,   default=96,
                    help="quadrature samples per track")
    ap.add_argument("--lr",          type=float, default=2e-3)
    ap.add_argument("--lr-warmup-iters", type=int,   default=200)
    ap.add_argument("--lr-min-ratio",    type=float, default=0.05)
    ap.add_argument("--grad-clip",   type=float, default=1.0)
    # regularisation
    ap.add_argument("--w-tv", type=float, default=2e-3,
                    help="total-variation regulariser weight")
    ap.add_argument("--w-bg", type=float, default=5e-3,
                    help="background prior weight (L2 toward --lambda-bg)")
    ap.add_argument("--lambda-bg", type=float, default=None,
                    help="background prior centre (cm^-1); default = LAMBDA_AIR")
    ap.add_argument("--tv-delta-cm", type=float, default=1.0,
                    help="finite-difference step for the TV regulariser (cm)")
    ap.add_argument("--n-reg-points", type=int, default=4096,
                    help="random points per call for regularisers")
    # network — hash-grid encoder
    ap.add_argument("--n-levels",           type=int,   default=16)
    ap.add_argument("--n-features-per-level", type=int, default=2)
    ap.add_argument("--log2-hashmap-size",  type=int,   default=19)
    ap.add_argument("--base-resolution",    type=int,   default=16)
    ap.add_argument("--per-level-scale",    type=float, default=1.5)
    ap.add_argument("--hidden-dim",         type=int,   default=128)
    ap.add_argument("--n-hidden-layers",    type=int,   default=1)
    # initial state
    ap.add_argument("--lambda-init", type=float, default=None,
                    help="initial value of lambda everywhere (cm^-1); "
                         "default = LAMBDA_AIR.  Set to lambda_concrete for "
                         "concrete-background scenes.")
    # measurement model
    ap.add_argument("--sigma-meas-angle", type=float, default=0.0)
    ap.add_argument("--sigma-meas-pos",   type=float, default=0.0)
    ap.add_argument("--use-log-term", action="store_true",
                    help="enable the Highland log correction in the NLL")
    # misc
    ap.add_argument("--device",    default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--seed",      type=int, default=0)
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()

    # device sanity checks
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available — falling back to CPU")
        args.device = "cpu"
    elif args.device == "mps":
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not has_mps:
            print("WARNING: MPS not available — falling back to CPU")
            args.device = "cpu"

    torch.manual_seed(args.seed)

    # dataset
    dataset = MuonDataset.load(args.dataset)
    n_tracks = len(dataset)
    print(f"loaded {n_tracks:,} tracks from {args.dataset}")

    # reconstruction volume
    box = VolumeBox(
        lo=torch.tensor(args.box_lo, dtype=torch.float32),
        hi=torch.tensor(args.box_hi, dtype=torch.float32),
    )
    print(f"box: {args.box_lo} -> {args.box_hi} cm  "
          f"(size {[f'{s:.1f}' for s in box.size.tolist()]} cm)")

    # field
    from mint.physics import LAMBDA_AIR
    lambda_init = args.lambda_init if args.lambda_init is not None else LAMBDA_AIR
    field_model = LambdaField(
        box=box,
        encoder="hash",
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden_layers,
        n_levels=args.n_levels,
        n_features_per_level=args.n_features_per_level,
        log2_hashmap_size=args.log2_hashmap_size,
        base_resolution=args.base_resolution,
        per_level_scale=args.per_level_scale,
        lambda_air=lambda_init,
    )
    n_params = sum(p.numel() for p in field_model.parameters())
    print(f"LambdaField: {n_params:,} trainable parameters")

    # train
    cfg = TrainConfig(
        n_iters=args.n_iters,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        lr=args.lr,
        lr_warmup_iters=args.lr_warmup_iters,
        lr_min_ratio=args.lr_min_ratio,
        grad_clip=args.grad_clip,
        w_tv=args.w_tv,
        w_bg=args.w_bg,
        lambda_bg=args.lambda_bg,
        n_reg_points=args.n_reg_points,
        tv_delta_cm=args.tv_delta_cm,
        sigma_meas_angle=args.sigma_meas_angle,
        sigma_meas_pos=args.sigma_meas_pos,
        use_log_term=args.use_log_term,
        log_every=args.log_every,
        device=args.device,
        seed=args.seed,
    )
    t0 = time.time()
    state = train(field_model, dataset, cfg)
    elapsed = time.time() - t0
    print(f"\ntrained {cfg.n_iters} iters in {elapsed:.1f} s "
          f"({elapsed / max(1, cfg.n_iters):.3f} s/iter)")

    # save
    torch.save(
        {
            "state_dict": field_model.state_dict(),
            "config":     vars(args),
            "losses":     state.losses,
            "box_lo":     box.lo.cpu(),
            "box_hi":     box.hi.cpu(),
        },
        args.out,
    )
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
