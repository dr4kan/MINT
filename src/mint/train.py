"""Training loop for the MINT neural field."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .data import MuonDataset
from .field import LambdaField
from .forward import integrate_lambda
from .losses import background_prior, gaussian_nll, total_variation


@dataclass
class TrainConfig:
    """Hyperparameters for :func:`train`."""

    # optimisation
    n_iters: int = 4000
    batch_size: int = 8192
    n_samples: int = 96
    lr: float = 2e-3
    weight_decay: float = 0.0
    # learning-rate schedule (linear warmup -> cosine decay)
    lr_warmup_iters: int = 200      # 0 disables warmup
    lr_min_ratio: float = 0.05      # 1.0 disables cosine decay
    grad_clip: float = 1.0
    # regularisation weights (J = NLL + w_tv * TV + w_bg * background_prior)
    w_tv: float = 2e-3
    w_bg: float = 5e-3
    n_reg_points: int = 4096
    tv_delta_cm: float = 1.0
    tv_edge_aware: bool = False
    tv_eps_grad: float = 1.0
    # background prior centre (default = LAMBDA_AIR, set explicitly for
    # non-air scenes such as concrete or water).  ``None`` falls back to
    # the field's own ``lambda_air`` attribute.
    lambda_bg: float | None = None
    # measurement noise (per-projection)
    sigma_meas_angle: float = 0.0
    sigma_meas_pos: float = 0.0
    # Highland logarithmic correction in NLL
    use_log_term: bool = False
    # misc
    log_every: int = 50
    device: str = "cpu"
    seed: int = 0


@dataclass
class TrainState:
    """Returned by :func:`train`; holds the per-iteration loss history."""

    losses: list[dict] = field(default_factory=list)


def _lr_schedule(it: int, cfg: TrainConfig) -> float:
    """Linear warmup over ``cfg.lr_warmup_iters`` then cosine decay to
    ``cfg.lr * cfg.lr_min_ratio``."""
    base = cfg.lr
    if cfg.lr_warmup_iters > 0 and it < cfg.lr_warmup_iters:
        return base * (it + 1) / cfg.lr_warmup_iters
    if cfg.lr_min_ratio < 1.0:
        progress = max(0.0, min(1.0,
            (it - cfg.lr_warmup_iters) / max(1, cfg.n_iters - cfg.lr_warmup_iters)))
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return base * (cfg.lr_min_ratio + (1.0 - cfg.lr_min_ratio) * cos)
    return base


def train(
    field_model: LambdaField,
    dataset: MuonDataset,
    cfg: TrainConfig,
    *,
    on_log=None,
) -> TrainState:
    """Train ``field_model`` on ``dataset`` according to ``cfg``.

    Optimises the objective

    .. math::

       \\mathcal{J}(\\theta) = -\\log\\mathcal{L}(\\theta)
                              + w_{\\mathrm{TV}}\\,\\mathcal{R}_{\\mathrm{TV}}
                              + w_{\\mathrm{bg}}\\,\\mathcal{R}_{\\mathrm{bg}}

    by Adam, with the stratified-jittered quadrature of
    :func:`mint.forward.integrate_lambda` used to evaluate ``T_i`` for
    every track in a mini-batch.  A linear warmup followed by cosine
    decay (see :func:`_lr_schedule`) is applied to the learning rate.

    Args:
        field_model: the :class:`~mint.field.LambdaField` to train.
                     Its parameters are modified in place.
        dataset:     :class:`~mint.data.MuonDataset` of training tracks.
        cfg:         hyperparameters; see :class:`TrainConfig`.
        on_log:      optional callable ``(iter, loss_terms_dict)`` used in
                     place of ``print`` when an iteration is logged.

    Returns:
        :class:`TrainState` with per-iteration loss terms.

    Notes:
        Iterations that produce a non-finite loss or gradient are
        skipped (the optimiser is not stepped and the parameters are
        not corrupted) and logged with ``loss = nan``.
    """
    device = torch.device(cfg.device)
    field_model.to(device)
    field_model.box = field_model.box.to(device)
    dataset = dataset.to(device)

    g = torch.Generator(device="cpu").manual_seed(cfg.seed)

    optim = torch.optim.Adam(
        field_model.parameters(),
        lr=cfg.lr, weight_decay=cfg.weight_decay, eps=1e-15,
    )

    lambda_bg = cfg.lambda_bg if cfg.lambda_bg is not None else field_model.lambda_air

    state = TrainState()
    n_total = len(dataset)

    for it in range(1, cfg.n_iters + 1):
        new_lr = _lr_schedule(it, cfg)
        for pg in optim.param_groups:
            pg["lr"] = new_lr

        idx = torch.randint(0, n_total, (cfg.batch_size,), generator=g)
        batch_entry = dataset.entry_point[idx]
        batch_exit  = dataset.exit_point[idx]
        batch_obs   = dataset.observations[idx]
        batch_p     = dataset.momentum_mev[idx]

        T, L = integrate_lambda(
            field_model, batch_entry, batch_exit,
            n_samples=cfg.n_samples, jitter=True,
        )

        nll = gaussian_nll(
            T, L, batch_p, batch_obs,
            sigma_meas_angle=cfg.sigma_meas_angle,
            sigma_meas_pos=cfg.sigma_meas_pos,
            use_log_term=cfg.use_log_term,
        )

        loss = nll
        loss_terms = {"nll": nll.detach().item()}

        if cfg.w_tv > 0:
            tv = total_variation(
                field_model,
                n_points=cfg.n_reg_points,
                delta_cm=cfg.tv_delta_cm,
                edge_aware=cfg.tv_edge_aware,
                eps_grad=cfg.tv_eps_grad,
            )
            loss = loss + cfg.w_tv * tv
            loss_terms["tv"] = tv.detach().item()

        if cfg.w_bg > 0:
            bg = background_prior(
                field_model, n_points=cfg.n_reg_points, lambda_bg=lambda_bg,
            )
            loss = loss + cfg.w_bg * bg
            loss_terms["bg"] = bg.detach().item()

        # guard against pathological iterations (non-finite loss/grad).
        if not torch.isfinite(loss):
            loss_terms.update({"loss": float("nan"), "iter": it})
            state.losses.append(loss_terms)
            optim.zero_grad(set_to_none=True)
            continue

        optim.zero_grad(set_to_none=True)
        loss.backward()

        bad_grad = any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in field_model.parameters()
        )
        if bad_grad:
            optim.zero_grad(set_to_none=True)
            loss_terms.update({"loss": float("nan"), "iter": it})
            state.losses.append(loss_terms)
            continue

        if cfg.grad_clip and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(field_model.parameters(), cfg.grad_clip)
        optim.step()

        loss_terms.update({"loss": loss.detach().item(), "iter": it})
        state.losses.append(loss_terms)
        if it % cfg.log_every == 0 or it == 1:
            if on_log is not None:
                on_log(it, loss_terms)
            else:
                msg = " ".join(
                    f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in loss_terms.items()
                )
                print(msg)

    return state


@torch.no_grad()
def evaluate_grid(
    field_model: LambdaField,
    *,
    nx: int = 64,
    ny: int = 64,
    nz: int = 64,
    chunk: int = 65536,
) -> torch.Tensor:
    """Evaluate ``field_model`` on a regular grid spanning its box.

    Returns:
        ``(nx, ny, nz)`` tensor of ``lambda`` values (cm^-1).
    """
    device = next(field_model.parameters()).device
    box = field_model.box
    xs = torch.linspace(box.lo[0].item(), box.hi[0].item(), nx, device=device)
    ys = torch.linspace(box.lo[1].item(), box.hi[1].item(), ny, device=device)
    zs = torch.linspace(box.lo[2].item(), box.hi[2].item(), nz, device=device)
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
    pts = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
    out = []
    for start in range(0, pts.shape[0], chunk):
        out.append(field_model(pts[start:start + chunk]))
    return torch.cat(out).reshape(nx, ny, nz)


__all__ = ["TrainConfig", "TrainState", "train", "evaluate_grid"]
