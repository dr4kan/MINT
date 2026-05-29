# Copyright (c) 2026 Davide Pagano
# University of Brescia (Italy)
# davide.pagano@unibs.it
#
# Part of MINT — Muon Implicit Neural Tomography.
# Distributed under the MIT License (see LICENSE in the project root).

"""Differentiable line integral of ``lambda`` along straight muon paths.

For each track the path through the volume is approximated by the
straight segment from the entry to the exit point and the line integral

.. math::

   T_i = \\int_0^{L_i} \\lambda_\\theta(\\mathbf{r}_i(s))\\,\\mathrm{d}s

is estimated by stratified jittered Monte Carlo quadrature with ``N``
samples (NeRF-style).  The estimator is unbiased and the gradient
flows cleanly through every quadrature sample into both the hash
embeddings and the MLP decoder.
"""

from __future__ import annotations

import torch

from .field import LambdaField, VolumeBox


def stratified_samples(
    n_rays: int,
    n_samples: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
    jitter: bool = True,
) -> torch.Tensor:
    """Return ``(n_rays, n_samples)`` quadrature points in ``(0, 1)``.

    The unit interval is partitioned into ``n_samples`` equal strata;
    one sample is drawn inside each stratum.  When ``jitter=True``,
    samples are drawn uniformly inside their stratum (recommended,
    breaks aliasing with the hash grid); when ``jitter=False``, samples
    are at the stratum midpoints (deterministic midpoint rule).
    """
    edges = torch.linspace(0.0, 1.0, n_samples + 1, device=device, dtype=dtype)
    centers = 0.5 * (edges[:-1] + edges[1:]).expand(n_rays, n_samples).contiguous()
    if jitter:
        widths = 1.0 / n_samples
        u = torch.rand(n_rays, n_samples, device=device, dtype=dtype, generator=generator)
        centers = centers + (u - 0.5) * widths
    return centers


def integrate_lambda(
    field: LambdaField,
    entry: torch.Tensor,        # (N, 3)
    exit_: torch.Tensor,        # (N, 3)
    *,
    n_samples: int = 96,
    jitter: bool = True,
    chunk: int | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the per-track ``T_i`` and path length ``L_i``.

    Args:
        field:     the :class:`~mint.field.LambdaField` being optimised.
        entry:     ``(N, 3)`` entry points (cm).
        exit_:     ``(N, 3)`` exit points (cm).
        n_samples: number of quadrature samples per track.
        jitter:    use jittered (random) strata; recommended.
        chunk:     if set, integrate in sub-batches of at most ``chunk``
                   rays to bound peak memory.
        generator: optional CPU ``torch.Generator`` for reproducibility.

    Returns:
        ``(T, L)``:
            - ``T`` ``(N,)`` estimated line integral of ``lambda``.
            - ``L`` ``(N,)`` path length through the volume (cm).
    """
    assert entry.shape == exit_.shape and entry.shape[-1] == 3
    delta = exit_ - entry
    L = torch.linalg.norm(delta, dim=-1)  # (N,)

    n_rays = entry.shape[0]
    if chunk is None or chunk >= n_rays:
        return _integrate_block(field, entry, exit_, L, n_samples, jitter, generator), L

    Ts = []
    for start in range(0, n_rays, chunk):
        end = min(start + chunk, n_rays)
        Ts.append(
            _integrate_block(
                field,
                entry[start:end],
                exit_[start:end],
                L[start:end],
                n_samples,
                jitter,
                generator,
            )
        )
    return torch.cat(Ts, dim=0), L


def _integrate_block(
    field: LambdaField,
    entry: torch.Tensor,
    exit_: torch.Tensor,
    L: torch.Tensor,
    n_samples: int,
    jitter: bool,
    generator: torch.Generator | None,
) -> torch.Tensor:
    n_rays = entry.shape[0]
    device, dtype = entry.device, entry.dtype
    t = stratified_samples(
        n_rays, n_samples, device=device, dtype=dtype,
        generator=generator, jitter=jitter,
    )  # (N, S) in (0, 1)
    direction = (exit_ - entry).unsqueeze(1)              # (N, 1, 3)
    pts = entry.unsqueeze(1) + t.unsqueeze(-1) * direction  # (N, S, 3)
    lam_flat = field(pts.reshape(-1, 3)).reshape(n_rays, n_samples)
    # midpoint-rule quadrature: each stratum has width L / S
    return lam_flat.mean(dim=1) * L


def random_points_in_box(
    n: int,
    box: VolumeBox,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample ``n`` points uniformly inside ``box``.

    When a ``generator`` is given on a device different from the target
    device (typically CPU generator + MPS / CUDA target), samples are
    drawn on the generator's device first and then moved.  This avoids
    the ``"Expected a 'mps' device type for generator"`` error.
    """
    target = torch.device(device) if device is not None else box.lo.device
    if generator is not None and generator.device != target:
        u = torch.rand(n, 3, dtype=dtype, generator=generator)
        u = u.to(device=target)
    else:
        u = torch.rand(n, 3, device=target, dtype=dtype, generator=generator)
    return box.lo.to(device=target, dtype=dtype) + u * box.size.to(device=target, dtype=dtype)


__all__ = ["stratified_samples", "integrate_lambda", "random_points_in_box"]
