# Copyright (c) 2026 Davide Pagano
# University of Brescia (Italy)
# davide.pagano@unibs.it
#
# Part of MINT — Muon Implicit Neural Tomography.
# Distributed under the MIT License (see LICENSE in the project root).

"""Loss components for the MINT optimisation problem.

The 4x4 MCS covariance decomposes into two independent 2x2 blocks (one
per projection plane) of identical analytic form

.. math::

   M = \\begin{pmatrix} a & c \\\\ c & b \\end{pmatrix},\\quad
   a = \\theta_0^2,\\quad
   b = L^2 \\theta_0^2 / 3,\\quad
   c = L\\,\\theta_0^2 / 2.

So ``log det Sigma = 2 log(ab - c^2)`` and the quadratic form
factorises as

.. math::

   y^\\top \\Sigma^{-1} y = \\frac{1}{ab - c^2}
       \\sum_{k} \\bigl[ b\\,\\Delta\\theta_k^2
                       - 2c\\,\\Delta\\theta_k\\,\\Delta r_k
                       + a\\,\\Delta r_k^2 \\bigr] .

Everything is built directly from ``theta_0^2``, avoiding any explicit
4x4 inverse or determinant computation.
"""

from __future__ import annotations

import math

import torch

from .field import LambdaField
from .forward import random_points_in_box
from .physics import LAMBDA_AIR, theta0_squared

_LOG_2PI = math.log(2.0 * math.pi)


def gaussian_nll(
    T: torch.Tensor,                # (N,)
    L: torch.Tensor,                # (N,)
    p_mev: torch.Tensor,            # (N,)
    obs: torch.Tensor,              # (N, 4) = (dtheta_x, dtheta_y, dx, dy)
    *,
    sigma_meas_angle: float = 0.0,
    sigma_meas_pos: float = 0.0,
    use_log_term: bool = False,
    reduction: str = "mean",
    eps: float = 1e-30,
) -> torch.Tensor:
    """Per-track 4-D Gaussian NLL with the full block-diagonal covariance.

    Args:
        T:                 (N,) integrated inverse radiation length
                           along each track.
        L:                 (N,) path length through the volume (cm).
        p_mev:             (N,) muon momentum (MeV/c).
        obs:               (N, 4) per-track observation
                           ``(dtheta_x, dtheta_y, dx, dy)``.
        sigma_meas_angle:  per-projection angular measurement noise (rad).
        sigma_meas_pos:    per-projection lateral measurement noise (cm).
        use_log_term:      include the ``[1 + 0.038 ln T]^2`` correction.
        reduction:         ``"mean"`` (default), ``"sum"``, or ``"none"``.
        eps:               lower clamp for the per-block determinant.

    Returns:
        Scalar loss (``mean`` / ``sum``) or per-track NLL (``none``).
    """
    th2 = theta0_squared(T, p_mev, use_log_term=use_log_term)
    a = th2 + sigma_meas_angle * sigma_meas_angle
    b = L * L * th2 / 3.0 + sigma_meas_pos * sigma_meas_pos
    c = L * th2 / 2.0
    det = (a * b - c * c).clamp_min(eps)

    dtx, dty, dx, dy = obs[..., 0], obs[..., 1], obs[..., 2], obs[..., 3]
    qx = (b * dtx * dtx - 2.0 * c * dtx * dx + a * dx * dx) / det
    qy = (b * dty * dty - 2.0 * c * dty * dy + a * dy * dy) / det

    nll = 0.5 * (qx + qy) + torch.log(det).clamp(min=-50.0, max=50.0) + 2.0 * _LOG_2PI

    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    if reduction == "none":
        return nll
    raise ValueError(f"unknown reduction: {reduction!r}")


def total_variation(
    field: LambdaField,
    n_points: int = 1024,
    delta_cm: float = 1.0,
    *,
    generator: torch.Generator | None = None,
    edge_aware: bool = False,
    eps_grad: float = 1.0,
) -> torch.Tensor:
    """Stochastic total-variation regulariser.

    Averages the finite-difference gradient magnitude of ``lambda`` at
    random points inside the volume; one finite difference along each
    axis.  Cheap (a single forward pass on ``4N`` points).

    With ``edge_aware=True``, switches to a Perona-Malik / saturating form

    .. math::

       \\mathrm{TV}_{\\mathrm{edge}}
           = \\bigl\\langle
                 \\lVert\\nabla\\lambda\\rVert^2 /
                 (\\lVert\\nabla\\lambda\\rVert^2 + \\varepsilon_g^2)
             \\bigr\\rangle ,

    which smooths flat regions but saturates at sharp boundaries
    (does not penalise step transitions).  ``eps_grad`` should be
    smaller than the typical material-boundary gradient and larger
    than the reconstruction noise.

    Args:
        field:      the :class:`~mint.field.LambdaField` to regularise.
        n_points:   number of random base points per call.
        delta_cm:   finite-difference step (cm).
        generator:  optional CPU ``torch.Generator`` for reproducibility.
        edge_aware: switch to the saturating Perona-Malik form.
        eps_grad:   gradient scale for the edge-aware variant.
    """
    box = field.box
    base = random_points_in_box(n_points, box, dtype=torch.float32, generator=generator)
    base = base.to(next(field.parameters()).device)
    delta = float(delta_cm)
    e = torch.eye(3, device=base.device, dtype=base.dtype) * delta
    pts = torch.cat([base, base + e[0], base + e[1], base + e[2]], dim=0)
    lam = field(pts)
    n = base.shape[0]
    l0, lx, ly, lz = lam[:n], lam[n:2 * n], lam[2 * n:3 * n], lam[3 * n:]
    if edge_aware:
        gx = (lx - l0) / delta
        gy = (ly - l0) / delta
        gz = (lz - l0) / delta
        g2 = gx * gx + gy * gy + gz * gz
        return (g2 / (g2 + eps_grad * eps_grad)).mean()
    tv = ((lx - l0).abs() + (ly - l0).abs() + (lz - l0).abs()) / delta
    return tv.mean()


def background_prior(
    field: LambdaField,
    n_points: int = 1024,
    *,
    lambda_bg: float = LAMBDA_AIR,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Soft L2 prior pulling ``lambda`` toward ``lambda_bg`` at random points.

    For most MST scenes the background is air (``lambda_bg = LAMBDA_AIR``,
    the default).  For scenes inside a denser medium (e.g.\\ a concrete
    cask or a water pool) pass the corresponding ``lambda_bg`` value.

    This prior gently constrains regions that are not crossed by any
    muon track; it has no effect on regions where the data dominate
    the gradient.
    """
    pts = random_points_in_box(n_points, field.box, dtype=torch.float32, generator=generator)
    pts = pts.to(next(field.parameters()).device)
    lam = field(pts)
    return ((lam - lambda_bg) ** 2).mean()


# Backward-compatible alias (the paper calls this term ``R_air``).
air_prior = background_prior


__all__ = ["gaussian_nll", "total_variation", "background_prior", "air_prior"]
