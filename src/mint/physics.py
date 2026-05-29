"""Multiple Coulomb Scattering (MCS) physics for MINT.

We use the Highland / PDG approximation for the projected RMS scattering
angle of a charged particle of momentum ``p`` (MeV/c) and velocity
``beta * c`` traversing a material of integrated inverse radiation length
``T = int lambda(s) ds`` along the track:

.. math::

   \\theta_0^2 = \\left(\\frac{13.6\\,\\mathrm{MeV}}{p\\beta}\\right)^2
                 T\\,[1 + 0.038\\,\\ln T]^2 ,

with ``lambda = 1/X_0``.  The logarithmic correction is optional and
disabled by default (it makes the per-track Jacobian noisier).

For each projection plane (``x`` or ``y``) the joint distribution of the
angular deflection ``Delta theta`` and the lateral offset ``Delta r`` at
the exit of a slab of thickness ``L`` is a zero-mean 2-D Gaussian with
covariance

.. math::

   \\mathrm{Var}(\\Delta\\theta) &= \\theta_0^2 \\\\
   \\mathrm{Var}(\\Delta r)      &= L^2\\,\\theta_0^2 / 3 \\\\
   \\mathrm{Cov}(\\theta, r)     &= L\\,\\theta_0^2 / 2 .

The two projections are statistically independent, so the joint
4-D covariance for ``y = (dtheta_x, dtheta_y, dx, dy)`` is a permutation
of two identical 2x2 blocks.
"""

from __future__ import annotations

import torch

HIGHLAND_C: float = 13.6           # MeV
HIGHLAND_LOG_COEF: float = 0.038
MUON_MASS_MEV: float = 105.6583755

# Reference air radiation length (dry air NTP, PDG).
X0_AIR_CM: float = 30420.0
LAMBDA_AIR: float = 1.0 / X0_AIR_CM


def beta_from_momentum(
    p_mev: torch.Tensor,
    mass_mev: float = MUON_MASS_MEV,
) -> torch.Tensor:
    """Relativistic ``beta`` from momentum (MeV/c)."""
    e = torch.sqrt(p_mev * p_mev + mass_mev * mass_mev)
    return p_mev / e


def theta0_squared(
    T: torch.Tensor,
    p_mev: torch.Tensor,
    mass_mev: float = MUON_MASS_MEV,
    use_log_term: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Highland projected variance ``theta_0^2`` in rad^2.

    Args:
        T:           integrated inverse radiation length along the track.
        p_mev:       muon momentum in MeV/c.
        mass_mev:    particle mass in MeV/c^2.  Defaults to the muon mass.
        use_log_term: include the ``[1 + 0.038 ln T]^2`` correction.
        eps:         lower clamp for ``T`` inside the logarithm.
    """
    beta = beta_from_momentum(p_mev, mass_mev)
    pbeta = p_mev * beta
    base = (HIGHLAND_C / pbeta).pow(2) * T
    if use_log_term:
        log_t = torch.log(T.clamp_min(eps))
        corr = (1.0 + HIGHLAND_LOG_COEF * log_t).pow(2)
        return base * corr
    return base


def mcs_covariance(
    T: torch.Tensor,
    L: torch.Tensor,
    p_mev: torch.Tensor,
    *,
    sigma_meas_angle: float = 0.0,
    sigma_meas_pos: float = 0.0,
    use_log_term: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Build the 4x4 covariance for ``y = (dtheta_x, dtheta_y, dx, dy)``.

    Args:
        T:                 (N,) integrated inverse radiation length.
        L:                 (N,) path length through the volume.
        p_mev:             (N,) muon momentum (MeV/c).
        sigma_meas_angle:  per-projection angular measurement noise (rad),
                           added in quadrature to the angular variance.
        sigma_meas_pos:    per-projection lateral measurement noise (cm),
                           added in quadrature to the positional variance.
        use_log_term:      include the logarithmic Highland correction.

    Returns:
        Sigma:  (N, 4, 4) tensor.

    Note:
        The full 4x4 matrix is rarely needed in practice — both the
        log-likelihood and its gradient can be evaluated analytically
        from ``theta_0^2`` directly.  See :func:`mint.losses.gaussian_nll`.
    """
    th2 = theta0_squared(T, p_mev, use_log_term=use_log_term, eps=eps)  # (N,)
    var_theta = th2 + sigma_meas_angle * sigma_meas_angle
    var_pos = L * L * th2 / 3.0 + sigma_meas_pos * sigma_meas_pos
    cov_tp = L * th2 / 2.0

    n = T.shape[0]
    Sigma = T.new_zeros(n, 4, 4)
    Sigma[:, 0, 0] = var_theta
    Sigma[:, 1, 1] = var_theta
    Sigma[:, 2, 2] = var_pos
    Sigma[:, 3, 3] = var_pos
    Sigma[:, 0, 2] = cov_tp
    Sigma[:, 2, 0] = cov_tp
    Sigma[:, 1, 3] = cov_tp
    Sigma[:, 3, 1] = cov_tp
    return Sigma


__all__ = [
    "HIGHLAND_C",
    "HIGHLAND_LOG_COEF",
    "MUON_MASS_MEV",
    "X0_AIR_CM",
    "LAMBDA_AIR",
    "beta_from_momentum",
    "theta0_squared",
    "mcs_covariance",
]
