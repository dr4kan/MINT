# Copyright (c) 2026 Davide Pagano
# University of Brescia (Italy)
# davide.pagano@unibs.it
#
# Part of MINT — Muon Implicit Neural Tomography.
# Distributed under the MIT License (see LICENSE in the project root).

"""MINT — Muon Implicit Neural Tomography.

A differentiable neural-field framework for cosmic-ray muon scattering
tomography.  The continuous inverse-radiation-length field
``lambda(r) = 1/X_0(r)`` is parameterised by a multi-resolution hash
encoder + small MLP decoder and fit by stochastic gradient descent on
the analytic Highland MCS log-likelihood.

The convenient top-level imports below cover the typical usage; see the
individual sub-modules for full APIs.
"""

from .data import MuonDataset, MuonTrack
from .field import LambdaField, VolumeBox
from .forward import integrate_lambda
from .losses import gaussian_nll, total_variation, background_prior
from .physics import (
    HIGHLAND_C,
    LAMBDA_AIR,
    MUON_MASS_MEV,
    X0_AIR_CM,
    mcs_covariance,
    theta0_squared,
)
from .train import TrainConfig, TrainState, evaluate_grid, train

__version__ = "0.1.0"

__all__ = [
    # data
    "MuonDataset", "MuonTrack",
    # field
    "LambdaField", "VolumeBox",
    # forward
    "integrate_lambda",
    # losses
    "gaussian_nll", "total_variation", "background_prior",
    # physics
    "HIGHLAND_C", "LAMBDA_AIR", "MUON_MASS_MEV", "X0_AIR_CM",
    "mcs_covariance", "theta0_squared",
    # training
    "TrainConfig", "TrainState", "evaluate_grid", "train",
]
