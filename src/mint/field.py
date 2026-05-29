"""Neural field for the inverse radiation length.

The MINT field represents the inverse radiation length

.. math::

   \\lambda_\\theta(\\mathbf{r}) =
       \\mathrm{softplus}\\bigl(g_\\theta \\circ h_\\theta(\\mathbf{r})\\bigr)

as the composition of a multi-resolution hash encoder ``h_theta`` and a
small MLP decoder ``g_theta``, with softplus enforcing positivity.

The output bias of the last linear layer is initialised so that
``lambda`` at start-up approximately matches the inverse radiation
length of dry air (~3.3e-5 cm^-1), which keeps the per-track integrated
scatter ``T_i`` away from zero during the first training iterations and
prevents divergence of the Highland NLL.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .encoding import FrequencyEncoding, HashGridEncoding
from .physics import LAMBDA_AIR


@dataclass
class VolumeBox:
    """Axis-aligned reconstruction volume (cm)."""

    lo: torch.Tensor   # (3,)
    hi: torch.Tensor   # (3,)

    def to(self, device, dtype=None) -> "VolumeBox":
        return VolumeBox(
            lo=self.lo.to(device=device, dtype=dtype) if dtype else self.lo.to(device),
            hi=self.hi.to(device=device, dtype=dtype) if dtype else self.hi.to(device),
        )

    @property
    def size(self) -> torch.Tensor:
        return self.hi - self.lo

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Map cm coordinates into ``[0, 1]^3`` (the encoder's domain)."""
        return (x - self.lo) / self.size

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        return ((x >= self.lo) & (x <= self.hi)).all(dim=-1)


class LambdaField(nn.Module):
    """Coordinate-based field ``lambda_theta(r)`` returning a non-negative scalar.

    Args:
        box:                  reconstruction volume.
        encoder:              ``"hash"`` (default, recommended) or
                              ``"frequency"`` (NeRF-style sinusoidal).
        hidden_dim:           width of the MLP hidden layer.
        n_hidden_layers:      number of hidden layers in the MLP.
        n_levels:             hash encoder — number of resolution levels.
        n_features_per_level: hash encoder — feature dimension per entry.
        log2_hashmap_size:    hash encoder — ``log2`` of the table size.
        base_resolution:      hash encoder — coarsest grid resolution.
        per_level_scale:      hash encoder — geometric scale factor.
        n_freqs:              frequency encoder — number of harmonics.
        lambda_air:           value of ``lambda`` at initialisation
                              (the last-layer bias is set so that
                              ``softplus(bias) = lambda_air``).
        out_scale:            multiplicative scale of the output.
    """

    def __init__(
        self,
        box: VolumeBox,
        *,
        encoder: str = "hash",
        hidden_dim: int = 128,
        n_hidden_layers: int = 1,
        # hash-grid hyperparameters
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        per_level_scale: float = 1.5,
        # frequency-encoding hyperparameters
        n_freqs: int = 6,
        # output transform
        lambda_air: float = LAMBDA_AIR,
        out_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.box = box
        if encoder == "hash":
            self.encoder = HashGridEncoding(
                n_levels=n_levels,
                n_features_per_level=n_features_per_level,
                log2_hashmap_size=log2_hashmap_size,
                base_resolution=base_resolution,
                per_level_scale=per_level_scale,
            )
        elif encoder == "frequency":
            self.encoder = FrequencyEncoding(n_freqs=n_freqs, include_input=True)
        else:
            raise ValueError(f"unknown encoder: {encoder!r}")

        layers: list[nn.Module] = []
        in_dim = self.encoder.output_dim
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)]
            in_dim = hidden_dim
        layers += [nn.Linear(in_dim, 1)]
        self.mlp = nn.Sequential(*layers)

        self.lambda_air = float(lambda_air)
        self.out_scale = float(out_scale)
        # softplus(b) = lambda_air / out_scale  -> bias the last layer accordingly
        target = max(self.lambda_air / self.out_scale, 1e-12)
        b0 = math.log(math.expm1(target)) if target < 20 else target  # inverse softplus
        with torch.no_grad():
            last_linear = self.mlp[-1]
            assert isinstance(last_linear, nn.Linear)
            last_linear.bias.fill_(b0)
            last_linear.weight.mul_(0.01)

    def forward(self, x_cm: torch.Tensor) -> torch.Tensor:
        """Evaluate ``lambda_theta`` at the world coordinates ``x_cm``.

        Args:
            x_cm: ``(N, 3)`` query points in cm.

        Returns:
            ``(N,)`` non-negative ``lambda`` values in cm^-1.
        """
        u = self.box.normalize(x_cm)
        feat = self.encoder(u)
        raw = self.mlp(feat).squeeze(-1)
        return torch.nn.functional.softplus(raw) * self.out_scale


__all__ = ["LambdaField", "VolumeBox"]
