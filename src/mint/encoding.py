# Copyright (c) 2026 Davide Pagano
# University of Brescia (Italy)
# davide.pagano@unibs.it
#
# Part of MINT — Muon Implicit Neural Tomography.
# Distributed under the MIT License (see LICENSE in the project root).

"""Coordinate encodings for the neural field.

This module provides two encoders that wrap a 3-D query coordinate
``r in [0, 1]^3`` into a feature vector ``f in R^F`` consumed by the
MLP decoder:

* :class:`HashGridEncoding` — multi-resolution hash encoding of
  M\\"uller et al. 2022 (Instant-NGP), implemented in pure PyTorch.
  This is the encoder used in the MINT paper.

* :class:`FrequencyEncoding` — sinusoidal positional encoding
  (NeRF-style), provided as a baseline / fallback.

Both encoders are fully autograd-friendly and run on CPU, CUDA, and
Apple MPS without any C++/CUDA extension.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# Three large primes from the original Instant-NGP paper.
_PRIMES = (1, 2654435761, 805459861)


def _spatial_hash(coords_int: torch.Tensor, table_size: int) -> torch.Tensor:
    """Spatial hash function (Instant-NGP).

    Maps integer lattice coordinates to a row index of the embedding
    table.  Inputs of shape ``(..., 3)`` (int64); output of shape
    ``(...,)`` in ``[0, table_size)``.
    """
    h = coords_int[..., 0] * _PRIMES[0]
    h = h ^ (coords_int[..., 1] * _PRIMES[1])
    h = h ^ (coords_int[..., 2] * _PRIMES[2])
    return h % table_size


class HashGridEncoding(nn.Module):
    """Multi-resolution hash grid encoding for 3-D inputs in ``[0, 1]^3``.

    At each of ``n_levels`` levels the volume is overlaid with a regular
    grid of resolution ``base_resolution * per_level_scale ** level``.
    The eight corners of the cell enclosing the query are hashed into a
    small fixed-size table of learnable embeddings; trilinear
    interpolation between the eight hashed embeddings produces the
    per-level feature vector.  Per-level outputs are concatenated.

    Args:
        n_levels:             number of resolution levels.
        n_features_per_level: feature dimension per hash entry.
        log2_hashmap_size:    ``log2`` of the per-level table size.
        base_resolution:      coarsest resolution (cells per side).
        per_level_scale:      geometric scale factor between consecutive
                              levels.
    """

    def __init__(
        self,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        per_level_scale: float = 1.5,
    ) -> None:
        super().__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.table_size = 1 << log2_hashmap_size
        self.base_resolution = base_resolution
        self.per_level_scale = per_level_scale

        resolutions = [
            int(math.floor(base_resolution * (per_level_scale ** lvl)))
            for lvl in range(n_levels)
        ]
        self.register_buffer(
            "resolutions",
            torch.tensor(resolutions, dtype=torch.int64),
            persistent=False,
        )
        self.embeddings = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(self.table_size, n_features_per_level).uniform_(-1e-4, 1e-4)
                )
                for _ in range(n_levels)
            ]
        )

    @property
    def output_dim(self) -> int:
        return self.n_levels * self.n_features_per_level

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``x`` of shape ``(N, 3)`` in ``[0, 1]`` to features of
        shape ``(N, n_levels * n_features_per_level)``."""
        assert x.shape[-1] == 3
        device = x.device
        corner_offsets = torch.tensor(
            [
                [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
            ],
            dtype=torch.int64,
            device=device,
        )  # (8, 3)
        out = []
        for lvl in range(self.n_levels):
            res = int(self.resolutions[lvl].item())
            scaled = x * res
            base = torch.floor(scaled).to(torch.int64)         # (N, 3)
            frac = scaled - base.to(scaled.dtype)              # (N, 3)
            corners = base.unsqueeze(1) + corner_offsets.unsqueeze(0)  # (N, 8, 3)
            idx = _spatial_hash(corners, self.table_size)      # (N, 8)
            feats = self.embeddings[lvl][idx]                  # (N, 8, F)
            # trilinear interpolation weights
            fx, fy, fz = frac.unbind(-1)
            wx = torch.stack([1 - fx, fx], dim=-1)
            wy = torch.stack([1 - fy, fy], dim=-1)
            wz = torch.stack([1 - fz, fz], dim=-1)
            w = (
                wx[:, [0, 1, 0, 1, 0, 1, 0, 1]]
                * wy[:, [0, 0, 1, 1, 0, 0, 1, 1]]
                * wz[:, [0, 0, 0, 0, 1, 1, 1, 1]]
            )  # (N, 8)
            interp = (feats * w.unsqueeze(-1)).sum(dim=1)      # (N, F)
            out.append(interp)
        return torch.cat(out, dim=-1)


class FrequencyEncoding(nn.Module):
    """Sinusoidal positional encoding (NeRF-style).

    Provided as a portable baseline.  For most MST scenes the hash grid
    encoding is preferred (see :class:`HashGridEncoding`).
    """

    def __init__(self, n_freqs: int = 6, include_input: bool = True) -> None:
        super().__init__()
        self.n_freqs = n_freqs
        self.include_input = include_input
        freqs = 2.0 ** torch.arange(n_freqs).float() * math.pi
        self.register_buffer("freqs", freqs, persistent=False)

    @property
    def output_dim(self) -> int:
        d_in = 3
        return d_in * (1 if self.include_input else 0) + 2 * 3 * self.n_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xc = 2.0 * x - 1.0
        scaled = xc.unsqueeze(-1) * self.freqs        # (..., 3, n_freqs)
        sins = torch.sin(scaled)
        coss = torch.cos(scaled)
        out = torch.cat([sins, coss], dim=-1).flatten(-2)
        if self.include_input:
            out = torch.cat([xc, out], dim=-1)
        return out


__all__ = ["HashGridEncoding", "FrequencyEncoding"]
