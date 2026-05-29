# Copyright (c) 2026 Davide Pagano
# University of Brescia (Italy)
# davide.pagano@unibs.it
#
# Part of MINT — Muon Implicit Neural Tomography.
# Distributed under the MIT License (see LICENSE in the project root).

"""Muon-track data containers.

The minimum information that MINT requires from each reconstructed
muon track is

* the entry and exit *points* on the boundary of the reconstruction
  volume (cm);
* the entry and exit *directions* (unit vectors) measured by the
  upstream and downstream trackers;
* the muon momentum in MeV/c.

From these, the 4-D per-track observation
``(dtheta_x, dtheta_y, dx, dy)`` --- projected angular deflections and
lateral offsets --- is derived automatically when the dataset is
constructed.  Tracks are stored as flat tensors so that a
:class:`~torch.utils.data.DataLoader` (or simple advanced indexing)
can hand whole batches to the training step without per-sample Python
overhead.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class MuonTrack:
    """A single reconstructed muon track.

    All quantities are stored in cm and rad, momentum in MeV/c.
    Directions are unit vectors with ``uz > 0`` (forward-going).
    """

    entry_point: torch.Tensor    # (3,)
    entry_dir:   torch.Tensor    # (3,) unit
    exit_point:  torch.Tensor    # (3,)
    exit_dir:    torch.Tensor    # (3,) unit
    momentum_mev: float


def projected_deflections(
    entry_dir: torch.Tensor,
    exit_dir: torch.Tensor,
) -> torch.Tensor:
    """Projected angular deflections ``(dtheta_x, dtheta_y)`` in radians.

    Uses the convention that the muon is forward-going along ``z``.
    For each projection plane ``k in {x, y}``:

    .. math::

       \\Delta\\theta_k = \\arctan\\!\\bigl(u^{\\mathrm{out}}_k /
                                            u^{\\mathrm{out}}_z\\bigr)
                       - \\arctan\\!\\bigl(u^{\\mathrm{in}}_k /
                                            u^{\\mathrm{in}}_z\\bigr) .
    """
    uz_in = entry_dir[..., 2].clamp_min(1e-6)
    uz_out = exit_dir[..., 2].clamp_min(1e-6)
    th_x = torch.atan2(exit_dir[..., 0], uz_out) - torch.atan2(entry_dir[..., 0], uz_in)
    th_y = torch.atan2(exit_dir[..., 1], uz_out) - torch.atan2(entry_dir[..., 1], uz_in)
    return torch.stack([th_x, th_y], dim=-1)


def lateral_offsets(
    entry_point: torch.Tensor,
    entry_dir: torch.Tensor,
    exit_point: torch.Tensor,
) -> torch.Tensor:
    """Lateral offsets ``(dx, dy)`` between the actual exit point and
    the straight-line extrapolation of the entry track to the exit
    ``z``-plane.
    """
    uz = entry_dir[..., 2].clamp_min(1e-6)
    dz = exit_point[..., 2] - entry_point[..., 2]
    pred_x = entry_point[..., 0] + dz * entry_dir[..., 0] / uz
    pred_y = entry_point[..., 1] + dz * entry_dir[..., 1] / uz
    return torch.stack([
        exit_point[..., 0] - pred_x,
        exit_point[..., 1] - pred_y,
    ], dim=-1)


class MuonDataset(Dataset):
    """Tensor-backed dataset of ``N`` muon tracks.

    Args:
        entry_point:  ``(N, 3)`` entry positions on the upstream tracker (cm).
        entry_dir:    ``(N, 3)`` upstream direction (unit vector).
        exit_point:   ``(N, 3)`` exit positions on the downstream tracker (cm).
        exit_dir:     ``(N, 3)`` downstream direction (unit vector).
        momentum_mev: ``(N,)``  muon momentum in MeV/c.

    Attributes:
        observations: ``(N, 4)`` per-track observable
                      ``(dtheta_x, dtheta_y, dx, dy)``.
    """

    def __init__(
        self,
        entry_point: torch.Tensor,
        entry_dir: torch.Tensor,
        exit_point: torch.Tensor,
        exit_dir: torch.Tensor,
        momentum_mev: torch.Tensor,
    ) -> None:
        assert entry_point.shape == exit_point.shape == entry_dir.shape == exit_dir.shape
        assert entry_point.ndim == 2 and entry_point.shape[1] == 3
        assert momentum_mev.shape == (entry_point.shape[0],)
        self.entry_point = entry_point
        self.entry_dir = entry_dir
        self.exit_point = exit_point
        self.exit_dir = exit_dir
        self.momentum_mev = momentum_mev
        self._dtheta = projected_deflections(entry_dir, exit_dir)
        self._dr = lateral_offsets(entry_point, entry_dir, exit_point)
        self.observations = torch.cat([self._dtheta, self._dr], dim=-1)  # (N, 4)

    def __len__(self) -> int:
        return self.entry_point.shape[0]

    def __getitem__(self, idx):
        return {
            "entry": self.entry_point[idx],
            "exit":  self.exit_point[idx],
            "obs":   self.observations[idx],
            "p":     self.momentum_mev[idx],
        }

    def to(self, device) -> "MuonDataset":
        return MuonDataset(
            self.entry_point.to(device),
            self.entry_dir.to(device),
            self.exit_point.to(device),
            self.exit_dir.to(device),
            self.momentum_mev.to(device),
        )

    def save(self, path: str) -> None:
        """Save to a PyTorch ``.pt`` file readable by :meth:`load`."""
        torch.save(
            {
                "entry_point":  self.entry_point,
                "entry_dir":    self.entry_dir,
                "exit_point":   self.exit_point,
                "exit_dir":     self.exit_dir,
                "momentum_mev": self.momentum_mev,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "MuonDataset":
        """Load a dataset previously saved with :meth:`save`."""
        d = torch.load(path, weights_only=True)
        return cls(
            d["entry_point"],
            d["entry_dir"],
            d["exit_point"],
            d["exit_dir"],
            d["momentum_mev"],
        )


__all__ = ["MuonTrack", "MuonDataset", "projected_deflections", "lateral_offsets"]
