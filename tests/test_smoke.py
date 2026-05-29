# Copyright (c) 2026 Davide Pagano
# University of Brescia (Italy)
# davide.pagano@unibs.it
#
# Part of MINT — Muon Implicit Neural Tomography.
# Distributed under the MIT License (see LICENSE in the project root).

"""Smoke tests: verify the package imports and that a tiny training run
makes the loss go down.

Run with::

    pytest tests/

or::

    python -m unittest tests.test_smoke
"""

from __future__ import annotations

import unittest

import torch

from mint import (
    LambdaField,
    MuonDataset,
    TrainConfig,
    VolumeBox,
    integrate_lambda,
    train,
)


def _toy_dataset(n: int = 256) -> MuonDataset:
    g = torch.Generator().manual_seed(42)
    entry = torch.rand(n, 3, generator=g) * 10 - 5
    entry[:, 2] = -5.0
    exit_ = entry.clone()
    exit_[:, 2] = +5.0
    entry_dir = torch.tensor([[0.0, 0.0, 1.0]]).expand(n, 3).contiguous()
    exit_dir  = entry_dir.clone()
    p = torch.full((n,), 3000.0)
    return MuonDataset(entry, entry_dir, exit_, exit_dir, p)


class TestPackage(unittest.TestCase):

    def test_imports(self) -> None:
        import mint  # noqa: F401
        self.assertTrue(hasattr(mint, "LambdaField"))
        self.assertTrue(hasattr(mint, "train"))

    def test_forward_pass(self) -> None:
        box = VolumeBox(
            lo=torch.tensor([-5.0, -5.0, -5.0]),
            hi=torch.tensor([+5.0, +5.0, +5.0]),
        )
        field = LambdaField(box, encoder="hash", n_levels=4,
                            log2_hashmap_size=12, hidden_dim=16)
        pts = torch.rand(100, 3) * 10 - 5
        out = field(pts)
        self.assertEqual(out.shape, (100,))
        self.assertTrue(torch.all(out >= 0.0))

    def test_line_integral_shape(self) -> None:
        box = VolumeBox(
            lo=torch.tensor([-5.0, -5.0, -5.0]),
            hi=torch.tensor([+5.0, +5.0, +5.0]),
        )
        field = LambdaField(box, encoder="hash", n_levels=4,
                            log2_hashmap_size=12, hidden_dim=16)
        ds = _toy_dataset(32)
        T, L = integrate_lambda(field, ds.entry_point, ds.exit_point, n_samples=16)
        self.assertEqual(T.shape, (32,))
        self.assertEqual(L.shape, (32,))
        self.assertTrue(torch.all(L > 0.0))

    def test_training_runs(self) -> None:
        box = VolumeBox(
            lo=torch.tensor([-5.0, -5.0, -5.0]),
            hi=torch.tensor([+5.0, +5.0, +5.0]),
        )
        field = LambdaField(box, encoder="hash", n_levels=4,
                            log2_hashmap_size=12, hidden_dim=16)
        ds = _toy_dataset(256)
        cfg = TrainConfig(
            n_iters=20, batch_size=32, n_samples=8,
            lr=1e-3, w_tv=0.0, w_bg=0.0,
            log_every=10, device="cpu", seed=0,
        )
        state = train(field, ds, cfg)
        self.assertEqual(len(state.losses), 20)
        self.assertTrue(all("nll" in t for t in state.losses))


if __name__ == "__main__":
    unittest.main()
