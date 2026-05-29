# MINT — Muon Implicit Neural Tomography

A differentiable neural-field framework for cosmic-ray muon scattering
tomography (MST).

MINT represents the inverse radiation length

```
λ(r) = 1 / X₀(r),       r ∈ ℝ³
```

as a coordinate-based neural network with a multi-resolution hash-grid
backbone, and fits its parameters by stochastic gradient descent on the
analytic Highland multiple-Coulomb-scattering log-likelihood of
per-track angular deflections and lateral offsets.

The result is a continuous, mesh-free reconstruction that adapts
spatial resolution to the data and avoids the per-voxel
Fisher-information starvation of fixed-grid methods (PoCA, MLEM) in the
sparse-track regime.

> If you use this code in your research, please cite the accompanying
> paper:
>
> D. Pagano, *A differentiable neural-field forward model for cosmic-ray
> muon scattering tomography*, in preparation.

---

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Data format](#data-format)
- [Library API](#library-api)
- [Command-line tools](#command-line-tools)
- [How MINT works (short)](#how-mint-works-short)
- [Project layout](#project-layout)
- [License](#license)

---

## Installation

MINT requires Python ≥ 3.10 and PyTorch ≥ 2.0.  It is implemented in
pure PyTorch — no C++/CUDA extension is required — and runs on CPU,
NVIDIA GPU (CUDA) and Apple Silicon (MPS).

```bash
git clone https://github.com/USERNAME/mint.git
cd mint
pip install -e .
```

To run the tests and the linter:

```bash
pip install -e ".[dev]"
pytest tests/
```

---

## Quick start

The 30-second demonstration: train MINT on a small synthetic dataset
and print a slice of the reconstructed `λ` field.

```bash
python examples/train_minimal.py
```

For a real reconstruction, you need (a) a muon dataset in the
[expected format](#data-format) and (b) the bounding box of your
reconstruction volume.  Then:

```bash
# Train
python scripts/train_mint.py \
    --dataset  mydata.pt \
    --box-lo  -50 -50 -50 \
    --box-hi   50  50  50 \
    --device   cuda \
    --n-iters  4000 \
    --out      mint_model.pt

# Evaluate on a 128³ grid
python scripts/eval_mint.py \
    --model  mint_model.pt \
    --grid   128 128 128 \
    --out    reconstruction.pt
```

The output checkpoint `reconstruction.pt` contains a
`(128, 128, 128)` tensor of `λ` values together with the corresponding
axis vectors, ready to be visualised with any volumetric plotting
library (e.g.\ `matplotlib`, `plotly`, `napari`, `pyvista`).

---

## Data format

A muon dataset is a PyTorch `.pt` file containing a single dictionary
with five tensors (one row per track):

| Key             | Shape   | Units          | Description                                        |
|-----------------|---------|----------------|----------------------------------------------------|
| `entry_point`   | `(N,3)` | cm             | Track entry position on the upstream tracker.       |
| `entry_dir`     | `(N,3)` | unit vector    | Upstream direction (forward-going, `uz > 0`).       |
| `exit_point`    | `(N,3)` | cm             | Track exit position on the downstream tracker.      |
| `exit_dir`      | `(N,3)` | unit vector    | Downstream direction.                               |
| `momentum_mev`  | `(N,)`  | MeV/c          | Muon momentum.                                      |

The 4-D per-track observation `(dθ_x, dθ_y, dx, dy)` (projected angular
deflections and lateral offsets) is derived automatically from the
entry/exit positions and directions when the dataset is constructed.

You can build the file directly from Python:

```python
import torch
from mint import MuonDataset

ds = MuonDataset(
    entry_point=torch.from_numpy(my_entry_xyz).float(),
    entry_dir=torch.from_numpy(my_entry_dir).float(),
    exit_point=torch.from_numpy(my_exit_xyz).float(),
    exit_dir=torch.from_numpy(my_exit_dir).float(),
    momentum_mev=torch.from_numpy(my_momentum).float(),
)
ds.save("mydata.pt")
```

**Coordinate system.**  MINT assumes muons travel along the positive
`z` direction (`uz > 0`).  The reconstruction box is axis-aligned but
otherwise arbitrary; entry and exit points must lie on its boundary
faces.

---

## Library API

```python
import torch
from mint import (
    MuonDataset, LambdaField, VolumeBox,
    TrainConfig, train, evaluate_grid,
)

# 1) Load data and define the reconstruction volume.
ds = MuonDataset.load("mydata.pt")
box = VolumeBox(
    lo=torch.tensor([-50., -50., -50.]),
    hi=torch.tensor([+50., +50., +50.]),
)

# 2) Instantiate the neural field.
field = LambdaField(
    box=box,
    encoder="hash",
    n_levels=16,
    log2_hashmap_size=19,
    base_resolution=16,
    per_level_scale=1.5,
    hidden_dim=128,
)

# 3) Train.
cfg = TrainConfig(
    n_iters=4000,
    batch_size=8192,
    n_samples=96,
    lr=2e-3,
    w_tv=2e-3,
    w_bg=5e-3,
    device="cuda",
)
state = train(field, ds, cfg)

# 4) Inference on a regular grid.
field.eval()
grid = evaluate_grid(field, nx=128, ny=128, nz=128)   # (128,128,128) tensor in cm⁻¹
```

### Custom evaluation points

Once trained, the field can be queried at any 3-D point:

```python
pts = torch.tensor([[0., 0., 0.], [10., -5., 2.]])
with torch.no_grad():
    lam = field(pts)   # (2,) tensor in cm⁻¹
```

---

## Command-line tools

Two convenience scripts are installed as console entry points after
`pip install`:

```bash
mint-train  --help     # equivalent to: python scripts/train_mint.py
mint-eval   --help     # equivalent to: python scripts/eval_mint.py
```

Both expose every relevant hyperparameter (network width, hash-table
size, regulariser weights, optimiser settings, …) so MINT can be
driven end-to-end without writing any Python.

### Recommended starting hyperparameters

The defaults below mirror those used in the paper; they are sensible
for most large-volume muon-scattering geometries.

| Hyperparameter                | Default    | Note                                                                                 |
|-------------------------------|-----------:|--------------------------------------------------------------------------------------|
| `--n-iters`                   | `4000`     | Increase to `8000–12000` for sparse-data scenarios.                                  |
| `--batch-size`                | `8192`     | Largest value that fits in VRAM.                                                     |
| `--n-samples`                 | `96`       | Quadrature points per track.                                                         |
| `--lr` / `--lr-min-ratio`     | `2e-3` / `0.05` | Adam with cosine decay.                                                          |
| `--w-tv`                      | `2e-3`     | Total-variation regulariser; set to `0` to disable.                                  |
| `--w-bg`                      | `5e-3`     | Background L2 prior toward `--lambda-bg`; set to `0` to disable.                     |
| `--lambda-bg` / `--lambda-init` | `LAMBDA_AIR` | Override for scenes embedded in a denser medium (concrete, water, …).            |
| `--log2-hashmap-size`         | `19`       | Increase by 1 for larger volumes (memory doubles).                                   |

---

## How MINT works (short)

For a track with entry/exit points `(r_in, r_out)` and momentum `p`,
the integrated inverse radiation length

```
T = ∫₀ᴸ λ_θ(r(s)) ds
```

is estimated by **stratified jittered Monte-Carlo quadrature** with
`N` samples per track.  The Highland MCS formula gives the projected
angular variance

```
θ₀² = (13.6 MeV / pβ)² · T
```

from which the full 4×4 covariance of the per-track observation
`y = (dθ_x, dθ_y, dx, dy)` factorises into two independent 2×2 blocks.
This admits a closed-form Gaussian NLL with no matrix decomposition.

The training objective is

```
J(θ) = NLL(θ) + α · R_TV(θ) + β · R_bg(θ)
```

where `R_TV` is a stochastic total-variation regulariser (`α` defaults
to `2 × 10⁻³`) and `R_bg` is a soft L2 prior toward `λ_air` (`β`
defaults to `5 × 10⁻³`).  Both regularisers are evaluated on a few
thousand random points per iteration.

The encoder is the multi-resolution hash grid of Müller et al. 2022
(Instant-NGP), and the decoder is a single-hidden-layer MLP of
width 128; the final softplus enforces `λ > 0` smoothly.  The bias of
the last linear layer is initialised so that `λ ≈ λ_air` everywhere
at iteration 1, which keeps the per-track `T` away from zero and
prevents the `1/T` Highland singularity from blowing up the first
gradient.

---

## Project layout

```
mint/
├── README.md                # this file
├── LICENSE                  # MIT
├── pyproject.toml           # PEP 621 packaging
├── src/mint/                # the package
│   ├── __init__.py
│   ├── data.py              # MuonDataset, observation derivation
│   ├── encoding.py          # multi-resolution hash + frequency encoders
│   ├── field.py             # LambdaField, VolumeBox
│   ├── forward.py           # stratified-jittered line integral
│   ├── losses.py            # Gaussian NLL, TV, background prior
│   ├── physics.py           # Highland constants and MCS covariance
│   └── train.py             # training loop + grid evaluation
├── scripts/
│   ├── train_mint.py        # CLI: train MINT on a saved dataset
│   └── eval_mint.py         # CLI: evaluate a trained model on a grid
├── examples/
│   └── train_minimal.py     # 30-second smoke demo (no external data)
└── tests/
    └── test_smoke.py        # imports, forward pass, mini training run
```

The repository contains **only** the MINT reconstruction code.
Simulation, baseline algorithms (PoCA, MLEM), and benchmark-specific
analysis scripts used in the paper are kept in a separate research
repository.

---

## License

MIT — see [`LICENSE`](LICENSE).
