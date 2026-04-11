# Thermodynamic Diffusion: A Minimal DDPM Implementation

> **Paper**: [Thermodynamic Diffusion: A Minimal DDPM Implementation](https://doi.org/10.5281/zenodo.19507595) (Zenodo, 2026)

A tiny image generator that demos the thermodynamic foundations of diffusion models as the mathematical substrate the architecture was derived from.

---

## What This Is

This project implements a **Denoising Diffusion Probabilistic Model (DDPM)** from first principles, foregrounding the thermodynamic interpretation that was present in the original formulation but often obscured in modern ML implementations.

**The core insight**: Diffusion models are not just "adding and removing noise". They implement **thermodynamic relaxation** (forward: signal dissipating toward maximum entropy) and **learned energy-gradient descent** (reverse: recovering structure by following learned probability gradients).

### Try It Yourself

```bash
# Train on MNIST (~30-60 minutes on GPU)
python scripts/train.py

# Generate digits from noise
python scripts/generate.py

# Visualise the diffusion trajectory
python scripts/visualise.py
```

---

## The Idea in Plain English

### Forward Process: Dissipation

Start with a clear image (like a handwritten digit). Gradually add Gaussian noise over 1000 steps. By the end, the image is indistinguishable from random noise — it has reached **thermodynamic equilibrium** (maximum entropy).

This is not "destroying" the image. The information is still there, just distributed across the noise pattern in a way that's hard to see.

### Reverse Process: Formation

Train a neural network to predict what noise was added at each step. Once trained, start from pure noise and iteratively remove the predicted noise. After 1000 steps, structure emerges — a recognisable digit forms from what was initially random.

This is **formation via learned energy-gradient traversal**. The network has learned the "score function" — the direction of steepest probability increase at every point in the diffusion trajectory.

### Why This Matters

The original 2015 paper by Sohl-Dickstein et al. explicitly framed this using **non-equilibrium statistical mechanics**. The 2020 DDPM paper by Ho et al. simplified the training objective but kept the thermodynamic scaffold. This implementation makes that scaffold visible again.

---

## Visualising the Process

### Forward Trajectory (Dissipation)

A single digit relaxing into equilibrium over 1000 timesteps:

```
t=0     t=100   t=250   t=500   t=750   t=999
[clear] → [slight blur] → [fuzzy] → [mostly noise] → [noise] → [equilibrium]
```

### Reverse Trajectory (Formation)

Structure emerging from equilibrium:

```
t=999   t=750   t=500   t=250   t=100   t=0
[noise] → [faint structure] → [blurry digit] → [recognisable] → [clear] → [sharp]
```

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/BrutchsamaJeanLouis/Caloric
cd Caloric

# Install dependencies
pip install -e .
```

### Training

```bash
# Train for 50 epochs on MNIST
python scripts/train.py

# Outputs:
# - outputs/checkpoints/constraint_field_epoch_*.pt
# - outputs/samples/generated_samples_*.png
# - outputs/logs/training_loss.csv
```

### Generation

```bash
# Generate 64 digits from noise
python scripts/generate.py --num-samples 64

# Output: outputs/samples/generated_samples_*.png
```

### Visualisation

```bash
# Visualise forward and reverse trajectories
python scripts/visualise.py

# Output: outputs/trajectories/trajectory_*.png
```

---

## The Mathematics (Brief)

### Forward Process

At each timestep `t`, we add Gaussian noise:

```
q(x_t | x_{t-1}) = N(x_t; √(1 - β_t) · x_{t-1}, β_t · I)
```

Where `β_t` is the **dissipation schedule** (controls noise injection rate).

Closed-form sampling at any timestep:

```
q(x_t | x_0) = N(x_t; √(ᾱ_t) · x_0, (1 - ᾱ_t) · I)
```

Where `ᾱ_t` is the cumulative product of `(1 - β_s)` from `s=1` to `t`. As `t → T`, `ᾱ_t → 0` and `x_t → N(0, I)` — equilibrium.

### Reverse Process

The learned reverse:

```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t² · I)
```

The network `ε_θ` predicts the noise component, which is equivalent to estimating the **score function** `∇_x log p(x_t)` — the gradient of the energy landscape.

### Training Objective

Simple noise prediction:

```
L = E_{t, x_0, ε} [ ||ε - ε_θ(x_t, t)||² ]
```

We train the network to learn the exact entropy gradient at every point along the dissipation trajectory.

---

## Project Structure

```
Caloric/
├── src/
│   ├── thermodynamics.py      # Forward process, dissipation schedule
│   ├── constraint_field.py    # U-Net architecture (the learned energy landscape)
│   ├── formation.py           # Reverse process / sampling
│   ├── training.py            # Training loop
│   └── utils.py               # Visualisation, logging, I/O
├── scripts/
│   ├── train.py               # Training entry point
│   ├── generate.py            # Generation entry point
│   └── visualise.py           # Trajectory visualisation
├── configs/
│   └── default.yaml           # All hyperparameters
├── outputs/                   # Generated samples, checkpoints, logs
└── README.md                  # This file
```

---

## Configuration

All hyperparameters in `configs/default.yaml`:

```yaml
# Dissipation schedule
T: 1000                        # Total timesteps
beta_start: 0.0001             # Initial noise rate
beta_end: 0.02                 # Terminal noise rate

# Model
base_channels: 64
channel_multipliers: [1, 2, 4]
num_res_blocks: 2

# Training
batch_size: 128
learning_rate: 2e-4
num_epochs: 50
```

---

## What You'll See

After training and generation:

1. **Generated digits**: 8×8 grid of recognisable handwritten digits (0-9)
2. **Trajectory visualisations**: Side-by-side forward (dissipation) and reverse (formation) trajectories
3. **Loss curve**: Smooth convergence over 50 epochs

The generated digits should be clearly legible — this is a working generative model, not just a research prototype.

---

## For Developers

### Thermodynamic Terminology

This codebase uses thermodynamic terminology throughout (see the [paper](https://doi.org/10.5281/zenodo.19507595) for details):

| Term | Meaning |
|------|---------|
| `forward_process` | Adding noise; thermodynamic relaxation |
| `reverse_process` | Learned denoising; energy-gradient descent |
| `dissipation_schedule` | Beta schedule controlling noise rate |
| `equilibrium` | Terminal isotropic Gaussian (max entropy) |
| `constraint_field` | The trained U-Net |
| `formation` | Structure emergence during reverse process |
| `score` | ∇_x log p(x); gradient of log-probability |

### Extending Beyond MNIST

The implementation is dataset-agnostic. To use other datasets:

1. Implement a data loader (replace `torchvision.datasets.MNIST`)
2. Adjust `image_size` and `channels` in config
3. May need to tune `beta_schedule` and `num_epochs`

---

## References

- **Paper**: [Thermodynamic Diffusion: A Minimal DDPM Implementation](https://doi.org/10.5281/zenodo.19507595) (Zenodo, 2026)
- Sohl-Dickstein, J. et al. (2015). *Deep Unsupervised Learning using Nonequilibrium Thermodynamics.* ICML.
- Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS.
- Song, Y. & Ermon, S. (2019). *Generative Modeling by Estimating Gradients of the Data Distribution.* NeurIPS.

---

## License

MIT License — feel free to use, modify, and share.

---

## Citation

If you use this code or ideas in your work:

```bibtex
@software{jeanlouis2026thermodynamicdiffusion,
  author = {Jean-Louis, Brutchsama},
  title = {Thermodynamic Diffusion: A Minimal DDPM Implementation},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.19507595},
  url = {https://doi.org/10.5281/zenodo.19507595}
}
```
