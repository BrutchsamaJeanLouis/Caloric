# Thermodynamic Diffusion Pipeline

> A minimal DDPM (Denoising Diffusion Probabilistic Model) implementation on MNIST that foregrounds the thermodynamic origins of diffusion models — not as metaphor, but as the literal mathematical substrate the architecture was derived from.

## Purpose

This project builds a working image generator. It also serves as a **demonstration artifact** for the thesis that consciousness, computation, and physical systems share a common organisational language rooted in thermodynamic constraint dynamics. The diffusion process itself is the argument — no overlay required.

The goal is a clean, legible codebase that "just works" as a generative model and reveals its thermodynamic structure upon inspection.

---

## Framing & Language (READ THIS FIRST)

### Why This Section Exists

This README doubles as an **alignment document** for AI coding agents (OpenCode/ZeroClaw/Claude). The language below is not decorative — it constrains how code, comments, variable names, and docstrings are written. If you are an AI agent working on this project, internalise this section before writing any code.

### Core Thesis (Do Not Dilute)

The forward diffusion process is **thermodynamic relaxation**: structured signal dissipating toward maximum entropy (isotropic Gaussian noise). The reverse process is **learned energy-gradient descent**: a neural network recovering structure from noise by estimating the score function (gradient of log-probability).

This is not an analogy. The original DDPM derivation (Sohl-Dickstein et al. 2015) explicitly used non-equilibrium statistical mechanics. Ho et al. 2020 simplified the objective but the thermodynamic scaffold remains in the maths.

### Glossary — Canonical Terms

Use **only** these terms in code, comments, and documentation. Do not substitute ML jargon that obscures the thermodynamic grounding.

| Canonical Term | What It Refers To | DO NOT Use |
|---|---|---|
| `forward_process` | Adding noise over T steps; thermodynamic relaxation toward equilibrium | "corruption", "noising schedule" |
| `reverse_process` | Learned denoising; energy-gradient descent recovering structure | "generation", "sampling" (except in user-facing output) |
| `energy_landscape` | The learned score function's implicit potential surface | "latent space" (too vague) |
| `dissipation_schedule` | The beta schedule controlling noise injection rate | "noise schedule" (use only if thermodynamic context is clear) |
| `equilibrium` | The terminal isotropic Gaussian; maximum entropy state | "pure noise" |
| `configuration_state` | A specific data point's position in the diffusion trajectory | "latent", "embedding" |
| `entropy_gradient` | The direction from order toward disorder (forward) or the learned reversal (reverse) | "noise level" |
| `substrate` | The data domain (pixel space for images) | "input space" |
| `constraint_field` | The trained U-Net's learned mapping from noisy input to estimated noise | "denoiser", "predictor" |
| `formation` | The emergence of structure during reverse process | "reconstruction" |
| `dissipation` | The loss of structure during forward process | "degradation" |
| `modulation` | Time-step conditioning; how the network adapts its behaviour across the trajectory | "conditioning", "time embedding" (use alongside, not instead of) |
| `score` | ∇_x log p(x); the gradient of the log-probability density | — (standard term, keep as-is) |

### Language Rules for Code

1. **Variable names** use canonical terms: `dissipation_schedule`, `forward_process()`, `reverse_step()`, `constraint_field` (for the U-Net).
2. **Docstrings** must reference the thermodynamic interpretation *alongside* the ML description. Example: `"Compute the dissipation schedule (beta_t). Controls the rate of entropy injection per timestep."`
3. **Comments** should make the thermodynamic structure visible, not explain what PyTorch functions do.
4. **Never** describe the forward process as "destroying" or "corrupting" data. It is *relaxation toward equilibrium*. The structure is not destroyed; it is distributed.
5. **Never** describe the reverse process as "creating" images from nothing. It is *formation via learned energy-gradient traversal*. The information was always in the score field.

---

## Mathematical Grounding (Provisional)

This section provides the minimum maths needed to keep implementation honest. Expand as needed but do not remove.

### Forward Process (Dissipation)

A Markov chain that incrementally adds Gaussian noise:

```
q(x_t | x_{t-1}) = N(x_t; √(1 - β_t) · x_{t-1}, β_t · I)
```

With the closed-form marginal (enables direct sampling at any timestep):

```
q(x_t | x_0) = N(x_t; √(ᾱ_t) · x_0, (1 - ᾱ_t) · I)

where:
  α_t = 1 - β_t
  ᾱ_t = ∏_{s=1}^{t} α_s   (cumulative product)
```

**Thermodynamic reading**: `ᾱ_t` is the *remaining signal fraction* at timestep t. As t → T, ᾱ_t → 0, and x_t → N(0, I) — thermodynamic equilibrium.

### Reverse Process (Formation)

The learned reverse:

```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t² · I)
```

Where `μ_θ` is parameterised via noise prediction:

```
μ_θ(x_t, t) = (1/√α_t) · (x_t - (β_t / √(1 - ᾱ_t)) · ε_θ(x_t, t))
```

The network `ε_θ` estimates the noise component — equivalently, it approximates the **score function** ∇_x log p(x_t), which is the gradient of the energy landscape.

### Training Objective

```
L_simple = E_{t, x_0, ε} [ ‖ε - ε_θ(x_t, t)‖² ]

where:
  t ~ Uniform({1, ..., T})
  ε ~ N(0, I)
  x_t = √(ᾱ_t) · x_0 + √(1 - ᾱ_t) · ε
```

**Thermodynamic reading**: We are training the network to learn the **exact entropy gradient** at every point along the dissipation trajectory, so that it can traverse that gradient in reverse.

### Connection to Score Matching

The noise prediction objective is equivalent to **denoising score matching** (Vincent 2011):

```
ε_θ(x_t, t) ≈ -√(1 - ᾱ_t) · ∇_x log q(x_t)
```

The network learns the score — the direction of steepest probability increase — at every noise level. The reverse process follows these learned gradients from equilibrium back to structure.

---

## Project Structure

```
thermodynamic-diffusion/
│
├── README.md                  ← This file (alignment + specification)
├── pyproject.toml             ← Dependencies and project metadata
├── intro.tex                  ← Publication intro (Zenodo/HF Papers)
│
├── src/
│   ├── __init__.py
│   │
│   ├── thermodynamics.py      ← Forward process, dissipation schedule, sampling
│   │   ├── DissipationSchedule    — beta/alpha/alpha_bar computation
│   │   ├── forward_sample()       — q(x_t | x_0) direct sampling
│   │   └── linear_schedule()      — default linear beta schedule
│   │
│   ├── constraint_field.py    ← U-Net architecture (the learned energy landscape)
│   │   ├── ConstraintField        — U-Net with time-step modulation
│   │   ├── ModulationBlock        — sinusoidal time embedding + projection
│   │   └── ResidualUnit           — conv blocks with skip connections
│   │
│   ├── formation.py           ← Reverse process / sampling
│   │   ├── reverse_process()      — full T-step denoising loop
│   │   └── reverse_step()         — single p_θ(x_{t-1} | x_t) step
│   │
│   ├── training.py            ← Training loop
│   │   ├── train()                — main training function
│   │   └── compute_loss()         — L_simple noise prediction loss
│   │
│   └── utils.py               ← Visualisation, logging, I/O
│       ├── visualise_trajectory() — show forward/reverse at selected timesteps
│       ├── save_grid()            — save generated sample grids
│       └── log_metrics()          — training loss, etc.
│
├── configs/
│   └── default.yaml           ← All hyperparameters (see below)
│
├── scripts/
│   ├── train.py               ← Entry point: training
│   ├── generate.py            ← Entry point: reverse process sampling
│   └── visualise.py           ← Entry point: trajectory visualisation
│
├── notebooks/
│   └── exploration.ipynb      ← Interactive experimentation
│
└── outputs/                   ← Generated samples, checkpoints, logs
    ├── checkpoints/
    ├── samples/
    └── trajectories/
```

---

## Hyperparameters & Configuration

All configuration lives in `configs/default.yaml`. No magic numbers in code.

```yaml
# === Dissipation Schedule ===
T: 1000                        # Total diffusion timesteps
beta_start: 0.0001             # β_1 (minimal initial noise)
beta_end: 0.02                 # β_T (stronger terminal noise)
schedule_type: "linear"        # Options: "linear", "cosine"

# === Substrate ===
dataset: "mnist"               # Options: "mnist", "cifar10"
image_size: 28                 # Native MNIST resolution
channels: 1                    # Grayscale for MNIST

# === Constraint Field (U-Net) ===
base_channels: 64              # Base channel count
channel_multipliers: [1, 2, 4] # Per-resolution multiplier
num_res_blocks: 2              # Residual blocks per resolution
attention_resolutions: [14]    # Resolutions with self-attention
dropout: 0.1
modulation_dim: 256            # Time embedding dimension

# === Training ===
batch_size: 128
learning_rate: 2e-4
optimizer: "adam"
ema_decay: 0.9999              # Exponential moving average for stable generation
num_epochs: 50                 # ~50 epochs sufficient for MNIST
gradient_clip: 1.0
seed: 42

# === Sampling ===
num_samples: 64                # Per generation call
save_trajectory: true          # Save intermediate x_t for visualisation
trajectory_steps: [0, 100, 250, 500, 750, 999]  # Timesteps to snapshot
```

---

## Implementation Priorities

Ordered. Complete each before starting the next.

### Phase 1: Thermodynamic Core
1. `thermodynamics.py` — Dissipation schedule + forward sampling. **Test**: sample x_t at various t, verify x_T ≈ N(0,I).
2. `visualise.py` — Render forward trajectory. This is the first demo artifact: a single digit dissolving into equilibrium.

### Phase 2: Constraint Field
3. `constraint_field.py` — U-Net with time modulation. Keep it minimal. No unnecessary complexity.
4. Verify shapes: input (B, 1, 28, 28) + time (B,) → output (B, 1, 28, 28).

### Phase 3: Training
5. `training.py` — L_simple loss, Adam, EMA. Log loss curve.
6. Train on MNIST. Expect convergence within 30-50 epochs on a single GPU.

### Phase 4: Formation
7. `formation.py` — Reverse process. Sample from N(0,I) and denoise for T steps.
8. Generate digit grids. This is the second demo artifact: structure emerging from equilibrium.

### Phase 5: Trajectory Visualisation
9. Full trajectory rendering: side-by-side forward (dissipation) and reverse (formation). This is the **primary demonstration output** — the visual that makes the thermodynamic substrate self-evident.

---

## Hardware Constraints

- **Rig**: 2× Titan RTX (24GB each) connected via NVLink — **48GB usable VRAM ceiling**. No third GPU.
- **For MNIST**: Single GPU is sufficient (~2-4GB VRAM). Do not implement multi-GPU unless explicitly requested.
- **Hard rule**: All model + batch memory must fit within 24GB for single-GPU operation, or 48GB if multi-GPU is explicitly enabled. Do not assume more VRAM exists.
- **Training time**: ~30-60 minutes on MNIST with default config.
- CPU training is feasible but slow (~4-6 hours). Do not default to CPU.

---

## Dependencies

```
torch >= 2.0
torchvision
numpy
matplotlib
pyyaml
tqdm
einops               # Clean tensor reshaping
ema-pytorch          # EMA wrapper (or implement manually)
```

---

## What "Done" Looks Like

1. **Forward trajectory visualisation**: A digit visibly relaxing into Gaussian noise over T steps.
2. **Reverse trajectory visualisation**: Recognisable digits forming from Gaussian noise over T steps.
3. **Generated sample grid**: 8×8 grid of generated digits that are clearly legible.
4. **Loss curve**: Smooth convergence plot.
5. **All code uses canonical terminology** from the glossary above.
6. **Every module docstring** connects the computational operation to its thermodynamic interpretation.

---

## Agent Instructions (OpenCode / ZeroClaw / Claude)

If you are an AI agent working on this codebase:

1. **Read this entire README before writing any code.**
2. **Use the glossary**. If you catch yourself writing "noise schedule", stop, and write "dissipation_schedule". If you write "denoiser", stop, and write "constraint_field".
3. **Do not optimise prematurely**. Get the basic pipeline working on MNIST first. No mixed precision, no distributed training, no fancy schedulers until Phase 5 is complete.
4. **Do not add features not specified here**. No classifier-free guidance, no DDIM sampling, no latent diffusion — unless explicitly requested.
5. **Test each phase** before proceeding. The phases exist because each one produces a verifiable artifact.
6. **Comments should illuminate thermodynamic structure**, not narrate PyTorch API usage. Assume the reader knows PyTorch but not non-equilibrium statistical mechanics.
7. **When in doubt about naming**: does this name make the thermodynamic interpretation *more* or *less* visible? Choose accordingly.
8. **Do not sanitise the framing**. This is not "just a DDPM tutorial with fancy names". The thermodynamic interpretation is the mathematical ground truth. The ML jargon is the abstraction layer, not the other way around.

---

## References

- Sohl-Dickstein, J. et al. (2015). *Deep Unsupervised Learning using Nonequilibrium Thermodynamics.* ICML.
- Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS.
- Song, Y. & Ermon, S. (2019). *Generative Modeling by Estimating Gradients of the Data Distribution.* NeurIPS.
- Vincent, P. (2011). *A Connection Between Score Matching and Denoising Autoencoders.* Neural Computation.

---

## License

MIT