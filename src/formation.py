import torch
from typing import Optional, List, Tuple
from .thermodynamics import DissipationSchedule


def reverse_step(
    x_t: torch.Tensor,
    t: torch.Tensor,
    constraint_field: torch.nn.Module,
    dissipation_schedule: DissipationSchedule,
    sigma: float = 0.0,
) -> torch.Tensor:
    """
    Single step of the reverse process (formation).

    Computes p_θ(x_{t-1} | x_t) using the learned constraint field to estimate
    the score function (gradient of the energy landscape).

    The reverse step follows:
        μ_θ(x_t, t) = (1/√α_t) · (x_t - (β_t / √(1 - ᾱ_t)) · ε_θ(x_t, t))
        x_{t-1} = μ_θ(x_t, t) + σ · ε   (where ε ~ N(0, I), σ=0 for t=1)

    Thermodynamic Interpretation:
        The constraint field ε_θ estimates the entropy gradient at configuration
        state x_t. We traverse this learned gradient to move from disorder toward
        structure. This is energy-gradient descent on the learned potential surface.

    Args:
        x_t (torch.Tensor): Current configuration state at timestep t.
        t (torch.Tensor): Current timestep (scalar or batch tensor).
        constraint_field (torch.nn.Module): The learned energy landscape mapper.
        dissipation_schedule (DissipationSchedule): The dissipation schedule.
        sigma (float): Noise scale for the reverse step. Set to 0 for t=1.

    Returns:
        torch.Tensor: The formed configuration state x_{t-1}.
    """
    device = x_t.device

    # Ensure t is the right shape for indexing
    if t.dim() == 0:
        t = t.unsqueeze(0)

    # Get schedule parameters
    alphas_bar_t, betas_t = dissipation_schedule.get_params(t)
    alphas_t = dissipation_schedule.alphas[t]

    # Reshape for broadcasting: (batch_size,) -> (batch_size, 1, 1, 1)
    shape = (x_t.shape[0],) + (1,) * (x_t.dim() - 1)
    alphas_bar_t = alphas_bar_t.view(shape)
    betas_t = betas_t.view(shape)
    alphas_t = alphas_t.view(shape)

    # Predict the noise (entropy gradient estimate)
    with torch.no_grad():
        epsilon_pred = constraint_field(x_t, t)

    one_minus_alphas_bar = 1 - alphas_bar_t + 1e-8
    coeff = betas_t / torch.sqrt(one_minus_alphas_bar)
    mu_theta = (x_t - coeff * epsilon_pred) / torch.sqrt(alphas_t + 1e-8)

    # Add noise (except at t=1)
    if sigma > 0:
        noise = sigma * torch.randn_like(x_t)
        x_prev = mu_theta + noise
    else:
        x_prev = mu_theta

    return x_prev


def reverse_process(
    constraint_field: torch.nn.Module,
    dissipation_schedule: DissipationSchedule,
    batch_size: int = 64,
    channels: int = 1,
    image_size: int = 28,
    device: Optional[torch.device] = None,
    save_trajectory: bool = False,
    trajectory_steps: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, Optional[dict]]:
    """
    Full reverse process (formation) from equilibrium to structure.

    Starts from thermodynamic equilibrium (isotropic Gaussian N(0, I)) and
    traverses the learned energy landscape via T steps of energy-gradient descent,
    recovering structured data from noise.

    Thermodynamic Interpretation:
        We begin at maximum entropy (equilibrium) and follow the learned score
        function to traverse the energy landscape back toward low-entropy
        configurations. Structure emerges not from creation but from following
        the entropy gradients encoded in the constraint field.

    Args:
        constraint_field (torch.nn.Module): The trained energy landscape mapper.
        dissipation_schedule (DissipationSchedule): The dissipation schedule.
        batch_size (int): Number of samples to generate.
        channels (int): Number of image channels.
        image_size (int): Image size (assumed square).
        device (torch.device): Device to run sampling on.
        save_trajectory (bool): Whether to save intermediate states.
        trajectory_steps (List[int]): Timesteps to snapshot if saving trajectory.

    Returns:
        Tuple[torch.Tensor, Optional[dict]]:
            - Final formed samples x_0 (B, C, H, W)
            - Optional trajectory dict mapping timestep -> configuration state
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    constraint_field = constraint_field.to(device)
    constraint_field.eval()

    T = dissipation_schedule.T

    # Initialize from thermodynamic equilibrium (isotropic Gaussian)
    x_t = torch.randn(batch_size, channels, image_size, image_size, device=device)

    # Track trajectory if requested
    trajectory = {}
    if save_trajectory and trajectory_steps:
        if 0 in trajectory_steps:
            trajectory[0] = x_t.clone()

    # Reverse process: T -> 0
    for t in range(T, 0, -1):
        # Create timestep tensor
        t_tensor = torch.full((batch_size,), t - 1, device=device, dtype=torch.long)

        # Determine noise scale (sigma=0 for final step)
        if t == 1:
            sigma = 0.0
        else:
            # Use DDPM variance schedule
            alphas_bar_t = dissipation_schedule.alphas_bar[t - 1]
            betas_t = dissipation_schedule.betas[t - 1]
            # Simplified: use beta_t as variance proxy
            sigma = torch.sqrt(betas_t).item()

        # Single reverse step
        x_t = reverse_step(x_t, t_tensor, constraint_field, dissipation_schedule, sigma)

        # Save trajectory snapshot
        if save_trajectory and trajectory_steps and t in trajectory_steps:
            trajectory[t] = x_t.clone()

    # Save final state
    if save_trajectory and trajectory_steps and 0 in trajectory_steps:
        trajectory[0] = x_t.clone()

    return x_t, trajectory
