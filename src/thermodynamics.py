import torch
import numpy as np
from typing import Optional


class DissipationSchedule(torch.nn.Module):
    """
    Manages the thermodynamic relaxation of the signal.

    This class computes the dissipation schedule (beta_t), which controls the rate of
    entropy injection per timestep, driving the system from a structured state
    toward thermodynamic equilibrium (isotropic Gaussian noise).

    Attributes:
        T (int): Total number of diffusion timesteps.
        betas (torch.Tensor): The dissipation rate \beta_t for each timestep.
        alphas (torch.Tensor): The signal retention rate \alpha_t = 1 - \beta_t.
        alphas_bar (torch.Tensor): The remaining signal fraction \\bar{\\alpha}_t = \\prod_{s=1}^t \\alpha_s.
    """

    def __init__(
        self,
        T: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule_type: str = "linear",
    ):
        super().__init__()
        self.T = T

        if schedule_type == "linear":
            betas = torch.linspace(beta_start, beta_end, T)
        elif schedule_type == "cosine":
            # Cosine annealing schedule (Nichol & Dhariwal 2021)
            # Creates smoother dissipation with slower changes at beginning and end
            timesteps = torch.arange(T, dtype=torch.float64)
            betas = beta_end - 0.5 * (beta_end - beta_start) * (
                1 + torch.cos(torch.pi * timesteps / T)
            )
            betas = betas.to(torch.float32)
        else:
            raise NotImplementedError(
                f"Schedule type {schedule_type} is not yet implemented. "
                f"Options: 'linear', 'cosine'"
            )

        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # Register as buffers to ensure they are moved to the correct device
        # and saved in state_dict, but not treated as trainable parameters.
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_bar", alphas_bar)

    def get_params(self, t: torch.Tensor):
        """
        Retrieve schedule parameters for a specific set of timesteps.

        Args:
            t (torch.Tensor): Timesteps (integers from 0 to T-1).

        Returns:
            tuple: (alpha_bar_t, beta_t)
        """
        # t is 0-indexed in tensors, but usually referred to as 1..T in papers.
        return self.alphas_bar[t], self.betas[t]


def forward_sample(
    x_0: torch.Tensor, t: torch.Tensor, dissipation_schedule: DissipationSchedule
) -> torch.Tensor:
    """
    Implements the forward process (thermodynamic relaxation).

    Samples x_t from the closed-form marginal q(x_t | x_0), which describes
    the state of the system after t steps of dissipation.

    The process follows: q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

    Thermodynamic Interpretation:
        alpha_bar_t represents the remaining signal fraction. As t -> T,
        alpha_bar_t -> 0, and the system reaches thermodynamic equilibrium (isotropic Gaussian).

    Args:
        x_0 (torch.Tensor): The initial structured state (original data).
        t (torch.Tensor): The timestep of dissipation.
        dissipation_schedule (DissipationSchedule): The schedule defining entropy injection.

    Returns:
        torch.Tensor: The dissipated configuration state x_t.
    """
    # Ensure t is on the same device as x_0
    t = t.to(x_0.device)

    # Retrieve alpha_bar for the given timesteps
    alphas_bar_t = dissipation_schedule.alphas_bar[t]

    # Reshape for broadcasting: (batch_size,) -> (batch_size, 1, 1, 1) for images
    shape = (alphas_bar_t.shape[0],) + (1,) * (x_0.dim() - 1)
    alphas_bar_t = alphas_bar_t.view(shape)

    # Sample epsilon from N(0, I)
    epsilon = torch.randn_like(x_0)

    # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    x_t = torch.sqrt(alphas_bar_t) * x_0 + torch.sqrt(1 - alphas_bar_t) * epsilon

    return x_t
