import torch
from typing import Optional, Tuple
from tqdm import tqdm
import copy

from .thermodynamics import DissipationSchedule, forward_sample


def compute_loss(
    x_0: torch.Tensor,
    constraint_field: torch.nn.Module,
    dissipation_schedule: DissipationSchedule,
) -> torch.Tensor:
    """
    Compute the L_simple training objective (noise prediction loss).

    The training objective is:
        L_simple = E_{t, x_0, ε} [ ‖ε - ε_θ(x_t, t)‖² ]

    where:
        t ~ Uniform({1, ..., T})
        ε ~ N(0, I)
        x_t = √(ᾱ_t) · x_0 + √(1 - ᾱ_t) · ε

    Thermodynamic Interpretation:
        We train the constraint field to learn the exact entropy gradient
        at every point along the dissipation trajectory. By minimizing the
        error between predicted and actual noise, the network learns to
        estimate the score function ∇_x log p(x_t), enabling reverse
        traversal of the energy landscape.

    Args:
        x_0 (torch.Tensor): Original data (B, C, H, W).
        constraint_field (torch.nn.Module): The energy landscape mapper.
        dissipation_schedule (DissipationSchedule): The dissipation schedule.

    Returns:
        torch.Tensor: The mean squared error loss.
    """
    batch_size = x_0.shape[0]
    T = dissipation_schedule.T

    # Sample timesteps uniformly
    t = torch.randint(1, T + 1, (batch_size,), device=x_0.device)
    t = t - 1  # Convert to 0-indexed

    # Sample noise
    epsilon = torch.randn_like(x_0)

    # Forward process: get x_t from x_0
    x_t = forward_sample(x_0, t, dissipation_schedule)

    # Predict noise
    epsilon_pred = constraint_field(x_t, t)

    # Compute MSE loss
    loss = torch.mean((epsilon - epsilon_pred) ** 2)

    return loss


class EMA:
    """
    Exponential Moving Average for stable generation.

    Maintains a running average of model parameters to improve sample quality.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow_params = {
            name: param.clone().detach() for name, param in model.named_parameters()
        }

    def update(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            self.shadow_params[name] = (
                self.decay * self.shadow_params[name]
                + (1 - self.decay) * param.detach()
            )

    def copy_to(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow_params[name])

    def store(self, model: torch.nn.Module):
        self.backup = {name: param.clone() for name, param in model.named_parameters()}

    def restore(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            param.data.copy_(self.backup[name])


def train(
    constraint_field: torch.nn.Module,
    dissipation_schedule: DissipationSchedule,
    train_loader: torch.utils.data.DataLoader,
    num_epochs: int,
    learning_rate: float,
    batch_size: int,
    gradient_clip: float,
    ema_decay: float,
    device: Optional[torch.device] = None,
    log_interval: int = 100,
) -> Tuple[torch.nn.Module, list]:
    """
    Train the constraint field on the dissipation reversal task.

    Thermodynamic Interpretation:
        Training teaches the constraint field to estimate the entropy gradient
        (score function) at every configuration state along the dissipation
        trajectory. Once learned, this enables energy-gradient descent from
        equilibrium back to structured data.

    Args:
        constraint_field (torch.nn.Module): The energy landscape mapper.
        dissipation_schedule (DissipationSchedule): The dissipation schedule.
        train_loader (DataLoader): Training data loader.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for Adam optimizer.
        batch_size (int): Batch size.
        gradient_clip (float): Gradient clipping value.
        ema_decay (float): EMA decay rate.
        device (torch.device): Device to train on.
        log_interval (int): Print log every N iterations.

    Returns:
        Tuple[torch.nn.Module, list]:
            - Trained constraint field (with EMA weights applied)
            - List of average losses per epoch
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    constraint_field = constraint_field.to(device)
    dissipation_schedule = dissipation_schedule.to(device)

    optimizer = torch.optim.Adam(constraint_field.parameters(), lr=learning_rate)
    ema = EMA(constraint_field, ema_decay)

    epoch_losses = []

    for epoch in range(num_epochs):
        constraint_field.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            leave=False,
        )

        for batch_idx, (x, _) in enumerate(pbar):
            x = x.to(device)
            # Data should already be normalized to [-1, 1] in the DataLoader transform

            optimizer.zero_grad()
            loss = compute_loss(x, constraint_field, dissipation_schedule)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(constraint_field.parameters(), gradient_clip)
            optimizer.step()

            ema.update(constraint_field)

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % log_interval == 0:
                pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / num_batches
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.6f}")

    # Apply EMA weights to the final model
    ema.copy_to(constraint_field)

    return constraint_field, epoch_losses
