import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import os


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """
    Denormalize tensor from [-1, 1] to [0, 1] range for visualization.
    """
    return (x + 1) / 2


def save_grid(
    samples: torch.Tensor,
    save_path: str,
    normalize: bool = True,
) -> None:
    """
    Save a grid of generated samples to disk.

    Args:
        samples (torch.Tensor): Generated samples (B, C, H, W).
        save_path (str): Path to save the grid image.
        normalize (bool): Whether to denormalize from [-1, 1] to [0, 1].
    """
    if normalize:
        samples = denormalize(samples)

    # Ensure values are in [0, 1]
    samples = torch.clamp(samples, 0, 1)

    # Convert to numpy
    samples_np = samples.cpu().numpy()

    # Create grid
    grid = make_grid(samples_np, nrow=8, padding=2)

    # Save
    plt.imsave(save_path, grid, cmap="gray")


def make_grid(images: np.ndarray, nrow: int = 8, padding: int = 2) -> np.ndarray:
    """
    Create a grid of images for visualization.

    Args:
        images (np.ndarray): Images (B, C, H, W) or (B, H, W).
        nrow (int): Number of columns in the grid.
        padding (int): Padding between images.

    Returns:
        np.ndarray: Grid image.
    """
    if images.ndim == 3:
        images = images[:, None, :, :]  # Add channel dimension

    batch_size, channels, height, width = images.shape
    ncols = nrow
    nrows = (batch_size + ncols - 1) // ncols

    grid_height = nrows * height + (nrows - 1) * padding
    grid_width = ncols * width + (ncols - 1) * padding

    if channels == 1:
        grid = np.zeros((grid_height, grid_width), dtype=np.float32)
    else:
        grid = np.zeros((grid_height, grid_width, channels), dtype=np.float32)

    for i, img in enumerate(images):
        row = i // ncols
        col = i % ncols

        y_start = row * (height + padding)
        x_start = col * (width + padding)

        if channels == 1:
            grid[y_start : y_start + height, x_start : x_start + width] = img
        else:
            grid[y_start : y_start + height, x_start : x_start + width] = img

    return grid


def visualise_trajectory(
    forward_traj: Dict[int, torch.Tensor],
    reverse_traj: Dict[int, torch.Tensor],
    save_path: str,
    trajectory_steps: List[int],
) -> None:
    """
    Visualize forward (dissipation) and reverse (formation) trajectories side by side.

    This is the primary demonstration output that makes the thermodynamic substrate
    self-evident: showing structure relaxing into equilibrium and back.

    Args:
        forward_traj (Dict[int, Tensor]): Forward process snapshots {timestep: x_t}.
        reverse_traj (Dict[int, Tensor]): Reverse process snapshots {timestep: x_t}.
        save_path (str): Path to save the visualization.
        trajectory_steps (List[int]): Timesteps to visualize.
    """
    fig, axes = plt.subplots(
        2, len(trajectory_steps), figsize=(4 * len(trajectory_steps), 8)
    )

    if len(trajectory_steps) == 1:
        axes = axes.reshape(-1, 1)

    # Top row: forward process (dissipation)
    for col, t in enumerate(trajectory_steps):
        ax = axes[0, col]
        if t in forward_traj:
            x_t = forward_traj[t].cpu().squeeze()
            x_t = denormalize(x_t).clamp(0, 1)
            ax.imshow(x_t, cmap="gray")
        ax.set_title(f"t={t}\n(Dissipation)")
        ax.axis("off")

    # Bottom row: reverse process (formation)
    for col, t in enumerate(trajectory_steps):
        ax = axes[1, col]
        if t in reverse_traj:
            x_t = reverse_traj[t].cpu().squeeze()
            x_t = denormalize(x_t).clamp(0, 1)
            ax.imshow(x_t, cmap="gray")
        ax.set_title(f"t={t}\n(Formation)")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def log_metrics(
    epoch_losses: List[float],
    save_path: str,
) -> None:
    """
    Save training loss curve to disk.

    Args:
        epoch_losses (List[float]): Loss values per epoch.
        save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    loss: float,
    save_path: str,
) -> None:
    """
    Save training checkpoint.

    Args:
        model (torch.nn.Module): The model.
        optimizer (torch.optim.Optimizer, optional): The optimizer.
        epoch (int): Current epoch.
        loss (float): Current loss.
        save_path (str): Path to save the checkpoint.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "loss": loss,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(checkpoint, save_path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict:
    """
    Load training checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to restore.

    Returns:
        dict: Checkpoint data.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def save_trajectory_animation(
    forward_traj: Dict[int, torch.Tensor],
    reverse_traj: Dict[int, torch.Tensor],
    save_path: str,
    trajectory_steps: List[int],
    fps: int = 10,
) -> None:
    """
    Create an animated GIF showing forward (dissipation) and reverse (formation) trajectories.

    Thermodynamic Interpretation:
        This animation visualizes the complete thermodynamic cycle: structured signal
        relaxing toward equilibrium (forward) and structure emerging from equilibrium
        via learned energy-gradient traversal (reverse).

    Args:
        forward_traj (Dict[int, Tensor]): Forward process snapshots {timestep: x_t}.
        reverse_traj (Dict[int, Tensor]): Reverse process snapshots {timestep: x_t}.
        save_path (str): Path to save the GIF.
        trajectory_steps (List[int]): Timesteps to animate (must be sorted ascending).
        fps (int): Frames per second for the animation.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    num_steps = len(trajectory_steps)
    fig, axes = plt.subplots(2, 1, figsize=(6, 6))

    im_fwd = None
    if trajectory_steps[0] in forward_traj:
        x_t = forward_traj[trajectory_steps[0]].cpu().squeeze()
        x_t = denormalize(x_t).clamp(0, 1)
        im_fwd = axes[0].imshow(x_t, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Forward Process (Dissipation)", fontsize=12)
    axes[0].axis("off")

    im_rev = None
    if trajectory_steps[0] in reverse_traj:
        x_t = reverse_traj[trajectory_steps[0]].cpu().squeeze()
        x_t = denormalize(x_t).clamp(0, 1)
        im_rev = axes[1].imshow(x_t, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Reverse Process (Formation)", fontsize=12)
    axes[1].axis("off")

    plt.tight_layout()

    def update(frame_idx):
        t = trajectory_steps[frame_idx]

        if t in forward_traj and im_fwd is not None:
            x_t = forward_traj[t].cpu().squeeze()
            x_t = denormalize(x_t).clamp(0, 1)
            im_fwd.set_data(x_t)

        if t in reverse_traj and im_rev is not None:
            x_t = reverse_traj[t].cpu().squeeze()
            x_t = denormalize(x_t).clamp(0, 1)
            im_rev.set_data(x_t)

        axes[0].set_title(f"Dissipation t={t}", fontsize=12)
        axes[1].set_title(f"Formation t={t}", fontsize=12)

        return [img for img in [im_fwd, im_rev] if img is not None]

    anim = FuncAnimation(fig, update, frames=num_steps, interval=1000 / fps, blit=True)
    anim.save(save_path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close()
