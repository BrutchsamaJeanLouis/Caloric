#!/usr/bin/env python3
"""
Visualisation entry point for the thermodynamic diffusion model.

This script creates the primary demonstration output: side-by-side
forward (dissipation) and reverse (formation) trajectories that make
the thermodynamic substrate self-evident.
"""

import argparse
import os
import yaml
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.thermodynamics import DissipationSchedule, forward_sample
from src.constraint_field import ConstraintField
from src.formation import reverse_process
from src.utils import visualise_trajectory


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])


def main():
    parser = argparse.ArgumentParser(
        description="Visualise thermodynamic diffusion trajectories"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cuda, cpu, or auto)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for visualisations",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to visualise",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Visualising on device: {device}")

    # Create output directories
    os.makedirs(os.path.join(args.output_dir, "trajectories"), exist_ok=True)

    # Create dissipation schedule
    dissipation_schedule = DissipationSchedule(
        T=config["T"],
        beta_start=config["beta_start"],
        beta_end=config["beta_end"],
        schedule_type=config["schedule_type"],
    )
    dissipation_schedule = dissipation_schedule.to(device)

    # Create and load constraint field
    constraint_field = ConstraintField(
        base_channels=config["base_channels"],
        channel_multipliers=config["channel_multipliers"],
        num_res_blocks=config["num_res_blocks"],
        attention_resolutions=config["attention_resolutions"],
        dropout=config["dropout"],
        modulation_dim=config["modulation_dim"],
    )
    constraint_field = constraint_field.to(device)

    load_checkpoint(args.checkpoint, constraint_field)
    constraint_field.eval()
    print(f"Loaded checkpoint from: {args.checkpoint}")

    # Load a sample from MNIST for forward trajectory
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2.0 * x - 1.0),
        ]
    )

    dataset = MNIST(root="data", train=True, download=True, transform=transform)
    x_0, _ = dataset[0]
    x_0 = x_0.unsqueeze(0).to(device)  # (1, 1, 28, 28)

    trajectory_steps = config.get("trajectory_steps", [0, 100, 250, 500, 750, 999])

    # Forward process: dissipation trajectory
    print("Computing forward trajectory (dissipation)...")
    forward_traj = {}
    with torch.no_grad():
        for t in trajectory_steps:
            if t == 0:
                forward_traj[0] = x_0.clone()
            else:
                t_tensor = torch.tensor([t - 1], device=device, dtype=torch.long)
                x_t = forward_sample(x_0, t_tensor, dissipation_schedule)
                forward_traj[t] = x_t.clone()

    # Reverse process: formation trajectory
    print("Computing reverse trajectory (formation)...")
    _, reverse_traj = reverse_process(
        constraint_field=constraint_field,
        dissipation_schedule=dissipation_schedule,
        batch_size=1,
        channels=config["channels"],
        image_size=config["image_size"],
        device=device,
        save_trajectory=True,
        trajectory_steps=trajectory_steps,
    )

    # Visualise side-by-side trajectory
    traj_path = os.path.join(args.output_dir, "trajectories", "full_trajectory.png")
    visualise_trajectory(forward_traj, reverse_traj, traj_path, trajectory_steps)
    print(f"Saved full trajectory visualisation to: {traj_path}")

    print("Visualisation complete!")


if __name__ == "__main__":
    main()
