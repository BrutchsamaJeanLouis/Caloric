#!/usr/bin/env python3
"""
Generation entry point for the thermodynamic diffusion model.

This script runs the reverse process (formation) to generate samples
from thermodynamic equilibrium using a trained constraint field.
"""

import argparse
import os
import yaml
import torch

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.thermodynamics import DissipationSchedule
from src.constraint_field import ConstraintField
from src.formation import reverse_process
from src.utils import save_grid


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])


def main():
    parser = argparse.ArgumentParser(
        description="Generate samples from thermodynamic diffusion model"
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
        "--num-samples",
        type=int,
        default=None,
        help="Override number of samples to generate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for generated samples",
    )
    parser.add_argument(
        "--save-trajectory",
        action="store_true",
        help="Save intermediate trajectory states",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override with command-line arguments
    if args.num_samples is not None:
        config["num_samples"] = args.num_samples

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Generating on device: {device}")

    # Create output directories
    os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)
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

    # Generate samples via reverse process (formation)
    trajectory_steps = config.get("trajectory_steps", [0, 100, 250, 500, 750, 999])

    samples, trajectory = reverse_process(
        constraint_field=constraint_field,
        dissipation_schedule=dissipation_schedule,
        batch_size=config["num_samples"],
        channels=config["channels"],
        image_size=config["image_size"],
        device=device,
        save_trajectory=args.save_trajectory,
        trajectory_steps=trajectory_steps,
    )

    # Save sample grid
    grid_path = os.path.join(args.output_dir, "samples", "generated_grid.png")
    save_grid(samples, grid_path)
    print(f"Saved sample grid to: {grid_path}")

    # Save trajectory if requested
    if args.save_trajectory and trajectory:
        traj_path = os.path.join(
            args.output_dir, "trajectories", "formation_trajectory.png"
        )
        from src.utils import visualise_trajectory

        # Create empty forward trajectory for visualization
        forward_traj = {}
        visualise_trajectory(forward_traj, trajectory, traj_path, trajectory_steps)
        print(f"Saved formation trajectory to: {traj_path}")

    print("Generation complete!")


if __name__ == "__main__":
    main()
