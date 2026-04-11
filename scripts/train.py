#!/usr/bin/env python3
"""
Training entry point for the thermodynamic diffusion model.

This script trains the constraint field (U-Net) on MNIST to learn the
entropy gradient for reversing the dissipation process.
"""

import argparse
import os
import yaml
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.thermodynamics import DissipationSchedule
from src.constraint_field import ConstraintField
from src.training import train
from src.utils import save_checkpoint, log_metrics


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train thermodynamic diffusion model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cuda, cpu, or auto)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for checkpoints and logs",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Convert numeric config values to proper types (YAML may read them as strings)
    config["learning_rate"] = float(config["learning_rate"])
    config["ema_decay"] = float(config["ema_decay"])
    config["gradient_clip"] = float(config["gradient_clip"])
    config["dropout"] = float(config["dropout"])

    # Override config with command-line arguments
    if args.epochs is not None:
        config["num_epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Training on device: {device}")

    # Create output directories
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)

    # Set seed for reproducibility
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed(config["seed"])

    # Create dissipation schedule
    dissipation_schedule = DissipationSchedule(
        T=config["T"],
        beta_start=config["beta_start"],
        beta_end=config["beta_end"],
        schedule_type=config["schedule_type"],
    )
    dissipation_schedule = dissipation_schedule.to(device)

    # Create constraint field (U-Net)
    constraint_field = ConstraintField(
        base_channels=config["base_channels"],
        channel_multipliers=config["channel_multipliers"],
        num_res_blocks=config["num_res_blocks"],
        attention_resolutions=config["attention_resolutions"],
        dropout=config["dropout"],
        modulation_dim=config["modulation_dim"],
    )
    constraint_field = constraint_field.to(device)

    # Create data loader
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2.0 * x - 1.0),  # Normalize to [-1, 1]
        ]
    )

    train_dataset = MNIST(root="data", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Number of training batches: {len(train_loader)}")

    # Train
    constraint_field, epoch_losses = train(
        constraint_field=constraint_field,
        dissipation_schedule=dissipation_schedule,
        train_loader=train_loader,
        num_epochs=config["num_epochs"],
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
        gradient_clip=config["gradient_clip"],
        ema_decay=config["ema_decay"],
        device=device,
    )

    # Save final checkpoint
    checkpoint_path = os.path.join(
        args.output_dir, "checkpoints", "final_checkpoint.pth"
    )
    save_checkpoint(
        constraint_field,
        None,  # Optimizer not needed for inference
        config["num_epochs"],
        epoch_losses[-1],
        checkpoint_path,
    )
    print(f"Saved final checkpoint to: {checkpoint_path}")

    # Save loss curve
    loss_curve_path = os.path.join(args.output_dir, "logs", "loss_curve.png")
    log_metrics(epoch_losses, loss_curve_path)
    print(f"Saved loss curve to: {loss_curve_path}")

    print("Training complete!")


if __name__ == "__main__":
    main()
