#!/usr/bin/env python3
import argparse
import os
import yaml
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.thermodynamics import DissipationSchedule, forward_sample
from src.constraint_field import ConstraintField
from src.formation import reverse_process


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])


def create_schedule_comparison(
    checkpoint: str,
    output_path: str,
    device: torch.device,
):
    config = load_config("configs/default.yaml")

    linear_schedule = DissipationSchedule(
        T=config["T"],
        beta_start=config["beta_start"],
        beta_end=config["beta_end"],
        schedule_type="linear",
    ).to(device)

    cosine_schedule = DissipationSchedule(
        T=config["T"],
        beta_start=config["beta_start"],
        beta_end=config["beta_end"],
        schedule_type="cosine",
    ).to(device)

    model = ConstraintField(
        base_channels=config["base_channels"],
        channel_multipliers=config["channel_multipliers"],
        num_res_blocks=config["num_res_blocks"],
        attention_resolutions=config["attention_resolutions"],
        dropout=config["dropout"],
        modulation_dim=config["modulation_dim"],
    ).to(device)
    load_checkpoint(checkpoint, model)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2.0 * x - 1.0),
        ]
    )
    dataset = MNIST(root="data", train=True, download=True, transform=transform)
    x_0, _ = dataset[0]
    x_0 = x_0.unsqueeze(0).to(device)

    trajectory_steps = [0, 100, 250, 500, 750, 999]

    linear_forward = {}
    with torch.no_grad():
        for t in trajectory_steps:
            if t == 0:
                linear_forward[t] = x_0.clone()
            else:
                t_tensor = torch.tensor([t - 1], device=device, dtype=torch.long)
                x_t = forward_sample(x_0, t_tensor, linear_schedule)
                linear_forward[t] = x_t.clone()

    _, linear_reverse = reverse_process(
        constraint_field=model,
        dissipation_schedule=linear_schedule,
        batch_size=1,
        channels=config["channels"],
        image_size=config["image_size"],
        device=device,
        save_trajectory=True,
        trajectory_steps=trajectory_steps,
    )

    cosine_forward = {}
    with torch.no_grad():
        for t in trajectory_steps:
            if t == 0:
                cosine_forward[t] = x_0.clone()
            else:
                t_tensor = torch.tensor([t - 1], device=device, dtype=torch.long)
                x_t = forward_sample(x_0, t_tensor, cosine_schedule)
                cosine_forward[t] = x_t.clone()

    _, cosine_reverse = reverse_process(
        constraint_field=model,
        dissipation_schedule=cosine_schedule,
        batch_size=1,
        channels=config["channels"],
        image_size=config["image_size"],
        device=device,
        save_trajectory=True,
        trajectory_steps=trajectory_steps,
    )

    fig, axes = plt.subplots(4, 6, figsize=(18, 12))

    for i, t in enumerate(trajectory_steps):
        img = linear_forward[t].cpu().squeeze().clamp(-1, 1).numpy()
        axes[0, i].imshow(img, cmap="gray")
        axes[0, i].set_title(f"t={t}")
        axes[0, i].axis("off")
    axes[0, 0].set_ylabel("Linear Forward", fontsize=12, fontweight="bold")

    for i, t in enumerate(trajectory_steps):
        img = linear_reverse[t].cpu().squeeze().clamp(-1, 1).numpy()
        axes[1, i].imshow(img, cmap="gray")
        axes[1, i].axis("off")
    axes[1, 0].set_ylabel("Linear Reverse", fontsize=12, fontweight="bold")

    for i, t in enumerate(trajectory_steps):
        img = cosine_forward[t].cpu().squeeze().clamp(-1, 1).numpy()
        axes[2, i].imshow(img, cmap="gray")
        axes[2, i].axis("off")
    axes[2, 0].set_ylabel("Cosine Forward", fontsize=12, fontweight="bold")

    for i, t in enumerate(trajectory_steps):
        img = cosine_reverse[t].cpu().squeeze().clamp(-1, 1).numpy()
        axes[3, i].imshow(img, cmap="gray")
        axes[3, i].axis("off")
    axes[3, 0].set_ylabel("Cosine Reverse", fontsize=12, fontweight="bold")

    plt.suptitle(
        "Dissipation Schedule Comparison: Linear vs Cosine",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Comparison saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare linear vs cosine schedules")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/trained_30epochs.pth",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/trajectories/schedule_comparison.png",
        help="Output path for comparison image",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cuda, cpu, or auto)",
    )

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Creating comparison on device: {device}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    create_schedule_comparison(
        checkpoint=args.checkpoint,
        output_path=args.output,
        device=device,
    )

    print("Done!")


if __name__ == "__main__":
    main()
