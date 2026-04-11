import torch
from src.constraint_field import ConstraintField
import yaml


def verify_constraint_field():
    # Load hyperparameters from configs/default.yaml
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize model
    model = ConstraintField(
        base_channels=config["base_channels"],
        channel_multipliers=config["channel_multipliers"],
        num_res_blocks=config["num_res_blocks"],
        attention_resolutions=config["attention_resolutions"],
        dropout=config["dropout"],
        modulation_dim=config["modulation_dim"],
    )

    # Dummy input: (B, 1, 28, 28) and time (B,)
    B = 4
    x = torch.randn(B, 1, 28, 28)
    t = torch.randint(0, 1000, (B,)).float()

    try:
        output = model(x, t)
        print(f"Input shape: {x.shape}")
        print(f"Time shape: {t.shape}")
        print(f"Output shape: {output.shape}")

        assert output.shape == (B, 1, 28, 28), (
            f"Expected (B, 1, 28, 28), got {output.shape}"
        )
        print("Verification SUCCESS: Output shape is correct.")
    except Exception as e:
        print(f"Verification FAILED: {e}")
        raise e


if __name__ == "__main__":
    verify_constraint_field()
