import time

time.sleep(30)  # Allow LLM agents to unload and free VRAM before training
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.constraint_field import ConstraintField
from src.thermodynamics import DissipationSchedule
from src.training import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1),
    ]
)
train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)

model = ConstraintField(
    base_channels=64,
    channel_multipliers=[1, 2, 4],
    num_res_blocks=2,
    attention_resolutions=[14],
    dropout=0.1,
    modulation_dim=256,
)
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

ds = DissipationSchedule(
    T=1000, beta_start=0.0001, beta_end=0.02, schedule_type="linear"
)

trained_model, losses = train(
    constraint_field=model,
    dissipation_schedule=ds,
    train_loader=train_loader,
    num_epochs=30,
    learning_rate=2e-4,
    batch_size=128,
    gradient_clip=1.0,
    ema_decay=0.9999,
    device=device,
    log_interval=100,
)

torch.save(
    {
        "model_state_dict": trained_model.state_dict(),
        "epoch": 30,
        "loss": losses[-1],
    },
    "outputs/checkpoints/trained_30epochs.pth",
)

print(f"Final loss: {losses[-1]:.6f}")
print("Saved to outputs/checkpoints/trained_30epochs.pth")
