import torch
from src.thermodynamics import DissipationSchedule, forward_sample


def verify_thermodynamics():
    # 1. Setup
    T = 1000
    schedule = DissipationSchedule(T=T)

    # Create a fake "digit" x_0: (Batch=1000, C=1, H=28, W=28)
    # We use a value of 1.0 to clearly see the dissipation
    x_0 = torch.ones((1000, 1, 28, 28))

    # 2. Sample x_T (Equilibrium state)
    # t = T-1 is the final timestep
    t = torch.full((1000,), T - 1, dtype=torch.long)
    x_T = forward_sample(x_0, t, schedule)

    # 3. Verify distribution of x_T
    # At T=1000, x_T should be approximately N(0, I)
    mean = x_T.mean().item()
    var = x_T.var().item()

    print(f"Verification of x_T (T={T}):")
    print(f"Mean: {mean:.4f} (Expected: ~0.0)")
    print(f"Variance: {var:.4f} (Expected: ~1.0)")

    # Assertions with tolerance
    assert abs(mean) < 0.1, f"Mean {mean} is too far from 0"
    assert abs(var - 1.0) < 0.2, f"Variance {var} is too far from 1"
    print("Verification SUCCESS: x_T is approximately N(0, I)")


if __name__ == "__main__":
    verify_thermodynamics()
