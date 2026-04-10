"""Tests for cosine dissipation schedule implementation."""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.thermodynamics import DissipationSchedule


def test_cosine_schedule_exists():
    """Test that cosine schedule can be created."""
    schedule = DissipationSchedule(schedule_type="cosine")
    assert schedule is not None
    assert schedule.T == 1000
    print("[PASS] Cosine schedule created successfully")


def test_cosine_beta_bounds():
    """Test that beta values are within expected bounds."""
    beta_start = 0.0001
    beta_end = 0.02
    schedule = DissipationSchedule(
        T=1000, beta_start=beta_start, beta_end=beta_end, schedule_type="cosine"
    )

    min_beta = schedule.betas.min().item()
    max_beta = schedule.betas.max().item()

    assert min_beta >= beta_start - 1e-9, f"Min beta {min_beta} < {beta_start}"
    assert max_beta <= beta_end + 1e-9, f"Max beta {max_beta} > {beta_end}"
    print(
        f"[PASS] Beta bounds: [{min_beta:.6f}, {max_beta:.6f}] within [{beta_start}, {beta_end}]"
    )


def test_cosine_monotonic_increase():
    """Test that beta values monotonically increase."""
    schedule = DissipationSchedule(schedule_type="cosine")

    # Check that each beta is greater than the previous
    is_monotonic = (schedule.betas[1:] > schedule.betas[:-1]).all().item()
    assert is_monotonic, "Beta values should monotonically increase"
    print("[PASS] Beta values monotonically increase")


def test_cosine_smoothness():
    """Test that beta changes are smooth (no large jumps)."""
    schedule = DissipationSchedule(schedule_type="cosine")

    # Compute differences between consecutive betas
    diffs = torch.diff(schedule.betas)

    # Max difference should be reasonable (< 0.001 for T=1000)
    max_diff = diffs.max().item()
    assert max_diff < 0.001, f"Max beta difference {max_diff} too large"
    print(f"[PASS] Beta changes are smooth (max diff: {max_diff:.6f})")


def test_linear_vs_cosine():
    """Compare linear and cosine schedules."""
    linear = DissipationSchedule(schedule_type="linear")
    cosine = DissipationSchedule(schedule_type="cosine")

    # Both should have same bounds
    assert torch.allclose(linear.betas[0], cosine.betas[0], atol=1e-6)
    assert torch.allclose(linear.betas[-1], cosine.betas[-1], atol=1e-6)

    # But different distributions
    linear_mid = linear.betas[500].item()
    cosine_mid = cosine.betas[500].item()

    print(f"[PASS] Linear mid (t=500): {linear_mid:.6f}")
    print(f"[PASS] Cosine mid (t=500): {cosine_mid:.6f}")
    print(
        f"  Cosine is {'slower' if cosine_mid < linear_mid else 'faster'} at midpoint"
    )


def test_alphas_and_alphas_bar():
    """Test that alphas and alphas_bar are correctly computed."""
    schedule = DissipationSchedule(schedule_type="cosine")

    # alphas = 1 - betas
    expected_alphas = 1.0 - schedule.betas
    assert torch.allclose(schedule.alphas, expected_alphas)

    # alphas_bar should be cumulative product
    expected_alphas_bar = torch.cumprod(schedule.alphas, dim=0)
    assert torch.allclose(schedule.alphas_bar, expected_alphas_bar)

    # alphas_bar should decrease from ~1 to ~0
    assert schedule.alphas_bar[0].item() > 0.99
    assert schedule.alphas_bar[-1].item() < 0.01

    print(
        f"[PASS] Alphas bar: [{schedule.alphas_bar[0].item():.4f}, ..., {schedule.alphas_bar[-1].item():.4f}]"
    )


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Cosine Dissipation Schedule")
    print("=" * 60)

    test_cosine_schedule_exists()
    test_cosine_beta_bounds()
    test_cosine_monotonic_increase()
    test_cosine_smoothness()
    test_linear_vs_cosine()
    test_alphas_and_alphas_bar()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
