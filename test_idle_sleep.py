"""Test that idle sleep is present in retrain.py for LLM agent VRAM unloading."""

import ast


def test_import_time_exists():
    """Verify that 'import time' exists in retrain.py."""
    with open("retrain.py", "r") as f:
        content = f.read()

    assert "import time" in content, "import time not found in retrain.py"


def test_time_sleep_30_exists():
    """Verify that 'time.sleep(30)' exists in retrain.py."""
    with open("retrain.py", "r") as f:
        content = f.read()

    assert "time.sleep(30)" in content, "time.sleep(30) not found in retrain.py"


def test_sleep_before_torch_import():
    """Verify that time.sleep(30) appears before torch import."""
    with open("retrain.py", "r") as f:
        content = f.read()

    sleep_pos = content.find("time.sleep(30)")
    torch_pos = content.find("import torch")

    assert sleep_pos < torch_pos, "time.sleep(30) should appear before import torch"


if __name__ == "__main__":
    test_import_time_exists()
    print("[OK] import time exists")

    test_time_sleep_30_exists()
    print("[OK] time.sleep(30) exists")

    test_sleep_before_torch_import()
    print("[OK] sleep appears before torch import")

    print("\nAll tests passed!")
