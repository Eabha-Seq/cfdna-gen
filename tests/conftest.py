"""Pytest configuration and fixtures."""

import pytest
import torch


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seeds for reproducibility."""
    torch.manual_seed(42)
    yield


@pytest.fixture
def device():
    """Return the test device (CPU for CI, GPU if available locally)."""
    return "cpu"
