"""
Shared fixtures for student model tests.

Provides:
    device: CPU device (tests are CPU-only)
    batch_input: Synthetic (2, 3, 640, 640) random tensor
    student_model: StudentModel instance in eval mode on CPU
"""

import pytest
import torch

from src.models.student.student_model import StudentModel


@pytest.fixture
def device():
    """CPU device — all student tests run on CPU."""
    return torch.device("cpu")


@pytest.fixture
def batch_input(device):
    """Synthetic batch input: (2, 3, 640, 640) random tensor on CPU."""
    return torch.randn(2, 3, 640, 640, device=device)


@pytest.fixture
def student_model(device):
    """StudentModel in eval mode on CPU."""
    model = StudentModel()
    model.eval()
    model.to(device)
    return model
