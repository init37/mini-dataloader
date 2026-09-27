import pytest
from torch import nn, optim

from mini_dataloader.factory.optimizer import optimizerFactory


# Fixture for creating a mock model
@pytest.fixture
def mock_model():
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(10, 10)

        def forward(self, x):
            return self.layer(x)

    return MockModel()


# Test cases for optimizerFactory function
def test_optimizerFactory_happy_path(mock_model):
    optimizer = optimizerFactory(mock_model, 0.01, "adam")
    assert isinstance(optimizer, optim.Adam)
    assert optimizer.defaults["lr"] == 0.01


def test_optimizerFactory_wrong_model_type():
    with pytest.raises(TypeError):
        optimizerFactory(None, 0.01, "adam")


def test_optimizerFactory_wrong_lr_type():
    with pytest.raises(TypeError):
        optimizerFactory(nn.Linear(10, 10), "0.01", "adam")


def test_optimizerFactory_wrong_optimizer_name():
    with pytest.raises(ValueError):
        optimizerFactory(nn.Linear(10, 10), 0.01, "sgd")
