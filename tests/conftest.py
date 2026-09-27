import pytest
import torch

from mini_dataloader.data.dataset import FashionMNISTDataset


@pytest.fixture
def create_fashion_mnist_dataset():
    X = torch.rand(10, 28, 28)
    y = torch.randint(0, 10, (10,))
    return FashionMNISTDataset(X, y)
