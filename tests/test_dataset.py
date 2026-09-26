import pytest
import torch

from mini_dataloader.data.dataset import FashionMNISTDataset


# Fixture for creating a FashionMNISTDataset instance
@pytest.fixture
def create_fashion_mnist_dataset():
    X = torch.rand(10, 28, 28)
    y = torch.randint(0, 10, (10,))
    return FashionMNISTDataset(X, y)


# Test cases for Dataset class
@pytest.mark.parametrize("index", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_dataset_getitem(create_fashion_mnist_dataset, index):
    dataset = create_fashion_mnist_dataset
    item = dataset.__getitem__(index)
    assert isinstance(item, tuple)
    assert isinstance(item[0], torch.Tensor)
    assert isinstance(item[1], torch.Tensor)
    assert item[0].shape == (28, 28)
    assert item[1].shape == ()


def test_dataset_len(create_fashion_mnist_dataset):
    dataset = create_fashion_mnist_dataset
    assert len(dataset) == 10


def test_dataset_keys(create_fashion_mnist_dataset):
    dataset = create_fashion_mnist_dataset
    keys = dataset.keys()
    assert isinstance(keys, list)
    assert all(isinstance(k, int) for k in keys)
    assert len(keys) == 10


# Test cases for FashionMNISTDataset class
def test_fashion_mnist_dataset_init():
    X = torch.rand(10, 28, 28)
    y = torch.randint(0, 10, (10,))
    dataset = FashionMNISTDataset(X, y)
    assert isinstance(dataset.X, torch.Tensor)
    assert isinstance(dataset.y, torch.Tensor)
    assert dataset.X.shape == (10, 28, 28)
    assert dataset.y.shape == (10,)


def test_fashion_mnist_dataset_getitem(create_fashion_mnist_dataset):
    dataset = create_fashion_mnist_dataset
    item = dataset.__getitem__(0)
    assert isinstance(item, tuple)
    assert isinstance(item[0], torch.Tensor)
    assert isinstance(item[1], torch.Tensor)
    assert item[0].shape == (28, 28)
    assert item[1].shape == ()


def test_fashion_mnist_dataset_len(create_fashion_mnist_dataset):
    dataset = create_fashion_mnist_dataset
    assert len(dataset) == 10


def test_fashion_mnist_dataset_keys(create_fashion_mnist_dataset):
    dataset = create_fashion_mnist_dataset
    keys = dataset.keys()
    assert isinstance(keys, list)
    assert all(isinstance(k, int) for k in keys)
    assert len(keys) == 10
