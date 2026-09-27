from unittest.mock import patch

import pytest
import torch

from mini_dataloader.data.dataset import Dataset, FashionMNISTDataset, datasetFactory


@pytest.fixture
def mock_fashion_mnist():
    with patch("torchvision.datasets.FashionMNIST") as mock_fashion:
        yield mock_fashion


@pytest.fixture
def sample_data():
    X = torch.rand(10, 28, 28)
    y = torch.randint(0, 10, (10,))
    return X, y


# Test cases for Dataset class
def test_dataset_abstract_methods():
    with pytest.raises(TypeError):
        Dataset()  # Attempting to instantiate abstract class


# Test cases for FashionMNISTDataset class
def test_fashion_mnist_dataset_init(sample_data):
    X, y = sample_data
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


# Test cases for datasetFactory function
def test_datasetFactory_happy_path(mock_fashion_mnist, sample_data):
    X, y = sample_data
    mock_fashion_mnist.return_value = [(X[i], y[i]) for i in range(len(X))]

    ds = datasetFactory("fashion", True)
    assert isinstance(ds, FashionMNISTDataset)
    assert torch.all(ds.X == X)
    assert torch.all(ds.y == y)


def test_datasetFactory_wrong_name(mock_fashion_mnist):
    with pytest.raises(ValueError):
        datasetFactory("wrong_name", True)


def test_datasetFactory_wrong_train_value(mock_fashion_mnist):
    with pytest.raises(TypeError):
        datasetFactory("fashion", "wrong_value")


def test_datasetFactory_not_train(mock_fashion_mnist, sample_data):
    X, y = sample_data
    mock_fashion_mnist.return_value = [(X[i], y[i]) for i in range(len(X))]

    ds = datasetFactory("fashion", False)
    assert isinstance(ds, FashionMNISTDataset)
    assert torch.all(ds.X == X)
    assert torch.all(ds.y == y)


def test_datasetFactory_transform(mock_fashion_mnist, sample_data):
    X, y = sample_data
    mock_fashion_mnist.return_value = [(X[i], y[i]) for i in range(len(X))]

    ds = datasetFactory("fashion", True)
    assert isinstance(ds, FashionMNISTDataset)
    assert torch.all(ds.X == X)
    assert torch.all(ds.y == y)
