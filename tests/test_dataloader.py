import pytest
import torch

from mini_dataloader.data.dataloader import DataLoader
from mini_dataloader.data.dataset import FashionMNISTDataset
from mini_dataloader.data.sampler import DefaultSampler


# Fixture for creating a FashionMNISTDataset instance
@pytest.fixture
def create_fashion_mnist_dataset():
    X = torch.rand(10, 28, 28)
    y = torch.randint(0, 10, (10,))
    return FashionMNISTDataset(X, y)


# Fixture for creating a DefaultSampler instance
@pytest.fixture
def create_default_sampler(create_fashion_mnist_dataset):
    return DefaultSampler(
        indices=list(range(len(create_fashion_mnist_dataset))),
        shuffle=True,
        batch_size=3,
    )


# Fixture for creating a DataLoader instance
@pytest.fixture
def create_dataloader(create_fashion_mnist_dataset, create_default_sampler):
    return DataLoader(
        dataset=create_fashion_mnist_dataset,
        batch_size=3,
        shuffle=True,
        sampler=create_default_sampler,
    )


# Test cases for DataLoader class
def test_dataloader_init(create_fashion_mnist_dataset, create_default_sampler):
    dataloader = DataLoader(
        dataset=create_fashion_mnist_dataset,
        batch_size=3,
        shuffle=False,
        sampler=create_default_sampler,
    )
    assert dataloader.dataset == create_fashion_mnist_dataset
    assert dataloader.batch_size == 3
    assert isinstance(dataloader.sampler, DefaultSampler)


def test_dataloader_iter(create_dataloader):
    dataloader = create_dataloader
    iterator = iter(dataloader)
    batches = list(iterator)
    assert len(batches) == 4
    assert all(len(batch[0]) == 3 and len(batch[1]) == 3 for batch in batches[:-1])
    assert 1 <= len(batches[-1][0]) <= 3
    assert len(batches[-1][0]) == len(batches[-1][1])


def test_dataloader_len(create_dataloader):
    dataloader = create_dataloader
    assert len(dataloader) == 4


def test_dataloader_shuffle(create_dataloader):
    dataloader = create_dataloader
    iterator = iter(dataloader)
    batches = list(iterator)
    assert len(batches) == 4
    assert all(len(batch[0]) == 3 and len(batch[1]) == 3 for batch in batches[:-1])
    assert 1 <= len(batches[-1][0]) <= 3
    assert len(batches[-1][0]) == len(batches[-1][1])
