import pytest
import torch

from data.dataloader import DataLoader
from data.dataset import Dataset
from data.sampler import BatchSampler


class TensorDataset(Dataset[int]):
    def __init__(self, data: torch.Tensor) -> None:
        self.data = data

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    def keys(self) -> list[int]:
        return list(range(self.data.shape[0]))


def test_dataloader_returns_batches() -> None:
    dataset = TensorDataset(torch.tensor([1, 2, 3, 4]))

    loader = DataLoader(
        dataset,
        batch_size=2,
    )

    batches = list(loader)

    assert len(batches) == 2
    assert torch.equal(batches[0], torch.tensor([1, 2]))
    assert torch.equal(batches[1], torch.tensor([3, 4]))


def test_dataloader_last_batch_can_be_smaller() -> None:
    dataset = TensorDataset(torch.tensor([1, 2, 3, 4, 5]))

    loader = DataLoader(
        dataset,
        batch_size=2,
    )

    batches = list(loader)

    assert len(batches) == 3
    assert torch.equal(batches[0], torch.tensor([1, 2]))
    assert torch.equal(batches[1], torch.tensor([3, 4]))
    assert torch.equal(batches[2], torch.tensor([5]))


def test_dataloader_without_batching() -> None:
    dataset = TensorDataset(torch.tensor([1, 2, 3]))

    loader = DataLoader(dataset)

    batches = list(loader)

    assert len(batches) == 3
    assert all(batch.ndim == 1 for batch in batches)

    for batch, expected in zip(
        batches,
        [
            torch.tensor([1]),
            torch.tensor([2]),
            torch.tensor([3]),
        ],
    ):
        assert torch.equal(batch, expected)


def test_dataloader_preserves_order_without_shuffle() -> None:
    dataset = TensorDataset(torch.arange(10))

    loader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=False,
    )

    result = torch.cat(list(loader))

    assert torch.equal(result, torch.arange(10))


def test_dataloader_shuffle_contains_all_elements() -> None:
    dataset = TensorDataset(torch.arange(100))

    loader = DataLoader(
        dataset,
        batch_size=10,
        shuffle=True,
    )

    result = torch.cat(list(loader))

    assert torch.equal(
        torch.sort(result).values,
        torch.arange(100),
    )


def test_dataloader_with_custom_sampler() -> None:
    dataset = TensorDataset(torch.arange(6))

    class FixedSampler(BatchSampler[int]):
        def __iter__(self):
            yield [5, 2]
            yield [0, 4]
            yield [1, 3]

        def __len__(self) -> int:
            return 3

    loader = DataLoader(
        dataset,
        sampler=FixedSampler(),
    )

    batches = list(loader)

    assert torch.equal(batches[0], torch.tensor([5, 2]))
    assert torch.equal(batches[1], torch.tensor([0, 4]))
    assert torch.equal(batches[2], torch.tensor([1, 3]))


def test_dataloader_invalid_batch_size() -> None:
    dataset = TensorDataset(torch.arange(10))

    with pytest.raises(ValueError, match="batch_size must be greater than 0"):
        DataLoader(
            dataset,
            batch_size=0,
        )


def test_dataloader_empty_dataset() -> None:
    dataset = TensorDataset(torch.tensor([]))

    loader = DataLoader(
        dataset,
        batch_size=4,
    )

    assert list(loader) == []
