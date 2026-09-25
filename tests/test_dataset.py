import pytest
import torch

from data.dataset import Dataset


def test_dataset_is_abstract():
    with pytest.raises(TypeError):
        Dataset()  # pyright: ignore[reportAbstractUsage]


def test_dataset_requires_implementation():
    class BrokenDataset(Dataset[int]):
        pass

    with pytest.raises(TypeError):
        BrokenDataset()  # pyright: ignore[reportAbstractUsage]


def test_concrete_dataset_works():
    class BasicDataset(Dataset[int]):
        def __init__(self, values: list[torch.Tensor]):
            self.values = values

        def __getitem__(self, index: int) -> torch.Tensor:
            return self.values[index]

        def __len__(self) -> int:
            return len(self.values)

        def keys(self) -> list[int]:
            return list(range(len(self.values)))

    ds = BasicDataset([torch.tensor([1]), torch.tensor([2]), torch.tensor([3])])
    assert ds[0] == torch.tensor([1])
    assert len(ds) == 3
