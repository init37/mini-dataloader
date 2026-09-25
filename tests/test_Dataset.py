import pytest

from data.dataset import Dataset


def test_dataset_is_abstract():
    with pytest.raises(TypeError):
        Dataset()  # pyright: ignore[reportAbstractUsage]


def test_dataset_requires_implementation():
    class BrokenDataset(Dataset[int, str]):
        pass

    with pytest.raises(TypeError):
        BrokenDataset()  # pyright: ignore[reportAbstractUsage]


def test_concrete_dataset_works():
    class BasicDataset(Dataset[int, str]):
        def __init__(self, values: list[str]):
            self.values = values

        def __getitem__(self, index: int) -> str:
            return self.values[index]

        def __len__(self) -> int:
            return len(self.values)

    ds = BasicDataset(["a", "b", "c"])
    assert ds[0] == "a"
    assert len(ds) == 3
