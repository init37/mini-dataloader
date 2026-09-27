from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence

import torch
import torchvision


class Dataset[K: Hashable](ABC):
    X: torch.Tensor
    y: torch.Tensor

    @abstractmethod
    def __getitem__(self, index: K) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def keys(self) -> Sequence[K]: ...


class FashionMNISTDataset(Dataset[int]):
    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.X = X
        self.y = y

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.X[index], self.y[index])

    def __len__(self) -> int:
        return self.X.shape[0]

    def keys(self) -> list[int]:
        return list(range(len(self)))


def datasetFactory(name: str, train: bool) -> Dataset:
    if not isinstance(train, bool):
        raise TypeError("train argument must be a boolean")

    if name.lower() != "fashion":
        raise ValueError(f"Unsupported dataset name: {name}")

    transform = torchvision.transforms.ToTensor()
    temp_set = torchvision.datasets.FashionMNIST(
        root="./db",
        train=train,
        transform=transform,
        download=True,
    )
    X = torch.stack([image for image, _ in temp_set])
    y = torch.tensor([label for _, label in temp_set]).to(torch.long)
    return FashionMNISTDataset(X, y)
