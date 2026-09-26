from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence

import torch


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
