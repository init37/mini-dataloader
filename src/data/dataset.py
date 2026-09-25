from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence

import torch


class Dataset[K: Hashable](ABC):
    @abstractmethod
    def __getitem__(self, index: K) -> torch.Tensor: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def keys(self) -> Sequence[K]: ...
