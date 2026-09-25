from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from random import shuffle


class BatchSampler[K](ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[list[K]]: ...
    @abstractmethod
    def __len__(self) -> int: ...


class DefaultSampler[K](BatchSampler[K]):
    def __init__(
        self,
        indices: Sequence[K],
        shuffle: bool = False,
        batch_size: int = 1,
    ) -> None:
        self.indices = list(indices)
        self.shuffle = shuffle
        self.batch_size = batch_size
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

    def __len__(self) -> int:
        return (len(self.indices) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[K]]:
        indices = self.indices.copy()
        if self.shuffle:
            shuffle(indices)
        for i in range(0, len(indices), self.batch_size):
            yield indices[i : i + self.batch_size]
