from abc import ABC, abstractmethod
from collections.abc import Iterator
from random import shuffle


class Sampler[K](ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[K]: ...
    @abstractmethod
    def __len__(self) -> int: ...


class DefaultSampler(Sampler[list[int]]):
    def __init__(
        self, indicies: list[int], shuffle: bool = False, batch_size: int = 1
    ) -> None:
        self.indicies = indicies
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __len__(self) -> int:
        return (len(self.indicies) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        indicies = self.indicies.copy()
        if self.shuffle:
            shuffle(indicies)
        for i in range(0, len(self.indicies), self.batch_size):
            yield indicies[i : i + self.batch_size]
