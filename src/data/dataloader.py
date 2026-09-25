from collections.abc import Hashable, Iterator

import torch

from .dataset import Dataset
from .sampler import BatchSampler, DefaultSampler


class DataLoader[K: Hashable]:
    def __init__(
        self,
        dataset: Dataset[K],
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: BatchSampler[K] | None = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        if sampler is not None:
            self.shuffle = False
            self.sampler = sampler
        else:
            self.shuffle = shuffle
            self.sampler = DefaultSampler[K](
                indices=self.dataset.keys(),
                shuffle=self.shuffle,
                batch_size=self.batch_size,
            )

    def __iter__(self) -> Iterator[torch.Tensor]:
        for keys in self.sampler:
            yield torch.stack([self.dataset[key] for key in keys])

    def __len__(self) -> int:
        return len(self.sampler)
