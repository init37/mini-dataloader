from collections.abc import Hashable

from .dataset import Dataset
from .sampler import BatchSampler, DefaultSampler


class DataLoader[K: Hashable, V]:
    def __init__(
        self,
        dataset: Dataset[K, V],
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
            self.sampler = DefaultSampler(
                indices=self.dataset.keys(),
                shuffle=self.shuffle,
                batch_size=self.batch_size,
            )
