from collections.abc import Hashable

from .dataset import Dataset
from .sampler import Sampler


class DataLoader[K: Hashable, V]:
    def __init__(
        self,
        dataset: Dataset[K, V],
        batch_size: int | None = 1,
        shuffle: bool = False,
        sampler: Sampler[K] | None = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampler = sampler

        if sampler is not None:
            self.shuffle = False
        else:
            self.shuffle = shuffle
