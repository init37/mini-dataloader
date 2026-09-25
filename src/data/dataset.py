from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence


class Dataset[K: Hashable, V](ABC):
    @abstractmethod
    def __getitem__(self, index: K) -> V: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def keys(self) -> Sequence[K]: ...
