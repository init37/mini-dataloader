import time
from collections.abc import Callable
from functools import wraps
from typing import Any

import torch


def timed[**P, T](
    func: Callable[P, T],
) -> Callable[P, T]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            epoch_time = time.perf_counter() - start_time
            print(f"elapsed={epoch_time:.4f}s")
        return result

    return wrapper
