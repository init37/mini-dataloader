import time
from collections.abc import Callable
from functools import wraps
from typing import Any

import torch


def timed[**P](
    func: Callable[P, tuple[float, float]],
) -> Callable[P, tuple[float, float]]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> tuple[float, float]:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - start_time
        print(f"elapsed={epoch_time:.4f}s")
        return result

    return wrapper
