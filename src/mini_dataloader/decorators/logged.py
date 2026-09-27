import logging
from collections.abc import Callable
from functools import wraps
from typing import Any

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def logged[**P, T](func: Callable[P, tuple[T, T]]) -> Callable[P, tuple[T, T]]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> tuple[T, T]:
        result = func(*args, **kwargs)
        try:
            loss, acc = result
        except TypeError, ValueError:
            raise ValueError("not enough values to unpack (expected 2)") from None

        log.info("Loss: %s, Accuracy: %s", loss, acc)
        return loss, acc

    return wrapper
