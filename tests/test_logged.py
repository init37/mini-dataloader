from unittest.mock import MagicMock, patch

import pytest

from mini_dataloader.decorators.logged import logged


@pytest.fixture
def mock_logger():
    """Fixture to mock the logger's info method."""
    with patch("mini_dataloader.decorators.logged.log") as mock_log:
        yield mock_log


def test_logged_decorator_happy_path(mock_logger):
    """
    Test the logged decorator for a function that returns typical loss and accuracy.
    Verifies that the logger is called correctly and the original return value is preserved.
    """

    @logged
    def training_step():
        return 0.123, 0.987

    loss, acc = training_step()

    assert loss == 0.123
    assert acc == 0.987
    mock_logger.info.assert_called_once_with("Loss: %s, Accuracy: %s", 0.123, 0.987)


def test_logged_decorator_with_arguments(mock_logger):
    """
    Test the logged decorator with a function that accepts positional and keyword arguments.
    Ensures arguments are passed correctly, logging occurs, and return value is preserved.
    """

    @logged
    def evaluate(data, model_id=1):
        # Simulate some computation
        if model_id == 1:
            return 0.05 * len(data), 0.95
        return 0.10 * len(data), 0.90

    data = [1, 2, 3, 4, 5]
    loss, acc = evaluate(data, model_id=1)

    assert loss == 0.25
    assert acc == 0.95
    mock_logger.info.assert_called_once_with("Loss: %s, Accuracy: %s", 0.25, 0.95)

    mock_logger.info.reset_mock()
    loss, acc = evaluate(data, model_id=2)
    assert loss == 0.50
    assert acc == 0.90
    mock_logger.info.assert_called_once_with("Loss: %s, Accuracy: %s", 0.50, 0.90)


@pytest.mark.parametrize(
    "loss_val, acc_val, expected_loss_str, expected_acc_str",
    [
        (0.0, 1.0, "0.0", "1.0"),  # Edge: min/max values
        (0.5, 0.5, "0.5", "0.5"),
        (
            -0.1,
            1.1,
            "-0.1",
            "1.1",
        ),  # Edge: unusual values (though not typical for loss/acc)
        ("N/A", "UNKNOWN", "N/A", "UNKNOWN"),  # Edge: non-numeric values
        (None, None, "None", "None"),  # Edge: None values
        (
            MagicMock(),
            MagicMock(),
            "<MagicMock id=",
            "<MagicMock id=",
        ),  # Edge: objects (partial match)
    ],
    ids=[
        "zero_one",
        "half_half",
        "unusual_values",
        "non_numeric",
        "none_values",
        "mock_objects",
    ],
)
def test_logged_decorator_various_return_types(
    mock_logger, loss_val, acc_val, expected_loss_str, expected_acc_str
):
    """
    Test the logged decorator with functions returning various types for loss and accuracy.
    Ensures logging handles different data types and original values are returned.
    """

    @logged
    def get_values():
        return loss_val, acc_val

    returned_loss, returned_acc = get_values()

    assert returned_loss == loss_val
    assert returned_acc == acc_val
    if isinstance(
        loss_val, MagicMock
    ):  # Special handling for mock objects to check their string representation
        mock_logger.info.assert_called_once()
        assert mock_logger.info.call_args[0][0] == "Loss: %s, Accuracy: %s"
        assert str(mock_logger.info.call_args[0][1]).startswith(expected_loss_str)
        assert str(mock_logger.info.call_args[0][2]).startswith(expected_acc_str)
    else:
        mock_logger.info.assert_called_once_with(
            "Loss: %s, Accuracy: %s", loss_val, acc_val
        )


def test_logged_decorator_function_raises_exception(mock_logger):
    """
    Test that the logged decorator correctly handles exceptions raised by the decorated function.
    The decorator should re-raise the exception, and no logging should occur before the exception
    if the error happens before return.
    """

    class CustomError(Exception):
        pass

    @logged
    def function_that_errors():
        raise CustomError("Operation failed")
        # The logging line will not be reached

    with pytest.raises(CustomError, match="Operation failed"):
        function_that_errors()

    mock_logger.info.assert_not_called()


@pytest.mark.parametrize(
    "invalid_return",
    [
        0.123,  # Not a tuple
        (0.123,),  # Tuple of wrong length
        (0.123, 0.987, 0.5),  # Tuple of wrong length
        "single_string",
        None,
    ],
    ids=["float", "tuple_one_element", "tuple_three_elements", "string", "None"],
)
def test_logged_decorator_function_returns_invalid_type(mock_logger, invalid_return):
    """
    Test that the logged decorator raises a TypeError if the decorated function
    does not return a tuple of exactly two elements, as specified by the type hint.
    """

    @logged
    def function_with_bad_return():
        return invalid_return

    # Python's type hints are for static analysis, not runtime enforcement by default.
    # The `logged` decorator expects `tuple[T, T]`, but the runtime behavior for
    # unpacking `loss, acc = func(...)` will raise a ValueError if `func`
    # does not return an iterable of length 2.
    with pytest.raises(ValueError, match="not enough values to unpack"):
        function_with_bad_return()

    mock_logger.info.assert_not_called()
