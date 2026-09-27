from unittest.mock import MagicMock, patch

import pytest

from mini_dataloader.decorators.timed import timed


@patch("time.perf_counter")
@patch("builtins.print")
def test_timed_decorator_happy_path_no_cuda(mock_print, mock_perf_counter):
    """
    Test the timed decorator for a simple function when CUDA is not available.
    Verifies function execution, timing, and print output.
    """
    mock_perf_counter.side_effect = [0.0, 1.0]  # Simulate 1 second elapsed time

    @timed
    def my_function():
        return "hello"

    result = my_function()

    assert result == "hello"
    assert mock_perf_counter.call_count == 2
    mock_print.assert_called_once_with("elapsed=1.0000s")


@patch("time.perf_counter")
@patch("builtins.print")
@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.synchronize")
def test_timed_decorator_happy_path_with_cuda(
    mock_synchronize, mock_is_available, mock_print, mock_perf_counter
):
    """
    Test the timed decorator for a simple function when CUDA is available.
    Verifies function execution, timing, print output, and CUDA synchronization.
    """
    mock_perf_counter.side_effect = [0.0, 0.5]  # Simulate 0.5 seconds elapsed time

    @timed
    def my_function():
        return 123

    result = my_function()

    assert result == 123
    assert mock_perf_counter.call_count == 2
    mock_is_available.assert_called_once()
    mock_synchronize.assert_called_once()
    mock_print.assert_called_once_with("elapsed=0.5000s")


@patch("time.perf_counter")
@patch("builtins.print")
def test_timed_decorator_function_with_args_kwargs(mock_print, mock_perf_counter):
    """
    Test the timed decorator with a function that accepts positional and keyword arguments.
    Ensures arguments are passed correctly and the decorated function returns the expected result.
    """
    mock_perf_counter.side_effect = [0.0, 0.1]

    @timed
    def multiply(a, b, c=1):
        return a * b * c

    result = multiply(2, 3, c=4)

    assert result == 24
    mock_print.assert_called_once_with("elapsed=0.1000s")


@patch("time.perf_counter")
@patch("builtins.print")
def test_timed_decorator_function_raises_exception(mock_print, mock_perf_counter):
    """
    Test that the timed decorator correctly handles exceptions raised by the decorated function.
    The timing information should still be printed before the exception is re-raised.
    """
    mock_perf_counter.side_effect = [0.0, 0.2]

    class CustomError(Exception):
        pass

    @timed
    def function_that_errors():
        raise CustomError("Something went wrong")

    with pytest.raises(CustomError, match="Something went wrong"):
        function_that_errors()

    assert mock_perf_counter.call_count == 2
    mock_print.assert_called_once_with("elapsed=0.2000s")


@pytest.mark.parametrize(
    "return_value",
    [
        None,
        [],
        {"key": "value"},
        0,
        True,
    ],
    ids=["None", "empty_list", "dict", "zero", "boolean"],
)
@patch("time.perf_counter")
@patch("builtins.print")
def test_timed_decorator_various_return_types(
    mock_print, mock_perf_counter, return_value
):
    """
    Test the timed decorator with functions returning various data types.
    Ensures the original return value is preserved.
    """
    mock_perf_counter.side_effect = [0.0, 0.05]

    @timed
    def my_function():
        return return_value

    result = my_function()

    assert result == return_value
    mock_print.assert_called_once_with("elapsed=0.0500s")


@patch("time.perf_counter", MagicMock(return_value=0.0))  # Ensure consistent start
@patch("builtins.print")
def test_timed_decorator_very_short_execution(mock_print):
    """
    Test the timed decorator with a function that executes extremely quickly,
    resulting in zero or near-zero elapsed time.
    """
    # Simulate zero elapsed time
    with patch("time.perf_counter", side_effect=[0.0, 0.0]):

        @timed
        def quick_function():
            pass

        quick_function()
        mock_print.assert_called_once_with("elapsed=0.0000s")
        mock_print.reset_mock()

    # Simulate a very small non-zero elapsed time
    with patch("time.perf_counter", side_effect=[0.0, 1e-6]):

        @timed
        def quick_function_small():
            pass

        quick_function_small()
        mock_print.assert_called_once_with("elapsed=0.0000s")  # Due to formatting
        mock_print.reset_mock()

    with patch("time.perf_counter", side_effect=[0.0, 1e-4]):

        @timed
        def quick_function_small_formatted():
            pass

        quick_function_small_formatted()
        mock_print.assert_called_once_with("elapsed=0.0001s")
