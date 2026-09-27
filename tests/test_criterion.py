import pytest
from torch import nn

from mini_dataloader.factory.criterion import criterionFactory


class TestCriterionFactory:
    """Tests for the criterionFactory function."""

    @pytest.mark.parametrize(
        "criterion_name, expected_type",
        [
            ("CE", nn.CrossEntropyLoss),
        ],
    )
    def test_criterion_factory_happy_path(self, criterion_name, expected_type):
        """
        Test that criterionFactory returns the correct criterion type for supported names.
        """
        criterion = criterionFactory(criterion_name)
        assert isinstance(criterion, expected_type)

    @pytest.mark.parametrize(
        "unsupported_name",
        [
            "MSE",
            "BCE",
            "L1Loss",
            "UnknownCriterion",
            "",
            "ce",  # Case sensitivity check
        ],
    )
    def test_criterion_factory_unsupported_criterion_raises_value_error(
        self, unsupported_name
    ):
        """
        Test that criterionFactory raises a ValueError for unsupported criterion names.
        """
        with pytest.raises(ValueError) as excinfo:
            criterionFactory(unsupported_name)
        assert f"Unsupported criterion: {unsupported_name}" in str(excinfo.value)
