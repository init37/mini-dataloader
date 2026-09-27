from torch import nn


def criterionFactory(name: str) -> nn.Module:
    if not name == "CE":
        raise ValueError(f"Unsupported criterion: {name}")
    return nn.CrossEntropyLoss()
