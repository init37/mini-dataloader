from torch import nn, optim


def optimizerFactory(model: nn.Module, lr: float, name: str) -> optim.Optimizer:
    if not isinstance(model, nn.Module):
        raise TypeError("Unsupported model API")
    if not isinstance(lr, float):
        raise TypeError("Unsupported lr value")
    if name.lower() != "adam":
        raise ValueError(f"Unsupported optimizer name: {name}")
    return optim.Adam(model.parameters(), lr)
