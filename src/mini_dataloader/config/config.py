from dataclasses import dataclass


@dataclass
class TrainingConfig:
    lr: float = 1e-4
    batch_size: int = 8
    epochs: int = 5
    device: str = "cuda"
    shuffle: bool = True
    optimizer: str = "Adam"
    criterion: str = "CE"
