from dataclasses import dataclass


@dataclass
class TrainingConfig:
    lr: float = 1e-4
    batch_size: int = 8
    epochs: int = 50
    device: str = "cuda"
    shuffle: bool = True
