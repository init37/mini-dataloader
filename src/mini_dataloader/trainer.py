import torch
from torch import nn, optim

from mini_dataloader.config.config import TrainingConfig
from mini_dataloader.data.dataloader import DataLoader


class Trainer:
    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
    ):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train_one_epoch(
        self,
        dataloader: DataLoader,
    ) -> float:
        total_loss = 0.0
        for images, targets in dataloader:
            images = images.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            features = self.model(images)
            loss = self.criterion(features, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss
        return total_loss / len(dataloader)

    def train(self, dataloader: DataLoader) -> None:
        self.model.train()
        for epoch in range(self.config.epochs):
            loss = self.train_one_epoch(
                dataloader,
            )
            print(f"Epoch {epoch + 1}: loss={loss:.4f}")
