import torch
from torch import nn

from mini_dataloader.config.config import TrainingConfig
from mini_dataloader.data.dataloader import DataLoader


class Trainer:
    def __init__(self, config: TrainingConfig, model: nn.Module):
        self.config = config
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train_one_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        total_loss = 0.0
        for images, targets in dataloader:
            images = images.to(self.device)
            targets = targets.to(self.device)
            optimizer.zero_grad()
            features = self.model(images)
            loss = criterion(features, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss
        return total_loss / len(dataloader)

    def train(self, train_dataloader: DataLoader) -> None:
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        criterion = nn.CrossEntropyLoss()
        dataloader = train_dataloader
        for epoch in range(self.config.epochs):
            loss = self.train_one_epoch(
                dataloader,
                optimizer,
                criterion,
            )
            print(f"Epoch {epoch + 1}: loss={loss:.4f}")
