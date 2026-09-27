from types import TracebackType

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
    ) -> tuple[float, float]:
        total_loss = 0.0
        correct = 0
        total = 0
        for images, targets in dataloader:
            images = images.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            features = self.model(images)
            loss = self.criterion(features, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss
            predictions = features.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
        return total_loss / len(dataloader), correct / total

    def train(self, dataloader: DataLoader) -> None:
        with TrainingMode(self.model):
            for epoch in range(self.config.epochs):
                loss, acc = self.train_one_epoch(
                    dataloader,
                )
                print(f"Epoch {epoch + 1}: loss={loss:.4f}, Accuracy: {acc:.4f}")

    def eval(self, dataloader: DataLoader) -> None:
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for image, target in dataloader:
                image = image.to(self.device)
                target = target.to(self.device)
                features = self.model(image)
                loss = self.criterion(features, target)
                total_loss += loss.item()
                prediction = features.argmax(dim=1)
                correct += (prediction == target).sum().item()
                total += target.size(0)
        acc = correct / total
        loss = total_loss / len(dataloader)
        print(f"Test set: loss={loss:.4f}, Accuracy: {acc:.4f}")


class TrainingMode:
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def __enter__(self) -> None:
        self.model.train()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.model.eval()
