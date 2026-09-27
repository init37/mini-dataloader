from torch import nn

from mini_dataloader.config.config import TrainingConfig
from mini_dataloader.data.dataloader import DataLoader
from mini_dataloader.data.dataset import datasetFactory
from mini_dataloader.trainer import Trainer

train_dataset = datasetFactory("fashion", True)
train_dataloader = DataLoader(train_dataset)

model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(64 * 7 * 7, 10),
)

config = TrainingConfig(lr=1e-4, batch_size=8, epochs=10, device="cuda", shuffle=True)
trainer = Trainer(config, model)
trainer.train(train_dataloader)
