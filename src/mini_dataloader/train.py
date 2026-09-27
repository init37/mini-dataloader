from torch import nn

from mini_dataloader.config.config import TrainingConfig
from mini_dataloader.data.dataloader import DataLoader
from mini_dataloader.data.dataset import datasetFactory
from mini_dataloader.factory.criterion import criterionFactory
from mini_dataloader.factory.optimizer import optimizerFactory
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
config = TrainingConfig()
optimizer = optimizerFactory(model, config.lr, config.optimizer)
criterion = criterionFactory(config.criterion)
trainer = Trainer(config, model, optimizer, criterion)
trainer.train(train_dataloader)
