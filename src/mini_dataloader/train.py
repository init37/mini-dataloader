import torch
import torchvision
from torch import nn

from mini_dataloader.config.config import TrainingConfig
from mini_dataloader.data.dataloader import DataLoader
from mini_dataloader.data.dataset import FashionMNISTDataset
from mini_dataloader.trainer import Trainer

transform = torchvision.transforms.ToTensor()

training_set = torchvision.datasets.FashionMNIST(
    "./db", train=True, transform=transform, download=True
)
validation_set = torchvision.datasets.FashionMNIST(
    "./db", train=False, transform=transform, download=True
)

X_train = torch.stack([image for image, _ in training_set])
y_train = torch.tensor([label for _, label in training_set])

X_test = torch.stack([image for image, _ in validation_set])
y_test = torch.tensor([label for _, label in validation_set])

train_dataset = FashionMNISTDataset(X_train, y_train)
test_dataset = FashionMNISTDataset(X_test, y_test)

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
