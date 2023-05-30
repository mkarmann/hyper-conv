"""
Basic code taken from: https://www.pytorchlightning.ai/index.html#join-slack
"""
import io
import shutil

import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl


class MNISTClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28 * 28))

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)


def get_train_and_val_data():
    # data
    dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
    train_data, val_data = random_split(dataset, [55000, 5000])

    return train_data, val_data


def train_model(train_data, val_data, use_hype_conv=False):
    train_loader = DataLoader(train_data, batch_size=64)
    val_loader = DataLoader(val_data, batch_size=256)

    # model
    model = MNISTClassifier()

    # training
    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min")
        ],
        logger=CSVLogger('.')
    )
    trainer.fit(model, train_loader, val_loader)

    # Final validation score
    best_model = MNISTClassifier.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    return best_model


def validate_model(model, val_data):
    val_loader = DataLoader(val_data, batch_size=256)
    trainer = pl.Trainer()
    val_loss = trainer.validate(model, val_loader)[0]['val_loss']
    shutil.rmtree(trainer.logger.log_dir)
    return val_loss


if __name__ == '__main__':
    train_data, val_data = get_train_and_val_data()
    model = train_model(train_data, val_data)
    val_loss = validate_model(model, val_data)
    print(val_loss)