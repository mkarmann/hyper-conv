"""
The main script for training the Classification models

It contains the implementation of the Hyper-Network and Hyper-Convolutions as well as the classifier
The MNIST data is automatically downloaded and then split in train and validation sets.

After the training is finished, the trained models can be found in the lightning_logs folder.
There are also results from previous runs stored already.

The entire training pipeline is using pytorch lightning and therefore
the start code was taken form the example of: https://www.pytorchlightning.ai/index.html#join-slack
The Hyper-Network's architecture was looked up from the authors repository: https://github.com/tym002/Hyper-Convolution
"""
import os.path
import shutil

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl


class HyperConv(nn.Module):
    """
    Hyper Convolution Layer

    If the use_hyper_net is set to true, weights will be predicted by a hyper net
    if not, it will be like a default convolutional layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias=False, use_hyper_net=True, **kwargs):
        """
        :param in_channels: in channels of the convolution
        :param out_channels: out channels of the convolution
        :param kernel_size: kernel size of the convolution
        :param bias: wether or not the convolution uses a bias term, this is not supported in combination with
            use_hyper_net=True
        :param kwargs: Args passed to the pytorch convolution function
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_hyper_net = use_hyper_net
        self.kwargs = kwargs

        if use_hyper_net:
            if bias:
                raise Exception('Bias for HyperConv is not supported by this implementation!')

            # Create position tensor with shape (batch, channel y, x)
            # Which is (1, 2, kernel_size_y, kernel_size_x)
            y, x = torch.meshgrid(torch.arange(kernel_size[0]) - 0.5 * kernel_size[0] + 0.5, torch.arange(kernel_size[1]) - 0.5 * kernel_size[1] + 0.5, indexing='ij')
            self.pos = torch.nn.Parameter(torch.concat([y[None, :, :], x[None, :, :]], dim=0)[None], requires_grad=False)

            # The hyper net architecture. Just like described in the paper this is implemented as a convolution
            num_c = self.in_channels * self.out_channels
            self.main = nn.Sequential(
                nn.Conv2d(2, 16, (1, 1)),
                nn.LeakyReLU(0.1),
                nn.Conv2d(16, 16, (1, 1)),
                nn.LeakyReLU(0.1),
                nn.Conv2d(16, 16, (1, 1)),
                nn.LeakyReLU(0.1),
                nn.Conv2d(16, num_c, (1, 1), bias=False),
                nn.BatchNorm2d(num_c)           # This layer was not included in the original implementation, but provides more stability during training and making the weights close to a std of 1 around a mean of zero
            )
        else:
            # If no hyper convolution, just take a regular convolution
            self.main = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)

    def get_weights(self) -> torch.Tensor:
        """
        Get the weights of the convolution.

        If this layer uses hyper convolution, it will return the predicted weights from the hyper net
        :return: weight tensor in the shape of (out_channels, in_channels, kernel_height, kernel_width)
        """
        if self.use_hyper_net:
            return self.main(self.pos).reshape(self.out_channels, self.in_channels, self.pos.shape[2], self.pos.shape[3])
        else:
            return self.main.weight

    def forward(self, x):
        if self.use_hyper_net:
            return F.conv2d(x, self.get_weights(), **self.kwargs)
        else:
            return self.main(x)


class ResidualBlock(nn.Module):
    """
    Residual block with the option to use hyper convolutions
    """
    def __init__(self, ch, kernel_size, use_hyper_conv=False):
        """
        Initialization

        :param ch: Number of input and output channels
        :param kernel_size: Kernel size for both convoluation in the residual block
        :param use_hyper_conv: If true, the block will use hyper convolutions
        """
        super().__init__()
        self.main = nn.Sequential(
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            HyperConv(ch, ch, kernel_size, padding='same', bias=False, use_hyper_net=use_hyper_conv),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            HyperConv(ch, ch, kernel_size, padding='same', bias=False, use_hyper_net=use_hyper_conv),
        )

    def forward(self, x):
        return self.main(x) + x


class MNISTClassifier(pl.LightningModule):
    """
    Classification model

    It consists of 4 times 2 residual blocks with 1x1 convolutions and max pooling in between.
    """
    def __init__(self, kernel=(5,5), use_hyper_conv=False):
        super().__init__()
        self.save_hyperparameters()
        ch = 8
        self.main = nn.Sequential(
            nn.Conv2d(1, ch, (1, 1), padding='same'),
            ResidualBlock(ch, kernel, use_hyper_conv=self.hparams.use_hyper_conv),
            ResidualBlock(ch, kernel, use_hyper_conv=self.hparams.use_hyper_conv),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.Conv2d(ch, ch*2, (1, 1), padding='same', bias=False),
            nn.MaxPool2d(2, 2),
            ResidualBlock(ch*2, kernel, use_hyper_conv=self.hparams.use_hyper_conv),
            ResidualBlock(ch*2, kernel, use_hyper_conv=self.hparams.use_hyper_conv),
            nn.BatchNorm2d(ch*2),
            nn.ReLU(),
            nn.Conv2d(ch*2, ch*4, (1, 1), padding='same', bias=False),
            nn.MaxPool2d(2, 2),
            ResidualBlock(ch * 4, kernel, use_hyper_conv=self.hparams.use_hyper_conv),
            ResidualBlock(ch * 4, kernel, use_hyper_conv=self.hparams.use_hyper_conv),
            nn.Conv2d(ch * 4, ch * 8, (1, 1), padding='same', bias=False),
            nn.MaxPool2d(2, 2),
            ResidualBlock(ch * 8, kernel, use_hyper_conv=self.hparams.use_hyper_conv),
            ResidualBlock(ch * 8, kernel, use_hyper_conv=self.hparams.use_hyper_conv),
            nn.BatchNorm2d(ch * 8),
            nn.ReLU(),
            nn.MaxPool2d((3, 3)),
            nn.Flatten(),
            nn.Linear(ch * 8, 10)
        )

    def forward(self, x):
        x = self.main(x)
        return x

    def configure_optimizers(self):
        # Lower lr for more stability (1e-3 created to much variance when training the hypernet)
        # Also added a bit of weight decay
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-6)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        pred = self(x)
        loss = F.cross_entropy(pred, y)
        acc = torch.mean(torch.eq(torch.argmax(pred, 1), y).float())
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        pred = self(x)
        loss = F.cross_entropy(pred, y)
        acc = torch.mean(torch.eq(torch.argmax(pred, 1), y).float())
        self.log('val_loss', loss)
        self.log('val_acc', acc)


def get_train_and_val_data(seed=1):
    """
    Getting fixed train and test data split (if seed is kept the same)

    :param seed: Seed for the split
    :return: tuple with (train_data, test_data)
    """
    dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
    torch.manual_seed(seed)
    train_data, val_data = random_split(dataset, [55000, 5000])

    return train_data, val_data


def train_model(train_data, val_data, kernel=(5, 5), use_hype_conv=False) -> MNISTClassifier:
    """
    Train a classification model

    :param train_data: Training data
    :param val_data: Validation data
    :param kernel: kernel size for all residual block convolutions of the model
    :param use_hype_conv: If true, uses hyper convolutions for the residual blocks in the model
    :return: model with best validation loss
    """
    train_loader = DataLoader(train_data, batch_size=64)
    val_loader = DataLoader(val_data, batch_size=256)

    # model
    model = MNISTClassifier(kernel=kernel, use_hyper_conv=use_hype_conv)

    # training
    model_checkpoint = ModelCheckpoint(monitor='val_loss', mode='min')
    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=10),
            model_checkpoint
        ],
        logger=CSVLogger('.')
    )
    trainer.fit(model, train_loader, val_loader)

    # Copy best checkpoint
    shutil.copy(model_checkpoint.best_model_path, os.path.join(os.path.dirname(os.path.dirname(model_checkpoint.best_model_path)), 'best.ckpt'))

    # Load best model
    best_model = MNISTClassifier.load_from_checkpoint(model_checkpoint.best_model_path)
    return best_model


def evaluate_model_accuracy(model, data) -> float:
    """
    Get the models accuracy on data

    :param model: model to test
    :param data: Data to test
    :return: accuracy
    """
    val_loader = DataLoader(data, batch_size=256)
    trainer = pl.Trainer(logger=CSVLogger('tmp'))
    acc = trainer.validate(model, val_loader)[0]['val_acc']
    shutil.rmtree(trainer.logger.log_dir)
    return acc


if __name__ == '__main__':
    train_data, val_data = get_train_and_val_data()
    model = train_model(train_data, val_data, kernel=(5, 5), use_hype_conv=True)
