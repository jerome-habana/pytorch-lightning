# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import argparse

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
import torchvision

import pytorch_lightning as pl
from pytorch_lightning.utilities import _HPU_AVAILABLE

from pytorch_lightning.callbacks import Callback

from pytorch_lightning.utilities.hpu_datamodule import HPUDataModule


class ConvolutionOnHPU(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.l1 = torch.nn.Linear(224 * 224 * 16, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x)
        acc = self.accuracy(probs, y)
        return acc

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = self.accuracy(logits, y)
        return acc

    def accuracy(self, logits, y):
        acc = torch.sum(torch.eq(torch.argmax(logits, -1), y).to(torch.float32)) / len(y)
        return acc

    def validation_epoch_end(self, outputs) -> None:
        self.log("val_acc", torch.stack(outputs).mean(), prog_bar=True)

    def test_epoch_end(self, outputs) -> None:
        self.log("test_acc", torch.stack(outputs).mean())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

# Init our model
model = ConvolutionOnHPU()

#imagenet dataset
def load_data(traindir, valdir):
    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    
    dataset_test = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    print("Creating data loaders")

    return dataset, dataset_test

def main(args):

    data_path = args.data_path
    train_dir = os.path.join(data_path, 'train')
    val_dir = os.path.join(data_path, 'val')
    train_ds, val_ds = load_data(train_dir, val_dir)

    data_module = HPUDataModule(train_ds, val_ds)

    # Initialize a trainer
    trainer = pl.Trainer(devices=1, accelerator="hpu", max_epochs=1, precision=32)

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
    trainer.validate(model, datamodule=data_module)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='PyLight ImageNet Training')
    parser.add_argument('--data-path', default='/software/data/pytorch/imagenet/ILSVRC2012/', help='dataset')
    args = parser.parse_args()
    
    main(args)
