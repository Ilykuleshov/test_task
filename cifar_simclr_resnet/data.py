import torch
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from pytorch_lightning import LightningDataModule
from torchvision.datasets import CIFAR10
from torchvision import transforms as T

from .const import REPO_ROOT


class CIFARModule(LightningDataModule):
    def __init__(self, batch_size: int=128, train_size: float=1., num_workers: int=4) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.train_size = train_size
        self.num_workers=num_workers

    def setup(self, stage):
        generator=torch.Generator().manual_seed(42)

        transform = T.Compose([
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float32)
        ])
        train_set = CIFAR10(root=REPO_ROOT / 'data', train=True, download=True, transform=transform)
        control_set = CIFAR10(root=REPO_ROOT / 'data', train=False, download=True, transform=transform)
        
        self.train_ds = random_split(train_set, [self.train_size, 1 - self.train_size], generator=generator)[0]
        self.val_ds, self.test_ds = random_split(control_set, [0.5, 0.5], generator=generator)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
