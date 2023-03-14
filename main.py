from pytorch_lightning.cli import LightningCLI
from cifar_simclr_resnet.data import CIFARModule
from cifar_simclr_resnet.model import ResNetSupervised

LightningCLI(model_class=ResNetSupervised, datamodule_class=CIFARModule)
