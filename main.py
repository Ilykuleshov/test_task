from pytorch_lightning.cli import LightningCLI
from cifar_simclr_resnet.data import CIFARModule
from cifar_simclr_resnet.model import ResNetSupervised

cli = LightningCLI(model_class=ResNetSupervised, datamodule_class=CIFARModule, run=False, save_config_callback=None)
cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)
cli.trainer.test(model=cli.model, datamodule=cli.datamodule)
