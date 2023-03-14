from typing import Dict
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
from torchmetrics import Accuracy
from pytorch_lightning import LightningModule

from .const import REPO_ROOT


class ResNetSupervised(LightningModule):
    def __init__(self, pretrained: bool=False, feat_dim=128, num_classes=10) -> None:
        super().__init__()
        
        torch.set_float32_matmul_precision('medium')
        backbone = resnet18(num_classes=feat_dim)
        dim_mlp = backbone.fc.in_features
        backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), backbone.fc)

        if pretrained:
            weights = torch.load(REPO_ROOT / 'weights/pretrained_weights.pth')
            backbone.load_state_dict(weights)
        
        self.backbone = backbone
        self.classifier = nn.Linear(feat_dim, num_classes)

        self.loss = nn.CrossEntropyLoss()
        self.metrics = nn.ModuleDict({
            k: Accuracy(task='multiclass', num_classes=num_classes)
            for k in ['Train', 'Val', 'Test']
        })

    def forward(self, x):
        return self.classifier(self.backbone(x))

    def step(self, batch, key: str):
        images, labels = batch
        predictions = self(images)
        loss = F.cross_entropy(predictions, labels)
        metric = self.metrics[key](predictions, labels)

        self.log(f'{key}/loss', loss)
        self.log(f'{key}/accuracy', self.metrics[key])

        return loss if key == 'Train' else metric

    def training_step(self, batch): return self.step(batch, 'Train')
    def validation_step(self, batch, batch_idx): return self.step(batch, 'Val')
    def test_step(self, batch, batch_idx): return self.step(batch, 'Test')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)
