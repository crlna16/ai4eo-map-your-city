from typing import Any, List

import torch
from torch.nn import functional as F
from torch import optim
from torch import nn

import timm

from pytorch_lightning import LightningModule

class TIMMCollection(nn.Module):
    '''
    Pretrained TIMM Models

    '''
    def __init__(self,
                 num_classes,
                 model_id,
                 is_pretrained
                 ):
        super().__init__()
        self.model = timm.create_model(model_id, pretrained=is_pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


class SimpleConvNet(nn.Module):
    '''
    A simple ConvNet for the Sentinel-2 data

    '''

    def __init__(self, 
                 num_classes,
                 in_channels=12,
                 mid_channels=36,
                 kernel_size=5,
                 mid_units=128,
                 dropout=0.1
                 ):
        super().__init__()

        flattened_shape = 2

        self.backbone = nn.Sequential(
                         nn.Conv2d(in_channels, out_channels, kernel_size),
                         nn.MaxPool2d(),
                         nn.Conv2d(out_channels, out_channels, kernel_size),
                         nn.MaxPool2d(),
                         nn.Flatten(),
                         nn.Linear(mid_units, mid_units),
                         nn.Dropout(dropout),
                         nn.Linear(mid_units, num_classes)
                         )
