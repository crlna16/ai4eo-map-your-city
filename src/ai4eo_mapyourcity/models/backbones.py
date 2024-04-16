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
                 in_channels,
                 out_channels,
                 kernel_size,
                 pool_size,
                 mid_units,
                 dropout,
                 model_id
                 ):
        super().__init__()

        flattened_size = 6084  # TODO
        kernel_tuple = (kernel_size, kernel_size)
        pool_tuple = (pool_size, pool_size)

        self.backbone = nn.Sequential(
                         nn.Conv2d(in_channels, out_channels, kernel_tuple),
                         nn.MaxPool2d(pool_tuple),
                         nn.Conv2d(out_channels, out_channels, kernel_tuple),
                         nn.MaxPool2d(pool_tuple),
                         nn.Flatten(),
                         nn.Linear(flattened_size, mid_units),
                         nn.Dropout(dropout),
                         nn.Linear(mid_units, num_classes)
                         )

    def forward(self, x):
        return self.backbone(x)
