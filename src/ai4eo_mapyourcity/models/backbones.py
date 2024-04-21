from typing import Any, List

import torch
from torch.nn import functional as F
from torch import optim
from torch import nn

import timm

import numpy as np

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

class TIMMCollectionCombined(nn.Module):
    '''
    Pretrained TIMM Models

    '''
    def __init__(self,
                 num_classes,
                 model_id,
                 is_pretrained,
                 num_models,
                 out_features,
                 mid_features
                 ):
        super().__init__()

        self.num_models = num_models
        self.out_features = out_features
        self.mid_features = mid_features
        self.num_classes = num_classes
        
        if self.num_models == 2:
            self.model1 = timm.create_model(model_id, pretrained=is_pretrained, num_classes=0)
            self.model2 = timm.create_model(model_id, pretrained=is_pretrained, num_classes=0)

        if self.num_models == 3:
            self.model3 = timm.create_model(model_id, pretrained=is_pretrained, num_classes=0)

        self.head = nn.Sequential(nn.Flatten(),
                                  nn.Linear(self.num_models * self.out_features, self.mid_features),
                                  nn.Linear(self.mid_features, self.num_classes))

    def forward(self, x):
        '''
        Combine models before the original classification head stage
        '''

        if self.num_models == 2:
            embeddings1 = self.model1(x[0])
            embeddings2 = self.model2(x[1])
            xcat = torch.cat([embeddings1, embeddings2])

        if self.num_models == 3:
            embeddings3 = self.model3(x[2])
            xcat = torch.cat([xcat, embeddings3])

        # common classification head
        return self.head(xcat.reshape(-1, self.num_models * self.out_features))

class SimpleConvNet(nn.Module):
    '''
    A simple ConvNet for the Sentinel-2 data

    '''

    def __init__(self, 
                 input_size,
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

        # calculate the flattened size
        S = input_size - kernel_size + 1
        S = np.ceil((S - 1) / pool_size)
        S = S - kernel_size + 1
        S = np.ceil((S - 1) / pool_size)
        flattened_size = int( S * S * out_channels)

        kernel_tuple = (kernel_size, kernel_size)
        pool_tuple = (pool_size, pool_size)

        self.backbone = nn.Sequential(
                         nn.Conv2d(in_channels, out_channels, kernel_tuple),
                         nn.MaxPool2d(pool_tuple),
                         nn.Conv2d(out_channels, out_channels, kernel_tuple),
                         nn.MaxPool2d(pool_tuple),
                         nn.Flatten(),
                         nn.Linear(flattened_size, 2 * mid_units),
                         nn.Dropout(dropout),
                         nn.Linear(2 * mid_units, mid_units),
                         nn.Dropout(dropout),
                         nn.Linear(mid_units, num_classes)
                         )

    def forward(self, x):
        return self.backbone(x)
