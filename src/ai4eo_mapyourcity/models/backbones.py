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
    Pretrained TIMM Models combined

    fusion_mode:
    - concatenate: concatenate the embeddings of all 2 / 3
      models, apply classification layer
    - average: calculate the average of the embeddings of all models, position-wise
      , apply classification layer
    - sum: position-wise sum, apply classification layer

    '''
    def __init__(self,
                 num_classes,
                 model_id,
                 is_pretrained,
                 num_models,
                 out_features,
                 fusion_mode
                 ):
        super().__init__()

        self.num_models = num_models
        self.out_features = out_features
        self.num_classes = num_classes
        self.fusion_mode = fusion_mode
        
        self.model1 = timm.create_model(model_id, pretrained=is_pretrained, num_classes=0)

        if self.num_models >= 2:
            self.model2 = timm.create_model(model_id, pretrained=is_pretrained, num_classes=0)

        if self.num_models >= 3:
            self.model3 = timm.create_model(model_id, pretrained=is_pretrained, num_classes=0)

        match self.fusion_mode:
            case 'concatenate':
                self.head = nn.Linear(self.num_models * self.out_features, self.num_classes)
            case 'average':
                self.head = nn.Linear(self.out_features, self.num_classes)
            case 'sum':
                self.head = nn.Linear(self.out_features, self.num_classes)
            case 'attention':
                self.extra_attention = nn.TransformerEncoderLayer(self.out_features, 8, 512, batch_first=True)
                self.head = nn.Linear(self.out_features, self.num_classes)
            case _:
                raise ValueError('Invalid fusion mode: ', self.fusion_mode)


    def forward(self, x):
        '''
        Combine models before the original classification head stage
        '''

        if self.num_models >= 2:
            embeddings1 = self.model1(x[0])
            embeddings2 = self.model2(x[1])
            xcat = torch.cat([embeddings1, embeddings2], axis=-1)

        if self.num_models == 3:
            embeddings3 = self.model3(x[2])
            xcat = torch.cat([xcat, embeddings3], axis=-1)

        match self.fusion_mode:
            case 'concatenate':
                # common classification head
                return self.head(xcat)
            case 'average':
                xcat = xcat.reshape(-1, self.num_models, self.out_features)
                xcat = torch.mean(xcat, axis=1)
                return self.head(xcat)
            case 'sum':
                xcat = xcat.reshape(-1, self.num_models, self.out_features)
                xcat = torch.sum(xcat, axis=1)
                return self.head(xcat)
            case 'attention':
                xcat = xcat.reshape(-1, self.num_models, self.out_features)
                xcat = self.extra_attention(xcat)
                xcat = torch.sum(xcat, axis=1)
                return self.head(xcat)
                

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
