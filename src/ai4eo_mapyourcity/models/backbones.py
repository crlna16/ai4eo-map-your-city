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
    - max: position-wise max, apply classification layer
    - learned_weighted_average: average with learnable weights
    - attention: one self-attention layer, then average

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

        self.fusion = nn.Module()

        match self.fusion_mode:
            case 'concatenate':
                self.fusion.head = nn.Linear(self.num_models * self.out_features, self.num_classes)
            case 'average':
                self.fusion.head = nn.Linear(self.out_features, self.num_classes)
            case 'sum':
                self.fusion.head = nn.Linear(self.out_features, self.num_classes)
            case 'max':
                self.fusion.head = nn.Linear(self.out_features, self.num_classes)
            case 'learned_weighted_average':
                self.fusion.weights = nn.ParameterList([nn.Parameter(torch.rand(self.num_models))])
                self.fusion.head = nn.Sequential(nn.LayerNorm(self.out_features), nn.Linear(self.out_features, self.num_classes))
            case 'attention':
            # TODO https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/modules/fusions/attention_fusion.py
                self.fusion.extra_attention = nn.TransformerEncoderLayer(self.out_features, 8, 512, batch_first=True)
                self.fusion.head = nn.Linear(self.out_features, self.num_classes)
            case _:
                raise ValueError('Invalid fusion mode: ', self.fusion_mode)

        self.add_module('fusion', self.fusion)


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
                return self.fusion.head(xcat)
            case 'average':
                xcat = xcat.reshape(-1, self.num_models, self.out_features)
                xcat = torch.mean(xcat, axis=1)
                return self.fusion.head(xcat)
            case 'sum':
                xcat = xcat.reshape(-1, self.num_models, self.out_features)
                xcat = torch.sum(xcat, axis=1)
                return self.fusion.head(xcat)
            case 'max':
                xcat = xcat.reshape(-1, self.num_models, self.out_features)
                xcat = torch.max(xcat, axis=1).values
                return self.fusion.head(xcat)
            case 'learned_weighted_average':
                xcat = xcat.reshape(-1, self.num_models, self.out_features)
                xcat = torch.einsum('bij,i -> bij', xcat, self.fusion.weights[0])
                xcat = torch.mean(xcat, axis=1)
                return self.fusion.head(xcat)
            case 'attention':
                xcat = xcat.reshape(-1, self.num_models, self.out_features)
                xcat = self.fusion.extra_attention(xcat)
                xcat = torch.sum(xcat, axis=1)
                return self.fusion.head(xcat)
                

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
