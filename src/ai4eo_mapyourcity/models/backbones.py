'''Backbones for MapYourCity challenge'''

from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F

import timm

import numpy as np

from ai4eo_mapyourcity import utils
log = utils.get_logger(__name__)

class TIMMCollection(nn.Module):
    '''
    Pretrained TIMM Model backbone.

    Attributes:
        model (torch.nn.Module): Pretrained model from TIMM.
    '''
    def __init__(self,
                 num_classes: int,
                 model_id: str,
                 is_pretrained: bool
                 ):
        '''
        Initialize the TIMMCollection instance.

        Arguments:
            num_classes (int): Number of classes.
            is_pretrained (bool): If True, use pretrained weights.
            model_id (str): The model to select (check TIMM collection on Huggingface for options)

        '''
        super().__init__()
        self.model = timm.create_model(model_id, pretrained=is_pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

class TIMMCollectionCombined(nn.Module):
    '''
    A model combining several pretrained TIMM models for different modalities.

    Supports late fusion with the following modes:
    - concatenate: concatenate the embeddings of all 2 / 3
      models, apply classification layer
    - average: calculate the average of the embeddings of all models, position-wise
      , apply classification layer
    - sum: position-wise sum, apply classification layer
    - max: position-wise max, apply classification layer
    - learned_weighted_average: average with learnable weights
    - attention: one self-attention layer, then average

    Attributes:
        num_models (int): Number of models / modalities
        model_id (Dict): TIMM model ID to use for each modality.
        is_pretrained (bool): If True, use pretrained weights.
        num_classes (int): Number of classes.
        out_features (Dict): Size of the embedding created by TIMM when stripping classification head.
        fusion_mode (str): Which fusion mode to select in late fusion.
        model1 (nn.Module): TODO merge to dict
        model2 (nn.Module): 
        model3 (nn.Module): 
        fusion (nn.Module): Fusion module.
    '''
    def __init__(self,
                 num_classes,
                 model_id: Dict[str, str],
                 is_pretrained,
                 out_features: Dict[str, int],
                 fusion_mode
                 ):
        '''
        Initialize TIMMCollectionCombined.

        Arguments:
            model_id (str): TIMM model ID to use for each modality.
            is_pretrained (bool): If True, use pretrained weights.
            num_classes (int): Number of classes.
            out_features (int): Size of the embedding created by TIMM when stripping classification head.
            fusion_mode (str): Which fusion mode to select in late fusion.
        '''
        super().__init__()

        self.out_features = max(out_features.values())
        self.num_classes = num_classes
        self.fusion_mode = fusion_mode

        model_dict = {}
        for key, value in model_id.items():
            if value is None:
                log.info(f'Skipping modality {key}')
            else:
                log.info(f'Creating TIMM model {model_id[key]} for modality {key}')
                model_dict[key] = timm.create_model(model_id[key], pretrained=is_pretrained, num_classes=0)

        self.models = nn.ModuleDict(model_dict)

        self.num_models = len(self.models.keys())

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
            case 'attention':
                self.fusion.attention = nn.ModuleDict({'topview':nn.Linear(self.out_features, 1),
                                                       'streetview':nn.Linear(self.out_features, 1),
                                                       'sentinel2':nn.Linear(self.out_features, 1)})
                self.fusion.head = nn.Linear(self.out_features, self.num_classes)
            case _:
                raise ValueError('Invalid fusion mode: ', self.fusion_mode)

        self.add_module('fusion', self.fusion)


    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        '''
        Combine models before the original classification head stage

        Arguments:
            x (Dict[str, torch.Tensor]): Images as key-value pairs. Matches with ModuleDict.
        '''

        embeddings = {}
        for key, encoder in self.models.items():
            if not key in x: # missing in input
                print(f'{key} is missing in input dict')
                continue
            embeddings[key] = encoder(x[key]).unsqueeze(1)


        match self.fusion_mode:
            case 'concatenate':
                # common classification head
                xcat = torch.cat(list(embeddings.values()), axis=1)
                xcat = xcat.flatten()
                return self.fusion.head(xcat)
            case 'average':
                xcat = torch.cat(list(embeddings.values()), axis=1)
                xcat = torch.mean(xcat, axis=1)
                return self.fusion.head(xcat)
            case 'sum':
                xcat = torch.cat(list(embeddings.values()), axis=1)
                xcat = torch.sum(xcat, axis=1)
                return self.fusion.head(xcat)
            case 'max':
                xcat = torch.cat(list(embeddings.values()), axis=1)
                xcat = torch.max(xcat, axis=1).values
                return self.fusion.head(xcat)
            case 'learned_weighted_average':
                xcat = torch.cat(list(embeddings.values()), axis=1)
                xcat = torch.einsum('bij,i -> bij', xcat, self.fusion.weights[0])
                xcat = torch.mean(xcat, axis=1)
                return self.fusion.head(xcat)
            case 'attention':
                # attention for each embedding
                atts = {}
                for key, embedding in embeddings.items():
                    atts[key] = self.fusion.attention[key](embedding)

                # attention weight
                acat = torch.cat(list(atts.values()), axis=1)
                aweights = F.softmax(acat, dim=1)

                weighted_embeddings = {}
                for i, (key, embedding) in enumerate(embeddings.items()):
                    weighted_embeddings[key] = embedding * aweights[:, i]

                xcat = torch.cat(list(weighted_embeddings.values()), axis=1)
                # fuse by averaging
                xcat = torch.mean(xcat, axis=1)
                # apply standard classification layer
                return self.fusion.head(xcat)

class SimpleConvNet(nn.Module):
    '''
    A convolutional neural network architecture.

    Attributes:
        backbone (nn.Sequential): Model.
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
        '''
        Initialize SimpleConvNet.

        Arguments:
            input_size (int): Image input size (assumes quadratic shape)
            num_classes (int): Number of classes
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels (of convolutional block)
            kernel_size (int): Convolutional kernel size.
            pool_size (int): MaxPool size.
            mid_units (int): Number of neurons after flattening.
            dropout (float): Dropout percentage.
            model_id (str): Argument only listed for consistency with other backbones.
        '''
        super().__init__()

        # calculate the flattened size
        mysize = input_size - kernel_size + 1
        mysize = np.ceil((mysize - 1) / pool_size)
        mysize = mysize - kernel_size + 1
        mysize = np.ceil((mysize - 1) / pool_size)
        flattened_size = int( mysize * mysize * out_channels)

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
