#!/usr/bin/env python

import torch
import numpy as np
import os
import pandas as pd

from torchvision.transforms import v2

import lightning as L
from torch.utils.data import DataLoader, Dataset

from PIL import Image
import rasterio
import timm

class MapYourCityDataset(Dataset):
    '''
    Dataset for MapYourCity data

    Base class

    Arguments:
      data_path (str) : data root path
      csv_path (str) : csv file describing the split
      img_type (str) : choice of ['streetview', 'topview', 'sentinel-2'] TODO
      transform (str) : choice of ['default', 'resize', 'center_crop']
      model (str) : timm model identifier
      is_training (bool) : if True, apply data augmentation (applies to default case only)

    '''
    def __init__(self,
                 options,
                 split='train'
                 ):
        super().__init__()

        self.img_file = options['img_file']
        self.loader = None # assigned by subclass
        self.transforms = None # assigned by subclass

        # list of files
        if split in ['train', 'valid']:
            csv_path = os.path.join(options["fold_dir"], f'split_{split}_{options["fold"]}.csv')
            data_path = os.path.join(options["data_dir"], 'train', 'data')
        elif split == 'test':
            csv_path = os.path.join(options["data_dir"], 'test', 'test-set.csv')
            data_path = os.path.join(options["data_dir"], 'test', 'data')

        df = pd.read_csv(csv_path)

        self.pids = df['pid'].values

        if 'label' in df.columns:
            self.labels = df['label'].values
        else: # test stage
            self.labels = [0] * len(self.pids)

        self.image_paths = [os.path.join(data_path, pid, self.img_file) for pid in self.pids]

        # not all folders have a streetview image
        if self.img_file == 'street.jpg':
            is_valid = [os.path.exists(imp) for imp in self.image_paths]
            self.pids = [self.pids[i] for i in range(len(self.pids)) if is_valid[i]]
            self.labels = [self.labels[i] for i in range(len(self.labels)) if is_valid[i]]
            self.image_paths = [self.image_paths[i] for i in range(len(self.image_paths)) if is_valid[i]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.transforms( self.loader( self.image_paths[idx] ) )
        label = self.labels[idx]
        pid = self.pids[idx]

        return img, label, pid

class PhotoDataset(MapYourCityDataset):
    '''
    Dataset for MapYourCity data
    - street photos
    - orthophotos

    Arguments:
      data_path (str) : data root path
      csv_path (str) : csv file describing the split
      img_type (str) : choice of ['streetview', 'topview', 'sentinel-2'] TODO
      transform (str) : choice of ['default', 'resize', 'center_crop']
      model (str) : timm model identifier
      is_training (bool) : if True, apply data augmentation (applies to default case only)

    '''
    def __init__(self,
                 options,
                 split='train'
                 ):
        super().__init__(options, split)

        self.loader = self._photo_loader

        # assign transforms
        if options['model_id'] is None:
            config = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        else:
            config = timm.data.resolve_model_data_config(options['model_id'])
            input_size = int( config['input_size'][1] / config['crop_pct'] )

        print('xyz', config)

        match options['transform']:
            case 'default':
                if split == 'train':
                    trafo1 = [ v2.RandomResizedCrop(size=(input_size, input_size), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=3),
                               v2.RandomHorizontalFlip(p=0.5),
                               v2.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4), hue=None),]
                elif split in ['valid', 'test']:
                    trafo1 = [ v2.Resize(size=(input_size, input_size), interpolation=3),]

            case 'resize':
                trafo1 = [ v2.Resize(size=(input_size, input_size), interpolation=2), ]

            case 'center_crop':
                trafo1 = [ v2.CenterCrop(size=(input_size, input_size)), ]

            case _:
                raise ValueError('Invalid choice:', transform)

        trafo0 = [v2.ToImage()] 
        trafo2 = [v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=config['mean'], std=config['std'])]

        self.transforms = v2.Compose(trafo0 + trafo1 + trafo2)

    def _photo_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            img.load()

            return img.convert("RGB")

class Sentinel2Dataset(MapYourCityDataset):
    '''
    Dataset for MapYourCity data
    
    Sentinel-2 only

    TODO

    Arguments:
      data_path (str) : data root path
      csv_path (str) : csv file describing the split
      img_type (str) : choice of ['streetview', 'topview', 'sentinel-2']
      transform (str) : choice of ['default', 'resize', 'center_crop', 'random_crop', 'none']
      sentinel2_mode (str) : choice of ['rgb', 'ndvi']
      model (str) : timm model identifier
      is_training (bool) : if True, apply data augmentation (applies to default case only)
      augment (float) : probability for data augmentation (default: 0.0)

    '''
    def __init__(self,
                 options,
                 split='train'
                 ):
        super().__init__(options, split)

        self.use_ndvi = options['use_ndvi']
        # TODO add other bands

        self.loader = self._sentinel2_loader

        mean = [0., 0., 0.]
        std = [1., 1., 1.]

        trafo0 = [] 
        trafo1 = [] # TODO
        trafo2 = [v2.ToDtype(torch.float32, scale=True)]

        self.transforms = v2.Compose(trafo0 + trafo1 + trafo2)

    def _sentinel2_loader(self, path):
        '''
        Load Sentinel-2 and extract the RGB channels
        
        Apply factor of 3 x 10^-4 as in demo notebook
        
        Return as Image for compliance with transforms
        '''
        with rasterio.open(path) as f:
            # TODO add support for selecting bands and indices
            s2 = f.read()
            s2 = np.transpose(s2, [1,2,0]) * 3e-4
            # NIR - RED
            ndvi = (s2[...,7] - s2[...,3]) / (s2[...,7] + s2[...,3])
            # SWIR - NIR
            ndbi = (s2[...,10] - s2[...,7]) / (s2[...,10] + s2[...,7])
            # NIR - RGB
            ndwi = (s2[...,2] - s2[...,7]) / (s2[...,2] + s2[...,7])

            stacked = np.dstack([s2, ndvi, ndwi, ndbi]).astype(np.float32)

            # need channels first
            stacked = np.transpose(stacked, [2,0,1])

            return stacked

class MapYourCityDataModule(L.LightningDataModule):
    '''
    Generic datamodule for MapYourCity data

    Arguments:
      batch_size (int) : the mini-batch size
      num_workers (int) : the number of workers
      pin_memory (bool) : if True, pin GPU memory
      dataset_options (dict) : dictionary of options to pass on to the dataset

    '''
    def __init__(self,
                 batch_size,
                 num_workers,
                 pin_memory,
                 dataset_options
                 ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset_options = dataset_options

    def setup(self, stage='fit'):
        '''
        Train, valid and test data
        '''

        match self.dataset_options['img_file']:
            case 'street.jpg' | 'orthophoto.tif':
                self.train_data = PhotoDataset(self.dataset_options, split='train')
                self.valid_data = PhotoDataset(self.dataset_options, split='valid')
                self.test_data = PhotoDataset(self.dataset_options, split='test')
            case 's2_l2a.tif':
                self.train_data = Sentinel2Dataset(self.dataset_options, split='train')
                self.valid_data = Sentinel2Dataset(self.dataset_options, split='valid')
                self.test_data = Sentinel2Dataset(self.dataset_options, split='test')
            case _:
                raise ValueError('Invalid file', self.dataset_options['img_file'])


    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(dataset=self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=True)

    def valid_dataloader(self):
        return DataLoader(dataset=self.valid_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=False)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=False)
