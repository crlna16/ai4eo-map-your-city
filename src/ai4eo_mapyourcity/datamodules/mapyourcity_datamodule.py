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
                 data_path,
                 csv_path,
                 img_type,
                 transform,
                 sentinel2_mode,
                 model,
                 is_training,
                 augment=0.0
                 ):
        super().__init__()

        self.img_type = img_type
        self.label_file = 'label.txt'
        self.sentinel2_mode = sentinel2_mode
        self.augment = augment

        config = timm.data.resolve_model_data_config(model)
        true_size = int(config['input_size'][1] / config['crop_pct'])
        true_size = int(config['input_size'][1])

        match self.img_type:
            case 'streetview':
                self.img_file = 'street.jpg'
                self.loader = self._photo_loader
            case 'topview':
                self.img_file = 'orthophoto.tif'
                self.loader = self._photo_loader
            case 'sentinel-2':
                self.img_file = 's2_l2a.tif'
                if self.sentinel2_mode == 'rgb':
                    self.loader = self._sentinel_rgb_loader
                elif self.sentinel2_mode == 'ndvi':
                    self.loader = self._sentinel_ndvi_loader
            case _:
                raise ValueError('Invalid choice:', self.img_type)

        # assign transforms
        match transform:
            case 'default':
                if is_training:
                    self.transforms = v2.Compose([v2.ToImage(),
                                        v2.RandomResizedCrop(size=(true_size, true_size), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=3),
                                        v2.RandomHorizontalFlip(p=0.5),
                                        v2.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4), hue=None),
                                        v2.ToDtype(torch.float32, scale=True),
                                        v2.Normalize(mean=config['mean'], std=config['std'])
                                        ])
                else:
                    self.transforms = v2.Compose([v2.ToImage(),
                                        v2.Resize(size=(true_size, true_size), interpolation=3),
                                        v2.ToDtype(torch.float32, scale=True),
                                        v2.Normalize(mean=config['mean'], std=config['std'])
                                        ])
            case 'resize':
                self.transforms = v2.Compose([v2.ToImage(),
                                      v2.Resize(size=(true_size, true_size), interpolation=2),
                                      v2.RandomHorizontalFlip(p=self.augment),
                                      v2.RandomVerticalFlip(p=self.augment),
                                      v2.ToDtype(torch.float32, scale=True),
                                      v2.Normalize(mean=config['mean'], std=config['std'])
                                      ])
            case 'center_crop':
                self.transforms = v2.Compose([v2.ToImage(),
                                      v2.CenterCrop(size=(true_size, true_size)),
                                      v2.RandomHorizontalFlip(p=self.augment),
                                      v2.RandomVerticalFlip(p=self.augment),
                                      v2.ToDtype(torch.float32, scale=True),
                                      v2.Normalize(mean=config['mean'], std=config['std'])
                                      ])
            case 'random_crop':
                self.transforms = v2.Compose([v2.ToImage(),
                                      v2.RandomCrop(size=(true_size, true_size)),
                                      v2.RandomHorizontalFlip(p=self.augment),
                                      v2.RandomVerticalFlip(p=self.augment),
                                      v2.ToDtype(torch.float32, scale=True),
                                      v2.Normalize(mean=config['mean'], std=config['std'])
                                      ])
            case 'none':
                self.transforms = v2.Compose([v2.ToImage(),
                                      v2.ToDtype(torch.float32, scale=True),
                                      ])
            case _:
                raise ValueError('Invalid choice:', transform)


        df = pd.read_csv(csv_path)

        self.pids = df['pid'].values

        if 'label' in df.columns:
            self.labels = df['label'].values
        else: # test stage
            self.labels = [0] * len(self.pids)

        self.image_paths = [os.path.join(data_path, pid, self.img_file) for pid in self.pids]

        # not all folders have a streetview image
        if self.img_type == 'streetview':
            is_valid = [os.path.exists(imp) for imp in self.image_paths]
            self.pids = [self.pids[i] for i in range(len(self.pids)) if is_valid[i]]
            self.labels = [self.labels[i] for i in range(len(self.labels)) if is_valid[i]]
            self.image_paths = [self.image_paths[i] for i in range(len(self.image_paths)) if is_valid[i]]

    def _photo_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            img.load()

            return img.convert("RGB")

    def _sentinel_rgb_loader(self, path):
        '''
        Load Sentinel-2 and extract the RGB channels
        
        Apply factor of 3 x 10^-4 as in demo notebook
        
        Return as Image for compliance with transforms
        '''
        with rasterio.open(path) as f:
            s2 = f.read()
            s2 = np.transpose(s2,[1,2,0])

            s2_rgb = s2[...,[3,2,1]] * 3e-4
            compl_trafo = v2.Compose([v2.ToImage(), v2.ToDtype(torch.uint8, scale=True)])
            return compl_trafo(s2_rgb)

    def _sentinel_ndvi_loader(self, path):
        '''Return NDVI, NDBI, and NDWI instead of RGB'''
        with rasterio.open(path) as f:
            s2 = f.read()
            s2 = np.transpose(s2,[1,2,0])
            # NIR - RED
            ndvi = (s2[:,:,7] - s2[:,:,3]) / (s2[:,:,7] + s2[:,:,3])
            # SWIR - NIR
            ndbi = (s2[:,:,10] - s2[:,:,7]) / (s2[:,:,10] + s2[:,:,7])
            # NIR - RGB
            ndwi = (s2[:,:,2] - s2[:,:,7]) / (s2[:,:,2] + s2[:,:,7])

            stacked = np.dstack([ndvi, ndwi, ndbi])
            stacked = (stacked + 1.0) / 2.
            compl_trafo = v2.Compose([v2.ToImage(), v2.ToDtype(torch.uint8, scale=True)])
            return compl_trafo(stacked)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.transforms( self.loader( self.image_paths[idx] ) )
        label = self.labels[idx]
        pid = self.pids[idx]

        return img, label, pid

class MapYourCityDataModule(L.LightningDataModule):
    def __init__(self,
                 data_dir, fold, fold_dir, batch_size, num_workers,
                 pin_memory, img_type, transform, sentinel2_mode,
                 model, augment):
        super().__init__()

        self.data_dir = data_dir
        self.fold = fold
        self.fold_dir = fold_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.img_type = img_type
        self.transform = transform
        self.sentinel2_mode = sentinel2_mode
        self.augment = augment
        self.model = model

    def setup(self, stage='fit'):
        '''
        Train, valid and test data
        '''
        self.train_data = MapYourCityDataset(os.path.join(self.data_dir, 'train', 'data'),
                                             os.path.join(self.fold_dir, f'split_train_{self.fold}.csv'),
                                             self.img_type, self.transform, self.sentinel2_mode,
                                             self.model, True, self.augment)
        self.valid_data = MapYourCityDataset(os.path.join(self.data_dir, 'train', 'data'),
                                             os.path.join(self.fold_dir, f'split_valid_{self.fold}.csv'),
                                             self.img_type, self.transform, self.sentinel2_mode,
                                             self.model, False)
        self.test_data = MapYourCityDataset(os.path.join(self.data_dir, 'test', 'data'),
                                             os.path.join(self.data_dir, 'test',  f'test-set.csv'),
                                             self.img_type, self.transform, self.sentinel2_mode,
                                             self.model, False)

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(dataset=self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def valid_dataloader(self):
        return DataLoader(dataset=self.valid_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
