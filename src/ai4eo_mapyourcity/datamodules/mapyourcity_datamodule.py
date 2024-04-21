#!/usr/bin/env python

import os
import torch
import numpy as np
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

        # get config from pretrained model 
        if options['model_id'] is None:
            self.config = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        else:
            self.config = timm.data.resolve_model_data_config(options['model_id'])
            self.input_size = int( self.config['input_size'][1] / self.config['crop_pct'] )

        self.loader = None # assigned by subclass
        self.transforms = None # assigned by subclass

        # list of files
        if split in ['train', 'valid']:
            csv_path = os.path.join(options["fold_dir"], options["fold_key"], f'split_{split}_{options["fold"]}.csv')
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
        self.image_roots = [os.path.join(data_path, pid) for pid in self.pids]

        # not all folders have a streetview image
        if split == 'test' and self.img_file == 'street.jpg':
            self.fake_streetview = Image.fromarray(np.zeros([512,1024,3]).astype(np.uint8))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.transforms( self.loader( self.image_paths[idx] ) )
        label = self.labels[idx]
        pid = self.pids[idx]

        return img, label, pid

class CombinedDataset(MapYourCityDataset):
    '''
    TODO
    '''

    def __init__(self,
                 options,
                 split='train'
                 ):
        super().__init__(options, split)

        self.use_topview = options['use_topview']
        self.topview_options = options['dataset_options_topview']
        self.use_streetview = options['use_streetview']
        self.streetview_options = options['dataset_options_streetview']
        self.use_sentinel2 = options['use_sentinel2']
        self.sentinel2_options = options['dataset_options_sentinel2']

        self.loader = self._combined_loader
        self.sources = int(self.use_topview) + int(self.use_streetview) + int(self.use_sentinel2)

        # TODO transforms
        match options['transform']:
            case 'default':
                if split == 'train':
                    trafo1 = [ v2.RandomResizedCrop(size=(self.input_size, self.input_size), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=3),
                               v2.RandomHorizontalFlip(p=0.5),
                               v2.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4), hue=None),]
                elif split in ['valid', 'test']:
                    trafo1 = [ v2.Resize(size=(self.input_size, self.input_size), interpolation=3),]

            case 'resize':
                trafo1 = [ v2.Resize(size=(self.input_size, self.input_size), interpolation=3), ]

            case _:
                raise ValueError('Invalid choice:', transform)

        trafo0 = [v2.ToImage()] 
        trafo2 = [v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=self.config['mean'], std=self.config['std'])]

        self.transforms = v2.Compose(trafo0 + trafo1 + trafo2)

    def _combined_loader(self, path):
        '''
        Loader for different samples in one PID

        Returns: 
            list / tuple TODO of images at location
        '''

        imgs = []

        if self.use_topview:
            with open(os.path.join(path, self.topview_options['img_file']), "rb") as f:
                img = Image.open(f)
                img.load()

            imgs.append(img.convert("RGB"))

        if self.use_streetview:
            if not os.path.exists(os.path.join(path, self.streetview_options['img_file'])):
                img = self.fake_streetview
                return img.convert("RGB")

            with open(os.path.join(path, self.streetview_options['img_file']), "rb") as f:
                img = Image.open(f)
                img.load()

            imgs.append(img.convert("RGB"))

        if self.use_sentinel2:
            
            with rasterio.open(os.path.join(path, self.sentinel2_options['img_file'])) as f:
                # Need to calculate indices before selecting channels
                s2 = f.read() * 3e-4

            # NIR - RED
            ndvi = (s2[7] - s2[3]) / (s2[7] + s2[3])
            #ndvi = ndvi[np.newaxis, ...]
            # SWIR - NIR
            ndbi = (s2[10] - s2[7]) / (s2[10] + s2[7])
            #ndbi = ndbi[np.newaxis, ...]
            # NIR - RGB
            ndwi = (s2[2] - s2[7]) / (s2[2] + s2[7])
            #ndwi = ndwi[np.newaxis, ...]

            p_tl = np.stack([ndvi, ndwi, ndbi])
            p_tr = s2[[3,2,1]] # top right - RGB
            p_bl = s2[[4,5,6]] # bottom left - Vegetation red edge
            p_br = s2[[8, 10, 11]] # bottom right - NIR/SWIR

            patch = np.dstack([np.hstack([p_tl, p_tr]), np.hstack([p_bl, p_br])])
            patch = np.nan_to_num(patch).transpose(2,1,0)

            imgs.append( patch )

        return tuple(imgs)

    def __getitem__(self, idx):
        label = self.labels[idx]
        pid = self.pids[idx]

        if self.sources == 2:
            img0, img1 = self.loader(self.image_roots[idx])
            img0 = self.transforms(img0)
            img1 = self.transforms(img1)
            
            return img0, img1, label, pid

        elif self.sources == 3:
            img0, img1, img2 = self.loader(self.image_roots[idx])
            img0 = self.transforms(img0)
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)

            return img0, img1, img2, label, pid

class PhotoDataset(MapYourCityDataset):
    '''
    Dataset for MapYourCity data
    - street photos
    - orthophotos

    Arguments:
        options - dict with options
        split - specifies split
    '''
    def __init__(self,
                 options,
                 split='train'
                 ):
        super().__init__(options, split)

        self.loader = self._photo_loader

        match options['transform']:
            case 'default':
                if split == 'train':
                    trafo1 = [ v2.RandomResizedCrop(size=(self.input_size, self.input_size), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=3),
                               v2.RandomHorizontalFlip(p=0.5),
                               v2.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4), hue=None),]
                elif split in ['valid', 'test']:
                    trafo1 = [ v2.Resize(size=(self.input_size, self.input_size), interpolation=3),]

            case 'resize':
                trafo1 = [ v2.Resize(size=(self.input_size, self.input_size), interpolation=2), ]

            case 'center_crop':
                trafo1 = [ v2.CenterCrop(size=(self.input_size, self.input_size)), ]

            case _:
                raise ValueError('Invalid choice:', transform)

        trafo0 = [v2.ToImage()] 
        trafo2 = [v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=self.config['mean'], std=self.config['std'])]

        self.transforms = v2.Compose(trafo0 + trafo1 + trafo2)

    def _photo_loader(self, path):
        if not os.path.exists(path):
            img = self.fake_streetview
            return img.convert("RGB")

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

        self.reference_bands = ['B01','B02', 'B03', 'B04','B05','B06','B07','B08','B8A','B09','B11','B12']

        match options['transform']:
            case 'default':
                self.use_ndvi = options['use_ndvi']
                self.use_ndwi = options['use_ndwi']
                self.use_ndbi = options['use_ndbi']
                self.use_bands = options['use_bands']

                self.channel_idx = [self.reference_bands.index(b) for b in self.use_bands]

                self.loader = self._sentinel2_loader

                mean = [0., 0., 0.]
                std = [1., 1., 1.]

                # TODO add augmentation
                self.transforms = v2.ToDtype(torch.float32, scale=True)
            case 'patch':
                # Create a 3 x 128 x 128 patch
                self.loader = self._sentinel2_patch_loader
                # TODO add augmentation
                self.transforms = v2.Compose([v2.ToImage(),
                                              v2.Resize(self.input_size), 
                                              v2.ToDtype(torch.float32, scale=True)])
                                              #v2.Normalize(mean=self.config['mean'], std=self.config['std'])])

    def _sentinel2_patch_loader(self, path):
        '''
        Load Sentinel-2 and create patches by stacking the 12 
        channels side by side --> 3 x 128 x 128

        The channels with 60m resolution are replaced by
        normalized indices
        
        Apply factor of 3 x 10^-4 as in demo notebook
        
        Returns:
          Stacked array, channels first (C, W, H)
        '''
        with rasterio.open(path) as f:
            # Need to calculate indices before selecting channels
            s2 = f.read() * 3e-4

        # NIR - RED
        ndvi = (s2[7] - s2[3]) / (s2[7] + s2[3])
        #ndvi = ndvi[np.newaxis, ...]
        # SWIR - NIR
        ndbi = (s2[10] - s2[7]) / (s2[10] + s2[7])
        #ndbi = ndbi[np.newaxis, ...]
        # NIR - RGB
        ndwi = (s2[2] - s2[7]) / (s2[2] + s2[7])
        #ndwi = ndwi[np.newaxis, ...]

        p_tl = np.stack([ndvi, ndwi, ndbi])
        p_tr = s2[[3,2,1]] # top right - RGB
        p_bl = s2[[4,5,6]] # bottom left - Vegetation red edge
        p_br = s2[[8, 10, 11]] # bottom right - NIR/SWIR

        patch = np.dstack([np.hstack([p_tl, p_tr]), np.hstack([p_bl, p_br])])
        patch = np.nan_to_num(patch).transpose(2,1,0)

        return patch


    def _sentinel2_loader(self, path):
        '''
        Load Sentinel-2 and extract the requested channels
        
        Apply factor of 3 x 10^-4 as in demo notebook
        
        Returns:
          Stacked array, channels first (C, W, H)
        '''
        with rasterio.open(path) as f:
            # Need to calculate indices before selecting channels
            s2 = f.read() * 3e-4
            # NIR - RED
            ndvi = (s2[7] - s2[3]) / (s2[7] + s2[3])
            ndvi = ndvi[np.newaxis, ...]
            # SWIR - NIR
            ndbi = (s2[10] - s2[7]) / (s2[10] + s2[7])
            ndbi = ndbi[np.newaxis, ...]
            # NIR - RGB
            ndwi = (s2[2] - s2[7]) / (s2[2] + s2[7])
            ndwi = ndwi[np.newaxis, ...]

            stacked = s2[self.channel_idx]
            if self.use_ndvi:
                stacked = np.vstack([stacked, ndvi])
            if self.use_ndwi:
                stacked = np.vstack([stacked, ndwi])
            if self.use_ndbi:
                stacked = np.vstack([stacked, ndbi])

            stacked = np.nan_to_num(stacked)

            return stacked.astype(np.float32)

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

        self.train_data = None
        self.valid_data = None
        self.test_data = None

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

class MapYourCityCombinedDataModule(L.LightningDataModule):
    '''
    Generic datamodule for MapYourCity data
    Combine two or three input sources

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

        self.train_data = None
        self.valid_data = None
        self.test_data = None

    def setup(self, stage='fit'):
        '''
        Train, valid and test data
        '''

        self.train_data = CombinedDataset(self.dataset_options, split='train')
        self.valid_data = CombinedDataset(self.dataset_options, split='valid')
        self.test_data = CombinedDataset(self.dataset_options, split='test')

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(dataset=self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=True)

    def valid_dataloader(self):
        return DataLoader(dataset=self.valid_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=False)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=False)
