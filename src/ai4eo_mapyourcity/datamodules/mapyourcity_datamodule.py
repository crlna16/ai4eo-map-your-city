#!/usr/bin/env python
'''Datamodule for the MapYourCity challenge'''

from typing import Dict

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
    Base class for MapYourCity challenge data.

    Attributes:
        img_file (str): Image file name.
        config (Dict): Model config. Assigned from pretrained model if available.
        input_size (int): Image input size.
        transforms (torchvision.v2.Compose): Composition of transforms to apply on the image.
        pids (List): List of all PIDs.
        labels (List): List of all labels.
        image_paths (List): List of all image paths.
        image_roots (List): List of all image root folders.
        fake_streetview (PIL.Image): Streetview photo to use in case of missing modality.
    '''
    def __init__(self,
                 options: Dict,
                 split: str = 'train'
                 ):
        '''
        Initialize the MapYourCityDataset.

        If the model config specifies a pretrained TIMM model, retrieve its configuration.
        Else assign a config for consistency.

        Construct the image transforms based on the config.

        Construct all paths to image root directories for the given data split.

        Store all PIDs and labels.

        Arguments:
            options (Dict): Dictionary with dataset options.
            split (str, optional): Dataset split. Defaults to 'train'.
        '''
        super().__init__()

        self.img_file = options['img_file']

        # get config from pretrained model
        if options['model_id'] is None:
            self.config = {'mean':[0.5, 0.5, 0.5], 'std':[0.5, 0.5, 0.5]}
        else:
            self.config = timm.data.resolve_model_data_config(options['model_id'])
            self.input_size = int( self.config['input_size'][1] / self.config['crop_pct'] )

        self.transforms = lambda x: x  # assigned by subclass # TODO move to function

        # list of files
        if split in ['train', 'valid']:
            csv_path = os.path.join(options["fold_dir"],
                                    options["fold_key"],
                                    f'split_{split}_{options["fold"]}.csv')
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
            x = np.zeros([284,284,3])
            x[...,0] = 0.4850 * 256
            x[...,1] = 0.4560 * 256
            x[...,2] = 0.4060 * 256
            self.fake_streetview = Image.fromarray(x.astype(np.uint8))

    def loader(self, path: str):
        '''
        Open the image file at path.

        Needs to be implemented in any subclass.

        Args:
            path (str): Full path to image file.

        Returns:
            Image.
        '''
        raise NotImplementedError('Loader needs to be implemented in the subclass.')

    def __len__(self):
        '''
        Returns:
            int: Length of the dataset.
        '''
        return len(self.labels)

    def __getitem__(self, idx: int):
        '''
        Retrieve an item from the dataset.

        Args:
            idx (int): Index to retrieve.

        Returns:
            PIL.Image: Transformed image.
            int: Class label.
            str: Sample PID.
        '''
        img = self.transforms( self.loader( self.image_paths[idx] ) )
        label = self.labels[idx]
        pid = self.pids[idx]

        return img, label, pid

class CombinedDataset(Dataset):
    '''
    Dataset for two or three input modalities.

    Attributes:
        use_topview (bool): If True, use topview (orthophoto) samples.
        topview_options (Dict): Dictionary with options to create topview dataset.
        topview_dataset (PhotoDataset): Topview dataset.
        use_streetview (bool): If True, use streetview samples.
        streetview_options (Dict): Dictionary with options to create streetview dataset.
        streetview_dataset (PhotoDataset): Streetview dataset.
        use_sentinel2 (bool): If True, use Sentinel-2 samples.
        sentinel2_options (Dict): Dictionary with options to create Sentinel-2 dataset.
        sentinel2_dataset (Sentinel2Dataset): Sentinel-2 dataset.
        sources (int): Number of modalities.
    '''

    def __init__(self,
                 options: Dict,
                 split: str = 'train'
                 ):
        '''
        Create datasets for topview, streetview, and Sentinel-2 data as specified.

        Arguments:
            options (Dict): Dictionary with dataset options.
            split (str, optional): Dataset split. Defaults to 'train'.
        '''
        super().__init__()

        self.use_topview = options['use_topview']
        self.topview_options = options['dataset_options_topview']
        self.use_streetview = options['use_streetview']
        self.streetview_options = options['dataset_options_streetview']
        self.use_sentinel2 = options['use_sentinel2']
        self.sentinel2_options = options['dataset_options_sentinel2']

        if self.use_topview:
            self.topview_dataset = PhotoDataset({**options, **options['dataset_options_topview']},
                                                split)
        if self.use_streetview:
            self.streetview_dataset = PhotoDataset({**options, **options['dataset_options_streetview']},
                                                   split)
        if self.use_sentinel2:
            self.sentinel2_dataset = Sentinel2Dataset({**options, **options['dataset_options_sentinel2']},
                                                      split)

        self.sources = int(self.use_topview) + int(self.use_streetview) + int(self.use_sentinel2)

    def __len__(self):
        '''
        Returns:
            int: Length of the dataset.
        '''
        if self.use_topview:
            return len(self.topview_dataset.labels)
        if self.use_streetview:
            return len(self.streetview_dataset.labels)
        if self.use_sentinel2:
            return len(self.sentinel2_dataset.labels)
        return None

    def __getitem__(self, idx: int):

        imgs = []

        if self.use_topview:
            img, label, pid = self.topview_dataset.__getitem__(idx)
            imgs.append(img)
        if self.use_streetview:
            img, label, pid = self.streetview_dataset.__getitem__(idx)
            imgs.append(img)
        if self.use_sentinel2:
            img, label, pid = self.sentinel2_dataset.__getitem__(idx)
            imgs.append(img)

        return tuple(imgs), label, pid

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
                 options: Dict,
                 split: str = 'train'
                 ):
        super().__init__(options, split)

        match options['transform']:
            case 'default':
                if split == 'train':
                    trafo1 = [ v2.RandomResizedCrop(size=(self.input_size, self.input_size),
                                                    scale=(0.08, 1.0),
                                                    ratio=(0.75, 1.3333),
                                                    interpolation=3),
                               v2.RandomHorizontalFlip(p=0.5),
                               v2.ColorJitter(brightness=(0.6, 1.4),
                                              contrast=(0.6, 1.4),
                                              saturation=(0.6, 1.4),
                                              hue=None),]
                elif split in ['valid', 'test']:
                    trafo1 = [ v2.Resize(size=(self.input_size, self.input_size), interpolation=3),]

            case 'resize':
                trafo1 = [ v2.Resize(size=(self.input_size, self.input_size), interpolation=2), ]

            case 'center_crop':
                trafo1 = [ v2.CenterCrop(size=(self.input_size, self.input_size)), ]

            case _:
                raise ValueError('Invalid choice:', options['transform'])

        trafo0 = [v2.ToImage()]
        trafo2 = [v2.ToDtype(torch.float32, scale=True),
                  v2.Normalize(mean=self.config['mean'], std=self.config['std'])]

        self.transforms = v2.Compose(trafo0 + trafo1 + trafo2)

    def loader(self, path: str):
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
                 options: Dict,
                 split: str = 'train'
                 ):
        super().__init__(options, split)

        self.transform = options['transform']

        self.reference_bands = ['B01','B02', 'B03', 'B04','B05','B06',
                                'B07','B08','B8A','B09','B11','B12']

        match options['transform']:
            case 'default':
                self.use_ndvi = options['use_ndvi']
                self.use_ndwi = options['use_ndwi']
                self.use_ndbi = options['use_ndbi']
                self.use_bands = options['use_bands']

                self.channel_idx = [self.reference_bands.index(b) for b in self.use_bands]

                self.transforms = v2.ToDtype(torch.float32, scale=True)
            case 'patch':
                # Create a 3 x 128 x 128 patch
                self.loader = self._sentinel2_patch_loader
                if split == 'train':
                    trafo1 = [ v2.Resize(size=(self.input_size, self.input_size),
                                         interpolation=3),
                               v2.ColorJitter(brightness=(0.6, 1.4),
                                              contrast=(0.6, 1.4),
                                              saturation=(0.6, 1.4),
                                              hue=None),]
                elif split in ['valid', 'test']:
                    trafo1 = [ v2.Resize(size=(self.input_size, self.input_size), interpolation=3),]

                trafo0 = [v2.ToImage()]
                trafo2 = [v2.ToDtype(torch.float32, scale=True),
                          v2.Normalize(mean=self.config['mean'], std=self.config['std'])]

                self.transforms = v2.Compose(trafo0 + trafo1 + trafo2)

    def loader(self, path: str):
        '''
        Open the image file at path.

        Returns the image file as patch or sub-selection of channels.

        TODO separate the load logic from the patch logic.

        Args:
            path (str): Full path to image file.

        Returns:
            Image.
        '''

        match self.transform:
            case 'default':
                return self._sentinel2_loader(path)
            case 'patch':
                return self._sentinel2_patch_loader(path)
            case _:
                raise ValueError('Invalid transform option')

    def _sentinel2_patch_loader(self, path: str):
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


    def _sentinel2_loader(self, path: str):
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
                 batch_size: int,
                 num_workers: int,
                 pin_memory: bool,
                 dataset_options: Dict
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
        '''Returns: DataLoader'''
        return DataLoader(dataset=self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=True)

    def valid_dataloader(self):
        '''Returns: DataLoader'''
        return DataLoader(dataset=self.valid_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=False)

    def test_dataloader(self):
        '''Returns: DataLoader'''
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
                 batch_size: int,
                 num_workers: int,
                 pin_memory: bool,
                 dataset_options: Dict
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
        Construct train, valid, and test data
        '''

        self.train_data = CombinedDataset(self.dataset_options, split='train')
        self.valid_data = CombinedDataset(self.dataset_options, split='valid')
        self.test_data = CombinedDataset(self.dataset_options, split='test')

    def prepare_data(self):
        pass

    def train_dataloader(self):
        '''Returns: DataLoader'''
        return DataLoader(dataset=self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=True)

    def valid_dataloader(self):
        '''Returns: DataLoader'''
        return DataLoader(dataset=self.valid_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=False)

    def test_dataloader(self):
        '''Returns: DataLoader'''
        return DataLoader(dataset=self.test_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=False)
