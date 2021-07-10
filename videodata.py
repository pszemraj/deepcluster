"""
This file is based off of fungidata.py and will create image dataset off of videos and load them into the model


"""

import os
from dataclasses import dataclass
from enum import Enum

import numpy as np
# TODO finish converting this file from fungi to process videos
import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset

import img_transforms


class RawData(Enum):
    '''Number of rows in the image raw data'''
    N_ROWS = 15695
    '''Name of headers in raw data file'''
    HEADERS = ['Kingdom', 'Division', 'Subdivision', 'Class', 'Order', 'Family', 'Genus', 'Species', 'InstanceIndex',
               'ImageName']
    '''video level specification names'''
    LEVELS = HEADERS[:-2]


@dataclass
class DataGetKeys:
    '''Shared keys for the return of the datasets __getitem__ dictionary'''
    image: str = 'image'
    label: str = 'label'
    idx: str = 'idx'


#
# Various video Datasets. These are accessed in the Learner via the factory function, `factory`, see below
#
class videoFullBasicData(Dataset):
    '''video Dataset. Properties: full image, basic transformation of image channels, no appended data to __getitem__

    Args:
        csv_file (str): Path to CSV file with table-of-contents of the video raw data
        img_root_dir (str): Path to the root directory of video images
        selector (optional): Pandas IndexSlice or callable that is passed to the Pandas `.loc` method in
            order to select a subset of images on basis of MultiIndex values. Defaults to None.
        iselector (optional): Colletion of integer indices or callable that is passed to the Pandas `.iloc`
            method in order to select a subset of images. This is applied after any `selector` filtering.

    Attributes:
          returnkey : Keys to access values of return dictionary for __getitem__

    '''

    def __init__(self, csv_file, img_root_dir, selector=None, iselector=None, min_dim=224, square=True):
        super(videoFullBasicData, self).__init__()

        self._core = _videoDataCore(csv_file, img_root_dir, selector=selector, iselector=iselector)
        self._core._set_level_attr(self)
        self.returnkey = DataGetKeys()
        del self.returnkey.label
        del self.returnkey.idx
        self._transform = img_transforms.StandardTransform(min_dim=min_dim, square=square)

    def __len__(self):
        return self._core.__len__()

    def __getitem__(self, idx):
        image = self._transform(self._core[idx]['image'])
        return {self.returnkey.image: image}


class videoFullBasicLabelledData(Dataset):
    '''video Dataset. Properties: full image, basic transformation of image channels, label appended to __getitem__

    Args:
        csv_file (str): Path to CSV file with table-of-contents of the video raw data
        img_root_dir (str): Path to the root directory of video images
        label_keys (iterable of str): Collection of strings pass to the Pandas `.query` method in order
            to define subsets of the data that should be assigned integer class labels. If None, the
            indexing of the class returns the image tensor object, if not None, the indexing of the class
            returns the image tensor object and the integer class label.
        selector (optional): Pandas IndexSlice or callable that is passed to the Pandas `.loc` method in
            order to select a subset of images on basis of MultiIndex values. Defaults to None.
        iselector (optional): Colletion of integer indices or callable that is passed to the Pandas `.iloc`
            method in order to select a subset of images. This is applied after any `selector` filtering.
        min_dim (int, optional): min_dim (int): Length of shortest dimension of transformed output image
        square (bool, optional): If True, the source image (after resizing of shortest dimension) is cropped at
            the centre such that output image is square. Defaults to False

    Attributes:
          returnkey : Keys to access values of return dictionary for __getitem__

    '''

    def __init__(self, csv_file, img_root_dir, label_keys, selector=None, iselector=None, min_dim=224, square=False):
        super(videoFullBasicLabelledData, self).__init__()

        self._core = _videoDataCore(csv_file, img_root_dir, selector=selector, iselector=iselector,
                                    label_keys=label_keys)
        self._core._set_level_attr(self)
        self.returnkey = DataGetKeys()
        del self.returnkey.idx
        self._transform = img_transforms.StandardTransform(min_dim=min_dim, square=square)

    def __len__(self):
        return self._core.__len__()

    def __getitem__(self, idx):
        raw_out = self._core[idx]
        image = self._transform(raw_out['image'])
        label = raw_out['label']
        return {self.returnkey.image: image, self.returnkey.label: label}


class videoFullAugLabelledData(Dataset):
    '''video Dataset. Properties: full image, augmentation transformation of image channels, label appended to __getitem__

    Args:
        csv_file (str): Path to CSV file with table-of-contents of the video raw data
        img_root_dir (str): Path to the root directory of video images
        label_keys (iterable of str): Collection of strings pass to the Pandas `.query` method in order
            to define subsets of the data that should be assigned integer class labels. If None, the
            indexing of the class returns the image tensor object, if not None, the indexing of the class
            returns the image tensor object and the integer class label.
        selector (optional): Pandas IndexSlice or callable that is passed to the Pandas `.loc` method in
            order to select a subset of images on basis of MultiIndex values. Defaults to None.
        iselector (optional): Colletion of integer indices or callable that is passed to the Pandas `.iloc`
            method in order to select a subset of images. This is applied after any `selector` filtering.

    Attributes:
          returnkey : Keys to access values of return dictionary for __getitem__

    '''

    def __init__(self, csv_file, img_root_dir, label_keys, aug_multiplicity, aug_label, min_dim=224, square=False,
                 selector=None, iselector=None):
        super(videoFullAugLabelledData, self).__init__()

        self._core = _videoDataCore(csv_file, img_root_dir, selector=selector, iselector=iselector,
                                    label_keys=label_keys)
        self._core._set_level_attr(self)
        self.returnkey = DataGetKeys()
        del self.returnkey.idx

        self._transform = [img_transforms.StandardTransform(min_dim=min_dim, square=square)]

        self.aug_multiplicity = aug_multiplicity
        for k_aug_transform in range(self.aug_multiplicity):
            self._transform.append(img_transforms.DataAugmentTransform(augmentation_label=aug_label,
                                                                       min_dim=min_dim, square=square))

    def __len__(self):
        return self._core.__len__() * self.aug_multiplicity

    def __getitem__(self, idx):
        idx_video = int(np.floor(idx / (1 + self.aug_multiplicity)))
        idx_aug_transform = idx % (1 + self.aug_multiplicity)
        raw_out = self._core[idx_video]
        image = self._transform[idx_aug_transform](raw_out['image'])
        label = raw_out['label']
        return {self.returnkey.image: image, self.returnkey.label: label}


class videoFullBasicIdxData(videoFullBasicData):
    '''video Dataset. Properties: full image, basic transformation of image channels, image index appended data to __getitem__

    Args:
        csv_file (str): Path to CSV file with table-of-contents of the video raw data
        img_root_dir (str): Path to the root directory of video images
        selector (optional): Pandas IndexSlice or callable that is passed to the Pandas `.loc` method in
            order to select a subset of images on basis of MultiIndex values. Defaults to None.
        iselector (optional): Colletion of integer indices or callable that is passed to the Pandas `.iloc`
            method in order to select a subset of images. This is applied after any `selector` filtering.

    Attributes:
          returnkey : Keys to access values of return dictionary for __getitem__

    '''

    def __init__(self, csv_file, img_root_dir, selector=None, iselector=None):
        super(videoFullBasicIdxData, self).__init__(csv_file=csv_file, img_root_dir=img_root_dir,
                                                    selector=selector, iselector=iselector,
                                                    square=True)

        self.returnkey = DataGetKeys()
        del self.returnkey.label

    def __getitem__(self, idx):
        ret_dict = super().__getitem__(idx)
        ret_dict[self.returnkey.idx] = idx
        return ret_dict


class videoGridBasicData(Dataset):
    '''video Dataset. Properties: grid image, basic transformation of image channels, no appended data to __getitem__

    Args:
        csv_file (str): Path to CSV file with table-of-contents of the video raw data
        img_root_dir (str): Path to the root directory of video images
        selector (optional): Pandas IndexSlice or callable that is passed to the Pandas `.loc` method in
            order to select a subset of images on basis of MultiIndex values. Defaults to None.
        iselector (optional): Colletion of integer indices or callable that is passed to the Pandas `.iloc`
            method in order to select a subset of images. This is applied after any `selector` filtering.
        img_input_dim (int): Length and height of square of source image to be sliced by grid. Defaults to 224.
        img_n_splits (int): Number of slices per side, thus total number of slices for one source image
            will be `img_n_splits * img_n_splits`. Defaults to 6.
        crop_step_size (int): Number of pixels between grid lines. Defaults to 32.
        crop_dim (int): Length and height of grid squares. Defaults to 64.

    Attributes:
          returnkey : Keys to access values of return dictionary for __getitem__

    '''

    def __init__(self, csv_file, img_root_dir, selector=None, iselector=None,
                 img_input_dim=224, img_n_splits=6, crop_step_size=32, crop_dim=64):
        super(videoGridBasicData, self).__init__()

        self._core = _videoDataCore(csv_file, img_root_dir, selector=selector, iselector=iselector)
        self._core._set_level_attr(self)
        self.returnkey = DataGetKeys()
        del self.returnkey.label
        del self.returnkey.idx
        self._transform = img_transforms.OverlapGridTransform(img_input_dim=img_input_dim,
                                                              img_n_splits=img_n_splits,
                                                              crop_step_size=crop_step_size,
                                                              crop_dim=crop_dim)

    def __len__(self):
        return self._core.__len__() * self._transform.n_blocks

    def __getitem__(self, idx):
        idx_video = int(np.floor(idx / self._transform.n_blocks))
        idx_sub = idx % self._transform.n_blocks
        raw_out = self._core[idx_video]
        img_crops = self._transform(raw_out['image'])
        return {self.returnkey.image: img_crops[idx_sub]}


class videoGridBasicIdxData(videoGridBasicData):
    '''video Dataset. Properties: grid image, basic transformation of image channels, image index appended data to __getitem__

    Args:
        csv_file (str): Path to CSV file with table-of-contents of the video raw data
        img_root_dir (str): Path to the root directory of video images
        selector (optional): Pandas IndexSlice or callable that is passed to the Pandas `.loc` method in
            order to select a subset of images on basis of MultiIndex values. Defaults to None.
        iselector (optional): Colletion of integer indices or callable that is passed to the Pandas `.iloc`
            method in order to select a subset of images. This is applied after any `selector` filtering.
        img_input_dim (int): Length and height of square of source image to be sliced by grid. Defaults to 224.
        img_n_splits (int): Number of slices per side, thus total number of slices for one source image
            will be `img_n_splits * img_n_splits`. Defaults to 6.
        crop_step_size (int): Number of pixels between grid lines. Defaults to 32.
        crop_dim (int): Length and height of grid squares. Defaults to 64.

    Attributes:
          returnkey : Keys to access values of return dictionary for __getitem__

    '''

    def __init__(self, csv_file, img_root_dir, selector=None, iselector=None,
                 img_input_dim=224, img_n_splits=6, crop_step_size=32, crop_dim=64):
        super(videoGridBasicIdxData, self).__init__(csv_file, img_root_dir, selector=selector, iselector=iselector,
                                                    img_input_dim=img_input_dim, img_n_splits=img_n_splits,
                                                    crop_step_size=crop_step_size, crop_dim=crop_dim)

        self.returnkey = DataGetKeys()
        del self.returnkey.label

    def __getitem__(self, idx):
        ret_dict = super().__getitem__(idx)
        ret_dict[self.returnkey.idx] = idx
        return ret_dict


class _videoDataCore(object):
    '''The core data class that contains all logic related to the raw data files and their construction.

    Args:
        csv_file (str): CSV file with table-of-contents of the video raw data
        img_root_dir (str): Path to the root directory of video images
        selector (optional): Pandas IndexSlice or callable that is passed to the Pandas `.loc` method in
            order to select a subset of images on basis of MultiIndex values. Defaults to None.
        iselector (optional): Colletion of integer indices or callable that is passed to the Pandas `.iloc`
            method in order to select a subset of images. This is applied after any `selector` filtering.
        label_keys (iterable of str): Collection of strings pass to the Pandas `.query` method in order
            to define subsets of the data that should be assigned integer class labels. If None, the
            indexing of the class returns the image tensor object, if not None, the indexing of the class
            returns the image tensor object and the integer class label.

    '''

    def __init__(self, csv_file, img_root_dir, selector=None, iselector=None, label_keys=None):

        self.img_toc = pd.read_csv(csv_file, index_col=(0, 1, 2, 3, 4, 5, 6, 7, 8))
        self.img_root_dir = img_root_dir
        self.label_keys = label_keys

        if not selector is None:
            self.img_toc = self.img_toc.loc[selector]

        if not iselector is None:
            self.img_toc = self.img_toc.iloc[iselector]

        if not label_keys is None:
            self.img_toc = pd.concat(self._assign_label(self.label_keys))

    def _set_level_attr(self, obj):
        '''Add attributes to input object that denotes how numerous different levels of video data are.

        Args:
            obj : Object to add attributes to. Typically the `self` of a video Dataset

        '''
        for level in RawData.LEVELS.value:
            setattr(obj, 'n_{}'.format(level.lower()), self._n_x(level))
            setattr(obj, 'n_instances_{}'.format(level.lower()), self._n_instances_x(level))

    def __getitem__(self, idx):
        '''Retrieve raw data from disk

        Args:
            idx: index to retrieve

        Returns:
            raw_data (dict): Raw data retrieved, which is a dictionary with keys "image" and "label", the former
                with value the image raw data, as represented by `skimage.io.imread`, the latter with value the
                associated image label (if applicable).

        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.img_toc.iloc[idx]
        img_name = row[0]

        rel_path = list(row.name)[1:-1]
        rel_path.append(img_name)
        img_name = os.path.join(self.img_root_dir, *tuple(rel_path))
        image = io.imread(img_name)

        if not self.label_keys is None:
            label = row[1]
        else:
            label = None

        return {'image': image, 'label': label}

    def __len__(self):
        return len(self.img_toc)

    def _assign_label(self, l_keys, int_start=0):
        '''Assign label to data based on family, genus, species selections

        The label keys are query strings for Pandas, as described here:
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html

        Each query string define a class. The query string can refer to individual species, genus, family etc. or
        collections thereof. For example, the tuple `('Family == "Cantharellaceae"', 'Family == "Amanitaceae"')`
        defines two labels for all video in either of the two families.

        Args:
            l_keys (iterable): list of query strings for Pandas DataFrame, where each query string defines a class to be
                assigned a unique integer label.
            int_start (int, optional): the first integer class label. Defaults to 0.

        Returns:
            category_slices (list): List of DataFrames each corresponding to the categories. The list can be
                concatenated in order to form a single DataFrame

        '''
        category_slices = []
        for label_int, query_label in enumerate(l_keys):
            subset_label = self.img_toc.query(query_label)

            if len(subset_label) > 0:
                subset_label.loc[:, 'ClassLabel'] = label_int + int_start
                subset_label = subset_label.astype({'ClassLabel': 'int64'})
                category_slices.append(subset_label)

        return category_slices

    def _n_x(self, x_label):
        '''Compute number of distinct types of video at a given step in the hierarchy'''
        return len(self.img_toc.groupby(x_label))

    def _n_instances_x(self, x_label):
        '''Compute number of images for each type of video at a given step in the hierarchy'''
        return self.img_toc.groupby(x_label).count()[RawData.HEADERS.value[-1]].to_dict()

    @property
    def label_semantics(self):
        '''The dictionary that maps '''
        return dict([(count, label_select) for count, label_select in enumerate(self.label_keys)])


#
# Various video Dataset builders, which instantiate a video Dataset. The builders are used by the factory
# function `factory`, see below.
#

class videoFullBasicDataBuilder(object):
    def __init__(self):
        self._instance = None

    def __call__(self, csv_file, img_root_dir, selector=None, iselector=None,
                 img_input_dim=224, square=False, **_ignored):
        self._instance = videoFullBasicData(csv_file=csv_file, img_root_dir=img_root_dir,
                                            selector=selector, iselector=iselector, square=square,
                                            min_dim=img_input_dim)
        return self._instance


class videoFullBasicLabelledDataBuilder(object):
    def __init__(self):
        self._instance = None

    def __call__(self, csv_file, img_root_dir, label_keys, selector=None, iselector=None,
                 min_dim=224, square=False, **_ignored):
        self._instance = videoFullBasicLabelledData(csv_file=csv_file, img_root_dir=img_root_dir,
                                                    label_keys=label_keys,
                                                    selector=selector, iselector=iselector,
                                                    min_dim=min_dim, square=square)
        return self._instance


class videoFullAugLabelledDataBuilder(object):
    def __init__(self):
        self._instance = None

    def __call__(self, csv_file, img_root_dir, label_keys, aug_multiplicity, aug_label,
                 min_dim=224, square=False, selector=None, iselector=None, **_ignored):
        self._instance = videoFullAugLabelledData(csv_file=csv_file, img_root_dir=img_root_dir,
                                                  label_keys=label_keys,
                                                  min_dim=min_dim, square=square,
                                                  aug_multiplicity=aug_multiplicity,
                                                  aug_label=aug_label,
                                                  selector=selector, iselector=iselector)
        return self._instance


class videoFullBasicIdxDataBuilder(object):
    def __init__(self):
        self._instance = None

    def __call__(self, csv_file, img_root_dir, selector=None, iselector=None, **_ignored):
        self._instance = videoFullBasicIdxData(csv_file=csv_file, img_root_dir=img_root_dir,
                                               selector=selector, iselector=iselector)
        return self._instance


class videoGridBasicDataBuilder(object):
    def __init__(self):
        self._instance = None

    def __call__(self, csv_file, img_root_dir,
                 img_input_dim, img_n_splits, crop_step_size, crop_dim,
                 selector=None, iselector=None, **_ignored):
        self._instance = videoGridBasicData(csv_file=csv_file, img_root_dir=img_root_dir,
                                            selector=selector, iselector=iselector,
                                            img_input_dim=img_input_dim, img_n_splits=img_n_splits,
                                            crop_step_size=crop_step_size, crop_dim=crop_dim)
        return self._instance


class videoGridBasicIdxDataBuilder(object):
    def __init__(self):
        self._instance = None

    def __call__(self, csv_file, img_root_dir,
                 img_input_dim, img_n_splits, crop_step_size, crop_dim,
                 selector=None, iselector=None, **_ignored):
        self._instance = videoGridBasicIdxData(csv_file=csv_file, img_root_dir=img_root_dir,
                                               selector=selector, iselector=iselector,
                                               img_input_dim=img_input_dim, img_n_splits=img_n_splits,
                                               crop_step_size=crop_step_size, crop_dim=crop_dim)
        return self._instance


class videoDataFactory(object):
    '''Interface to video data factories.

    Typical usage involves the invocation of the `create` method, which returns a specific video dataset.

    '''

    def __init__(self):
        self._builders = {}

    def register_builder(self, key, builder):
        '''Register a builder

        Args:
            key (str): Key to the builder, which can be invoked by `create` method
            builder: A video Data Builder instance

        '''
        self._builders[key] = builder

    @property
    def keys(self):
        return self._builders.keys()

    def create(self, key, csv_file, img_root_dir, selector=None, iselector=None, **kwargs):
        '''Method to create a video data set through a uniform interface

        Args:
            key (str): The name of the type of dataset to create. The available keys available in attribute `keys`
            csv_file (str): CSV file with table-of-contents of the video raw data
            img_root_dir (str): Path to the root directory of video images
            selector (optional): Pandas IndexSlice or callable that is passed to the Pandas `.loc` method in
                order to select a subset of images on basis of MultiIndex values. Defaults to None.
            iselector (optional): Colletion of integer indices or callable that is passed to the Pandas `.iloc`
                method in order to select a subset of images. This is applied after any `selector` filtering.
            **kwargs: Additional arguments to be passed to the specific dataset builder.
        '''
        try:
            builder = self._builders[key]
        except KeyError:
            raise ValueError('Unregistered data builder: {}'.format(key))
        return builder(csv_file=csv_file, img_root_dir=img_root_dir, selector=selector, iselector=iselector,
                       **kwargs)


# The available pre-registrered video data set factory method. It can be imported and the `create` method has a
# uniform interface for the creation of one of many possible variants of a video data set.
factory = videoDataFactory()
factory.register_builder('full basic', videoFullBasicDataBuilder())
factory.register_builder('full basic labelled', videoFullBasicLabelledDataBuilder())
factory.register_builder('full aug labelled', videoFullAugLabelledDataBuilder())
factory.register_builder('full basic idx', videoFullBasicIdxDataBuilder())
factory.register_builder('grid basic', videoGridBasicDataBuilder())
factory.register_builder('grid basic idx', videoGridBasicIdxDataBuilder())
