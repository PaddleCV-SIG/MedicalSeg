# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import paddle
import numpy as np
from PIL import Image

from paddleseg3d.cvlibs import manager
from paddleseg3d.transforms import Compose
import paddleseg3d.transforms.functional as F


@manager.DATASETS.add_component
class MedicalDataset(paddle.io.Dataset):
    """
    Pass in a custom dataset that conforms to the format.

    Args:
        transforms (list): Transforms for image.
        dataset_root (str): The dataset directory.
        num_classes (int): Number of classes.
        mode (str, optional): which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
        train_path (str, optional): The train dataset file. When mode is 'train', train_path is necessary.
            The contents of train_path file are as follow:
            image1.jpg ground_truth1.png
            image2.jpg ground_truth2.png
        val_path (str. optional): The evaluation dataset file. When mode is 'val', val_path is necessary.
            The contents is the same as train_path
        test_path (str, optional): The test dataset file. When mode is 'test', test_path is necessary.
            The annotation file is not necessary in test_path file.
        separator (str, optional): The separator of dataset list. Default: ' '.
        edge (bool, optional): Whether to compute edge while training. Default: False

        Examples:

            import paddleseg3d.transforms as T
            from paddleseg.datasets import MedicalDataset

            transforms = [T.RandomPaddingCrop(crop_size=(512,512)), T.Normalize()]
            dataset_root = 'dataset_root_path'
            train_path = 'train_path'
            num_classes = 2
            dataset = MedicalDataset(transforms = transforms,
                              dataset_root = dataset_root,
                              num_classes = 2,
                              train_path = train_path,
                              mode = 'train')

    """

    def __init__(self,
                 transforms,
                 dataset_root,
                 result_dir,
                 num_classes,
                 mode='train',
                 post_transforms=None,
                 train_path=None,
                 val_path=None,
                 test_path=None,
                 separator=' ',
                 ignore_index=255,
                 edge=False):
        self.dataset_root = dataset_root
        self.result_dir = result_dir
        self.transforms = Compose(transforms)
        self.post_transforms = Compose(post_transforms) if post_transforms is not None else None
        self.file_list = list()
        self.mode = mode.lower()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.edge = edge

        if self.mode not in ['train', 'val', 'test']:
            raise ValueError(
                "mode should be 'train', 'val' or 'test', but got {}.".format(
                    self.mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        if not os.path.exists(self.dataset_root):
            raise FileNotFoundError('there is not `dataset_root`: {}.'.format(
                self.dataset_root))

        if self.mode == 'train':
            if train_path is None:
                raise ValueError(
                    'When `mode` is "train", `train_path` is necessary, but it is None.'
                )
            elif not os.path.exists(train_path):
                raise FileNotFoundError(
                    '`train_path` is not found: {}'.format(train_path))

        elif self.mode == 'val':
            if val_path is None:
                raise ValueError(
                    'When `mode` is "val", `val_path` is necessary, but it is None.'
                )
            elif not os.path.exists(val_path):
                raise FileNotFoundError(
                    '`val_path` is not found: {}'.format(val_path))

        elif self.mode == "test":
            if test_path is None:
                raise ValueError(
                    'When `mode` is "test", `test_path` is necessary, but it is None.'
                )
            elif not os.path.exists(test_path):
                raise FileNotFoundError(
                    '`test_path` is not found: {}'.format(test_path))

    def __getitem__(self, idx):
        raise NotImplementedError

    def post_transform(self, pred, label=None):
        """Save the preprocessed images to the result_dir"""
        if self.post_transforms is not None:
            if label is None:
                pred = map(lambda x: self.post_transforms(x), pred)
            else:
                pred, label = map(lambda x, y: self.post_transforms(x, y), zip(pred, label))
        return pred if label is None else (pred, label)

    def __len__(self):
        raise NotImplementedError
