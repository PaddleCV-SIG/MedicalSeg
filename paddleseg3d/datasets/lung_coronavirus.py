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
import sys
import numpy as np

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.."))

from paddleseg3d.utils.download import download_file_and_uncompress
from paddleseg3d.utils.env_util import seg_env
from paddleseg3d.cvlibs import manager
from paddleseg3d.transforms import Compose
from paddleseg3d.datasets import MedicalDataset

URL = ' '  # todo: add coronavirus url


@manager.DATASETS.add_component
class LungCoronavirus(MedicalDataset):
    """
    The Lung cornavirus dataset is ...(todo: add link and description)

    Args:
        dataset_root (str): The dataset directory. Default: None
        result_root(str): The directory to save the result file. Default: None
        transforms (list): Transforms for image.
        mode (str, optional): Which part of dataset to use. it is one of ('train', 'val'). Default: 'train'.
    """
    NUM_CLASSES = 2

    def __init__(self,
                 dataset_root=None,
                 result_dir=None,
                 transforms=None,
                 edge=False,
                 mode='train'):
        self.dataset_root = dataset_root
        self.result_dir = result_dir
        self.transforms = Compose(transforms)
        self.mode = mode.lower()
        self.file_list = list()
        self.num_classes = self.NUM_CLASSES
        self.ignore_index = 255  # todo: if labels only have 1/0, thus ignore_index is not necessary

        if mode not in ['train', 'val']:
            raise ValueError(
                "`mode` should be 'train' or 'val', but got {}.".format(mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        if self.dataset_root is None:
            self.dataset_root = download_file_and_uncompress(
                url=URL,
                savepath=seg_env.DATA_HOME,
                extrapath=seg_env.DATA_HOME)
        elif not os.path.exists(self.dataset_root):
            self.dataset_root = os.path.normpath(self.dataset_root)
            savepath, extraname = self.dataset_root.rsplit(
                sep=os.path.sep, maxsplit=1)
            self.dataset_root = download_file_and_uncompress(
                url=URL,
                savepath=savepath,
                extrapath=savepath,
                extraname=extraname)

        if mode == 'train':
            file_path = os.path.join(self.dataset_root, 'train_list.txt')
        elif mode == 'val':
            file_path = os.path.join(self.dataset_root, 'val_list.txt')

        with open(file_path, 'r') as f:
            for line in f:
                items = line.strip().split()
                if len(items) != 2:
                    raise Exception("File list format incorrect! It should be"
                                    " image_name label_name\\n")
                else:
                    image_path = os.path.join(self.dataset_root, items[0])
                    grt_path = os.path.join(self.dataset_root, items[1])
                self.file_list.append([image_path, grt_path])


if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    dataset = LungCoronavirus(
        dataset_root="data/lung_coronavirus/lung_coronavirus_phase0",
        result_dir="data/lung_coronavirus/lung_coronavirus_phase1",
        transforms=[],
        mode="train")

    for img, label in dataset:
        print(img.shape, label.shape)
        print("image val", img.min(), img.max())
        print("label val", label.min(), label.max(), np.unique(label))
