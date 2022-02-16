# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
"""
The file structure is as following:
prostate
|--prostate_raw
│   ├── TrainData
│   │   ├── Case49.raw
│   │   ├── Case49.mhd
│   │   ├── Case49_segmentation.raw
│   │   ├── Case49_segmentation.mhd
│   │   ├── ...
│   ├── TestData
│   │   ├── Case29.mhd
│   │   ├── Case29.raw
│   │   ├── ...
├── prostate_phase0
│   ├── images
│   │   ├── Case49.npy
│   │   ├── ...
│   ├── labels
│   │   ├── Case49_segmentation.npy
│   │   ├── ...
│   ├── test_images
│   │   ├── Case49.npy
│   │   ├── ...
│   ├── train_list.txt
│   ├── val_list.txt
│   └── test_list.txt

support:
1. download and uncompress the file.
2. save the data as the above format.
3. read the preprocessed data into train_list.txt, val_list.txt, test_list.txt

"""
import os
import sys
import zipfile
import functools
import numpy as np
import nibabel as nib

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             ".."))

from prepare import Prep
from paddleseg3d.datasets.preprocess_utils import HUNorm, resample

urls = {"prostate.zip": ""}

### TODO Prostate is MRI image, it needs other preprocess technique.


class Prep_luna(Prep):

    def __init__(self):
        self.dataset_root = "data/prostate"
        self.phase_path = os.path.join(self.dataset_root, "prostate_phase0/")
        super().__init__(phase_path=self.phase_path,
                         dataset_root=self.dataset_root)

        self.raw_data_path = os.path.join(self.dataset_root, "prostate_raw/")

        self.image_dir = os.path.join(self.raw_data_path, "TrainData")
        self.label_dir = os.path.join(self.raw_data_path, "TrainData")
        self.test_image_dir = os.path.join(self.raw_data_path, "TestData")
        self.urls = urls

    def convert_path(self):
        """convert nii.gz file to numpy array in the right directory"""

        print("Start convert images to numpy array, please wait patiently")
        self.load_save(
            self.image_dir,
            save_path=self.image_path,
            preprocess=[
                functools.partial(
                    HUNorm, HU_min=-1250,
                    HU_max=250),  # TODO replace HTNorm with other module
                functools.partial(resample, new_shape=[128, 128, 64], order=1)
            ],
            valid_suffix='mhd',
            filter_key={"segmentation": False})

        print("start convert labels to numpy array, please wait patiently")

        self.load_save(self.label_dir,
                       self.label_path,
                       preprocess=[
                           functools.partial(resample,
                                             new_shape=[128, 128, 64],
                                             order=0)
                       ],
                       valid_suffix='mhd',
                       filter_key={"segmentation": True},
                       tag="label")

    def generate_txt(self):
        """generate the train_list.txt and val_list.txt"""

        txtname = [
            os.path.join(self.phase_path, 'train_list.txt'),
            os.path.join(self.phase_path, 'val_list.txt'),
            os.path.join(self.phase_path, 'test_list.txt')
        ]

        label_files = os.listdir(self.label_path)
        image_files = [
            name.replace("_LobeSegmentation", "") for name in label_files
        ]

        self.split_files_txt(txtname[0],
                             image_files,
                             label_files,
                             train_split=45)
        self.split_files_txt(txtname[1],
                             image_files,
                             label_files,
                             train_split=45)


if __name__ == "__main__":
    prep = Prep_luna()
    # prep.uncompress_file(num_zipfiles=4)
    prep.convert_path()
    prep.generate_txt()
