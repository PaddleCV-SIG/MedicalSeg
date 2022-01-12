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
"""
File: prepare.py
This is the prepare class for all relavent prepare file

support:
1. download and uncompress the file.
2. save the data as the above format.

"""
import os
import sys
import glob
import time
import random
import zipfile
import functools
import numpy as np
import nibabel as nib
import nrrd
import SimpleITK as sitk

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

from prepare_utils import list_files
from paddleseg3d.datasets.preprocess_utils import uncompressor
from paddleseg3d.datasets.preprocess_utils import HU2float32, resample


class Prep:
    def __init__(self, phase_path=None, dataset_root=None):
        self.raw_data_path = None
        self.image_dir = None
        self.label_dir = None
        self.urls = None

        self.dataset_root = dataset_root
        self.phase_path = phase_path
        self.image_path = os.path.join(self.phase_path, "images")
        self.label_path = os.path.join(self.phase_path, "labels")
        os.makedirs(self.image_path, exist_ok=True)
        os.makedirs(self.label_path, exist_ok=True)

    def uncompress_file(self, num_zipfiles):
        uncompress_tool = uncompressor(
            download_params=(self.urls, self.dataset_root, True))
        """unzip all the file in the root directory"""
        zipfiles = glob.glob(os.path.join(self.dataset_root, "*.zip"))

        assert len(zipfiles) == num_zipfiles, print(
            "The file directory should include {} zip file, but there is only {}"
            .format(num_zipfiles, len(zipfiles)))

        for f in zipfiles:
            extract_path = os.path.join(self.raw_data_path,
                                        f.split("/")[-1].split('.')[0])
            uncompress_tool._uncompress_file(
                f, extract_path, delete_file=False, print_progress=True)

    def load_save(self,
                  file_dir,
                  load_type=np.float32,
                  savepath=None,
                  preprocess=None,
                  filter_suffix=None,
                  tag="image"):
        """
        Load the file in file dir, preprocess it and save it to the directory.
        """
        files = list_files(file_dir, filter_suffix=filter_suffix)
        assert len(files) != 0, print(
            "The data directory you assigned is wrong, there is no file in it."
        )

        for f in files:
            filename = f.split("/")[-1]
            if "nii.gz" in filename:
                f_np = nib.load(f).get_fdata(dtype=load_type)
                file_suffix = "nii.gz"
            elif "nrrd" in filename:
                f_np, header = nrrd.read(f)
                file_suffix = "nrrd"
            elif "mhd" in filename or "raw" in filename:
                itkimage = sitk.ReadImage(f)
                # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
                f_np = sitk.GetArrayFromImage(itkimage)
                file_suffix = "mhd"
            else:
                raise NotImplementedError
            file_prefix = filename[:-len(file_suffix) - 1]

            if preprocess is not None:
                for op in preprocess:
                    f_np = op(f_np)

            np.save(os.path.join(savepath, file_prefix), f_np)

        print("Sucessfully convert medical images to numpy array!")

    def convert_path(self):
        """convert nii.gz file to numpy array in the right directory"""
        raise NotImplementedError

    def generate_txt(self):
        """generate the train_list.txt and val_list.txt"""
        raise NotImplementedError

    def write_txt(self, txt, image_files, label_files, train_split=None):
        if train_split is None:
            train_split = int(0.8 * len(image_files))

        with open(txt, 'w') as f:
            if "train" in txt:
                image_names = image_files[:train_split]
                label_names = label_files[:train_split]

            else:
                image_names = image_files[train_split:]
                label_names = label_files[train_split:]

            for i in range(len(image_names)):
                string = "{} {}\n".format('images/' + image_names[i],
                                          'labels/' + label_names[i])
                f.write(string)

        print("successfully write to {}".format(txt))
