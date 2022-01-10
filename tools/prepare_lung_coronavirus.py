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
File: prepare_lung_coronavirus.py
we assume that you have download the dataset through the link and save them as following:
lung_coronavirus
|--20_ncov_scan.zip
|--infection.zip
|--lung_infection.zip
|--lung_mask.zip

support:
1. uncompress the file and save the img as the following format
2. save your img as the rules.

lung_coronavirus_phase0
|
|--images
|--labels

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

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

from utils import list_files
from paddleseg3d.datasets.preprocess_utils import uncompressor
from paddleseg3d.datasets.preprocess_utils import HU2float32, resample

urls = {
    "lung_infection.zip":
    "https://bj.bcebos.com/v1/ai-studio-online/432237969243497caa4d389c33797ddb2a9fa877f3104e4a9a63bd31a79e4fb8?responseContentDisposition=attachment%3B%20filename%3DLung_Infection.zip&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2020-05-10T03%3A42%3A16Z%2F-1%2F%2Faccd5511d56d7119555f0e345849cca81459d3783c547eaa59eb715df37f5d25",
    "lung_mask.zip":
    "https://bj.bcebos.com/v1/ai-studio-online/96f299c5beb046b4a973fafb3c39048be8d5f860bd0d47659b92116a3cd8a9bf?responseContentDisposition=attachment%3B%20filename%3DLung_Mask.zip&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2020-05-10T03%3A41%3A14Z%2F-1%2F%2Fb8e23810db1081fc287a1cae377c63cc79bac72ab0fb835d48a46b3a62b90f66",
    "infection_mask.zip":
    "https://bj.bcebos.com/v1/ai-studio-online/2b867932e42f4977b46bfbad4fba93aa158f16c79910400b975305c0bd50b638?responseContentDisposition=attachment%3B%20filename%3DInfection_Mask.zip&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2020-05-10T03%3A42%3A37Z%2F-1%2F%2Fabd47aa33ddb2d4a65555795adef14826aa68b20c3ee742dff2af010ae164252",
    "20_ncov_scan.zip":
    "https://bj.bcebos.com/v1/ai-studio-online/12b02c4d5f9d44c5af53d17bbd4f100888b5be1dbc3d40d6b444f383540bd36c?responseContentDisposition=attachment%3B%20filename%3D20_ncov_scan.zip&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2020-05-10T14%3A54%3A21Z%2F-1%2F%2F1d812ca210f849732feadff9910acc9dcf98ae296988546115fa7b987d856b85"
}


class Prep:
    dataset_root = "data/lung_coronavirus"
    phase0_path = os.path.join(dataset_root, "lung_coronavirus_phase0/")
    raw_data_path = os.path.join(dataset_root, "lung_coronavirus_raw/")
    image_dir = os.path.join(raw_data_path, "20_ncov_scan")
    label_dir = os.path.join(raw_data_path, "lung_mask")

    def __init__(self, phase_path=phase0_path, train_split=15):
        self.train_split = train_split
        self.phase_path = phase_path
        self.image_path = os.path.join(self.phase_path, "images")
        self.label_path = os.path.join(self.phase_path, "labels")
        os.makedirs(self.image_path, exist_ok=True)
        os.makedirs(self.label_path, exist_ok=True)

    def uncompress_file(self, num_zipfiles):
        uncompress_tool = uncompressor(
            urls=urls, savepath=self.dataset_root, print_progress=True)
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
                  tag="image"):
        """
        Load the file in file dir, preprocess it and save it to the directory.
        """
        files = list_files(file_dir)
        assert len(files) != 0, print(
            "The data directory you assigned is wrong, there is no file in it."
        )

        for f in files:
            filename = f.split("/")[-1].split(".")[0]
            nii_np = nib.load(f).get_fdata(dtype=load_type)

            if preprocess is not None:
                for op in preprocess:
                    nii_np = op(nii_np)

            np.save(os.path.join(savepath, filename), nii_np)

        print("Sucessfully convert medical images to numpy array!")

    def convert_path(self):
        """convert nii.gz file to numpy array in the right directory"""
        import pdb
        pdb.set_trace()

        print("Start convert images to numpy array, please wait patiently")
        self.load_save(
            self.image_dir,
            load_type=np.float32,
            savepath=self.image_path,
            preprocess=[
                HU2float32,
                functools.partial(resample, new_shape=[128, 128, 128])
            ])
        print("start convert labels to numpy array, please wait patiently")

        self.load_save(
            self.label_dir,
            np.float32,
            self.label_path,
            preprocess=[
                functools.partial(
                    resample, new_shape=[128, 128, 128], order=0)
            ],
            tag="label")

    def generate_txt(self):
        """generate the train_list.txt and val_list.txt"""

        def write_txt(txt, files):
            with open(txt, 'w') as f:
                if "train" in txt:
                    image_names = files[:self.train_split]
                else:
                    image_names = files[self.train_split:]

                label_names = [
                    name.replace("_org_covid-19-pneumonia-", "_").replace(
                        "-dcm", "").replace("_org_", "_")
                    for name in image_names
                ]  # todo: remove specific for this class

                for i in range(len(image_names)):
                    string = "{} {}\n".format('images/' + image_names[i],
                                              'labels/' + label_names[i])
                    f.write(string)
            print("successfully write to {}".format(txt))

        txtname = [
            os.path.join(self.phase_path, 'train_list.txt'),
            os.path.join(self.phase_path, 'val_list.txt')
        ]

        files = os.listdir(self.image_path)
        random.shuffle(files)
        write_txt(txtname[0], files)
        write_txt(txtname[1], files)


if __name__ == "__main__":
    prep = Prep()
    prep.uncompress_file(num_zipfiles=4)
    prep.convert_path()
    prep.generate_txt()
