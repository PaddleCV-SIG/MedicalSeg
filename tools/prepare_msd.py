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
The file structure is as following:
MSD_test
|--MSD_raw
│   ├── Taskxx_xx
│   │   ├── imagesTr
│   │   │   ├── xxx_001.nii.gz
│   │   │   ├── ...
│   │   ├── imagesTs
│   │   │   ├── xxx_002.nii.gz
│   │   │   ├── ...
│   │   ├── labelsTr
│   │   │   ├── xxx_001.nii.gz
│   │   │   ├── ...
|--MSD_decathlon
│   ├── Taskxx_xx
│   │   ├── imagesTr
│   │   │   ├── xxx_001.nii.gz
│   │   │   ├── ...
│   │   ├── imagesTs
│   │   │   ├── xxx_002.nii.gz
│   │   │   ├── ...
│   │   ├── labelsTr
│   │   │   ├── xxx_001.nii.gz
│   │   │   ├── ...


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
import argparse
import functools
import numpy as np
import nibabel as nib

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             ".."))

from prepare import Prep
from paddleseg3d.datasets.preprocess_utils import uncompressor, crawl_and_remove_hidden_from_decathlon, split_4d
from paddleseg3d.datasets.preprocess_utils import HUNorm, resample, label_remap

urls = {
    "Task04_hippocampus.tar":
    "https://bj.bcebos.com/v1/ai-studio-online/7c29c43dc678458e8126f64e519ac8fd56e777858390417e9bc6892547fc0ace?responseContentDisposition=attachment%3B%20filename%3DTask04_Hippocampus.tar&、authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-01-29T05%3A10%3A31Z%2F-1%2F%2F6c4845ceef1f990cef7f5ada245759eb5e619d54fef50ef7f6313bd5e5036161",
    "Task06_lung.tar":
    "https://bj.bcebos.com/v1/ai-studio-online/9591d9811aea45aa96a2b85bd737a727ea52cc8efa7442b79134062b2302fbc0?responseContentDisposition=attachment%3B%20filename%3DTask06_Lung.tar&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-01-18T08%3A34%3A58Z%2F-1%2F%2Fb9816d90a62b661d388e84717c13b0afb6fd714c062ba9ddd4a10122e22a9267",
}  # TODO: Add urls and test uncompress file as aforementioned format


class Prep_nnunet_msd(Prep):
    """
    Args:
        msd_dataset_name (str): MSD dataset name such as "Task04_hippocampus".
    """
    def __init__(self, msd_dataset_name):
        self.task_name = msd_dataset_name
        self.dataset_root = os.path.join('data', 'MSD_test')
        self.phase_path = os.path.join(self.dataset_root,
                                       msd_dataset_name + '_phase0/')
        self.raw_data_path = os.path.join(self.dataset_root, 'MSD_raw')
        self.decathlon_data_path = os.path.join(self.dataset_root,
                                                'MSD_decathlon')
        self.cropped_data_path = os.path.join(self.dataset_root,
                                              'MSD_cropped/')
        self.urls = {msd_dataset_name: urls[msd_dataset_name + '.tar']}

    def uncompress_file(self, num_tarfiles):
        uncompress_tool = uncompressor(download_params=(self.urls,
                                                        self.dataset_root,
                                                        True))
        """unzip all the file in the root directory"""
        tarfiles = glob.glob(os.path.join(self.dataset_root, "*.tar"))

        assert len(tarfiles) == num_tarfiles, print(
            "The file directory should include {} tar file, but there is only {}"
            .format(num_tarfiles, len(tarfiles)))

        for f in tarfiles:
            extract_path = self.raw_data_path
            uncompress_tool._uncompress_file(f,
                                             extract_path,
                                             delete_file=False,
                                             print_progress=True)

    def convert_to_decathlon(self):
        data_name = self.task_name
        data_name = data_name[0:7] + data_name[7].capitalize() + data_name[8:]
        self.raw_data_path = os.path.join(self.raw_data_path, data_name)
        crawl_and_remove_hidden_from_decathlon(self.raw_data_path)
        split_4d(self.raw_data_path,
                 num_processes=8,
                 output_dir=self.decathlon_data_path)


def parse_args():
    parser = argparse.ArgumentParser(description='MSD dataset prepare.')
    parser.add_argument("--task_name",
                        dest="task_name",
                        help="MSD dataset name.",
                        default=None,
                        type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args.task_name)
    prep = Prep_nnunet_msd(args.task_name)
    prep.uncompress_file(1)
    prep.convert_to_decathlon()
