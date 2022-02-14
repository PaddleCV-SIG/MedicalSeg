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
Details about MSD datasets: http://medicaldecathlon.com/

The file structure is as following:
lung_coronavirus
|--20_ncov_scan.zip
|--infection.zip
|--lung_infection.zip
|--lung_mask.zip
|--lung_coronavirus_raw
│   ├── 20_ncov_scan
│   │   ├── coronacases_org_001.nii.gz
│   │   ├── ...
│   ├── infection_mask
│   ├── lung_infection
│   ├── lung_mask
├── lung_coronavirus_phase0
│   ├── images
│   ├── labels
│   │   ├── coronacases_001.npy
│   │   ├── ...
│   │   └── radiopaedia_7_85703_0.npy
│   ├── train_list.txt
│   └── val_list.txt
support:
1. download and uncompress the file.
2. save the data as the above format.

"""
import os
import os.path as osp
import sys
import functools

import numpy as np

sys.path.append(osp.join(osp.dirname(osp.realpath(__file__)), ".."))

from prepare import Prep
from paddleseg3d.datasets.preprocess_utils import HUNorm, resample

tasks = {
    1: {
        "Task01_BrainTumour.tar":
        "https://bj.bcebos.com/v1/ai-studio-online/netdisk/975fea1d4c8549b883b2b4bb7e6a82de84392a6edd054948b46ced0f117fd701?responseContentDisposition=attachment%3B%20filename%3DTask01_BrainTumour.tar&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-01-21T18%3A50%3A30Z%2F-1%2F%2F283ea6f8700c129903e3278ea38a54eac2cf087e7f65197268739371898aa1b3",
    },
    2: "Task02_Heart",
    3: "Task03_Liver",
    4: {
        "Task04_Hippocampus.tar":
        "https://bj.bcebos.com/v1/ai-studio-online/1bf93142b1284f69a2a2a4e84248a0fe2bdb76c3b4ba4ddf82754e23d8820dfe?responseContentDisposition=attachment%3B%20filename%3DTask04_Hippocampus.tar&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-02-14T17%3A09%3A53Z%2F-1%2F%2Fc53aa0df7f8810277261a00458d0af93df886c354c27498607bb8e2fb64a3d90"
    },
    5: "Task05_Prostate",
    6: "Task06_Lung",
    7: "Task07_Pancreas",
    8: "Task08_HepaticVessel",
    9: "Task09_Spleen",
    10: "Task10_Colon",
}


class Prep_msd(Prep):

    def __init__(self, task_id):
        self.dataset_name = list(tasks[task_id].keys())[0].split(".")[0]
        super().__init__(
            dataset_fdr=f"msd/{self.dataset_name}",
            urls=tasks[task_id],
            image_fdr=f"{self.dataset_name}/{self.dataset_name}/imagesTr",
            label_fdr=f"{self.dataset_name}/{self.dataset_name}/labelsTr",
            phase_fdr="phase0",
        )

    def convert_path(self):
        """convert medical imaging file to numpy array in the right directory"""

        print("Start convert images to numpy array, please wait patiently")
        self.load_save(
            self.image_dir,
            savepath=self.image_path,
            preprocess=[
                HUNorm,
                functools.partial(resample, new_shape=[128, 128, 128],
                                  order=1),
            ],
            filter={
                "filter_suffix": None,
                "filter_key": None
            },
        )
        print("start convert labels to numpy array, please wait patiently")

        self.load_save(
            self.label_dir,
            self.label_path,
            preprocess=[
                functools.partial(resample, new_shape=[128, 128, 128],
                                  order=0),
            ],
            filter={
                "filter_suffix": None,
                "filter_key": None
            },
            tag="label",
        )

    def generate_txt(self, train_split=15):
        """generate the train_list.txt and val_list.txt"""

        txtname = [
            osp.join(self.phase_path, "train_list.txt"),
            osp.join(self.phase_path, "val_list.txt"),
        ]

        image_files = os.listdir(self.image_path)
        label_files = [
            name.replace("_org_covid-19-pneumonia-",
                         "_").replace("-dcm", "").replace("_org_", "_")
            for name in image_files
        ]

        self.split_files_txt(txtname[0], image_files, label_files)
        self.split_files_txt(txtname[1], image_files, label_files)


if __name__ == "__main__":
    prep = Prep_msd(task_id=4)
    prep.uncompress_file(num_zipfiles=1)
    prep.convert_path()
    prep.generate_txt()
