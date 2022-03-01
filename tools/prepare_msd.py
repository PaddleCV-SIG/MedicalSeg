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
        "https://bj.bcebos.com/v1/ai-studio-online/netdisk/975fea1d4c8549b883b2b4bb7e6a82de84392a6edd054948b46ced0f117fd701?responseContentDisposition=attachment%3B%20filename%3DTask01_BrainTumour.tar&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-01-21T18%3A50%3A30Z%2F-1%2F%2F283ea6f8700c129903e3278ea38a54eac2cf087e7f65197268739371898aa1b3"
    },  # 4d
    2: {
        "Task02_Heart.tar":
        "https://bj.bcebos.com/v1/ai-studio-online/netdisk/44a1e00baf55489db5d95d79f2e56e7230b6f87687604ab0889e0deb45ba289e?responseContentDisposition=attachment%3B%20filename%3DTask02_Heart.tar&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-01-21T18%3A30%3A22Z%2F-1%2F%2F3c23a084e9bbbc57d8d6435eb014b7fb8c4160395a425bc94da5b55a08fc14de"
    },  # 3d zxy
    3: {
        "Task03_Liver.tar":
        "https://bj.bcebos.com/v1/ai-studio-online/netdisk/e641b1b7f364472c885147b6c500842f559ee6ae03494b78b5d140d53db35907?responseContentDisposition=attachment%3B%20filename%3DTask03_Liver.tar&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-01-21T18%3A49%3A33Z%2F-1%2F%2F83b1b4e70026a2a568dcfbbf60fb06f0ae27a847e7ebe5ba7b2efe60fc6b16a5"
    },  # 3d
    4: {
        "Task04_Hippocampus.tar":
        "https://bj.bcebos.com/v1/ai-studio-online/1bf93142b1284f69a2a2a4e84248a0fe2bdb76c3b4ba4ddf82754e23d8820dfe?responseContentDisposition=attachment%3B%20filename%3DTask04_Hippocampus.tar&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-02-14T17%3A09%3A53Z%2F-1%2F%2Fc53aa0df7f8810277261a00458d0af93df886c354c27498607bb8e2fb64a3d90"
    },  # 3d zyx
    5: {
        "Task05_Prostate.tar":
        "https://bj.bcebos.com/v1/ai-studio-online/netdisk/aca74eceef674a74bff647998413ebf25a33ad44e04643d7b796e05eecbc9891?responseContentDisposition=attachment%3B%20filename%3DTask05_Prostate.tar&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-01-21T18%3A28%3A58Z%2F-1%2F%2F610d78c178a2f5eeb5d8f6c7ec48ef52f7d6899b5ed8484f213ff1e03d266bd8"
    },  # 4d
    6: {
        "Task06_Lung.tar":
        "https://bj.bcebos.com/v1/ai-studio-online/netdisk/c42c621dc5c0490baaec935e1efd899478615f02add040649764c80c5f46805a?responseContentDisposition=attachment%3B%20filename%3DTask06_Lung.tar&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-01-21T18%3A59%3A27Z%2F-1%2F%2Fd4a6b5b382136af96395a8acc6d18d4e88ac744314c517f19f3a71417be3d12c"
    },  # 3d zxy
    7: {
        "Task07_Pancreas.tar":
        "https://bj.bcebos.com/v1/ai-studio-online/netdisk/d94f22313d764d808b15b240da0335a9cf0ca0e806ce418f9213f9db9e56a5a8?responseContentDisposition=attachment%3B%20filename%3DTask07_Pancreas.tar&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-01-21T18%3A34%3A45Z%2F-1%2F%2F3a17fb265c8fcdac91de8f15e7e2352a31783bbb121755ad27c28685ce047afa"
    },  # 3d
    8: {
        "Task08_HepaticVessel.tar":
        "https://bj.bcebos.com/v1/ai-studio-online/netdisk/51ff9421bfa648449f12e65a68862215c6b5b85f91de49aab1c16626c62c3af6?responseContentDisposition=attachment%3B%20filename%3DTask08_HepaticVessel.tar&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-01-21T18%3A35%3A23Z%2F-1%2F%2Fa664645e0b0c99e351f31352701dbe163de3fbe6e96eac11539629b5e6658360"
    },  # 3d
    9: {
        "Task09_Spleen.tar":
        "https://bj.bcebos.com/v1/ai-studio-online/netdisk/c02462f396f14b13a50d2c9ff01f86fc471c7bff8df24994af7bd8b2298dc843?responseContentDisposition=attachment%3B%20filename%3DTask09_Spleen.tar&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-01-21T18%3A45%3A46Z%2F-1%2F%2Faf6f10f658fbe9569eb423fc1b7bd464aead582ef89cd7c135dcae002bc3cb09"
    },  # 3d zyx
    10: {
        "Task10_Colon.tar":
        "https://bj.bcebos.com/v1/ai-studio-online/netdisk/062aa5a52cc44597a87f56c5ef1371c7acb52f73a2c946be9fea347dedec5058?responseContentDisposition=attachment%3B%20filename%3DTask10_Colon.tar&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-01-21T18%3A42%3A03Z%2F-1%2F%2F106546582e748224f0833e100fc74d1bf3ff7fe4f4370d43bb487b10c3f5deae"
    },  # 3d zyx
}


class Prep_msd(Prep):

    def __init__(self, task_id):
        self.dataset_name = list(tasks[task_id].keys())[0].split(".")[0]
        super().__init__(
            dataset_fdr=f"{self.dataset_name}",
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
            save_path=self.image_path,
            preprocess=[
                HUNorm,
                functools.partial(resample, new_shape=[128, 128, 128],
                                  order=1),
            ],
        )
        print("start convert labels to numpy array, please wait patiently")

        self.load_save(
            self.label_dir,
            self.label_path,
            preprocess=[
                functools.partial(resample, new_shape=[128, 128, 128],
                                  order=0),
            ],
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


# 3d: 2 3 4 6 7 8 9 10

if __name__ == "__main__":
    assert (
        len(sys.argv) == 2
    ), "Please specify msd task id. \n usage: python tools/prepare_msd.py task_id"
    prep = Prep_msd(task_id=int(sys.argv[1]))
    prep.uncompress_file(num_zipfiles=1)
    prep.convert_path()
    prep.generate_txt()
    # prep.visualize(alpha=0.1, mode="save", idx=-1)
