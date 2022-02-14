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
3. read the preprocessed data into train.txt and val.txt

"""
import os
import os.path as osp
import sys
import glob
import zipfile

import numpy as np
import nibabel as nib
import nrrd
import SimpleITK as sitk

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             ".."))

from paddleseg3d.datasets.preprocess_utils import uncompressor, sitk_read, list_files

# DEBUG:
import matplotlib.pyplot as plt


class Prep:

    def __init__(
        self,
        dataset_fdr,
        urls,
        image_fdr,
        label_fdr,
        phase_fdr="phase0",
        raw_fdr="raw",
        datasets_root="data",
    ):
        # self.raw_data_path = None
        # self.image_dir = None
        # self.label_dir = None
        # self.urls = None

        self.dataset_root = osp.join(datasets_root, dataset_fdr)
        self.phase_path = osp.join(self.dataset_root, phase_fdr)
        self.raw_data_path = osp.join(self.dataset_root, raw_fdr)
        self.image_dir = osp.join(self.raw_data_path, image_fdr)
        self.label_dir = osp.join(self.raw_data_path, label_fdr)
        self.urls = urls

        self.image_path = os.path.join(self.phase_path, "images")
        self.label_path = os.path.join(self.phase_path, "labels")

        os.makedirs(self.image_path, exist_ok=True)
        os.makedirs(self.label_path, exist_ok=True)

    def uncompress_file(self, num_zipfiles):
        uncompress_tool = uncompressor(download_params=(self.urls,
                                                        self.dataset_root,
                                                        True))
        """unzip all the file in the dataset_root directory"""
        zipfiles = []
        for ext in ["zip", "tar", "tar.gz", "tgz"]:
            zipfiles += glob.glob(os.path.join(self.dataset_root, f"*.{ext}"))

        assert len(zipfiles) == num_zipfiles, print(
            "The file directory should include {} zip file, but there is only {}"
            .format(num_zipfiles, len(zipfiles)))

        for f in zipfiles:
            extract_path = os.path.join(self.raw_data_path,
                                        f.split("/")[-1].split(".")[0])
            uncompress_tool._uncompress_file(f,
                                             extract_path,
                                             delete_file=False,
                                             print_progress=True)

    def load_save(self,
                  file_dir,
                  savepath=None,
                  preprocess=None,
                  filter=None,
                  tag="image"):
        # TODO: add support for multiprocess
        """
        Load the needed file with filter, preprocess it transfer to the correct type and save it to the directory.
        """
        files = list_files(file_dir, **filter)

        assert len(files) != 0, print(
            "The data directory you assigned is wrong, there is no file in it."
        )

        for f in files:
            filename = f.split("/")[-1]
            # load data based on format, axis should be in order z, y, x

            if "nii.gz" in filename:
                try:
                    f_nps = sitk_read(f, split=True)
                except RuntimeError:
                    # TODO: nib read 4d series
                    f_np = nib.load(f).get_fdata(dtype="float32")
                    f_nps = [f_np.swapaxes(0, 2)]
                file_suffix = "nii.gz"
            elif "nrrd" in filename:
                f_np, header = nrrd.read(f)
                file_suffix = "nrrd"
                f_nps = [f_np]
            elif "mhd" in filename or "raw" in filename:
                # TODO： check if all data read with sitk need shuffle axes, this is needed in luna16_lobe51
                f_nps = sitk_read(f, split=True)
                file_suffix = "mhd"
            else:
                raise NotImplementedError
            file_prefix = filename[:-len(file_suffix) - 1]

            # plt.imshow(f_nps[0][10])
            # plt.show()

            # Set image to a uniform format before save.
            if tag == "image":
                f_nps = [f_np.astype("float32") for f_np in f_nps]
            else:
                f_nps = [f_np.astype("int64") for f_np in f_nps]

            for idx, f_np in enumerate(f_nps):
                if preprocess is not None:
                    for op in preprocess:
                        f_np = op(f_np)
                part_id = f"-s{idx}" if len(f_nps) != 1 else ""
                np.save(os.path.join(savepath, f"{file_prefix}{part_id}"),
                        f_np)

        print("Sucessfully converted medical images to numpy array!")

    def convert_path(self):
        """convert nii.gz file to numpy array in the right directory"""
        raise NotImplementedError

    def generate_txt(self):
        """generate the train_list.txt and val_list.txt"""
        raise NotImplementedError

    # TODO add data visualize method, such that data can be checked every time after preprocess.
    def visualize(self):
        pass
        # imga = Image.fromarray(np.int8(imga))
        # #当要保存的图片为灰度图像时，灰度图像的 numpy 尺度是 [1, h, w]。需要将 [1, h, w] 改变为 [h, w]
        # imgb = np.squeeze(imgb)

        # # imgb = Image.fromarray(np.int8(imgb))
        # plt.figure(figsize=(12, 6))
        # plt.subplot(1,2,1),plt.xticks([]),plt.yticks([]),plt.imshow(imga)
        # plt.subplot(1,2,2),plt.xticks([]),plt.yticks([]),plt.imshow(imgb)
        # plt.show()

    def write_txt(self, txt, image_names, label_names=None):
        """
        write the image_names and label_names on the txt file like this:

        images/image_name labels/label_name
        ...

        or this when label is None.

        images/image_name
        ...

        """
        with open(txt, "w") as f:
            for i in range(len(image_names)):
                if label_names is not None:
                    string = "{} {}\n".format("images/" + image_names[i],
                                              "labels/" + label_names[i])
                else:
                    string = "{}\n".format("images/" + image_names[i])

                f.write(string)

        print("successfully write to {}".format(txt))

    def split_files_txt(self,
                        txt,
                        image_files,
                        label_files=None,
                        train_split=None):
        """
        split filenames and write the image names and label names on train.txt, val.txt or test.txt

        Args:
        txt(string): the path to the txt file, for example: "data/train.txt"
        image_files(list|tuple): the list of image names.
        label_files(list|tuple): the list of label names, order is corresponding with the image_files.
        train_split(float): Number of the training files

        """
        if train_split is None:
            train_split = int(0.8 * len(image_files))

        if "train" in txt:
            image_names = image_files[:train_split]
            label_names = label_files[:train_split]
        elif "val" in txt:
            image_names = image_files[train_split:]
            label_names = label_files[train_split:]
        elif "test" in txt:
            self.write_txt(txt, image_names)

            return
        else:
            raise NotImplementedError(
                "The txt split except for train.txt, val.txt and test.txt is not implemented yet."
            )

        self.write_txt(txt, image_names, label_names)
