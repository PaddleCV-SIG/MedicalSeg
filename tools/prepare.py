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
import random

import numpy as np
import nibabel as nib
import nrrd
import SimpleITK as sitk
import cv2

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             ".."))

from paddleseg3d.datasets.preprocess_utils import uncompressor, sitk_read, list_files

# DEBUG:
import matplotlib.pyplot as plt
"""
axis: z y x
matplolib: h w c
sitk: c z y x
"""


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
        visualize_fdr="vis",
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
        self.visualize_path = osp.join(self.dataset_root, visualize_fdr)

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
            # load data based on ext, result axis should be in order z, y, x
            if "nii.gz" in filename:
                try:
                    f_nps = sitk_read(f, split=True)
                except RuntimeError:
                    # TODO: nib read 4d series
                    # TODO: support orient
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
                part_id = "-t" + str(idx).zfill(3) if len(f_nps) != 1 else ""
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
    def visualize(self, idx=None, alpha=0.8, mode="show"):
        if mode == "save":
            os.makedirs(self.visualize_path, exist_ok=True)
        # 1. read training list
        # TODO: add other two lists
        with open(osp.join(self.phase_path, "train_list.txt"), "r") as f:
            train_list = f.readlines()
        train_list = [l.strip().split(" ") for l in train_list]
        # 2. generate id of record to visualize
        if idx is None:
            idxs = range(len(train_list))
        elif idx == -1:
            idxs = [int(random.random() * len(train_list))]
        else:
            idxs = [idx]

        for idx in idxs:
            image_path, label_path = train_list[idx]
            image = np.load(osp.join(self.phase_path, image_path)) * 255
            image = image.astype("uint8")
            label = np.load(osp.join(self.phase_path, label_path))
            label = label.astype("uint8")
            assert (
                image.shape == label.shape
            ), f"image shape: {image.shape} != label shape: {label.shape}"
            label_max = label.max()
            color_step = int(255 / (label_max + 1))
            for z_idx in range(image.shape[0]):
                # 3. visualize image and label
                # 3.1 get image and label slice, prep
                image_slice = image[z_idx, :, :]
                image_slice = cv2.cvtColor(image_slice, cv2.COLOR_GRAY2RGB)
                label_slice = label[z_idx, :, :]
                # 3.2 show two images directly
                if mode == "show":
                    if label_slice.sum() != 0:
                        plt.subplot(1, 2, 1)
                        plt.imshow(image_slice)
                        plt.subplot(1, 2, 2)
                        plt.imshow(label_slice)
                        manager = plt.get_current_fig_manager()
                        manager.full_screen_toggle()
                        plt.show()
                    continue
                # 3.3 generate color label mask
                curr_color = 0
                label_mask = np.zeros([*label_slice.shape, 3], dtype="uint8")
                for ann in range(1, label_max + 1):
                    curr_color += color_step
                    color = cv2.applyColorMap(
                        np.array([curr_color]).astype("uint8"),
                        cv2.COLORMAP_JET)
                    mask = label_slice == ann
                    for i in range(label_slice.shape[0]):
                        for j in range(label_slice.shape[1]):
                            if mask[i][j]:
                                label_mask[i, j, :] = color[0][0]
                # 3.4 blend image and label mask
                for i in range(label_slice.shape[0]):
                    for j in range(label_slice.shape[1]):
                        if (label_mask[i, j] != [0, 0, 0]).any():
                            image_slice[i, j] = (label_mask[i, j] * alpha +
                                                 image_slice[i, j] *
                                                 (1 - alpha)).astype("uint8")
                # 3.5 save blended image
                vis_name = f"{osp.basename(image_path.split('.')[0])}-z{str(z_idx)}.png"
                image_slice = cv2.cvtColor(image_slice, cv2.COLOR_RGB2BGR)
                cv2.imwrite(osp.join(self.visualize_path, vis_name),
                            image_slice)

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
