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
import time

import numpy as np
import nibabel as nib
import nrrd
import SimpleITK as sitk

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

from paddleseg3d.utils import get_image_list
from paddleseg3d.datasets.preprocess_utils import uncompressor

# DEBUG:
import matplotlib.pyplot as plt
import cv2

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
        os.makedirs(self.visualize_path, exist_ok=True)

    def uncompress_file(self, num_zipfiles):
        uncompress_tool = uncompressor(download_params=(self.urls,
                                                        self.dataset_root,
                                                        True))
        """unzip all the file in the root directory"""
        zipfiles = []
        for ext in ["zip", "tar", "tar.gz", "tgz"]:
            zipfiles += glob.glob(os.path.join(self.dataset_root, f"*.{ext}"))

        assert len(zipfiles) == num_zipfiles, print(
            "The file directory should include {} zip file, but there is only {}"
            .format(num_zipfiles, len(zipfiles)))

        for f in zipfiles:
            extract_path = os.path.join(self.raw_data_path,
                                        f.split("/")[-1].split('.')[0])
            uncompress_tool._uncompress_file(f,
                                             extract_path,
                                             delete_file=False,
                                             print_progress=True)

    @staticmethod
    def load_medical_data(f):
        """ Load data of various format into numpy array, zyx format
        Args:
            f (str): the absolute path to the file that you want to load

        Returns:
            [np.array], dict: a list of 3d volumes in image, metadata in image header
        """
        filename = osp.basename(f)
        images = []

        if "nrrd" in filename:
            f_np, metadata = nrrd.read(f)
            f_nps = [f_np]
        else:
            itkimage = sitk.ReadImage(f)
            metadata = {}
            for key in itkimage.GetMetaDataKeys():
                metadata[key] = itkimage.GetMetaData(key)
            if itkimage.GetDimension() == 4:
                extract = sitk.ExtractImageFilter()
                s = list(itkimage.GetSize())
                s[-1]=0
                extract.SetSize(s)
                for slice_idx in range(itkimage.GetSize()[-1]):
                    extract.SetIndex([0,0,0, slice_idx])
                    sitk_volume = extract.Execute(itkimage)
                    images.append(sitk_volume)
            else:
                images = [itkimage]

            images = [sitk.DICOMOrient(img, 'LPS') for img in images]
            f_nps = [sitk.GetArrayFromImage(img) for img in images]
            f_nps = [np.transpose(f_np, [2, 1, 0]) for f_np in f_nps]

        # DEBUG: to be removed
        for f_np in f_nps:
            vis = f_np[f_np.shape[0]//2,:,:]
            cv2.imwrite(osp.join("./data/vis", f"{filename}-{f_np.shape[0]//2}.png" ), vis/vis.max()*255)
        print(metadata)
        return f_nps, metadata


    @staticmethod
    def load_save(image_folder,
                  label_folder,
                  save_path,
                  preprocesses=[],
                  intermediate_save_paths=[],
                  image_save_folder="image",
                  label_save_folder="label",
                  image_suffix=None,
                  label_suffix=None,
                  filter_key={}):
                  # TODO: maybe support filtering for scans satisfying certain pixel spacing and have more than certain number of slices. Can be useful on datasets obtained from hospital imaging department
        """Load and filter image and label, preprocess them and save on disk.
        Args:
            image_folder (str): Folder containing images
            label_folder (str): Folder containing labels
            save_path (str): Save base path
            intermediate_save_paths (str): Specify save paths for intermediate prep results
            preprocesses (list): List of preprocessing functions
            image_save_folder (str): Image will be saved to save_path/image_save_folder
            label_save_folder (str): Label will be saved to save_path/label_save_folder
            image_suffix ([str]): Only include images with these suffix
            label_suffix ([str]): Only include labels with these suffix
            filter_key ({"": bool}): Strings image and label name should or shouldn't contain
        """

        # 0. check image and label input path exists
        if not osp.exists(image_folder):
            raise RuntimeError(f"image_folder `{image_folder}` doesn't exist")
        if not osp.exists(label_folder):
            raise RuntimeError(f"label_folder `{label_folder}` doesn't exist")

        # 1. get all the image and label paths under specified path
        image_names = get_image_list(image_folder, image_suffix, filter_key)
        label_names = get_image_list(label_folder, label_suffix, filter_key)

        assert len(image_names) == len(label_names), f"The number of images {len(image_names)} and labels {len(label_names)} are not equal!"

        image_names.sort()
        label_names.sort()

        print("Followings are all included image, labels and their matching. ")
        for img, lab in zip(image_names, label_names):
            print(f"{img}\t{lab}")

        image_paths = [osp.join(image_folder, n) for n in image_names]
        label_paths = [osp.join(label_folder, n) for n in label_names]

        # 2. read with load_medical_data
        images = [Prep.load_medical_data(path) for path in image_paths]
        labels = [Prep.load_medical_data(path) for path in label_paths]

        # 3. preprocess data and save
        image_save_folder = osp.join(save_path, image_save_folder)
        label_save_folder = osp.join(save_path, label_save_folder)
        os.makedirs(image_save_folder, exist_ok=True)
        os.makedirs(label_save_folder, exist_ok=True)

        # TODO: muli-thread
        for pair_idx in range(len(images)):
            img_vols, _ = images[pair_idx]
            lab_vol, _ = labels[pair_idx]
            lab_vol = lab_vol[0] # 4D series typically have 3D label
            img_name = osp.basename(image_paths[pair_idx])
            lab_name = osp.basename(label_paths[pair_idx])

            for vol_idx, img_vol in enumerate(img_vols):
                for op in preprocesses:
                    img_vol, lab_vol = op(img_vol, lab_vol)  # TODO: all ops need to change

                img_vol = img_vol.astype("float32")
                lab_vol = lab_vol.astype("int64")

                vol_idx = str(-vol_idx) if len(img_vols) != 1 else ""
                np.save(osp.join(image_save_folder, img_name + vol_idx), img_vol)
                np.save(osp.join(label_save_folder, lab_name + vol_idx), lab_vol)

    print("Sucessfully convert medical images to numpy array!")

    def convert_path(self):
        """convert nii.gz file to numpy array in the right directory"""
        raise NotImplementedError

    def generate_txt(self):
        """generate the train_list.txt and val_list.txt"""
        raise NotImplementedError

    # TODO add data visualize method, such that data can be checked every time after preprocess.
    def visualize(self):
        # sitk.LabelMapContourOverlay
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
        with open(txt, 'w') as f:
            for i in range(len(image_names)):
                if label_names is not None:
                    string = "{} {}\n".format('images/' + image_names[i],
                                              'labels/' + label_names[i])
                else:
                    string = "{}\n".format('images/' + image_names[i])

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
