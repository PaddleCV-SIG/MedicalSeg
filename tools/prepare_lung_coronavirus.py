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
import tarfile
import time
import random
import zipfile
import functools
import numpy as np

import nibabel as nib

lasttime = time.time()
FLUSH_INTERVAL = 0.1


def list_names(path):
    """list all the filename in a given path recursively"""
    fname = []
    for root, _, f_names in os.walk(path):
        for f in f_names:
            fname.append(os.path.join(root, f))

    return fname


class uncompressor:
    def __init__(self):
        pass

    def _uncompress_file_zip(self, filepath, extrapath):
        files = zipfile.ZipFile(filepath, 'r')
        filelist = files.namelist()
        rootpath = filelist[0]
        total_num = len(filelist)
        for index, file in enumerate(filelist):
            files.extract(file, extrapath)
            yield total_num, index, rootpath
        files.close()
        yield total_num, index, rootpath

    def progress(self, str, end=False):
        global lasttime
        if end:
            str += "\n"
            lasttime = 0
        if time.time() - lasttime >= FLUSH_INTERVAL:
            sys.stdout.write("\r%s" % str)
            lasttime = time.time()
            sys.stdout.flush()

    def _uncompress_file_tar(self, filepath, extrapath, mode="r:gz"):
        files = tarfile.open(filepath, mode)
        filelist = files.getnames()
        total_num = len(filelist)
        rootpath = filelist[0]
        for index, file in enumerate(filelist):
            files.extract(file, extrapath)
            yield total_num, index, rootpath
        files.close()
        yield total_num, index, rootpath

    def _uncompress_file(self, filepath, extrapath, delete_file,
                         print_progress):
        if print_progress:
            print("Uncompress %s" % os.path.basename(filepath))

        if filepath.endswith("zip"):
            handler = self._uncompress_file_zip
        elif filepath.endswith("tgz"):
            handler = functools.partial(self._uncompress_file_tar, mode="r:*")
        else:
            handler = functools.partial(self._uncompress_file_tar, mode="r")

        for total_num, index, rootpath in handler(filepath, extrapath):
            if print_progress:
                done = int(50 * float(index) / total_num)
                self.progress("[%-50s] %.2f%%" %
                              ('=' * done, float(100 * index) / total_num))
        if print_progress:
            self.progress("[%-50s] %.2f%%" % ('=' * 50, 100), end=True)

        if delete_file:
            os.remove(filepath)

        return rootpath


class Prep:
    dataset_root = "../data/lung_coronavirus"
    phase0_path = os.path.join(dataset_root, "lung_coronavirus_phase0/")
    image_dir = os.path.join(dataset_root, "20_ncov_scan")
    label_dir = os.path.join(dataset_root, "lung_mask")

    def __init__(self, phase_path=phase0_path, train_split=15):
        self.train_split = train_split
        self.phase_path = phase_path
        self.image_path = os.path.join(self.phase_path, "images")
        self.label_path = os.path.join(self.phase_path, "labels")
        os.makedirs(self.image_path, exist_ok=True)
        os.makedirs(self.label_path, exist_ok=True)

        self.uncompress_tool = uncompressor()

    def uncompress_file(self, num_zipfiles):
        """unzip all the file in the root directory"""
        zipfiles = glob.glob(os.path.join(self.dataset_root, "*.zip"))

        assert len(zipfiles) == num_zipfiles, print(
            "The file directory should include {} zip file, but there is only {}"
            .format(num_zipfiles, len(zipfiles)))

        for f in zipfiles:
            extract_path = os.path.join(self.dataset_root,
                                        f.split("/")[-1].split('.')[0])
            self.uncompress_tool._uncompress_file(
                f, extract_path, delete_file=False, print_progress=True)

    def convert_path(self, image_dir=image_dir, label_dir=label_dir):
        """convert nii.gz file to numpy array in the right directory"""

        def load_save(names, load_type=np.float32, savepath=self.image_path):
            for f in names:
                filename = f.split("/")[-1].split(".")[0]
                nii_np = nib.load(f).get_fdata(dtype=load_type)
                np.save(os.path.join(savepath, filename), nii_np)

        image_names = list_names(image_dir)
        label_names = list_names(label_dir)

        print("start to convert images to numpy array, please wait patiently")
        load_save(image_names, np.float32, self.image_path)
        print("start to convert labels to numpy array, please wait patiently")
        load_save(label_names, np.float32, self.label_path)
        print("Sucessfully convert medical images to numpy array!")

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
                ]  #todo: remove specific for this class

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
    # prep.uncompress_file(num_zipfiles=4)
    # prep.convert_path()
    prep.generate_txt()
