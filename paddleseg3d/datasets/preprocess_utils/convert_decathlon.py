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
import os
import shutil
import numpy as np
import SimpleITK as sitk
from multiprocessing import Pool
from .file_utils import subdirs, subfiles, remove_trailing_slash


def crawl_and_remove_hidden_from_decathlon(folder):
    folder = remove_trailing_slash(folder)
    assert folder.split('/')[-1].startswith("Task"), "This does not seem to be a decathlon folder. Please give me a " \
                                                     "folder that starts with TaskXX and has the subdirs imagesTr, " \
                                                     "labelsTr and imagesTs"
    subf = subdirs(folder, join=False)
    assert 'imagesTr' in subf, "This does not seem to be a decathlon folder. Please give me a " \
                                                     "folder that starts with TaskXX and has the subdirs imagesTr, " \
                                                     "labelsTr and imagesTs"
    assert 'imagesTs' in subf, "This does not seem to be a decathlon folder. Please give me a " \
                                                     "folder that starts with TaskXX and has the subdirs imagesTr, " \
                                                     "labelsTr and imagesTs"
    assert 'labelsTr' in subf, "This does not seem to be a decathlon folder. Please give me a " \
                                                     "folder that starts with TaskXX and has the subdirs imagesTr, " \
                                                     "labelsTr and imagesTs"
    _ = [os.remove(i) for i in subfiles(folder, prefix=".")]
    _ = [
        os.remove(i)
        for i in subfiles(os.path.join(folder, 'imagesTr'), prefix=".")
    ]
    _ = [
        os.remove(i)
        for i in subfiles(os.path.join(folder, 'labelsTr'), prefix=".")
    ]
    _ = [
        os.remove(i)
        for i in subfiles(os.path.join(folder, 'imagesTs'), prefix=".")
    ]


def split_4d_nifti(filename, output_folder):
    img_itk = sitk.ReadImage(filename)
    dim = img_itk.GetDimension()
    file_base = filename.split("/")[-1]
    if dim == 3:
        shutil.copy(
            filename,
            os.path.join(output_folder, file_base[:-7] + "_0000.nii.gz"))
        return
    elif dim != 4:
        raise RuntimeError(
            "Unexpected dimensionality: %d of file %s, cannot split" %
            (dim, filename))
    else:
        img_npy = sitk.GetArrayFromImage(img_itk)
        spacing = img_itk.GetSpacing()
        origin = img_itk.GetOrigin()
        direction = np.array(img_itk.GetDirection()).reshape(4, 4)
        # now modify these to remove the fourth dimension
        spacing = tuple(list(spacing[:-1]))
        origin = tuple(list(origin[:-1]))
        direction = tuple(direction[:-1, :-1].reshape(-1))
        for i, t in enumerate(range(img_npy.shape[0])):
            img = img_npy[t]
            img_itk_new = sitk.GetImageFromArray(img)
            img_itk_new.SetSpacing(spacing)
            img_itk_new.SetOrigin(origin)
            img_itk_new.SetDirection(direction)
            sitk.WriteImage(
                img_itk_new,
                os.path.join(output_folder,
                             file_base[:-7] + "_%04.0d.nii.gz" % i))


def split_4d(input_folder,
             num_processes=8,
             overwrite_task_output_id=None,
             output_dir='./'):
    assert os.path.isdir(os.path.join(input_folder, "imagesTr")) and os.path.isdir(os.path.join(input_folder, "labelsTr")) and \
           os.path.isfile(os.path.join(input_folder, "dataset.json")), \
        "The input folder must be a valid Task folder from the Medical Segmentation Decathlon with at least the " \
        "imagesTr and labelsTr subdirs and the dataset.json file"

    while input_folder.endswith("/"):
        input_folder = input_folder[:-1]

    full_task_name = input_folder.split("/")[-1]

    assert full_task_name.startswith(
        "Task"
    ), "The input folder must point to a folder that starts with TaskXX_"

    first_underscore = full_task_name.find("_")
    assert first_underscore == 6, "Input folder start with TaskXX with XX being a 3-digit id: 00, 01, 02 etc"

    input_task_id = int(full_task_name[4:6])
    if overwrite_task_output_id is None:
        overwrite_task_output_id = input_task_id

    task_name = full_task_name[7:]
    output_folder = os.path.join(
        output_dir, "Task%03.0d_" % overwrite_task_output_id + task_name)
    if os.path.isdir(output_folder):
        shutil.rmtree(output_folder)
    files = []
    output_dirs = []
    os.makedirs(output_folder, exist_ok=True)
    for subdir in ["imagesTr", "imagesTs"]:
        curr_out_dir = os.path.join(output_folder, subdir)
        if not os.path.isdir(curr_out_dir):
            os.mkdir(curr_out_dir)
        curr_dir = os.path.join(input_folder, subdir)
        nii_files = [
            os.path.join(curr_dir, i) for i in os.listdir(curr_dir)
            if i.endswith(".nii.gz")
        ]
        nii_files.sort()
        for n in nii_files:
            files.append(n)
            output_dirs.append(curr_out_dir)

    shutil.copytree(os.path.join(input_folder, "labelsTr"),
                    os.path.join(output_folder, "labelsTr"))
    p = Pool(num_processes)
    p.starmap(split_4d_nifti, zip(files, output_dirs))
    p.close()
    p.join()
    shutil.copy(os.path.join(input_folder, "dataset.json"), output_folder)
