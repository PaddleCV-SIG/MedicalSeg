# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import math

import numpy as np
import SimpleITK as sitk
import pydicom

# DEBUG:
import matplotlib.pyplot as plt


def load_slices(dcm_dir):
    """
    Load dcm like images
    Return img array and [z,y,x]-ordered origin and spacing
    """

    dcm_list = [os.path.join(dcm_dir, i) for i in os.listdir(dcm_dir)]
    indices = np.array([pydicom.dcmread(i).InstanceNumber for i in dcm_list])
    dcm_list = np.array(dcm_list)[indices.argsort()]

    itkimage = sitk.ReadImage(dcm_list)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing


def load_series(mhd_path):
    """
    Load mhd, nii like images
    Return img array and [z,y,x]-ordered origin and spacing
    """

    itkimage = sitk.ReadImage(mhd_path)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing


def reverse_axes(image):
    return np.transpose(image, tuple(reversed(range(image.ndim))))


def sitk_read(volume_path, orient=True, split=False, load_type=np.float32):
    # 1. get scan data
    # sitk wont load files without orthonormal direction cosines, try/catch next line
    sitk_f = sitk.ReadImage(volume_path)
    volume_np = sitk.GetArrayFromImage(sitk_f).astype(load_type)
    dim = sitk_f.GetDimension()

    # 2. change dimensions to z, y, x
    # s = volume_np.shape
    # if len(volume_np.shape) == 3:
    #     if s[0] == s[1]:

    print("volume_np.shape", volume_np.shape)
    print(sitk_f.GetDirection())
    print(dim, sitk_f.GetDepth(), sitk_f.GetHeight(), sitk_f.GetWidth())
    # 3. correct scan orientation, currently only works with 3d image
    if orient and dim == 3:  # TODO: add 4d orient support
        # sitk_f.SetDirection((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
        # (-1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        # volume_np = volume_np[:, ::-1, :]
        volume_np = reverse_axes(sitk.GetArrayFromImage(sitk_f))  # xyz
        direction = np.asarray(sitk_f.GetDirection())
        s = int(math.sqrt(direction.size))
        cosine = direction.reshape(s, s)
        cosine_inv = np.linalg.inv(np.round(cosine))
        swap = np.argmax(abs(cosine_inv), axis=0)
        flip = np.sum(cosine_inv, axis=0)
        volume_np = np.transpose(volume_np, tuple(swap))
        volume_np = volume_np[tuple(slice(None, None, int(f)) for f in flip)]
        volume_np = np.rot90(volume_np, -1)
        volume_np = np.transpose(volume_np, (2, 0, 1))

    # 4. split 4d series
    if not split:
        return volume_np
    if dim == 3:
        return [volume_np]
    elif dim == 4:
        volumes = []
        for idx in range(volume_np.shape[0]):
            volumes.append(volume_np[idx])
        return volumes
    else:
        raise RuntimeError(
            f"{volume_path} has {dim} dimensions, expecting 3D or 4D files")
