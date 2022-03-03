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

import numpy as np
import SimpleITK as sitk
import scipy


def resample(image,
             label,
             spacing=None,
             new_spacing=[1.0, 1.0, 1.0],
             new_shape=None,
             order=1):
    """
    Resample image from the original spacing to new_spacing, e.g. 1x1x1

    image(numpy array): 3D numpy array of raw HU values from CT series in [z, y, x] order.
    spacing(list|tuple): float * 3, raw CT spacing in [z, y, x] order.
    new_spacing: float * 3, new spacing used for resample, typically 1x1x1,
        which means standardizing the raw CT with different spacing all into
        1x1x1 mm.
    new_shape(list|tuple): the new shape of resampled numpy array.
    order(int): order for resample function scipy.ndimage.interpolation.zoom

    return: 3D binary numpy array with the same shape of the image after,
        resampling. The actual resampling spacing is also returned.
    """
    if new_shape is None:
        # just np.array(spacing) ?
        spacing = np.array([spacing[0], spacing[1], spacing[2]])
        new_shape = np.round(image.shape * spacing / new_spacing)
    else:
        new_shape = np.array(new_shape)

    resize_factor = new_shape / image.shape

    image_new = scipy.ndimage.zoom(
        image, resize_factor, mode='nearest', order=order)
    label_new = scipy.ndimage.zoom(
        label, resize_factor, mode='nearest', order=0)

    return image_new, label_new

def foreground_bb(image, background_value=0):
    """Get a bounding box for pixels != background_value in image.

    Args:
        image (np.ndarray): medical image
        background_value (int): value for background, pixels != this value are considered foreground
    Returns:
        list: 2d list, foreground for axis is in range [ret[axis][0], ret[axis][0])

    """
    assert image.ndim == 3, f"Only supports 3d image while received image of {image.ndim}d"

    mask = image != background_value
    bb = []
    for axis in range(3):
        reduce_axis = tuple(idx for idx in range(3) if idx != axis)
        non_zero = np.any(mask, axis=reduce_axis)
        idxs = np.where(non_zero)[0]
        bb.append([idxs[0], idxs[-1]+1])
    return bb

def crop_to_foreground(image, label=None, background=0):
    bb = foreground_bb(image, background)
    bb = tuple(slice(b[0], b[1]) for b in bb)
    print(bb)
    image = image[bb]
    if label is not None:
        label = label[bb]
    return image, label

# DEBUG: for foreground_bb crop_to_foreground
# image = np.array([
#     [
#         [0,0,0,0,0],
#         [0,0,0,0,0],
#         [0,0,0,0,0],
#         [0,0,0,0,0],
#         [0,0,0,0,0],
#         [0,0,0,0,0],
#     ],
#     [
#         [0,0,0,0,0],
#         [0,1,0,0,0],
#         [0,0,0,0,0],
#         [0,0,0,1,0],
#         [0,0,0,0,0],
#         [0,0,0,0,0],
#     ],
#     [
#         [0,0,0,0,0],
#         [0,1,0,0,0],
#         [0,0,0,0,0],
#         [0,0,0,0,0],
#         [0,0,0,1,0],
#         [0,0,0,0,0],
#     ],
# ])
# print(image.ndim, image.shape)
# print(foreground_bb(image))
#
# image = crop_to_foreground(image)
# print(image.shape)
# print(image)
# DEBUG:  debug ends
