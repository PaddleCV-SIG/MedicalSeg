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


def HU2float32(image, HU_min=-1200, HU_max=600, HU_nan=-2000, dtype="float32"):
    """
    Convert HU unit into uint8 values. First bound HU values by predfined min
    and max, and then normalize. Due to paddle.nn.conv3D doesn't support uint8, we need to convert
    the returned image as float32

    image: 3D numpy array of raw HU values from CT series in [z, y, x] order.
    HU_min: float, min HU value.
    HU_max: float, max HU value.
    HU_nan: float, value for nan in the raw CT image.
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    image = np.nan_to_num(image, copy=False, nan=HU_nan)

    # normalize to [0, 1]
    image = (image - HU_min) / ((HU_max - HU_min) / 255)
    np.clip(image, 0, 255, out=image)
    image = image.astype(dtype)

    return image
