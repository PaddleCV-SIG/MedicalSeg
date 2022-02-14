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


def list_files(path, filter_suffix=None, filter_key=None, include=None):
    """
    list all the filename in a given path recursively. if needed filter the names with suffix or key string
    When filter_key is not None，use include key to select only filenames with key in it.

    Args:

    """
    fname = []
    for root, _, f_names in os.walk(path):
        for f in f_names:
            # skip all hidden files
            if f.startswith("."):
                continue
            if filter_suffix is not None:
                if f[-len(filter_suffix):] == filter_suffix:
                    fname.append(os.path.join(root, f))
            elif filter_key is not None:
                assert include is not None, print(
                    'if you want to filter with filter key, "include" need to be True or False.'
                )

                if (filter_key in f and include) or (filter_key not in f
                                                     and not include):
                    fname.append(os.path.join(root, f))
            else:
                fname.append(os.path.join(root, f))

    return fname
