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


def subdirs(folder: str,
            join: bool = True,
            prefix: str = None,
            suffix: str = None,
            sort: bool = True):
    if join:
        j = os.path.join
    else:
        j = lambda x, y: y
    res = [
        j(folder, i) for i in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, i)) and (
            prefix is None or i.startswith(prefix)) and (
                suffix is None or i.endswith(suffix))
    ]
    if sort:
        res.sort()
    return res


def subfiles(folder: str,
             join: bool = True,
             prefix: str = None,
             suffix: str = None,
             sort: bool = True):
    if join:
        j = os.path.join
    else:
        j = lambda x, y: y
    res = [
        j(folder, i) for i in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, i)) and (
            prefix is None or i.startswith(prefix)) and (
                suffix is None or i.endswith(suffix))
    ]
    if sort:
        res.sort()
    return res


def remove_trailing_slash(filename: str):
    while filename.endswith('/'):
        filename = filename[:-1]
    return filename
