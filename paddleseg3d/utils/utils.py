# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import contextlib
import filelock
import os
import os.path as osp
import random
from urllib.parse import urlparse, unquote

import tempfile
import numpy as np

import paddle

from paddleseg3d.utils import logger, seg_env
from paddleseg3d.utils.download import download_file_and_uncompress


@contextlib.contextmanager
def generate_tempdir(directory: str = None, **kwargs):
    '''Generate a temporary directory'''
    directory = seg_env.TMP_HOME if not directory else directory
    with tempfile.TemporaryDirectory(dir=directory, **kwargs) as _dir:
        yield _dir


def load_entire_model(model, pretrained):
    if pretrained is not None:
        load_pretrained_model(model, pretrained)
    else:
        logger.warning('Not all pretrained params of {} are loaded, ' \
                       'training from scratch or a pretrained backbone.'.format(model.__class__.__name__))


def download_pretrained_model(pretrained_model):
    """
    Download pretrained model from url.
    Args:
        pretrained_model (str): the url of pretrained weight
    Returns:
        str: the path of pretrained weight
    """
    assert urlparse(pretrained_model).netloc, "The url is not valid."

    pretrained_model = unquote(pretrained_model)
    savename = pretrained_model.split('/')[-1]
    if not savename.endswith(('tgz', 'tar.gz', 'tar', 'zip')):
        savename = pretrained_model.split('/')[-2]
    else:
        savename = savename.split('.')[0]

    with generate_tempdir() as _dir:
        with filelock.FileLock(os.path.join(seg_env.TMP_HOME, savename)):
            pretrained_model = download_file_and_uncompress(
                pretrained_model,
                savepath=_dir,
                extrapath=seg_env.PRETRAINED_MODEL_HOME,
                extraname=savename)
            pretrained_model = os.path.join(pretrained_model, 'model.pdparams')
    return pretrained_model


def load_pretrained_model(model, pretrained_model):
    if pretrained_model is not None:
        logger.info(
            'Loading pretrained model from {}'.format(pretrained_model))

        if urlparse(pretrained_model).netloc:
            pretrained_model = download_pretrained_model(pretrained_model)

        if os.path.exists(pretrained_model):
            para_state_dict = paddle.load(pretrained_model)

            model_state_dict = model.state_dict()
            keys = model_state_dict.keys()
            num_params_loaded = 0
            for k in keys:
                if k not in para_state_dict:
                    logger.warning("{} is not in pretrained model".format(k))
                elif list(para_state_dict[k].shape) != list(
                        model_state_dict[k].shape):
                    logger.warning(
                        "[SKIP] Shape of pretrained params {} doesn't match.(Pretrained: {}, Actual: {})"
                        .format(k, para_state_dict[k].shape,
                                model_state_dict[k].shape))
                else:
                    model_state_dict[k] = para_state_dict[k]
                    num_params_loaded += 1
            model.set_dict(model_state_dict)
            logger.info("There are {}/{} variables loaded into {}.".format(
                num_params_loaded, len(model_state_dict),
                model.__class__.__name__))

        else:
            raise ValueError(
                'The pretrained model directory is not Found: {}'.format(
                    pretrained_model))
    else:
        logger.info(
            'No pretrained model to load, {} will be trained from scratch.'.
            format(model.__class__.__name__))


def resume(model, optimizer, resume_model):
    if resume_model is not None:
        logger.info('Resume model from {}'.format(resume_model))
        if os.path.exists(resume_model):
            resume_model = os.path.normpath(resume_model)
            ckpt_path = os.path.join(resume_model, 'model.pdparams')
            para_state_dict = paddle.load(ckpt_path)
            ckpt_path = os.path.join(resume_model, 'model.pdopt')
            opti_state_dict = paddle.load(ckpt_path)
            model.set_state_dict(para_state_dict)
            optimizer.set_state_dict(opti_state_dict)

            iter = resume_model.split('_')[-1]
            iter = int(iter)
            return iter
        else:
            raise ValueError(
                'Directory of the model needed to resume is not Found: {}'.
                format(resume_model))
    else:
        logger.info('No model needed to resume.')


def worker_init_fn(worker_id):
    np.random.seed(random.randint(0, 100000))


def get_image_list(image_path, valid_suffix=None, filter_key={}):
    """Get image list from image name or image directory name with valid suffix.

    If needed, filter_key can be used to whether 'include' the key word.
    When filter_key is not None，it indicates whether filenames should include certain key.


    Args:
    image_path(str): the image or image folder where you want to get a image list from.
    valid_suffix(tuple): Contain only the suffix you want to include.
    filter_key(dict): the key and whether you want to include it. e.g.:{"segmentation": True} will futher filter the imagename with segmentation in it.

    """
    # TODO: maybe add filter for pixel spacing and 2d slice count. that can be useful for dicom

    if valid_suffix is None:
        valid_suffix = (
            'nii.gz', 'nii', 'dcm', 'nrrd', 'mhd', 'raw', 'npy', 'mha'
        )

    image_list = []
    # 1. load a single image
    if osp.isfile(image_path):
        if osp.basename(image_path).endswith(valid_suffix):
            image_list = [image_path]
        else:
            raise FileNotFoundError(
                "{} doesn't end with any of the supported suffix: {}.".format(image_path, valid_suffix))

    # 2. load image in a directory
    elif osp.isdir(image_path):
        for root, dirs, files in os.walk(image_path):
            for f in files:
                if '.ipynb_checkpoints' in root:
                    continue
                if f.endswith(valid_suffix):
                    image_list.append(osp.join(root, f))

    # 3. not dir not file, image_path doesn't exist
    else:
        raise FileNotFoundError(
            'image_path {} is not found. It should be a path of image, or a directory containing images.'.format(image_path)
        )

    # 4. filter based on filter_key
    def satisfy_filter(path):
        f_name = osp.basename(path)
        for key, val in filter_key:
            if (key in f_name) is not val:
                return False
        return True

    image_list = list(filter(satisfy_filter, image_list))
    # TODO: dcm should only return one file in a series

    if len(image_list) == 0:
        raise RuntimeError(
            'No image found in `image_path`={}'.format(image_path))

    return image_list
