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

import argparse
import os
import shutil
import paddle
import numpy as np
import time
import pickle
from typing import Tuple, List
import sys

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from paddleseg3d.cvlibs import Config
# from paddleseg3d.core import evaluate
from paddleseg3d.utils import get_sys_env, logger, config_check, utils
from paddleseg3d.datasets import MSDDataset
from paddleseg3d.utils import metric, TimeAverager, calculate_eta, logger, progbar, loss_computation, add_image_vdl, sum_tensor
import paddle.nn.functional as F

from nnunet_tools.predict_next_stage import predict_next_stage
from nnunet_tools.utils import save_segmentation_nifti_from_softmax
from nnunet_tools.evaluator import NiftiEvaluator, aggregate_scores
from nnunet_tools.postprocessing import determine_postprocessing, subfiles


def to_one_hot(seg, all_seg_labels=None):
    if all_seg_labels is None:
        all_seg_labels = np.unique(seg)
    result = np.zeros((len(all_seg_labels), *seg.shape), dtype=seg.dtype)
    for i, l in enumerate(all_seg_labels):
        result[i][seg == l] = 1
    return result

    
def evaluate(model,
             eval_dataset,
             losses,
             num_workers=0,
             print_detail=True,
             auc_roc=False,
             writer=None,
             save_dir=None):
    """
    Launch evalution.

    Args:
        model（nn.Layer): A sementic segmentation model.
        eval_dataset (paddle.io.Dataset): Used to read and process validation datasets.
        losses(dict): Used to calculate the loss. e.g: {"types":[loss_1...], "coef": [0.5,...]}
        num_workers (int, optional): Num workers for data loader. Default: 0.
        print_detail (bool, optional): Whether to print detailed information about the evaluation process. Default: True.
        auc_roc(bool, optional): whether add auc_roc metric.
        writer: visualdl log writer.
        save_dir(str, optional): the path to save predicted result.

    Returns:
        float: The mIoU of validation datasets.
        float: The accuracy of validation datasets.
    """
    model.eval()

    if isinstance(eval_dataset, MSDDataset):
        loader = eval_dataset
    else:
        raise UserWarning('only support nnunet!')

    # val params====nnunet params
    model.network.inference_apply_nonlin = lambda x: F.softmax(x, 1)
    plans = loader.plans
    if 'segmentation_export_params' in plans.keys():
        force_separate_z = plans['segmentation_export_params']['force_separate_z']
        interpolation_order = plans['segmentation_export_params']['interpolation_order']
        interpolation_order_z = plans['segmentation_export_params']['interpolation_order_z']
    else:
        force_separate_z = None
        interpolation_order = 1
        interpolation_order_z = 0 
    output_base = args.val_save_folder
    output_folder = os.path.join(output_base, 'fold_{}'.format(loader.fold))
    validation_raw_folder = os.path.join(output_folder, 'validation_raw')
    if not os.path.exists(validation_raw_folder):
        os.makedirs(validation_raw_folder, exist_ok=True)
    pred_args = {'do_mirroring': True,
                         'use_sliding_window': True,
                         'step_size': 0.5,
                         'save_softmax': True,
                         'use_gaussian': True,
                         'overwrite': True,
                         'validation_folder_name': args.val_save_folder,
                         'debug': False,
                         'all_in_gpu': False,
                         'segmentation_export_kwargs': None,
                         }
    if not loader.data_aug_params['do_mirror']:
        raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
    mirror_axes = loader.data_aug_params['mirror_axes']

    print('start evaluating...')
    pred_gt_tuples = []
    gt_niftis_folder = os.path.join(loader.preprocessed_dir, "gt_segmentations")  # gt dir
    for k in loader.dataset_val.keys():
        with open(loader.dataset[k]['properties_file'], 'rb') as f:
            properties = pickle.load(f)
        fname = properties['list_of_data_files'][0].split("/")[-1][:-12]
        pred_gt_tuples.append(
            [
                os.path.join(validation_raw_folder, fname + '.nii.gz'),
                os.path.join(gt_niftis_folder, fname + '.nii.gz'),
            ]
        )
        if os.path.exists(os.path.join(validation_raw_folder, fname + '.nii.gz')):
            print('{} already exists, skip.'.format(os.path.join(validation_raw_folder, fname + '.nii.gz')))
            continue
        data = np.load(loader.dataset[k]['data_file'])['data']
        print(k, data.shape)
        data[-1][data[-1] == -1] = 0
        data = data[:-1]
        with paddle.no_grad():
            if loader.stage == 1:
                seg_pre_path = os.path.join(loader.folder_with_segs_from_prev_stage, k + "_segFromPrevStage.npz")
                if not os.path.exists(seg_pre_path):
                    raise UserWarning('cannot find stage 1 segmentation result for {}.'.format(seg_pre_path))
                seg_from_prev_stage = np.load(seg_pre_path)['data'][None]
                data = np.concatenate((data, to_one_hot(seg_from_prev_stage[0], range(1, loader.num_classes))))
                # print('stage 2: ', data.shape, data.shape, to_one_hot(seg_from_prev_stage[0], range(1, loader.num_classes)).shape)
            argmax_pred, softmax_pred = model.network.predict_3D(data, do_mirroring=pred_args['do_mirroring'], mirror_axes=mirror_axes,
                                        use_sliding_window=pred_args['use_sliding_window'], step_size=pred_args['step_size'],
                                        patch_size=loader.patch_size, regions_class_order=None,
                                        use_gaussian=pred_args['use_gaussian'], pad_border_mode='constant',
                                        pad_kwargs=None, all_in_gpu=pred_args['all_in_gpu'], verbose=True,
                                        mixed_precision=args.precision=='fp16')
        softmax_pred = softmax_pred.transpose([0] + [i + 1 for i in loader.transpose_backward])
        # print(argmax_pred.dtype, softmax_pred.dtype, argmax_pred.shape, softmax_pred.shape)
        softmax_fnmae = os.path.join(validation_raw_folder, fname + '.npz')

        if np.prod(softmax_pred.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save
            np.save(os.path.join(validation_raw_folder, fname + ".npy"), softmax_pred)
            softmax_pred = os.path.join(validation_raw_folder, fname + ".npy")

        save_segmentation_nifti_from_softmax(softmax_pred, os.path.join(validation_raw_folder, fname + '.nii.gz'), 
            properties, interpolation_order, None, None, None, softmax_fnmae, None, force_separate_z,
            interpolation_order_z,    
        )
    
    # 对比
    _ = aggregate_scores(pred_gt_tuples, labels=list(range(loader.num_classes)),
                             json_output_file=os.path.join(validation_raw_folder, "summary.json"),
                             json_name=" val tiled %s" % (str(pred_args['use_sliding_window'])),
                             json_author="justld",
                             json_task='task', num_threads=2)
    # postprocessing
    determine_postprocessing(output_folder, gt_niftis_folder, 'validation_raw',
                                     final_subf_name='validation_raw' + "_postprocessed", debug=False)

    # 复制gt
    gt_nifti_folder = os.path.join(output_base, "gt_niftis")

    if not os.path.exists(gt_nifti_folder):
        os.makedirs(gt_nifti_folder, exist_ok=True)
    print('copy gt from {} to {}.'.format(gt_niftis_folder, gt_nifti_folder))
    for f in subfiles(gt_niftis_folder, suffix=".nii.gz"):
        success = False
        attempts = 0
        e = None
        while not success and attempts < 10:
            try:
                shutil.copy(f, gt_nifti_folder)
                success = True
            except OSError as e:
                attempts += 1
        if not success:
            print("Could not copy gt nifti file %s into folder %s" % (f, gt_nifti_folder))
            if e is not None:
                raise e

    if args.predict_next_stage:
        predict_next_stage(model, plans, loader, loader.dataset_directory, os.path.join(loader.dataset_directory, plans['data_identifier'] + "_stage%d" % 1), args.precision=='fp16')


def parse_args():
    parser = argparse.ArgumentParser(description='Model evaluation')

    # params of evaluate
    parser.add_argument("--config",
                        dest="cfg",
                        help="The config file.",
                        default=None,
                        type=str)

    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for evaluation',
        type=str,
        default=
        "saved_model/vnet_lung_coronavirus_128_128_128_15k/best_model/model.pdparams"
    )
    parser.add_argument(
        '--predict_next_stage', 
        action='store_true', 
        default=False, 
        help='whether predict stage 2 training data.')

    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The path to save result',
        type=str,
        default="saved_model/vnet_lung_coronavirus_128_128_128_15k/best_model")
    parser.add_argument(
        '--val_save_folder',
        dest='val_save_folder',
        help='The path to val data predicted result',
        type=str,
        default="val_save_folder")

    parser.add_argument('--num_workers',
                        dest='num_workers',
                        help='Num workers for data loader',
                        type=int,
                        default=0)

    parser.add_argument(
        '--print_detail',  # the dest cannot have space in it
        help='Whether to print evaluate values',
        type=bool,
        default=True)

    parser.add_argument('--use_vdl',
                        help='Whether to use visualdl to record result images',
                        type=bool,
                        default=True)

    parser.add_argument('--auc_roc',
                        help='Whether to use auc_roc metric',
                        type=bool,
                        default=False)

    parser.add_argument(
        "--precision",
        default="fp32",
        type=str,
        choices=["fp32", "fp16"],
        help="Use AMP (Auto mixed precision) if precision='fp16'. If precision='fp32', the training is normal."
    )

    return parser.parse_args()


def main(args):
    env_info = get_sys_env()
    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else 'cpu'

    paddle.set_device(place)
    if not args.cfg:
        raise RuntimeError('No configuration file specified.')

    cfg = Config(args.cfg)
    losses = cfg.loss

    val_dataset = cfg.val_dataset
    if val_dataset is None:
        raise RuntimeError(
            'The verification dataset is not specified in the configuration file.'
        )
    elif len(val_dataset) == 0:
        raise ValueError(
            'The length of val_dataset is 0. Please check if your dataset is valid'
        )

    msg = '\n---------------Config Information---------------\n'
    msg += str(cfg)
    msg += '------------------------------------------------'
    logger.info(msg)

    model = cfg.model
    if args.model_path:
        utils.load_entire_model(model, args.model_path)
        logger.info('Loaded trained params of model successfully')

    config_check(cfg, val_dataset=val_dataset)

    evaluate(model,
             val_dataset,
             losses,
             num_workers=args.num_workers,
             print_detail=args.print_detail,
             auc_roc=args.auc_roc,
             save_dir=args.save_dir)


if __name__ == '__main__':
    args = parse_args()
    main(args)
