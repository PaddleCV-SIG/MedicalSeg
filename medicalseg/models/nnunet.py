# Implementation of this model is borrowed and modified
# (from torch to paddle) from here:
# https://github.com/black0017/MedicalZooPytorch

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
import sys
sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.."))
import pickle
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

from paddleseg3d.cvlibs import manager
from paddleseg3d.utils import utils
from .generic_unet import Generic_UNet


@manager.MODELS.add_component
class NNUNet(nn.Layer):
    def __init__(self, plans_path, stage=None, num_classes=0):
        super().__init__()
        self.plans_path = plans_path
        self.stage = stage
        plans = self.load_plans_file(plans_path)
        self.process_plans(plans)

        if self.threeD:
            conv_op = nn.Conv3D
            dropout_op = nn.Dropout3D
            norm_op = nn.InstanceNorm3D
        else:
            conv_op = nn.Conv2D
            dropout_op = nn.Dropout2D
            norm_op = nn.InstanceNorm2D

        norm_op_kwargs = {'epsilon': 1e-5}
        dropout_op_kwargs = {'p': 0}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, 1e-2,
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
    
    def load_plans_file(self, plans_path):
        with open(plans_path, 'rb') as f:
            plans = pickle.load(f)
        return plans

    def process_plans(self, plans):
        if self.stage is None:
            assert len(list(plans['plans_per_stage'].keys())) == 1, \
                "If self.stage is None then there can be only one stage in the plans file. That seems to not be the " \
                "case. Please specify which stage of the cascade must be trained"
            self.stage = list(plans['plans_per_stage'].keys())[0]
        self.plans = plans

        stage_plans = self.plans['plans_per_stage'][self.stage]
        # self.batch_size = stage_plans['batch_size']
        self.net_pool_per_axis = stage_plans['num_pool_per_axis']
        self.patch_size = np.array(stage_plans['patch_size']).astype(int)
        # self.do_dummy_2D_aug = stage_plans['do_dummy_2D_data_aug']

        if 'pool_op_kernel_sizes' not in stage_plans.keys():
            assert 'num_pool_per_axis' in stage_plans.keys()
            self.print_to_log_file("WARNING! old plans file with missing pool_op_kernel_sizes. Attempting to fix it...")
            self.net_num_pool_op_kernel_sizes = []
            for i in range(max(self.net_pool_per_axis)):
                curr = []
                for j in self.net_pool_per_axis:
                    if (max(self.net_pool_per_axis) - j) <= i:
                        curr.append(2)
                    else:
                        curr.append(1)
                self.net_num_pool_op_kernel_sizes.append(curr)
        else:
            self.net_num_pool_op_kernel_sizes = stage_plans['pool_op_kernel_sizes']

        if 'conv_kernel_sizes' not in stage_plans.keys():
            self.print_to_log_file("WARNING! old plans file with missing conv_kernel_sizes. Attempting to fix it...")
            self.net_conv_kernel_sizes = [[3] * len(self.net_pool_per_axis)] * (max(self.net_pool_per_axis) + 1)
        else:
            self.net_conv_kernel_sizes = stage_plans['conv_kernel_sizes']

        # self.pad_all_sides = None  # self.patch_size
        # self.intensity_properties = plans['dataset_properties']['intensityproperties']
        # self.normalization_schemes = plans['normalization_schemes']
        self.base_num_features = plans['base_num_features']
        self.num_input_channels = plans['num_modalities']
        self.num_classes = plans['num_classes'] + 1  # background is no longer in num_classes
        if self.stage == 1:
            self.num_input_channels += (self.num_classes - 1)  # for seg from prev stage
        
        
        self.classes = plans['all_classes']
        # self.use_mask_for_norm = plans['use_mask_for_norm']
        # self.only_keep_largest_connected_component = plans['keep_only_largest_region']
        # self.min_region_size_per_class = plans['min_region_size_per_class']
        # self.min_size_per_class = None  # DONT USE THIS. plans['min_size_per_class']

        # if plans.get('transpose_forward') is None or plans.get('transpose_backward') is None:
        #     print("WARNING! You seem to have data that was preprocessed with a previous version of nnU-Net. "
        #           "You should rerun preprocessing. We will proceed and assume that both transpose_foward "
        #           "and transpose_backward are [0, 1, 2]. If that is not correct then weird things will happen!")
        #     plans['transpose_forward'] = [0, 1, 2]
        #     plans['transpose_backward'] = [0, 1, 2]
        # self.transpose_forward = plans['transpose_forward']
        # self.transpose_backward = plans['transpose_backward']

        if len(self.patch_size) == 2:
            self.threeD = False
        elif len(self.patch_size) == 3:
            self.threeD = True
        else:
            raise RuntimeError("invalid patch size in plans file: %s" % str(self.patch_size))

        if "conv_per_stage" in plans.keys():  # this ha sbeen added to the plans only recently
            self.conv_per_stage = plans['conv_per_stage']
        else:
            self.conv_per_stage = 2
    
    def forward(self, x):
        return [self.network(x)]






