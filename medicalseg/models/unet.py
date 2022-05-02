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
from ossaudiodev import control_labels
import sys
from telnetlib import OUTMRK
from turtle import forward

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.."))

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from medicalseg.cvlibs import manager
from medicalseg.utils import utils


class EncoderBlock(nn.Layer):
    def __init__(
        self,
        name_scope,
        in_channels=None,
        kernel_number=8,
        downsample=True,
        norm=True,
        kernel_size=3,
    ):
        super(EncoderBlock, self).__init__(name_scope=name_scope)
        self.norm = norm
        if in_channels is None:
            in_channels = kernel_number // 2

        self.lrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout3D(p=0.6)

        if norm:
            self.norm1 = nn.InstanceNorm3D(kernel_number)
            self.norm2 = nn.InstanceNorm3D(kernel_number)
        self.norm3 = nn.InstanceNorm3D(kernel_number)

        self.conv1 = nn.Conv3D(
            in_channels=in_channels,
            out_channels=kernel_number,
            kernel_size=kernel_size,
            stride=2 if downsample else 1,
            padding="SAME",
            bias_attr=False,
        )
        self.conv2 = nn.Conv3D(
            in_channels=kernel_number,
            out_channels=kernel_number,
            kernel_size=kernel_size,
            stride=1,
            padding="SAME",
            bias_attr=False,
        )
        self.conv3 = nn.Conv3D(
            in_channels=kernel_number,
            out_channels=kernel_number,
            kernel_size=kernel_size,
            stride=1,
            padding="SAME",
            bias_attr=False,
        )

    def forward(self, x):
        out = self.conv1(x)
        residual = out
        if self.norm:
            out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.dropout(out)
        if self.norm:
            out = self.norm2(out)
        out = self.lrelu(out)
        out = self.conv3(out)
        out += residual
        out = self.norm3(out)
        out = self.lrelu(out)
        return out


class DecoderBlock(nn.Layer):
    def __init__(self, name_scope, kernel_number, num_classes, kernel_size=3, stride=1):
        super(DecoderBlock, self).__init__(name_scope=name_scope)
        self.dropout = nn.Dropout3D(p=0.6)
        self.lrelu = nn.LeakyReLU()

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", data_format="NCDHW")
        self.conv1 = nn.Conv3D(
            in_channels=kernel_number * 2,
            out_channels=kernel_number,
            kernel_size=kernel_size,
            stride=stride,
            padding="SAME",
            bias_attr=False,
        )
        self.norm1 = nn.InstanceNorm3D(kernel_number)

        self.conv2 = nn.Conv3D(
            in_channels=kernel_number * 2,
            out_channels=kernel_number,
            kernel_size=kernel_size,
            stride=stride,
            padding="SAME",
            bias_attr=False,
        )
        self.norm2 = nn.InstanceNorm3D(kernel_number)

        self.conv3 = nn.Conv3D(
            in_channels=kernel_number,
            out_channels=kernel_number,
            kernel_size=kernel_size,
            stride=stride,
            padding="SAME",
            bias_attr=False,
        )

        self.shortcut_conv = nn.Conv3D(
            in_channels=kernel_number,
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            padding="SAME",
            bias_attr=False,
        )

        self.norm3 = nn.InstanceNorm3D(kernel_number)

    def forward(self, x, skip):
        out = self.upsample(x)

        out = paddle.concat([out, skip], axis=1)

        out = self.conv1(out)
        out = self.norm1(out)
        out = self.lrelu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.lrelu(out)
        shortcut = out

        out = self.conv3(shortcut)
        out = self.norm3(out)
        out = self.lrelu(out)

        shortcut = self.shortcut_conv(shortcut)
        return out, shortcut


@manager.MODELS.add_component
class UNet(nn.Layer):
    """
    Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
    """

    def __init__(self, in_channels, num_classes, pretrained=None, base_n_kernel=8):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.best_loss = 1000000

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", data_format="NCDHW")
        self.pad = nn.Pad3D([1, 0, 1, 0, 1, 0])
        self.padded = False

        self.encb1 = EncoderBlock(
            "encoder1", in_channels, base_n_kernel * 2**0, downsample=False, norm=False
        )  # 8, orig
        self.encb2 = EncoderBlock("encoder2", kernel_number=base_n_kernel * 2**1)  # 16, orig/2
        self.encb3 = EncoderBlock("encoder3", kernel_number=base_n_kernel * 2**2)  # 32, orig/4
        self.encb4 = EncoderBlock("encoder4", kernel_number=base_n_kernel * 2**3)  # 64, orig/8

        self.encb5 = EncoderBlock(
            "encoder5", in_channels=base_n_kernel * 2**3, kernel_number=base_n_kernel * 2**4
        )  # 128, orig/16

        self.decb4 = DecoderBlock("decoder4", base_n_kernel * 2**3, num_classes)
        self.decb3 = DecoderBlock("decoder3", base_n_kernel * 2**2, num_classes)
        self.decb2 = DecoderBlock("decoder2", base_n_kernel * 2**1, num_classes)
        self.decb1 = DecoderBlock("decoder1", base_n_kernel * 2**0, num_classes)

        self.decconv = nn.Conv3D(
            in_channels=base_n_kernel, out_channels=num_classes, kernel_size=1, stride=1
        )

        self.outputconv = nn.Conv3D(
            in_channels=num_classes * 2, out_channels=num_classes, kernel_size=1, stride=1
        )

    def forward(self, x):
        if x.shape[2] % 2 != 0:
            x = self.pad(x)
            self.padded = True

        enc1 = self.encb1(x)
        enc2 = self.encb2(enc1)
        enc3 = self.encb3(enc2)
        enc4 = self.encb4(enc3)
        enc5 = self.encb5(enc4)

        out, ds4 = self.decb4(enc5, enc4)
        out, ds3 = self.decb3(enc4, enc3)
        out, ds2 = self.decb2(enc3, enc2)
        out, _ = self.decb1(enc2, enc1)

        out = self.decconv(out)

        ds4_up = self.upsample(ds4)
        ds3 += ds4_up
        ds3_up = self.upsample(ds3)
        ds2 += ds3_up
        ds2_up = self.upsample(ds2)
        out = out + ds2_up

        if self.padded:
            out = out[:, :, 1:, 1:, 1:]

        return [out]


if __name__ == "__main__":
    size = 64
    num_classes = 3
    input = paddle.static.InputSpec([None, 1, size, size, size], "float32", "x")
    label = paddle.static.InputSpec([None, num_classes, size, size, size], "int64", "label")

    unet = UNet(in_channels=1, num_classes=3)

    paddle.Model(unet, input, label).summary()

    input = paddle.rand((2, 1, size, size, size))
    print("input", input.shape)


    output = unet(input)
    print("output", output[0].shape)
