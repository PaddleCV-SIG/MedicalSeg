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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import os
import sys

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.."))

from paddleseg3d.cvlibs import manager
from paddleseg3d.utils import utils


class LUConv(nn.Layer):

    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = nn.ELU() if elu else nn.PReLU(nchan)
        self.conv1 = nn.Conv3D(nchan, nchan, kernel_size=5, padding=2)

        self.bn1 = nn.BatchNorm3D(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))

        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Layer):

    def __init__(self, in_channels, elu):
        super(InputTransition, self).__init__()
        self.num_features = 16
        self.in_channels = in_channels

        self.conv = nn.Conv3D(self.in_channels,
                              self.num_features,
                              kernel_size=5,
                              padding=2)

        self.bn = nn.BatchNorm3D(self.num_features)

        self.relu = nn.ELU() if elu else nn.PReLU(self.num_features)

    def forward(self, x):
        out = self.conv(x)
        repeat_rate = int(self.num_features / self.in_channels)
        out = self.bn(out)
        x_tile = x.tile([1, repeat_rate, 1, 1, 1])
        return self.relu(paddle.add(out, x_tile))


class DownTransition(nn.Layer):

    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2 * inChans
        self.if_dropout = dropout
        # todo: replace with module
        self.down_conv = nn.Conv3D(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm3D(outChans)
        self.relu1 = nn.ELU() if elu else nn.PReLU(outChans)
        self.relu2 = nn.ELU() if elu else nn.PReLU(outChans)
        self.dropout = nn.Dropout3D()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.dropout(down) if self.if_dropout else down
        out = self.ops(out)
        out = paddle.add(out, down)
        out = self.relu2(out)

        return out


class UpTransition(nn.Layer):

    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.Conv3DTranspose(inChans,
                                          outChans // 2,
                                          kernel_size=2,
                                          stride=2)

        self.bn1 = nn.BatchNorm3D(outChans // 2)
        self.relu1 = nn.ELU() if elu else nn.PReLU(outChans // 2)
        self.relu2 = nn.ELU() if elu else nn.PReLU(outChans)
        self.if_dropout = dropout
        self.dropout1 = nn.Dropout3D()
        self.dropout2 = nn.Dropout3D()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.dropout1(x) if self.if_dropout else x
        skipx = self.dropout2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = paddle.concat((out, skipx), 1)
        out = self.ops(xcat)
        out = self.relu2(paddle.add(out, xcat))

        return out


class OutputTransition(nn.Layer):

    def __init__(self, in_channels, num_classes, elu):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3D(in_channels,
                               num_classes,
                               kernel_size=5,
                               padding=2)
        self.bn1 = nn.BatchNorm3D(num_classes)

        self.conv2 = nn.Conv3D(num_classes, num_classes, kernel_size=1)
        self.relu1 = nn.ELU() if elu else nn.PReLU(num_classes)

    def forward(self, x):
        # convolve 32 down to channels as the desired classes
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        return out


@manager.MODELS.add_component
class VNet(nn.Layer):
    """
    Implementations based on the Vnet paper: https://arxiv.org/abs/1606.04797
    """

    def __init__(self,
                 elu=False,
                 in_channels=1,
                 num_classes=4,
                 pretrained=None):
        super().__init__()
        self.best_loss = 1000000
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.in_tr = InputTransition(in_channels, elu=elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=True)
        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32, num_classes, elu)

        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return [
            out,
        ]

    def test(self):
        import numpy as np
        np.random.seed(1)
        a = np.random.rand(1, self.in_channels, 32, 32, 32)
        input_tensor = paddle.to_tensor(a, dtype='float32')

        ideal_out = paddle.rand((1, self.num_classes, 32, 32, 32))
        out = self.forward(input_tensor)
        print("out", out.mean(), input_tensor.mean())

        assert ideal_out.shape == out.shape
        paddle.summary(self, (1, self.in_channels, 32, 32, 32))

        print("Vnet test is complete")


@manager.MODELS.add_component
class VNetLight(nn.Layer):
    """
    A lighter version of Vnet that skips down_tr256 and up_tr256 in oreder to reduce time and space complexity
    """

    def __init__(self, elu=True, in_channels=1, num_classes=4):
        super().__init__()
        self.best_loss = 1000000
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.in_tr = InputTransition(in_channels, elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
        self.up_tr128 = UpTransition(128, 128, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32, num_classes, elu)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out = self.up_tr128(out128, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out

    def test(self, device='cpu'):
        input_tensor = paddle.rand([1, self.in_channels, 32, 32, 32])
        ideal_out = paddle.rand([1, self.num_classes, 32, 32, 32])
        out = self.forward(input_tensor)
        assert ideal_out.shape == out.shape
        paddle.summary(self, (self.in_channels, 32, 32, 32))

        print("Vnet light test is complete")


if __name__ == "__main__":
    m = VNet(in_channels=1, num_classes=3)
    # m.test()
    x = paddle.randn([2, 1, 128, 128, 128])
    out = m(x)
    out[0].backward()
    for name, tensor in m.named_parameters():
        grad = tensor.grad
        if grad is not None:
            print(name, grad.mean())
