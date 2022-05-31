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
import sys
sys.path.append("tools")
import argparse
import os
from tools.batchgenerators.utilities.file_and_folder_operations import *
from medicalseg.core.training.model_restore import restore_model
import paddle
import yaml
import shutil
import numpy as np

# from medicalseg.cvlibs import Config
from medicalseg.utils import logger


def parse_args():
    parser = argparse.ArgumentParser(description='Model export.')
    # params of training
    parser.add_argument(
        "--plan",
        dest="plan",
        help="The config file.Such as model_best.model.pkl",
        default="None",
        type=str,
        required=True)
    parser.add_argument(
        "--stage",
        dest="stage",
        help="define witch stage the model is 0 or 1",
        default=0,
        type=int,
        required=False)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the exported model',
        type=str,
        default='./output')
    parser.add_argument('-m', '--model', help="2d, 3d_lowres, 3d_fullres or 3d_cascade_fullres. Default: 3d_fullres",
                        default="3d_fullres", required=False)
    parser.add_argument(
        '--check_point',
        dest='check_point',
        help='The path of model for export.Such as model_best.model',
        type=str,
        default=None)
    parser.add_argument(
        '--without_argmax',
        dest='without_argmax',
        help='Do not add the argmax operation at the end of the network',
        action='store_true')
    parser.add_argument(
        '--with_softmax',
        dest='with_softmax',
        help='Add the softmax operation at the end of the network',
        action='store_true')
    parser.add_argument(
        "--input_shape",
        nargs='+',
        help="Export the model with fixed input shape, such as 1 3 1024 1024.",
        type=int,
        default=None)

    return parser.parse_args()


class SavedSegmentationNet(paddle.nn.Layer):
    def __init__(self, net, without_argmax=False, with_softmax=False):
        super().__init__()
        self.net = net
        self.post_processer = PostPorcesser(without_argmax, with_softmax)

    def forward(self, x):
        outs = self.net(x)
        outs = self.post_processer(outs)
        return outs


class PostPorcesser(paddle.nn.Layer):
    def __init__(self, without_argmax, with_softmax):
        super().__init__()
        self.without_argmax = without_argmax
        self.with_softmax = with_softmax

    def forward(self, outs):
        new_outs = []
        for out in outs:
            if self.with_softmax:
                out = paddle.nn.functional.softmax(out, axis=1)
            if not self.without_argmax:
                out = paddle.argmax(out, axis=1)
            new_outs.append(out)
        return new_outs


def main(args):
    os.environ['MEDICALSEG_EXPORT_STAGE'] = 'True'
    stage=args.stage
    check_point = args.check_point
    pkl_file=args.plan
    plan=load_pickle(pkl_file)
    # print(plan)
    num_classes=plan["plans"]['num_classes']
    plan=plan["plans"]["plans_per_stage"][stage]
    shape=[int(i) for i in plan['patch_size']]

    input_shape=[1,1+stage*num_classes,*shape]
    trainer = restore_model(pkl_file,check_point)
    net=trainer.network


    logger.info('Loaded trained params of model successfully.')


    shape = input_shape

    if not args.without_argmax or args.with_softmax:
        new_net = SavedSegmentationNet(net, args.without_argmax,
                                       args.with_softmax)
    else:
        new_net = net

    new_net.eval()
    # test_input=paddle.to_tensor(np.random.random(shape),dtype=paddle.float32)
    #
    # out_put=new_net(test_input)
    # print(f"output is {out_put}")
    new_net = paddle.jit.to_static(
        new_net,
        input_spec=[paddle.static.InputSpec(
            shape=shape, dtype='float32')])  # export is export to static graph
    save_path = os.path.join(args.save_dir, 'model')
    paddle.jit.save(new_net, save_path)
    shutil.copy(pkl_file,args.save_dir)
    logger.info(f'Model is saved in {args.save_dir}.')


if __name__ == '__main__':
    args = parse_args()
    main(args)
