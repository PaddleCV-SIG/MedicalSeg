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
import pickle
sys.path.insert(0,"..")
LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(LOCAL_PATH, '..', '..'))

print(sys.path)
from tools.utilities.sitk_stuff import copy_geometry
from tools.utilities.one_hot_encoding import to_one_hot
from tools.batchgenerators.augmentations.utils import resize_segmentation
import argparse
import codecs

import SimpleITK as sitk
from skimage.transform import resize
import yaml
import numpy as np
import functools

from paddle.inference import create_predictor, PrecisionType
from paddle.inference import Config as PredictConfig
from tqdm import tqdm
import medicalseg.transforms as T
from medicalseg.cvlibs import manager
from medicalseg.utils import get_sys_env, logger, get_image_list
from medicalseg.utils.visualize import get_pseudo_color_map
from tools import HUnorm, resample
from tools import Prep


def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument(
        "--model_path",
        dest="model_path",
        default=None,
        type=str,
        required=True)
    parser.add_argument(
        '--image_path',
        dest='image_path',
        help='The directory or path or file list of the images to be predicted.',
        type=str,
        default=None,
        required=True)

    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the predict result.',
        type=str,
        default='./output')
    parser.add_argument(
        '--device',
        choices=['cpu', 'gpu'],
        default="gpu",
        help="Select which device to inference, defaults to gpu.")

    parser.add_argument(
        "--precision",
        default="fp32",
        type=str,
        choices=["fp32", "fp16", "int8"],
        help='The tensorrt precision.')
    parser.add_argument(
        "--lower_path",
        default=r"D:\work\cache\output_model\3d_lowres",
        type=str,
        help='3d_lower path when we use cascade it is necessary'
    )

    parser.add_argument(
        '--cpu_threads',
        default=10,
        type=int,
        help='Number of threads to predict when using cpu.')

    parser.add_argument(
        "--model_name",
        default="3d_cascade_fullres",
        type=str,
        help='witch stratage nnunet is used'
    )

    parser.add_argument(
        '--with_argmax',
        dest='with_argmax',
        help='Perform argmax operation on the predict result.',
        action='store_true')
    return parser.parse_args()


def load_pickle(file: str, mode: str = 'rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a

class DeployConfig:
    def __init__(self, path,stage=0):
        self.path=path
        self.stage=stage
        self.model=os.path.join(self.path,'model.pdmodel')
        self.params=os.path.join(self.path,"model.pdiparams")
    def read_plan(self):
        planfile=[file for file in os.listdir(self.path) if file.endswith(".pkl")]
        plan=load_pickle(os.path.join(self.path,planfile[0]))['plans']
        patch_size=plan['plans_per_stage'][self.stage]['patch_size']
        target_spacing=plan['plans_per_stage'][self.stage]['current_spacing']
        maxbound=list(plan['dataset_properties']['intensityproperties'].values())[-1]['percentile_99_5']
        minbound=list(plan['dataset_properties']['intensityproperties'].values())[-1]['percentile_00_5']
        sd=list(plan['dataset_properties']['intensityproperties'].values())[-1]['sd']
        mean=list(plan['dataset_properties']['intensityproperties'].values())[-1]['mean']
        num_classes=plan['num_classes']

        return target_spacing,patch_size,maxbound,minbound,sd,mean,num_classes


class SegPredictor:
    def __init__(self, model_path,device="gpu",stage=0):
        self.cfg = DeployConfig(model_path,stage)
        #获得model名字，注意如果是cascade
        self._init_base_config()
        if device == 'cpu':
            self._init_cpu_config()
        else:
            self._init_gpu_config()
        self.predictor = create_predictor(self.pred_cfg)
        target_spacing,patch_size,maxbound,minbound,sd,mean,num_classes=self.cfg.read_plan()
        self.target_spacing=target_spacing
        self.patch_size=patch_size
        self.maxbound=maxbound
        self.minbound=minbound
        self.sd=sd
        self.mean=mean
        self.num_classes=num_classes

    def make_steps(self):
        if len(self.patch_size)==2:
            z_size=1
        else:
            z_size=self.patch_size[0]
        x_size=self.patch_size[-2]
        y_size=self.patch_size[-1]
        self.predict_steps=[]
        for z_start in range(0,self.newshape[0],z_size):
            for x_start in range(0,self.newshape[1],x_size):
                for y_start in range(0,self.newshape[2],y_size):
                    z_end=min(z_start+z_size,self.newshape[0])
                    x_end=min(x_start+x_size,self.newshape[1])
                    y_end=min(y_start+y_size,self.newshape[2])
                    step=[[max(z_end-z_size,0),z_end],[max(x_end-x_size,0),x_end],[max(y_end-y_size,0),y_end]]
                    # print("we will deal step:{}".format(step))
                    self.predict_steps.append(step)



    def data_normalize(self,image_array):
        image_array = np.clip(image_array, self.minbound, self.maxbound)
        image_array= (image_array - self.mean) / self.sd
        return image_array

    def preprocess(self,image_path,pre_array=None):
        img=sitk.ReadImage(image_path)
        original_spacing=np.array(img.GetSpacing())[[2, 1, 0]]
        img_array= sitk.GetArrayFromImage(img)
        self.shape=img_array.shape
        self.newshape=np.round(((np.array(original_spacing) / np.array(self.target_spacing)).astype(float) * self.shape)).astype(int)
        new_array=resize(img_array,self.newshape)
        self.new_array=self.data_normalize(new_array)
        self.new_array=self.new_array[None]
        if not pre_array is None:
            pre_array=resize_segmentation(pre_array,self.newshape)
            pre_onehot=to_one_hot(pre_array,range(1,self.num_classes))
            self.new_array=np.concatenate([self.new_array,pre_onehot],0)


    def get_output(self,work_array):
        if(len(self.patch_size)==3):
            array_shape = work_array.shape
            input_array=np.zeros([array_shape[0],*self.patch_size])

            input_array[0:array_shape[0],0:array_shape[1],0:array_shape[2],0:array_shape[3]]=work_array
            input_array=input_array[None]
        else:
            array_shape = work_array.shape
            input_array=np.zeros([array_shape[0],*self.patch_size])
            input_array[0:array_shape[1], 0:array_shape[2], 0:array_shape[3]] = work_array[0]
            input_array = input_array[None]
        input_array=input_array.astype(np.float32)

        input_names = self.predictor.get_input_names()
        input_handle = self.predictor.get_input_handle(input_names[0])
        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])
        input_handle.reshape(input_array.shape)
        input_handle.copy_from_cpu(input_array)
        self.predictor.run()
        output = output_handle.copy_to_cpu()
        if (len(self.patch_size) == 3):
            return output[0,0:array_shape[1], 0:array_shape[2], 0:array_shape[3]]
        else:
            return output[0:array_shape[1], 0:array_shape[2], 0:array_shape[3]]

    def get_result(self,path,post_back=False,pre_array=None):
        self.preprocess(path,pre_array)
        self.make_steps()
        result=np.zeros((self.newshape))
        for step in tqdm(self.predict_steps):
            z_start,z_end=step[0]
            x_start,x_end=step[1]
            y_start,y_end=step[2]
            result[z_start:z_end,x_start:x_end,y_start:y_end]=self.get_output(self.new_array[:,z_start:z_end,
                                                                                   x_start:x_end,y_start:y_end])
        if post_back:
            result=resize_segmentation(result,self.shape)
        return result

    def _init_base_config(self):
        "初始化基础配置"
        self.pred_cfg = PredictConfig(self.cfg.model, self.cfg.params)

        self.pred_cfg.disable_glog_info()
        self.pred_cfg.enable_memory_optim()
        self.pred_cfg.switch_ir_optim(True)

    def _init_cpu_config(self):
        """
        Init the config for x86 cpu.
        """
        logger.info("Use CPU")
        self.pred_cfg.disable_gpu()
        self.pred_cfg.set_cpu_math_library_num_threads(10)

    def _init_gpu_config(self):
        """
        Init the config for nvidia gpu.
        """
        logger.info("Use GPU")
        self.pred_cfg.enable_use_gpu(100, 0)

def save_output(output_file,result,input_file):
    img_in = sitk.ReadImage(input_file)
    img_out_itk = sitk.GetImageFromArray(result)
    img_out_itk = copy_geometry(img_out_itk, img_in)
    sitk.WriteImage(img_out_itk, output_file)

def main(args):
    model_path=args.model_path
    image_path = args.image_path
    save_path=args.save_dir
    lower_path=args.lower_path
    model_name = args.model_name
    device=args.device
    if model_name=="3d_cascade_fullres":
        assert len(lower_path)>0
    imgs_list = get_image_list(
        image_path)  # get image list from image path

    if model_name=="3d_cascade_fullres":
        presegpredictor=SegPredictor(lower_path,device,stage=0)
        segpredictor = SegPredictor(model_path,device,stage=1)
        for img in imgs_list:
            f_name=os.path.basename(img)
            save_name=os.path.join(save_path,f_name)
            pre_array=presegpredictor.get_result(img)
            result=segpredictor.get_result(img,True,pre_array)
            print(result.shape)
            save_output(save_name,result,img)

    else:
        segpredictor = SegPredictor(args.model_path, device, stage=0)
        for img in imgs_list:
            f_name = os.path.basename(img)
            save_name = os.path.join(save_path, f_name)
            result=segpredictor.get_result(img,True)
            print(result.shape)
            save_output(save_name, result, img)







if __name__ == '__main__':
    args = parse_args()
    main(args)
