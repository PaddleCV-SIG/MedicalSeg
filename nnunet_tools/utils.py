import sys
import os
from copy import deepcopy
import pickle
from typing import Union, Tuple

import numpy as np
import SimpleITK as sitk
from paddleseg3d.datasets.preprocess_utils import get_lowres_axis, get_do_separate_z, resample_data_or_seg, resize_segmentation


def save_segmentation_nifti_from_softmax(segmentation_softmax: Union[str, np.ndarray], out_fname: str,
                                         properties_dict: dict, order: int = 1,
                                         region_class_order: Tuple[Tuple[int]] = None,
                                         seg_postprogess_fn: callable = None, seg_postprocess_args: tuple = None,
                                         resampled_npz_fname: str = None,
                                         non_postprocessed_fname: str = None, force_separate_z: bool = None,
                                         interpolation_order_z: int = 0, verbose: bool = True):
    if verbose: print("force_separate_z:", force_separate_z, "interpolation order:", order)

    if isinstance(segmentation_softmax, str):
        assert os.path.isfile(segmentation_softmax), "If isinstance(segmentation_softmax, str) then " \
                                             "isfile(segmentation_softmax) must be True"
        del_file = deepcopy(segmentation_softmax)
        if segmentation_softmax.endswith('.npy'):
            segmentation_softmax = np.load(segmentation_softmax)
        elif segmentation_softmax.endswith('.npz'):
            segmentation_softmax = np.load(segmentation_softmax)['softmax']
        os.remove(del_file)

    # first resample, then put result into bbox of cropping, then save
    current_shape = segmentation_softmax.shape
    shape_original_after_cropping = properties_dict.get('size_after_cropping')
    shape_original_before_cropping = properties_dict.get('original_size_of_raw_data')
    # current_spacing = dct.get('spacing_after_resampling')
    # original_spacing = dct.get('original_spacing')

    if np.any([i != j for i, j in zip(np.array(current_shape[1:]), np.array(shape_original_after_cropping))]):
        if force_separate_z is None:
            if get_do_separate_z(properties_dict.get('original_spacing')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(properties_dict.get('original_spacing'))
            elif get_do_separate_z(properties_dict.get('spacing_after_resampling')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(properties_dict.get('spacing_after_resampling'))
            else:
                do_separate_z = False
                lowres_axis = None
        else:
            do_separate_z = force_separate_z
            if do_separate_z:
                lowres_axis = get_lowres_axis(properties_dict.get('original_spacing'))
            else:
                lowres_axis = None

        if lowres_axis is not None and len(lowres_axis) != 1:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False

        if verbose: print("separate z:", do_separate_z, "lowres axis", lowres_axis)
        seg_old_spacing = resample_data_or_seg(segmentation_softmax, shape_original_after_cropping, is_seg=False,
                                               axis=lowres_axis, order=order, do_separate_z=do_separate_z,
                                               order_z=interpolation_order_z)
        # seg_old_spacing = resize_softmax_output(segmentation_softmax, shape_original_after_cropping, order=order)
    else:
        if verbose: print("no resampling necessary")
        seg_old_spacing = segmentation_softmax

    if resampled_npz_fname is not None:
        np.savez_compressed(resampled_npz_fname, softmax=seg_old_spacing.astype(np.float16))
        # this is needed for ensembling if the nonlinearity is sigmoid
        if region_class_order is not None:
            properties_dict['regions_class_order'] = region_class_order
        with open(resampled_npz_fname[:-4] + ".pkl", 'wb') as f:
            pickle.dump(properties_dict, f)
        # save_pickle(properties_dict, resampled_npz_fname[:-4] + ".pkl")

    if region_class_order is None:
        seg_old_spacing = seg_old_spacing.argmax(0)
    else:
        seg_old_spacing_final = np.zeros(seg_old_spacing.shape[1:])
        for i, c in enumerate(region_class_order):
            seg_old_spacing_final[seg_old_spacing[i] > 0.5] = c
        seg_old_spacing = seg_old_spacing_final

    bbox = properties_dict.get('crop_bbox')

    if bbox is not None:
        seg_old_size = np.zeros(shape_original_before_cropping, dtype=np.uint8)
        for c in range(3):
            bbox[c][1] = np.min((bbox[c][0] + seg_old_spacing.shape[c], shape_original_before_cropping[c]))
        seg_old_size[bbox[0][0]:bbox[0][1],
        bbox[1][0]:bbox[1][1],
        bbox[2][0]:bbox[2][1]] = seg_old_spacing
    else:
        seg_old_size = seg_old_spacing

    if seg_postprogess_fn is not None:
        seg_old_size_postprocessed = seg_postprogess_fn(np.copy(seg_old_size), *seg_postprocess_args)
    else:
        seg_old_size_postprocessed = seg_old_size

    seg_resized_itk = sitk.GetImageFromArray(seg_old_size_postprocessed.astype(np.uint8))
    seg_resized_itk.SetSpacing(properties_dict['itk_spacing'])
    seg_resized_itk.SetOrigin(properties_dict['itk_origin'])
    seg_resized_itk.SetDirection(properties_dict['itk_direction'])
    sitk.WriteImage(seg_resized_itk, out_fname)

    if (non_postprocessed_fname is not None) and (seg_postprogess_fn is not None):
        seg_resized_itk = sitk.GetImageFromArray(seg_old_size.astype(np.uint8))
        seg_resized_itk.SetSpacing(properties_dict['itk_spacing'])
        seg_resized_itk.SetOrigin(properties_dict['itk_origin'])
        seg_resized_itk.SetDirection(properties_dict['itk_direction'])
        sitk.WriteImage(seg_resized_itk, non_postprocessed_fname)

        