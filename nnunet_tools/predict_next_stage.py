import os
import numpy as np
from copy import deepcopy
from typing import Tuple, List
import paddle
import sys

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from multiprocessing import Pool
from paddleseg3d.datasets.preprocess_utils.preprocessing import resample_data_or_seg


def predict_preprocessed_data_return_seg_and_softmax(model, loader, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if pad_border_mode == 'constant' and pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        if do_mirroring and mirror_axes is None:
            mirror_axes = loader.data_aug_params['mirror_axes']

        if do_mirroring:
            assert loader.data_aug_params["do_mirror"], "Cannot do mirroring as test time augmentation when training " \
                                                      "was done without mirroring"

        # valid = list((SegmentationNetwork, nn.DataParallel))
        # assert isinstance(self.network, tuple(valid))

        current_mode = model.training
        model.eval()
        with paddle.no_grad():
            argmax_pred, softmax_pred = model.network.predict_3D(data, do_mirroring=do_mirroring, mirror_axes=mirror_axes,
                                        use_sliding_window=use_sliding_window, step_size=step_size,
                                        patch_size=loader.patch_size, regions_class_order=None,
                                        use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                                        pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose,
                                        mixed_precision=mixed_precision)
        model.training = current_mode
        return argmax_pred, softmax_pred


def resample_and_save(predicted, target_shape, output_file, force_separate_z=False,
                      interpolation_order=1, interpolation_order_z=0):
    if isinstance(predicted, str):
        assert os.path.isfile(predicted), "If isinstance(segmentation_softmax, str) then " \
                                  "isfile(segmentation_softmax) must be True"
        del_file = deepcopy(predicted)
        predicted = np.load(predicted)
        os.remove(del_file)

    predicted_new_shape = resample_data_or_seg(predicted, target_shape, False, order=interpolation_order,
                                               do_separate_z=force_separate_z, order_z=interpolation_order_z)
    seg_new_shape = predicted_new_shape.argmax(0)
    np.savez_compressed(output_file, data=seg_new_shape.astype(np.uint8))


def predict_next_stage(model, plans, loader, output_folder, stage_to_be_predicted_folder, mixed_precision):
    output_folder = os.path.join(output_folder, "pred_next_stage")
    os.makedirs(output_folder, exist_ok=True)

    if 'segmentation_export_params' in plans.keys():
        force_separate_z = plans['segmentation_export_params']['force_separate_z']
        interpolation_order = plans['segmentation_export_params']['interpolation_order']
        interpolation_order_z = plans['segmentation_export_params']['interpolation_order_z']
    else:
        force_separate_z = None
        interpolation_order = 1
        interpolation_order_z = 0

    export_pool = Pool(2)
    results = []

    for pat in loader.dataset_val.keys():
        print(pat, 'predict next stage...')
        data_file = loader.dataset_val[pat]['data_file']
        data_preprocessed = np.load(data_file)['data'][:-1]
        data_file_nofolder = data_file.split("/")[-1]
        data_file_nextstage = os.path.join(stage_to_be_predicted_folder, data_file_nofolder)
        data_nextstage = np.load(data_file_nextstage)['data']
        target_shp = data_nextstage.shape[1:]
        output_file = os.path.join(output_folder, data_file_nextstage.split("/")[-1][:-4] + "_segFromPrevStage.npz")
        if os.path.exists(output_file):
            print("{} already exists, skip.".format(output_file))
            continue

        predicted_probabilities = predict_preprocessed_data_return_seg_and_softmax(model, loader,
            data_preprocessed, do_mirroring=loader.data_aug_params["do_mirror"],
            mirror_axes=loader.data_aug_params['mirror_axes'], mixed_precision=mixed_precision)[1]

        if np.prod(predicted_probabilities.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save
            np.save(output_file[:-4] + ".npy", predicted_probabilities)
            predicted_probabilities = output_file[:-4] + ".npy"

        results.append(export_pool.starmap_async(resample_and_save, [(predicted_probabilities, target_shp, output_file,
                                                                      force_separate_z, interpolation_order,
                                                                      interpolation_order_z)]))

    _ = [i.get() for i in results]
    export_pool.close()
    export_pool.join()

