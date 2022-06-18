import argparse
from copy import deepcopy
from typing import Tuple, Union, List
import os
import sys
import numpy as np
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)
from multiprocessing import Process, Queue
import paddle
import SimpleITK as sitk
import shutil
from multiprocessing import Pool
import paddle.nn.functional as F
from paddleseg3d.models import NNUNet
from paddleseg3d.datasets.preprocess_utils import GenericPreprocessor, PreprocessorFor2D
from paddleseg3d.transforms import default_2D_augmentation_params, default_3D_augmentation_params
from paddleseg3d.datasets.preprocess_utils import resample_data_or_seg, get_do_separate_z, get_lowres_axis, resize_segmentation
from nnunet_tools.postprocessing import load_postprocessing, load_remove_save


class Predictor:
    def __init__(self, plan_path, stage=0):
        self.model = NNUNet(plans_path=plan_path, stage=stage)
        self.model.network.inference_apply_nonlin = lambda x: F.softmax(x, 1)
        self.plans = self.model.plans
        self.num_classes = self.model.num_classes
        self.threeD = self.model.threeD
        self.stage = stage

        self.patch_size = self.model.patch_size
        self.regions_class_order = None

        # 预测时用到一些增强的参数
        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
        else:
            self.data_aug_params = default_2D_augmentation_params

    def load_checkpoint(self, param_path):
        print("======load checkpoint from {}=======".format(param_path))
        params = paddle.load(param_path)
        self.model.set_state_dict(params)

    def load_params(self, params):
        self.model.set_state_dict(params)

    def load_checkpoint_ram(self, params, no_use=False):
        self.model.set_state_dict(params)

    def preprocess_patient(self, input_files):
        if self.threeD:
            preprocessor_class = GenericPreprocessor
        else:
            preprocessor_class = PreprocessorFor2D

        preprocessor = preprocessor_class(self.model.normalization_schemes, self.model.use_mask_for_norm,
                                          self.model.transpose_forward, self.model.intensity_properties)
        d, s, properties = preprocessor.preprocess_test_case(input_files,
                                                             self.plans['plans_per_stage'][self.stage][
                                                                 'current_spacing'])
        return d, s, properties

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        if pad_border_mode == 'constant' and pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        if do_mirroring and mirror_axes is None:
            mirror_axes = self.data_aug_params['mirror_axes']

        if do_mirroring:
            assert self.data_aug_params["do_mirror"], "Cannot do mirroring as test time augmentation when training " \
                                                      "was done without mirroring"

        self.model.eval()
        with paddle.no_grad():
            ret = self.model.network.predict_3D(data, do_mirroring=do_mirroring, mirror_axes=mirror_axes,
                                        use_sliding_window=use_sliding_window, step_size=step_size,
                                        patch_size=self.patch_size, regions_class_order=self.regions_class_order,
                                        use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                                        pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose,
                                        mixed_precision=mixed_precision)
        return ret
    

def to_one_hot(seg, all_seg_labels=None):
    if all_seg_labels is None:
        all_seg_labels = np.unique(seg)
    result = np.zeros((len(all_seg_labels), *seg.shape), dtype=seg.dtype)
    for i, l in enumerate(all_seg_labels):
        result[i][seg == l] = 1
    return result


def preprocess_save_to_queue(preprocess_fn, q, list_of_lists, output_files, segs_from_prev_stage, classes,
                             transpose_forward):
    # suppress output
    # sys.stdout = open(os.devnull, 'w')

    errors_in = []
    for i, l in enumerate(list_of_lists):
        try:
            output_file = output_files[i]
            print("preprocessing", output_file)
            d, _, dct = preprocess_fn(l)
            # print(output_file, dct)
            if segs_from_prev_stage[i] is not None:
                assert os.path.isfile(segs_from_prev_stage[i]) and segs_from_prev_stage[i].endswith(
                    ".nii.gz"), "segs_from_prev_stage" \
                                " must point to a " \
                                "segmentation file"
                seg_prev = sitk.GetArrayFromImage(sitk.ReadImage(segs_from_prev_stage[i]))
                # check to see if shapes match
                img = sitk.GetArrayFromImage(sitk.ReadImage(l[0]))
                assert all([i == j for i, j in zip(seg_prev.shape, img.shape)]), "image and segmentation from previous " \
                                                                                 "stage don't have the same pixel array " \
                                                                                 "shape! image: %s, seg_prev: %s" % \
                                                                                 (l[0], segs_from_prev_stage[i])
                seg_prev = seg_prev.transpose(transpose_forward)
                seg_reshaped = resize_segmentation(seg_prev, d.shape[1:], order=1)
                seg_reshaped = to_one_hot(seg_reshaped, classes)
                d = np.vstack((d, seg_reshaped)).astype(np.float32)
            """There is a problem with python process communication that prevents us from communicating objects 
            larger than 2 GB between processes (basically when the length of the pickle string that will be sent is 
            communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long 
            enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually 
            patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will 
            then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either 
            filename or np.ndarray and will handle this automatically"""
            print(d.shape)
            if np.prod(d.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save, 4 because float32 is 4 bytes
                print(
                    "This output is too large for python process-process communication. "
                    "Saving output temporarily to disk")
                np.save(output_file[:-7] + ".npy", d)
                d = output_file[:-7] + ".npy"
            q.put((output_file, (d, dct)))
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            print("error in", l)
            print(e)
    q.put("end")
    if len(errors_in) > 0:
        print("There were some errors in the following cases:", errors_in)
        print("These cases were ignored.")
    else:
        print("This worker has ended successfully, no errors to report")
    # restore output
    # sys.stdout = sys.__stdout__


def load_model_and_checkpoint_files(folder, folds=None, mixed_precision=None, checkpoint_name="model_best", plan_path=None, stage=0):
    """
    used for if you need to ensemble the five models of a cross-validation. This will restore the model from the
    checkpoint in fold 0, load all parameters of the five folds in ram and return both. This will allow for fast
    switching between parameters (as opposed to loading them from disk each time).
    This is best used for inference and test prediction
    :param folder:
    :param folds:
    :param mixed_precision: if None then we take no action. If True/False we overwrite what the model has in its init
    :return:
    """
    if isinstance(folds, str):
        folds = [os.path.join(folder, "all")]
        assert os.path.isdir(folds[0]), "no output folder for fold %s found" % folds
    elif isinstance(folds, (list, tuple)):
        if len(folds) == 1 and folds[0] == "all":
            folds = [os.path.join(folder, "all")]
        else:
            folds = [os.path.join(folder, "fold_%d" % i) for i in folds]
        assert all([os.path.isdir(i) for i in folds]), "list of folds specified but not all output folders are present"
    elif isinstance(folds, int):
        folds = [os.path.join(folder, "fold_%d" % folds)]
        assert all([os.path.isdir(i) for i in folds]), "output folder missing for fold %d" % folds
    elif folds is None:
        print("folds is None so we will automatically look for output folders (not using \'all\'!)")
        folds = subfolders(folder, prefix="fold")
        print("found the following folds: ", folds)
    else:
        raise ValueError("Unknown value for folds. Type: %s. Expected: list of int, int, str or None", str(type(folds)))

    # trainer = restore_model(join(folds[0], "%s.model.pkl" % checkpoint_name), fp16=mixed_precision)
    # trainer.output_folder = folder
    # trainer.output_folder_base = folder
    # trainer.update_fold(0)
    # trainer.initialize(False)
    predictor = Predictor(plan_path=plan_path, stage=stage)
    print('construct model, plan path: {}, stage: {}.'.format(plan_path, stage))
    all_best_model_files = [os.path.join(i, "model.pdparams") for i in folds]
    print("using the following model files: ", all_best_model_files)
    all_params = [paddle.load(i) for i in all_best_model_files]
    predictor.load_params(all_params[0])
    return predictor, all_params



def preprocess_multithreaded(trainer, list_of_lists, output_files, num_processes=2, segs_from_prev_stage=None):
    if segs_from_prev_stage is None:
        segs_from_prev_stage = [None] * len(list_of_lists)

    num_processes = min(len(list_of_lists), num_processes)

    classes = list(range(1, trainer.num_classes))
    assert isinstance(trainer, Predictor)
    q = Queue(1)
    processes = []
    for i in range(num_processes):
        pr = Process(target=preprocess_save_to_queue, args=(trainer.preprocess_patient, q,
                                                            list_of_lists[i::num_processes],
                                                            output_files[i::num_processes],
                                                            segs_from_prev_stage[i::num_processes],
                                                            classes, trainer.plans['transpose_forward']))
        pr.start()
        processes.append(pr)

    try:
        end_ctr = 0
        while end_ctr != num_processes:
            item = q.get()
            if item == "end":
                end_ctr += 1
                continue
            else:
                yield item

    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()  # this should not happen but better safe than sorry right
            p.join()

        q.close()


def predict_cases(model, list_of_lists, output_filenames, folds, save_npz, num_threads_preprocessing,
                  num_threads_nifti_save, segs_from_prev_stage=None, do_tta=True, mixed_precision=True,
                  overwrite_existing=False,
                  all_in_gpu=False, step_size=0.5, checkpoint_name="model_final_checkpoint",
                  segmentation_export_kwargs: dict = None, disable_postprocessing: bool = False, plan_path=None, stage=0, postprocessing_json_path=None):
    """
    :param segmentation_export_kwargs:
    :param model: folder where the model is saved, must contain fold_x subfolders
    :param list_of_lists: [[case0_0000.nii.gz, case0_0001.nii.gz], [case1_0000.nii.gz, case1_0001.nii.gz], ...]
    :param output_filenames: [output_file_case0.nii.gz, output_file_case1.nii.gz, ...]
    :param folds: default: (0, 1, 2, 3, 4) (but can also be 'all' or a subset of the five folds, for example use (0, )
    for using only fold_0
    :param save_npz: default: False
    :param num_threads_preprocessing:
    :param num_threads_nifti_save:
    :param segs_from_prev_stage:
    :param do_tta: default: True, can be set to False for a 8x speedup at the cost of a reduced segmentation quality
    :param overwrite_existing: default: True
    :param mixed_precision: if None then we take no action. If True/False we overwrite what the model has in its init
    :return:
    """
    assert len(list_of_lists) == len(output_filenames)
    if segs_from_prev_stage is not None: assert len(segs_from_prev_stage) == len(output_filenames)

    pool = Pool(num_threads_nifti_save)
    results = []

    cleaned_output_files = []
    for o in output_filenames:
        dr, f = os.path.split(o)
        if len(dr) > 0:
            # maybe_mkdir_p(dr)
            os.makedirs(dr, exist_ok=True)
        if not f.endswith(".nii.gz"):
            f, _ = os.path.splitext(f)
            f = f + ".nii.gz"
        cleaned_output_files.append(os.path.join(dr, f))

    if not overwrite_existing:
        print("number of cases:", len(list_of_lists))
        # if save_npz=True then we should also check for missing npz files
        not_done_idx = [i for i, j in enumerate(cleaned_output_files) if (not os.path.isfile(j)) or (save_npz and not os.path.isfile(j[:-7] + '.npz'))]

        cleaned_output_files = [cleaned_output_files[i] for i in not_done_idx]
        list_of_lists = [list_of_lists[i] for i in not_done_idx]
        if segs_from_prev_stage is not None:
            segs_from_prev_stage = [segs_from_prev_stage[i] for i in not_done_idx]

        print("number of cases that still need to be predicted:", len(cleaned_output_files))

    print("emptying cuda cache")
    paddle.device.cuda.empty_cache()

    print("loading parameters for folds,", folds)
    trainer, params = load_model_and_checkpoint_files(model, folds, mixed_precision=mixed_precision,
                                                      checkpoint_name=checkpoint_name, plan_path=plan_path, stage=stage)

    if segmentation_export_kwargs is None:
        if 'segmentation_export_params' in trainer.plans.keys():
            force_separate_z = trainer.plans['segmentation_export_params']['force_separate_z']
            interpolation_order = trainer.plans['segmentation_export_params']['interpolation_order']
            interpolation_order_z = trainer.plans['segmentation_export_params']['interpolation_order_z']
        else:
            force_separate_z = None
            interpolation_order = 1
            interpolation_order_z = 0
    else:
        force_separate_z = segmentation_export_kwargs['force_separate_z']
        interpolation_order = segmentation_export_kwargs['interpolation_order']
        interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

    print("starting preprocessing generator")
    print("file list: ", list_of_lists)
    print("preprocess output files: ", cleaned_output_files)
    preprocessing = preprocess_multithreaded(trainer, list_of_lists, cleaned_output_files, num_threads_preprocessing,
                                             segs_from_prev_stage)
    print("starting prediction...")
    all_output_files = []
    for preprocessed in preprocessing:
        output_filename, (d, dct) = preprocessed
        all_output_files.append(all_output_files)
        if isinstance(d, str):
            data = np.load(d)
            os.remove(d)
            d = data

        print("predicting", output_filename)
        trainer.load_checkpoint_ram(params[0], False)
        softmax = trainer.predict_preprocessed_data_return_seg_and_softmax(
            d, do_mirroring=do_tta, mirror_axes=trainer.data_aug_params['mirror_axes'], use_sliding_window=True,
            step_size=step_size, use_gaussian=True, all_in_gpu=all_in_gpu,
            mixed_precision=mixed_precision)[1]

        for p in params[1:]:
            trainer.load_checkpoint_ram(p, False)
            softmax += trainer.predict_preprocessed_data_return_seg_and_softmax(
                d, do_mirroring=do_tta, mirror_axes=trainer.data_aug_params['mirror_axes'], use_sliding_window=True,
                step_size=step_size, use_gaussian=True, all_in_gpu=all_in_gpu,
                mixed_precision=mixed_precision)[1]

        if len(params) > 1:
            softmax /= len(params)

        transpose_forward = trainer.plans.get('transpose_forward')
        if transpose_forward is not None:
            transpose_backward = trainer.plans.get('transpose_backward')
            softmax = softmax.transpose([0] + [i + 1 for i in transpose_backward])

        if save_npz:
            npz_file = output_filename[:-7] + ".npz"
        else:
            npz_file = None

        if hasattr(trainer, 'regions_class_order'):
            region_class_order = trainer.regions_class_order
        else:
            region_class_order = None

        """There is a problem with python process communication that prevents us from communicating objects 
        larger than 2 GB between processes (basically when the length of the pickle string that will be sent is 
        communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long 
        enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually 
        patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will 
        then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either 
        filename or np.ndarray and will handle this automatically"""
        bytes_per_voxel = 4
        if all_in_gpu:
            bytes_per_voxel = 2  # if all_in_gpu then the return value is half (float16)
        if np.prod(softmax.shape) > (2e9 / bytes_per_voxel * 0.85):  # * 0.85 just to be save
            print(
                "This output is too large for python process-process communication. Saving output temporarily to disk")
            np.save(output_filename[:-7] + ".npy", softmax)
            softmax = output_filename[:-7] + ".npy"

        results.append(pool.starmap_async(save_segmentation_nifti_from_softmax,
                                          ((softmax, output_filename, dct, interpolation_order, region_class_order,
                                            None, None,
                                            npz_file, None, force_separate_z, interpolation_order_z),)
                                          ))

    print("inference done. Now waiting for the segmentation export to finish...")
    _ = [i.get() for i in results]

    # now apply postprocessing
    # first load the postprocessing properties if they are present. Else raise a well visible warning
    print('post processing=========')
    results = []
    # pp_file = os.path.join(model, "postprocessing.json")
    pp_file = postprocessing_json_path
    if os.path.isfile(pp_file):
        print("postprocessing...")
        shutil.copy(pp_file, os.path.abspath(os.path.dirname(output_filenames[0])))
        # for_which_classes stores for which of the classes everything but the largest connected component needs to be
        # removed
        for_which_classes, min_valid_obj_size = load_postprocessing(pp_file)
        results.append(pool.starmap_async(load_remove_save,
                                            zip(output_filenames, output_filenames,
                                                [for_which_classes] * len(output_filenames),
                                                [min_valid_obj_size] * len(output_filenames))))
        _ = [i.get() for i in results]
    else:
        print("WARNING! Cannot run postprocessing because the postprocessing file is missing. Make sure to run "
                "consolidate_folds in the output folder of the model first!\nThe folder you need to run this in is "
                "%s" % model)

    pool.close()
    pool.join()


def save_segmentation_nifti_from_softmax(segmentation_softmax: Union[str, np.ndarray], out_fname: str,
                                         properties_dict: dict, order: int = 1,
                                         region_class_order: Tuple[Tuple[int]] = None,
                                         seg_postprogess_fn: callable = None, seg_postprocess_args: tuple = None,
                                         resampled_npz_fname: str = None,
                                         non_postprocessed_fname: str = None, force_separate_z: bool = None,
                                         interpolation_order_z: int = 0, verbose: bool = True):
    """
    This is a utility for writing segmentations to nifty and npz. It requires the data to have been preprocessed by
    GenericPreprocessor because it depends on the property dictionary output (dct) to know the geometry of the original
    data. segmentation_softmax does not have to have the same size in pixels as the original data, it will be
    resampled to match that. This is generally useful because the spacings our networks operate on are most of the time
    not the native spacings of the image data.
    If seg_postprogess_fn is not None then seg_postprogess_fnseg_postprogess_fn(segmentation, *seg_postprocess_args)
    will be called before nifty export
    There is a problem with python process communication that prevents us from communicating objects
    larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
    communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long
    enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
    patching system python code.) We circumvent that problem here by saving softmax_pred to a npy file that will
    then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
    filename or np.ndarray for segmentation_softmax and will handle this automatically
    :param segmentation_softmax:
    :param out_fname:
    :param properties_dict:
    :param order:
    :param region_class_order:
    :param seg_postprogess_fn:
    :param seg_postprocess_args:
    :param resampled_npz_fname:
    :param non_postprocessed_fname:
    :param force_separate_z: if None then we dynamically decide how to resample along z, if True/False then always
    /never resample along z separately. Do not touch unless you know what you are doing
    :param interpolation_order_z: if separate z resampling is done then this is the order for resampling in z
    :param verbose:
    :return:
    """
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
        save_pickle(properties_dict, resampled_npz_fname[:-4] + ".pkl")

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
    print('sitk save: ', out_fname)
    sitk.WriteImage(seg_resized_itk, out_fname)
    # print('sitk exist: ', os.path.exists(out_fname), seg_old_size_postprocessed.min(), seg_old_size_postprocessed.max())

    if (non_postprocessed_fname is not None) and (seg_postprogess_fn is not None):
        seg_resized_itk = sitk.GetImageFromArray(seg_old_size.astype(np.uint8))
        seg_resized_itk.SetSpacing(properties_dict['itk_spacing'])
        seg_resized_itk.SetOrigin(properties_dict['itk_origin'])
        seg_resized_itk.SetDirection(properties_dict['itk_direction'])
        sitk.WriteImage(seg_resized_itk, non_postprocessed_fname)
