import argparse
import os
import pickle
import sys
import json
from itertools import combinations
from collections import OrderedDict
import numpy as np

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from typing import Tuple, List, Union
import shutil
from multiprocess import Pool
from nnunet_tools.evaluator import aggregate_scores
from nnunet_tools.postprocessing import determine_postprocessing, subfiles, join, load_json, save_json
from nnunet_tools.utils import save_segmentation_nifti_from_softmax


def merge(args):
    file1, file2, properties_file, out_file = args
    if not os.path.isfile(out_file):
        res1 = np.load(file1)['softmax']
        res2 = np.load(file2)['softmax']
        with open(properties_file, 'rb') as f:
            props = pickle.load(f)
        print("ensemble {} and {}.".format(file1, file2))
        mn = np.mean((res1, res2), 0)
        # Softmax probabilities are already at target spacing so this will not do any resampling (resampling parameters
        # don't matter here)
        save_segmentation_nifti_from_softmax(mn, out_file, props, 3, None, None, None, force_separate_z=None,
                                             interpolation_order_z=0)


def ensemble(training_output_folder1, training_output_folder2, output_folder, plan_path, validation_folder, folds, gt_dir, allow_ensembling: bool = True):
    print("\nEnsembling folders\n", training_output_folder1, "\n", training_output_folder2)

    output_folder_base = output_folder
    output_folder = join(output_folder_base, "ensembled_raw")

    with open(plan_path, 'rb') as f:
        plans = pickle.load(f)

    files1 = []
    files2 = []
    property_files = []
    out_files = []
    gt_segmentations = []

    folder_with_gt_segs = gt_dir
    # in the correct shape and we need the original geometry to restore the niftis

    for f in range(folds):
        validation_folder_net1 = os.path.join(training_output_folder1, "fold_%d" % f, validation_folder)
        validation_folder_net2 = os.path.join(training_output_folder2, "fold_%d" % f, validation_folder)

        if not os.path.isdir(validation_folder_net1):
            raise AssertionError("Validation directory missing: %s. Please rerun validation with `python nnunet_tools/nnunet_fold_val.py --config {config_path} --model_path {model_path} --precision fp16 --save_dir {save_dir} --val_save_folder {val predicted save dir}`" % validation_folder_net1)
        if not os.path.isdir(validation_folder_net2):
            raise AssertionError("Validation directory missing: %s. Please rerun validation with `python nnunet_tools/nnunet_fold_val.py --config {config_path} --model_path {model_path} --precision fp16 --save_dir {save_dir} --val_save_folder {val predicted save dir}`" % validation_folder_net2)

        # we need to ensure the validation was successful. We can verify this via the presence of the summary.json file
        if not os.path.isfile(join(validation_folder_net1, 'summary.json')):
            raise AssertionError("Validation directory incomplete: %s. Please rerun validation with `python nnunet_tools/nnunet_fold_val.py --config {config_path} --model_path {model_path} --precision fp16 --save_dir {save_dir} --val_save_folder {val predicted save dir}`" % validation_folder_net1)
        if not os.path.isfile(join(validation_folder_net2, 'summary.json')):
            raise AssertionError("Validation directory missing: %s. Please rerun validation with `python nnunet_tools/nnunet_fold_val.py --config {config_path} --model_path {model_path} --precision fp16 --save_dir {save_dir} --val_save_folder {val predicted save dir}`" % validation_folder_net2)

        patient_identifiers1_npz = [i[:-4] for i in subfiles(validation_folder_net1, False, None, 'npz', True)]
        patient_identifiers2_npz = [i[:-4] for i in subfiles(validation_folder_net2, False, None, 'npz', True)]

        # we don't do postprocessing anymore so there should not be any of that noPostProcess
        patient_identifiers1_nii = [i[:-7] for i in subfiles(validation_folder_net1, False, None, suffix='nii.gz', sort=True) if not i.endswith("noPostProcess.nii.gz") and not i.endswith('_postprocessed.nii.gz')]
        patient_identifiers2_nii = [i[:-7] for i in subfiles(validation_folder_net2, False, None, suffix='nii.gz', sort=True) if not i.endswith("noPostProcess.nii.gz") and not i.endswith('_postprocessed.nii.gz')]

        if not all([i in patient_identifiers1_npz for i in patient_identifiers1_nii]):
            raise AssertionError("Missing npz files in folder %s. Please run the validation for all models and folds with the '--npz' flag." % (validation_folder_net1))
        if not all([i in patient_identifiers2_npz for i in patient_identifiers2_nii]):
            raise AssertionError("Missing npz files in folder %s. Please run the validation for all models and folds with the '--npz' flag." % (validation_folder_net2))

        patient_identifiers1_npz.sort()
        patient_identifiers2_npz.sort()

        assert all([i == j for i, j in zip(patient_identifiers1_npz, patient_identifiers2_npz)]), "npz filenames do not match. This should not happen."

        os.makedirs(output_folder, exist_ok=True)

        for p in patient_identifiers1_npz:
            files1.append(os.path.join(validation_folder_net1, p + '.npz'))
            files2.append(os.path.join(validation_folder_net2, p + '.npz'))
            property_files.append(os.path.join(validation_folder_net1, p) + ".pkl")
            out_files.append(os.path.join(output_folder, p + ".nii.gz"))
            gt_segmentations.append(os.path.join(folder_with_gt_segs, p + ".nii.gz"))

    p = Pool(4)
    p.map(merge, zip(files1, files2, property_files, out_files))
    p.close()
    p.join()

    if not os.path.isfile(os.path.join(output_folder, "summary.json")) and len(out_files) > 0:
        aggregate_scores(tuple(zip(out_files, gt_segmentations)), labels=plans['all_classes'],
                     json_output_file=os.path.join(output_folder, "summary.json"), num_threads=4)

    print('running postprocessing===============')
    # now lets also look at postprocessing. We cannot just take what we determined in cross-validation and apply it
    # here because things may have changed and may also be too inconsistent between the two networks
    determine_postprocessing(output_folder_base, folder_with_gt_segs, "ensembled_raw", "temp",
                                "ensembled_postprocessed", 4, dice_threshold=0)

    out_dir_all_json = os.path.join(output_folder_base, "summary_jsons")
    json_out = load_json(os.path.join(output_folder_base, "ensembled_postprocessed", "summary.json"))

    json_out["experiment_name"] = output_folder_base.split("/")[-1]
    save_json(json_out, os.path.join(output_folder_base, "ensembled_postprocessed", "summary.json"))

    # maybe_mkdir_p(out_dir_all_json)
    os.makedirs(out_dir_all_json, exist_ok=True)
    shutil.copy(os.path.join(output_folder_base, "ensembled_postprocessed", "summary.json"),
                os.path.join(out_dir_all_json, "%s.json" % (output_folder_base.split("/")[-1])))

def get_mean_foreground_dice(json_file):
    results = load_json(json_file)
    return get_foreground_mean(results)


def get_foreground_mean(results):
    results_mean = results['results']['mean']
    dice_scores = [results_mean[i]['Dice'] for i in results_mean.keys() if i != "0" and i != 'mean']
    return np.mean(dice_scores)


def foreground_mean(filename):
    with open(filename, 'r') as f:
        res = json.load(f)
    class_ids = np.array([int(i) for i in res['results']['mean'].keys() if (i != 'mean')])
    class_ids = class_ids[class_ids != 0]
    class_ids = class_ids[class_ids != -1]
    class_ids = class_ids[class_ids != 99]

    tmp = res['results']['mean'].get('99')
    if tmp is not None:
        _ = res['results']['mean'].pop('99')

    metrics = res['results']['mean']['1'].keys()
    res['results']['mean']["mean"] = OrderedDict()
    for m in metrics:
        foreground_values = [res['results']['mean'][str(i)][m] for i in class_ids]
        res['results']['mean']["mean"][m] = np.nanmean(foreground_values)
    with open(filename, 'w') as f:
        json.dump(res, f, indent=4, sort_keys=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Model evaluation')

    parser.add_argument(
        '--nnunet_2d_val_dir',
        dest='nnunet_2d',
        help='The path to val nnunet 2d predict.',
        type=str,
        default=None)
    parser.add_argument(
        '--nnunet_3d_val_dir',
        dest='nnunet_3d',
        help='The path to val nnunet 2d predict.',
        type=str,
        default=None)
    parser.add_argument(
        '--nnunet_3d_cascade_val_dir',
        dest='nnunet_3d_cascade',
        help='The path to val nnunet 3d cascade predict.',
        type=str,
        default=None)
    parser.add_argument(
        '--ensemble_output',
        dest='ensemble_output',
        help='The path to ensemble output.',
        type=str,
        default='ensemble_output')
    parser.add_argument(
        '--plan_2d_path',
        dest='plan_2d_path',
        help='The path to 2d plan path.',
        type=str,
        default=None)
    parser.add_argument(
        '--folds',
        dest='folds',
        help='The number of folds.',
        type=int,
        default=5)
    parser.add_argument(
        '--gt_dir',
        dest='gt_dir',
        help='The path of ground truth.',
        type=str,
        default=None)
    return parser.parse_args()


def main(args):
    model_output_dir = []
    if args.nnunet_2d is not None:
        model_output_dir.append(args.nnunet_2d)
    if args.nnunet_3d is not None:
        model_output_dir.append(args.nnunet_3d)   
    if args.nnunet_3d_cascade is not None:
        model_output_dir.append(args.nnunet_3d_cascade) 
    assert len(model_output_dir) >= 2, "The number of ensemble models must greater than 2."
    assert args.gt_dir is not None, "The ground truth dir cannot be None."

    validation_folder = "validation_raw"
    folds = args.folds

    results = {}
    all_results = {}
    # 结合输出，做ensemble
    for m1, m2 in combinations(model_output_dir, 2):
        ensemble_name = "ensemble_" + m1.split('/')[-1] + "__" + m2.split('/')[-1] 
        output_folder_base = os.path.join(args.ensemble_output, "ensembles", ensemble_name)
        os.makedirs(output_folder_base, exist_ok=True)

        print("ensembling", m1, m2)
        ensemble(m1, m2, output_folder_base, args.plan_2d_path, validation_folder, folds, args.gt_dir)
        # ensembling will automatically do postprocessingget_foreground_mean

        # now get result of ensemble
        results[ensemble_name] = get_mean_foreground_dice(os.path.join(output_folder_base, "ensembled_raw", "summary.json"))
        summary_file = os.path.join(output_folder_base, "ensembled_raw", "summary.json")
        foreground_mean(summary_file)
        all_results[ensemble_name] = load_json(summary_file)['results']['mean']

    # now print all mean foreground dice and highlight the best
    # foreground_dices = list(results.values())
    # best = np.max(foreground_dices)
    # for k, v in results.items():
    #     print(k, v)

if __name__ == '__main__':
    args = parse_args()
    main(args)
