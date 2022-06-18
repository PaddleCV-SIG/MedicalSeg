import argparse
import os
import sys


parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from typing import Tuple, List, Union
import shutil
from nnunet_tools.evaluator import aggregate_scores
from nnunet_tools.postprocessing import determine_postprocessing, subfiles, join, load_json


def collect_cv_niftis(cv_folder: str, output_folder: str, validation_folder_name: str = 'validation_raw',
                      folds: tuple = (0, 1, 2, 3, 4)):
    validation_raw_folders = [os.path.join(cv_folder, "fold_%d" % i, validation_folder_name) for i in folds]
    exist = [os.path.isdir(i) for i in validation_raw_folders]

    if not all(exist):
        raise RuntimeError("some folds are missing. Please run the full 5-fold cross-validation. "
                           "The following folds seem to be missing: %s" %
                           [i for j, i in enumerate(folds) if not exist[j]])

    # now copy all raw niftis into cv_niftis_raw
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    for f in folds:
        # niftis = subfiles(validation_raw_folders[f], suffix=".nii.gz")
        folder = validation_raw_folders[f]
        niftis = [os.path.join(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i)) and i.endswith(".nii.gz")]
        for n in niftis:
            shutil.copy(n, output_folder)


def consolidate_folds(output_folder_base, validation_folder_name: str = 'validation_raw',
                      advanced_postprocessing: bool = False, folds: Tuple[int] = (0, 1, 2, 3, 4)):
    """
    Used to determine the postprocessing for an experiment after all five folds have been completed. In the validation of
    each fold, the postprocessing can only be determined on the cases within that fold. This can result in different
    postprocessing decisions for different folds. In the end, we can only decide for one postprocessing per experiment,
    so we have to rerun it
    :param folds:
    :param advanced_postprocessing:
    :param output_folder_base:experiment output folder (fold_0, fold_1, etc must be subfolders of the given folder)
    :param validation_folder_name: dont use this
    :return:
    """
    output_folder_raw = os.path.join(output_folder_base, "cv_niftis_raw")
    if os.path.isdir(output_folder_raw):
        shutil.rmtree(output_folder_raw)

    output_folder_gt = os.path.join(output_folder_base, "gt_niftis")
    collect_cv_niftis(output_folder_base, output_folder_raw, validation_folder_name,
                      folds)

    num_niftis_gt = len(subfiles(join(output_folder_base, "gt_niftis"), suffix='.nii.gz'))
    # count niftis in there
    num_niftis = len(subfiles(output_folder_raw, suffix='.nii.gz'))
    if num_niftis != num_niftis_gt:
        raise AssertionError("If does not seem like you trained all the folds! Train all folds first!")

    # load a summary file so that we can know what class labels to expect
    summary_fold0 = load_json(join(output_folder_base, "fold_0", validation_folder_name, "summary.json"))['results'][
        'mean']
    classes = [int(i) for i in summary_fold0.keys()]
    niftis = subfiles(output_folder_raw, join=False, suffix=".nii.gz")
    test_pred_pairs = [(join(output_folder_raw, i), join(output_folder_gt, i)) for i in niftis]

    # determine_postprocessing needs a summary.json file in the folder where the raw predictions are. We could compute
    # that from the summary files of the five folds but I am feeling lazy today
    aggregate_scores(test_pred_pairs, labels=classes, json_output_file=join(output_folder_raw, "summary.json"),
                     num_threads=4)

    determine_postprocessing(output_folder_base, output_folder_gt, 'cv_niftis_raw',
                             final_subf_name="cv_niftis_postprocessed", processes=4,
                             advanced_postprocessing=advanced_postprocessing)


def parse_args():
    parser = argparse.ArgumentParser(description='Model evaluation')

    parser.add_argument(
        '--gt_dir',
        dest='gt_dir',
        help='The path to gt result',
        type=str,
        default=None)
    parser.add_argument(
        '--val_dir',
        dest='val_dir',
        help='The path to val data result',
        type=str,
        default=None)
    return parser.parse_args()


def main(args):
    output_folder = args.val_dir
    gt_dir = args.gt_dir
    folds = [i for i in range(5)]

    postprocessing_json = os.path.join(output_folder, "postprocessing.json")
    cv_niftis_folder = join(output_folder, "cv_niftis_raw")
    
    if not os.path.isfile(postprocessing_json) or not os.path.isdir(cv_niftis_folder):
        print("running missing postprocessing." )
        consolidate_folds(output_folder, folds=folds)
        assert os.path.isfile(postprocessing_json), "Postprocessing json missing, expected: %s" % postprocessing_json
        assert os.path.isdir(cv_niftis_folder), "Folder with niftis from CV missing, expected: %s" % cv_niftis_folder




if __name__ == '__main__':
    args = parse_args()
    main(args)
