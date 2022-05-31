


from tools.batchgenerators.utilities.file_and_folder_operations import *


def pretend_to_be_nnUNetTrainer(folder, checkpoints=("model_best.model.pkl", "model_final_checkpoint.model.pkl")):
    pretend_to_be_other_trainer(folder, "nnUNetTrainer", checkpoints)


def pretend_to_be_other_trainer(folder, new_trainer_name, checkpoints=("model_best.model.pkl", "model_final_checkpoint.model.pkl")):
    folds = subdirs(folder, prefix="fold_", join=False)

    if isdir(join(folder, 'all')):
        folds.append('all')

    for c in checkpoints:
        for f in folds:
            checkpoint_file = join(folder, f, c)
            if isfile(checkpoint_file):
                a = load_pickle(checkpoint_file)
                a['name'] = new_trainer_name
                save_pickle(a, checkpoint_file)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Use this script to change the nnunet trainer class of a saved '
                                                 'model. Useful for models that were trained with trainers that do '
                                                 'not support inference (multi GPU trainers) or for trainer classes '
                                                 'whose source code is not available. For this to work the network '
                                                 'architecture must be identical between the original trainer '
                                                 'class and the trainer class we are changing to. This script is '
                                                 'experimental and only to be used by advanced users.')
    parser.add_argument('-i', help='Folder containing the trained model. This folder is the one containing the '
                                   'fold_X subfolders.')
    parser.add_argument('-tr', help='Name of the new trainer class')
    args = parser.parse_args()
    pretend_to_be_other_trainer(args.i, args.tr)
