import os


def list_files(path):
    """list all the filename in a given path recursively"""
    fname = []
    for root, _, f_names in os.walk(path):
        for f in f_names:
            fname.append(os.path.join(root, f))

    return fname
