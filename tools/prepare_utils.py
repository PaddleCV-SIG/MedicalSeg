import os
import re


def list_files(path, filter_suffix=None):
    """list all the filename in a given path recursively"""

    fname = []
    for root, _, f_names in os.walk(path):
        for f in f_names:
            if filter_suffix is not None:
                if f[-len(filter_suffix):] == filter_suffix:
                    fname.append(os.path.join(root, f))
            else:
                fname.append(os.path.join(root, f))

    return fname
