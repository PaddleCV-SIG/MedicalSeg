import os


def list_files(path, filter_suffix=None, filter_key=None, include=None):
    """
    list all the filename in a given path recursively. if needed filter the names with suffix or key string
    When filter_key is not None，use include key to select only filenames with key in it.

    Args:

    """
    fname = []
    for root, _, f_names in os.walk(path):
        for f in f_names:
            if filter_suffix is not None:
                if f[-len(filter_suffix):] == filter_suffix:
                    fname.append(os.path.join(root, f))
            elif filter_key is not None:
                assert include is not None, print(
                    "if you want to filter with filter key, \"include\" need to be True or False."
                )

                if (filter_key in f and include) or (filter_key not in f
                                                     and not include):
                    fname.append(os.path.join(root, f))
            else:
                fname.append(os.path.join(root, f))

    return fname
