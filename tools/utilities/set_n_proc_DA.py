
import subprocess
import os


def get_allowed_n_proc_DA():
    hostname = subprocess.getoutput(['hostname'])

    if 'nnUNet_n_proc_DA' in os.environ.keys():
        return int(os.environ['nnUNet_n_proc_DA'])

    if hostname in ['hdf19-gpu16', 'hdf19-gpu17', 'e230-AMDworkstation']:
        return 16

    if hostname in ['Fabian',]:
        return 12

    if hostname.startswith('hdf19-gpu') or hostname.startswith('e071-gpu'):
        return 12
    elif hostname.startswith('e230-dgx1'):
        return 10
    elif hostname.startswith('hdf18-gpu') or hostname.startswith('e132-comp'):
        return 16
    elif hostname.startswith('e230-dgx2'):
        return 6
    elif hostname.startswith('e230-dgxa100-'):
        return 32
    else:
        return None