import os
import subprocess
import random, string
import pathlib
import warnings
from typing import Union, List, Optional

import numpy as np
import h5py

from .pty_base import ifft2, update_object, update_probe, stft2, istft2, get_naive_phase_init
from .pty_data import get_object_and_probe, get_scan, get_diffraction_patterns


def get_git_revision_hash():
    try:
        # from https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=os.path.dirname(os.path.realpath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode('ascii').strip()
    except subprocess.CalledProcessError as e:
        warnings.warn(f'Could not get git revision hash: {e}')
        return '[unavailable]'


def get_random_runname(length):
    x = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(length))
    return x


def save_results(folder: pathlib.Path, data_idx: int, O_final, P_final, Psi_final, O_gt, P_gt, Ak, rk, stft_gt):
    """
    Saves results to HDF5 file `folder/<data_idx>.h5` with the object/probe index `data_idx`.
      All parameters besides `file` and `data_idx` are numpy arrays.

    Appends to the file if it already exists. Note that this will error when results
      already exist for the given object index `data_idx`, ensuring we don't overwrite old results.

    Will use Ak, rk and stft_gt to compute in a sense 'optimal' estimates for Psi, O and P,
      given the amplitudes Ak which may be affected by Poisson noise, by combining them with
      ground-truth phases from stft_gt.
    """
    Psi_gt = ifft2(stft_gt)
    stft_poisson = Ak * np.exp(1j * np.angle(stft_gt))
    Psi_poisson = ifft2(stft_poisson)
    O_poisson, _ = update_object(Psi_poisson, rk, P_gt, O_gt.shape, eps=1e-12)
    P_poisson, _ = update_probe(Psi_poisson, rk, O_poisson, P_gt.shape, eps=1e-12)

    dskw = dict(compression='lzf')
    folder = pathlib.Path(folder)
    file = folder / f'{data_idx}.h5'
    with h5py.File(file, 'a') as f:
        f.create_dataset('O_final', data=O_final, **dskw)
        f.create_dataset('P_final', data=P_final, **dskw)
        f.create_dataset('Psi_final', data=Psi_final, **dskw)
        f.create_dataset('O_gt', data=O_gt, **dskw)
        f.create_dataset('P_gt', data=P_gt, **dskw)
        f.create_dataset('Psi_gt', data=Psi_gt, **dskw)
        f.create_dataset('O_poisson', data=O_poisson, **dskw)
        f.create_dataset('P_poisson', data=P_poisson, **dskw)
        f.create_dataset('Psi_poisson', data=Psi_poisson, **dskw)


def get_result(folder_or_file: pathlib.Path, data_idx: Optional[int], key_or_keys: Union[str, List[str]]):
    """
    Loads results from HDF5 file for the key(s) `key_or_keys`.
    When `data_idx` is not None, loads from `folder_or_file/<data_idx.h5>` for the object index `data_idx`.
    When `data_idx` is None, loads from `folder_or_file` directly.
    
    Returns a single value or a list of values, depending on whether `key_or_keys` is a single key or a list of keys.
    """
    folder_or_file = pathlib.Path(folder_or_file)
    h5file = folder_or_file / f'{data_idx}.h5' if data_idx is not None else folder_or_file
    
    with h5py.File(h5file, 'r') as f:
        if isinstance(key_or_keys, list):
            return [f[key][:] for key in key_or_keys]
        else:
            return f[key_or_keys][:]


def get_data(data_idx, scandens, lamb, Ifac, naive_phase_init, ctype, ftype):
    # Initialize / read object and probe
    O, P = get_object_and_probe(data_idx, small=True, normprobe=True, ctype=ctype)

    # Initialize ground-truth STFT, from-STFT inverted object, and (noiseless) diffraction patterns
    rk = get_scan(scandens, O, P)
    stft_gt = stft2(O, rk, P)
    O_gt = istft2(stft_gt, rk, P, O.shape, eps=1e-12)
    Ak_gt = np.abs(stft_gt)

    # Simulate Poisson noise statistics and re-scale diffraction patterns. Use same random seed as default for
    # central-region reconstruction script, so we get consistent diffraction patterns
    np.random.seed(675820+data_idx)
    Ak = get_diffraction_patterns(Ak_gt, lamb=lamb, Ifac=Ifac).astype(ftype)

    # Get initial exit wave guess from the final diffraction patterns Ak
    Psik0 = ifft2(get_naive_phase_init(Ak, naive_phase_init)).astype(ctype)

    return Ak, rk, Psik0, O_gt, P, stft_gt
