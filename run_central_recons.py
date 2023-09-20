import pathlib
import multiprocessing
from functools import partial

import numpy as np
import tqdm
import h5py

from src.pty_base import fft2, ifft2, stft2, spectral_convergence
from src.pty_algs import run_psi_alg, parse_alg
from src.pty_data import get_diffraction_patterns, get_scan, get_norm_probe
from src.pty_data import get_object_and_probe


ftype = np.float32
ctype = np.complex64


def get_central_recon(data_idx, scandens, Kcent, Jcent, lamb, algdesc, probe_seed, A_seed, ftype=ftype, ctype=ctype):
    O, P = get_object_and_probe(data_idx, small=True, normprobe=True, ctype=ctype)
    rk = get_scan(scandens, O, P)
    stft_gt = stft2(O, rk, P)
    Ak_gt = np.abs(stft_gt)

    np.random.seed(A_seed)
    Ak = get_diffraction_patterns(Ak_gt, lamb=lamb, Ifac=1).astype(ftype)
    rcent = rk[:Kcent]
    Acent = Ak[:Kcent]
    Psicent0 = ifft2(Acent + 0j).astype(ctype)

    np.random.seed(probe_seed)
    P0_rand = get_norm_probe(np.random.randn(*P.shape) + 1j*np.random.randn(*P.shape), Acent).astype(ctype)
    alg = parse_alg(algdesc, Acent)
    final_state = run_psi_alg(
        alg, Jcent, Psicent0, rcent, P0_rand, Acent,
        tqdm=False, track_states=False, probe_update=True,
        Oshape=O.shape
    )
    Pcent = alg.get_probe(final_state)
    Ocent = alg.get_object(final_state)
    Psicent = alg.get_psi(final_state)
    sc = spectral_convergence(fft2(Psicent), Acent, rcent, Pcent, O.shape, eps_pC=1e-12)
    return Pcent, Ocent, Psicent, sc


def get_best_central_of(tries, *args, A_seed, base_probe_seed=None, **kwargs):
    """Uses nonintrusive SC metric to get best of several tries (minimum SC)"""
    results = []

    for t in range(tries):
        kwargs.update({'probe_seed': base_probe_seed+t if base_probe_seed is not None else None})
        results.append(get_central_recon(*args, A_seed=A_seed, **kwargs))

    best_idx = np.argmin([result[3] for result in results])
    return results[best_idx]


def get_best(data_idx, lamb, algdesc, scandens=10, base_probe_seed=591):
    A_seed = 675820+data_idx  # as in run_recon.py and run_recon_classical.py -- makes results reproducible
    best = get_best_central_of(
        10, data_idx, scandens=scandens, Kcent=20, Jcent=250, lamb=lamb,
        algdesc=algdesc, A_seed=A_seed, base_probe_seed=base_probe_seed*(1+data_idx)
    )
    return data_idx, best
    

if __name__ == '__main__':
    ALGDESC_DEFAULT = 'Fhybrid_200Xdm1,M100,0.9+50Xer1e-12,1'

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('outname', type=str)
    parser.add_argument('lamb', type=float)
    parser.add_argument('--nproc', type=int, default=15)
    parser.add_argument('--algdesc', type=str, default=ALGDESC_DEFAULT, help=f'Default: {ALGDESC_DEFAULT}')
    parser.add_argument('--scandens', type=int, default=10, choices=[10, 15, 20, 30])
    parser.add_argument('--base-probe-seed', type=int, default=591)
    args = parser.parse_args()

    basefolder = pathlib.Path('data/central_recons/')
    basefolder.mkdir(parents=True, exist_ok=True)

    with multiprocessing.Pool(processes=args.nproc) as pool:
        mapper = partial(
            get_best,
            lamb=args.lamb,
            algdesc=args.algdesc,
            scandens=args.scandens,
            base_probe_seed=args.base_probe_seed,
        )

        n_objects = 30  # assumes 'small' subset
        res = pool.imap_unordered(mapper, range(n_objects))
        for data_idx, (Pcent, Ocent, Psicent, sc) in tqdm.tqdm(res, total=n_objects):
            with h5py.File(basefolder / f'{args.outname}.h5', 'a') as f:
                f.create_dataset(f'{data_idx}_O_central', data=Ocent)
                f.create_dataset(f'{data_idx}_P_central', data=Pcent)
                f.create_dataset(f'{data_idx}_Psi_central', data=Psicent)
                f.create_dataset(f'{data_idx}_sc_central', data=sc)
