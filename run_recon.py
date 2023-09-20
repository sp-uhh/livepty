import argparse
import sys, pathlib
import pickle
import h5py

import numpy as np

from src.pty_base import ifft2, stft2, istft2, get_naive_phase_init,  ObjectProbeUpdateMode
from src.pty_algs import parse_alg, parse_eps
from src.pty_live import run_rtisi_psi, RealTimeProbeUpdate
from src.pty_data import get_scan, get_object_and_probe, get_norm_probe, get_diffraction_patterns
from src.run_utils import get_git_revision_hash, get_random_runname, save_results


ctype = np.complex64
ftype = np.float32

parser = argparse.ArgumentParser()
parser.add_argument(
    'obj_probe_idx', type=int,
    help='Index of object and probe (from our simulated dataset) to use. '
        'Should be between 0 and 29, or between 0 and 89 when using --full.')
parser.add_argument(
    'iters', type=int,
    help='Number of algorithm iterations per scan index')
parser.add_argument(
    'buffersize', type=int,
    help='Number of exit waves in the buffer')
parser.add_argument(
    '--runname', type=str, required=False,
    help='Name of this run. Should be unique, will throw error if results already exist. '
        'get_random_runname() is called if this is not passed.')
parser.add_argument(
    '--alg', type=str,
    help='Algorithm to run for live reconstruction. Check parse_alg function from `algs_psi.py`. '
        'Examples: "dm1,1e-12" or "Fhybrid_8Xdm1,1e-12+2Xer1e-12".')
parser.add_argument(
    '--commit-eps', type=str, required=True,
    help='Epsilon to use for object/probe update after each exit wave commit.')
parser.add_argument(
    '--commit-pq', type=float, required=True,
    help='Probe quantile clip to use for probe update after each exit wave commit. '
        '0.95 is a good default. Pass 1 to turn off.')
parser.add_argument(
    '--phi0-idxs', nargs='*', default=[-1],
    help='Buffer-relative indices to apply Phi0 estimation to. [-1] by default (like in RTISI/RTISI-LA).')
parser.add_argument(
    '--scandens', type=int, choices=[10, 15, 20, 30], default=10,
    help='Archimedes Spiral scan density (in px). 10 by default.')
parser.add_argument(
    '--Ifac', type=float, default=1,
    help='Global factor to apply to simulated intensities. Useful to set epsilons relative to. 1 by default.')
parser.add_argument(
    '--poisson-lambda-max', type=float, default=1e9,
    help='Expected max. intensity for all patterns for Poisson noise simulation. 1e9 by default.')
parser.add_argument(
    '--use-gt-probe', action='store_true',
    help='Use the ground-truth probe for reconstruction. Overrides --central-from-file if passed.')
parser.add_argument(
    '--rt-probe-update', type=RealTimeProbeUpdate.from_string, choices=list(RealTimeProbeUpdate),
    default=str(RealTimeProbeUpdate.EACH_ITERATION),
    help='Mode for real-time probe update. "each_iteration" by default.')
parser.add_argument(
    '--central-from-file', type=str,
    help='Load central reconstruction from HDF5 file.')
parser.add_argument(
    '--kmax', type=int,
    help="Pass to limit number of total exit waves processed, for debugging etc.")
parser.add_argument(
    '--full', action='store_true',
    help="Use full dataset (40+40+10 object/probe pairs) instead of 'smol' dataset (10+10+10, default).")
parser.add_argument(
    '--naive-phase-init', type=str, choices=['zero', 'uniform'], default='zero',
    help='Naive phase initialization for exit waves. Recommended default is zero.')
args = parser.parse_args()

# get index of object&probe pair
data_idx = args.obj_probe_idx

# Initialize output path / check if already exists
runname = args.runname if args.runname is not None else get_random_runname(8)
outpath = pathlib.Path(f'results/{runname}/')
result_file = outpath / f'{data_idx}.h5'
print(f"Using runname: {runname}")
if result_file.exists():
    print(f"runname {runname} already has a written output file for index {data_idx}! Exiting.")
    sys.exit(1)
outpath.mkdir(parents=True, exist_ok=True)

# save meta information (passed args, git commit) for future reference
with open(outpath / f'{data_idx}_meta.pkl', "wb") as output_file:
    pickle.dump({'type': 'rtisi', 'args': args, 'commit': get_git_revision_hash()}, output_file)

# Initialize / read object and probe
O, P = get_object_and_probe(data_idx, small=not args.full, normprobe=True, ctype=ctype)

# Initialize ground-truth STFT, from-STFT inverted object, and (noiseless) diffraction patterns
rk = get_scan(args.scandens, O, P)
stft_gt = stft2(O, rk, P)
O_gt = istft2(stft_gt, rk, P, O.shape, eps=1e-12)
Ak_gt = np.abs(stft_gt)

# Simulate Poisson noise statistics and re-scale diffraction patterns. Use same random seed as default for
# central-region reconstruction script, so we get consistent diffraction patterns
np.random.seed(675820+data_idx)
Ak = get_diffraction_patterns(Ak_gt, lamb=args.poisson_lambda_max, Ifac=args.Ifac).astype(ftype)

# Get initial exit wave guess from the final diffraction patterns Ak
Psik0 = ifft2(get_naive_phase_init(Ak, args.naive_phase_init)).astype(ctype)

# The buffersize determines how much information we have access to in the first step, so get it here already
B = args.buffersize

# O/P/Psi initialization, depending on options
if args.use_gt_probe:
    P0 = P
    O0 = None
    A_alg = Ak[:B]
elif args.central_from_file is not None:
    print(f"Loading central reconstruction from HDF5 file {args.central_from_file}")
    with h5py.File(args.central_from_file, 'r') as f:
        Ocent = f[f'{data_idx}_O_central'][:]
        Pcent = f[f'{data_idx}_P_central'][:]
        Psicent = f[f'{data_idx}_Psi_central'][:]
    P0 = Pcent
    O0 = Ocent
    Psik0[:Psicent.shape[0]] = Psicent[:]
    A_alg = Ak[:Psicent.shape[0]]
else:
    # uninformed initialization
    A_alg = Ak[:B]
    P0_rand = get_norm_probe(np.random.randn(*P.shape) + 1j*np.random.randn(*P.shape), A_alg)
    P0 = P0_rand
    O0 = None

# Initialize real-time algorithm
eps_commit = parse_eps(args.commit_eps, A_alg)
pq_commit = args.commit_pq
alg = parse_alg(args.alg, A_alg)
print(f"Using algorithm: {alg}")

# Run reconstruction!
Of, Pf, Psif, _, _, _ = run_rtisi_psi(
    Ak, rk, O.shape, P0,
    B=B, alg=alg, iters=args.iters,
    phi0_est_idxs=args.phi0_idxs,
    naive_phase_init=args.naive_phase_init,
    probe_update=args.rt_probe_update,
    object_probe_update_mode=ObjectProbeUpdateMode.PROBE_THEN_OBJECT,
    pA_before_commit=True,
    track_interm=None,
    eps_commit=eps_commit,
    pq_commit=pq_commit,
    Psik0=Psik0,
    O0=O0,
    kmax=args.kmax,
)

# Save results
print(f"Saving results to {outpath}")
P_gt = P
save_results(outpath, data_idx, Of, Pf, Psif, O_gt, P_gt, Ak, rk, stft_gt)
print("Done!")
