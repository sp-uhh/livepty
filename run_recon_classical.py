import argparse
import sys, pathlib
import pickle

import numpy as np

from src.pty_base import ifft2, stft2, istft2, get_naive_phase_init
from src.pty_data import get_scan, get_object_and_probe, get_norm_probe, get_diffraction_patterns
from src.pty_algs import run_psi_alg, parse_alg
from src.run_utils import get_git_revision_hash, get_random_runname, save_results


ftype = np.float32
ctype = np.complex64

parser = argparse.ArgumentParser()
parser.add_argument(
    'obj_probe_idx', type=int,
    help='Index of object and probe (from our simulated dataset) to use. Should be between 0 and 89, or between 0 and 29 when using --smol.')
parser.add_argument(
    'iters', type=int,
    help='Number of total algorithm iterations')
parser.add_argument(
    '--runname', type=str, required=False,
    help='Name of this run. Should be unique, will throw error if reconstructed object already exists. get_random_runname() is called if this is not passed.')
parser.add_argument(
    '--alg', type=str,
    help="Algorithm to run for classical reconstruction. Example: 'Fhybrid_200Xdm1,1e-12+50Xer1e-12'.")
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
    help='Use the ground-truth probe for reconstruction. If passed, probe update is turned off.')
parser.add_argument(
    '--kmax', type=int,
    help="Pass to limit number of total frames processed, for debugging etc.")
parser.add_argument(
    '--full', action='store_true',
    help="Use full dataset (40+40+10 object/probe pairs) instead of 'smol' dataset (10+10+10, default).")
parser.add_argument(
    '--naive-phase-init', type=str, choices=['zero', 'uniform'], default='zero',
    help='Naive phase initialization for frames. Default is zero.')
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
    pickle.dump({'type': 'classical', 'args': args, 'commit': get_git_revision_hash()}, output_file)

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

# Either use ground-truth probe or reconstruct from random complex Gaussian noise.
if args.use_gt_probe:
    P0 = P
    O0 = None
    probe_update = False
else:
    P0_rand = get_norm_probe(np.random.randn(*P.shape) + 1j*np.random.randn(*P.shape), Ak)
    P0 = P0_rand
    O0 = None
    probe_update = True

# Initialize classical algorithm(s)
iters = args.iters
alg = parse_alg(args.alg, Ak)
print(f"Using algorithm: {alg} with {iters} iterations")

# Run reconstruction!
final_state = run_psi_alg(
    alg, iters, Psik0, rk, P0, Ak, Oshape=O.shape,
    probe_update=probe_update, tqdm=True, track_states=False,
)
Of = alg.get_object(final_state)
Pf = alg.get_probe(final_state)
Psif = alg.get_psi(final_state)

print(f"Saving results to {outpath}")
P_gt = P
save_results(outpath, data_idx, Of, Pf, Psif, O_gt, P_gt, Ak, rk, stft_gt)
print("Done!")
