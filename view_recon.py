import pathlib
import argparse

import matplotlib.pyplot as plt
import numpy as np

from src.pty_data import plot_ampl_phase, ccmap_img
from src.run_utils import get_result
from src.pty_base import clip_max_to_quantile_complex, clip_max_to_quantile_float, zero_phase_ramp, get_gamma

def zero_ramps(x, win=None, clipmax=None, cut=None):
    x_clip = clip_max_to_quantile_complex(x, clipmax)
    x_win = x_clip * (win if win is not None else 1)
    _, ramp = zero_phase_ramp(x_win, return_ramp=True, cut=cut)
    return x * ramp

def get_zero_win(O_gt):
    return 1 / (np.maximum(np.abs(O_gt), 1e-4) * np.exp(1j*np.angle(O_gt)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result_file', type=pathlib.Path)
    parser.add_argument('--center-only', action='store_true',
        help='Only show central 300x300 part of object estimate')
    parser.add_argument('--zero-phase-ramp', action='store_true',
        help='Zero out phase ramp in object estimate')
    parser.add_argument('--apply-gamma', action='store_true',
        help='Apply "gamma" correction factor to object estimate (as also used in E0 error metric)')
    parser.add_argument('--complex-plot', action='store_true',
        help='Plot amplitude and phase with lightness/hue instead of in two subplots')
    parser.add_argument('--phase-cmap', type=str, default='hsv',
        help='Colormap to use for phases. Default: "hsv". Also recommended: "twilight"')
    args = parser.parse_args()

    O_est, P_est, O_gt, P_gt = get_result(args.result_file, None, ['O_final', 'P_final', 'O_gt', 'P_gt'])
    phase_cmap = args.phase_cmap

    if args.center_only:
        cut = (slice(106,406), slice(106,406))
        O_est = O_est[cut]
        O_gt = O_gt[cut]
    if args.zero_phase_ramp:
        midy = O_est.shape[0]//2
        midx = O_est.shape[1]//2
        zerokw = dict(win=get_zero_win(O_gt), clipmax=1.0)
        O_est = zero_ramps(O_est, **zerokw)
    if args.apply_gamma:
        O_est = get_gamma(O_gt, O_est) * O_est

    if not args.complex_plot:
        fig, axs = plt.subplots(2, 2, sharex='row', sharey='row', figsize=(8,8))
        imkw = dict(cmaps=('Greys_r', phase_cmap))
        _, _, (imA, imP) = plot_ampl_phase(O_est, axs=axs[0,:], **imkw)
        _, _, (imAw, imPw) = plot_ampl_phase(P_est, axs=axs[1,:], **imkw)
        imA.set_clim(np.min(np.abs(O_est)), np.max(np.abs(O_est)))
        imAw.set_clim(np.min(np.abs(P_est)), np.max(np.abs(P_est)))
        plt.colorbar(imA, ax=axs[0,0])
        plt.colorbar(imAw, ax=axs[1,0])
        plt.colorbar(imP, ax=axs[0,1])
        plt.colorbar(imPw, ax=axs[1,1])
    else:
        # Use ccmap_img
        fig, axs = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(8,4))
        # Clip amplitudes at 0.95th percentile if --center-only is not passed, to avoid bad visualizations due to edge
        amp_tf = (lambda a: a) if args.center_only else (lambda a: clip_max_to_quantile_float(a, 0.95))
        imA = axs[0].imshow(ccmap_img(O_est, cmap=phase_cmap, amp_tf=amp_tf), interpolation='none')
        imAw = axs[1].imshow(ccmap_img(P_est, cmap=phase_cmap), interpolation='none')

    fig.show()
    plt.show()
