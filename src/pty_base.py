from enum import Enum

import numba
import numpy as np
from scipy.fft import fftfreq as fftfreq_np, fftshift as fftshift_np
from scipy.ndimage import center_of_mass

from .pty_data import get_power
from .util import ValueEqEnum
from .sigproc import fft2, ifft2, c2f


# ========== Algorithmic primitives ===========

def get_naive_phase_init(D, kind):
    if kind == 'zero':
        return D + 0j
    elif kind == 'uniform':
        return D * np.exp(1j * 2*np.pi * np.random.rand(*D.shape))
    elif isinstance(kind, tuple):
        minval, maxval = kind
        return D * np.exp(1j * np.random.uniform(minval, maxval, size=D.shape))


@numba.njit(cache=True)
def segment2(O, p, w):
    wsx, wsy = w.shape
    segments = np.empty((len(p), *w.shape), dtype=O.dtype)
    for i, p_ in enumerate(p):
        px, py = p_
        segments[i] = w * O[px:px+wsx, py:py+wsy]
    return segments


def stft2(O, p, w):
    return fft2(segment2(O, p, w))


@numba.njit(cache=True)
def _clip_max_complex(x, x_abs, vq):
    iunit = x.dtype.type(1j)  # numba hacks to ensure complex input type = output type
    return np.minimum(x_abs, vq) * np.exp(iunit * np.angle(x))

@numba.njit(cache=True)
def _clip_max_float(x, x_abs, vq):
    return np.minimum(x_abs, vq) * np.sign(x)

@numba.njit(cache=True)
def clip_max_to_quantile_float(x, quantile):
    if quantile is None or quantile == 1:
        return x
    x_abs = np.abs(x)
    x_abs_nonzero = x_abs.reshape(-1)
    x_abs_nonzero = x_abs_nonzero[x_abs_nonzero!=0]
    if x_abs_nonzero.size > 0:
        clipval = x_abs.dtype.type(np.quantile(x_abs_nonzero, quantile))  # numba type hack, would otherwise default to float64
        return _clip_max_float(x, x_abs, clipval)
    else:
        return x  # no nonzero entries in x_abs! avoid division by zero by returning the input

@numba.njit(cache=True)
def clip_max_to_quantile_complex(x, quantile):
    if quantile is None or quantile == 1:
        return x
    x_abs = np.abs(x)
    x_abs_nonzero = x_abs.reshape(-1)
    x_abs_nonzero = x_abs_nonzero[x_abs_nonzero!=0]
    if x_abs_nonzero.size > 0:
        clipval = x_abs.dtype.type(np.quantile(x_abs_nonzero, quantile))  # numba type hack, would otherwise default to float64
        return _clip_max_complex(x, x_abs, clipval)
    else:
        return x  # no nonzero entries in x_abs! avoid division by zero by returning the input


def update_object(Psi, p, w, Oshape, Otype=np.complex64, wsyn=None, eps=1e-12, ps_O=None):
    if wsyn is None:
        wsyn = w
    wsyn_conj = wsyn.conj()
    wprod = np.abs(w * wsyn_conj)
    wsx, wsy = w.shape

    if ps_O is None:
        Owin = np.zeros(Oshape, dtype=Otype)
        Wsum = np.zeros(Oshape, dtype=c2f[Otype])
    else:
        Owin, Wsum = ps_O
        Owin, Wsum = np.copy(Owin), np.copy(Wsum) 
    for i, p_ in enumerate(p):
        px, py = p_
        Owin[px:px+wsx, py:py+wsy] += Psi[i] * wsyn_conj
        Wsum[px:px+wsx, py:py+wsy] += wprod
    O_hat = Owin / np.maximum(Wsum, eps)

    return O_hat, (Owin, Wsum)


def update_probe(Psi, p, O, wshape, wtype=np.complex64, eps=1e-12, ps_P=None, clip_quantile=None, Ppower=None):
    wsx, wsy = wshape
    Oint = np.abs(O)**2
    Oconj = O.conj()

    if ps_P is None:
        Wavesum = np.zeros(wshape, dtype=wtype)
        Osum = np.zeros(wshape, dtype=c2f[wtype])
    else:
        Osum, Wavesum = ps_P
        Osum, Wavesum = np.copy(Osum), np.copy(Wavesum)

    q = clip_quantile
    for i, p_ in enumerate(p):
        px, py = p_
        Oint_local = Oint[px:px+wsx, py:py+wsy]
        Oconj_local = Oconj[px:px+wsx, py:py+wsy]
        Osum    += clip_max_to_quantile_float(Oint_local, q)
        Wavesum += Psi[i] * clip_max_to_quantile_complex(Oconj_local, q)
    w_hat = Wavesum / np.maximum(Osum, eps)

    if Ppower is not None:
        w_hat = w_hat / get_power(w_hat) * Ppower
    
    return w_hat, (Osum, Wavesum)


def istft2(X, p, w, Oshape, Otype=np.complex64, wsyn=None, eps=1e-8, ps_O=None):
    O, _ = update_object(ifft2(X), p, w, Oshape, Otype=Otype, wsyn=wsyn, eps=eps, ps_O=ps_O)
    return O


def pC(X, p, w, Oshape, eps=1e-8, ps_O=None):
    return stft2(
        istft2(X, p, w, Oshape, eps=eps, ps_O=ps_O),
        p,
        w
    )


def pA(X, D):
    return D * np.exp(1j*np.angle(X))


def pC_Psi(Psi, p, w, Oshape, eps=1e-8, ps_O=None):
    O_est, _ = update_object(Psi, p, w, Oshape, eps=eps, ps_O=ps_O)
    return segment2(O_est, p, w)


def pA_Psi(Psi, D):
    X = fft2(Psi)
    return ifft2(pA(X, D))


class ObjectProbeUpdateMode(ValueEqEnum):
    PROBE_THEN_OBJECT = 0  # as in Rodenburg's & Maiden's 2019 book chapter
    PROBE_WITH_PA_THEN_OBJECT = 1
    OBJECT_THEN_PROBE = 2
    OBJECT_THEN_PROBE_WITH_PA_PC = 3
    OBJECT_THEN_PROBE_WITH_PA = 4


def update_object_and_probe_psi_cached(
    Psikj, rk, Pj, Oj, eps, Ak,
    probe_update: bool, mode: ObjectProbeUpdateMode,
    ps_O, ps_P,
    p_Psi_probe=None, probe_clip_quantile=None,
):
    p_probe = rk if p_Psi_probe is None else p_Psi_probe[0]
    Psi_probe = Psikj if p_Psi_probe is None else p_Psi_probe[1]
    Ppower = get_power(Pj)
    Pkwargs = dict(eps=eps, ps_P=ps_P, clip_quantile=probe_clip_quantile, Ppower=Ppower)
    if not probe_update:
        _, ps_P2 = update_probe(Psi_probe, p_probe, Oj, Pj.shape, Pj.dtype, **Pkwargs)
        O2, ps_O2 = update_object(Psikj, rk, Pj, Oj.shape, Oj.dtype, eps=eps, ps_O=ps_O)
        return O2, Pj, ps_O2, ps_P2

    if mode == ObjectProbeUpdateMode.PROBE_THEN_OBJECT:
        w2, ps_P2 = update_probe(Psi_probe, p_probe, Oj, Pj.shape, Pj.dtype, **Pkwargs)
        O2, ps_O2 = update_object(Psikj, rk, w2, Oj.shape, Oj.dtype, eps=eps, ps_O=ps_O)
    elif mode == ObjectProbeUpdateMode.PROBE_WITH_PA_THEN_OBJECT:
        w2, ps_P2 = update_probe(pA_Psi(Psi_probe, Ak), p_probe, Oj, Pj.shape, Pj.dtype, **Pkwargs)
        O2, ps_O2 = update_object(Psikj, rk, w2, Oj.shape, eps=eps, ps_O=ps_O)
    elif mode == ObjectProbeUpdateMode.OBJECT_THEN_PROBE:
        O2, ps_O2 = update_object(Psikj, rk, Pj, Oj.shape, Oj.dtype, eps=eps, ps_O=ps_O)
        w2, ps_P2 = update_probe(Psi_probe, p_probe, O2, Pj.shape, Pj.dtype, **Pkwargs)
    elif mode == ObjectProbeUpdateMode.OBJECT_THEN_PROBE_WITH_PA:
        O2, ps_O2 = update_object(Psikj, rk, Pj, Oj.shape, Oj.dtype, eps=eps, ps_O=ps_O)
        w2, ps_P2 = update_probe(pA_Psi(Psi_probe, Ak), p_probe, O2, Pj.shape, Pj.dtype, **Pkwargs)
    elif mode == ObjectProbeUpdateMode.OBJECT_THEN_PROBE_WITH_PA_PC:
        O2, ps_O2 = update_object(Psikj, rk, Pj, Oj.shape, Oj.dtype, eps=eps, ps_O=ps_O)
        # we don't need to overlap-add again after getting O2, so the 'pC' is performed via segment2(update_object(...)) here
        w2, ps_P2 = update_probe(pA_Psi(segment2(O2, p_probe, Pj), Ak), p_probe, O2, Pj.shape, Pj.dtype, **Pkwargs)
    else:
        raise ValueError(f"Unknown ObjectProbeUpdateMode: {mode}")

    return O2, w2, ps_O2, ps_P2


# ========== Error metrics ==========

def spectral_convergence(X, D, p, w, Oshape, eps_pC=1e-12):
    # X should be a spectrogram
    return np.linalg.norm(D - np.abs(pC(X, p, w, Oshape, eps=eps_pC))) / np.linalg.norm(D)


def spectral_convergence_psi(Psi, D, p, w, Oshape, eps_pC=1e-12):
    return spectral_convergence(fft2(Psi), D, p, w, Oshape, eps_pC=1e-12)


def get_gamma(O, O_est):
    """
    From Maiden and Rodenburg 2009, Eqn. 9.
    Note that for input tensors with more than two dimensions, all but the two last dims are kept.
    This is so the gamma is calculated per input and channel, rather than per batch.
    """
    num = np.sum(O*O_est.conj(), axis=(-2, -1), keepdims=True)
    denom = np.sum(np.abs(O_est)**2, axis=(-2, -1), keepdims=True)
    return num / denom


def zero_phase_ramp(O, return_ramp=False, cut=None):
    if cut is not None:
        Ocut = O[cut]
    else:
        Ocut = O
    ny, nx = np.int32(Ocut.shape[0]), np.int32(Ocut.shape[1])
    cyx = center_of_mass((np.abs(fft2(Ocut)) ** 2))
    dx, dy = cyx[1] - nx / 2, cyx[0] - ny / 2
    y, x = np.meshgrid(
        fftshift_np(fftfreq_np(O.shape[1], d=O.shape[1]/Ocut.shape[1])),
        fftshift_np(fftfreq_np(O.shape[0], d=O.shape[0]/Ocut.shape[0])),
        indexing='ij'
    )
    ramp = np.exp(-2j * np.pi * (x * dx + y * dy))
    O_zeroramp = O * ramp
    if return_ramp:
        return O_zeroramp, ramp
    else:
        return O_zeroramp


def E0(O, O_est, remove_phase_ramp=False):
    """
    From Maiden and Rodenburg 2009, Eqn. 8.
    Optionally zeroes the phase ramp (only!) in O_est when zero_phase_ramp=True is passed.
    """
    if remove_phase_ramp:
        O_est = zero_phase_ramp(O_est)
    gamma = get_gamma(O, O_est)
    return np.sum(np.abs(O - gamma*O_est)**2) / np.sum(np.abs(O)**2)


def get_phase_gamma(O, O_est):
    return np.exp(1j*np.angle(get_gamma(O, O_est)))