import warnings
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import skimage
import h5py


# =========== Object / Probe data ===========

# Load from generated .h5 file (see prepare_data.py)
# get path relative to this code file (not the caller):
h5path = pathlib.Path(__file__).parent.parent / 'data/objects_probes.h5'
h5path = h5path.resolve().expanduser().absolute()

def ensure_data():
    if not h5path.exists():
        raise ValueError(f"Could not find data HDF5 file {h5path}! Run prepare_data.py first.")


def get_object(i):
    ensure_data()
    with h5py.File(h5path, 'r') as f:
        return f[f'objects'][i][:]

def get_probe(i):
    ensure_data()
    with h5py.File(h5path, 'r') as f:
        return f[f'probes'][i][:]


def get_probe_coef(i):
    ensure_data()
    with h5py.File(h5path, 'r') as f:
        return f[f'probe_coefs'][i][:]


def get_probe_r(i):
    ensure_data()
    with h5py.File(h5path, 'r') as f:
        return f[f'probe_rs'][i][:]


def get_object_small(i):
    ensure_data()
    with h5py.File(h5path, 'r') as f:
        return f[f'objects_small'][i][:]


def get_probe_small(i):
    ensure_data()
    with h5py.File(h5path, 'r') as f:
        return f[f'probes_small'][i][:]


def get_probe_coef_small(i):
    ensure_data()
    with h5py.File(h5path, 'r') as f:
        return f[f'probe_coefs_small'][i][:]


def get_probe_r_small(i):
    ensure_data()
    with h5py.File(h5path, 'r') as f:
        return f[f'probe_rs_small'][i][:]


def get_object_and_probe(kO, small, normprobe, ctype):
    ensure_data()
    if small:
        O, P = get_object_small(kO), get_probe_small(kO)
    else:
        O, P = get_object(kO), get_probe(kO)
    if normprobe:
        # normalize probe to power=1
        P = P / get_power(P)
    O, P = O.astype(ctype), P.astype(ctype)
    return O, P


# ========== Scan construction ==========

def get_scan(scandens, O, w):
    if not w.shape == (64, 64) and O.shape == (512, 512):
        warnings.warn("get_scan() was built for object size 512x512 and probe size 64x64! Your sizes are different.")
    if scandens == 10:
        return get_spiral_pm(10, 1500, w, O)
    elif scandens == 15:
        return get_spiral_pm(15, 700, w, O)
    elif scandens == 20:
        return get_spiral_pm(20, 390, w, O)
    elif scandens == 30:
        return get_spiral_pm(30, 170, w, O)
    else:
        raise ValueError(f"Unknown scan density {scandens}!")


# ========== Measurement simulation ==========

def get_beamstop(arrsize, rpix):
    arr = np.ones(arrsize)
    cx, cy = arr.shape[0]//2, arr.shape[1]//2
    xx, yy = np.meshgrid(np.arange(arrsize[0]), np.arange(arrsize[1]))
    arr[(xx-cx)**2 + (yy-cy)**2 < rpix**2] = 0.0
    return arr


def simulate_poisson_noise(difpats, photons_per_pixel, mode, rescale=True):
    if mode == 'max':
        intensity_normfac = np.max(difpats)
    elif mode == 'maxpower':
        intensity_normfac = np.max(np.sum(difpats, axis=(1,2)))
    elif mode == 'mean':
        intensity_normfac = np.mean(difpats)

    normalized_patterns = difpats / intensity_normfac
    noisy_patterns = np.random.poisson(normalized_patterns * photons_per_pixel)
    if rescale:
        # Go back to original data scale to ensure comparability for different `avg_photons_per_pixel` values
        return noisy_patterns * intensity_normfac / photons_per_pixel
    else:
        return noisy_patterns


# ========== Scan construction ==========

def spiral_archimedes(a, n):
    """
    [Taken from PyNX codebase]

    Creates n points spiral of step a, with a between successive points
    on the spiral. Returns the x,y coordinates of the spiral points.

    This is an Archimedes spiral. the equation is:
      r=(a/2*pi)*theta
      the stepsize (radial distance between successive passes) is a
      the curved absciss is: s(theta)=(a/2*pi)*integral[t=0->theta](sqrt(1*t**2))dt
    """
    vr, vt = [0], [0]
    t = np.pi
    while len(vr) < n:
        vt.append(t)
        vr.append(a * t / (2 * np.pi))
        t += 2 * np.pi / np.sqrt(1 + t ** 2)
    vt, vr = np.array(vt), np.array(vr)
    return vr * np.cos(vt), vr * np.sin(vt)


def spiral_fermat(dmax, n):
    """"
    [Taken from PyNX codebase, currently unused]
    
    Creates a Fermat spiral with n points distributed in a circular area with
    diameter<= dmax. Returns the x,y coordinates of the spiral points. The average
    distance between points can be roughly estimated as 0.5*dmax/(sqrt(n/pi))

    http://en.wikipedia.org/wiki/Fermat%27s_spiral
    """
    c = 0.5 * dmax / np.sqrt(n)
    vr, vt = [], []
    t = .4
    goldenAngle = np.pi * (3 - np.sqrt(5))
    while t < n:
        vr.append(c * np.sqrt(t))
        vt.append(t * goldenAngle)
        t += 1
    vt, vr = np.array(vt), np.array(vr)
    return vr * np.cos(vt), vr * np.sin(vt)


def get_spiral_pm(a, n, w, O, toint=True):
    wsx, wsy = w.shape
    spx, spy = spiral_archimedes(a, n)
    spx = spx + O.shape[0]//2 - wsx//2
    spy = spy + O.shape[1]//2 - wsy//2
    pm = np.array([spx, spy])
    if toint:
        pm = np.round(pm).astype(np.int32)
    return pm.T


def get_diffraction_patterns(D_gt, lamb, Ifac):
    Dint = D_gt**2
    Dint_norm = Dint / np.max(Dint)
    D = np.sqrt(simulate_poisson_noise(Dint_norm, lamb, mode='max', rescale=True).astype(np.float32) * Ifac)
    return D


# ========== Object / Probe generation ==========


def rgb_to_complex(img_rgb, mode, alpha=0, beta=1):
    if mode == 'from_hsv':
        # complex object from HSV image with phase from hue, magnitudes from value.
        # saturation is discarded. phase range is [-pi, pi]
        img_hsv = skimage.color.rgb2hsv(img_rgb)
        A = img_hsv[:,:,2]
        P = img_hsv[:,:,0]*2*np.pi - np.pi
    elif mode == 'gray_phase':
        # phase-only object from gray-converted image. phase range is [-pi, pi]
        img_gray = skimage.color.rgb2gray(img_rgb)
        A = 1
        P = 2*np.pi * (img_gray-0.5)
    else:
        raise NotImplementedError()

    O = (A+alpha)/(1+alpha) * np.exp(1j * P*beta)
    return O.astype(np.complex64)


def get_power(x):
    return np.linalg.norm(np.abs(x).reshape(-1), ord=None)


def get_norm_probe(w, D=None):
    if D is None:
        return w / get_power(w)
    else:
        # See Rodenburg&Maiden 2019 book chapter, sec. 17.9.3 "Implementing Ptychographic Algorithms on the Computer"
        w2norm = get_power(w)
        D2norms = np.linalg.norm(np.abs(D).reshape(-1, D.shape[1]*D.shape[2]), axis=(1,), ord=None)
        return w / w2norm * np.max(D2norms)


# ========== Plotting ==========

def plot_ampl_phase(x, log=False, cmaps=('viridis', 'twilight'), axs=None, clim=None, ):
    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)
    else:
        fig = None
    cmap_a, cmap_p = cmaps
    imA = axs[0].imshow(np.abs(x),
                  norm=mpl.colors.LogNorm() if log else mpl.colors.Normalize(), cmap=cmap_a,
                  interpolation='none')
    imP = axs[1].imshow(np.angle(x),
                  norm=mpl.colors.CenteredNorm(halfrange=np.pi), cmap=cmap_p,
                  interpolation='none')
    if clim is not None:
        if isinstance(clim, str) and clim.startswith('q'):
            q = int(clim[1:]) / 100.
            clim = (np.min(np.abs(x)), np.quantile(np.abs(x), q))
        imA.set_clim(clim)
    if fig is not None:
        fig.tight_layout()
    return fig, axs, (imA, imP)


def ccmap_img(cimg, amp_tf=lambda a: a, inv=False, mult=True, cmap='hsv'):
    """
    Maps a complex array to rgba values with information about amplitude and phase, by using
    `phase_cmap` ('twilight' by default) as the colormap for the phase information and mapping
    the normalized amplitudes to:

        - mult=False: The opacity (i.e., large amplitudes map to high saturation, when the background is white).
        - mult=True: The brightness, by multiplying all RGB channels with the normalized amplitudes.
    """
    cmp = mpl.colormaps.get_cmap(cmap)
    a = np.abs(cimg)
    p = np.angle(cimg)
    pn = (p + np.pi) / (2*np.pi)
    phase = cmp(pn)
    if amp_tf is not None:
        a = amp_tf(a)
    a = (a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a))

    if mult:
        rgba = phase
        fac = (1-a) if inv else a
        fac = fac[..., None]
        rgba[...,:3] *= fac
        rgba[...,3] = 1
    else:
        rgba = phase
        rgba[...,3] = (1-a) if inv else a
    return rgba
