"""
Module for Zernike probe function generation
"""

import numpy as np
from zernike import RZern


DEFAULT_SCALES = np.array([
    0,       # Piston
    # Order 1
    0.2,   # X-tilt
    0.2,   # Y-tilt
    # Order 2
    0.2,     # Defocus
    0.2,     # Horizontal astigmatism
    0.2,     # Vertical astigmatism
    # Order 3
    0.2,     # Vertical coma
    0.2,     # Horizontal coma
    0.2,     # Vertical trefoil
    0.2,     # Horizontal trefoil
    # Order 4
    0.2,     # Spherical aberration
    0.2,     # Vertical secondary astigmatism
    0.2,     # Horizontal secondary astigmatism
    0.2,     # Vertical quadrafoil
    0.2,     # Horizontal quadrafoil
])


def get_zern(maxdeg, r=3, pixels=100, cls=RZern):
    zern = cls(maxdeg)
    range_ = np.linspace(-r, r, pixels)
    xv, yv = np.meshgrid(range_, range_)
    zern.make_cart_grid(xv, yv, unit_circle=True)
    return zern


def get_probe(pixels, coefs, oversample=16, maxdeg=4, r=1.2):
    # Get Zernike polynomial instance
    zern = get_zern(maxdeg, r=r, pixels=pixels*oversample, cls=RZern)
    if not coefs.shape == (zern.nk,):
        raise ValueError(f"Coeffient shape {coefs.shape} does not match expected shape {(zern.nk,)}!")

    # Construct circle to use as an amplitude mask later
    c = np.zeros((zern.nk,))
    c[0] = 1.
    circle = zern.eval_grid(c, matrix=True)
    circle[np.isnan(circle)] = 0.0

    # Calculate phase aberrations by evaluating the Zernike polynomial
    phase = zern.eval_grid(coefs, matrix=True)
    phase[np.isnan(phase)] = 0.0  # Remove NaNs, these would mess up our Fourier Transform
    phi = circle * np.exp(1j * phase)

    # Fourier transform the wavefront at the aperture to get a probe function
    phi_ft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(phi), norm='ortho'))

    # Center-crop the probe function to the target number of pixels
    x, y = phi_ft.shape
    l, r = pixels//2, pixels//2+pixels%2
    probe = phi_ft[int(y/2-l):int(y/2+r), int(x/2-l):int(x/2+r)]

    return probe


def get_random_probe(pixels, oversample=16, maxdeg=4, r=1.2, scales=DEFAULT_SCALES):
    coefs = np.random.normal(loc=0.0, scale=scales)#, size=(zern.nk,))
    probe = get_probe(pixels, coefs, oversample=oversample, maxdeg=maxdeg, r=r)
    return probe
