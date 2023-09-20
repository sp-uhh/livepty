import numpy as np
import scipy as sp


f2c = {np.float32: np.complex64, np.float64: np.complex128,
       np.complex64: np.complex64, np.complex128: np.complex128}
c2f = {np.complex64: np.float32, np.complex128: np.float64,
       np.float32: np.float32, np.float64: np.float64}
f2c = {**f2c, **{np.dtype(k): np.dtype(v) for k, v in f2c.items()}}
c2f = {**c2f, **{np.dtype(k): np.dtype(v) for k, v in c2f.items()}}

fftshift = lambda x: sp.fft.fftshift(x, axes=(-2, -1))
ifftshift = lambda x: sp.fft.ifftshift(x, axes=(-2, -1))
fft2_unshifted = lambda x: sp.fft.fft2(x, axes=(-2, -1), norm='ortho')
ifft2_unshifted = lambda x: sp.fft.ifft2(x, axes=(-2, -1), norm='ortho')
fft2 = lambda x: fftshift(fft2_unshifted(ifftshift(x)))
ifft2 = lambda x: fftshift(ifft2_unshifted(ifftshift(x)))
for shape in [(16, 16), (17, 17), (16, 17), (17, 16)]:
    xtest = np.random.randn(*shape)
    assert np.isclose(ifft2(fft2(xtest)), xtest).all()