import argparse
import pathlib

import h5py
import skimage
from skimage import io as skio
import numpy as np

from src.pty_data import rgb_to_complex
from src.pty_probes import get_probe, DEFAULT_SCALES


def get_random_probe_coefs():
    coefs = np.random.normal(loc=0.0, scale=probe_coef_scales)
    return coefs


def get_random_probe_and_coefs():
    maxdeg = 4
    coefs = get_random_probe_coefs()
    r = np.random.uniform(3.5, 5.5)
    pixels = 64
    oversample = 8
    probe = get_probe(pixels, coefs, oversample=oversample, maxdeg=maxdeg, r=r)
    return probe, coefs, r


if __name__ != '__main__':
    raise ValueError("This script should be run directly, not imported.")


parser = argparse.ArgumentParser()
parser.add_argument('DIV2K_HR_test_512crop', type=str, help='Path to folder containing DIV2K_HR_test_512crop images')
args = parser.parse_args()


# =========== Object generation ===========

# Load images
base_folder = pathlib.Path(args.DIV2K_HR_test_512crop).resolve().expanduser().absolute()
pngs = base_folder.glob('*.png')
pngs = np.array(list(sorted(pngs)))
imgs_rgb = np.array([skio.imread(png_path) for png_path in pngs])
assert len(imgs_rgb) == 100, "Expected 100 images in folder!"
print(f"Found {len(imgs_rgb)} images in folder. Processing to get 90 objects: 40x'from_hsv', 40x'gray_phase', and 10x'paired'...")

# 40x from_hsv, 40x gray_phase, the 20 others in pairs as magnitude + phase (done separately)
modes = ['from_hsv']*40 + ['gray_phase']*40

# array holding all final complex-valued objects
objects = []

# process first 40+40 images according to `modes`
np.random.seed(31879)
alphas = np.random.uniform(0.0, 1.0, len(modes))
betas = np.random.uniform(0.3, 0.99, len(modes))
for img_rgb, mode, alpha, beta in zip(imgs_rgb, modes, alphas, betas):
    O = rgb_to_complex(img_rgb, mode, alpha=alpha, beta=beta)
    objects.append(O)

# process final 20 images as pairs. generate new alphas and betas for that
nleft = len(pngs) - len(modes)
assert nleft%2 == 0
np.random.seed(50322)
alphas2 = np.random.uniform(0.0, 1.0, nleft//2)
betas2 = np.random.uniform(0.3, 0.99, nleft//2)
for i, (alpha, beta) in enumerate(zip(alphas2, betas2)):
    img1 = imgs_rgb[len(modes)+2*i]
    img2 = imgs_rgb[len(modes)+2*i+1]
    ampl = (skimage.color.rgb2gray(img1)+alpha)/(1+alpha)
    phase = beta * (2*np.pi*skimage.color.rgb2gray(img2) - np.pi)
    O = ampl * np.exp(1j * phase)
    objects.append(O)
print("Done generating objects.")


# =========== Probe generation ===========

print(f"Generating {len(objects)} associated probes...")
probe_coef_scales = DEFAULT_SCALES

# arrays holding all complex probe coefficients and their 64x64 probes. will have same length as objects array
probe_coefs = []
probes = []
probe_rs = []
np.random.seed(54985127)
for i in range(len(objects)):
    probe, coefs, r = get_random_probe_and_coefs()
    probes.append(probe)
    probe_coefs.append(coefs)
    probe_rs.append(r)
print("Done generating probes.")


# =========== Saving ===========

# Convert to numpy arrays and construct 'small' subset
print("Constructing 'small' subset of full dataset.")
objects_small = np.array(objects[:10] + objects[40:50] + objects[80:90])
probes_small = np.array(probes[:10] + probes[40:50] + probes[80:90])
probe_coefs_small = np.array(probe_coefs[:10] + probe_coefs[40:50] + probe_coefs[80:90])
probe_rs_small = np.array(probe_rs[:10] + probe_rs[40:50] + probe_rs[80:90])

# Save to single HDF5 file
outpath = pathlib.Path('data/objects_probes.h5').resolve().expanduser().absolute()
print("Saving all to HDF5 file:", outpath)
dskw = dict(compression="gzip",  compression_opts=5)
with h5py.File(outpath, 'w') as f:
    f.create_dataset('objects', data=np.array(objects), **dskw)
    f.create_dataset('probes', data=np.array(probes), **dskw)
    f.create_dataset('probe_coefs', data=np.array(probe_coefs), **dskw)
    f.create_dataset('probe_rs', data=np.array(probe_rs), **dskw)
    f.create_dataset('objects_small', data=objects_small, **dskw)
    f.create_dataset('probes_small', data=probes_small, **dskw)
    f.create_dataset('probe_coefs_small', data=probe_coefs_small, **dskw)
    f.create_dataset('probe_rs_small', data=probe_rs_small, **dskw)

print("Done!")
