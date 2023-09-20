# Live Iterative Ptychography with projection-based algorithms

Accompanying code repository for our work
[**Live Iterative Ptychography with projection-based algorithms**](https://arxiv.org/abs/2309.08639).
Developed by Simon Welker and Tal Peer, based in part on the bachelor thesis of Johannes Kolhoff.

We kindly ask you to cite our work in your publication when using any of our research or code:

```bib
@article{welker2023livePty,
  title={Live Iterative Ptychography with projection-based algorithms},
  author={Welker, Simon and Peer, Tal and Chapman, Henry N. and Gerkmann, Timo},
  journal={arXiv preprint arXiv:2309.08639},
  date={2023-09-14},
  eprint={2309.08639},
  eprinttype={arxiv},
  pubstate={preprint},
}
```


## Setup

We recommend using Python 3.9 or newer. Create a new virtual environment and run

```bash
pip install -r requirements.txt
```


## Data preparation

If you wish to use our simulated dataset based on [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/), you first need to
download the DIV2K validation set from [the following URL](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip),
extract it, and then run the following ImageMagick command inside your copy of the `DIV2K/DIV2K_valid_HR/` folder:

```bash
for f in *.png; do convert "$f" -resize 512x512^ -gravity Center -crop 512x512+0+0 +repage "512crop/${f}"; done
```

which generates a folder of all images resized and center-cropped to 512x512 (as we used it). Then run:

```bash
python prepare_data.py <the_512crop_subfolder_you_just_generated>/
```

which generates all 90 test objects and probes and stores them in the file `data/objects_probes.h5` of this repository.


## Repository structure

- Library code is inside `src/`
- Scripts to run reconstructions are:
    - `run_recon.py` (live reconstruction)
    - `run_recon_classical.py` (non-live reconstruction)
    - `run_central_recons.py` (for pre-reconstruction of the central region).
- Results will be stored as HDF5 files under `results/`
    - `view_recon.py` can be used for visualization of a single result file (e.g. `results/run1/0.h5`).
      We recommend using the options `--zero-phase-ramp --apply-gamma` for best viewing experience.
- Data is stored inside `data/`
    - This data is not provided with the repository and must be generated, via `prepare_data.py`
      (for simulated test objects and probes) and `run_central_recons.py` (for pre-reconstructed central regions).
