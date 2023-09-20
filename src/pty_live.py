from typing import Optional, List

import numpy as np
from tqdm.auto import tqdm as TQDM

from .pty_base import ifft2, stft2, pA_Psi, get_naive_phase_init
from .pty_base import update_object_and_probe_psi_cached, ObjectProbeUpdateMode
from .pty_base import ValueEqEnum
from .pty_algs import IterativeAlgorithm


class RealTimeProbeUpdate(ValueEqEnum):
    NONE = 0
    EACH_ITERATION = 1  # most natural, as it is analogous to how the object is handled
    AFTER_COMMIT_ONLY = 2
    EACH_ITERATION_ONLY = 3


def get_and_apply_phi0(Psi_fluid, r_fluid, A_fluid, O, P, idxs):
    if not idxs:  # None or empty list
        return
    if not isinstance(idxs, list):
        idxs = [idxs]  # could be made more efficient for single-index case, but hey.
    Phi0 = np.angle(stft2(O, r_fluid[idxs], P))
    Psi_fluid[idxs] = ifft2(A_fluid[idxs] * np.exp(1j * Phi0))


def run_rtisi_psi(
    Ak: np.array, rk: np.array, Oshape: tuple, P0: np.array,
    B: int, alg: IterativeAlgorithm, iters: int,
    phi0_est_idxs: List[int], naive_phase_init: str = 'zero',
    Psik0: Optional[np.array] = None, O0: Optional[np.array] = None,
    pA_before_commit: bool = True,
    probe_update: RealTimeProbeUpdate = RealTimeProbeUpdate.NONE,
    object_probe_update_mode: ObjectProbeUpdateMode = ObjectProbeUpdateMode.PROBE_THEN_OBJECT,
    kmax: int = None, track_interm: Optional[List[str]] = None,
    eps_commit: float = 1e-12, pq_commit: Optional[float] = None,
):
    """
    Run generalized ptycho-RTISI for diffraction patterns `D` (shape (N, m, n)),
    scan position offsets `p` (shape (N, 2)), initial-object `O0` (shape (K, L)),
    using buffersize `B` and `alg` as the real-time capable iterative algorithm,
    running for `iters` iterations before freezing each exit wave.

    Returns (O_final, w_final, Psi_final), with final object estimate `O_final`, probe `w_final`
    and exit-wave collection `Psi_final`.

    `phi0_est_idxs` sets the indices for which Phi0 should be estimated (always based on previous object estimate).
    Can be empty list / None to disable Phi0 estimation. [-1] is a good default.

    `naive_phase_init` sets the style of naive phase initialization, used for the first few exit waves and
    (by default, may be overridden dependent on phi0_est_variant) each new incoming exit wave.
    Can be 'zero' or 'uniform' [0, 2pi).

    `Psi0` can be passed if an estimate for Psi has been calculated before (e.g., in a previous reconstruction run).
    If None, then Psi0 will be determined from `Ak` and `naive_phase_init`.

    `pA_before_commit` determines whether a final amplitude projection pA should be applied
    before committing each exit wave, and is True by default.

    `probe_update` sets the way of doing the probe update. NONE (no probe update) by default.

    `track_interm` can be a list of strings, determining which intermediate results to track. Available are
    'Psi', 'O', 'P', 'ps_O', 'ps_P'. If None (default), no intermediate results are tracked.

    `kmax` optionally limits the number of diffraction patterns to iterate over, useful for quick debugging or experiments.
    """
    kmax = kmax if kmax is not None else len(rk)
    rdtype = Ak.dtype
    cdtype = (Ak[0] + 1j).dtype

    # Use O0 if it was passed, otherwise initialize as array of ones (free space).
    O0 = np.ones(Oshape, dtype=cdtype) if O0 is None else O0
    # Use Psi0 if it was passed, otherwise initialize as zero-phase.
    if Psik0 is not None:
        Psikj = Psik0
    else:
        Psikj = ifft2(Ak + 0j)

    Oj = O0
    Pj = P0
    # Assign initial partial-object and partial-probe iterates as zeros. This is sensible
    # when assuming these should only contain *committed* iterates, as there is no committed
    # information at this point in the algorithm.
    ps_O = (np.zeros_like(Oj), np.zeros(Oj.shape, rdtype))
    ps_P = (np.zeros(Pj.shape, rdtype), np.zeros_like(Pj))

    # Track initial estimates if track_interm is not None
    if track_interm is not None:
        initial = {'Psi': Psikj, 'O': Oj, 'P': Pj, 'ps_O': ps_O, 'ps_P': ps_P}
        all_interms = [{k: v for k, v in initial.items() if k in track_interm}]

    for k in TQDM(range(0, kmax, 1)):  # Process k exit waves sequentially. This is the 'outer loop'
        fluid_slice = slice(k, k+B)
        # Use B exit waves starting from the k-th exit wave, treat them as fluid
        r_fluid = rk[fluid_slice]
        # get associated diffraction patterns and (potential) past exit wave estimates for fluid exit waves
        A_fluid = Ak[fluid_slice]
        Psi_fluid = Psikj[fluid_slice]

        if Psik0 is None:
            # Use zero-phase difpat and naive phase initialization to get an initial estimate for the new incoming exit wave. Specific phase-init variants may overwrite this.
            # If Psi0 was provided, this step is skipped to make use of the previous exit-wave estimation.
            Psi_fluid[-1] = ifft2(get_naive_phase_init(A_fluid[-1], kind=naive_phase_init))

        # Get and apply an initial phase estimate to exit waves in the buffer.
        get_and_apply_phi0(Psi_fluid, r_fluid, A_fluid, Oj, Pj, idxs=phi0_est_idxs)

        # Get bounding box of fluid exit waves for efficient calculations
        rmin, rmax = r_fluid.min(axis=0), r_fluid.max(axis=0)
        rmax += np.array(Pj.shape)  # add probe shape to get max coordinates
        rmin, rmax = rmin.astype(int), rmax.astype(int)
        r_fluid_shift = r_fluid - rmin
        # crop out region of interest from Oj as well as from partial sums in ps_O
        cut = (slice(rmin[0], rmax[0]), slice(rmin[1], rmax[1]))
        Oj_crop = Oj[cut]
        ps_O_crop = (ps_O[0][cut], ps_O[1][cut])

        # Iterate Psi_fluid with given algorithm for `iters` many times. This is the 'inner loop'
        state = alg.initialize(Psi_fluid, r_fluid_shift, Pj, A_fluid, O0=Oj_crop)
        pu_iter = probe_update in (RealTimeProbeUpdate.EACH_ITERATION, RealTimeProbeUpdate.EACH_ITERATION_ONLY)
        for j in range(iters):
            state = alg.iterate(state, probe_update=pu_iter, partial_sums=(ps_O_crop, ps_P))
            if track_interm is not None:
                interms_dict = {
                    'Psi': alg.get_psi(state), 'O': alg.get_object(state), 'P': alg.get_probe(state),
                    'ps_O': ps_O, 'ps_P': ps_P
                }
                all_interms.append({k: v for k, v in interms_dict.items() if k in track_interm})

        # Get iterated Psi_fluid from algorithm state
        Psi_fluid = alg.get_psi(state)

        # Commit exit wave k. Optionally perform pA on it before committing (this will also
        # affect the `update_object_and_probe_psi_cached` call a bit below). Then update `Psi`
        # with the fluid exit waves so they can carry over to the next outer-loop iteration.
        if pA_before_commit:
            Psi_fluid[0] = pA_Psi(Psi_fluid[0], A_fluid[0])
        Psikj[fluid_slice] = Psi_fluid[:]
        
        # Update object and probe estimates as well as running sum variables by committing the k-th exit wave.
        if probe_update == RealTimeProbeUpdate.EACH_ITERATION_ONLY:
            # If EACH_ITERATION_ONLY, we need to get the probe from the iterated state: it will not be updated below.
            Pj = alg.get_probe(state)
        Oj, Pj, ps_O, ps_P = update_object_and_probe_psi_cached(
            Psikj[[k]], rk[[k]], Pj, Oj, eps_commit, A_fluid,
            probe_update=(probe_update in (RealTimeProbeUpdate.EACH_ITERATION, RealTimeProbeUpdate.AFTER_COMMIT_ONLY)),
            mode=object_probe_update_mode,
            ps_O=ps_O,
            ps_P=ps_P,
            probe_clip_quantile=pq_commit,
        )

    if track_interm is None:
        return Oj, Pj, Psikj, ps_O, ps_P, None
    else:
        return Oj, Pj, Psikj, ps_O, ps_P, all_interms
