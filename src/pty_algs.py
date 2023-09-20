from typing import List

import numpy as np
from tqdm.auto import tqdm as TQDM

from .pty_base import pA_Psi, pC_Psi
from .pty_base import ObjectProbeUpdateMode, update_object_and_probe_psi_cached


class IterativeAlgorithm:
    """
    Parent class for iterative algorithms that are implemented in a stateless way.
    """
    def initialize(self, Psik0, rk, P0, Ak, O0):
        """
        Returns a data structure that is a full representation of an algorithm state,
        given the initial exit-waves `Psik0`, scan positions `rk`, probe guess `P0`,
        diffraction patterns `Ak`, and object guess `O0`.
        Implementations can add arbitrary objects and iterates to the state, such as
        iterates from previous steps.

        The implementation of `iterate` must both receive and return such data structures.

        The default implementation of this method returns a tuple
        `(iterations, Psik0, rk, P0, Ak, O0)`,
        with iterations set to 0.

        NOTE: no object- or measurement-specific state should be stored on the algorithm itself! This is
        to make `IterativeAlgorithm` instances stateless and usable for several objects or partial objects
        in parallel / after each other.
        """
        state = (0, Psik0, rk, P0, Ak, O0)
        return state

    def iterate(self, state, probe_update, partial_sums=None):
        """
        Performs a single iteration of the algorithm from the given `state`, returning a new state.
        Performs a probe update if `probe_update=True`.

        If passed, can optionally make use of past partial sums `partial_sums = (ps_O, ps_P)`
        for the object and probe.
        """
        pass

    def finalize(self, state, partial_sums=None):
        """
        Finalizes the `state`, in particular w.r.t. the Psi and w estimation.
        May perform a step such as an amplitude projection, or may also be a no-op.
        By default, performs an amplitude projection on Psikj and leaves the rest alone.
        """
        (i, Psikj, rk, Pj, Ak, Oj) = state
        return (i, pA_Psi(Psikj, Ak), rk, Pj, Ak, Oj)

    def get_iteration_number(self, state):
        """
        Returns the iteration number from the given `state`.
        """
        return state[0]

    def get_psi(self, state):
        """
        Returns the current exit-wave estimate from the given `state`.
        """
        (_, Psikj, _, _, _, _) = state
        return Psikj

    def get_positions(self, state):
        """
        Returns the scan positions from the given `state`.
        """
        (_, _, rk, _, _, _) = state
        return rk

    def get_probe(self, state):
        """
        Returns the current probe estimate from the given `state`.
        """
        (_, _, _, Pj, _, _) = state
        return Pj

    def get_difpats(self, state):
        """
        Returns the diffraction patterns from the given `state`.
        """
        (_, _, _, _, Ak, _) = state
        return Ak

    def get_object(self, state):
        """
        Returns the current object estimate from the given `state`.
        """
        (_, _, _, _, _, Oj) = state
        return Oj

    def __repr__(self):
        return str(self)


class PsiER(IterativeAlgorithm):
    def __init__(self, eps=1e-8, reversed=False,
                 object_probe_update_mode=ObjectProbeUpdateMode.PROBE_THEN_OBJECT,
                 probe_clip_quantile=None):
        self.eps = eps
        self.reversed = reversed
        self.object_probe_update_mode = object_probe_update_mode
        self.probe_clip_quantile = probe_clip_quantile

    def __str__(self):
        return f"PsiER(eps={self.eps},probe_clip_quantile={self.probe_clip_quantile})"

    def iterate(self, state, probe_update, partial_sums=None):
        (i, Psikj, rk, Pj, Ak, Oj) = state
        eps = self.eps
        reversed_ = self.reversed
        object_probe_update_mode = self.object_probe_update_mode
        ps_O, ps_P = partial_sums if partial_sums is not None else (None, None)

        if not reversed_:
            PsiPC = pC_Psi(Psikj, rk, Pj, Oj.shape, eps=eps, ps_O=ps_O)
            Psi_next = pA_Psi(PsiPC, Ak)
        else:
            PsiPA = pA_Psi(Psikj, Ak)
            Psi_next = pC_Psi(PsiPA, rk, Pj, Oj.shape, eps=eps, ps_O=ps_O)

        O_next, P_next, _, _ = update_object_and_probe_psi_cached(
            Psi_next, rk, Pj, Oj, eps, Ak, probe_update=probe_update,
            mode=object_probe_update_mode, probe_clip_quantile=self.probe_clip_quantile,
            ps_O=ps_O, ps_P=ps_P
        )
        return (i+1, Psi_next, rk, P_next, Ak, O_next)


PsiGLA = PsiER


class PsiFGLA(IterativeAlgorithm):
    def __init__(self, alpha, eps=1e-8, reversed=False,
                 object_probe_update_mode=ObjectProbeUpdateMode.PROBE_THEN_OBJECT,
                 probe_clip_quantile=None):
        self.alpha = alpha  # only constant alpha for now
        self.eps = eps
        self.reversed = reversed
        self.object_probe_update_mode = object_probe_update_mode
        self.probe_clip_quantile = probe_clip_quantile

    def __str__(self):
        return f"PsiFGLA(alpha={self.alpha},eps={self.eps},probe_clip_quantile={self.probe_clip_quantile})"

    def initialize(self, Psik0, rk, P0, Ak, O0):
        xPsi = pC_Psi(Psik0, rk, P0, O0.shape, eps=self.eps)  # ensure xPsi, yPsi are in the C set
        yPsi = Psik0
        state = (0, xPsi, yPsi, rk, P0, Ak, O0)
        return state

    def iterate(self, state, probe_update, partial_sums=None):
        (i, xPsi, yPsi, rk, Pj, Ak, Oj) = state
        alpha = self.alpha
        eps = self.eps
        object_probe_update_mode = self.object_probe_update_mode
        ps_O, ps_P = partial_sums if partial_sums is not None else (None, None)

        if not reversed:
            yPsi_PC = pC_Psi(yPsi, rk, Pj, Oj.shape, eps=eps, ps_O=ps_O)
            xPsi_next = pA_Psi(yPsi_PC, Ak)
        else:
            yPsi_PA = pA_Psi(yPsi, Ak)
            xPsi_next = pC_Psi(yPsi_PA, rk, Pj, Oj.shape, eps=eps, ps_O=ps_O)
        yPsi_next = xPsi_next + alpha*(xPsi_next - xPsi)

        O_next, P_next, _, _ = update_object_and_probe_psi_cached(
            yPsi_next, rk, Pj, Oj, eps, Ak, probe_update=probe_update,
            mode=object_probe_update_mode, probe_clip_quantile=self.probe_clip_quantile,
            ps_O=ps_O, ps_P=ps_P
        )
        return (i+1, xPsi_next, yPsi_next, rk, P_next, Ak, O_next)

    def finalize(self, state, partial_sums=None):
        (i, xPsi, yPsi, rk, Pj, Ak, Oj) = state
        return (i, xPsi, pA_Psi(yPsi, Ak), rk, Pj, Ak, Oj)

    def get_psi(self, state):
        (_, _, yPsi, _, _, _, _) = state
        return yPsi

    def get_probe(self, state):
        (_, _, _, _, P, _, _) = state
        return P

    def get_object(self, state):
        (_, _, _, _, _, _, O) = state
        return O

    def get_positions(self, state):
        (_, _, rk, _, _, _, _) = state
        return rk

    def get_difpats(self, state):
        (_, _, _, _, _, Ak, _) = state
        return Ak


class PsiDM(IterativeAlgorithm):
    def __init__(self, beta, eps=1e-8,
                 object_probe_update_mode=ObjectProbeUpdateMode.PROBE_THEN_OBJECT,
                 probe_clip_quantile=None):
        self.beta = beta
        self.eps = eps
        self.object_probe_update_mode = object_probe_update_mode
        self.probe_clip_quantile = probe_clip_quantile

    def __str__(self):
        return f"PsiDM(beta={self.beta},eps={self.eps},probe_clip_quantile={self.probe_clip_quantile})"

    def iterate(self, state, probe_update, partial_sums=None):
        (i, Psikj, rk, Pj, Ak, Oj) = state
        beta = self.beta
        eps = self.eps
        ps_O, ps_P = partial_sums if partial_sums is not None else (None, None)

        if beta == 1:
            # optimized implementation for beta=1 which avoids extraneous projections
            PsiPC = pC_Psi(Psikj, rk, Pj, Oj.shape, eps=eps, ps_O=ps_O)
            PsiPARC = pA_Psi(2*PsiPC - Psikj, Ak)
            Psi_next = Psikj + PsiPARC - PsiPC
        elif beta == -1:
            # optimized implementation for beta=-1 which avoids extraneous projections
            PsiPA = pA_Psi(Psikj, Ak)
            PsiPCRA = pC_Psi(2*PsiPA - Psikj, rk, Pj, Oj.shape, eps=eps, ps_O=ps_O)
            Psi_next = Psikj + PsiPCRA - PsiPA
        else:
            PsiPA = pA_Psi(Psikj, Ak)
            PsiPC = pC_Psi(Psikj, rk, Pj, Oj.shape, eps=eps, ps_O=ps_O)
            fA = PsiPA - (PsiPA - Psikj) / beta
            fC = PsiPC + (PsiPC - Psikj) / beta
            pCfA = pC_Psi(fA, rk, Pj, Oj.shape, eps=eps, ps_O=ps_O)
            pAfC = pA_Psi(fC, Ak)
            Psi_next = Psikj + beta*(pAfC - pCfA)

        O_next, P_next, _, _ = update_object_and_probe_psi_cached(
            Psi_next, rk, Pj, Oj, eps, Ak, probe_update=probe_update,
            mode=self.object_probe_update_mode, probe_clip_quantile=self.probe_clip_quantile,
            ps_O=ps_O, ps_P=ps_P
        )
        return (i+1, Psi_next, rk, P_next, Ak, O_next)


class PsiRAAR(IterativeAlgorithm):
    def __init__(self, beta, eps=1e-8,
                 object_probe_update_mode=ObjectProbeUpdateMode.PROBE_THEN_OBJECT,
                 probe_clip_quantile=None):
        self.beta = beta
        self.eps = eps
        self.object_probe_update_mode = object_probe_update_mode
        self.probe_clip_quantile = probe_clip_quantile

    def __str__(self):
        return f"PsiRAAR(beta={self.beta},eps={self.eps},probe_clip_quantile={self.probe_clip_quantile})"

    def iterate(self, state, probe_update, partial_sums=None):
        (i, Psikj, rk, Pj, Ak, Oj) = state
        beta = self.beta
        eps = self.eps
        ps_O, ps_P = partial_sums if partial_sums is not None else (None, None)

        if beta >= 0:
            PsiPC = pC_Psi(Psikj, rk, Pj, Oj.shape, eps=eps, ps_O=ps_O)
            PsiRC = 2*PsiPC - Psikj
            PsiPARC = pA_Psi(PsiRC, Ak)
            PsiRARC = 2*PsiPARC - PsiRC
            Psi_next = 0.5*beta*(Psikj + PsiRARC) + (1-beta)*PsiPC
        else:
            # swap C and A sets -- this is a bit of an unusual definition, but lets us explore more
            beta = -beta  # make effective beta positive
            PsiPA = pA_Psi(Psikj, Ak)
            PsiRA = 2*PsiPA - Psikj
            PsiPCRA = pC_Psi(PsiRA, rk, Pj, Oj.shape, eps=eps, ps_O=ps_O)
            PsiRCRA = 2*PsiPCRA - PsiRA
            Psi_next = 0.5*beta*(Psikj + PsiRCRA) + (1-beta)*PsiPA

        O_next, P_next, _, _ = update_object_and_probe_psi_cached(
            Psi_next, rk, Pj, Oj, eps, Ak, probe_update=probe_update,
            mode=self.object_probe_update_mode, probe_clip_quantile=self.probe_clip_quantile,
            ps_O=ps_O, ps_P=ps_P,
        )
        return (i+1, Psi_next, rk, P_next, Ak, O_next)


class PsiHybrid(IterativeAlgorithm):
    def __init__(self, algs: List[IterativeAlgorithm], switches: List[int],
                 finalize_before_switch: bool = False):
        self.algs = algs
        self.switches = np.array(switches)
        self.finalize_before_switch = finalize_before_switch
        assert (np.sort(self.switches) == self.switches).all(), "Switches array needs to be sorted!"
        assert len(algs) == len(switches) + 1

    def __str__(self):
        return f"PsiHybrid(algs={self.algs},switches={self.switches},finalize_before_switch={self.finalize_before_switch})"

    def initialize(self, Psik0, rk, P0, Ak, O0):
        alg0state = self.algs[0].initialize(Psik0, rk, P0, Ak, O0)
        return (0, alg0state)

    def _get_alg_idx(self, i):
        return np.searchsorted(self.switches, i, side='right')

    def _get_alg(self, i):
        return self.algs[self._get_alg_idx(i)]

    def iterate(self, state, probe_update, partial_sums=None):
        (i, wrapped_state) = state
        alg_idx = self._get_alg_idx(i)
        alg_idx_prev = self._get_alg_idx(i-1)
        alg = self.algs[alg_idx]

        if alg_idx != alg_idx_prev:
            # Algorithm switch occurs in this iteration, reinitialize wrapped_state for next alg
            alg_prev = self.algs[alg_idx_prev]
            if self.finalize_before_switch:
                wrapped_state = alg_prev.finalize(wrapped_state)
            wrapped_state = alg.initialize(
                alg_prev.get_psi(wrapped_state),
                alg_prev.get_positions(wrapped_state),
                alg_prev.get_probe(wrapped_state),
                alg_prev.get_difpats(wrapped_state),
                alg_prev.get_object(wrapped_state),
            )
        wrapped_state_next = alg.iterate(wrapped_state, probe_update, partial_sums)                
        return (i+1, wrapped_state_next)

    def finalize(self, state, partial_sums=None):
        (i, wrapped_state) = state
        alg = self._get_alg(i-1)
        return (i, alg.finalize(wrapped_state))

    def get_psi(self, state):
        (i, wrapped_state) = state
        alg = self._get_alg(i-1)
        return alg.get_psi(wrapped_state)

    def get_positions(self, state):
        (i, wrapped_state) = state
        alg = self._get_alg(i-1)
        return alg.get_positions(wrapped_state)

    def get_probe(self, state):
        (i, wrapped_state) = state
        alg = self._get_alg(i-1)
        return alg.get_probe(wrapped_state)

    def get_difpats(self, state):
        (i, wrapped_state) = state
        alg = self._get_alg(i-1)
        return alg.get_difpats(wrapped_state)

    def get_object(self, state):
        (i, wrapped_state) = state
        alg = self._get_alg(i-1)
        return alg.get_object(wrapped_state)


def run_psi_alg(
    alg, iterations, Psik0, rk, P0, Ak,
    tqdm=True, track_states=False,
    probe_update=False, O0_init=None, Oshape=None,
):
    assert Psik0.shape == Ak.shape, "Exit waves (Psi) shape should match diffraction patterns (D) shape"

    # Set up state-tracking if `track_states=True`
    if track_states:
        tracked_states = []

    # Create a calculation box
    # shift offsets so minimum is at (0, 0)
    wsx, wsy = P0.shape
    if O0_init is None:
        if Oshape is None:
            pshift = rk - np.min(rk, axis=0)
            max_x, max_y = np.max(pshift, axis=0)
            max_x += wsx
            max_y += wsy
        else:
            pshift = rk
            max_x, max_y = Oshape
        O0 = np.ones((max_x, max_y), dtype=Psik0.dtype)
    else:
        pshift = rk
        O0 = np.copy(O0_init)
    
    alg_state = alg.initialize(Psik0, pshift, P0, Ak, O0)
    if track_states:
        tracked_states.append(alg_state)

    # Iterate algorithm state
    iterator = list(range(iterations))
    if tqdm:
        iterator = TQDM(iterator)
    for i in iterator:
        alg_state = alg.iterate(alg_state, probe_update=probe_update)
        if track_states:
            tracked_states.append(alg_state)

    # Finalize algorithm to get final state
    final_state = alg.finalize(alg_state)

    # Return
    if track_states:
        return final_state, tracked_states
    else:
        return final_state
    

def parse_eps(epsstr, D):
    if epsstr.startswith('M'):
        return np.max(D)/float(epsstr[1:])
    return float(epsstr)


def parse_alg(alg, D):
    try:
        if alg.startswith('dm'):
            params = alg[2:].split(',')
            beta = float(params[0])
            eps = parse_eps(params[1], D)
            pq = float(params[2]) if len(params) > 2 else None
            return PsiDM(beta=beta, eps=eps, probe_clip_quantile=pq)
        elif alg.startswith('raar'):
            params = alg[4:].split(',')
            beta = float(params[0])
            eps = parse_eps(params[1], D)
            pq = float(params[2]) if len(params) > 2 else None
            return PsiRAAR(beta=beta, eps=eps, probe_clip_quantile=pq)
        elif alg.startswith('fgla'):
            params = alg[4:].split(',')
            alpha = float(params[0])
            eps = parse_eps(params[1], D)
            pq = float(params[2]) if len(params) > 2 else None
            return PsiFGLA(alpha=alpha, eps=eps, probe_clip_quantile=pq)
        elif alg.startswith('er'):
            params = alg[2:].split(',')
            eps = parse_eps(params[0], D)
            pq = float(params[1]) if len(params) > 1 else None
            return PsiER(eps=eps, probe_clip_quantile=pq)
        elif alg.startswith('hybrid') or alg.startswith('Fhybrid'):
            finalize_before_switch = alg.startswith('F')
            switches_algs = alg[len('hybrid_')+1*finalize_before_switch:].split('+')
            times = [int(sw_alg.split('X')[0]) for sw_alg in switches_algs]
            switches = list(np.cumsum([0] + times))[1:-1]
            algs = [parse_alg(sw_alg.split('X')[1], D) for sw_alg in switches_algs]
            return PsiHybrid(algs=algs, switches=switches, finalize_before_switch=finalize_before_switch)
        else:
            raise ValueError(f"Unknown algorithm type '{alg}'. Valid types are 'dm', 'raar', 'fgla', 'er', 'hybrid', 'Fhybrid'")
    except Exception as e:
        print(f"Error parsing algorithm string '{alg}': {e}")
        raise e
