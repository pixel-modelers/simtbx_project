from __future__ import division, print_function

import time
import sys
import socket
import logging
import os
import numpy as np
from scipy.optimize import basinhopping, minimize


from libtbx.mpi4py import MPI
from simtbx.diffBragg import hopper_ensemble_utils, hopper_utils, utils
from simtbx.diffBragg.prep_stage2_input import prep_dataframe
from cctbx import miller, crystal, sgtbx
from dials.array_family import flex
from dxtbx.model import ExperimentList


COMM = MPI.COMM_WORLD

MAIN_LOGGER = logging.getLogger("diffBragg.main")

F32 = np.finfo(np.float32)


class TargetFuncEnsemble:

    def __init__(self, vary, xinit=None):
        self.vary = vary
        self.x0 = np.ones(len(self.vary), np.float64)  # initial full parameter list
        if xinit is not None:
            self.x0 = xinit
        self.niter = 0
        self.t_per_iter = np.array([])
        self._fhkl_res_bins = None  # cached resolution bin indices (lo/mid/hi)

    def jac(self, x, *args):
        if self.g is not None:
            return self.g[self.vary]

    @property
    def ave_t_per_iter(self):
        ave_t_per_iter = 0
        if self.t_per_iter.shape[0] > 1:
            ave_t_per_iter = np.mean(self.t_per_iter[1:] - self.t_per_iter[:-1])
        return ave_t_per_iter

    def __call__(self, x, *args, **kwargs):
        self.t_per_iter = np.append(self.t_per_iter, time.time())
        modelers = args[0]

        self.x0[self.vary] = x

        # sync the centric amplitudes
        if modelers.SIM.num_Fhkl_channels >1:
            num_fhkl_x = modelers.SIM.Num_ASU*modelers.SIM.num_Fhkl_channels
            fhkl_param_start = len(self.x0) - num_fhkl_x
            channel0_amps = self.x0[fhkl_param_start: fhkl_param_start+modelers.SIM.Num_ASU]
            centric_amps = channel0_amps[modelers.SIM.is_centric]
            for i_chan in range(1, modelers.SIM.num_Fhkl_channels):
                offset = fhkl_param_start + i_chan*modelers.SIM.Num_ASU
                np.put(self.x0, modelers.SIM.where_is_centric + offset, centric_amps)

        f,  self.g, ave_zscore_sig = target_func(self.x0, modelers)

        # resitribute all centric gradients into channel0 centrics
        if modelers.SIM.num_Fhkl_channels >1:
            num_fhkl_x = modelers.SIM.Num_ASU*modelers.SIM.num_Fhkl_channels
            fhkl_param_start = len(self.x0) - num_fhkl_x
            where_to_add_grad = modelers.SIM.where_is_centric + fhkl_param_start
            for i_chan in range(1, modelers.SIM.num_Fhkl_channels):
                chan_start = fhkl_param_start + i_chan*modelers.SIM.Num_ASU
                chan_grad = self.g[chan_start: chan_start+modelers.SIM.Num_ASU]
                chan_centric_grad = chan_grad[modelers.SIM.is_centric]
                np.add.at(self.g, where_to_add_grad, chan_centric_grad)

        self.niter += 1

        # gradient tracking by component
        num_fhkl_param = modelers.SIM.Num_ASU * modelers.SIM.num_Fhkl_channels
        g_shot = self.g[:-num_fhkl_param] if num_fhkl_param else self.g
        g_fhkl_all = self.g[-num_fhkl_param:] if num_fhkl_param else np.array([])
        g_fhkl = g_fhkl_all[:modelers.SIM.Num_ASU]  # channel 0
        gnorm_shot = np.linalg.norm(g_shot)
        gnorm_fhkl = np.linalg.norm(g_fhkl_all)
        gnorm_all = np.linalg.norm(self.g[self.vary])

        # cache resolution bins on first call (reuse d-spacings from DataModelers)
        if self._fhkl_res_bins is None and len(g_fhkl) > 0:
            d = modelers.get_fhkl_d_spacings()
            thirds = np.percentile(d, [33.3, 66.7])
            self._fhkl_res_bins = {
                "hi": d < thirds[0],
                "mid": (d >= thirds[0]) & (d < thirds[1]),
                "lo": d >= thirds[1],
            }
            if COMM.rank == 0:
                MAIN_LOGGER.info("Gradient res bins: hi=%.1f-%.1f mid=%.1f-%.1f lo=%.1f-%.1f A" % (
                    d.min(), thirds[0], thirds[0], thirds[1], thirds[1], d.max()))

        res_str = ""
        if self._fhkl_res_bins is not None:
            ghi = np.linalg.norm(g_fhkl[self._fhkl_res_bins["hi"]])
            gmid = np.linalg.norm(g_fhkl[self._fhkl_res_bins["mid"]])
            glo = np.linalg.norm(g_fhkl[self._fhkl_res_bins["lo"]])
            res_str = " | fhkl_g(lo=%.2e mid=%.2e hi=%.2e)" % (glo, gmid, ghi)

        min_info = "it=%d | t/it=%.4fs | F=%10.7g | sigZ=%10.7g | |g|=%.2e (shot=%.2e fhkl=%.2e)%s" \
                  % (self.niter, self.ave_t_per_iter, f, ave_zscore_sig,
                     gnorm_all, gnorm_shot, gnorm_fhkl, res_str)
        if COMM.rank==0:
            #print(min_info, flush=True)
            MAIN_LOGGER.info(min_info)
        if modelers.save_freq is not None and self.niter % modelers.save_freq == 0:
            modelers.save_up(self.x0, ref_iter=self.niter)
            if modelers.SIM.D.record_timings:
                modelers.SIM.D.show_timings()

        return f


def target_func(x, modelers):
    """

    :param x: refinement parameters
    :param modelers: instance of DataModelers class
    :return:
    """
    assert modelers.SIM is not None
    assert modelers.SIM.refining_Fhkl

    num_fhkl_params = modelers.SIM.Num_ASU * modelers.SIM.num_Fhkl_channels
    num_shot_params = len(modelers[0].P)  # all modelers will have same number of per-shot parameters to refine
    assert len(x) == num_fhkl_params + modelers.num_total_modelers * num_shot_params

    f = 0  # target functional
    g = np.zeros(modelers.num_total_modelers * num_shot_params)
    g_fhkl = np.zeros(num_fhkl_params)
    zscore_sigs = []
    per_shot_f = []  # per-shot total log-likelihood
    unmerged_rows = []  # list of (shot_idx, asu_indices, roi_loglikes)
    fcell_params = x[-num_fhkl_params:]
    for ii, i_shot in enumerate(modelers):
        shot_modeler = modelers[i_shot]
        shot_x_slice = modelers.x_slices[i_shot]
        per_shot_params = x[shot_x_slice]
        x_for_shot = np.hstack((per_shot_params, fcell_params))
        model_bragg, Jac = hopper_utils.model(x_for_shot, shot_modeler, modelers.SIM, compute_grad=True, update_spectrum=True,
                                              update_Fhkl_scales=ii==0)

        model_pix = model_bragg + shot_modeler.all_background

        if modelers.SIM.use_psf:
            model_pix, J = hopper_utils.convolve_model_with_psf(model_pix, Jac, shot_modeler, modelers.SIM)

        resid = shot_modeler.all_data - model_pix

        # data contributions to target function
        V = model_pix + shot_modeler.all_sigma_rdout**2
        resid_square = resid**2
        with np.errstate(invalid='ignore'):
            shot_fLogLike = (.5*(np.log(2*np.pi*V) + resid_square / V))
            zscore_sig = np.std((resid / np.sqrt(V))[shot_modeler.all_trusted])
        if shot_modeler.params.roi.allow_overlapping_spots:
            shot_fLogLike /= shot_modeler.all_freq

        # per-reflection (per-ROI) log-likelihood within this shot
        if not hasattr(shot_modeler, '_roi_unique') or shot_modeler._roi_unique is None:
            roi_id_arr = np.array(shot_modeler.roi_id)
            shot_modeler._roi_unique = np.unique(roi_id_arr)
            # first pixel index for each ROI → get its ASU HKL
            shot_modeler._asu_idx_per_roi = np.array([
                modelers.SIM.asu_map_int.get(
                    shot_modeler.hi_asu_perpix[np.searchsorted(roi_id_arr, rid)], -1)
                for rid in shot_modeler._roi_unique
            ], dtype=int)
            shot_modeler._roi_id_arr = roi_id_arr
        trusted_loglike = np.where(shot_modeler.all_trusted, shot_fLogLike, 0)
        roi_loglike = np.array([
            trusted_loglike[shot_modeler._roi_id_arr == rid].sum()
            for rid in shot_modeler._roi_unique
        ])
        unmerged_rows.append((i_shot, shot_modeler._asu_idx_per_roi.copy(), roi_loglike))

        shot_fLogLike = shot_fLogLike[shot_modeler.all_trusted].sum()   # negative log Likelihood target
        f += shot_fLogLike
        per_shot_f.append(shot_fLogLike)

        zscore_sigs.append(zscore_sig)

        # get this shots contribution to the gradient
        common_grad_term_all = (0.5 / V * (1 - 2 * resid - resid_square / V))
        if shot_modeler.params.roi.allow_overlapping_spots:
            common_grad_term_all /= shot_modeler.all_freq
        common_grad_term = common_grad_term_all[shot_modeler.all_trusted]

        shot_g = np.zeros(num_shot_params)
        for name in shot_modeler.non_fhkl_params:
            p = shot_modeler.P[name]
            Jac_p = Jac[p.xpos]
            shot_g[p.xpos] += (Jac_p[shot_modeler.all_trusted] * common_grad_term).sum()
        np.add.at(g, shot_x_slice, shot_g)

        spot_scale_p = shot_modeler.P["G_xtal0"]
        G = spot_scale_p.get_val(x[spot_scale_p.xpos])
        g_fhkl += modelers.SIM.D.add_Fhkl_gradients(
            shot_modeler.pan_fast_slow, resid, V, shot_modeler.all_trusted,
            shot_modeler.all_freq, modelers.SIM.num_Fhkl_channels, G)

    # add up target and gradients across all ranks
    f = COMM.bcast(COMM.reduce(f))

    # store unmerged per-reflection diagnostics locally (no MPI gather)
    if not hasattr(modelers, '_diag_unmerged'):
        modelers._diag_unmerged = []
    modelers._diag_unmerged.append(unmerged_rows)

    # average z-score sigma for reporting
    zscore_sigs = COMM.reduce(zscore_sigs)
    ave_zscore_sig = np.mean(COMM.bcast(zscore_sigs))

    # consider sanity checks on g, e.g. at this point it should be 0's outside of all x_slices on this rank
    g = COMM.bcast(COMM.reduce(g))
    g_fhkl = COMM.bcast(COMM.reduce(g_fhkl))

    if COMM.rank==0:
        t = time.time()
        for beta, how in [(modelers.params.betas.Fhkl, "ave"),
                          (modelers.params.betas.Friedel, "Friedel"),
                          (modelers.params.betas.Finit, "init")]:
            if beta is None:
                continue

            for i_chan in range(modelers.SIM.num_Fhkl_channels):
                fhkl_restraint_f, fhkl_restraint_grad = modelers.SIM.D.Fhkl_restraint_data(
                    i_chan,
                    beta,
                    modelers.params.use_geometric_mean_Fhkl,
                    how)
                f += fhkl_restraint_f
                fhkl_slice = slice(i_chan*modelers.SIM.Num_ASU, (i_chan+1)*modelers.SIM.Num_ASU, 1)
                np.add.at(g_fhkl, fhkl_slice, fhkl_restraint_grad)
        t = time.time()-t
        MAIN_LOGGER.debug("Fhkl restraint comp took %.4f sec" %t)
    f = COMM.bcast(f)
    g_fhkl = COMM.bcast(g_fhkl)

    g_fhkl *= modelers.SIM.Fhkl_scales * modelers.params.sigmas.Fhkl  # rescale Fhkl gradient

    g = np.append(g, g_fhkl)

    return f, g, ave_zscore_sig


class DataModelers:

    def __init__(self):
        self.data_modelers = {}
        self.x_slices = {}
        self.num_modelers = 0  # this is the number of modelers on this MPIrank
        self.num_total_modelers = 0  # this is a total summed across MPI ranks
        self.num_param_per_shot = 0
        self._vary = None  # flags for refined variables
        self.SIM = None  # sim_data.SimData instance (one per rank) to be shared amongst the data modelers
        self.cell_for_mtz = None  # unit cell for writing the mtz
        self.max_sigma = 1e20  # max sigma allowed for an optimized amplitude to be included in mtz
        self.outdir = None  # output folder, if None, defaults to the folder used when running hopper
        self.save_freq = None  # optional integer, if provided, save mtz files each 'save_freq' iterations
        self.npix_to_alloc = 0
        self.save_modeler_params = False  # if True, save modelers to pandas files at each iteration

    def get_fhkl_d_spacings(self):
        """Return array of d-spacings for each ASU index, cached after first call."""
        if not hasattr(self, '_fhkl_d_spacings') or self._fhkl_d_spacings is None:
            idx_to_asu = {idx: asu for asu, idx in self.SIM.asu_map_int.items()}
            uc_params = self.cell_for_mtz
            if uc_params is None:
                uc_params = self.data_modelers[list(self.data_modelers.keys())[0]].ucell_man.unit_cell_parameters
            from cctbx import uctbx
            uc = uctbx.unit_cell(tuple(uc_params))
            self._fhkl_d_spacings = np.array([uc.d(idx_to_asu[i]) for i in range(self.SIM.Num_ASU)])
        return self._fhkl_d_spacings

    def get_fhkl_sigmas(self):
        """Return per-reflection sigma array based on empirical gradient magnitudes.
        Accumulates |grad| over iterations, then sets sigma ~ 1/mean(|grad|) so all
        reflections get similar effective step sizes.
        Cached after first call. Only used when params.sigmas.Fhkl < 0 (flag to enable)."""
        if not hasattr(self, '_fhkl_sigmas') or self._fhkl_sigmas is None:
            # not yet computed — return 1.0 (will be set after warmup)
            num_fhkl = self.SIM.Num_ASU * self.SIM.num_Fhkl_channels
            return np.ones(num_fhkl)
        return self._fhkl_sigmas

    def _accumulate_fhkl_grad(self, g_fhkl):
        """Accumulate |gradient| for empirical preconditioning."""
        if not hasattr(self, '_fhkl_grad_accum'):
            self._fhkl_grad_accum = np.zeros_like(g_fhkl)
            self._fhkl_grad_count = 0
        self._fhkl_grad_accum += np.abs(g_fhkl)
        self._fhkl_grad_count += 1

    def _compute_empirical_sigmas(self):
        """Compute sigmas from accumulated gradient magnitudes. sigma ~ 1/mean(|g|)."""
        if not hasattr(self, '_fhkl_grad_accum') or self._fhkl_grad_count == 0:
            return
        mean_abs_g = self._fhkl_grad_accum / self._fhkl_grad_count
        # avoid division by zero for unobserved reflections
        mean_abs_g = np.clip(mean_abs_g, 1e-10, None)
        raw = 1.0 / mean_abs_g
        # normalize so median of observed reflections = 1
        observed = mean_abs_g > 1e-9
        if observed.any():
            raw /= np.median(raw[observed])
        self._fhkl_sigmas = raw
        if COMM.rank == 0:
            MAIN_LOGGER.info("Empirical Fhkl sigmas: min=%.3e median=%.3f max=%.3e (from %d iters)"
                             % (self._fhkl_sigmas.min(), np.median(self._fhkl_sigmas[observed]),
                                self._fhkl_sigmas.max(), self._fhkl_grad_count))

    def set_Fhkl_channels(self):
        if self.SIM is None:
            raise AttributeError("cant set Fhkl channels without a SIM attribute")
        for i_shot, mod in self.data_modelers.items():
            mod.set_Fhkl_channels(self.SIM, set_in_diffBragg=False)
            self.data_modelers[i_shot] = mod

    def _determine_per_rank_max_num_pix(self):
        max_npix = 0
        for i_shot, modeler in self.data_modelers.items():
            x1, x2, y1, y2 = map(np.array, zip(*modeler.rois))
            npix = np.sum((x2-x1)*(y2-y1))
            max_npix = max(npix, max_npix)
        return max_npix

    def _mpi_set_allocation_volume(self):
        assert self.SIM is not None
        assert hasattr(self.SIM, "D")

        MAIN_LOGGER.debug("BEGIN DETERMINE MAX PIX")
        self.npix_to_alloc = self._determine_per_rank_max_num_pix()
        # TODO in case of randomize devices, shouldnt this be total max across all ranks?
        n = COMM.gather(self.npix_to_alloc)
        if COMM.rank == 0:
            n = max(n)
        self.npix_to_alloc = COMM.bcast(n)
        MAIN_LOGGER.debug("DONE DETERMINE MAX PIX (each GPU will allocate space for %d pixels" % self.npix_to_alloc)
        self.SIM.D.Npix_to_allocate = int(self.npix_to_alloc)  # needs to be int32

    def _mpi_sanity_check_num_params(self):
        num_param_per_shot = []
        for i_shot in self.data_modelers:
            mod = self.data_modelers[i_shot]
            num_param_per_shot.append( len(mod.P))
        num_param_per_shot = COMM.reduce(num_param_per_shot)
        if COMM.rank==0:
            assert len(set(num_param_per_shot)) == 1
            num_param_per_shot = num_param_per_shot[0]
        num_param_per_shot = COMM.bcast(num_param_per_shot)

        self.num_param_per_shot = num_param_per_shot

    def mpi_get_ave_cell(self):
        all_ucell_p = []
        for i_shot, mod in self.data_modelers.items():
            ucell_p = mod.ucell_man.unit_cell_parameters
            all_ucell_p.append(ucell_p)

        all_ucell_p = COMM.reduce(all_ucell_p)

        ave_ucell = None
        if COMM.rank==0:
            ave_ucell = np.vstack(all_ucell_p).mean(0)
            print("Setting average unit cell=", ave_ucell)
        ave_ucell = COMM.bcast(ave_ucell)
        return ave_ucell

    def _mpi_get_shots_per_rank(self):
        """
        :return:  dictionary of (rank, number of data modelers on that rank)
        """
        rank_and_numShot = [(COMM.rank, self.num_modelers)]
        rank_and_numShot = COMM.reduce(rank_and_numShot)
        if COMM.rank == 0:
            rank_and_numShot = dict(rank_and_numShot)
        rank_and_numShot = COMM.bcast(rank_and_numShot)
        return rank_and_numShot

    def set_device_id(self):
        assert self.SIM is not None
        dev = COMM.rank % self.params.refiner.num_devices
        MAIN_LOGGER.info("will use device %d on host %s" % (dev, socket.gethostname()))
        self.SIM.D.device_Id = dev

    def mpi_set_x_slices(self):
        """
        x_slices is a dict that should have the same keys as the data_modelers
        Each slice slices through a global paramater array
        """
        self._mpi_sanity_check_num_params()
        self._mpi_compute_num_total_modelers()
        shots_per_rank = self._mpi_get_shots_per_rank()
        start = 0
        npar = self.num_param_per_shot
        for i_rank in range(COMM.size):
            rank_nshots = shots_per_rank[i_rank]

            if COMM.rank==i_rank:
                assert rank_nshots==self.num_modelers  # sanity test
                for i_shot in range(self.num_modelers):
                    x_slice = slice(start+i_shot*npar, start+(i_shot+1)*npar, 1)
                    self.x_slices[i_shot] = x_slice
                break

            rank_npar = npar*rank_nshots
            start += rank_npar

    def _mpi_compute_num_total_modelers(self):
        self.num_total_modelers = COMM.bcast(COMM.reduce(self.num_modelers))

    def __getitem__(self, item):
        assert item in self.data_modelers, "shot %d not in data modelers!" % item
        return self.data_modelers[item]

    def __iter__(self):
        return self.data_modelers.__iter__()

    def add_modeler(self, modeler):
        assert isinstance(modeler, hopper_utils.DataModeler)
        self.data_modelers[self.num_modelers] = modeler
        self.num_modelers += 1

    def prep_for_refinement(self):
        assert self.SIM is not None, "set the sim_data.SimData instance first. example shown in simtbx/command_line/hopper_ensemble.py method load_inputs"
        assert self.SIM.refining_Fhkl
        num_fhkl_param = self.SIM.Num_ASU*self.SIM.num_Fhkl_channels
        num_param_total = self.num_param_per_shot*self.num_total_modelers + num_fhkl_param
        vary = np.zeros(num_param_total)

        for i_shot, x_slice in self.x_slices.items():
            shot_vary = np.ones(self.num_param_per_shot)
            for p in self.data_modelers[i_shot].P.values():
                if not p.refine:
                    shot_vary[p.xpos] = 0
            np.add.at(vary, x_slice, shot_vary)

        vary = COMM.bcast(COMM.reduce(vary))
        # TODO: actually, if there are more than 1 Fhkl channels, then we want to fix the centric Fhkls in all but 1 of the channels
        vary[-num_fhkl_param:] = self._get_fhkl_vary_flags() #1  # we will always vary the fhkl params in ensemble refinement (current default)
        vary = vary.astype(bool)

        # use the first data modeler to set the diffBragg internal refinement flags
        P = self.data_modelers[0].P
        num_ucell_p = len(self.data_modelers[0].ucell_man.variables)
        if P["lambda_offset"].refine:
            for lam_id in hopper_utils.LAMBDA_IDS:
                self.SIM.D.refine(lam_id)
        if P["RotXYZ0_xtal0"].refine:
            self.SIM.D.refine(hopper_utils.ROTX_ID)
            self.SIM.D.refine(hopper_utils.ROTY_ID)
            self.SIM.D.refine(hopper_utils.ROTZ_ID)
        if P["Nabc0"].refine:
            self.SIM.D.refine(hopper_utils.NCELLS_ID)
        if P["Ndef0"].refine:
            self.SIM.D.refine(hopper_utils.NCELLS_ID_OFFDIAG)
        if P["ucell0"].refine:
            for i_ucell in range(num_ucell_p):
                self.SIM.D.refine(hopper_utils.UCELL_ID_OFFSET + i_ucell)
        if P["eta_abc0"].refine:
            self.SIM.D.refine(hopper_utils.ETA_ID)
        if P["detz_shift"].refine:
            self.SIM.D.refine(hopper_utils.DETZ_ID)
        if P["Bfactor"].refine:
            self.SIM.D.refine(hopper_utils.BFACTOR_ID)
        if self.SIM.D.use_diffuse:
            self.SIM.D.refine(hopper_utils.DIFFUSE_ID)

        self._vary = vary

        self._set_mtz_data()
        self.set_device_id()

    def alloc_max_pix_per_shot(self):
        self._mpi_set_allocation_volume()

    def _get_fhkl_vary_flags(self):
        # we vary all Fhkl, however if there are more than 1 Fhkl channels
        # we only vary the centrics in the first channel, and then set those as the values in the other channels
        # (no anomalous scattering in centrics)

        num_fhkl_param = self.SIM.Num_ASU*self.SIM.num_Fhkl_channels
        fhkl_vary = np.ones(num_fhkl_param, int)

        if self.SIM.num_Fhkl_channels > 1:
            assert self.SIM.is_centric is not None
            for i_chan in range(1, self.SIM.num_Fhkl_channels):
                channel_slc = slice(i_chan*self.SIM.Num_ASU, (i_chan+1) *self.SIM.Num_ASU, 1)
                np.subtract.at(fhkl_vary, channel_slc, self.SIM.is_centric.astype(int))

        # only refine hkls that are present in the reflection tables
        all_nominal_hkl = set()
        for mod in self.data_modelers.values():
            all_nominal_hkl = all_nominal_hkl.union(mod.hi_asu_perpix)
        #TODO : is this memory intensive?
        all_nominal_hkl = COMM.gather(all_nominal_hkl)
        if COMM.rank == 0:
            # TODO: all_nominal_hkl is P1, asu_map_int is non-P1
            all_nominal_hkl = set(all_nominal_hkl[0]).union(*all_nominal_hkl[1:])
            is_anom = not self.params.merge_friedel
            all_nominal_hkl_sym = utils.map_hkl_list(all_nominal_hkl, is_anom, self.SIM.crystal.symbol)
            asu_inds_to_vary = [self.SIM.asu_map_int[h] for h in all_nominal_hkl_sym]
        else:
            asu_inds_to_vary = None
        asu_inds_to_vary = set(COMM.bcast(asu_inds_to_vary))
        for i_chan in range(self.SIM.num_Fhkl_channels):
            for i_asu in range(self.SIM.Num_ASU):
                if i_asu not in asu_inds_to_vary:
                    fhkl_vary[i_asu + self.SIM.Num_ASU*i_chan] = 0

        return fhkl_vary


    @property
    def params(self):
        """
        all data_modelers should have the exact same phil extract, so we just grab the first one
        :return:  None or phil extract object
        """
        if not self.data_modelers:
            raise ValueError("No added data modelers! therefore no params")
        return self.data_modelers[0].params

    def Minimize(self, save=True):
        """
        :param save:  save an optimized MTZ file when finished
        """
        assert self._vary is not None, "call prep_for_refinement() first..."

        target = TargetFuncEnsemble(self._vary)
        x0_for_refinement = target.x0[self._vary]

        fhkl_is_varied = self._get_fhkl_vary_flags()
        num_fhkl_refined = int(np.sum(fhkl_is_varied))
        bounds = [(None, None)] * len(x0_for_refinement)
        for i in np.arange(num_fhkl_refined, 0, -1):
            bounds[-i] = (None, 20)
        min_kwargs = {
            "args": (self,),
            "method": "L-BFGS-B",
            "jac": target.jac,
            "bounds": bounds,
            "options" : {
                "ftol": self.params.ftol,
                "gtol": 1e-12,
                "maxfun": 1e5,
                "maxiter": self.params.lbfgs_maxiter,
                "eps": 1e-20
            }
        }

        lbfgs_opts = {
            "ftol": 0,
            "gtol": 1e-15,
            "maxfun": int(1e5),
            "maxiter": int(self.params.lbfgs_maxiter),
            "maxcor": 50,
        }

        out = minimize(target, x0_for_refinement,
                       args=(self,),
                       method="L-BFGS-B",
                       jac=target.jac,
                       bounds=bounds,
                       options=lbfgs_opts)
        target.x0[self._vary] = out.x
        if COMM.rank == 0:
            print("STOP CONDITION:", out.message)
            print("  L-BFGS-B nit:", out.nit, " nfev:", out.nfev)

            # diagnostic: Fhkl parameter values at bounds
            num_fhkl_param = self.SIM.Num_ASU * self.SIM.num_Fhkl_channels
            x_fhkl = target.x0[-num_fhkl_param:]
            n_at_upper = int(np.sum(x_fhkl >= 7.99))
            n_at_lower = int(np.sum(x_fhkl <= -7.99))  # no lower bound, but check anyway
            print("  Fhkl params: min=%.3f  median=%.3f  max=%.3f  at_upper_bound(>=7.99)=%d/%d"
                  % (x_fhkl.min(), np.median(x_fhkl), x_fhkl.max(), n_at_upper, len(x_fhkl)))

            # diagnostic: Fhkl gradient stats by resolution
            g_fhkl = target.g[-num_fhkl_param:]
            d_spacings = self.get_fhkl_d_spacings()
            abs_g = np.abs(g_fhkl[:self.SIM.Num_ASU])
            print("\n  Fhkl gradient at convergence:")
            print("  %8s  %8s  %8s  %8s  %5s" % ("d_lo", "d_hi", "mean|g|", "max|g|", "n"))
            print("  " + "-" * 45)
            edges = np.percentile(d_spacings, np.linspace(0, 100, 6))
            for lo, hi in zip(edges[:-1], edges[1:]):
                mask = (d_spacings >= lo) & (d_spacings < hi + 1e-6)
                if mask.sum() == 0:
                    continue
                print("  %8.2f  %8.2f  %8.2e  %8.2e  %5d" % (
                    lo, hi, abs_g[mask].mean(), abs_g[mask].max(), mask.sum()))
            print("  Overall: mean|g|=%.2e  max|g|=%.2e" % (abs_g.mean(), abs_g.max()))
            print()

        # --- second pass: freeze low-res Fhkl, refine only high-res half ---
        # NOTE: we will revert this later, I dont want to edit further this chunk...
        #if self.params.refine_Fhkl_second_pass:
        #    # compute d-spacing for each Fhkl parameter
        #    idx_to_asu = {idx: asu for asu, idx in self.SIM.asu_map_int.items()}
        #    ave_ucell = self.mpi_get_ave_cell()  # MPI collective, all ranks must call
        #    from cctbx import uctbx
        #    unit_cell = uctbx.unit_cell(tuple(ave_ucell))
        #    d_spacings = np.array([unit_cell.d(idx_to_asu[i]) for i in range(self.SIM.Num_ASU)])
        #    d_median = np.median(d_spacings)

        #    # build mask: True for Fhkl params with d > d_median (low-res, to freeze)
        #    num_fhkl_param = self.SIM.Num_ASU * self.SIM.num_Fhkl_channels
        #    low_res_fhkl = np.zeros(len(target.vary), dtype=bool)
        #    for i_chan in range(self.SIM.num_Fhkl_channels):
        #        offset = len(target.vary) - num_fhkl_param + i_chan * self.SIM.Num_ASU
        #        for i_asu in range(self.SIM.Num_ASU):
        #            if d_spacings[i_asu] > d_median:
        #                low_res_fhkl[offset + i_asu] = True

        #    if COMM.rank == 0:
        #        n_freeze = int(low_res_fhkl.sum())
        #        n_remain = int(fhkl_is_varied.sum()) - n_freeze
        #        print("Second pass: freezing %d low-res Fhkl (d > %.2f A), refining %d high-res"
        #              % (n_freeze, d_median, n_remain))

        #    vary2 = target.vary.copy()
        #    vary2[low_res_fhkl] = False
        #    target.vary = vary2
        #    x0_for_refinement = target.x0[vary2]
        #    bounds2 = [(None, None)] * len(x0_for_refinement)
        #    n_fhkl_vary2 = int(vary2[-num_fhkl_param:].sum())
        #    for i in np.arange(n_fhkl_vary2, 0, -1):
        #        bounds2[-i] = (None, 8)
        #    min_kwargs2 = dict(min_kwargs)
        #    min_kwargs2["bounds"] = bounds2

        #    out = basinhopping(target, x0_for_refinement,
        #                       niter=self.params.niter,
        #                       minimizer_kwargs=min_kwargs2,
        #                       T=self.params.temp,
        #                       callback=None,
        #                       disp=False,
        #                       stepsize=self.params.stepsize)

        #    target.x0[vary2] = out.x
        #    if COMM.rank == 0:
        #        print("STOP CONDITION (pass 2):", out.message)
        #        print("  nit:", out.nit, " nfev:", out.nfev)
        #        res = out.lowest_optimization_result
        #        print("  L-BFGS-B:", res.message)
        #        print("  L-BFGS-B nit:", res.nit, " nfev:", res.nfev)

        # save unmerged per-reflection diagnostics (one file per rank)
        if hasattr(self, '_diag_unmerged'):
            import os, pickle
            diag_dir = self.outdir or "."
            idx_to_asu = {idx: asu for asu, idx in self.SIM.asu_map_int.items()}
            asu_hkls = [idx_to_asu[i] for i in range(self.SIM.Num_ASU)]
            d_spacings = self.get_fhkl_d_spacings()
            n_iter = len(self._diag_unmerged)
            shot_ids = []
            asu_idxs = []
            for shot_idx, asu_idx_arr, _ in self._diag_unmerged[0]:
                for asu_idx in asu_idx_arr:
                    shot_ids.append(shot_idx)
                    asu_idxs.append(asu_idx)
            shot_ids = np.array(shot_ids, dtype=int)
            asu_idxs = np.array(asu_idxs, dtype=int)
            n_rows = len(shot_ids)
            loglikes = np.zeros((n_iter, n_rows))
            for it, iter_data in enumerate(self._diag_unmerged):
                offset = 0
                for shot_idx, asu_idx_arr, roi_ll in iter_data:
                    n = len(roi_ll)
                    loglikes[it, offset:offset+n] = roi_ll
                    offset += n

            diag = {
                "shot_id": shot_ids,
                "asu_idx": asu_idxs,
                "loglike": loglikes,
                "asu_hkls": asu_hkls,
                "d_spacings": d_spacings,
            }
            diag_path = os.path.join(diag_dir, "stage2_diag_rank%d.pkl" % COMM.rank)
            with open(diag_path, "wb") as fh:
                pickle.dump(diag, fh)
            if COMM.rank == 0:
                print("Saved unmerged diagnostics (per-rank) → %s/stage2_diag_rank*.pkl" % diag_dir)
                print("  %d ranks, %d observations on rank 0 × %d iterations" % (COMM.size, n_rows, n_iter))

        if save:
            self.save_up(target.x0)
        return target.x0

    def _set_mtz_data(self):
        idx_to_asu = {idx: asu for asu, idx in self.SIM.asu_map_int.items()}
        asu_hkls = [idx_to_asu[i] for i in range(self.SIM.Num_ASU)]
        inds, amps = self.SIM.D.Fhkl_tuple
        amplitude_map = {h: amp for h, amp in zip(inds, amps)}
        assert set(asu_hkls).intersection(amplitude_map) == set(asu_hkls)
        self.initial_intens = np.array([amplitude_map[h]**2 for h in asu_hkls])
        self.flex_asu = flex.miller_index(asu_hkls)

    def save_up(self, x, ref_iter=None):
        """
        :param x: optimized parameters output by self.Minimize
        :param ref_iter: iteration number for optional saving during minimization (e.g. each X iterations)
        """
        assert self.outdir is not None

        cell_for_mtz = self.cell_for_mtz
        if self.cell_for_mtz is None:
            cell_for_mtz = tuple(self.mpi_get_ave_cell())
        sym = crystal.symmetry(cell_for_mtz, self.params.space_group)

        Fhkl_scale_hessian = np.zeros(self.SIM.Num_ASU * self.SIM.num_Fhkl_channels)
        for i_shot, mod in self.data_modelers.items():
            mod.best_model, _ = hopper_utils.model(x, mod, self.SIM, compute_grad=False, update_spectrum=True)
            mod.best_model_includes_background = False
            resid = mod.all_data - (mod.best_model+mod.all_background)
            V = mod.best_model + mod.all_sigma_rdout ** 2
            Gparam = mod.P["G_xtal0"]
            G = Gparam.get_val(x[Gparam.xpos])
            # here we must use the CPU method
            if i_shot % 100==0:
                MAIN_LOGGER.info("Getting Fhkl errors for shot %d/%d ... " % (i_shot+1, self.num_modelers))
            Fhkl_scale_hessian += self.SIM.D.add_Fhkl_gradients(
                mod.pan_fast_slow, resid, V, mod.all_trusted, mod.all_freq,
                self.SIM.num_Fhkl_channels, G, track=False, errors=True)
            # ------------

        Fhkl_scale_hessian = COMM.reduce(Fhkl_scale_hessian)

        if COMM.rank==0:
            # resitribute the Hessian for centrics
            if self.SIM.num_Fhkl_channels > 1:
                for i_chan in range(1, self.SIM.num_Fhkl_channels):
                    chan_start = i_chan * self.SIM.Num_ASU
                    chan_hess = Fhkl_scale_hessian[chan_start: chan_start + self.SIM.Num_ASU]
                    chan_centric_hess = chan_hess[self.SIM.is_centric]
                    np.add.at(Fhkl_scale_hessian, self.SIM.where_is_centric, chan_centric_hess)

                total_centric_hess = Fhkl_scale_hessian[self.SIM.where_is_centric]
                for i_chan in range(1, self.SIM.num_Fhkl_channels):
                    chan_start = i_chan * self.SIM.Num_ASU
                    where_to_put_hess = self.SIM.where_is_centric + chan_start
                    np.put(Fhkl_scale_hessian, where_to_put_hess, total_centric_hess)

            if not os.path.exists(self.outdir):
                os.makedirs(self.outdir)

            for i_chan in range(self.SIM.num_Fhkl_channels):

                mtz_prefix = "optimized_channel%d" % i_chan
                if ref_iter is not None:
                    mtz_prefix += "_iter%d" % ref_iter
                mtz_name = os.path.join(self.outdir, "%s.mtz" % mtz_prefix)

                fhkl_slice = slice(i_chan*self.SIM.Num_ASU, (i_chan+1)*self.SIM.Num_ASU,1)
                channel_scales = self.SIM.Fhkl_scales[fhkl_slice]
                channel_hessian = Fhkl_scale_hessian[fhkl_slice]
                with np.errstate(all='ignore'):
                    channel_scales_var = 1 / channel_hessian

                safe_vals = np.logical_and( channel_scales_var >= F32.min, channel_scales_var <= F32.max)
                is_finite = np.logical_and(safe_vals, ~np.isinf(channel_scales_var))

                #is_finite = ~np.isinf(channel_scales_var.astype(np.float32))  # should be finite float32
                is_reasonable = channel_scales_var < self.max_sigma
                is_positive = channel_hessian > 0
                sel = is_positive & is_finite & is_reasonable
                optimized_data = channel_scales[sel] * self.initial_intens[sel]
                optimized_sigmas = np.sqrt(channel_scales_var[sel]) * self.initial_intens[sel]
                channel_inds = self.flex_asu.select(flex.bool(sel))

                assert not np.any(np.isnan(optimized_sigmas)), "should be no nans here"
                mset = miller.set(sym, channel_inds, not self.params.merge_friedel)

                ma = miller.array(mset, flex.double(optimized_data), flex.double(optimized_sigmas))
                ma = ma.set_observation_type_xray_intensity().as_amplitude_array()
                ma.as_mtz_dataset(column_root_label="F").mtz_object().write(mtz_name)

        if self.save_modeler_params:

            num_fhkl_params = self.SIM.Num_ASU * self.SIM.num_Fhkl_channels
            fcell_params = x[-num_fhkl_params:]
            for i_shot, mod in self.data_modelers.items():
                temp = mod.params.tag
                if ref_iter is not None:
                    mod.params.tag = mod.params.tag + ".iter%d" % ref_iter
                else:
                    mod.params.tag = mod.params.tag + ".final"
                # TODO: x should be for this particular modeler (fhkl_slice)
                shot_x_slice = self.x_slices[i_shot]
                per_shot_params = x[shot_x_slice]
                x_for_shot = np.hstack((per_shot_params, fcell_params))
                mod.save_up(x_for_shot, self.SIM, COMM.rank,
                            save_modeler_file=False,
                            save_fhkl_data=False,
                            save_sim_info=False,
                            save_refl=False)
                mod.params.tag = temp


def get_gather_name(exper_name, gather_dir):
    gathered_name = os.path.splitext(os.path.basename(exper_name))[0]
    gathered_name += "_withData.refl"
    gathered_name = os.path.join(gather_dir, gathered_name)
    return os.path.abspath(gathered_name)


def load_inputs(pandas_table, params, exper_key="exp_name", refls_key='predictions',
                gather_dir=None, exper_idx_key="exp_idx"):

    pandas_table, work_distribution = prep_dataframe(pandas_table, refls_key,
                                       res_ranges_string=params.refiner.res_ranges)
    COMM.barrier()
    num_exp = len(pandas_table)
    first_exper_file = pandas_table[exper_key].values[0]
    first_exper = ExperimentList.from_file(first_exper_file, check_format=False)[0]
    detector = first_exper.detector
    if detector is None and params.refiner.reference_geom is None:
        raise RuntimeError("No detector in experiment, must provide a reference geom.")
    # TODO verify all shots have the same detector ?
    if params.refiner.reference_geom is not None:
        detector = ExperimentList.from_file(params.refiner.reference_geom, check_format=False)[
            0].detector
        MAIN_LOGGER.debug("Using reference geom from expt %s" % params.refiner.reference_geom)

    if COMM.size > num_exp:
        raise ValueError("Requested %d MPI ranks to process %d shots. Reduce number of ranks to %d"
                         % (COMM.size, num_exp, num_exp))

    exper_names = pandas_table[exper_key]
    exper_ids = pandas_table[exper_idx_key]
    name_ids = list(zip(exper_names, exper_ids))
    assert len(name_ids) == len(set(name_ids))
    worklist = work_distribution[COMM.rank]
    MAIN_LOGGER.info("EVENT: begin loading inputs")

    Fhkl_model = utils.load_Fhkl_model_from_params_and_expt(params, first_exper)

    Fhkl_model = Fhkl_model.expand_to_p1().generate_bijvoet_mates()
    Fhkl_model_indices = set(Fhkl_model.indices())
    shot_modelers = hopper_ensemble_utils.DataModelers()
    for ii, i_df in enumerate(worklist):
        exper_name = exper_names[i_df]
        exper_id = int(exper_ids[i_df])
        MAIN_LOGGER.info("EVENT: BEGIN loading experiment list")
        check_format = not (params.refiner.load_data_from_refl or params.load_data_from_refls)
        expt = hopper_utils.DataModeler.exper_json_single_file(exper_name, exper_id, check_format)
        expt_list = ExperimentList()
        expt_list.append(expt)
        MAIN_LOGGER.info("EVENT: DONE loading experiment list")
        expt.detector = detector  # in case of supplied ref geom

        exper_dataframe = pandas_table.query("%s=='%s'" % (exper_key, exper_name)).query("%s==%d" % (exper_idx_key, exper_id))

        refl_name = exper_dataframe[refls_key].values[0]
        refls = flex.reflection_table.from_file(refl_name)
        refls = refls.select(refls['id'] == exper_id)

        miller_inds = list( refls['miller_index'])
        is_not_000 = [h != (0, 0, 0) for h in miller_inds]
        is_in_Fhkl_model = [h in Fhkl_model_indices for h in miller_inds]
        MAIN_LOGGER.debug("Only refining %d/%d refls whose HKL are in structure factor model" %(np.sum(is_in_Fhkl_model), len(refls)))
        refl_sel = flex.bool(np.logical_and(is_not_000, is_in_Fhkl_model))
        refls = refls.select(refl_sel)

        exp_cry_sym = expt.crystal.get_space_group().type().lookup_symbol()
        if params.space_group is not None and exp_cry_sym.replace(" ", "") != params.space_group:
            gr = sgtbx.space_group_info(params.space_group).group()
            expt.crystal.set_space_group(gr)
            #raise ValueError("Crystals should all have the same space group symmetry")

        MAIN_LOGGER.info("EVENT: LOADING ROI DATA")
        shot_modeler = hopper_utils.DataModeler(params)
        shot_modeler.exper_name = exper_name
        shot_modeler.exper_idx = exper_id
        shot_modeler.refl_name = refl_name
        shot_modeler.rank = COMM.rank
        # TODO fix redundant phils
        if params.refiner.load_data_from_refl or params.load_data_from_refls:
            gathered = shot_modeler.GatherFromReflectionTable(expt, refls, sg_symbol=params.space_group)
            MAIN_LOGGER.debug("tried loading from reflection table")
        else:
            gathered = shot_modeler.GatherFromExperiment(expt, refls, sg_symbol=params.space_group)
            MAIN_LOGGER.debug("tried loading data from expt table")
        if not gathered:
            raise IOError("Failed to gather data from experiment %s", exper_name)
        else:
            MAIN_LOGGER.debug("successfully loaded data")
        MAIN_LOGGER.info("EVENT: DONE LOADING ROI")

        if gather_dir is not None:
            gathered_name = get_gather_name(exper_name, gather_dir)
            shot_modeler.dump_gathered_to_refl(gathered_name, do_xyobs_sanity_check=False)
            MAIN_LOGGER.info("SAVED ROI DATA TO %s" % gathered_name)
            all_data = shot_modeler.all_data.copy()
            all_roi_id = shot_modeler.roi_id.copy()
            all_bg = shot_modeler.all_background.copy()
            all_trusted = shot_modeler.all_trusted.copy()
            all_pids = np.array(shot_modeler.pids)
            all_rois = np.array(shot_modeler.rois)
            new_Modeler = hopper_utils.DataModeler(params)
            assert new_Modeler.GatherFromReflectionTable(exper_name, gathered_name, sg_symbol=params.space_group)
            assert np.allclose(new_Modeler.all_data, all_data)
            assert np.allclose(new_Modeler.all_background, all_bg)
            assert np.allclose(new_Modeler.rois, all_rois)
            assert np.allclose(new_Modeler.pids, all_pids)
            assert np.allclose(new_Modeler.all_trusted, all_trusted)
            assert np.allclose(new_Modeler.roi_id, all_roi_id)
            MAIN_LOGGER.info("Gathered file approved!")

        if gather_dir is not None:
            continue

        shot_modeler.set_parameters_for_experiment(best=exper_dataframe)
        shot_modeler.set_spectrum(spectra_file=exper_dataframe.spectrum_filename.values[0])
        MAIN_LOGGER.info("Will simulate %d energy channels" % len(shot_modeler.nanoBragg_beam_spectrum))

        # verify this
        shot_modeler.Umatrices = [shot_modeler.E.crystal.get_U()]

        MAIN_LOGGER.info(utils.memory_report('Rank 0 reporting memory usage'))
        if COMM.rank==0:
            print("Finished loading image %d / %d" % (ii + 1, len(worklist)), flush=True)

        shot_modelers.add_modeler(shot_modeler)

    if gather_dir is not None:
        if COMM.rank==0:
            pandas_table['ens.hopper.imported'] = [get_gather_name(f_exp, gather_dir) for f_exp in pandas_table[exper_key]]
            pd_name = os.path.join(params.outdir, "preImport_for_ensemble.pkl")
            pandas_table.to_pickle(pd_name)
            print("Wrote file %s to be used to re-run ens.hopper . Use optional ens.hopper arg '--refl ens.hopper.imported', and the phil param load_data_from_refl=True to load the imported data" % pd_name)
        COMM.barrier()
        sys.exit()
    shot_modelers.mpi_set_x_slices()

    assert shot_modelers.num_modelers > 0

    # use the first shot modeler to create a sim data instance:
    shot_modelers.SIM = hopper_utils.get_simulator_for_data_modelers(shot_modelers[0])

    shot_modelers.set_Fhkl_channels()

    return shot_modelers
