from __future__ import absolute_import, division, print_function

from libtbx.mpi4py import MPI
from simtbx.diffBragg import stage_two_utils

COMM = MPI.COMM_WORLD
if not hasattr(COMM, "rank"):
    COMM.rank=0
    COMM.size=1
import time
import warnings
import signal
import logging
from copy import deepcopy
from simtbx.diffBragg import hopper_io

LOGGER = logging.getLogger("diffBragg.main")
warnings.filterwarnings("ignore")


class SignalHandler:
    def __init__(self):
        self.t = time.time()

    def handle(self, signum, frame):
        t = time.time()-self.t
        print("Recived signal ",signum," after program running for %f sec" % t)
        raise BreakBecauseSignal


SIGHAND = SignalHandler()


class Bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

try:
    import pandas
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# TODO : consider PEP-8 ing these numpy imports, but do a NERSC massively MPI time-test first...
# for now, if it aint broke, dont fix it ...
import numpy as np
import os

from simtbx.diffBragg.refiners import BreakBecauseSignal, BreakToUseCurvatures
from dials.array_family import flex
from simtbx.diffBragg.refiners import BaseRefiner
from cctbx import miller, sgtbx
from simtbx.diffBragg.refiners.parameters import RangedParameter

from simtbx.diffBragg.hopper_utils import BFACTOR_ID, BFACTOR_ANISO_ID, NCELLS_ID, NCELLS_ID_OFFDIAG


class StageTwoRefiner(BaseRefiner):

    def __init__(self, shot_modelers, sgsymbol, params):
        BaseRefiner.__init__(self)

        self.params = params
        self.trad_conv_eps = self.params.refiner.tradeps
        self.calc_curvatures = self.params.refiner.curvatures
        self.break_signal = self.params.refiner.break_signal
        self.output_dir = self.params.refiner.io.output_dir  # directory to dump progress files, these can be used to restart simulation later
        self.save_model_freq = self.params.refiner.stage_two.save_model_freq
        self.use_nominal_h = self.params.refiner.stage_two.use_nominal_hkl

        self.saveZ_freq = self.params.refiner.stage_two.save_Z_freq  # save Z-score data every N function calls
        self.break_signal = None  # check for this signal during refinement, and break refinement if signal is received (see python signal module) TODO: make work with MPI
        self.save_model = False  # whether to save the model
        self.hiasu = None  # stores Hi_asu, counts, maps to and from fcell indices
        self.rescale_params = True  # whether to rescale parameters during refinement  # TODO this will always be true, so remove the ability to disable
        self.request_diag_once = False  # LBFGS refiner property
        self.min_multiplicity = self.params.refiner.stage_two.min_multiplicity
        self.restart_file = None  # output file from previous run refinement
        self.trial_id = 0  # trial id in case multiple trials are run in sequence
        self.x_init = None  # used to restart the refiner (e.g. self.x gets updated with this)
        self.log_fcells = True  # to refine Fcell using logarithms to avoid negative Fcells
        self.refine_crystal_scale = False  # whether to refine the crystal scale factor
        self.refine_Fcell = False  # whether to refine Fhkl for each shoebox ROI
        self.use_curvatures_threshold = 7  # how many positive curvature iterations required before breaking, after which simulation can be restart with use_curvatures=True
        self.verbose = True  # whether to print during iterations
        self.iterations = 0  # iteration counter , used internally
        self.target_eval_count = 0  # target function evaluation counter, used internally
        self.shot_ids = None  # for global refinement ,
        self.log2pi = np.log(np.pi*2)

        self._sig_hand = None  # method for handling the break_signal, e.g. SIGHAND.handle defined above (theres an MPI version in global_refiner that overwrites this in stage 2)
        self._is_trusted = None  # used during refinement, 1-D array or trusted pixels corresponding to the pixels in the ROI

        self.rank = COMM.rank

        self.Modelers = shot_modelers
        self.shot_ids = sorted(self.Modelers.keys())
        # part of the re-parameterization for the per-spot scale factors requires us to take the sqrt here
        for i_shot in self.shot_ids:
            self.Modelers[i_shot].PAR.Scale.init = np.sqrt(self.Modelers[i_shot].PAR.Scale.init)
        self.n_shots = len(shot_modelers)
        self.n_shots_total = COMM.bcast(COMM.reduce(self.n_shots))
        LOGGER.debug("Loaded %d shots across all ranks" % self.n_shots_total)
        self.f_vals = []  # store the functional over time

        self._ncells_id = 9  # diffBragg internal index for Ncells derivative manager
        self._detector_distance_id = 10  # diffBragg internal index for detector_distance derivative manager
        self._panelRotO_id = 14  # diffBragg internal index for derivative manager
        self._panelRotF_id = 17  # diffBragg internal index for derivative manager
        self._panelRotS_id = 18  # diffBragg internal index for derivative manager
        self._panelX_id = 15  # diffBragg internal index for  derivative manager
        self._panelY_id = 16  # diffBragg internal index for  derivative manager
        self._fcell_id = 11  # diffBragg internal index for Fcell derivative manager
        self._eta_id = 19  # diffBragg internal index for eta derivative manager
        self._lambda0_id = 12  # diffBragg interneal index for lambda derivatives
        self._lambda1_id = 13  # diffBragg interneal index for lambda derivatives
        self._ncells_def_id = 21

        self.symbol = sgsymbol
        self.space_group = sgtbx.space_group(sgtbx.space_group_info(symbol=self.symbol).type().hall_symbol())

        self.REGIONS = None  # detector regions for gain refinement (this is a labeled array, same shape as self.S.detector
        self.num_regions = None  # the number of unique regions
        self.unique_regions = None  # the unique regions as a 1-d np.array
        self.region_params = {}  # dictionary for storuing diffBragg/refiners/parameters.RangerParameter for gain correction params

        self.I_AM_ROOT = COMM.rank==0

    def _compute_n_param_per_shot(self):
        """Compute number of per-shot parameters from config.

        Layout: G (1) + B (1) + Nabc/Cholesky (3 or 6) + Baniso (0 or 6)
        """
        n = 2  # G + B always present
        test_mod = self.Modelers[self.shot_ids[0]]
        if getattr(test_mod.PAR, 'use_cholesky_Nabc', False):
            n += 6  # Cholesky L11..L33
        else:
            n += 3  # Na, Nb, Nc
        if hasattr(test_mod.PAR, 'Baniso') and test_mod.PAR.Baniso is not None:
            n += 6  # B11..B23
        return n

    def _load_gain_regions(self):
        npan = len(self.S.detector)
        nfast, nslow = self.S.detector[0].get_image_size()
        det_shape = npan, nslow, nfast
        self.REGIONS = stage_two_utils.regionize_detector(det_shape, self.params.refiner.region_size)
        self.unique_regions = np.unique(self.REGIONS)
        self.num_regions = len(self.unique_regions)

    def __call__(self, *args, **kwargs):
        _, _ = self.compute_functional_and_gradients()
        return self.x, self._f, self._g, self.d

    @property
    def n(self):
        """LBFGS property"""
        return len(self.x)

    @property
    def n_global_fcell(self):
        return self.hiasu.present_len

    @property
    def image_shape(self):
        panelXdim, panelYdim = self.S.detector[0].get_image_size()
        Npanels = len(self.S.detector)
        return Npanels, panelYdim, panelXdim

    @property
    def x(self):
        """LBFGS parameter array"""
        return self._x

    @x.setter
    def x(self, val):
        self._x = val

    def _check_keys(self, shot_dict):
        """checks that the dictionary keys are the same"""
        if not sorted(shot_dict.keys()) == self.shot_ids:
            raise KeyError("input data funky, check GlobalRefiner inputs")
        return shot_dict

    def _set_current_gain_per_pixel(self):
        M = self.Modelers[self._i_shot]
        M._gain_region_per_pixel = self.REGIONS[M.all_pid, M.all_slow, M.all_fast]
        M.all_gain = self._gain_per_region[M._gain_region_per_pixel]

    def _evaluate_averageI(self):
        """model_Lambda means expected intensity in the pixel"""
        # NOTE: gain correction is applied to te background fit, as the background was fit to the data
        self.model_Lambda = self.Modelers[self._i_shot].all_background + self.model_bragg_spots

    def make_output_dir(self):
        if self.I_AM_ROOT and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.Zdir = os.path.join(self.output_dir, "Z")
        self.model_dir = os.path.join(self.output_dir, "model")
        for dirname in (self.Zdir, self.model_dir):
            if self.params.debug_mode and self.I_AM_ROOT and not os.path.exists(dirname):
                os.makedirs(dirname)
        COMM.barrier()

    def _setup(self):
        # Here we go!  https://youtu.be/7VvkXA6xpqI
        if not self.params.debug_mode:
            LOGGER.info("Disabling saveZ and save_model because debug_mode=False")
            self.saveZ_freq = None
            self.save_model_freq = None
        LOGGER.info("Setup begins!")
        if self.refine_Fcell and not self.hiasu.from_idx:
            raise ValueError("Need to supply a non empty asu from idx map")
        if self.refine_Fcell and not self.hiasu.to_idx:
            raise ValueError("Need to supply a non empty idx from asu map")

        self.make_output_dir()
        self._load_gain_regions()

        self.shot_mapping = self._get_shot_mapping()
        self.n_total_shots = len(self.shot_mapping)

        test_shot = self.shot_ids[0]
        self.n_ucell_param = len(self.Modelers[test_shot].PAR.ucell_man.variables)  # not used
        self.n_param_per_shot = self._compute_n_param_per_shot()
        self._use_cholesky = getattr(self.Modelers[test_shot].PAR, 'use_cholesky_Nabc', False)
        self._refining_Baniso = (hasattr(self.Modelers[test_shot].PAR, 'Baniso')
                                 and self.Modelers[test_shot].PAR.Baniso is not None
                                 and not self.params.fix.Baniso)
        LOGGER.info("Per-shot params: %d (cholesky=%s, Baniso=%s)"
                     % (self.n_param_per_shot, self._use_cholesky, self._refining_Baniso))
        self.n_total_params = self.n_total_shots*self.n_param_per_shot + self.n_global_fcell + self.num_regions

        # Build per-shot xpos mappings
        # Layout: [G, B, Nabc/Chol (3 or 6), Baniso (0 or 6)] per shot
        self.spot_scale_xpos = {}
        self.Bfactor_xpos = {}
        self.Ncells_xstart = {}
        self.Baniso_xstart = {}
        n_nabc = 6 if self._use_cholesky else 3
        for shot_id in self.shot_ids:
            base = self.shot_mapping[shot_id] * self.n_param_per_shot
            self.spot_scale_xpos[shot_id] = base
            self.Bfactor_xpos[shot_id] = base + 1
            self.Ncells_xstart[shot_id] = base + 2
            if self._refining_Baniso:
                self.Baniso_xstart[shot_id] = base + 2 + n_nabc
        LOGGER.info("--0 create an Fcell mapping")
        if self.refine_Fcell:
            self._make_p1_equiv_mapping()

        # Make a mapping of panel id to parameter index and backwards
        self.pid_from_idx = {}
        self.idx_from_pid = {}

        self.x = flex.double(np.ones(self.n_total_params))
        LOGGER.info("--Setting up per shot parameters")

        self.fcell_xstart = self.n_total_shots*self.n_param_per_shot
        self.regions_xstart = self.fcell_xstart + self.n_global_fcell

        self._setup_region_refinement_parameters()
        self._setup_Bfactor_refinement_parameters()
        if self._use_cholesky:
            self._setup_cholesky_refinement_parameters()
        else:
            self._setup_ncells_refinement_parameters()
        if self._refining_Baniso:
            self._setup_Baniso_refinement_parameters()
        self._track_num_times_pixel_was_modeled()

        self._setup_nominal_hkl_p1()
        self._MPI_setup_global_params()
        self._MPI_sync_fcell_parameters()
        # reduce then broadcast fcell
        LOGGER.info("--combining parameters across ranks")
        self._MPI_sync_hkl_freq()  # FIXME does this do absolutely anything?

        if self.x_init is not None:
            LOGGER.info("Initializing with provided x_init array")
            self.x = self.x_init
        elif self.restart_file is not None:
            LOGGER.info("Restarting from parameter file %s" % self.restart_file)
            self.x = flex.double(np.load(self.restart_file)["x"])

        # setup the diffBragg instance
        self.D = self.S.D

        self.D.refine(self._fcell_id)
        if self.params.refiner.refine_Nabc:
            self.D.refine(self._ncells_id)
            if self._use_cholesky:
                self.D.refine(self._ncells_def_id)
        if not self.params.fix.B:
            self.D.refine(BFACTOR_ID)
        if self._refining_Baniso:
            self.D.refine(BFACTOR_ANISO_ID)
        self.D.initialize_managers()

        for sid in self.shot_ids:
            Modeler = self.Modelers[sid]
            Modeler.all_fcell_global_idx = np.array([self.hiasu.to_idx[h] for h in Modeler.hi_asu_perpix])
            Modeler.unique_i_fcell = set(Modeler.all_fcell_global_idx)
            Modeler.i_fcell_slices = self._get_i_fcell_slices(Modeler)
            self.Modelers[sid] = Modeler  # TODO: VERIFY IF THIS IS NECESSARY ?

        if getattr(self.params, 'auto_Fhkl_sigma', False) and self.refine_Fcell:
            self._compute_auto_sigma()

        self._MPI_barrier()
        LOGGER.info("Setup ends!")

    def _track_num_times_pixel_was_modeled(self):
        self.pixel_was_modeled = np.zeros(self.REGIONS.shape)
        self.region_was_modeled = np.zeros(self.num_regions)
        for i_shot in self.shot_ids:
            M = self.Modelers[i_shot]
            self.pixel_was_modeled[M.all_pid, M.all_slow, M.all_fast] += 1
            M._gain_region_per_pixel = self.REGIONS[M.all_pid, M.all_slow, M.all_fast]
            M._unique_gain_regions = set(M._gain_region_per_pixel)
            for i_reg in M._unique_gain_regions:
                self.region_was_modeled[i_reg] += 1

        self.pixel_was_modeled = self._MPI_reduce_broadcast(self.pixel_was_modeled)
        self.region_was_modeled = self._MPI_reduce_broadcast(self.region_was_modeled)

    def _setup_region_refinement_parameters(self):
        self.region_params = {}
        for i_reg in range(self.num_regions):
            minGain, maxGain = self.params.refiner.gain_map_min_max
            if self.params.refiner.gain_restraint is not None:
                center,beta = self.params.refiner.gain_restraint
            else:
                center = 1
                beta = 1e10
            p = RangedParameter(init=1, minval=minGain, maxval=maxGain, sigma=1, center=center, beta=beta)
            p.xpos = self.regions_xstart + i_reg
            p.name = "region%d" % i_reg
            self.region_params[p.name] = p

    def _setup_ncells_refinement_parameters(self):
        names = "Na", "Nb", "Nc"
        for i_shot in self.shot_ids:
            Ncells_params = self.Modelers[i_shot].PAR.Nabc
            for i_n, p in enumerate(Ncells_params):
                p.xpos = self.Ncells_xstart[i_shot] + i_n
                p.name = "%s_shot%d_rank%d" % ( names[i_n], i_shot, COMM.rank)

    def _setup_Bfactor_refinement_parameters(self):
        for i_shot in self.shot_ids:
            p = self.Modelers[i_shot].PAR.B
            p.xpos = self.Bfactor_xpos[i_shot]
            p.name = "Bfactor_shot%d_rank%d" % (i_shot, COMM.rank)

    def _setup_cholesky_refinement_parameters(self):
        names = "chol_L11", "chol_L21", "chol_L22", "chol_L31", "chol_L32", "chol_L33"
        for i_shot in self.shot_ids:
            chol_params = self.Modelers[i_shot].PAR.cholesky
            for i_c, p in enumerate(chol_params):
                p.xpos = self.Ncells_xstart[i_shot] + i_c
                p.name = "%s_shot%d_rank%d" % (names[i_c], i_shot, COMM.rank)

    def _setup_Baniso_refinement_parameters(self):
        names = ["Baniso%d" % i for i in range(6)]
        for i_shot in self.shot_ids:
            baniso_params = self.Modelers[i_shot].PAR.Baniso
            for i_b, p in enumerate(baniso_params):
                p.xpos = self.Baniso_xstart[i_shot] + i_b
                p.name = "%s_shot%d_rank%d" % (names[i_b], i_shot, COMM.rank)

    def _gain_restraints(self):
        if self.params.refiner.gain_restraint:
            for p in self.region_params.values():
                self.target_functional += p.get_restraint_val(self.x[p.xpos])
                self.grad[p.xpos] += p.get_restraint_deriv(self.x[p.xpos])

    def _get_i_fcell_slices(self, Modeler):
        """finds the boundaries for each fcell in the 1-D array of per-shot data"""
        # TODO move this to Data Modeler class ?
        splitter = np.where(np.diff(Modeler.all_fcell_global_idx) != 0)[0]+1
        npix = len(Modeler.all_fcell_global_idx)
        slices = [slice(V[0], V[-1]+1, 1) for V in np.split(np.arange(npix), splitter)]
        i_fcells = [V[0] for V in np.split(Modeler.all_fcell_global_idx, splitter)]
        i_fcell_slices = {}
        for i_fcell, slc in zip(i_fcells, slices):
            if i_fcell not in i_fcell_slices:
                i_fcell_slices[i_fcell] = [slc]
            else:
                i_fcell_slices[i_fcell].append(slc)
        return i_fcell_slices

    def _get_shot_mapping(self):
        """each modeled shot maps to an integer along interval [0,Nshots) """
        all_shot_ids = COMM.gather(self.shot_ids)
        shot_mapping = None
        if COMM.rank == 0:
            unique_shot_ids = set([sid for shot_ids in all_shot_ids for sid in shot_ids])
            shot_mapping = {shot_id: i_shot for i_shot, shot_id in enumerate(unique_shot_ids)}
        shot_mapping = COMM.bcast(shot_mapping)
        return shot_mapping

    def _make_p1_equiv_mapping(self):
        self.num_equivs_for_i_fcell = {}
        self.update_indices = []
        for i_fcell in range(self.n_global_fcell):
            hkl_asu = self.hiasu.from_idx[i_fcell]

            equivs = [i.h() for i in miller.sym_equiv_indices(self.space_group, hkl_asu).indices()]
            self.num_equivs_for_i_fcell[i_fcell] = len(equivs)
            self.update_indices += equivs
        self.update_indices = flex.miller_index(self.update_indices)

    def _MPI_setup_global_params(self):
        if self.I_AM_ROOT:
            LOGGER.info("--2 Setting up global parameters")
            if self.output_dir is not None:
                np.save(os.path.join(self.output_dir, "f_asu_map"), self.hiasu.from_idx)

            self._setup_fcell_params()

    def _setup_nominal_hkl_p1(self):
        Omatrix = np.reshape(self.S.crystal.Omatrix.elems, [3, 3])
        for i_shot in self.Modelers:
            MOD = self.Modelers[i_shot]
            nom_h = MOD.all_nominal_hkl
            nom_h_p1 = np.dot(nom_h, Omatrix).astype(np.int32)
            nom_h_p1 = list(map(tuple, nom_h_p1))
            self.Modelers[i_shot].all_nominal_hkl_p1 = nom_h_p1

    def _setup_fcell_params(self):
        if self.refine_Fcell:
            LOGGER.info("----loading fcell data")
            # this is the number of observations of hkl (accessed like a dictionary via global_fcell_index)
            LOGGER.info("---- -- counting hkl totes")
            LOGGER.info("compute HKL multiplicity")
            self.hkl_frequency = self.hiasu.present_idx_counter
            LOGGER.info("save HKL multiplicity")
            np.save(os.path.join(self.output_dir, "f_asu_multi"), self.hkl_frequency)
            LOGGER.info("Done ")

            LOGGER.info("local refiner symbol=%s ; nanoBragg crystal symbol: %s" % (self.symbol, self.S.crystal.symbol))
            self.fcell_init_from_i_fcell = []
            ma = self.S.crystal.miller_array
            LOGGER.info("make an Fhkl map")
            ma_map = {h: d for h,d in zip(ma.indices(), ma.data())}
            Omatrix = np.reshape(self.S.crystal.Omatrix.elems,[3,3])

            # TODO: Vectorize
            for i_fcell in range(self.n_global_fcell):
                asu_hkl = self.hiasu.from_idx[i_fcell]  # high symmetry
                P1_hkl = tuple(np.dot(Omatrix, asu_hkl).astype(int))
                fcell_val = ma_map[P1_hkl]
                self.fcell_init_from_i_fcell.append(fcell_val)
            self.fcell_init_from_i_fcell = np.array(self.fcell_init_from_i_fcell)

            self.fcell_sigmas_from_i_fcell = self.params.sigmas.Fhkl
            LOGGER.info("DONE make fcell_init")

    def _compute_auto_sigma(self):
        """Compute per-Fhkl sigma from diagonal Hessian across all shots and MPI ranks.

        Aggregates curvature contributions from each shot, reduces across MPI,
        then sets self.fcell_sigmas_from_i_fcell = 1/sqrt(|H_ii|) (rescaled).
        """
        LOGGER.info("Computing auto Fhkl sigma from Hessian diagonal across all shots")
        nfcell = self.n_global_fcell
        hessian_accum = np.zeros(nfcell)

        # Iterate over shots, accumulate Hessian diagonal for each Fhkl index
        for i_shot in self.shot_ids:
            Mod = self.Modelers[i_shot]
            gains = self.REGIONS[Mod.all_pid, Mod.all_slow, Mod.all_fast]
            # Use initial gain = 1 for sigma estimation
            scale = Mod.PAR.Scale.init ** 2  # squared because Scale stores sqrt

            # Set up SIM state for this shot
            self._i_shot = i_shot
            self._update_beams()
            self._update_umatrix()
            self._update_ucell()
            self._update_ncells()
            self._update_ncells_def()
            self._update_eta()
            self._update_Bfactor()
            self._update_Baniso()
            self._update_dxtbx_detector()

            # Run forward model
            pfs = Mod.pan_fast_slow
            self.D.add_diffBragg_spots(pfs)
            npix = len(Mod.all_data)
            model_pix = self.D.raw_pixels_roi[:npix].as_numpy_array()
            model_bragg = scale * model_pix
            model_lambda = Mod.all_background + model_bragg
            resid = Mod.all_data - model_lambda
            V = model_lambda + Mod.nominal_sigma_rdout ** 2

            # Compute per-Fhkl Hessian contribution from this shot
            # For each unique i_fcell in this shot, accumulate |dI/dF|^2 / V
            dF = self.D.get_derivative_pixels(self._fcell_id)
            dF_arr = dF[:npix].as_numpy_array()
            dF_scaled = scale * dF_arr

            for i_fcell in Mod.unique_i_fcell:
                for slc in Mod.i_fcell_slices[i_fcell]:
                    trust = Mod.all_trusted[slc]
                    v_slc = V[slc][trust]
                    d_slc = dF_scaled[slc][trust]
                    # Diagonal Hessian ≈ sum(d^2/v)
                    hessian_accum[i_fcell] += np.sum(d_slc**2 / v_slc)

            self.D.raw_pixels_roi *= 0

        # MPI reduce Hessian across ranks
        hessian_accum = self._MPI_reduce_broadcast(hessian_accum)

        # Convert to sigmas
        abs_hess = np.abs(hessian_accum)
        has_signal = abs_hess > 1e-12
        default_sigma = float(self.params.sigmas.Fhkl)
        sigmas = np.full(nfcell, default_sigma)
        raw_sigmas = np.full(nfcell, 0.0)
        raw_sigmas[has_signal] = 1.0 / np.sqrt(abs_hess[has_signal])

        if self.log_fcells:
            # For log parameterization: dI/dx = sigma*F*dI/dF
            # Already accounted for in get_deriv chain rule
            pass

        sigmas[has_signal] = raw_sigmas[has_signal]

        # Rescale to match default sigma magnitude
        if has_signal.any():
            median_observed = float(np.median(sigmas[has_signal]))
            scale_factor = default_sigma / median_observed
            sigmas[has_signal] *= scale_factor
            if self.I_AM_ROOT:
                LOGGER.info("Fhkl auto-sigma: rescale=%.4g (default=%.4g / median=%.4g)"
                            % (scale_factor, default_sigma, median_observed))

        sigmas = np.clip(sigmas, 1e-4, 100)
        self.fcell_sigmas_from_i_fcell = sigmas

        if self.I_AM_ROOT:
            LOGGER.info("Fhkl auto-sigma: min=%.4g, max=%.4g, median=%.4g, signal=%d/%d"
                         % (sigmas.min(), sigmas.max(), np.median(sigmas),
                            has_signal.sum(), nfcell))

    def _get_sausage_parameters(self, i_shot):
        pass

    def _get_rotXYZ(self, i_shot):
        vals = [self.Modelers[i_shot].RotXYZ[i_rot].init for i_rot in range(3)]
        return vals

    def _get_rotX(self, i_shot):
        pass

    def _get_rotY(self, i_shot):
        pass

    def _get_rotZ(self, i_shot):
        pass

    def _get_spectra_coefficients(self):
        pass

    def _get_ucell_vars(self, i_shot):
        vars = []
        for i in range(self.n_ucell_param):
            var = self.Modelers[i_shot].PAR.ucell[i].init
            vars.append(var)
        return vars

    def _get_panelRot_val(self, panel_id):
        pass

    def _get_panelXYZ_val(self, panel_id, i_shot=0):
        pass

    def _get_detector_distance_val(self, i_shot):
        return self.Modelers[i_shot].PAR.detz_shift.init

    def _get_ncells_def(self, i_shot):
        vals = [self.Modelers[i_shot].PAR.Ndef[i_N].init for i_N in range(3)]
        return vals

    def _get_ncells_abc(self, i_shot):
        if self.params.refiner.refine_Nabc:
            vals = []
            Nabc_p = self.Modelers[i_shot].PAR.Nabc
            for p in Nabc_p:
                xval = self.x[p.xpos]
                val = p.get_val(xval)
                vals.append(val)
        else:
            vals = [self.Modelers[i_shot].PAR.Nabc[i_N].init for i_N in range(3)]

        return vals

    def _get_eta(self, i_shot):
        # NOTE: refinement of eta not supported in this script
        vals = [self.Modelers[i_shot].PAR.eta[i_eta].init for i_eta in range(3)]
        return vals

    def _get_spot_scale(self, i_shot):
        xval = self.x[self.spot_scale_xpos[i_shot]]
        PAR = self.Modelers[i_shot].PAR
        sig = PAR.Scale.sigma
        init = PAR.Scale.init
        val = sig*(xval-1) + init
        return val

    def _get_bfactor(self, i_shot):
        PAR = self.Modelers[i_shot].PAR
        return PAR.B.get_val(self.x[PAR.B.xpos])

    def _get_bg_vals(self, i_shot, i_spot):
        pass

    def _send_ucell_gradients_to_derivative_managers(self):
        """Needs to be called once each time the orientation is updated"""
        pass

    def _run_diffBragg_current(self):
        LOGGER.info("run diffBragg for shot %d" % self._i_shot)
        pfs = self.Modelers[self._i_shot].pan_fast_slow
        if self.use_nominal_h:
            nom_h_p1 = self.Modelers[self._i_shot].all_nominal_hkl_p1
            self.D.add_diffBragg_spots(pfs, nom_h_p1)
        else:
            self.D.add_diffBragg_spots(pfs)
        LOGGER.info("finished diffBragg for shot %d" % self._i_shot)

    def _store_updated_Fcell(self):
        if not self.refine_Fcell:
            return
        xvals = self.x[self.fcell_xstart: self.fcell_xstart+self.n_global_fcell]
        if self.rescale_params and self.log_fcells:
            sigs = self.fcell_sigmas_from_i_fcell
            inits = self.fcell_init_from_i_fcell
            if self.log_fcells:
                vals = np.exp(sigs*(xvals - 1))*inits
            else:
                vals = sigs*(xvals - 1) + inits
                vals[vals < 0] = 0
        else:
            if self.log_fcells:
                vals = np.exp(xvals)
            else:
                vals = xvals
                vals [vals < 0] = 0
        self._fcell_at_i_fcell = vals

    def _update_Fcell(self):
        if not self.refine_Fcell:
            return
        update_amps = []
        for i_fcell in range(self.n_global_fcell):
            new_Fcell_amplitude = self._fcell_at_i_fcell[i_fcell]
            update_amps += [new_Fcell_amplitude] * self.num_equivs_for_i_fcell[i_fcell]

        update_amps = flex.double(update_amps)
        #hkl_min = self.S.D.h_min, self.S.D.k_min, self.S.D.l_min
        #hkl_range = self.S.D.h_range, self.S.D.k_range, self.S.D.l_range
        #hkl_min = COMM.gather(hkl_min)
        #hkl_range = COMM.gather(hkl_range)
        #if COMM.rank==0:
        #    assert len(set(hkl_min))==1
        #    assert len(set(hkl_range)) == 1
        self.S.D.quick_Fhkl_update((self.update_indices, update_amps))

    def _update_spectra_coefficients(self):
        pass

    def _update_eta(self):

        if self.S.umat_maker is not None:
            eta_vals = self._get_eta(self._i_shot)

            if not self.D.has_anisotropic_mosaic_spread:
                assert self.S.Umats_method == 2
                assert len(set(eta_vals))==1
                eta_vals = eta_vals[0]

            LOGGER.info("eta=%f" % eta_vals)
            self.S.update_umats_for_refinement(eta_vals)

    def _symmetrize_Flatt(self):
        if self.params.symmetrize_Flatt:
            # NOTE: RotXYZ refinement disabled for this script, so offsets always 0,0,0
            RXYZU = hopper_io.diffBragg_Umat(0,0,0,self.D.Umatrix)
            Cryst = deepcopy(self.S.crystal.dxtbx_crystal)
            B_realspace = self.get_refined_Bmatrix(self._i_shot, recip=False)
            A = RXYZU * B_realspace
            A_recip = A.inverse().transpose()
            Cryst.set_A(A_recip)
            symbol = self.S.crystal.space_group_info.type().lookup_symbol()
            self.D.set_mosaic_blocks_sym(Cryst, symbol , self.params.simulator.crystal.num_mosaicity_samples,
                                        refining_eta=False) # NOTE:no eta refinement in this stage 2 script (possible in ens.hopper)

    def _set_background_plane(self):
        self.tilt_plane = self.Modelers[self._i_shot].all_background[self.roi_sel]

    def _update_sausages(self):
        pass

    def _update_rotXYZ(self):
        pass

    def _update_ncells(self):
        Mod = self.Modelers[self._i_shot]
        if self._use_cholesky:
            chol = Mod.PAR.cholesky
            L11, L21, L22, L31, L32, L33 = [p.get_val(self.x[p.xpos]) for p in chol]
            Na = L11**2
            Nb = L21**2 + L22**2
            Nc = L31**2 + L32**2 + L33**2
            Nd = L11*L21
            Ne = L21*L31 + L22*L32
            Nf = L11*L31
            self.D.set_ncells_values((Na, Nb, Nc))
            self.D.Ncells_def = Nd, Ne, Nf
            self._chol_vals = (L11, L21, L22, L31, L32, L33)
        else:
            vals = self._get_ncells_abc(self._i_shot)
            self.D.set_ncells_values(tuple(vals))

    def _update_ncells_def(self):
        if self._use_cholesky:
            return  # already set in _update_ncells
        vals = self._get_ncells_def(self._i_shot)
        self.D.Ncells_def = tuple(vals)

    def _update_Bfactor(self):
        if not self.params.fix.B:
            self.D.Bfactor_image = self._get_bfactor(self._i_shot)

    def _update_Baniso(self):
        if not self._refining_Baniso:
            return
        Mod = self.Modelers[self._i_shot]
        vals = tuple(p.get_val(self.x[p.xpos]) for p in Mod.PAR.Baniso)
        self.D.Bfactor_aniso = vals

    def _update_dxtbx_detector(self):
        shiftZ = self._get_detector_distance_val(self._i_shot)
        self.S.D.shift_origin_z(self.S.detector,  shiftZ)

    def _extract_spectra_coefficient_derivatives(self):
        pass

    def _pre_extract_deriv_arrays(self):
        npix = len(self.Modelers[self._i_shot].all_data)
        self._model_pix = self.D.raw_pixels_roi[:npix].as_numpy_array()

        if self.refine_Fcell:
            dF = self.D.get_derivative_pixels(self._fcell_id)
            self._extracted_fcell_deriv = dF[:npix].as_numpy_array()
            if self.calc_curvatures:
                d2F = self.D.get_second_derivative_pixels(self._fcell_id)
                self._extracted_fcell_second_deriv = d2F[:npix].as_numpy_array()

        if self.params.refiner.refine_Nabc:
            self.dNabc = [d[:npix].as_numpy_array() for d in self.D.get_ncells_derivative_pixels()]
            if self._use_cholesky:
                self.dNdef = [d[:npix].as_numpy_array() for d in self.D.get_ncells_def_derivative_pixels()]
            if self.calc_curvatures:
                raise NotImplementedError("update the code")

        if not self.params.fix.B:
            self.dBfactor = self.D.get_Bfactor_derivative_pixels().as_numpy_array()[:npix]

        if self._refining_Baniso:
            self.dBaniso = [d[:npix].as_numpy_array() for d in self.D.get_Bfactor_aniso_derivative_pixels()]

    def _extract_sausage_derivs(self):
        pass

    def _extract_Umatrix_derivative_pixels(self):
        pass

    def _extract_Bmatrix_derivative_pixels(self):
        pass

    def _extract_ncells_def_derivative_pixels(self):
        pass

    def _extract_mosaic_parameter_m_derivative_pixels(self):
        pass

    def _extract_detector_distance_derivative_pixels(self):
        pass

    def _extract_panelRot_derivative_pixels(self):
        pass

    def _extract_panelXYZ_derivative_pixels(self):
        pass

    def _scale_Fcell_derivative_pixels(self):
        self.fcell_deriv = self.fcell_second_deriv = 0
        if self.refine_Fcell:
            SG = self.scale_fac
            self.fcell_deriv = SG*(self._extracted_fcell_deriv)
            # handles Nan's when Fcell is 0 for whatever reason
            if self.calc_curvatures:
                self.fcell_second_deriv = SG*self._extracted_fcell_second_deriv

    def _scale_Nabc_derivative_pixels(self):
        if self.params.refiner.refine_Nabc:
            self.dNabc = [self.scale_fac*d for d in self.dNabc]
            if self._use_cholesky:
                self.dNdef = [self.scale_fac*d for d in self.dNdef]

    def _scale_Bfactor_derivative_pixels(self):
        if not self.params.fix.B:
            self.dBfactor = self.scale_fac * self.dBfactor

    def _scale_Baniso_derivative_pixels(self):
        if self._refining_Baniso:
            self.dBaniso = [self.scale_fac*d for d in self.dBaniso]

    def _get_per_spot_scale(self, i_shot, i_spot):
        pass

    def _scale_pixel_data(self):
        self.model_bragg_spots_no_gains = self.scale_fac_no_gains*self._model_pix
        self.model_bragg_spots = self.scale_fac*self._model_pix
        self._scale_Fcell_derivative_pixels()
        self._scale_Nabc_derivative_pixels()
        self._scale_Bfactor_derivative_pixels()
        self._scale_Baniso_derivative_pixels()

    def _update_ucell(self):
        self.D.Bmatrix = self.Modelers[self._i_shot].PAR.Bmatrix

    def _update_umatrix(self):
        self.D.Umatrix = self.Modelers[self._i_shot].PAR.Umatrix

    def _update_beams(self):
        # sim_data instance has a nanoBragg beam object, which takes spectra and converts to nanoBragg xray_beams
        self.S.beam.spectrum = self.Modelers[self._i_shot].spectra
        self.D.xray_beams = self.S.beam.xray_beams

    def _get_panels_fasts_slows(self):
        pass

    def _set_current_gain_correction_map(self):
        self._gain_per_region = np.zeros(self.num_regions)
        for i_reg in range(self.num_regions):
            gain_x = self.x[self.regions_xstart+i_reg]
            gain = self.region_params["region%d" % i_reg].get_val(gain_x)
            self._gain_per_region[i_reg] = gain

    def compute_functional_gradients_diag(self):
        self.compute_functional_and_gradients()
        return self._f, self._g, self.d

    def compute_functional_and_gradients(self):
        t = time.time()
        out = self._compute_functional_and_gradients()
        t = time.time()-t
        LOGGER.info("Took %.4f sec to compute functional and grad" % t)
        return out

    def _compute_functional_and_gradients(self):
        LOGGER.info(Bcolors.OKBLUE+"BEGIN FUNC GRAD ; Eval %d" % self.target_eval_count+Bcolors.ENDC)
        #if self.verbose:
        #    self._print_iteration_header()

        self.target_functional = 0

        self.grad = flex.double(self.n_total_params)
        if self.calc_curvatures:
            self.curv = flex.double(self.n_total_params)

        LOGGER.info("start update Fcell")
        self._store_updated_Fcell()
        self._update_Fcell()  # update the structure factor with the new x
        LOGGER.info("done update Fcell")
        self._MPI_save_state_of_refiner()
        self._update_spectra_coefficients()  # updates the diffBragg lambda coefficients if refinining spectra

        # get the gain correction image?
        self._set_current_gain_correction_map()

        tshots = time.time()

        LOGGER.info("Iterate over %d shots" % len(self.shot_ids))
        self._shot_Zscores = []
        save_model = self.save_model_freq is not None and self.target_eval_count % self.save_model_freq == 0
        if save_model:
            self._save_model_dir = os.path.join(self.model_dir, "eval%d" % self.target_eval_count)

            if self.params.debug_mode and COMM.rank == 0 and not os.path.exists(self._save_model_dir):
                os.makedirs(self._save_model_dir)
            COMM.barrier()

        if self.target_eval_count % self.params.refiner.save_gain_freq == 0:
            self._save_optimized_gain_map()

        self.all_sigZ = []

        for self._i_shot in self.shot_ids:
            self._set_current_gain_per_pixel()
            gains = self.Modelers[self._i_shot].all_gain
            self.scale_fac_no_gains = self._get_spot_scale(self._i_shot)**2
            self.scale_fac = gains*self._get_spot_scale(self._i_shot)**2

            self.b_fac = self._get_bfactor(self._i_shot)

            # TODO: Omatrix update? All crystal models here should have the same to_primitive operation, ideally
            self._update_beams()
            self._update_umatrix()
            self._update_ucell()
            self._update_ncells()
            self._update_ncells_def()
            self._update_rotXYZ()
            self._update_eta()  # mosaic spread
            self._update_Bfactor()
            self._update_Baniso()
            self._symmetrize_Flatt()
            self._update_dxtbx_detector()
            self._update_sausages()

            self._run_diffBragg_current()

            # CHECK FOR SIGNAL INTERRUPT HERE
            if self.break_signal is not None:
                signal.signal(self.break_signal, self._sig_hand.handle)
                self._MPI_check_for_break_signal()

            # TODO pre-extractions for all parameters
            self._pre_extract_deriv_arrays()
            self._scale_pixel_data()
            self._evaluate_averageI()
            self._evaluate_log_averageI_plus_sigma_readout()

            self._derivative_convenience_factors()

            if self.saveZ_freq is not None and self.target_eval_count % self.saveZ_freq == 0:
                MOD = self.Modelers[self._i_shot]
                self._spot_Zscores = []
                for i_fcell in MOD.unique_i_fcell:
                    for slc in MOD.i_fcell_slices[i_fcell]:
                        sigZ = self._Zscore[slc]
                        trus = MOD.all_trusted[slc]
                        sigZ = sigZ[trus].std()
                        self._spot_Zscores.append((i_fcell, sigZ))
                self._shot_Zscores.append(self._spot_Zscores)

            if save_model:
                MOD = self.Modelers[self._i_shot]
                P = MOD.all_pid
                F = MOD.all_fast
                S = MOD.all_slow
                #G = MOD.all_gain
                M = self.model_Lambda
                B = MOD.all_background
                D = MOD.all_data
                C = self.model_bragg_spots
                Z = self._Zscore
                iF = MOD.all_fcell_global_idx
                iROI = MOD.roi_id
                trust = MOD.all_trusted

                model_info = {"p": P, "f": F, "s": S, "model": M,
                        "background": B, "data": D, "bragg": C,
                        "Zscore": Z, "i_fcell": iF, "trust": trust,
                        "i_roi": iROI}
                self._save_model(model_info)
            self._is_trusted = self.Modelers[self._i_shot].all_trusted
            self.target_functional += self._target_accumulate()
            self._spot_scale_derivatives()
            self._Bfactor_derivatives()
            self._Baniso_derivatives()
            if self._use_cholesky:
                self._accumulate_cholesky_derivatives()
            else:
                self._accumulate_Nabc_derivatives()
            self._Fcell_derivatives()
            self._gain_region_derivatives()

            trusted = self.Modelers[self._i_shot].all_trusted
            self.all_sigZ.append(np.std(self._Zscore[trusted]))
        tshots = time.time()-tshots
        LOGGER.info("Time rank worked on shots=%.4f" % tshots)
        self._MPI_barrier()
        tmpi = time.time()
        LOGGER.info("MPI aggregation of func and grad")
        self._mpi_aggregation()
        tmpi = time.time() - tmpi
        LOGGER.info("Time for MPIaggregation=%.4f" % tmpi)

        self._gain_restraints()

        LOGGER.info("Aliases")
        self._f = self.target_functional
        self._g = self.g = self.grad
        self.d = self.curv
        LOGGER.info("curvature analysis")
        self._curvature_analysis()

        # reset ROI pixels TODO: is this necessary
        LOGGER.info("Zero pixels")
        self.D.raw_pixels_roi *= 0
        self.gnorm = -1

        tsave = time.time()
        LOGGER.info("DUMP param and Zscore data")
        self._save_Zscore_data()
        tsave = time.time()-tsave
        LOGGER.info("Time to dump param and Zscore data: %.4f" % tsave)

        self.target_eval_count += 1
        self.f_vals.append(self.target_functional)

        if self.calc_curvatures and not self.use_curvatures:
            if self.num_positive_curvatures == self.use_curvatures_threshold:
                raise BreakToUseCurvatures

        LOGGER.info("DONE WITH FUNC GRAD")
        return self._f, self._g

    def callback_after_step(self, minimizer):
        self.iterations = minimizer.iter()

    def _save_model(self, model_info):
        LOGGER.info("SAVING MODEL FOR SHOT %d" % self._i_shot)
        df = pandas.DataFrame(model_info)
        df["shot_id"] = self._i_shot
        outdir = self._save_model_dir
        outname = os.path.join(outdir, "rank%d_shot%d_EVAL%d_ITER%d.pkl" % (COMM.rank, self._i_shot, self.target_eval_count, self.iterations))
        df.to_pickle(outname)

    def _save_Zscore_data(self):
        if self.saveZ_freq is None or not self.target_eval_count % self.saveZ_freq == 0:
            return
        outdir = os.path.join(self.Zdir, "rank%d_Zscore" % self.rank)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        fname = os.path.join(outdir, "sigZ_eval%d_iter%d_rank%d" % (self.target_eval_count, self.iterations, self.rank))
        np.save(fname, np.array(self._shot_Zscores, object))

    def _sanity_check_grad(self):
        pass

    def _gain_region_derivatives(self):
        if not self.params.refiner.refine_gain_map:
            return
        MOD = self.Modelers[self._i_shot]

        #dL_dG = 0.5*self.one_over_v* (MOD.all_background + 2*self.u*(MOD.all_data-MOD.all_background) - \
        #    self.u*self.u*MOD.all_background*self.one_over_v)
        dL_dG = 0.5*self.model_bragg_spots_no_gains*self.common_grad_term
        dL_dG /= MOD.all_freq

        reg_grad = np.zeros(self.num_regions)
        np.add.at(reg_grad, MOD._gain_region_per_pixel[MOD.all_trusted], dL_dG[MOD.all_trusted])

        #u_reg = set(MOD._gain_region_per_pixel)
        for i_reg in MOD._unique_gain_regions:
            xpos = self.regions_xstart+i_reg
            gain_x = self.x[xpos]
            d = self.region_params["region%d"%i_reg].get_deriv(gain_x, reg_grad[i_reg])
            self.grad[xpos] += d
        #self.grad[self.regions_xstart:self.regions_xstart+self.num_regions] += reg_grad

    def _Fcell_derivatives(self):
        if not self.refine_Fcell:
            return
        MOD = self.Modelers[self._i_shot]
        dumps = []
        for i_fcell in MOD.unique_i_fcell:

            multi = self.hkl_frequency[i_fcell]
            if multi < self.min_multiplicity:
                continue

            xpos = self.fcell_xstart + i_fcell
            Famp = self._fcell_at_i_fcell[i_fcell]
            sig = 1

            for slc in MOD.i_fcell_slices[i_fcell]:
                self.fcell_dI_dtheta = self.fcell_deriv[slc]

                if self.log_fcells:
                    # case 2 rescaling
                    sig_times_fcell = sig*Famp
                    d = sig_times_fcell*self.fcell_dI_dtheta
                else:
                    # case 1 rescaling
                    d = sig*self.fcell_dI_dtheta

                gterm = self.common_grad_term[slc]
                g_accum = d*gterm
                trust = MOD.all_trusted[slc]
                # NOTE : no need to normalize Fhkl gradients by the overlap rate - they should arise from different HKLs
                #freq = MOD.all_freq[slc]  # pixel frequency (1 is no overlaps)
                dump = (g_accum[trust].sum())*.5
                self.grad[xpos] += dump
                dumps.append(dump)
                if self.calc_curvatures:
                    raise NotImplementedError("No curvature for Fcell refinement")

    def _accumulate_Nabc_derivatives(self):
        if not self.params.refiner.refine_Nabc:
            return
        Mod = self.Modelers[self._i_shot]
        for i_n in range(3):
            p = Mod.PAR.Nabc[i_n]
            d = p.get_deriv(self.x[p.xpos],  self.dNabc[i_n])
            self.grad[p.xpos] += self._grad_accumulate(d)

    def _spot_scale_derivatives(self, return_derivatives=False):
        if not self.refine_crystal_scale:
            return
        S = np.sqrt(self.scale_fac_no_gains)
        dI_dtheta = (2./S)*self.model_bragg_spots
        d2I_dtheta2 = (2./S/S)*self.model_bragg_spots
        # second derivative is 0 with respect to scale factor
        sig = self.Modelers[self._i_shot].PAR.Scale.sigma
        d = dI_dtheta*sig
        d2 = d2I_dtheta2 *(sig**2)

        xpos = self.spot_scale_xpos[self._i_shot]
        self.grad[xpos] += self._grad_accumulate(d)
        if self.calc_curvatures:
            self.curv[xpos] += self._curv_accumulate(d, d2)

        if return_derivatives:
            return d, d2

    def _Bfactor_derivatives(self):
        if self.params.fix.B:
            return
        Mod = self.Modelers[self._i_shot]
        d = Mod.PAR.B.get_deriv(self.x[Mod.PAR.B.xpos], self.dBfactor)
        self.grad[self.Bfactor_xpos[self._i_shot]] += self._grad_accumulate(d)

    def _Baniso_derivatives(self):
        if not self._refining_Baniso:
            return
        Mod = self.Modelers[self._i_shot]
        for i_ba in range(6):
            p = Mod.PAR.Baniso[i_ba]
            d = p.get_deriv(self.x[p.xpos], self.dBaniso[i_ba])
            self.grad[p.xpos] += self._grad_accumulate(d)

    def _accumulate_cholesky_derivatives(self):
        if not self.params.refiner.refine_Nabc:
            return
        Mod = self.Modelers[self._i_shot]
        L11, L21, L22, L31, L32, L33 = self._chol_vals
        # Chain rule: dI/dLij from dI/dN{abc} and dI/dN{def}
        dI_dL = [
            self.dNabc[0]*2*L11 + self.dNdef[0]*L21 + self.dNdef[2]*L31,  # dI/dL11
            self.dNdef[0]*L11 + self.dNabc[1]*2*L21 + self.dNdef[1]*L31,  # dI/dL21
            self.dNabc[1]*2*L22 + self.dNdef[1]*L32,                       # dI/dL22
            self.dNdef[2]*L11 + self.dNdef[1]*L21 + self.dNabc[2]*2*L31,  # dI/dL31
            self.dNdef[1]*L22 + self.dNabc[2]*2*L32,                       # dI/dL32
            self.dNabc[2]*2*L33,                                            # dI/dL33
        ]
        for i_chol in range(6):
            p = Mod.PAR.cholesky[i_chol]
            d = p.get_deriv(self.x[p.xpos], dI_dL[i_chol])
            self.grad[p.xpos] += self._grad_accumulate(d)

    def _mpi_aggregation(self):
        # reduce the broadcast summed results:
        LOGGER.info("aggregate barrier")
        self._MPI_barrier()
        LOGGER.info("Functional")
        self.target_functional = self._MPI_reduce_broadcast(self.target_functional)
        LOGGER.info("gradients")
        self.grad = self._MPI_reduce_broadcast(self.grad)
        if self.calc_curvatures:
            self.curv = self._MPI_reduce_broadcast(self.curv)
        all_sigZ = COMM.reduce(self.all_sigZ)
        if COMM.rank==0:
            LOGGER.info("F=%10.7e, sigmaZ: mean=%f, median=%f" % (self.target_functional, np.mean(all_sigZ), np.median(all_sigZ) ))

    def _curvature_analysis(self):
        self.tot_neg_curv = 0
        self.neg_curv_shots = []
        if self.calc_curvatures:
            self.is_negative_curvature = self.curv.as_numpy_array() < 0
            self.tot_neg_curv = sum(self.is_negative_curvature)

        if self.calc_curvatures and not self.use_curvatures:
            if self.tot_neg_curv == 0:
                self.num_positive_curvatures += 1
                self.d = self.curv
                self._verify_diag()
            else:
                self.num_positive_curvatures = 0
                self.d = None

        if self.use_curvatures:
            assert self.tot_neg_curv == 0
            self.request_diag_once = False
            self.diag_mode = "always"  # TODO is this proper place to set ?
            self.d = self.curv
            self._verify_diag()
        else:
            self.d = None

    def _get_refinement_string_label(self):
        refine_str = "refining "
        if self.refine_Fcell:
            refine_str += "fcell, "
        if self.refine_ncells:
            refine_str += "Ncells, "
        if self.refine_ncells_def:
            refine_str += "Ncells_def, "
        if self.refine_Bmatrix:
            refine_str += "Bmat, "
        if self.refine_Umatrix:
            refine_str += "Umat, "
        if self.refine_crystal_scale:
            refine_str += "scale, "
        if self.refine_background_planes:
            refine_str += "bkgrnd, "
        if self.refine_detdist:
            refine_str += "detector_distance, "
        if self.refine_panelRotO:
            refine_str += "panelRotO, "
        if self.refine_panelRotF:
            refine_str += "panelRotF, "
        if self.refine_panelRotS:
            refine_str += "panelRotS, "
        if self.refine_panelXY:
            refine_str += "panelXY, "
        if self.refine_panelZ:
            refine_str += "panelZ, "
        if self.refine_lambda0:
            refine_str += "Lambda0 (offset), "
        if self.refine_lambda1:
            refine_str += "Lambda1 (scale), "
        if self.refine_per_spot_scale:
            refine_str += "Per-spot scales, "
        if self.refine_eta:
            refine_str += "Eta, "
        if self.refine_blueSausages:
            refine_str += "Mosaic texture, "
        return refine_str

    def _print_iteration_header(self):
        refine_str = self._get_refinement_string_label()
        border = "<><><><><><><><><><><><><><><><>"
        if self.use_curvatures:

            LOGGER.info(
                "%s%s%s%s\nTrial%d (%s): Compute functional and gradients eval %d %s(Using Curvatures)%s\n%s%s%s%s"
                % (Bcolors.HEADER, border,border,border, self.trial_id + 1, refine_str, self.target_eval_count + 1, Bcolors.OKGREEN, Bcolors.HEADER, border,border,border, Bcolors.ENDC))
        else:
            LOGGER.info("%s%s%s%s\n, Trial%d (%s): Compute functional and gradients eval %d PosCurva %d\n%s%s%s%s"
                  % (Bcolors.HEADER, border, border, border, self.trial_id + 1, refine_str, self.target_eval_count + 1, self.num_positive_curvatures, border, border,border, Bcolors.ENDC))

    def _save_optimized_gain_map(self):
        if not self.params.refiner.refine_gain_map:
            return
        if self.I_AM_ROOT and self.output_dir is not None:
            outf = os.path.join(self.output_dir, "gain_map" )
            LOGGER.info(Bcolors.WARNING+"Saving detector gain map!"+Bcolors.ENDC)
            np.savez(outf, gain_per_region=self._gain_per_region, region_shape=self.params.refiner.region_size,
                     det_shape=self.REGIONS.shape, adu_per_photon=self.params.refiner.adu_per_photon,
                     regions=self.REGIONS,
                     num_times_pixel_was_modeled=self.pixel_was_modeled,
                     num_times_region_was_modeled=self.region_was_modeled)
            LOGGER.info("Done Saving detector gain map!")

    def _MPI_save_state_of_refiner(self):
        if self.I_AM_ROOT and self.output_dir is not None and self.refine_Fcell:
            outf = os.path.join(self.output_dir, "_fcell_trial%d_eval%d_iter%d" % (self.trial_id, self.target_eval_count, self.iterations))
            np.savez(outf, fvals=self._fcell_at_i_fcell)

    def _target_accumulate(self):
        fterm = self.log2pi + self.log_v + self.u*self.u*self.one_over_v
        M = self.Modelers[self._i_shot]
        fterm /= M.all_freq
        if self._is_trusted is not None:
            fterm = fterm[self._is_trusted]
        fterm = 0.5*(fterm.sum())
        return fterm

    def _grad_accumulate(self, d):
        gterm = d * self.one_over_v * self.one_minus_2u_minus_u_squared_over_v
        M = self.Modelers[self._i_shot]
        gterm /= M.all_freq
        if self._is_trusted is not None:
            gterm = gterm[self._is_trusted]
        gterm = 0.5*gterm.sum()
        return gterm

    def _curv_accumulate(self, d, d2):
        cterm = self.one_over_v * (d2*self.one_minus_2u_minus_u_squared_over_v -
                                   d*d*(self.one_over_v_times_one_minus_2u_minus_u_squared_over_v -
                                        (2 + 2*self.u_times_one_over_v + self.u_u_one_over_v*self.one_over_v)))
        if self._is_trusted is not None:
            cterm = cterm[self._is_trusted]
        cterm = .5 * (cterm.sum())
        return cterm

    def _derivative_convenience_factors(self):
        Mod = self.Modelers[self._i_shot]
        self.u = Mod.all_data - self.model_Lambda
        self.one_over_v = 1. / (self.model_Lambda + Mod.nominal_sigma_rdout ** 2)
        self.one_minus_2u_minus_u_squared_over_v = 1 - 2 * self.u - self.u * self.u * self.one_over_v
        if self.calc_curvatures:
            self.u_times_one_over_v = self.u*self.one_over_v
            self.u_u_one_over_v = self.u*self.u_times_one_over_v
            self.one_over_v_times_one_minus_2u_minus_u_squared_over_v = self.one_over_v*self.one_minus_2u_minus_u_squared_over_v
        self.common_grad_term = self.one_over_v * self.one_minus_2u_minus_u_squared_over_v
        self._Zscore = self.u*np.sqrt(self.one_over_v)

    def _evaluate_log_averageI(self):  # for Poisson only stats
        try:
            self.log_Lambda = np.log(self.model_Lambda)
        except FloatingPointError:
            pass
        neg_lam = self.model_Lambda <=0
        M = self.Modelers[self._i_shot]
        if any(neg_lam[M.all_trusted].ravel()):
            self.log_Lambda[neg_lam] = 1e-6
            LOGGER.warning(Bcolors.WARNING+("NEGATIVE INTENSITY IN MODEL (negative_models=%d)!" % self.num_negative_model) + Bcolors.ENDC)
        #    raise ValueError("model of Bragg spots cannot have negative intensities...")
        self.log_Lambda[neg_lam] = 0

    def _evaluate_log_averageI_plus_sigma_readout(self):
        Mod = self.Modelers[self._i_shot]
        v = self.model_Lambda + Mod.nominal_sigma_rdout ** 2
        v_is_neg = (v <= 0).ravel()
        if any(v_is_neg[Mod.all_trusted]):
            LOGGER.warning(Bcolors.WARNING+"NEGATIVE INTENSITY IN MODEL!"+Bcolors.ENDC)
        #    raise ValueError("model of Bragg spots cannot have negative intensities...")
        self.log_v = np.log(v)
        self.log_v[v <= 0] = 0  # but will I ever negative_model ?

    def get_refined_Bmatrix(self, i_shot, recip=False):
        if recip:
            return self.Modelers[i_shot].PAR.ucell_man.B_recipspace
        else:
            return self.Modelers[i_shot].PAR.ucell_man.B_realspace

    def curvatures(self):
        return self.curv

    def _MPI_sync_hkl_freq(self):
            if self.refine_Fcell:
                if self.rank != 0:
                    self.hkl_frequency = None
                self.hkl_frequency = COMM.bcast(self.hkl_frequency)

    def _MPI_sync_fcell_parameters(self):
        if not self.I_AM_ROOT:
            self.sigma_for_res_id = None
            self.res_group_id_from_fcell_index = None
            self.resolution_ids_from_i_fcell = self.fcell_sigmas_from_i_fcell = self.fcell_init_from_i_fcell = None

        if self.rescale_params:
            if self.refine_Fcell:
                self.fcell_sigmas_from_i_fcell = COMM.bcast(self.fcell_sigmas_from_i_fcell)
                self.fcell_init_from_i_fcell = COMM.bcast(self.fcell_init_from_i_fcell)

    def _MPI_sync_panel_params(self):
        if not self.I_AM_ROOT:
            self.panelRot_params = None
            self.panelX_params = None
            self.panelY_params = None
            self.panelZ_params = None
        self.panelRot_params = COMM.bcast(self.panelRot_params)
        self.panelX_params = COMM.bcast(self.panelX_params)
        self.panelY_params = COMM.bcast(self.panelY_params)
        self.panelZ_params = COMM.bcast(self.panelZ_params)

    def _MPI_reduce_broadcast(self, var):
        var = COMM.reduce(var, MPI.SUM, root=0)
        var = COMM.bcast(var, root=0)
        return var

    def _MPI_barrier(self):
        COMM.barrier()
