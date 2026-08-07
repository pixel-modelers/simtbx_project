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
import csv
from copy import deepcopy
from simtbx.diffBragg import hopper_io
from simtbx.diffBragg import utils as diffBragg_utils

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

# Lazy-import helpers to avoid circular imports (geometry -> ensemble_refine_launcher -> stage_two_refiner)
UCELL_ID_OFFSET = 3  # from hopper_utils; duplicated here to avoid import
PAN_O_ID = 14
PAN_F_ID = 17
PAN_S_ID = 18
PAN_X_ID = 15
PAN_Y_ID = 16
PAN_Z_ID = 10
PAN_OFS_IDS = (PAN_O_ID, PAN_F_ID, PAN_S_ID)
PAN_XYZ_IDS = (PAN_X_ID, PAN_Y_ID, PAN_Z_ID)

# base per-shot parameters: scale, B-factor, Ncells abc (Na,Nb,Nc), Ncells def (Nd,Ne,Nf), RotXYZ
N_PARAM_PER_SHOT_BASE = 11
# backwards compat alias (used in places that don't depend on ucell count)
N_PARAM_PER_SHOT = N_PARAM_PER_SHOT_BASE


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

        self._rotX_id = 0  # diffBragg internal index for RotX derivative manager
        self._rotY_id = 1  # diffBragg internal index for RotY derivative manager
        self._rotZ_id = 2  # diffBragg internal index for RotZ derivative manager
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
        self._bfactor_id = 25

        self.symbol = sgsymbol
        self.space_group = sgtbx.space_group(sgtbx.space_group_info(symbol=self.symbol).type().hall_symbol())

        self.REGIONS = None  # detector regions for gain refinement (this is a labeled array, same shape as self.S.detector
        self.num_regions = None  # the number of unique regions
        self.unique_regions = None  # the unique regions as a 1-d np.array
        self.region_params = {}  # dictionary for storuing diffBragg/refiners/parameters.RangerParameter for gain correction params

        self.I_AM_ROOT = COMM.rank==0

        # --- Tier 0/0.5 diagnostics state ---
        self._diag_csv_file = None
        self._diag_csv_writer = None
        self._prev_x = None
        self._prev_f = None
        self._prev_g = None
        self._diag_fterm_volume = 0.0   # accumulated per eval
        self._diag_fterm_chisq = 0.0    # accumulated per eval
        self._diag_neg_v_count = 0      # pixels with v <= 0
        self._diag_neg_lam_count = 0    # pixels with model_Lambda <= 0
        self._diag_neg_v_shots = 0      # shots with at least one v <= 0 pixel
        self._diag_min_multi_skip = 0   # i_fcell skipped by min_multiplicity
        self._diag_min_multi_skip_pix = 0  # trusted pixel count for skipped i_fcell
        self._eval_shot_data = []  # per-shot (shot_id, G, B, sigZ, chisq, n_regions) for current eval

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
        self.n_ucell_param = len(self.Modelers[test_shot].PAR.ucell_man.variables)
        # per-shot layout: Scale(0), B(1), Na/Nb/Nc(2-4), Nd/Ne/Nf(5-7), RotX/Y/Z(8-10), [ucell(11..11+n_ucell)]
        if self.params.refiner.refine_ucell:
            self.n_params_per_shot = N_PARAM_PER_SHOT_BASE + self.n_ucell_param
        else:
            self.n_params_per_shot = N_PARAM_PER_SHOT_BASE

        # Panel geometry: global params (6 per panel group)
        self._setup_panel_groups()
        n_panel_params = self.n_panel_groups * 6 if self.params.refiner.refine_panel_geom else 0

        self.n_total_params = self.n_total_shots*self.n_params_per_shot + self.n_global_fcell + self.num_regions + n_panel_params

        self.spot_scale_xpos = {}
        self.Bfactor_xpos = {}
        self.Ncells_xstart = {}
        self.Ndef_xstart = {}
        self.RotXYZ_xstart = {}
        self.ucell_xstart = {}
        for shot_id in self.shot_ids:
            base = self.shot_mapping[shot_id]*self.n_params_per_shot
            self.spot_scale_xpos[shot_id] = base
            self.Bfactor_xpos[shot_id] = base + 1
            self.Ncells_xstart[shot_id] = base + 2
            self.Ndef_xstart[shot_id] = base + 5
            self.RotXYZ_xstart[shot_id] = base + 8
            if self.params.refiner.refine_ucell:
                self.ucell_xstart[shot_id] = base + N_PARAM_PER_SHOT_BASE
        LOGGER.info("--0 create an Fcell mapping")
        if self.refine_Fcell:
            #idx, data = self.S.D.Fhkl_tuple
            #self.idx_from_p1 = {h: i for i, h in enumerate(idx)}
            self._make_p1_equiv_mapping()
            # self.p1_from_idx = {i: h for i, h in zip(idx, data)}

        # Make a mapping of panel id to parameter index and backwards
        self.pid_from_idx = {}
        self.idx_from_pid = {}

        self.x = flex.double(np.ones(self.n_total_params))
        LOGGER.info("--Setting up per shot parameters")

        self.fcell_xstart = self.n_total_shots*self.n_params_per_shot
        self.regions_xstart = self.fcell_xstart + self.n_global_fcell
        self.panel_xstart = self.regions_xstart + self.num_regions  # panel geom params start after regions

        self._setup_region_refinement_parameters()
        self._setup_ncells_refinement_parameters()
        self._setup_ucell_refinement_parameters()
        self._setup_panel_refinement_parameters()
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
        if self.params.refiner.refine_Ndef:
            self.D.refine(self._ncells_def_id)
        if self.params.refiner.refine_RotXYZ:
            self.D.refine(self._rotX_id)
            self.D.refine(self._rotY_id)
            self.D.refine(self._rotZ_id)
        if self.params.refiner.refine_Bfactor:
            self.D.refine(self._bfactor_id)
            print("STAGE2_BFACTOR: B-factor refinement ENABLED (refine_Bfactor=True)", flush=True)
        else:
            print("STAGE2_BFACTOR: B-factor refinement DISABLED", flush=True)
        if self.params.refiner.refine_ucell:
            for i_uc in range(self.n_ucell_param):
                self.D.refine(UCELL_ID_OFFSET + i_uc)
            LOGGER.info("Ucell refinement ENABLED (%d params)" % self.n_ucell_param)
        if self.params.refiner.refine_panel_geom:
            for pid in PAN_OFS_IDS + PAN_XYZ_IDS:
                self.D.refine(pid)
            LOGGER.info("Panel geometry refinement ENABLED (%d groups, %d params)"
                        % (self.n_panel_groups, self.n_panel_groups * 6))
        self.D.initialize_managers()

        for sid in self.shot_ids:
            Modeler = self.Modelers[sid]
            Modeler.all_fcell_global_idx = np.array([self.hiasu.to_idx[h] for h in Modeler.hi_asu_perpix])
            Modeler.unique_i_fcell = set(Modeler.all_fcell_global_idx)
            Modeler.i_fcell_slices = self._get_i_fcell_slices(Modeler)
            self.Modelers[sid] = Modeler  # TODO: VERIFY IF THIS IS NECESSARY ?

        self._save_fcell_shot_coupling()
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
        ndef_names = "Nd", "Ne", "Nf"
        for i_shot in self.shot_ids:
            Ncells_params = self.Modelers[i_shot].PAR.Nabc
            for i_n, p in enumerate(Ncells_params):
                p.xpos = self.Ncells_xstart[i_shot] + i_n
                p.name = "%s_shot%d_rank%d" % ( names[i_n], i_shot, COMM.rank)
            Ndef_params = self.Modelers[i_shot].PAR.Ndef
            for i_n, p in enumerate(Ndef_params):
                p.xpos = self.Ndef_xstart[i_shot] + i_n
                p.name = "%s_shot%d_rank%d" % ( ndef_names[i_n], i_shot, COMM.rank)
            rot_names = "RotX", "RotY", "RotZ"
            RotXYZ_params = self.Modelers[i_shot].PAR.RotXYZ_params
            for i_r, p in enumerate(RotXYZ_params):
                p.xpos = self.RotXYZ_xstart[i_shot] + i_r
                p.name = "%s_shot%d_rank%d" % ( rot_names[i_r], i_shot, COMM.rank)
            self.Modelers[i_shot].PAR.B.xpos = self.Bfactor_xpos[i_shot]

    def _setup_ucell_refinement_parameters(self):
        if not self.params.refiner.refine_ucell:
            return
        ucell_names = self.Modelers[self.shot_ids[0]].PAR.ucell_man.variable_names
        for i_shot in self.shot_ids:
            for i_uc, p in enumerate(self.Modelers[i_shot].PAR.ucell):
                p.xpos = self.ucell_xstart[i_shot] + i_uc
                p.name = "%s_shot%d_rank%d" % (ucell_names[i_uc], i_shot, COMM.rank)

    def _setup_panel_groups(self):
        """Load panel group definitions. Called early in _setup before n_total_params is computed."""
        if self.params.refiner.refine_panel_geom:
            from simtbx.diffBragg import utils as _utils
            if self.params.refiner.panel_group_file is not None:
                self.panel_group_from_id = _utils.load_panel_group_file(
                    self.params.refiner.panel_group_file)
            else:
                npan = len(self.S.detector)
                self.panel_group_from_id = {pid: 0 for pid in range(npan)}
            self.panel_groups_sorted = sorted(set(self.panel_group_from_id.values()))
            self.n_panel_groups = len(self.panel_groups_sorted)

            # Build reference origins for each panel
            det = self.S.detector
            panels_per_group = {gid: [] for gid in self.panel_groups_sorted}
            for pid in self.panel_group_from_id:
                panels_per_group[self.panel_group_from_id[pid]].append(pid)
            self.panel_reference_from_id = {}
            for pid in self.panel_group_from_id:
                gid = self.panel_group_from_id[pid]
                ref_panel = det[panels_per_group[gid][0]]
                self.panel_reference_from_id[pid] = ref_panel.get_origin()

            # Store on SIM for update_detector compatibility
            self.S.panel_group_from_id = self.panel_group_from_id
            self.S.panel_reference_from_id = self.panel_reference_from_id
            self.S.panel_groups_refined = set(self.panel_groups_sorted)
        else:
            self.n_panel_groups = 0
            self.panel_groups_sorted = []
            self.panel_group_from_id = {}
            self.panel_reference_from_id = {}

    def _setup_panel_refinement_parameters(self):
        """Create RangedParameter objects for panel geometry (6 per group)."""
        self.panel_params = {}
        if not self.params.refiner.refine_panel_geom:
            return

        GEO = self.params.geometry
        DEG_TO_PI = np.pi / 180.
        vary_rots = [not fixed_flag for fixed_flag in GEO.fix.panel_rotations]
        vary_shifts = [not fixed_flag for fixed_flag in GEO.fix.panel_translations]
        sigma_rot = GEO.sigmas.panel_rot
        sigma_xyz = GEO.sigmas.panel_xyz

        for i_g, group_id in enumerate(self.panel_groups_sorted):
            base_xpos = self.panel_xstart + i_g * 6
            names_inits_sigmas_minmax_fix = [
                ("group%d_RotOrth" % group_id, 0, sigma_rot[0],
                 GEO.min.panel_rotations[0]*DEG_TO_PI, GEO.max.panel_rotations[0]*DEG_TO_PI, not vary_rots[0]),
                ("group%d_RotFast" % group_id, 0, sigma_rot[1],
                 GEO.min.panel_rotations[1]*DEG_TO_PI, GEO.max.panel_rotations[1]*DEG_TO_PI, not vary_rots[1]),
                ("group%d_RotSlow" % group_id, 0, sigma_rot[2],
                 GEO.min.panel_rotations[2]*DEG_TO_PI, GEO.max.panel_rotations[2]*DEG_TO_PI, not vary_rots[2]),
                ("group%d_ShiftX" % group_id, 0, sigma_xyz[0],
                 GEO.min.panel_translations[0]*1e-3, GEO.max.panel_translations[0]*1e-3, not vary_shifts[0]),
                ("group%d_ShiftY" % group_id, 0, sigma_xyz[1],
                 GEO.min.panel_translations[1]*1e-3, GEO.max.panel_translations[1]*1e-3, not vary_shifts[1]),
                ("group%d_ShiftZ" % group_id, 0, sigma_xyz[2],
                 GEO.min.panel_translations[2]*1e-3, GEO.max.panel_translations[2]*1e-3, not vary_shifts[2]),
            ]
            for j, (name, init, sigma, minv, maxv, fix) in enumerate(names_inits_sigmas_minmax_fix):
                p = RangedParameter(init=init, minval=minv, maxval=maxv, sigma=sigma,
                                    center=0, beta=1e10, fix=fix)
                p.xpos = base_xpos + j
                p.name = name
                self.panel_params[name] = p

        # Set up group_id_slices on each Modeler for efficient per-group pixel access
        from simtbx.diffBragg.refiners.geometry import set_group_id_slices
        for i_shot in self.shot_ids:
            set_group_id_slices(self.Modelers[i_shot], self.panel_group_from_id)

        LOGGER.info("Panel params: %d groups, %d params total, vary_rots=%s, vary_shifts=%s"
                     % (self.n_panel_groups, len(self.panel_params), vary_rots, vary_shifts))

    def _gain_restraints(self):
        if self.params.refiner.gain_restraint:
            for p in self.region_params.values():
                self.target_functional += p.get_restraint_val(self.x[p.xpos])
                self.grad[p.xpos] += p.get_restraint_deriv(self.x[p.xpos])

    def _bfactor_restraints(self):
        if not self.params.refiner.refine_Bfactor:
            return
        for i_shot in self.shot_ids:
            p = self.Modelers[i_shot].PAR.B
            if p.beta is None:
                return  # no restraint configured
            self.target_functional += p.get_restraint_val(self.x[p.xpos])
            self.grad[p.xpos] += p.get_restraint_deriv(self.x[p.xpos])

    def _Fhkl_restraints(self):
        """Restrain Fhkl envelope: penalize deviation of bin-mean intensity from initial.

        The restraint is:  f += 0.5 * sum_bins (I_mean_b - I_init_b)^2 / beta
        where I_mean_b = mean(F_i^2) for all i_fcell in bin b.

        Gradient w.r.t. x_i (log-parameterized):
           g_i += (I_mean_b - I_init_b) * 2 * sig_i * F_i^2 / (beta * N_b)
        where dF/dx = sig * F for the log parameterization.
        """
        beta = self.params.betas.Fhkl
        if beta is None or not self.refine_Fcell:
            return
        if not hasattr(self, '_fcell_bin_id'):
            return

        # Current F values and intensities
        F_current = self._fcell_at_i_fcell
        I_current = F_current ** 2

        # Compute current bin means
        I_mean = np.zeros(self._fcell_n_bins)
        for b in range(self._fcell_n_bins):
            mask = self._fcell_bin_id == b
            if mask.any():
                I_mean[b] = I_current[mask].mean()

        # Restraint target and gradient
        delta = I_mean - self._fcell_bin_Imean_init
        f_restraint = 0.5 * np.sum(delta ** 2 / beta)
        self.target_functional += f_restraint

        # Per-Fcell gradient: chain rule through bin mean and log parameterization
        sigs = self.fcell_sigmas_from_i_fcell
        if np.isscalar(sigs):
            sigs = np.full(self.n_global_fcell, sigs)
        for i_fcell in range(self.n_global_fcell):
            b = self._fcell_bin_id[i_fcell]
            N_b = self._fcell_bin_count[b]
            if N_b == 0:
                continue
            xpos = self.fcell_xstart + i_fcell
            # d(restraint)/dx_i = delta_b / beta * d(I_mean_b)/dx_i
            # d(I_mean_b)/dx_i = 2*sig_i*F_i^2 / N_b  (for log param: dF/dx = sig*F)
            g_i = delta[b] / beta * 2.0 * sigs[i_fcell] * I_current[i_fcell] / N_b
            self.grad[xpos] += g_i

        if self.I_AM_ROOT and (self.iterations < 3 or self.target_eval_count % 10 == 0):
            LOGGER.info("Fhkl_envelope_restraint: f=%.4e beta=%.2e delta_range=[%.4e, %.4e]"
                        % (f_restraint, beta, delta.min(), delta.max()))

    def _save_per_shot_sigZ(self):
        """Save per-shot and per-shoebox sigZ/scores at configured intervals.
        Output maps back to stage1 modeler.npy files for comparison."""
        if not self._collect_shoebox_scores:
            return

        # --- per-shot sigZ with modeler path ---
        local_rows = []
        for i, i_shot in enumerate(self.shot_ids):
            MOD = self.Modelers[i_shot]
            sigZ = self.all_sigZ[i] if i < len(self.all_sigZ) else np.nan
            modeler_path = ""
            if hasattr(MOD, 'pandas_table_row') and 'stage1_output_img' in MOD.pandas_table_row.index:
                modeler_path = str(MOD.pandas_table_row['stage1_output_img'])
            exp_name = ""
            if hasattr(MOD, 'pandas_table_row') and 'exp_name' in MOD.pandas_table_row.index:
                exp_name = str(MOD.pandas_table_row['exp_name'])
            B = self._get_bfactor(i_shot)
            G = self._get_spot_scale(i_shot)**2
            local_rows.append((i_shot, COMM.rank, sigZ, B, G, len(MOD.Hi), modeler_path, exp_name))

        all_rows = COMM.gather(local_rows, root=0)
        if COMM.rank == 0 and all_rows is not None and self.output_dir is not None:
            rows = [r for rank_rows in all_rows for r in rank_rows]
            rows.sort(key=lambda r: r[0])
            outpath = os.path.join(self.output_dir,
                                   "per_shot_sigZ_eval%d.csv" % self.target_eval_count)
            with open(outpath, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["shot_id", "rank", "sigZ", "B", "G", "n_refls",
                            "stage1_modeler", "exp_name"])
                w.writerows(rows)
            LOGGER.info("Saved per-shot sigZ to %s (%d shots)" % (outpath, len(rows)))

        # --- per-shoebox sigZ and score (collected during shot loop) ---
        all_shoebox_rows = COMM.gather(self._per_shoebox_data, root=0)
        if COMM.rank == 0 and all_shoebox_rows is not None and self.output_dir is not None:
            rows = [r for rank_rows in all_shoebox_rows for r in rank_rows]
            rows.sort(key=lambda r: (r[0], r[1]))
            outpath = os.path.join(self.output_dir,
                                   "per_shoebox_scores_eval%d.csv" % self.target_eval_count)
            with open(outpath, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["shot_id", "i_fcell", "sigZ", "cc", "ntrust", "chisq", "stage1_modeler"])
                w.writerows(rows)
            LOGGER.info("Saved per-shoebox scores to %s (%d shoeboxes)" % (outpath, len(rows)))

    def _gather_and_save_shot_params(self):
        """Gather per-shot (G, B, sigZ, chisq, n_regions) from all ranks and save npz every eval."""
        all_shot_data = COMM.gather(self._eval_shot_data, root=0)
        gathered = None
        if self.I_AM_ROOT and all_shot_data is not None and self.output_dir is not None:
            rows = [r for rank_rows in all_shot_data for r in rank_rows]
            rows.sort(key=lambda x: x[0])  # sort by shot_id
            gathered = rows
            shot_ids = np.array([r[0] for r in rows], dtype=np.int32)
            G_vals = np.array([r[1] for r in rows])
            B_vals = np.array([r[2] for r in rows])
            sigZ_vals = np.array([r[3] for r in rows])
            chisq_vals = np.array([r[4] for r in rows])
            n_regions = np.array([r[5] for r in rows], dtype=np.int32)
            outf = os.path.join(self.output_dir,
                                "shot_params_eval%d" % self.target_eval_count)
            np.savez(outf, shot_ids=shot_ids, G=G_vals, B=B_vals,
                     sigZ=sigZ_vals, chisq=chisq_vals, n_regions=n_regions)
            LOGGER.info("Saved shot params eval %d: %d shots, "
                        "G=[%.4f, %.4f] B=[%.4f, %.4f] chisq=[%.2f, %.2f]"
                        % (self.target_eval_count, len(rows),
                           G_vals.min(), G_vals.max(),
                           B_vals.min(), B_vals.max(),
                           chisq_vals.min(), chisq_vals.max()))
        return gathered

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

            # Set up resolution bins for Fhkl envelope restraint
            self._setup_fcell_restraint_bins()
            LOGGER.info("DONE make fcell_init")

    def _setup_fcell_restraint_bins(self):
        """Compute resolution bins for Fhkl envelope restraint.
        Each i_fcell gets assigned a bin based on d-spacing. The initial
        bin-mean intensity (mean of F_init^2) is stored as the restraint target."""
        uc = self.S.crystal.dxtbx_crystal.get_unit_cell()
        n_bins = self.params.Fhkl_dspace_bins

        # Compute d-spacing for each i_fcell
        d_spacings = np.array([uc.d(self.hiasu.from_idx[i])
                               for i in range(self.n_global_fcell)])

        # Equal-count binning by d-spacing (sorted ascending = high-res first)
        sorted_d = np.sort(d_spacings)
        bin_edges = [sorted_d[0] - 1e-6]
        for chunk in np.array_split(sorted_d, n_bins):
            if len(chunk):
                bin_edges.append(chunk[-1])
        bin_edges[-1] += 1e-6
        bin_edges = np.array(bin_edges)

        # Assign each i_fcell to a bin
        self._fcell_bin_id = np.digitize(d_spacings, bin_edges) - 1
        self._fcell_bin_id = np.clip(self._fcell_bin_id, 0, n_bins - 1)
        self._fcell_n_bins = n_bins

        # Compute initial bin-mean intensities (I = F^2)
        I_init = self.fcell_init_from_i_fcell ** 2
        self._fcell_bin_Imean_init = np.zeros(n_bins)
        self._fcell_bin_count = np.zeros(n_bins)
        for b in range(n_bins):
            mask = self._fcell_bin_id == b
            if mask.any():
                self._fcell_bin_Imean_init[b] = I_init[mask].mean()
                self._fcell_bin_count[b] = mask.sum()

        LOGGER.info("Fhkl envelope restraint: %d bins, d=[%.2f, %.2f] A, "
                     "bin counts=[%d, %d]"
                     % (n_bins, d_spacings.min(), d_spacings.max(),
                        int(self._fcell_bin_count.min()),
                        int(self._fcell_bin_count.max())))

    def _save_fcell_shot_coupling(self):
        """Save which shots observe each Fcell (once at setup). MPI-gathered."""
        local_coupling = {}  # i_fcell -> list of shot_ids on this rank
        for i_shot in self.shot_ids:
            MOD = self.Modelers[i_shot]
            for i_fcell in MOD.unique_i_fcell:
                local_coupling.setdefault(i_fcell, []).append(i_shot)

        all_coupling = COMM.gather(local_coupling, root=0)
        if self.I_AM_ROOT and self.output_dir is not None:
            merged = {}
            for rank_coupling in all_coupling:
                for i_fcell, shot_list in rank_coupling.items():
                    merged.setdefault(i_fcell, []).extend(shot_list)
            # Save as arrays: i_fcell -> number of shots, and per-fcell shot lists
            n_fcell = self.n_global_fcell
            multiplicity = np.zeros(n_fcell, dtype=np.int32)
            for i_fcell, shots in merged.items():
                multiplicity[i_fcell] = len(shots)
            # Save sparse representation: arrays of (i_fcell, shot_ids) pairs
            fcell_ids = []
            shot_ids_flat = []
            for i_fcell in sorted(merged.keys()):
                for sid in sorted(merged[i_fcell]):
                    fcell_ids.append(i_fcell)
                    shot_ids_flat.append(sid)
            outf = os.path.join(self.output_dir, "fcell_shot_coupling")
            np.savez(outf, multiplicity=multiplicity,
                     fcell_ids=np.array(fcell_ids, dtype=np.int32),
                     shot_ids=np.array(shot_ids_flat, dtype=np.int32))
            LOGGER.info("Saved Fcell-shot coupling: %d Fcells, %d entries, "
                        "multiplicity range [%d, %d], median=%.1f"
                        % (n_fcell, len(fcell_ids),
                           multiplicity[multiplicity > 0].min() if np.any(multiplicity > 0) else 0,
                           multiplicity.max(),
                           np.median(multiplicity[multiplicity > 0]) if np.any(multiplicity > 0) else 0))

    def _get_sausage_parameters(self, i_shot):
        pass

    def _get_rotXYZ(self, i_shot):
        if self.params.refiner.refine_RotXYZ:
            vals = []
            RotXYZ_p = self.Modelers[i_shot].PAR.RotXYZ_params
            for p in RotXYZ_p:
                xval = self.x[p.xpos]
                val = p.get_val(xval)
                vals.append(val)
        else:
            vals = [self.Modelers[i_shot].PAR.RotXYZ_params[i_rot].init for i_rot in range(3)]
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
        vals = []
        if self.params.refiner.refine_ucell:
            for i in range(self.n_ucell_param):
                p = self.Modelers[i_shot].PAR.ucell[i]
                vals.append(p.get_val(self.x[p.xpos]))
        else:
            for i in range(self.n_ucell_param):
                vals.append(self.Modelers[i_shot].PAR.ucell[i].init)
        return vals

    def _get_panelRot_val(self, panel_id):
        pass

    def _get_panelXYZ_val(self, panel_id, i_shot=0):
        pass

    def _get_detector_distance_val(self, i_shot):
        return self.Modelers[i_shot].PAR.detz_shift.init

    def _get_ncells_def(self, i_shot):
        if self.params.refiner.refine_Ndef:
            vals = []
            Ndef_p = self.Modelers[i_shot].PAR.Ndef
            for p in Ndef_p:
                xval = self.x[p.xpos]
                val = p.get_val(xval)
                vals.append(val)
        else:
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
        p = self.Modelers[i_shot].PAR.B
        return p.get_val(self.x[p.xpos])

    def _get_bg_vals(self, i_shot, i_spot):
        pass

    def _send_ucell_gradients_to_derivative_managers(self):
        """Set ucell derivative matrices on diffBragg. Called after ucell_man.variables are updated."""
        if not self.params.refiner.refine_ucell:
            return
        ucell_man = self.Modelers[self._i_shot].PAR.ucell_man
        for i_uc in range(self.n_ucell_param):
            self.D.set_ucell_derivative_matrix(
                i_uc + UCELL_ID_OFFSET,
                ucell_man.derivative_matrices[i_uc])

    def _run_diffBragg_current(self):
        LOGGER.info("run diffBragg for shot %d" % self._i_shot)
        PAR = self.Modelers[self._i_shot].PAR
        pfs = self.Modelers[self._i_shot].pan_fast_slow
        npix = len(self.Modelers[self._i_shot].all_data)
        nom_h_p1 = self.Modelers[self._i_shot].all_nominal_hkl_p1 if self.use_nominal_h else None

        if PAR.num_xtals > 1:
            # Multi-domain (blue sausage): loop over domains, accumulate raw pixels
            # weighted by (G_i / G_0)^2 so existing scale_fac = gains * G_0^2 gives
            # model = gains * Σ(G_i^2 * bragg_pix_i)
            if not hasattr(self, '_sausage_debug_printed'):
                print("SAUSAGE_DEBUG stage2 _run_diffBragg_current: num_xtals=%d, shot=%d" % (PAR.num_xtals, self._i_shot), flush=True)
                self._sausage_debug_printed = True
            G0 = PAR.Scale.init

            # Domain 0 (primary): ratio = 1
            self.D.Umatrix = PAR.Umatrix
            if self.params.symmetrize_Flatt:
                self._symmetrize_Flatt_for_Umat(PAR.Umatrix)
            if nom_h_p1 is not None:
                self.D.add_diffBragg_spots(pfs, nom_h_p1)
            else:
                self.D.add_diffBragg_spots(pfs)
            self._sausage_model_pix = self.D.raw_pixels_roi[:npix].as_numpy_array().copy()

            # Also accumulate Fhkl derivatives for domain 0
            if self.refine_Fcell:
                dF = self.D.get_derivative_pixels(self._fcell_id)
                self._sausage_fcell_deriv = dF[:npix].as_numpy_array().copy()

            # Domains 1..N-1
            for Umat, dom_scale in zip(PAR.other_Umats, PAR.other_spotscales):
                ratio_sq = (dom_scale / G0) ** 2
                self.D.Umatrix = Umat
                if self.params.symmetrize_Flatt:
                    self._symmetrize_Flatt_for_Umat(Umat)
                if nom_h_p1 is not None:
                    self.D.add_diffBragg_spots(pfs, nom_h_p1)
                else:
                    self.D.add_diffBragg_spots(pfs)
                self._sausage_model_pix += ratio_sq * self.D.raw_pixels_roi[:npix].as_numpy_array()
                if self.refine_Fcell:
                    dF = self.D.get_derivative_pixels(self._fcell_id)
                    self._sausage_fcell_deriv += ratio_sq * dF[:npix].as_numpy_array()

            # Restore primary Umatrix for any downstream code
            self.D.Umatrix = PAR.Umatrix
            if self.params.symmetrize_Flatt:
                self._symmetrize_Flatt_for_Umat(PAR.Umatrix)
        else:
            self._sausage_model_pix = None
            self._sausage_fcell_deriv = None
            if nom_h_p1 is not None:
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
            rotXYZ = self._get_rotXYZ(self._i_shot)
            RXYZU = hopper_io.diffBragg_Umat(rotXYZ[0], rotXYZ[1], rotXYZ[2], self.D.Umatrix)
            Cryst = deepcopy(self.S.crystal.dxtbx_crystal)
            B_realspace = self.get_refined_Bmatrix(self._i_shot, recip=False)
            A = RXYZU * B_realspace
            A_recip = A.inverse().transpose()
            Cryst.set_A(A_recip)
            symbol = self.S.crystal.space_group_info.type().lookup_symbol()
            self.D.set_mosaic_blocks_sym(Cryst, symbol , self.params.simulator.crystal.num_mosaicity_samples,
                                        refining_eta=False) # NOTE:no eta refinement in this stage 2 script (possible in ens.hopper)

    def _symmetrize_Flatt_for_Umat(self, Umat):
        """Helper for multi-domain: symmetrize mosaic blocks for a specific Umatrix."""
        RXYZU = hopper_io.diffBragg_Umat(0, 0, 0, Umat)
        Cryst = deepcopy(self.S.crystal.dxtbx_crystal)
        B_realspace = self.get_refined_Bmatrix(self._i_shot, recip=False)
        A = RXYZU * B_realspace
        A_recip = A.inverse().transpose()
        Cryst.set_A(A_recip)
        symbol = self.S.crystal.space_group_info.type().lookup_symbol()
        self.D.set_mosaic_blocks_sym(Cryst, symbol, self.params.simulator.crystal.num_mosaicity_samples,
                                     refining_eta=False)

    def _set_background_plane(self):
        self.tilt_plane = self.Modelers[self._i_shot].all_background[self.roi_sel]

    def _update_sausages(self):
        pass

    def _update_gonio(self):
        Mod = self.Modelers[self._i_shot]
        if getattr(Mod, 'osc_deg', None) is not None and Mod.osc_deg > 0:
            diffBragg_utils.update_SIM_with_gonio(
                self.S, delta_phi=Mod.osc_deg,
                num_phi_steps=Mod.phisteps)

    def _update_rotXYZ(self):
        vals = self._get_rotXYZ(self._i_shot)
        self.D.set_value(self._rotX_id, vals[0])
        self.D.set_value(self._rotY_id, vals[1])
        self.D.set_value(self._rotZ_id, vals[2])

    def _update_ncells(self):
        vals = self._get_ncells_abc(self._i_shot)
        self.D.set_ncells_values(tuple(vals))

    def _update_ncells_def(self):
        vals = self._get_ncells_def(self._i_shot)
        self.D.Ncells_def = tuple(vals)

    def _update_dxtbx_detector(self):
        shiftZ = self._get_detector_distance_val(self._i_shot)
        self.S.D.shift_origin_z(self.S.detector,  shiftZ)

    def _update_panel_geom(self):
        if not self.params.refiner.refine_panel_geom:
            return
        from simtbx.diffBragg.geom_utils import update_detector as _geom_update_detector
        _geom_update_detector(self.x, self.panel_params, self.S, save=None,
                              rank=COMM.rank, force=False)

    def _extract_spectra_coefficient_derivatives(self):
        pass

    def _pre_extract_deriv_arrays(self):
        npix = len(self.Modelers[self._i_shot].all_data)

        if self._sausage_model_pix is not None:
            # Multi-domain: use pre-accumulated weighted sum from _run_diffBragg_current
            self._model_pix = self._sausage_model_pix
            if self.refine_Fcell:
                self._extracted_fcell_deriv = self._sausage_fcell_deriv
                if self.calc_curvatures:
                    raise NotImplementedError("curvatures not implemented for multi-domain")
        else:
            self._model_pix = self.D.raw_pixels_roi[:npix].as_numpy_array()
            if self.refine_Fcell:
                dF = self.D.get_derivative_pixels(self._fcell_id)
                self._extracted_fcell_deriv = dF[:npix].as_numpy_array()
                if self.calc_curvatures:
                    d2F = self.D.get_second_derivative_pixels(self._fcell_id)
                    self._extracted_fcell_second_deriv = d2F[:npix].as_numpy_array()

        if self.params.refiner.refine_Nabc:
            self.dNabc = [d[:npix].as_numpy_array() for d in self.D.get_ncells_derivative_pixels()]
            if self.calc_curvatures:
                self.d2Nabc = [d[:npix].as_numpy_array() for d in self.D.get_ncells_second_derivative_pixels()]

        if self.params.refiner.refine_Ndef:
            self.dNdef = [d[:npix].as_numpy_array() for d in self.D.get_ncells_def_derivative_pixels()]
            if self.calc_curvatures:
                self.d2Ndef = [d[:npix].as_numpy_array() for d in self.D.get_ncells_def_second_derivative_pixels()]

        if self.params.refiner.refine_RotXYZ:
            rot_ids = [self._rotX_id, self._rotY_id, self._rotZ_id]
            self.dRotXYZ = [self.D.get_derivative_pixels(rid).as_numpy_array()[:npix] for rid in rot_ids]
            if self.calc_curvatures:
                self.d2RotXYZ = [self.D.get_second_derivative_pixels(rid).as_numpy_array()[:npix] for rid in rot_ids]

        if self.params.refiner.refine_Bfactor:
            self._dB = self.D.get_Bfactor_derivative_pixels()[:npix].as_numpy_array()
            if self.calc_curvatures:
                # d2I/dB2 = s^4 * I = -s^2 * dI/dB, where s^2 = q^2/4
                Bfactor_qterm = self.Modelers[self._i_shot].all_q_perpix**2 / 4.
                self._d2B = -Bfactor_qterm * self._dB

        if self.params.refiner.refine_ucell:
            self.dUcell = []
            for i_uc in range(self.n_ucell_param):
                d = self.D.get_derivative_pixels(UCELL_ID_OFFSET + i_uc).as_numpy_array()[:npix]
                self.dUcell.append(d)

        if self.params.refiner.refine_panel_geom:
            self._panel_derivs = []
            for pid in PAN_OFS_IDS + PAN_XYZ_IDS:
                try:
                    d = self.D.get_derivative_pixels(pid).as_numpy_array()[:npix]
                except ValueError:
                    d = None
                self._panel_derivs.append(d)

    def _extract_sausage_derivs(self):
        pass

    def _extract_Umatrix_derivative_pixels(self):
        if self.params.refiner.refine_RotXYZ:
            npix = len(self.Modelers[self._i_shot].all_data)
            rot_ids = [self._rotX_id, self._rotY_id, self._rotZ_id]
            self.dRotXYZ = [self.D.get_derivative_pixels(rid).as_numpy_array()[:npix] for rid in rot_ids]

    def _extract_Bmatrix_derivative_pixels(self):
        if not self.params.refiner.refine_ucell:
            return
        npix = len(self.Modelers[self._i_shot].all_data)
        self.dUcell = []
        for i_uc in range(self.n_ucell_param):
            d = self.D.get_derivative_pixels(UCELL_ID_OFFSET + i_uc).as_numpy_array()[:npix]
            self.dUcell.append(d)

    def _extract_ncells_def_derivative_pixels(self):
        if self.params.refiner.refine_Ndef:
            npix = len(self.Modelers[self._i_shot].all_data)
            self.dNdef = [d[:npix].as_numpy_array() for d in self.D.get_ncells_def_derivative_pixels()]

    def _extract_mosaic_parameter_m_derivative_pixels(self):
        pass

    def _extract_detector_distance_derivative_pixels(self):
        pass

    def _extract_panelRot_derivative_pixels(self):
        if not self.params.refiner.refine_panel_geom:
            return
        npix = len(self.Modelers[self._i_shot].all_data)
        self._panel_derivs = []
        for pid in PAN_OFS_IDS + PAN_XYZ_IDS:
            try:
                d = self.D.get_derivative_pixels(pid).as_numpy_array()[:npix]
            except ValueError:
                d = None
            self._panel_derivs.append(d)

    def _extract_panelXYZ_derivative_pixels(self):
        pass  # extracted together with panelRot in _extract_panelRot_derivative_pixels

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
            if self.calc_curvatures:
                self.d2Nabc = [self.scale_fac*d for d in self.d2Nabc]

    def _scale_Ndef_derivative_pixels(self):
        if self.params.refiner.refine_Ndef:
            self.dNdef = [self.scale_fac*d for d in self.dNdef]
            if self.calc_curvatures:
                self.d2Ndef = [self.scale_fac*d for d in self.d2Ndef]

    def _scale_RotXYZ_derivative_pixels(self):
        if self.params.refiner.refine_RotXYZ:
            self.dRotXYZ = [self.scale_fac*d for d in self.dRotXYZ]
            if self.calc_curvatures:
                self.d2RotXYZ = [self.scale_fac*d for d in self.d2RotXYZ]

    def _scale_ucell_derivative_pixels(self):
        if self.params.refiner.refine_ucell:
            self.dUcell = [self.scale_fac*d for d in self.dUcell]

    def _scale_panel_derivative_pixels(self):
        if self.params.refiner.refine_panel_geom:
            self._panel_derivs = [self.scale_fac*d if d is not None else None
                                  for d in self._panel_derivs]

    def _get_per_spot_scale(self, i_shot, i_spot):
        pass

    def _scale_pixel_data(self):
        #Mod = self.Modelers[self._i_shot]
        #self.Bfactor_qterm = Mod.all_q_perpix**2 / 4.
        #self._expBq = np.exp(-self.b_fac**2 * self.Bfactor_qterm)
        #self.model_bragg_spots = self._expBq*self.scale_fac*(self._model_pix)
        self.model_bragg_spots_no_gains = self.scale_fac_no_gains*self._model_pix
        self.model_bragg_spots = self.scale_fac*self._model_pix
        self._scale_Fcell_derivative_pixels()
        self._scale_Nabc_derivative_pixels()
        self._scale_Ndef_derivative_pixels()
        self._scale_RotXYZ_derivative_pixels()
        self._scale_ucell_derivative_pixels()
        self._scale_panel_derivative_pixels()

    def _update_ucell(self):
        if self.params.refiner.refine_ucell:
            ucell_man = self.Modelers[self._i_shot].PAR.ucell_man
            ucell_man.variables = self._get_ucell_vars(self._i_shot)
            self.D.Bmatrix = ucell_man.B_recipspace
            for i_uc in range(self.n_ucell_param):
                self.D.set_ucell_derivative_matrix(
                    i_uc + UCELL_ID_OFFSET,
                    ucell_man.derivative_matrices[i_uc])
        else:
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
        self._diag_fterm_volume = 0.0
        self._diag_fterm_chisq = 0.0
        self._diag_neg_v_count = 0
        self._diag_neg_lam_count = 0
        self._diag_neg_v_shots = 0
        self._diag_min_multi_skip = 0
        self._diag_min_multi_skip_pix = 0
        self._eval_shot_data = []

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
        self._collect_shoebox_scores = (self.saveZ_freq is not None
                                        and self.target_eval_count % self.saveZ_freq == 0)
        self._per_shoebox_data = [] if self._collect_shoebox_scores else None

        for self._i_shot in self.shot_ids:
            self._set_current_gain_per_pixel()
            gains = self.Modelers[self._i_shot].all_gain
            self.scale_fac_no_gains = self._get_spot_scale(self._i_shot)**2
            self.scale_fac = gains*self._get_spot_scale(self._i_shot)**2

            self.b_fac = self._get_bfactor(self._i_shot)
            self.D.Bfactor_image = self.b_fac
            if self._i_shot == self.shot_ids[0] and self.iterations < 2:
                print("STAGE2_BFACTOR: shot=%d B=%.4f (set on D.Bfactor_image)" % (self._i_shot, self.b_fac), flush=True)

            # TODO: Omatrix update? All crystal models here should have the same to_primitive operation, ideally
            #LOGGER.info("update models shot %d " % self._i_shot)
            self._update_beams()
            self._update_umatrix()
            self._update_ucell()
            self._update_ncells()
            self._update_ncells_def()
            self._update_rotXYZ()
            self._update_eta()  # mosaic spread
            self._symmetrize_Flatt()
            self._update_dxtbx_detector()
            self._update_panel_geom()
            self._update_sausages()
            self._update_gonio()

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

            if self._collect_shoebox_scores:
                MOD = self.Modelers[self._i_shot]
                modeler_path = ""
                if hasattr(MOD, 'pandas_table_row') and 'stage1_output_img' in MOD.pandas_table_row.index:
                    modeler_path = str(MOD.pandas_table_row['stage1_output_img'])
                # Compute per-pixel fterm for chi-sq decomposition
                fterm_perpix = (self.log2pi + self.log_v + self.u*self.u*self.one_over_v) / MOD.all_freq
                self._spot_Zscores = []
                for i_fcell in MOD.unique_i_fcell:
                    for slc in MOD.i_fcell_slices[i_fcell]:
                        trus = MOD.all_trusted[slc]
                        if not np.any(trus):
                            continue
                        Z_roi = self._Zscore[slc][trus]
                        dat_roi = MOD.all_data[slc][trus]
                        mod_roi = self.model_Lambda[slc][trus]
                        roi_sigZ = Z_roi.std() if len(Z_roi) > 1 else np.nan
                        roi_chisq = float(0.5 * fterm_perpix[slc][trus].sum())
                        # Pearson CC between data and model
                        cc = np.nan
                        if len(dat_roi) > 2:
                            d_dm = dat_roi - dat_roi.mean()
                            m_dm = mod_roi - mod_roi.mean()
                            denom = np.sqrt((d_dm**2).sum() * (m_dm**2).sum())
                            if denom > 0:
                                cc = float((d_dm * m_dm).sum() / denom)
                        self._spot_Zscores.append((i_fcell, roi_sigZ))
                        self._per_shoebox_data.append(
                            (self._i_shot, i_fcell, float(roi_sigZ), cc,
                             int(trus.sum()), roi_chisq, modeler_path))
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
            shot_chisq = self._target_accumulate()
            self.target_functional += shot_chisq
            self._spot_scale_derivatives()
            self._Bfactor_derivatives()
            self._accumulate_Nabc_derivatives()
            self._accumulate_Ndef_derivatives()
            self._accumulate_RotXYZ_derivatives()
            self._accumulate_ucell_derivatives()
            self._Fcell_derivatives()
            self._gain_region_derivatives()
            self._accumulate_panel_derivatives()

            trusted = self.Modelers[self._i_shot].all_trusted
            shot_sigZ = np.std(self._Zscore[trusted])
            self.all_sigZ.append(shot_sigZ)
            MOD = self.Modelers[self._i_shot]
            self._eval_shot_data.append((
                self._i_shot,
                float(self._get_spot_scale(self._i_shot)**2),  # G
                float(self._get_bfactor(self._i_shot)),        # B
                float(shot_sigZ),
                float(shot_chisq),
                len(MOD.unique_i_fcell),  # n_regions
            ))
        tshots = time.time()-tshots
        LOGGER.info("Time rank worked on shots=%.4f" % tshots)
        self._MPI_barrier()
        tmpi = time.time()
        LOGGER.info("MPI aggregation of func and grad")
        self._mpi_aggregation()
        tmpi = time.time() - tmpi
        LOGGER.info("Time for MPIaggregation=%.4f" % tmpi)

        self._gain_restraints()
        self._bfactor_restraints()
        self._Fhkl_restraints()

        # --- per-shot params (every eval) ---
        self._gathered_shot_data = self._gather_and_save_shot_params()

        # --- per-shot sigZ + per-shoebox scoring ---
        self._save_per_shot_sigZ()

        # --- Tier 0/0.5 diagnostics ---
        self._log_tier0_diagnostics()

        LOGGER.info("Aliases")
        self._f = self.target_functional
        self._g = self.g = self.grad
        self.d = self.curv
        LOGGER.info("curvature analysis")
        self._curvature_analysis()

        # reset ROI pixels TODO: is this necessary
        LOGGER.info("Zero pixels")
        self.D.raw_pixels_roi *= 0

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

    def _log_tier0_diagnostics(self):
        """Tier 0 + 0.5 per-evaluation diagnostics. All ranks participate in reduce, rank 0 writes."""
        # all ranks must participate in reduce
        fterm_vol = COMM.reduce(self._diag_fterm_volume, MPI.SUM, root=0)
        fterm_chisq = COMM.reduce(self._diag_fterm_chisq, MPI.SUM, root=0)
        neg_v = COMM.reduce(self._diag_neg_v_count, MPI.SUM, root=0)
        neg_lam = COMM.reduce(self._diag_neg_lam_count, MPI.SUM, root=0)
        neg_v_shots = COMM.reduce(self._diag_neg_v_shots, MPI.SUM, root=0)
        multi_skip = COMM.reduce(self._diag_min_multi_skip, MPI.SUM, root=0)
        multi_skip_pix = COMM.reduce(self._diag_min_multi_skip_pix, MPI.SUM, root=0)

        if not self.I_AM_ROOT:
            return

        x_np = self.x.as_numpy_array()
        g_np = self.grad.as_numpy_array()
        f = self.target_functional

        # --- global norms ---
        g_norm2 = np.sqrt(np.sum(g_np**2))
        g_norminf = np.max(np.abs(g_np))
        x_norm2 = np.sqrt(np.sum(x_np**2))
        tradeps = self.trad_conv_eps
        conv_ratio = g_norm2 / (tradeps * max(1.0, x_norm2))
        self.gnorm = g_norm2  # replace sentinel

        # --- per-block norms ---
        n_shots = self.n_total_shots
        npp = self.n_params_per_shot
        shot_indices = np.arange(n_shots)
        scale_idx = shot_indices * npp
        bfac_idx = shot_indices * npp + 1
        nabc_idx = np.concatenate([shot_indices*npp + k for k in (2,3,4)])
        ndef_idx = np.concatenate([shot_indices*npp + k for k in (5,6,7)])
        rotxyz_idx = np.concatenate([shot_indices*npp + k for k in (8,9,10)])

        g_scale = g_np[scale_idx]
        g_bfac = g_np[bfac_idx]
        g_nabc = g_np[nabc_idx]
        g_ndef = g_np[ndef_idx]
        g_rotxyz = g_np[rotxyz_idx]
        g_fcell = g_np[self.fcell_xstart : self.fcell_xstart + self.n_global_fcell]
        g_gain = g_np[self.regions_xstart : self.regions_xstart + self.num_regions]

        x_scale = x_np[scale_idx]
        x_bfac = x_np[bfac_idx]
        x_nabc = x_np[nabc_idx]
        x_ndef = x_np[ndef_idx]
        x_rotxyz = x_np[rotxyz_idx]
        x_fcell = x_np[self.fcell_xstart : self.fcell_xstart + self.n_global_fcell]
        x_gain = x_np[self.regions_xstart : self.regions_xstart + self.num_regions]

        def block_norms(g_block, x_block):
            return (np.sqrt(np.sum(g_block**2)),
                    np.max(np.abs(g_block)) if len(g_block) > 0 else 0.0,
                    np.sqrt(np.sum(x_block**2)),
                    int(np.sum(g_block == 0)))

        scale_gn2, scale_ginf, scale_xn2, scale_gzero = block_norms(g_scale, x_scale)
        bfac_gn2, bfac_ginf, bfac_xn2, bfac_gzero = block_norms(g_bfac, x_bfac)
        nabc_gn2, nabc_ginf, nabc_xn2, nabc_gzero = block_norms(g_nabc, x_nabc)
        ndef_gn2, ndef_ginf, ndef_xn2, ndef_gzero = block_norms(g_ndef, x_ndef)
        rotxyz_gn2, rotxyz_ginf, rotxyz_xn2, rotxyz_gzero = block_norms(g_rotxyz, x_rotxyz)
        fcell_gn2, fcell_ginf, fcell_xn2, fcell_gzero = block_norms(g_fcell, x_fcell)
        gain_gn2, gain_ginf, gain_xn2, gain_gzero = block_norms(g_gain, x_gain)

        # --- Tier 0.5: line-search reconstruction ---
        step_norm2 = dd_prev = dd_new = armijo = curv_ratio = np.nan
        step_scale_n2 = step_fcell_n2 = np.nan
        if self._prev_x is not None:
            s = x_np - self._prev_x
            step_norm2 = np.sqrt(np.sum(s**2))
            dd_prev = np.dot(self._prev_g, s)
            dd_new = np.dot(g_np, s)
            c1 = 1e-4
            armijo = f - self._prev_f - c1 * dd_prev
            curv_ratio = dd_new / dd_prev if abs(dd_prev) > 1e-300 else np.nan
            step_scale_n2 = np.sqrt(np.sum(s[scale_idx]**2))
            step_fcell_n2 = np.sqrt(np.sum(s[self.fcell_xstart:self.fcell_xstart+self.n_global_fcell]**2))

        # stash for next eval
        self._prev_x = x_np.copy()
        self._prev_f = f
        self._prev_g = g_np.copy()

        # --- sigmaZ stats (already reduced in _mpi_aggregation) ---
        all_sigZ = self._reduced_all_sigZ
        sigZ_mean = np.mean(all_sigZ) if all_sigZ is not None else np.nan
        sigZ_median = np.median(all_sigZ) if all_sigZ is not None else np.nan

        # --- Fcell physical values ---
        if hasattr(self, '_fcell_at_i_fcell') and self._fcell_at_i_fcell is not None:
            fcell_min = float(self._fcell_at_i_fcell.min())
            fcell_max = float(self._fcell_at_i_fcell.max())
            fcell_mean = float(self._fcell_at_i_fcell.mean())
        else:
            fcell_min = fcell_max = fcell_mean = np.nan

        # --- Extended diagnostics from gathered shot data ---
        median_G = std_G = median_B = std_B = np.nan
        top10_chisq_frac = np.nan
        n_fcells_changed_gt_10pct = 0
        if self._gathered_shot_data is not None and len(self._gathered_shot_data) > 0:
            G_vals = np.array([r[1] for r in self._gathered_shot_data])
            B_vals = np.array([r[2] for r in self._gathered_shot_data])
            chisq_vals = np.array([r[4] for r in self._gathered_shot_data])
            median_G = float(np.median(G_vals))
            std_G = float(np.std(G_vals))
            median_B = float(np.median(B_vals))
            std_B = float(np.std(B_vals))
            # Top-10% shoebox dominance: fraction of total chi-sq from top 10% of shots
            if len(chisq_vals) > 0:
                sorted_chisq = np.sort(chisq_vals)[::-1]
                n10 = max(1, len(sorted_chisq) // 10)
                top10_chisq_frac = float(sorted_chisq[:n10].sum() / sorted_chisq.sum()) if sorted_chisq.sum() > 0 else np.nan

        # Fcell change from initial values
        if (hasattr(self, '_fcell_at_i_fcell') and self._fcell_at_i_fcell is not None
                and hasattr(self, 'fcell_init_from_i_fcell') and self.fcell_init_from_i_fcell is not None):
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_change = np.abs(self._fcell_at_i_fcell - self.fcell_init_from_i_fcell) / np.maximum(self.fcell_init_from_i_fcell, 1e-30)
            n_fcells_changed_gt_10pct = int(np.sum(rel_change > 0.1))

        # --- write CSV ---
        if self._diag_csv_file is None and self.output_dir is not None:
            diag_path = os.path.join(self.output_dir, "stage2_diagnostics.csv")
            self._diag_csv_file = open(diag_path, "w", newline="")
            self._diag_csv_writer = csv.writer(self._diag_csv_file)
            self._diag_csv_writer.writerow([
                "eval", "iter", "wall_time",
                "f", "f_volume", "f_chisq", "f_diff",
                "g_norm2", "g_norminf", "x_norm2", "tradeps",
                "conv_ratio", "n_total_params", "n_global_fcell", "n_total_shots", "num_regions",
                "scale_gn2", "scale_ginf", "scale_xn2", "scale_gzero",
                "bfac_gn2", "bfac_ginf", "bfac_xn2", "bfac_gzero",
                "nabc_gn2", "nabc_ginf", "nabc_xn2", "nabc_gzero",
                "ndef_gn2", "ndef_ginf", "ndef_xn2", "ndef_gzero",
                "rotxyz_gn2", "rotxyz_ginf", "rotxyz_xn2", "rotxyz_gzero",
                "fcell_gn2", "fcell_ginf", "fcell_xn2", "fcell_gzero",
                "gain_gn2", "gain_ginf", "gain_xn2", "gain_gzero",
                "neg_v_pix", "neg_lam_pix", "neg_v_shots",
                "min_multi_skip", "min_multi_skip_pix",
                "sigZ_mean", "sigZ_median",
                "fcell_min", "fcell_max", "fcell_mean",
                "step_norm2", "dd_prev", "dd_new", "armijo", "curv_ratio",
                "step_scale_n2", "step_fcell_n2",
                "median_G", "std_G", "median_B", "std_B",
                "top10_chisq_frac", "n_fcells_changed_gt_10pct",
            ])

        if self._diag_csv_writer is not None:
            f_prev = self.f_vals[-1] if self.f_vals else np.nan
            f_diff = f - f_prev
            self._diag_csv_writer.writerow([
                self.target_eval_count, self.iterations, time.time(),
                f, fterm_vol, fterm_chisq, f_diff,
                g_norm2, g_norminf, x_norm2, tradeps,
                conv_ratio, self.n_total_params, self.n_global_fcell, self.n_total_shots, self.num_regions,
                scale_gn2, scale_ginf, scale_xn2, scale_gzero,
                bfac_gn2, bfac_ginf, bfac_xn2, bfac_gzero,
                nabc_gn2, nabc_ginf, nabc_xn2, nabc_gzero,
                ndef_gn2, ndef_ginf, ndef_xn2, ndef_gzero,
                rotxyz_gn2, rotxyz_ginf, rotxyz_xn2, rotxyz_gzero,
                fcell_gn2, fcell_ginf, fcell_xn2, fcell_gzero,
                gain_gn2, gain_ginf, gain_xn2, gain_gzero,
                neg_v, neg_lam, neg_v_shots,
                multi_skip, multi_skip_pix,
                sigZ_mean, sigZ_median,
                fcell_min, fcell_max, fcell_mean,
                step_norm2, dd_prev, dd_new, armijo, curv_ratio,
                step_scale_n2, step_fcell_n2,
                median_G, std_G, median_B, std_B,
                top10_chisq_frac, n_fcells_changed_gt_10pct,
            ])
            self._diag_csv_file.flush()

        LOGGER.info("DIAG eval=%d iter=%d conv_ratio=%.6e g_norm2=%.6e x_norm2=%.6e tradeps=%.2e "
                     "n_params=%d n_fcell=%d dd_prev=%.4e armijo=%.4e step=%.4e neg_v=%d"
                     % (self.target_eval_count, self.iterations, conv_ratio, g_norm2, x_norm2, tradeps,
                        self.n_total_params, self.n_global_fcell, dd_prev, armijo, step_norm2, neg_v))
        LOGGER.info("BLOCKS gn2: Scale=%.3e B=%.3e Nabc=%.3e Ndef=%.3e RotXYZ=%.3e Fcell=%.3e Gain=%.3e | "
                     "xn2: Scale=%.3e Fcell=%.3e | step_fcell=%.3e"
                     % (scale_gn2, bfac_gn2, nabc_gn2, ndef_gn2, rotxyz_gn2, fcell_gn2, gain_gn2,
                        scale_xn2, fcell_xn2, step_fcell_n2))

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
                # count skipped reflections and their trusted pixel count
                self._diag_min_multi_skip += 1
                n_pix = sum(int(MOD.all_trusted[slc].sum()) for slc in MOD.i_fcell_slices[i_fcell])
                self._diag_min_multi_skip_pix += n_pix
                continue

            xpos = self.fcell_xstart + i_fcell
            Famp = self._fcell_at_i_fcell[i_fcell]
            sig = self.fcell_sigmas_from_i_fcell if np.isscalar(self.fcell_sigmas_from_i_fcell) \
                else self.fcell_sigmas_from_i_fcell[i_fcell]

            for slc in MOD.i_fcell_slices[i_fcell]:
                self.fcell_dI_dtheta = self.fcell_deriv[slc]

                if self.log_fcells:
                    # case 2 rescaling: d(I)/d(x) = F * dI/dF (log parameterization)
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
                    fcell_d2I_dtheta = self.fcell_second_deriv[slc]
                    if self.log_fcells:
                        # d2I/dx2 = F*dI/dF + F^2*d2I/dF^2 for log param
                        d2 = sig_times_fcell*self.fcell_dI_dtheta + sig_times_fcell*sig_times_fcell*fcell_d2I_dtheta
                    else:
                        d2 = sig*sig*fcell_d2I_dtheta
                    # manual curv accumulate for this slice (can't use _curv_accumulate which uses full-shot _is_trusted)
                    one_over_v_slc = self.one_over_v[slc]
                    cterm = one_over_v_slc * (d2*self.one_minus_2u_minus_u_squared_over_v[slc] -
                                d*d*(self.one_over_v_times_one_minus_2u_minus_u_squared_over_v[slc] -
                                     (2 + 2*self.u_times_one_over_v[slc] + self.u_u_one_over_v[slc]*one_over_v_slc)))
                    self.curv[xpos] += 0.5 * cterm[trust].sum()

    def _accumulate_Nabc_derivatives(self):
        if not self.params.refiner.refine_Nabc:
            return
        Mod = self.Modelers[self._i_shot]
        for i_n in range(3):
            p = Mod.PAR.Nabc[i_n]
            d = p.get_deriv(self.x[p.xpos],  self.dNabc[i_n])
            self.grad[p.xpos] += self._grad_accumulate(d)
            if self.calc_curvatures:
                d2 = p.get_second_deriv(self.x[p.xpos], self.dNabc[i_n], self.d2Nabc[i_n])
                self.curv[p.xpos] += self._curv_accumulate(d, d2)

    def _accumulate_Ndef_derivatives(self):
        if not self.params.refiner.refine_Ndef:
            return
        Mod = self.Modelers[self._i_shot]
        for i_n in range(3):
            p = Mod.PAR.Ndef[i_n]
            d = p.get_deriv(self.x[p.xpos], self.dNdef[i_n])
            self.grad[p.xpos] += self._grad_accumulate(d)
            if self.calc_curvatures:
                d2 = p.get_second_deriv(self.x[p.xpos], self.dNdef[i_n], self.d2Ndef[i_n])
                self.curv[p.xpos] += self._curv_accumulate(d, d2)

    def _accumulate_RotXYZ_derivatives(self):
        if not self.params.refiner.refine_RotXYZ:
            return
        Mod = self.Modelers[self._i_shot]
        for i_r in range(3):
            p = Mod.PAR.RotXYZ_params[i_r]
            d = p.get_deriv(self.x[p.xpos], self.dRotXYZ[i_r])
            self.grad[p.xpos] += self._grad_accumulate(d)
            if self.calc_curvatures:
                d2 = p.get_second_deriv(self.x[p.xpos], self.dRotXYZ[i_r], self.d2RotXYZ[i_r])
                self.curv[p.xpos] += self._curv_accumulate(d, d2)

    def _accumulate_ucell_derivatives(self):
        if not self.params.refiner.refine_ucell:
            return
        Mod = self.Modelers[self._i_shot]
        for i_uc in range(self.n_ucell_param):
            p = Mod.PAR.ucell[i_uc]
            d = p.get_deriv(self.x[p.xpos], self.dUcell[i_uc])
            self.grad[p.xpos] += self._grad_accumulate(d)

    def _accumulate_panel_derivatives(self):
        if not self.params.refiner.refine_panel_geom:
            return
        MOD = self.Modelers[self._i_shot]
        names = "RotOrth", "RotFast", "RotSlow", "ShiftX", "ShiftY", "ShiftZ"
        for group_id in MOD.unique_panel_group_ids:
            for pixel_rng in MOD.group_id_slices[group_id]:
                trusted_pixels = MOD.all_trusted[pixel_rng]
                for i_name, name in enumerate(names):
                    par_name = "group%d_%s" % (group_id, name)
                    det_param = self.panel_params[par_name]
                    if det_param.fix:
                        continue
                    if self._panel_derivs[i_name] is None:
                        continue
                    pixderivs = self._panel_derivs[i_name][pixel_rng][trusted_pixels]
                    pixderivs = det_param.get_deriv(self.x[det_param.xpos], pixderivs)
                    # Use common_grad_term for panel derivatives (same target function)
                    gterm = self.common_grad_term[pixel_rng][trusted_pixels]
                    freq = MOD.all_freq[pixel_rng][trusted_pixels]
                    g_accum = 0.5 * (pixderivs * gterm / freq).sum()
                    self.grad[det_param.xpos] += g_accum

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
        if not self.params.refiner.refine_Bfactor:
            return
        p = self.Modelers[self._i_shot].PAR.B
        d_raw = self.scale_fac * self._dB
        d = p.get_deriv(self.x[p.xpos], d_raw)
        grad_val = self._grad_accumulate(d)
        self.grad[p.xpos] += grad_val
        if self.calc_curvatures:
            d2_raw = self.scale_fac * self._d2B
            d2 = p.get_second_deriv(self.x[p.xpos], d_raw, d2_raw)
            self.curv[p.xpos] += self._curv_accumulate(d, d2)
        if self._i_shot == self.shot_ids[0] and self.iterations < 2:
            print("STAGE2_BFACTOR: shot=%d B=%.4f xpos=%d grad=%.6e dB_range=[%.4e,%.4e]"
                  % (self._i_shot, self.b_fac, p.xpos, grad_val,
                     self._dB.min(), self._dB.max()), flush=True)

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
        self._reduced_all_sigZ = COMM.reduce(self.all_sigZ)
        if COMM.rank==0:
            LOGGER.info("F=%10.7e, sigmaZ: mean=%f, median=%f" % (self.target_functional, np.mean(self._reduced_all_sigZ), np.median(self._reduced_all_sigZ) ))

    def _get_curvature_block_indices(self):
        """Build dict mapping block name -> array of x-vector indices for ALL shots (global)."""
        blocks = {}
        npp = self.n_params_per_shot
        scale_idx, b_idx, nabc_idx, ndef_idx, rot_idx, ucell_idx = [], [], [], [], [], []
        # Use shot_mapping (global across all ranks) not shot_ids (local to this rank)
        for sid, i_shot in self.shot_mapping.items():
            base = i_shot * npp
            scale_idx.append(base)
            b_idx.append(base + 1)
            nabc_idx.extend([base+2, base+3, base+4])
            ndef_idx.extend([base+5, base+6, base+7])
            rot_idx.extend([base+8, base+9, base+10])
            if self.params.refiner.refine_ucell:
                for i_uc in range(self.n_ucell_param):
                    ucell_idx.append(base + N_PARAM_PER_SHOT_BASE + i_uc)
        blocks["Scale"] = np.array(scale_idx, dtype=int)
        blocks["B"] = np.array(b_idx, dtype=int)
        blocks["Nabc"] = np.array(nabc_idx, dtype=int)
        blocks["Ndef"] = np.array(ndef_idx, dtype=int)
        blocks["RotXYZ"] = np.array(rot_idx, dtype=int)
        if ucell_idx:
            blocks["Ucell"] = np.array(ucell_idx, dtype=int)
        blocks["Fcell"] = np.arange(self.fcell_xstart, self.fcell_xstart + self.n_global_fcell, dtype=int)
        if self.num_regions > 0:
            blocks["Gain"] = np.arange(self.regions_xstart, self.regions_xstart + self.num_regions, dtype=int)
        if self.params.refiner.refine_panel_geom and self.n_panel_groups > 0:
            blocks["Panel"] = np.arange(self.panel_xstart, self.panel_xstart + self.n_panel_groups * 6, dtype=int)
        return blocks

    def _print_curvature_diagnostics(self, curv_np, blocks, label=""):
        """Print per-block curvature statistics (rank 0 only)."""
        if COMM.rank != 0:
            return
        prefix = "CURV_DIAG eval=%d%s" % (self.target_eval_count, (" "+label) if label else "")
        n_total = len(curv_np)
        n_neg = (curv_np < 0).sum()
        n_zero = (curv_np == 0).sum()
        pos = curv_np[curv_np > 0]
        msg = ["%s: total=%d neg=%d(%.1f%%) zero=%d" %
               (prefix, n_total, n_neg, 100.*n_neg/max(n_total,1), n_zero)]
        if len(pos):
            msg.append("  pos_range=[%.3e, %.3e] median=%.3e" % (pos.min(), pos.max(), np.median(pos)))
        # per-block breakdown
        for bname, idx in blocks.items():
            if len(idx) == 0:
                continue
            bc = curv_np[idx]
            n_b = len(bc)
            neg_b = (bc < 0).sum()
            pos_b = bc[bc > 0]
            zero_b = (bc == 0).sum()
            if n_b == 0:
                continue
            line = "  %-6s: n=%-6d neg=%d(%.1f%%) zero=%d" % (bname, n_b, neg_b, 100.*neg_b/n_b, zero_b)
            if len(pos_b):
                line += " pos_med=%.3e [%.3e,%.3e]" % (np.median(pos_b), pos_b.min(), pos_b.max())
            if neg_b > 0:
                neg_vals = bc[bc < 0]
                line += " neg_med=%.3e [%.3e,%.3e]" % (np.median(neg_vals), neg_vals.min(), neg_vals.max())
            msg.append(line)
        print("\n".join(msg), flush=True)

    def _clamp_curvatures(self, curv_np, blocks):
        """Clamp negative curvatures and floor small ones so they can be used for L-BFGS preconditioning.

        After clamping negatives, applies a global floor so the dynamic range
        of curvatures doesn't exceed curvature_max_ratio (default 1e4).
        This prevents blocks with tiny curvatures (e.g. Scale) from getting
        disproportionately large steps.
        """
        mode = self.params.refiner.curvature_clamp
        neg = curv_np < 0
        n_neg = neg.sum()

        if mode == "abs":
            curv_np = np.abs(curv_np)

        elif mode == "median":
            # Per-block: replace negatives with median of positives
            for bname, idx in blocks.items():
                if len(idx) == 0:
                    continue
                bc = curv_np[idx]
                pos = bc > 0
                neg_b = ~pos & (bc != 0)  # skip zeros (unrefined params)
                if not neg_b.any():
                    continue
                if pos.any():
                    med = np.median(bc[pos])
                else:
                    med = np.median(np.abs(bc[bc != 0])) if (bc != 0).any() else 1.0
                bc[neg_b] = med
                curv_np[idx] = bc

            # Fallback: clamp any remaining negatives not in named blocks
            still_neg = curv_np < 0
            if still_neg.any():
                pos_all = curv_np[curv_np > 0]
                fallback = np.median(pos_all) if len(pos_all) else 1.0
                curv_np[still_neg] = fallback
                if COMM.rank == 0:
                    print("CURV_DIAG: fallback clamped %d remaining negatives -> %.3e"
                          % (still_neg.sum(), fallback), flush=True)

        # Per-block floor: compress dynamic range WITHIN each block.
        # Each block's curvatures are floored at block_median / max_ratio.
        # This preserves intra-block curvature information (e.g. strong vs weak Fcells)
        # while preventing outlier shots from dominating the step direction.
        # Cross-block scaling is handled by the curvature magnitudes themselves.
        max_ratio = self.params.refiner.curvature_max_ratio
        if max_ratio is not None and max_ratio > 0:
            for bname, idx in blocks.items():
                if len(idx) == 0:
                    continue
                bc = curv_np[idx]
                pos = bc > 0
                pos_vals = bc[pos]
                if len(pos_vals) == 0:
                    continue
                block_med = np.median(pos_vals)
                block_floor = block_med / max_ratio
                n_floored = (pos & (bc < block_floor)).sum()
                bc[pos] = np.maximum(bc[pos], block_floor)
                curv_np[idx] = bc
                if COMM.rank == 0 and n_floored > 0:
                    print("CURV_DIAG: floor[%s] %d/%d below %.3e (med=%.3e, ratio=%.0e)"
                          % (bname, n_floored, len(pos_vals), block_floor, block_med, max_ratio), flush=True)

        # Neutralize Scale curvatures: replace with Fcell median.
        # Scale has pathologically small curvatures (~0.05) due to exp parameterization,
        # causing L-BFGS to overshoot. Setting to Fcell median (~100) gives neutral
        # preconditioning while keeping all other blocks at physically meaningful values.
        if self.params.refiner.curvature_neutralize_scale and "Scale" in blocks and "Fcell" in blocks:
            fcell_pos = curv_np[blocks["Fcell"]]
            fcell_pos = fcell_pos[fcell_pos > 0]
            if len(fcell_pos):
                replacement = np.median(fcell_pos)
                scale_idx = blocks["Scale"]
                old_med = np.median(curv_np[scale_idx][curv_np[scale_idx] > 0]) if (curv_np[scale_idx] > 0).any() else 0
                curv_np[scale_idx] = replacement
                if COMM.rank == 0:
                    print("CURV_DIAG: neutralize Scale: %.3e -> %.3e (Fcell median)" % (old_med, replacement), flush=True)

        # Per-block normalization: rescale each block so all have the same median.
        # Preserves intra-block relative curvatures (strong Fcells vs weak Fcells)
        # but equalizes inter-block step scaling so no block dominates L-BFGS direction.
        if self.params.refiner.curvature_normalize:
            block_meds = {}
            for bname, idx in blocks.items():
                if len(idx) == 0:
                    continue
                bc = curv_np[idx]
                pos_vals = bc[bc > 0]
                if len(pos_vals):
                    block_meds[bname] = np.median(pos_vals)
            if block_meds:
                target = np.median(list(block_meds.values()))
                if COMM.rank == 0:
                    parts = ["CURV_DIAG: normalize target=%.3e" % target]
                for bname, idx in blocks.items():
                    if bname not in block_meds or block_meds[bname] == 0:
                        continue
                    scale = target / block_meds[bname]
                    bc = curv_np[idx]
                    bc[bc > 0] *= scale
                    curv_np[idx] = bc
                    if COMM.rank == 0:
                        parts.append("  %s: %.3e -> %.3e (x%.2e)" %
                                     (bname, block_meds[bname], target, scale))
                if COMM.rank == 0:
                    print("\n".join(parts), flush=True)

        return curv_np

    def _verify_diag(self):
        """Override base _verify_diag (which has stale IPython embed).
        Sets curvatures to 1000 where gradient=0, asserts positive, inverts for preconditioning."""
        sel = (self.g != 0)
        self.d.set_selected(~sel, 1000)
        assert self.d.select(sel).all_gt(0), \
            "Negative curvatures remain after clamping (%d of %d)" % (
                (self.d.select(sel) <= 0).count(True), sel.count(True))
        self.d = 1. / self.d

    def _curvature_analysis(self):
        self.tot_neg_curv = 0
        self.neg_curv_shots = []
        if self.calc_curvatures:
            curv_np = self.curv.as_numpy_array()
            self.is_negative_curvature = curv_np < 0
            self.tot_neg_curv = sum(self.is_negative_curvature)

            blocks = self._get_curvature_block_indices()

            # Print diagnostics before clamping
            self._print_curvature_diagnostics(curv_np, blocks, label="pre-clamp")

            # Apply clamping if enabled
            clamp_mode = self.params.refiner.curvature_clamp
            if clamp_mode != "none" and self.tot_neg_curv > 0:
                curv_np = self._clamp_curvatures(curv_np, blocks)
                self.curv = flex.double(curv_np)
                self.tot_neg_curv = 0
                # Print diagnostics after clamping
                self._print_curvature_diagnostics(curv_np, blocks, label="post-clamp")

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
        M = self.Modelers[self._i_shot]
        vol_term = (self.log2pi + self.log_v) / M.all_freq
        chisq_term = (self.u*self.u*self.one_over_v) / M.all_freq
        if self._is_trusted is not None:
            vol_term = vol_term[self._is_trusted]
            chisq_term = chisq_term[self._is_trusted]
        vol_sum = 0.5 * vol_term.sum()
        chisq_sum = 0.5 * chisq_term.sum()
        self._diag_fterm_volume += vol_sum
        self._diag_fterm_chisq += chisq_sum
        return vol_sum + chisq_sum

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
        # non-smoothness counters
        neg_lam = (self.model_Lambda <= 0).ravel()
        n_neg_lam_trusted = int(neg_lam[Mod.all_trusted].sum())
        n_neg_v_trusted = int(v_is_neg[Mod.all_trusted].sum())
        self._diag_neg_lam_count += n_neg_lam_trusted
        self._diag_neg_v_count += n_neg_v_trusted
        if n_neg_v_trusted > 0:
            self._diag_neg_v_shots += 1
        if any(v_is_neg[Mod.all_trusted]):
            LOGGER.warning(Bcolors.WARNING+"NEGATIVE INTENSITY IN MODEL!"+Bcolors.ENDC)
        self.log_v = np.log(v)
        self.log_v[v <= 0] = 0

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
