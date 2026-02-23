from __future__ import absolute_import, division, print_function
import time
import os
import json
from copy import deepcopy
import pandas
from simtbx.diffBragg.geom_utils import detector_model_derivs, update_detector
from simtbx.diffBragg.geom_utils import PAN_OFS_IDS, PAN_XYZ_IDS
from xfel.poly.recompute_mosaic_params import extract_mosaic_parameters_using_lambda_spread

from simtbx.diffBragg import hopper_io
import numpy as np
from cctbx import crystal, miller
from scipy.optimize import dual_annealing, basinhopping
from collections import Counter
from scitbx.matrix import sqr, col
from scipy.ndimage import binary_dilation
from dxtbx.model.experiment_list import ExperimentListFactory
from dxtbx.model import Spectrum
try:  # TODO keep backwards compatibility until we close the nxmx_writer_experimental branch
    from serialtbx.detector.jungfrau import get_pedestalRMS_from_jungfrau
except ModuleNotFoundError:
    from xfel.util.jungfrau import get_pedestalRMS_from_jungfrau
from simtbx.nanoBragg.utils import downsample_spectrum
from cctbx.array_family import flex
from simtbx.diffBragg import utils
from simtbx.diffBragg.refiners.parameters import RangedParameter, Parameters, PositiveParameter
from simtbx.diffBragg.attr_list import NB_BEAM_ATTRS, NB_CRYST_ATTRS, DIFFBRAGG_ATTRS
from simtbx.diffBragg import psf

try:
    from line_profiler import LineProfiler
except ImportError:
    LineProfiler = None


import logging
MAIN_LOGGER = logging.getLogger("diffBragg.main")
PROFILE_LOGGER = logging.getLogger("diffBragg.profile")

ROTX_ID = 0
ROTY_ID = 1
ROTZ_ID = 2
ROTXYZ_IDS = ROTX_ID, ROTY_ID, ROTZ_ID
NCELLS_ID = 9
NCELLS_ID_OFFDIAG = 21
UCELL_ID_OFFSET = 3
DETZ_ID = 10
FHKL_ID = 11
ETA_ID = 19
DIFFUSE_ID = 23
GONIO_ANGLE_ID = 24
BFACTOR_ID = 25
BFACTOR_ANISO_ID = 26
LAMBDA_IDS = 12, 13

DEG = 180 / np.pi


def write_SIM_logs(SIM, log=None, lam=None):
    """
    Logs properties of SIM.D (diffBragg instance), and SIM.crystal, SIM.beam (nanoBragg beam and crystal)
    These are important for reproducing results
    :param SIM: sim_data instance used during hopper refinement, member of the data modeler
    :param log: optional log file to dump attributes of SIM
    :param lam: optional lambda file to dump the spectra to disk, can be read usint diffBragg.utils.load_spectra_file
    """
    if log is not None:
        with open(log, "w") as o:
            print("<><><><><>", file=o)
            print("DIFFBRAGG", file=o)
            print("<><><><><>", file=o)
            for attr in DIFFBRAGG_ATTRS:
                val = getattr(SIM.D, attr)
                print(attr+": ", val, file=o)
            if SIM.refining_Fhkl is not None and SIM.refining_Fhkl:
                channels = SIM.D.get_Fhkl_channels()
                channels = ",".join(map(str, channels))
                print("Fhkl channels: ", channels, file=o)
            print("\n<><><>", file=o)
            print("BEAM", file=o)
            print("<><><>", file=o)
            for attr in NB_BEAM_ATTRS:
                val = getattr(SIM.beam, attr)
                print(attr+": ", val, file=o)
            print("\n<><><><>", file=o)
            print("CRYSTAL", file=o)
            print("<><><><>", file=o)
            for attr in NB_CRYST_ATTRS:
                val = getattr(SIM.crystal, attr)
                print(attr+": ", val, file=o)
    if lam is not None:
        wavelen, wt = zip(*SIM.beam.spectrum)
        utils.save_spectra_file(lam, wavelen, wt)


def free_SIM_mem(SIM):
    """
    Frees memory allocated to host CPU (and GPU device, if applicable).
    Using this method is critical for serial applications!
    :param SIM: sim_data instance used during hopper refinement, member of the data modeler
    """
    SIM.D.free_all()
    SIM.D.free_Fhkl2()
    try:
        SIM.D.gpu_free()
    except TypeError:
        pass  # occurs on CPU-only builds


def finalize_SIM(SIM, log=None, lam=None):
    """
    thin wrapper to free_SIM_mem and write_SIM_logs
    :param SIM: sim_data instance used during hopper refinement, member of the data modeler
    :param log: optional log file to dump attributes of SIM
    :param lam: optional lambda file to dump the spectra to disk, can be read usint diffBragg.utils.load_spectra_file
    """
    write_SIM_logs(SIM, log, lam)
    free_SIM_mem(SIM)


class DataModeler:

    """
    The data modeler stores information in two ways:
    1- lists whose length is the number of pixels being modeled
    2- lists whose length is the number of shoeboxes being modeled

    for example if one is modeling 3 shoeboxes whose dimensions are 10x10, then
    the objects below like self.all_data will have length 300, and other objects like self.selection_flags
    will have length 3
    """

    def __init__(self, params):
        """ params is a simtbx.diffBragg.hopper phil"""
        self.roiScalesPerPix = 1
        self.Fhkl_channel_ids = None # this should be a list the same length as the spectrum. specifies which Fhkl to refine, depending on the energy (for eg. two color experiment)
        self.no_rlp_info = False  # whether rlps are stored in the refls table
        self.params = params  # phil params (see diffBragg/phil.py)
        self._abs_path_params()
        self.num_xtals = 1
        self.E = None  # placeholder for the dxtbx.model.Experiment instance
        self.pan_fast_slow =None  # (pid, fast, slow) per pixel
        self.all_background =None  # background model per pixel (photon units)
        self.roi_id =None  # region of interest ID per pixel
        self.u_id = None  # set of unique region of interest ids
        self.all_freq = None  # flag for the h,k,l frequency of the observed pixel
        self.best_model = None  # best model value at each pixel
        self.best_model_includes_background = False  # whether the best model includes the background scattering estimate
        self.all_nominal_hkl_p1 = None  # nominal p1 hkl at each pixel
        self.all_nominal_hkl = None  # nominal hkl at each pixel
        self.all_data =None  # data at each pixel (photon units)
        self.all_sigma_rdout = None  # this is either a float or an array. if the phil param use_perpixel_dark_rms=True, then these are different per pixel, per shot
        self.all_gain = None  # gain value per pixel (used during diffBragg/refiners/stage_two_refiner)
        self.all_sigmas =None  # error model for each pixel (photon units)
        self.all_trusted =None  # trusted pixel flags (True is trusted and therefore used during refinement)
        self.npix_total =None  # total number of pixels
        self.all_fast =None  # fast-scan coordinate per pixel
        self.all_slow =None  # slow-scan coordinate per pixel
        self.all_pid = None  # panel id per pixel
        self.all_zscore = None # the estimated z-score values for each pixel, updated each iteration in the Target class
        self.rois=None  # region of interest (per spot)
        self.pids=None  # panel id (per spot)
        self.tilt_abc=None  # background plane constants (per spot), a,b are fast,slow scan components, c is offset
        self.selection_flags=None  # whether the spot was selected for refinement (sometimes poorly conditioned spots are rejected)
        self.tilt_cov = None  # covariance estimates from background fitting (not used)
        self.simple_weights = None  # not used
        self.refls_idx = None  # position of modeled spot in original refl array
        self.refls = None  # reflection table
        self.nominal_sigma_rdout = None   # the value of the readout noise in photon units
        self.exper_name = None  # optional name specifying where dxtbx.model.Experiment was loaded from
        self.refl_name = None  # optional name specifying where dials.array_family.flex.reflection_table refls were loaded from
        self.spec_name = None  # optional name specifying spectrum file(.lam)
        self.exper_idx = 0  # optional number specifying the index of the experiment in the experiment list
        self.rank = 0  # in case DataModelers are part of an MPI program, have a rank attribute for record keeping

        self.Hi = None  # miller index (P1)
        self.Hi_asu = None  # miller index (high symmetry)
        self.target = None  # placeholder for the Target class instance
        self.nanoBragg_beam_spectrum = None  # see spectrun property of NBBeam (nanoBragg_beam.py)

        # which attributes to save when pickling a data modeler
        self.saves = ["all_data", "all_background", "all_trusted", "best_model", "nominal_sigma_rdout",
                      "rois", "pids", "tilt_abc", "selection_flags", "refls_idx", "pan_fast_slow",
                      "Hi", "Hi_asu", "roi_id", "params", "all_pid", "all_fast", "all_slow", "best_model_includes_background",
                      "all_q_perpix", "all_sigma_rdout", "all_zscore", "roiScalesPerPix"]

    def filter_bad_scores(self, checker):
        assert self.params.roi.filter_scores.enable
        assert checker is not None
        if self.best_model is None:
            raise ValueError("cannot get the best model, there is no best_model attribute")

        ave_good = num_good = 0
        for i_roi, roi_id in enumerate(self.roi_id_unique):
            roi_slc = self.roi_id_slices[roi_id][0]
            x1, x2, y1, y2 = self.rois[roi_id]
            roi_sh = y2-y1, x2-x1
            #roi_sel = self.roi_id==i_roi
            mod = self.best_model[roi_slc].reshape(roi_sh)
            if not self.best_model_includes_background:
                bg = self.all_background[roi_slc].reshape(roi_sh)
                mod += bg
            dat = self.all_data[roi_slc].reshape(roi_sh)
            score = checker.score(dat, mod)
            if score < self.params.roi.filter_scores.cutoff:
                self.all_trusted[roi_slc] = False
            else:
                num_good += 1
                ave_good += score
        if num_good > 0:
            ave_good = ave_good / num_good
            MAIN_LOGGER.info("Found %d good roi fits (average good score = %.2f)" % (num_good, ave_good))
        else:
            MAIN_LOGGER.info("Score filter removed all spots, bad fit overall!")
        return num_good

    def filter_pixels(self, thresh):
        assert self.roi_id is not None
        assert self.all_trusted is not None
        assert self.all_zscore is not None

        if not hasattr(self, 'roi_id_slices') or self.roi_id_slices is None:
            self.set_slices('roi_id')

        ntrust = self.all_trusted.sum()

        sigz_per_shoebox = []
        for roi_id in self.roi_id_unique:
            slcs = self.roi_id_slices[roi_id]
            Zs = []
            for slc in slcs:
                trusted = self.all_trusted[slc]
                Zs += list(self.all_zscore[slc][trusted])
            if not Zs:
                sigz = np.nan
            else:
                sigz = np.std(Zs)
            sigz_per_shoebox.append(sigz)
        if np.all(np.isnan(sigz_per_shoebox)):
            MAIN_LOGGER.debug("All shoeboxes are nan, nothing to filter")
            return
        med_sigz = np.median([sigz for sigz in sigz_per_shoebox if not np.isnan(sigz)])
        sigz_per_shoebox = np.nan_to_num(sigz_per_shoebox, nan=med_sigz)
        shoebox_is_bad = utils.is_outlier(sigz_per_shoebox, thresh)
        nbad_pix = 0
        for i_roi, roi_id in enumerate(self.roi_id_unique):
            if shoebox_is_bad[i_roi]:
                for slc in self.roi_id_slices[roi_id]:
                    nbad_pix += slc.stop - slc.start
                    self.all_trusted[slc] = False

        #inds = np.arange(len(self.all_trusted))
        #Zs = self.all_zscore[self.all_trusted]
        #bad = utils.is_outlier(Zs, thresh=thresh)
        #inds_trusted = inds[self.all_trusted]
        #self.all_trusted[inds_trusted[bad]] = False
        MAIN_LOGGER.debug("Added %d pixels from %d shoeboxes to the untrusted list (%d / %d trusted pixels remain)"
                          % (nbad_pix, shoebox_is_bad.sum(), self.all_trusted.sum(), len(self.all_trusted)))

    def set_spectrum(self, spectra_file=None, spectra_stride=None, total_flux=None):

        # note , the following 3 settings will only be used if spectrum_from_imageset is False and gause_spec is False
        if spectra_file is None:
            spectra_file = self.params.simulator.spectrum.filename
        if spectra_stride is None:
            spectra_stride = self.params.simulator.spectrum.stride  # will only be used if spectra_file is not None
        if total_flux is None:
            total_flux = self.params.simulator.total_flux

        if self.params.spectrum_from_imageset:
            self.nanoBragg_beam_spectrum = downsamp_spec_from_params(self.params, self.E)
        elif self.params.gen_gauss_spec:
            self.nanoBragg_beam_spectrum = set_gauss_spec(None, self.params, self.E)
        elif spectra_file is not None:
            self.nanoBragg_beam_spectrum = utils.load_spectra_file(spectra_file, total_flux, spectra_stride, as_spectrum=True)
        else:
            assert total_flux is not None
            self.nanoBragg_beam_spectrum = [(self.E.beam.get_wavelength(), total_flux)]

    def set_Fhkl_channels(self, SIM, set_in_diffBragg=True):
        if self.nanoBragg_beam_spectrum is None:
            raise AttributeError("Needs nanoBragg_beam_spectrum property first!")
        energies = np.array([utils.ENERGY_CONV / wave for wave, _ in self.nanoBragg_beam_spectrum])
        Fhkl_channel_ids = np.zeros(len(energies), int)
        for i_channel, (en1, en2) in enumerate(zip(SIM.Fhkl_channel_bounds, SIM.Fhkl_channel_bounds[1:])):
            sel = (energies >= en1) * (energies < en2)
            Fhkl_channel_ids[sel] = i_channel
        # With beam divergence, sources = n_div_angles * n_wavelengths.
        # The source ordering is: for each div angle, all wavelengths.
        # Tile the per-wavelength channel IDs for each divergence angle.
        n_sources = SIM.D.xray_beams.size()
        n_wavelengths = len(energies)
        if n_sources > n_wavelengths:
            assert n_sources % n_wavelengths == 0, \
                "sources (%d) not a multiple of wavelengths (%d)" % (n_sources, n_wavelengths)
            Fhkl_channel_ids = np.tile(Fhkl_channel_ids, n_sources // n_wavelengths)
        if set_in_diffBragg:
            SIM.D.update_Fhkl_channels(Fhkl_channel_ids)
        self.Fhkl_channel_ids = Fhkl_channel_ids

    def at_minimum(self, x, f, accept):
        self.target.iteration = 0
        self.target.all_x = []
        self.target.x0[self.target.vary] = x
        self.target.hop_iter += 1
        #self.target.minima.append((f,self.target.x0,accept))
        self.target.lowest_x = x
        try:
            # TODO get SIM and i_shot in here so we can save_up each new global minima!
            if f < self.target.lowest_f:
                 self.target.lowest_f = f
                 MAIN_LOGGER.info("New minimum found!")
                 self.save_up(self.target.x0, SIM, self.rank, i_shot=i_shot)
        except NameError:
            pass

    def _abs_path_params(self):
        """adds absolute path to certain params"""
        if self.params.simulator.structure_factors.mtz_name is not None:
            self.params.simulator.structure_factors.mtz_name = os.path.abspath(self.params.simulator.structure_factors.mtz_name)

    def __getstate__(self):
        # TODO cleanup/compress
        return {name: getattr(self, name) for name in self.saves}

    def __setstate__(self, state):
        for name in state:
            setattr(self, name, state[name])

    def set_slices(self, attr_name):
        """finds the boundaries for each attr in the 1-D array of per-shot data
        :params: attr_name, str

        For example, in a DataModeler, roi_id is set for every pixel,
        >> Modeler.roi_id returns
        0 0 0 0 1 1 2 2 2 2 3 3 3 3 4 4 4 4 4 ...

        A call Modeler.set_slices("roi_id") adds attributes
            roi_id_slices
            roi_id_unique
        where roi_id_slices is a dictionary whose keys are the unique roi ids and whose values
        are a list of slices corresponding to each connected regeion of the same roi ids.

        In the example,
        >> Modeler.roi_id_slices[0][0]
        slice(0,4,1)
        """
        vals = self.__getattribute__(attr_name)
        # find where the vals change
        splitter = np.where(np.diff(vals) != 0)[0] + 1
        npix = len(self.all_data)
        slices = [slice(V[0], V[-1] + 1, 1) for V in np.split(np.arange(npix), splitter)]
        vals_ids = [V[0] for V in np.split(np.array(vals), splitter)]
        vals_id_slices = {}
        for i_vals, slc in zip(vals_ids, slices):
            if i_vals not in vals_id_slices:
                vals_id_slices[i_vals] = [slc]
            else:
                vals_id_slices[i_vals].append(slc)
        unique_vals = set(vals)
        self.__setattr__("%s_unique" % attr_name, unique_vals)
        logging.debug("Modeler has data on %d unique vals from attribute %s" % (len(unique_vals) , attr_name))
        self.__setattr__("%s_slices" % attr_name, vals_id_slices)

    @property
    def sigma_rdout(self):
        print("WARNING ,this attribute will soon be deprecated, use nominal_sigma_rdout instead!")
        return self.nominal_sigma_rdout

    def clean_up(self, SIM):
        free_SIM_mem(SIM)

    @staticmethod
    def exper_json_single_file(exp_file, i_exp=0, check_format=True):
        """
        load a single experiment from an exp_file
        If working with large combined experiment files, we only want to load
        one image at a time on each MPI rank, otherwise at least one rank would need to
        load the entire file into memory.
        :param exp_file: experiment list file
        :param i_exp: experiment id
        :param check_format: bool, verifies the format class of the experiment, set to False if loading data from refls
        :return:
        """
        exper_json = json.load(open(exp_file))
        nexper = len(exper_json["experiment"])
        assert 0 <= i_exp < nexper

        this_exper = exper_json["experiment"][i_exp]

        new_json = {'__id__': "ExperimentList", "experiment": [deepcopy(this_exper)]}

        for model in ['beam', 'detector', 'crystal', 'imageset', 'profile', 'scan', 'goniometer', 'scaling_model']:
            if model in this_exper:
                model_index = this_exper[model]
                new_json[model] = [exper_json[model][model_index]]
                new_json["experiment"][0][model] = 0
            else:
                new_json[model] = []
        explist = ExperimentListFactory.from_dict(new_json, check_format=check_format)
        assert len(explist) == 1
        return explist[0]

    def set_experiment(self, exp, load_imageset=True, exp_idx=0):
        """
        :param exp: experiment or filename
        :param load_imageset: whether to load the imageset (usually True)
        :param exp_idx: index corresponding to experiment in experiment list
        """
        if isinstance(exp, str):
            if not load_imageset:
                self.E = ExperimentListFactory.from_json_file(exp, False)[exp_idx]
            else:
                self.E = self.exper_json_single_file(exp, exp_idx)
        else:
            self.E = exp
        if self.params.opt_det is not None:
            opt_det_E = ExperimentListFactory.from_json_file(self.params.opt_det, False)[0]
            self.E.detector = opt_det_E.detector
            MAIN_LOGGER.info("Set the optimal detector from %s" % self.params.opt_det)

        if self.params.opt_beam is not None:
            opt_beam_E = ExperimentListFactory.from_json_file(self.params.opt_beam, False)[0]
            self.E.beam = opt_beam_E.beam
            MAIN_LOGGER.info("Set the optimal beam from %s" % self.params.opt_beam)

    def load_refls(self, ref, exp_idx=0):
        """
        :param ref: reflection table or filename
        :param exp_idx: index corresponding to experiment in experiment list
        """
        if isinstance(ref, str):
            from dials.array_family import flex as dials_flex
            refls = dials_flex.reflection_table.from_file(ref)
            # TODO: is this the proper way to select the id ?
            refls = refls.select(refls['id']==exp_idx)
        else:
            # assert is a reflection table. ..
            refls = ref
        if self.params.roi.strong_only and "is_strong" in list(refls.keys()):
            refls = refls.select(refls["is_strong"])
        return refls

    def is_duplicate_hkl(self, refls):
        nref = len(refls)
        is_duplicate = np.zeros(nref, bool)
        if len(set(refls['miller_index'])) < nref:
            hkls = refls['miller_index']
            dupe_hkl = {h for h, count in Counter(hkls).items() if count > 1}
            for i_ref in range(nref):
                hh = refls[i_ref]['miller_index']
                is_duplicate[i_ref] = hh in dupe_hkl

        return is_duplicate

    def GatherFromReflectionTable(self, exp, ref, sg_symbol=None):

        self.set_experiment(exp, load_imageset=False)
        self.refls = self.load_refls(ref)
        nref = len(self.refls)
        if nref ==0:
            return False
        self.refls_idx = list(range(nref))
        self.rois = [(x1, x2, y1, y2) for x1,x2,y1,y2,_,_ in self.refls["shoebox"].bounding_boxes()]
        self.pids = list(self.refls["panel"])

        npan = len(self.E.detector)
        nfast, nslow = self.E.detector[0].get_image_size()  # NOTE assumes all panels same shape
        img_data = np.zeros((npan, nslow, nfast))
        background = np.zeros_like(img_data)
        is_trusted = np.zeros((npan, nslow, nfast), bool)
        for i_ref in range(nref):
            ref = self.refls[i_ref]
            pid = ref['panel']
            x1, x2, y1, y2 = self.rois[i_ref]

            # these are the in-bounds limits (on the panel)
            x1_onPanel = max(x1,0)
            x2_onPanel = min(x2,nfast)
            y1_onPanel = max(y1,0)
            y2_onPanel = min(y2,nslow)

            xdim = x2_onPanel-x1_onPanel
            ydim = y2_onPanel-y1_onPanel

            sb = ref['shoebox']
            sb_ystart = y1_onPanel - y1
            sb_xstart = x1_onPanel - x1
            sb_sliceY = slice(sb_ystart, sb_ystart+ydim,1)
            sb_sliceX = slice(sb_xstart, sb_xstart+xdim,1)

            dat_sliceY = slice(y1_onPanel, y1_onPanel+ydim,1)
            dat_sliceX = slice(x1_onPanel, x1_onPanel+xdim,1)
            img_data[pid,   dat_sliceY, dat_sliceX] = sb.data.as_numpy_array()[0,sb_sliceY,sb_sliceX]
            sb_bkgrnd = sb.background.as_numpy_array()[0,sb_sliceY,sb_sliceX]
            background[pid, dat_sliceY, dat_sliceX] = sb_bkgrnd
            fg_code = 5 # dials.algorithms.MaskCode.Valid + MaskCode.Foreground ) # 5
            bg_code = 19 # MaskCode.Valid + MaskCode.Background + MaskCode.BackgroundUsed  # 19
            mask = sb.mask.as_numpy_array()[0,sb_sliceY,sb_sliceX]
            if self.params.refiner.refldata_trusted=="allValid":
                sb_trust = mask > 0
            elif self.params.refiner.refldata_trusted=="fg":
                sb_trust = mask==fg_code
            else:
                sb_trust = np.logical_or(mask==fg_code, mask==bg_code)

            # below_zero = sb_bkgrnd <= 0
            below_zero = sb_bkgrnd < 0
            if np.any(below_zero):
                nbelow = np.sum(below_zero)
                ntot = sb_bkgrnd.size
                MAIN_LOGGER.debug("background <= zero in %d/%d pixels from shoebox %d! Marking those pixels as untrusted!" %  ( nbelow, ntot, i_ref ))
                sb_trust[below_zero] = False

            is_trusted[pid, dat_sliceY,dat_sliceX] = sb_trust

            self.rois[i_ref] = x1_onPanel, x2_onPanel, y1_onPanel, y2_onPanel


        if self.params.refiner.refldata_to_photons:
            MAIN_LOGGER.debug("Re-scaling reflection data to photon units: conversion factor=%f" % self.params.refiner.adu_per_photon)
            img_data /= self.params.refiner.adu_per_photon
            background /= self.params.refiner.adu_per_photon

        # can be used for Bfactor modeling
        self.Q = np.linalg.norm(self.refls["rlp"], axis=1)
        self.nominal_sigma_rdout = self.params.refiner.sigma_r / self.params.refiner.adu_per_photon

        self.Hi = list(self.refls["miller_index"])
        if sg_symbol is not None:
            self.Hi_asu = utils.map_hkl_list(self.Hi, True, sg_symbol)
        else:
            self.Hi_asu = self.Hi

        self.data_to_one_dim(img_data, is_trusted, background)
        return True

    def GatherFromExperiment(self, exp, ref, remove_duplicate_hkl=True, sg_symbol=None, exp_idx=0):
        """

        :param exp: experiment list filename , or experiment object
        :param ref: reflection table filename, or reflection table instance
        :param remove_duplicate_hkl: search for miller index duplicates and remove
        :param sg_symbol: space group lookup symbol P43212
        :param exp_idx: index of the experiment in the experiment list
        :return:
        """
        self.set_experiment(exp, load_imageset=True, exp_idx=exp_idx)

        refls = self.load_refls(ref, exp_idx=exp_idx)
        if len(refls)==0:
            MAIN_LOGGER.warning("no refls loaded!")
            return False

        if "rlp" not in list(refls.keys()):
            try:
                utils.add_rlp_column(refls, self.E)
                assert "rlp" in list(refls[0].keys())
            except KeyError:
                self.no_rlp_info = True
        img_data = utils.image_data_from_expt(self.E)
        img_data /= self.params.refiner.adu_per_photon
        is_trusted = np.ones(img_data.shape, bool)
        hotpix_mask = None
        if self.params.roi.hotpixel_mask is not None:
            is_trusted = utils.load_mask(self.params.roi.hotpixel_mask)
            hotpix_mask = ~is_trusted
        self.nominal_sigma_rdout = self.params.refiner.sigma_r / self.params.refiner.adu_per_photon

        roi_packet = utils.get_roi_background_and_selection_flags(
            refls, img_data, shoebox_sz=self.params.roi.shoebox_size,
            reject_edge_reflections=self.params.roi.reject_edge_reflections,
            reject_roi_with_hotpix=self.params.roi.reject_roi_with_hotpix,
            background_mask=None, hotpix_mask=hotpix_mask,
            bg_thresh=self.params.roi.background_threshold,
            use_robust_estimation=not self.params.roi.fit_tilt,
            set_negative_bg_to_zero=self.params.roi.force_negative_background_to_zero,
            pad_for_background_estimation=self.params.roi.pad_shoebox_for_background_estimation,
            sigma_rdout=self.nominal_sigma_rdout, deltaQ=self.params.roi.deltaQ, experiment=self.E,
            weighted_fit=self.params.roi.fit_tilt_using_weights,
            allow_overlaps=self.params.roi.allow_overlapping_spots,
            ret_cov=True, skip_roi_with_negative_bg=self.params.roi.skip_roi_with_negative_bg,
            only_high=self.params.roi.only_filter_zingers_above_mean, centroid=self.params.roi.centroid)

        if roi_packet is None:
            return False

        self.rois, self.pids, self.tilt_abc, self.selection_flags, background, self.tilt_cov = roi_packet

        if remove_duplicate_hkl and not self.no_rlp_info:
            is_not_a_duplicate = ~self.is_duplicate_hkl(refls)
            self.selection_flags = np.logical_and( self.selection_flags, is_not_a_duplicate)
        else:
            self.selection_flags  = np.array(self.selection_flags)

        if self.params.refiner.res_ranges is not None:
            # TODO add res ranges support for GatherFromReflectionTable
            if self.no_rlp_info:
                raise NotImplementedError("Cannot set resolution limits when processing refls that are missing the RLP column")
            res_flags = np.zeros(len(refls)).astype(bool)
            res = 1. / np.linalg.norm(refls["rlp"], axis=1)
            for dmin,dmax in utils.parse_reso_string(self.params.refiner.res_ranges):
                MAIN_LOGGER.debug("Parsing res range %.3f - %.3f Angstrom" % (dmin, dmax))
                in_resShell = np.logical_and(res >= dmin, res < dmax)
                res_flags[in_resShell] = True

            MAIN_LOGGER.info("Resolution filter removed %d/%d refls outside of all resolution ranges " \
                              % (sum(~res_flags), len(refls)))
            self.selection_flags[~res_flags] = False

        if "miller_index" in list(refls.keys()):
            self.Hi = list(refls["miller_index"])
            if sg_symbol is not None:
                self.Hi_asu = utils.map_hkl_list(self.Hi, True, sg_symbol)
            else:
                self.Hi_asu = self.Hi

        # Random ROI subsampling (e.g. for hopper_heatup speed)
        roi_frac = getattr(self.params.roi, 'fraction', None)
        if roi_frac is not None and 0 < roi_frac < 1:
            selected_idx = np.where(self.selection_flags)[0]
            n_keep = max(1, int(round(len(selected_idx) * roi_frac)))
            if n_keep < len(selected_idx):
                rng = np.random.RandomState(42)  # reproducible across trials
                drop_idx = rng.choice(selected_idx, size=len(selected_idx) - n_keep, replace=False)
                self.selection_flags[drop_idx] = False
                MAIN_LOGGER.info("ROI subsampling: kept %d/%d ROIs (fraction=%.2f)"
                                 % (n_keep, len(selected_idx), roi_frac))

        if sum(self.selection_flags) == 0:
            MAIN_LOGGER.info("No pixels slected, continuing")
            return False
        self.refls = refls
        self.refls_idx = [i_roi for i_roi in range(len(refls)) if self.selection_flags[i_roi]]

        self.rois = [roi for i_roi, roi in enumerate(self.rois) if self.selection_flags[i_roi]]
        self.tilt_abc = [abc for i_roi, abc in enumerate(self.tilt_abc) if self.selection_flags[i_roi]]
        self.pids = [pid for i_roi, pid in enumerate(self.pids) if self.selection_flags[i_roi]]
        self.tilt_cov = [cov for i_roi, cov in enumerate(self.tilt_cov) if self.selection_flags[i_roi]]
        if self.Hi is not None:
            self.Hi =[hi for i_roi, hi in enumerate(self.Hi) if self.selection_flags[i_roi]]
            self.Hi_asu =[hi_asu for i_roi, hi_asu in enumerate(self.Hi_asu) if self.selection_flags[i_roi]]

        if not self.no_rlp_info:
            self.Q = [np.linalg.norm(rlp_vec) for i_roi, rlp_vec in enumerate(refls["rlp"]) if self.selection_flags[i_roi]]

        self.data_to_one_dim(img_data, is_trusted, background)
        return True

    def data_to_one_dim(self, img_data, is_trusted, background):
        all_data = []
        all_sigma_rdout = []
        all_pid = []
        all_fast = []
        all_slow = []
        all_fast_relative = []
        all_slow_relative = []
        all_trusted = []
        all_sigmas = []
        all_background = []
        roi_id = []
        all_q_perpix = []
        all_refls_idx = []
        pixel_counter = np.zeros_like(img_data)
        self.all_nominal_hkl = []
        self.hi_asu_perpix = []
        numOutOfRange = 0
        perpixel_dark_rms = None
        if self.params.use_perpixel_dark_rms:
            perpixel_dark_rms = get_pedestalRMS_from_jungfrau(self.E)
            perpixel_dark_rms /= self.params.refiner.adu_per_photon
        if self.params.try_strong_mask_only:
            strong_mask_img = utils.strong_spot_mask(self.refls, self.E.detector)

        roi_datamax = []
        for i_roi in range(len(self.rois)):
            pid = self.pids[i_roi]
            x1, x2, y1, y2 = self.rois[i_roi]
            Y, X = np.indices((y2 - y1, x2 - x1))
            data = img_data[pid, y1:y2, x1:x2].copy()
            pixel_counter[pid, y1:y2, x1:x2] += 1

            data = data.ravel()
            all_background += list(background[pid, y1:y2, x1:x2].ravel())
            trusted = is_trusted[pid, y1:y2, x1:x2].ravel()
            datamax = data[trusted].max()
            roi_datamax.append(datamax)
            if perpixel_dark_rms is not None:
                sigma_rdout = perpixel_dark_rms[pid, y1:y2, x1:x2].ravel()
            else:
                sigma_rdout = np.ones_like(data)*self.nominal_sigma_rdout

            if self.params.roi.mask_outside_trusted_range:
                if self.params.roi.trusted_range is not None:
                    minDat, maxDat = self.params.roi.trusted_range
                    assert minDat < maxDat
                else:
                    minDat, maxDat = self.E.detector[pid].get_trusted_range()
                data_out_of_range = np.logical_or(data < minDat, data > maxDat)
                if self.params.roi.mask_all_if_any_outside_trusted_range:
                    if np.any(data_out_of_range):
                        data_out_of_range[:] = True
                

                trusted[data_out_of_range] = False
                numOutOfRange +=np.sum(data_out_of_range)

            if self.params.mask_highest_values is not None:
                trusted[np.argsort(data)[-self.params.mask_highest_values:]] = False

            if self.params.try_strong_mask_only:
                is_strong_spot = strong_mask_img[pid, y1:y2, x1:x2].ravel()
                if self.params.dilate_strong_mask is not None:
                    assert self.params.dilate_strong_mask >= 1
                    is_strong_spot = binary_dilation(is_strong_spot, iterations=self.params.dilate_strong_mask)
                trusted = np.logical_and(trusted, is_strong_spot)

            all_trusted += list(trusted)
            #TODO ignore invalid value warning (handled below), or else mitigate it!

            with np.errstate(invalid='ignore'):
                all_sigmas += list(np.sqrt(data + sigma_rdout ** 2))

            all_sigma_rdout += list(sigma_rdout)
            all_fast += list(X.ravel() + x1)
            all_fast_relative += list(X.ravel())
            all_slow += list(Y.ravel() + y1)
            all_slow_relative += list(Y.ravel())
            all_data += list(data)
            npix = len(data)  # np.sum(trusted)
            all_pid += [pid] * npix
            roi_id += [i_roi] * npix
            all_refls_idx += [self.refls_idx[i_roi]] * npix
            if not self.no_rlp_info:
                all_q_perpix += [self.Q[i_roi]]*npix
            if self.Hi is not None:
                self.all_nominal_hkl += [tuple(self.Hi[i_roi])]*npix
                self.hi_asu_perpix += [self.Hi_asu[i_roi]] * npix
                #self.all_nominal_hkl += [tuple(self.Hi[i_roi])]*npix
                #self.hi_asu_perpix += [self.Hi_asu[i_roi]] * npix

        if self.params.roi.mask_outside_trusted_range:
            MAIN_LOGGER.debug("Found %d pixels outside of trusted range" % numOutOfRange)
        all_freq = []
        for i_roi in range(len(self.rois)):
            pid = self.pids[i_roi]
            x1, x2, y1, y2 = self.rois[i_roi]
            freq = pixel_counter[pid, y1:y2, x1:x2].ravel()
            all_freq += list(freq)
        self.all_freq = np.array(all_freq, np.int32)  # if no overlapping pixels, this should be an array of 1's
        if not self.params.roi.allow_overlapping_spots:
            if not np.all(self.all_freq==1):
                print(set(self.all_freq))
                raise ValueError("There are overlapping regions of interest, despite the command to not allow overlaps")

        self.all_q_perpix = np.array(all_q_perpix)
        pan_fast_slow = np.ascontiguousarray((np.vstack([all_pid, all_fast, all_slow]).T).ravel())
        self.pan_fast_slow = flex.size_t(pan_fast_slow)
        self.all_background = np.array(all_background)
        self.roi_id = np.array(roi_id)
        self.all_data = np.array(all_data)
        if np.allclose(all_sigma_rdout, self.nominal_sigma_rdout):
            self.all_sigma_rdout = self.nominal_sigma_rdout
        else:
            self.all_sigma_rdout = np.array(all_sigma_rdout)
        self.all_sigmas = np.array(all_sigmas)
        # note rare chance for sigmas to be nan if the args of sqrt is below 0
        self.all_trusted = np.logical_and(np.array(all_trusted), ~np.isnan(all_sigmas))

        if self.params.roi.filter_bright_peaks.enable:
            roi_datamax = np.array(roi_datamax)
            is_bright_peak = utils.is_outlier(roi_datamax, thresh=self.params.roi.filter_bright_peaks.thresh)
            for i_roi in range(len(self.rois)):
                if is_bright_peak[i_roi]:
                    self.all_trusted[self.roi_id==i_roi] = False

        if self.params.roi.skip_roi_with_negative_bg:
            # Dont include pixels whose background model is below 0
            self.all_trusted[self.all_background < 0] = False

        self.npix_total = len(all_data)
        self.all_fast = np.array(all_fast)
        self.all_slow = np.array(all_slow)
        self.all_pid = np.array(all_pid)
        #self.simple_weights = 1/self.all_sigmas**2
        self.u_id = set(self.roi_id)
        self.all_refls_idx = np.array(all_refls_idx)

        MAIN_LOGGER.debug("Modeler has %d/ %d trusted pixels" % (self.all_trusted.sum() , self.npix_total))

    def dump_gathered_to_refl(self, output_name, do_xyobs_sanity_check=False):
        """after running GatherFromExperiment, dump the gathered results
        (data, background etc) to a new reflection file which can then be used to run
        diffBragg without the raw data in the experiment (this exists mainly for portability, and
        unit tests)"""
        from dials.model.data import Shoebox
        from dials.array_family import flex as dials_flex
        shoeboxes = []
        R = dials_flex.reflection_table()
        for i_roi, i_ref in enumerate(self.refls_idx):
            roi_sel = self.roi_id==i_roi
            x1, x2, y1, y2 = self.rois[i_roi]
            roi_shape = y2-y1, x2-x1
            roi_img = self.all_data[roi_sel].reshape(roi_shape).astype(np.float32)  #NOTE this has already been converted to photon units
            roi_bg = self.all_background[roi_sel].reshape(roi_shape).astype(np.float32)

            sb = Shoebox((x1, x2, y1, y2, 0, 1))
            sb.allocate()
            sb.data = dials_flex.float(np.ascontiguousarray(roi_img[None]))
            sb.background = dials_flex.float(np.ascontiguousarray(roi_bg[None]))

            dials_mask = np.zeros(roi_img.shape).astype(np.int32)
            mask = self.all_trusted[roi_sel].reshape(roi_shape)
            dials_mask[mask] = dials_mask[mask] + 1  # MaskCode.Valid=1
            sb.mask = dials_flex.int(np.ascontiguousarray(dials_mask[None]))

            # quick sanity test
            if do_xyobs_sanity_check:
                ref = self.refls[i_ref]
                x,y,_ = ref['xyzobs.px.value']
                assert x1 <= x <= x2, "exp %s; refl %d, %f %f %f" % (output_name, i_ref, x1,x,x2)
                assert y1 <= y <= y2, "exp %s; refl %d, %f %f %f" % (output_name, i_ref, y1,y,y2)

            R.extend(self.refls[i_ref: i_ref+1])
            shoeboxes.append(sb)

        R['shoebox'] = dials_flex.shoebox(shoeboxes)
        R['id'] = dials_flex.int(len(R), 0)
        R.as_file(output_name)

    def set_parameters_for_experiment(self, best=None):
        if self.params.symmetrize_Flatt and not self.params.fix.eta_abc:
            if not self.params.simulator.crystal.has_isotropic_mosaicity:
                raise NotImplementedError("if fix.eta_abc=False and symmetrize_Flatt=True, then eta must be isotropic. Set simulator.crystal.has_isotropic_mosaicity=True")
        ParameterTypes = {"ranged": RangedParameter, "positive": PositiveParameter}
        ParameterType = RangedParameter  # most params currently only this type

        if self.params.centers.Nvol is not None:
            assert self.params.betas.Nvol is not None

        if best is not None:
            # set the crystal Umat (rotational displacement) and Bmat (unit cell)
            # Umatrix
            # NOTE: just set the best Amatrix here
            #C = deepcopy(self.E.crystal)
            #crystal = self.E.crystal
            #self.E.crystal = crystal

            ## TODO , currently need this anyway
            ucparam = best[["a","b","c","al","be","ga"]].values[0]
            ucman = utils.manager_from_params(ucparam)
            self.E.crystal.set_B(ucman.B_recipspace)
            self.E.crystal.set_A(best.Amats.values[0])

            # mosaic block
            self.params.init.Nabc = tuple(best.ncells.values[0])
            self.params.init.Ndef = tuple(best.ncells_def.values[0])
            # scale factor
            self.params.init.G = best.spot_scales.values[0]

            if "detz_shift_mm" in list(best):
                self.params.init.detz_shift = best.detz_shift_mm.values[0]

            if "gonio_angle" in list(best):
                self.params.init.gonio_angle = best.gonio_angle.values[0]

            # TODO: set best eta_abc params
            self.params.init.eta_abc = tuple(best.eta_abc.values[0])
            self.params.init.eta_abc = tuple([eta_val if eta_val > self.params.mins.eta_abc[i_eta] else self.params.mins.eta_abc[i_eta]+1e-2 for i_eta, eta_val in enumerate(self.params.init.eta_abc)])

            lam0, lam1 = get_lam0_lam1_from_pandas(best)
            self.params.init.spec = lam0, lam1

        init = self.params.init
        sigma = self.params.sigmas
        mins = self.params.mins
        maxs = self.params.maxs
        centers = self.params.centers
        betas = self.params.betas
        fix = self.params.fix
        types = self.params.types
        P = Parameters()
        if self.params.init.random_Gs is not None:
            init.G = np.random.choice(self.params.init.random_Gs)
        for i_xtal in range(self.num_xtals):
            for ii in range(3):

                p = ParameterType(init=0, sigma=sigma.RotXYZ[ii],
                                  minval=mins.RotXYZ[ii], maxval=maxs.RotXYZ[ii],
                                  fix=fix.RotXYZ[ii], name="RotXYZ%d_xtal%d" % (ii,i_xtal),
                                  center=0 if betas.RotXYZ is not None else None,
                                  beta=betas.RotXYZ)
                P.add(p)

            p = ParameterTypes[types.G](init=init.G + init.G*0.01*i_xtal, sigma=sigma.G,
                              minval=mins.G, maxval=maxs.G,
                              fix=fix.G, name="G_xtal%d" %i_xtal,
                              center=centers.G, beta=betas.G)
            P.add(p)


        # these parameters are equal for all texture-domains within a crystal
        fix_Nabc = [fix.Nabc]*3
        if self.params.simulator.crystal.has_isotropic_ncells:
            fix_Nabc = [fix_Nabc[0], True, True]

        fix_difsig = [fix.diffuse_sigma]*3
        if self.params.isotropic.diffuse_sigma:
            fix_difsig = [fix_difsig[0], True, True]

        fix_difgam = [fix.diffuse_gamma]*3
        if self.params.isotropic.diffuse_gamma:
            fix_difgam = [fix_difgam[0], True, True]

        if not fix.eta_abc:
            assert all([eta> 0 for eta in init.eta_abc])
        if tuple(init.eta_abc ) == (0,0,0):
            mins.eta_abc=[-1e-10,-1e-10,-1e-10]

        fix_eta = [fix.eta_abc]*3
        if self.params.simulator.crystal.has_isotropic_mosaicity:
            fix_eta = [fix_eta[0], True, True]

        if self.params.init.random_Nabcs is not None:
            init.Nabc = np.random.choice(self.params.init.random_Nabcs, replace=True, size=3)

        self.use_cholesky_Nabc = getattr(self.params, 'use_cholesky_Nabc', False)

        if self.use_cholesky_Nabc:
            # Cholesky mode: parameterize NABC = L^T * L with 6 lower-triangular elements
            # L = [[L11,  0,   0 ],
            #      [L21, L22,  0 ],
            #      [L31, L32, L33]]
            # Initialize from diagonal Nabc: L11=sqrt(Na), L22=sqrt(Nb), L33=sqrt(Nc), off-diag=0
            if init.cholesky is not None:
                chol_init = list(init.cholesky)
            else:
                chol_init = [np.sqrt(init.Nabc[0]), 0, np.sqrt(init.Nabc[1]),
                             0, 0, np.sqrt(init.Nabc[2])]

            # Auto-compute cholesky bounds from Nabc bounds
            # NABC = L^T * L, so: Na = L11^2, Nb = L21^2 + L22^2, Nc = L31^2 + L32^2 + L33^2
            # Diagonal: L_ii in [sqrt(N_min_i), sqrt(N_max_i)]
            # Off-diagonal: [-sqrt(N_max_j), sqrt(N_max_j)] where j is the row
            chol_mins = list(mins.cholesky)
            chol_maxs = list(maxs.cholesky)
            if getattr(self.params, 'cholesky_bounds_from_Nabc', True):
                Na_min, Nb_min, Nc_min = mins.Nabc
                Na_max, Nb_max, Nc_max = maxs.Nabc
                # L11: sqrt(Na) range
                chol_mins[0] = np.sqrt(max(0, Na_min))   # L11 min
                chol_maxs[0] = np.sqrt(Na_max)            # L11 max
                # L21: off-diag for row 2, bounded by sqrt(Nb_max)
                chol_mins[1] = -np.sqrt(Nb_max)           # L21 min
                chol_maxs[1] = np.sqrt(Nb_max)            # L21 max
                # L22: sqrt(Nb) range (but L21 also contributes, so allow full range)
                chol_mins[2] = np.sqrt(max(0, Nb_min)) * 0.1  # allow shrinkage since L21 contributes
                chol_maxs[2] = np.sqrt(Nb_max)            # L22 max
                # L31: off-diag for row 3, bounded by sqrt(Nc_max)
                chol_mins[3] = -np.sqrt(Nc_max)           # L31 min
                chol_maxs[3] = np.sqrt(Nc_max)            # L31 max
                # L32: off-diag for row 3, bounded by sqrt(Nc_max)
                chol_mins[4] = -np.sqrt(Nc_max)           # L32 min
                chol_maxs[4] = np.sqrt(Nc_max)            # L32 max
                # L33: sqrt(Nc) range
                chol_mins[5] = np.sqrt(max(0, Nc_min)) * 0.1  # allow shrinkage since L31,L32 contribute
                chol_maxs[5] = np.sqrt(Nc_max)            # L33 max
                if self.params.logging.rank0_level == "high":
                    print("RANK%04d | Cholesky bounds from Nabc: Na=[%.1f,%.1f] Nb=[%.1f,%.1f] Nc=[%.1f,%.1f]" % (
                        self.rank, Na_min, Na_max, Nb_min, Nb_max, Nc_min, Nc_max))
                    print("RANK%04d |   L11=[%.2f,%.2f] L21=[%.2f,%.2f] L22=[%.2f,%.2f] L31=[%.2f,%.2f] L32=[%.2f,%.2f] L33=[%.2f,%.2f]" % (
                        self.rank, chol_mins[0], chol_maxs[0], chol_mins[1], chol_maxs[1],
                        chol_mins[2], chol_maxs[2], chol_mins[3], chol_maxs[3],
                        chol_mins[4], chol_maxs[4], chol_mins[5], chol_maxs[5]))

            chol_names = ["chol_L11", "chol_L21", "chol_L22", "chol_L31", "chol_L32", "chol_L33"]
            for ii in range(6):
                p = ParameterType(init=chol_init[ii], sigma=sigma.cholesky[ii],
                                  minval=chol_mins[ii], maxval=chol_maxs[ii],
                                  fix=fix.Nabc,  # reuse fix.Nabc to control Cholesky refinement
                                  name=chol_names[ii],
                                  center=centers.cholesky[ii] if centers.cholesky is not None else None,
                                  beta=betas.cholesky[ii] if betas.cholesky is not None else None)
                P.add(p)
            # Still need Nabc and Ndef params as fixed placeholders (for code that reads them)
            for ii in range(3):
                p = ParameterTypes[types.Nabc](init=init.Nabc[ii], sigma=sigma.Nabc[ii],
                                  minval=mins.Nabc[ii], maxval=maxs.Nabc[ii],
                                  fix=True, name="Nabc%d" % (ii,))
                P.add(p)
                p = ParameterType(init=init.Ndef[ii], sigma=sigma.Ndef[ii],
                                  minval=mins.Ndef[ii], maxval=maxs.Ndef[ii],
                                  fix=True, name="Ndef%d" % (ii,))
                P.add(p)
        else:
            for ii in range(3):
                # Mosaic domain tensor
                p = ParameterTypes[types.Nabc](init=init.Nabc[ii], sigma=sigma.Nabc[ii],
                                  minval=mins.Nabc[ii], maxval=maxs.Nabc[ii],
                                  fix=fix_Nabc[ii], name="Nabc%d" % (ii,),
                                  center=centers.Nabc[ii] if centers.Nabc is not None else None,
                                  beta=betas.Nabc[ii] if betas.Nabc is not None else None)
                P.add(p)

                p = ParameterType(init=init.Ndef[ii], sigma=sigma.Ndef[ii],
                                  minval=mins.Ndef[ii], maxval=maxs.Ndef[ii],
                                  fix=fix.Ndef, name="Ndef%d" % (ii,),
                                  center=centers.Ndef[ii] if centers.Ndef is not None else None,
                                  beta=betas.Ndef[ii] if betas.Ndef is not None else None)
                P.add(p)

        # diffuse gamma, sigma, and mosaic spread (shared by both Cholesky and legacy modes)
        for ii in range(3):
            p = ParameterTypes[types.diffuse_gamma](init=init.diffuse_gamma[ii], sigma=sigma.diffuse_gamma[ii],
                              minval=mins.diffuse_gamma[ii], maxval=maxs.diffuse_gamma[ii],
                              fix=fix_difgam[ii], name="diffuse_gamma%d" % (ii,),
                              center=centers.diffuse_gamma[ii] if centers.diffuse_gamma is not None else None,
                              beta=betas.diffuse_gamma[ii] if betas.diffuse_gamma is not None else None)
            P.add(p)

            p = ParameterTypes[types.diffuse_sigma](init=init.diffuse_sigma[ii], sigma=sigma.diffuse_sigma[ii],
                              minval=mins.diffuse_sigma[ii], maxval=maxs.diffuse_sigma[ii],
                              fix=fix_difsig[ii], name="diffuse_sigma%d" % (ii,),
                              center=centers.diffuse_sigma[ii] if centers.diffuse_sigma is not None else None,
                              beta=betas.diffuse_sigma[ii] if betas.diffuse_sigma is not None else None)
            P.add(p)

            # mosaic spread (mosaicity)
            p = ParameterType(init=init.eta_abc[ii], sigma=sigma.eta_abc[ii],
                              minval=mins.eta_abc[ii], maxval=maxs.eta_abc[ii],
                              fix=fix_eta[ii], name="eta_abc%d" % (ii,),
                              center=centers.eta_abc[ii] if centers.eta_abc is not None else None,
                              beta=betas.eta_abc[ii] if betas.eta_abc is not None else None)
            P.add(p)

        ucell_man = utils.manager_from_crystal(self.E.crystal)
        ucell_vary_perc = self.params.ucell_edge_perc / 100.
        for i_uc, (name, val) in enumerate(zip(ucell_man.variable_names, ucell_man.variables)):
            if "Ang" in name:
                minval = val - ucell_vary_perc * val
                maxval = val + ucell_vary_perc * val
                if name == 'a_Ang':
                    cent = centers.ucell_a
                    beta = betas.ucell_a
                elif name== 'b_Ang':
                    cent = centers.ucell_b
                    beta = betas.ucell_b
                else:
                    cent = centers.ucell_c
                    beta = betas.ucell_c
            else:
                val_in_deg = val * 180 / np.pi
                minval = (val_in_deg - self.params.ucell_ang_abs) * np.pi / 180.
                maxval = (val_in_deg + self.params.ucell_ang_abs) * np.pi / 180.
                if name=='alpha_rad':
                    cent = centers.ucell_alpha
                    beta = betas.ucell_alpha
                elif name=='beta_rad':
                    cent = centers.ucell_beta
                    beta = betas.ucell_beta
                else:
                    cent = centers.ucell_gamma
                    beta = betas.ucell_gamma
                if cent is not None:
                    cent = cent*np.pi / 180.

            p = ParameterType(init=val, sigma=sigma.ucell[i_uc],
                              minval=minval, maxval=maxval, fix=fix.ucell,
                              name="ucell%d" % (i_uc,),
                              center=cent,
                              beta=beta)
            MAIN_LOGGER.info(
                "Unit cell variable %s (currently=%f) is bounded by %f and %f" % (name, val, minval, maxval))
            P.add(p)

        self.ucell_man = ucell_man

        p = ParameterType(init=init.detz_shift*1e-3, sigma=sigma.detz_shift,
                          minval=mins.detz_shift*1e-3, maxval=maxs.detz_shift*1e-3,
                          fix=fix.detz_shift,name="detz_shift",
                          center=centers.detz_shift,
                          beta=betas.detz_shift)
        P.add(p)

        p = ParameterType(init=init.gonio_angle, sigma=sigma.gonio_angle,
                          minval=mins.gonio_angle, maxval=maxs.gonio_angle,
                          fix=fix.gonio_angle,name="gonio_angle",
                          center=centers.gonio_angle,
                          beta=betas.gonio_angle)
        P.add(p)

        # per-image B-factor
        p = ParameterType(init=init.B, sigma=sigma.B,
                          minval=mins.B, maxval=maxs.B,
                          fix=fix.B, name="Bfactor",
                          center=centers.B,
                          beta=betas.B)
        P.add(p)

        # anisotropic B-factor (6 components: β11, β22, β33, β12, β13, β23)
        for ii in range(6):
            p = ParameterType(init=init.Baniso[ii], sigma=sigma.Baniso[ii],
                              minval=mins.Baniso[ii], maxval=maxs.Baniso[ii],
                              fix=fix.Baniso, name="Baniso%d" % ii,
                              center=centers.Baniso[ii] if centers.Baniso is not None else None,
                              beta=betas.Baniso[ii] if betas.Baniso is not None else None)
            P.add(p)

        if not self.params.fix.perRoiScale or self.params.use_perRoiScale:
            self.set_slices("roi_id")  # this creates roi_id_unique
            refls_have_scales = "scale_factor" in list(self.refls.keys())
            for roi_id in self.roi_id_unique:
                slc = self.roi_id_slices[roi_id][0]
                if refls_have_scales:
                    refl_idx = int(self.all_refls_idx[slc][0])
                    init_scale = self.refls[refl_idx]["scale_factor"]
                else:
                    init_scale = 1
                p = PositiveParameter(init=init_scale, sigma=self.params.sigmas.roiPerScale,
                                  maxval=1e12,
                                  fix=fix.perRoiScale, name="scale_roi%d" % roi_id,
                                  center=1,
                                  beta=1e12)
                #p.misc_data = {"refl_idx": refl_idx}
                if isinstance(self.all_q_perpix, np.ndarray) and self.all_q_perpix.size:
                    q = self.all_q_perpix[slc][0]
                    reso = 1./q
                    hkl = self.all_nominal_hkl[slc][0]
                    p.misc_data={}
                    p.misc_data["reso"]= reso
                    p.misc_data["hkl"]=hkl
                P.add(p)

        # two parameters for optimizing the spectrum
        p = RangedParameter(init=self.params.init.spec[0], sigma=self.params.sigmas.spec[0],
                            minval=mins.spec[0], maxval=maxs.spec[0], fix=fix.spec,
                            name="lambda_offset", center=centers.spec[0] if centers.spec is not None else None,
                            beta=betas.spec[0] if betas.spec is not None else None)
        P.add(p)
        p = RangedParameter(init=self.params.init.spec[1], sigma=self.params.sigmas.spec[1],
                            minval=mins.spec[1], maxval=maxs.spec[1], fix=fix.spec,
                            name="lambda_scale", center=centers.spec[1] if centers.spec is not None else None,
                            beta=betas.spec[1] if betas.spec is not None else None)
        P.add(p)

        GEO = self.params.geometry
        DEG_TO_PI = np.pi/180
        vary_rots = [not fixed_flag for fixed_flag in GEO.fix.panel_rotations]
        o = RangedParameter(name="group0_RotOrth",
                            init=0,
                            sigma=1,  # TODO
                            minval=GEO.min.panel_rotations[0] * DEG_TO_PI,
                            maxval=GEO.max.panel_rotations[0] * DEG_TO_PI,
                            fix=not vary_rots[0], center=0, beta=GEO.betas.panel_rot[0], is_global=True)

        f = RangedParameter(name="group0_RotFast",
                            init=0,
                            sigma=1,  # TODO
                            minval=GEO.min.panel_rotations[1] * DEG_TO_PI,
                            maxval=GEO.max.panel_rotations[1] * DEG_TO_PI,
                            fix=not vary_rots[1], center=0, beta=GEO.betas.panel_rot[1],
                            is_global=True)

        s = RangedParameter(name="group0_RotSlow",
                            init=0,
                            sigma=1,  # TODO
                            minval=GEO.min.panel_rotations[2] * DEG_TO_PI,
                            maxval=GEO.max.panel_rotations[2] * DEG_TO_PI,
                            fix=not vary_rots[2], center=0, beta=GEO.betas.panel_rot[2],
                            is_global=True)

        vary_shifts = [not fixed_flag for fixed_flag in GEO.fix.panel_translations]
        # vary_shifts = [True]*3
        x = RangedParameter(name="group0_ShiftX", init=0,
                            sigma=1,
                            minval=GEO.min.panel_translations[0] * 1e-3, maxval=GEO.max.panel_translations[0] * 1e-3,
                            fix=not vary_shifts[0], center=0, beta=GEO.betas.panel_xyz[0],
                            is_global=True)
        y = RangedParameter(name="group0_ShiftY", init=0,
                            sigma=1,
                            minval=GEO.min.panel_translations[1] * 1e-3, maxval=GEO.max.panel_translations[1] * 1e-3,
                            fix=not vary_shifts[1], center=0, beta=GEO.betas.panel_xyz[1],
                            is_global=True)
        z = RangedParameter(name="group0_ShiftZ", init=0,
                            sigma=1,
                            minval=GEO.min.panel_translations[2] * 1e-3, maxval=GEO.max.panel_translations[2] * 1e-3,
                            fix=not vary_shifts[2], center=0, beta=GEO.betas.panel_xyz[2],
                            is_global=True)
        for p in [o,f,s,x,y,z]:
            P.add(p)
        P.refining_detector = any(vary_rots + vary_shifts)

        # iterating over this dict is time-consuming when refinine Fhkl, so we split up the names here:
        self.non_fhkl_params = [name for name in P if not name.startswith("scale_roi") and not name.startswith("Fhkl_")]
        self.scale_roi_names = [name for name in P if name.startswith("scale_roi")]
        self.P = P

        for name in self.P:
            p = self.P[name]
            if (p.beta is not None and p.center is None) or (p.center is not None and p.beta is None):
                raise RuntimeError("To use restraints, must specify both center and beta for param %s" % name)

    def get_data_model_pairs(self, reorder=False, return_stats=False):
        """
        :param reorder: whether to reorder sort by resolution
        :return: 4 lists of sub-images: data, models, trusted, bragg
        """
        if self.best_model is None:
            raise ValueError("cannot get the best model, there is no best_model attribute")
        all_dat_img, all_mod_img = [], []
        all_trusted = []
        all_bragg = []
        all_sigma_rdout = []
        all_d_perpix = 1/self.all_q_perpix
        spot_d = []
        spot_hkl = []
        spot_pid_and_cent = []
        spot_scale = []

        for i_roi in range(len(self.rois)):
            x1, x2, y1, y2 = self.rois[i_roi]
            roi_sel = self.roi_id==i_roi
            mod = self.best_model[roi_sel].reshape((y2 - y1, x2 - x1))
            try:
                if all_d_perpix.size:
                    res = all_d_perpix[roi_sel] .mean()
                else:
                    res = None
                cent_x = (x1+x2) /2.
                cent_y = (y1+y2) / 2.
                pid = self.all_pid[roi_sel].ravel()[0]
                scale = 1
                try:
                    scale = self.roiScalesPerPix[roi_sel].ravel()[0]
                except:
                    pass
                spot_scale.append(scale)
                spot_pid_and_cent.append((pid, cent_x, cent_y))
                hkl = None
                if self.Hi:
                    hkl = self.Hi[i_roi]
                spot_hkl.append(hkl)
                spot_d.append(res)
            except IndexError:
                pass
            if self.all_trusted is not None:
                trusted = self.all_trusted[self.roi_id == i_roi].reshape((y2 - y1, x2 - x1))
                all_trusted.append(trusted)
            else:
                all_trusted.append(None)

            dat = self.all_data[self.roi_id == i_roi].reshape((y2 - y1, x2 - x1))
            all_dat_img.append(dat)
            if isinstance(self.all_sigma_rdout, np.ndarray):
                sig = self.all_sigma_rdout[self.roi_id==i_roi].reshape((y2-y1, x2-x1))
                all_sigma_rdout.append(sig)
            if self.all_background is not None:
                bg = self.all_background[self.roi_id == i_roi].reshape((y2-y1, x2-x1))
                if self.best_model_includes_background:
                    all_bragg.append(mod-bg)
                    all_mod_img.append(mod)
                else:
                    all_bragg.append(mod)
                    all_mod_img.append(mod+bg)
            else:  # assume mod contains background
                all_mod_img.append(mod)
                all_bragg.append(None)
        ret_subimgs = [all_dat_img, all_mod_img, all_trusted, all_bragg]
        if all_sigma_rdout:
            ret_subimgs += [all_sigma_rdout]
        if reorder:
            if spot_d and spot_d[0] is not None:
                order = np.argsort(spot_d)[::-1]
                spot_d = np.array(spot_d)[order]
                spot_hkl = np.array(spot_hkl)[order]
                spot_scale = np.array(spot_scale)[order]
                spot_pid_and_cent = np.array(spot_pid_and_cent)[order]

                for i in range(len(ret_subimgs)):
                    imgs = ret_subimgs[i]
                    imgs = [imgs[i] for i in order]
                    ret_subimgs[i] = imgs
        if return_stats:
            stats = {"spot_d": spot_d, "spot_hkl": spot_hkl, "spot_pid_and_cent": spot_pid_and_cent,
                     "spot_scale": spot_scale}
            return ret_subimgs + [stats]
        else:
            return ret_subimgs

    def set_diffBragg_refinement_flags(self, SIM):
        """Tell diffBragg which parameters will be refined (must be called before first GPU kernel)."""
        if self.P["lambda_offset"].refine:
            for lam_id in LAMBDA_IDS:
                SIM.D.refine(lam_id)
        if self.P["RotXYZ0_xtal0"].refine:
            SIM.D.refine(ROTX_ID)
            SIM.D.refine(ROTY_ID)
            SIM.D.refine(ROTZ_ID)
        if getattr(self, 'use_cholesky_Nabc', False) and "chol_L11" in self.P and self.P["chol_L11"].refine:
            SIM.D.refine(NCELLS_ID)
            SIM.D.refine(NCELLS_ID_OFFDIAG)
        else:
            if self.P["Nabc0"].refine:
                SIM.D.refine(NCELLS_ID)
            if self.P["Ndef0"].refine:
                SIM.D.refine(NCELLS_ID_OFFDIAG)
        for db_id, name in zip(PAN_OFS_IDS + PAN_XYZ_IDS, ["RotOrth", "RotFast", "RotSlow", "ShiftX", "ShiftY", "ShiftZ"]):
            pname = f"group0_{name}"
            if pname in self.P and self.P[pname].refine:
                SIM.D.refine(db_id)
        if self.P["ucell0"].refine:
            for i_ucell in range(len(self.ucell_man.variables)):
                SIM.D.refine(UCELL_ID_OFFSET + i_ucell)
        if self.P["eta_abc0"].refine:
            SIM.D.refine(ETA_ID)
        if self.P["detz_shift"].refine:
            SIM.D.refine(DETZ_ID)
        if self.P["gonio_angle"].refine:
            SIM.D.refine(GONIO_ANGLE_ID)
        if SIM.D.use_diffuse:
            SIM.D.refine(DIFFUSE_ID)
        if self.P["Bfactor"].refine:
            SIM.D.refine(BFACTOR_ID)
        if self.P["Baniso0"].refine:
            SIM.D.refine(BFACTOR_ANISO_ID)

    def Minimize(self, x0, SIM, i_shot=0):
        self.target = target = TargetFunc(SIM=SIM, niter_per_J=self.params.niter_per_J, profile=self.params.profile)

        # set up the refinement flags
        vary = np.ones(len(x0), bool)
        if SIM.refining_Fhkl:
            assert len(x0) == len(self.P)+SIM.Num_ASU*SIM.num_Fhkl_channels
        else:
            assert len(x0) == len(self.P)
        for p in self.P.values():
            if not p.refine:
                vary[p.xpos] = False

        target.vary = vary  # fixed flags
        target.x0 = np.array(x0, np.float64)  # initial full parameter list
        x0_for_refinement = target.x0[vary]

        if self.params.method is None:
            method = "Nelder-Mead"
        else:
            method = self.params.method

        maxfev = None
        if self.params.nelder_mead_maxfev is not None:
            maxfev = self.params.nelder_mead_maxfev #* self.npix_total

        at_min = None
        if self.params.logging.show_params_at_minimum:
            at_min = target.at_minimum

        if self.params.niter >0:
            assert self.params.hopper_save_freq is None
            at_min = self.at_minimum

        callback_kwargs = {"SIM":SIM, "i_shot": i_shot, "save_freq": self.params.hopper_save_freq}
        callback = lambda x: self.callback(x, callback_kwargs)
        target.terminate_after_n_converged_iterations = self.params.terminate_after_n_converged_iter
        target.percent_change_of_converged = self.params.converged_param_percent_change
        if method in ["L-BFGS-B", "BFGS", "CG", "dogleg", "SLSQP", "Newton-CG", "trust-ncg", "trust-krylov", "trust-exact", "trust-ncg"]:
            self.set_diffBragg_refinement_flags(SIM)

            min_kwargs = {'args': (self,SIM, True), "method": method, "jac": target.jac,
                          'hess': self.params.hess, 'callback':callback}
            if method=="L-BFGS-B":
                min_kwargs["options"] = {"ftol": self.params.ftol, "gtol": 1e-12, "maxfun":1e5,
                                         "maxiter":self.params.lbfgs_maxiter, "eps":1e-20}

        else:
            min_kwargs = {'args': (self,SIM, False), "method": method,
                          'callback': callback,
                          'options': {'maxiter': maxfev, 'adaptive': True,
                                      'fatol': self.params.nelder_mead_fatol}}

        if self.params.global_method=="basinhopping":
            HOPPER = basinhopping

            try:
                out = HOPPER(target, x0_for_refinement,
                                   niter=self.params.niter,
                                   minimizer_kwargs=min_kwargs,
                                   T=self.params.temp,
                                   callback=at_min,
                                   disp=False,
                                   stepsize=self.params.stepsize)
                target.x0[vary] = out.x
                MAIN_LOGGER.debug("Optimal results after basinhopping:")
                f, g, modelpix, J, sigZ, debug_s = target_func(target.x0, None, self, SIM,
                                                            compute_grad=False, return_all_zscores=False)
                MAIN_LOGGER.debug("<><><><><><><><><\n"+debug_s+"\n<><><><><><><><>")
            except StopIteration:
                pass

        else:
            bounds = [(-100,100)] * len(x0_for_refinement)  # TODO decide about bounds, usually x remains close to 1 during refinement
            print("Beginning the annealing process")
            args = min_kwargs.pop("args")
            if self.params.dual.no_local_search:
                compute_grads = args[-1]
                if compute_grads:
                    print("Warning, parameters setup to compute gradients, swicthing off because no_local_search=True")
                args = list(args)
                args[-1] = False  # switch off grad
                args = tuple(args)
            out = dual_annealing(target, bounds=bounds, args=args,
                                 no_local_search=self.params.dual.no_local_search,
                                 x0=x0_for_refinement,
                                 accept=self.params.dual.accept,
                                 visit=self.params.dual.visit,
                                 maxiter=self.params.niter,
                                 local_search_options=min_kwargs,
                                 callback=at_min)
            target.x0[vary] = out.x

        return target.x0

    def get_best_hop(self):
        """
        after refinement, explore all_hop_id, all_sigz, and all_f to find which basinhop resulted in the best minimization
        """
        assert self.target.all_f
        assert self.target.all_hop_id
        assert self.target.all_sigZ
        funcs_hops_sigz = list(zip(self.target.all_f, self.target.all_hop_id, self.target.all_sigZ))
        hop_id = lambda x: x[1] # group by hop id
        gb = groupby(sorted(funcs_hops_sigz,key=hop_id), key=hop_id)
        info = {hop_id:list(traces) for hop_id,traces in gb}

        min_f = np.inf
        min_sigz = np.inf
        hop_winner = 0
        for hop_id in info:
            f, _, sigz = zip(*info[hop_id])
            if min(f) < min_f:
                min_f = min(f)
                min_sigz = min(sigz)
                hop_winner = hop_id
                niter = len(f)
        return min_sigz, niter, hop_winner

    def callback(self, x, kwargs):
        save_freq = kwargs["save_freq"]
        i_shot = kwargs["i_shot"]
        SIM = kwargs["SIM"]
        target = self.target
        if save_freq is not None and target.iteration % save_freq==0 and target.iteration> 0:
            xall = target.x0.copy()
            xall[target.vary] = x
            self.save_up(xall, SIM, rank=self.rank, i_shot=i_shot)
        return

        rescaled_vals = np.zeros_like(xall)
        all_perc_change = []
        for name in self.P:
            if name.startswith("Fhkl_"):
                continue
            p = self.P[name]

            if not p.refine:
                continue
            xpos = p.xpos
            val = p.get_val(xall[xpos])
            log_s = "Iter %d: %s = %1.2g ." % (target.iteration, name, val)
            if name.startswith("scale_roi") and p.misc_data is not None:
                reso, (h, k, l) = p.misc_data
                log_s += "reso=%1.3f Ang. (h,k,l)=%d,%d,%d ." % (reso, h, k, l)
            rescaled_vals[xpos] = val
            if target.prev_iter_vals is not None:
                prev_val = target.prev_iter_vals[xpos]
                if val - prev_val == 0 and prev_val == 0:
                    val_percent_diff = 0
                elif prev_val == 0:
                    val_percent_diff = np.abs(val - prev_val) / val * 100.
                else:
                    val_percent_diff = np.abs(val - prev_val) / prev_val * 100.
                log_s += " Percent change = %1.2f%%" % val_percent_diff
                all_perc_change.append(val_percent_diff)

            if verbose:
                MAIN_LOGGER.debug(log_s)

        if all_perc_change:
            all_perc_change = np.abs(all_perc_change)
            ave_perc_change = np.mean(all_perc_change)
            num_perc_change_small = np.sum(all_perc_change < target.percent_change_of_converged)
            max_perc_change = np.max(all_perc_change)
            if num_perc_change_small == len(all_perc_change):
                target.all_converged_params += 1
            else:
                target.all_converged_params = 0
            if verbose:
                MAIN_LOGGER.info(
                    "Iter %d: Mean percent change = %1.2f%%. Num param with %%-change < %1.2f: %d/%d. Max %%-change=%1.2f%%"
                    % (target.iteration, ave_perc_change, target.percent_change_of_converged,
                       num_perc_change_small, len(all_perc_change), max_perc_change))

        target.prev_iter_vals = rescaled_vals
        if target.terminate_after_n_converged_iterations is not None and target.all_converged_params >= target.terminate_after_n_converged_iterations:
            # at this point prev_iter_vals are the converged parameters!
            raise StopIteration()  # Refinement has reached convergence!

    def save_up(self, x, SIM, rank=0, i_shot=0,
                save_fhkl_data=True, save_modeler_file=True,
                save_refl=True,
                save_sim_info=True,
                save_traces=True,
                save_pandas=True, save_expt=True, checker=None):
        """

        :param x: l-bfgs refinement parameters (reparameterized, e.g. unbounded)
        :param SIM: sim_data.SimData instance
        :param rank: MPI rank Id
        :param i_shot: shot index for this rank (assuming each rank processes more than one shot, this should increment)
        :param save_fhkl_data: whether to write mtz files
        :param save_modeler_file: whether to write the DataModeler .npy file (a pickle file)
        :param save_refl: whether to write a reflection table for this shot with updated xyzcal.px from diffBragg models
        :param save_sim_info: whether to write a text file showing the diffBragg state
        :param save_traces: whether to write a text file showing the refinement target functional and sigmaZ per iter
        :param save_pandas: whether to write a single-shot pandas dataframe containing optimized diffBragg params
        :param save_expt: whether to save a single-shot experiment file for this shot with optimized crystal model
        :return: returns the single shot pandas dataframe (whether or not it was written)
        """
        assert self.exper_name is not None
        assert self.refl_name is not None
        Modeler = self
        LOGGER = logging.getLogger("refine")
        if not Modeler.params.fix.perRoiScale or Modeler.params.use_perRoiScale:
            Modeler.best_model, opt_Jac = model(x, Modeler, SIM,  compute_grad=True, dont_rescale_gradient=True)
            variance_s = get_variance_s(Modeler, opt_Jac)
        else:
            Modeler.best_model, _ = model(x, Modeler, SIM, compute_grad=False)
        Modeler.best_model_includes_background = False
        LOGGER.info("Optimized values for i_shot %d:" % i_shot)

        basename = os.path.splitext(os.path.basename(self.exper_name))[0]

        if save_fhkl_data and SIM.refining_Fhkl:
            fhkl_scale_dir = hopper_io.make_rank_outdir(Modeler.params.outdir, "Fhkl_scale", rank)

            # ------------
            # here we run the command add_Fhkl_gradients one more time
            # , only this time we track all the asu indices that influence the model
            # This is a special call that requires openMP to have num_threads=1 because
            # I do not not how to interact with an unordered_set in openMP
            resid = self.all_data - (self.best_model + self.all_background)  # here best model is just the Bragg portion, hence we add background
            V = self.best_model + self.all_sigma_rdout ** 2
            Gparam = self.P["G_xtal0"]
            G = Gparam.get_val(x[Gparam.xpos])
            # here we must use the CPU method
            force_cpu = SIM.D.force_cpu
            SIM.D.force_cpu = True
            MAIN_LOGGER.info("Getting Fhkl errors (forcing CPUkernel usage)... might take some time")
            Fhkl_scale_errors = SIM.D.add_Fhkl_gradients(
                self.pan_fast_slow, resid, V, self.all_trusted, self.all_freq,
                SIM.num_Fhkl_channels, G, track=True, errors=True)
            SIM.D.force_gpu = force_cpu
            # ------------

            inds = np.sort(np.array(SIM.D.Fhkl_gradient_indices))

            num_asu = len(SIM.asu_map_int)
            idx_to_asu = {idx:asu for asu,idx in SIM.asu_map_int.items()}
            all_nominal_hkl = set(self.hi_asu_perpix)
            for i_chan in range(SIM.num_Fhkl_channels):
                sel = (inds >= i_chan*num_asu) * (inds < (i_chan+1)*num_asu)
                if not np.any(sel):
                    continue
                inds_chan= inds[sel] - i_chan*num_asu
                assert np.max(inds_chan) < num_asu
                asu_hkls = []
                is_nominal_hkl = []
                scale_facs = []
                scale_vars = []
                for i_hkl in inds_chan:
                    asu = idx_to_asu[i_hkl]
                    xpos = i_hkl + i_chan*num_asu
                    #xval = x[xpos]
                    scale_fac = SIM.Fhkl_scales[xpos]
                    hessian_term = Fhkl_scale_errors[xpos]
                    with np.errstate(all='ignore'):
                        scale_var = 1/hessian_term
                    asu_hkls.append(asu)
                    scale_facs.append(scale_fac)
                    scale_vars.append(scale_var)
                    is_nominal_hkl.append(asu in all_nominal_hkl)
                scale_fname = os.path.join(fhkl_scale_dir, "%s_%s_%d_%d_channel%d_scale.npz"\
                                     % (Modeler.params.tag, basename, i_shot, self.exper_idx, i_chan))
                np.savez(scale_fname, asu_hkl=asu_hkls, scale_fac=scale_facs, scale_var=scale_vars,
                         is_nominal_hkl=is_nominal_hkl)


        # TODO: pretty formatting ?
        if Modeler.target is not None:
            # hop number, gradient descent index (resets with each new hop), target functional
            trace0, trace1, trace2 = Modeler.target.all_hop_id, Modeler.target.all_f, Modeler.target.all_sigZ
            trace_data = np.array([trace0, trace1, trace2]).T

            if save_traces:
                rank_trace_outdir = hopper_io.make_rank_outdir(Modeler.params.outdir, "traces", rank)
                trace_path = os.path.join(rank_trace_outdir, "%s_%s_%d_%d_traces.txt"
                                          % (Modeler.params.tag, basename, i_shot, self.exper_idx))
                np.savetxt(trace_path, trace_data, fmt="%s")

            Modeler.niter = len(trace0)
            Modeler.sigz = trace2[-1]

        shot_df = hopper_io.save_to_pandas(x, Modeler, SIM, self.exper_name, Modeler.params, Modeler.E, i_shot,
                                           self.refl_name, None, rank, write_expt=save_expt, write_pandas=save_pandas,
                                           exp_idx=self.exper_idx)

        if isinstance(Modeler.all_sigma_rdout, np.ndarray):
            data_subimg, model_subimg, trusted_subimg, bragg_subimg, sigma_rdout_subimg = Modeler.get_data_model_pairs()
        else:
            data_subimg, model_subimg, trusted_subimg, bragg_subimg = Modeler.get_data_model_pairs()
            sigma_rdout_subimg = None

        wavelen_subimg = []
        if SIM.D.store_ave_wavelength_image:
            bm = Modeler.best_model.copy()
            Modeler.best_model = SIM.D.ave_wavelength_image().as_numpy_array()
            Modeler.best_model_includes_background = True
            _, wavelen_subimg, _, _ = Modeler.get_data_model_pairs()
            Modeler.best_model = bm
            Modeler.best_model_includes_background = False

        if save_refl:

            Fp1 = SIM.D.Fhkl  # crystal.miller_array
            Fp1_map = {h: amp for h, amp in zip(Fp1.indices(), Fp1.data())}

            rank_refls_outdir = hopper_io.make_rank_outdir(Modeler.params.outdir, "refls", rank)
            new_refls_file = os.path.join(rank_refls_outdir, "%s_%s_%d_%d.refl"
                                          % (Modeler.params.tag, basename, i_shot, self.exper_idx))
            new_refls = deepcopy(Modeler.refls)
            has_xyzcal = 'xyzcal.px' in list(new_refls.keys())
            if has_xyzcal:
                new_refls['dials.xyzcal.px'] = deepcopy(new_refls['xyzcal.px'])
            per_refl_scales = flex.double(len(new_refls), 1)
            per_refl_variance_s = flex.double(len(new_refls), 1)
            per_refl_dbI = flex.double(len(new_refls),1)
            per_refl_dbIvar= flex.double(len(new_refls),1)
            per_refl_scores = flex.double(len(new_refls), -1)
            per_refl_ntrust = flex.int(len(new_refls), -1)
            per_refl_sigz = flex.double(len(new_refls), -1)
            new_xycalcs = flex.vec3_double(len(Modeler.refls), (np.nan, np.nan, np.nan))
            #sigmaZs = []
            for i_roi in range(len(data_subimg)):
                dat = data_subimg[i_roi]
                fit = model_subimg[i_roi]
                trust = trusted_subimg[i_roi]
                if sigma_rdout_subimg is not None:
                    sig = np.sqrt(fit + sigma_rdout_subimg[i_roi] ** 2)
                else:
                    sig = np.sqrt(fit + Modeler.nominal_sigma_rdout ** 2)
                Z = (dat - fit) / sig
                sigmaZ = np.nan
                if np.any(trust):
                    sigmaZ = Z[trust].std()
                #sigmaZs.append(sigmaZ)
                if bragg_subimg[0] is not None:
                    if np.any(bragg_subimg[i_roi] > 0):
                        ref_idx = Modeler.refls_idx[i_roi]
                        ref = Modeler.refls[ref_idx]
                        I = bragg_subimg[i_roi]
                        Y, X = np.indices(bragg_subimg[i_roi].shape)
                        x1, x2, y1, y2 = Modeler.rois[i_roi]
                        com_x, com_y, _ = ref["xyzobs.px.value"]
                        com_x = int(com_x - x1 - 0.5)
                        com_y = int(com_y - y1 - 0.5)
                        # make sure at least some signal is at the centroid! otherwise this is likely a neighboring spot
                        try:
                            if I[com_y, com_x] == 0:
                                continue
                        except IndexError:
                            continue
                        X += x1
                        Y += y1
                        Isum = I.sum()
                        xcom = (X * I).sum() / Isum
                        ycom = (Y * I).sum() / Isum
                        com = xcom + .5, ycom + .5, 0
                        new_xycalcs[ref_idx] = com
                        if not Modeler.params.fix.perRoiScale or Modeler.params.use_perRoiScale:
                            scale_p = Modeler.P["scale_roi%d" % i_roi]
                            scale = scale_p.get_val(x[scale_p.xpos])
                            scale_var = variance_s[scale_p.xpos]
                            per_refl_scales[ref_idx] = scale
                            per_refl_variance_s[ref_idx] = scale_var
                            if "miller_index" in new_refls:
                                h,k,l = new_refls["miller_index"][ref_idx]
                                Famp = Fp1_map[(h,k,l)]
                                I_hkl = (Famp**2) *scale
                                Ivar_hkl = (I_hkl/scale)**2 * scale_var
                                per_refl_dbI[ref_idx] = I_hkl
                                per_refl_dbIvar[ref_idx] = Ivar_hkl

                        if checker is not None:
                            score = checker.score(dat, fit)
                            per_refl_scores[ref_idx] = score
                        per_refl_ntrust[ref_idx] = int(trust.sum())
                        per_refl_sigz[ref_idx] = sigmaZ

            new_refls["xyzcal.px"] = new_xycalcs
            new_refls["scores"] = per_refl_scores
            new_refls["ntrust"] = per_refl_ntrust
            new_refls["sigz"] = per_refl_sigz
            new_refls["scale_factor"] = per_refl_scales
            new_refls["intensity.db.value"] = per_refl_dbI
            new_refls["intensity.db.variance"] = per_refl_dbIvar
            # TODO : convert to xyzcal.px and xyzobx.px.value to mm

            # try to set new deltapsi.cal
            E = deepcopy(Modeler.E)
            E.crystal.set_A(shot_df.Amats.values[0])
            new_refls["deff"] = flex.double(len(new_refls), -1)
            new_refls["eta_eff"] = flex.double(len(new_refls), -1)
            new_refls["dpsi.new"] = flex.double(len(new_refls), 0)
            for _ in range(10):
                try:
                  _, dpsi_new, deff, eta_eff = extract_mosaic_parameters_using_lambda_spread(
                            E, new_refls, verbose=False, Deff_init=np.random.uniform(100,900))
                  new_refls["dpsi.new"] = dpsi_new
                  new_refls["deff"] = flex.double(len(new_refls), deff)
                  new_refls["eta_eff"] = flex.double(len(new_refls), eta_eff)
                  break
                except Exception as err:
                    pass
                    #if "Model has diverged" in str(err):
                    #    continue
            if Modeler.params.filter_unpredicted_refls_in_output:
                sel = [not np.isnan(x) for x, y, z in new_xycalcs]
                new_refls = new_refls.select(flex.bool(sel))

            # filter the diffBragg intensities for wild outliers
            if not Modeler.params.fix.perRoiScale or Modeler.params.use_perRoiScale:
                val = new_refls['intensity.db.value'].as_numpy_array()
                var = new_refls['intensity.db.variance'].as_numpy_array()
                #is_inf_var = np.isinf(var)
                is_neg_var = var < 0
                is_one_val = val==1
                is_zero_val = val==0
                is_large_var = var > 1e18
                goodies = (~is_zero_val) * (~is_neg_var) * (~is_one_val) * (~is_large_var)
                #baddies = np.logical_or(~is_one_val, ~is_inf_var)
                #goodies = ~baddies
                new_refls = new_refls.select(flex.bool(goodies))
            eid = new_refls.experiment_identifiers()
            eid_key = list(set(new_refls['id']))
            if len(eid_key) == 1:
                eid_key = eid_key[0]
                eid[eid_key] = new_refls_file
                new_refls.as_file(new_refls_file)
                shot_df["save_up_refls"] = os.path.abspath(new_refls_file)
            else:
                shot_df["save_up_refls"] = ""

        if save_modeler_file:
            rank_imgs_outdir = hopper_io.make_rank_outdir(Modeler.params.outdir, "imgs", rank)
            modeler_file = os.path.join(rank_imgs_outdir,
                                        "%s_%s_%d_%d_modeler.npy"
                                        % (Modeler.params.tag, basename, i_shot, self.exper_idx))
            np.save(modeler_file, Modeler)
        if save_sim_info:
            spectrum_file = os.path.join(rank_imgs_outdir,
                                         "%s_%s_%d_%d_spectra.lam"
                                         % (Modeler.params.tag, basename, i_shot, self.exper_idx))
            rank_SIMlog_outdir = hopper_io.make_rank_outdir(Modeler.params.outdir, "simulator_state", rank)
            SIMlog_path = os.path.join(rank_SIMlog_outdir, "%s_%s_%d_%d.txt"
                                       % (Modeler.params.tag, basename, i_shot, self.exper_idx))
            write_SIM_logs(SIM, log=SIMlog_path, lam=spectrum_file)

        if Modeler.params.refiner.debug_pixel_panelfastslow is not None:
            # TODO separate diffBragg logger
            utils.show_diffBragg_state(SIM.D, Modeler.params.refiner.debug_pixel_panelfastslow)

        # Per-spot diagnostic CSV (lightweight, does not require debug_mode)
        if getattr(Modeler.params, 'save_spot_diagnostics', False):
            rank_diag_outdir = hopper_io.make_rank_outdir(Modeler.params.outdir, "spot_diagnostics", rank)
            diag_path = os.path.join(rank_diag_outdir, "%s_%s_%d_%d_spots.csv"
                                     % (Modeler.params.tag, basename, i_shot, self.exper_idx))

            # Extract refined shot-level parameters (same for all ROIs in this shot)
            G_val = Modeler.P["G_xtal0"].get_val(x[Modeler.P["G_xtal0"].xpos])
            B_val = Modeler.P["Bfactor"].get_val(x[Modeler.P["Bfactor"].xpos])
            rot_names = ["RotXYZ%d_xtal0" % i for i in range(3)]
            rot_vals = [Modeler.P[n].get_val(x[Modeler.P[n].xpos]) for n in rot_names]
            chol_names = ["chol_L11", "chol_L21", "chol_L22", "chol_L31", "chol_L32", "chol_L33"]
            if chol_names[0] in Modeler.P:
                L_vals = [Modeler.P[n].get_val(x[Modeler.P[n].xpos]) for n in chol_names]
                Na = L_vals[0]**2
                Nb = L_vals[1]**2 + L_vals[2]**2
                Nc = L_vals[3]**2 + L_vals[4]**2 + L_vals[5]**2
            else:
                Na_p, Nb_p, Nc_p = Modeler.P["Nabc0"], Modeler.P["Nabc1"], Modeler.P["Nabc2"]
                Na = Na_p.get_val(x[Na_p.xpos])
                Nb = Nb_p.get_val(x[Nb_p.xpos])
                Nc = Nc_p.get_val(x[Nc_p.xpos])

            # Refined unit cell parameters
            try:
                nucell = len(Modeler.ucell_man.variables)
                ucell_p = [Modeler.P["ucell%d" % i] for i in range(nucell)]
                ucell_var = [p.get_val(x[p.xpos]) for p in ucell_p]
                Modeler.ucell_man.variables = ucell_var
                uc_params = Modeler.ucell_man.unit_cell_parameters
                uc_a, uc_b, uc_c = uc_params[0], uc_params[1], uc_params[2]
                uc_al, uc_be, uc_ga = uc_params[3], uc_params[4], uc_params[5]
            except Exception:
                uc_a = uc_b = uc_c = uc_al = uc_be = uc_ga = np.nan

            # Detector geometry for lab-frame coordinates
            det = Modeler.E.detector

            rows = []
            for i_roi in range(len(data_subimg)):
                dat = data_subimg[i_roi]
                fit = model_subimg[i_roi]
                trust = trusted_subimg[i_roi]
                # sigma_z
                if sigma_rdout_subimg is not None:
                    sig = np.sqrt(fit + sigma_rdout_subimg[i_roi] ** 2)
                else:
                    sig = np.sqrt(fit + Modeler.nominal_sigma_rdout ** 2)
                Z = (dat - fit) / sig
                sigmaZ = np.nan
                ntrust = 0
                if np.any(trust):
                    sigmaZ = Z[trust].std()
                    ntrust = int(trust.sum())
                # HKL and d-spacing
                hkl = Modeler.Hi[i_roi] if Modeler.Hi else (0, 0, 0)
                hkl_asu = Modeler.Hi_asu[i_roi] if Modeler.Hi_asu else hkl
                d_spacing = 1.0 / Modeler.Q[i_roi] if hasattr(Modeler, 'Q') and Modeler.Q else np.nan
                # panel and centroid (pixels)
                x1, x2, y1, y2 = Modeler.rois[i_roi]
                pid = Modeler.pids[i_roi]
                cent_x = (x1 + x2) / 2.0
                cent_y = (y1 + y2) / 2.0
                # Lab-frame coordinates (mm) from detector geometry
                try:
                    panel = det[pid]
                    lab_xyz = panel.get_pixel_lab_coord((cent_x, cent_y))
                    det_x_mm = lab_xyz[0]
                    det_y_mm = lab_xyz[1]
                    det_z_mm = lab_xyz[2]
                except Exception:
                    det_x_mm = det_y_mm = det_z_mm = np.nan
                # tilt plane
                ta, tb, tc = Modeler.tilt_abc[i_roi] if Modeler.tilt_abc else (np.nan, np.nan, np.nan)
                # tilt plane residual variance (diagonal of covariance)
                cov = Modeler.tilt_cov[i_roi] if hasattr(Modeler, 'tilt_cov') and Modeler.tilt_cov else None
                if cov is not None and hasattr(cov, '__len__'):
                    try:
                        bg_resid_var = float(np.trace(cov) / 3.0)
                    except (TypeError, ValueError):
                        bg_resid_var = np.nan
                else:
                    bg_resid_var = np.nan
                # data/model max in trusted region
                dat_max = float(dat[trust].max()) if np.any(trust) else np.nan
                mod_max = float(fit[trust].max()) if np.any(trust) else np.nan
                # integrated signal (sum over trusted pixels, data - background)
                bg = Modeler.all_background[Modeler.roi_id == i_roi] if hasattr(Modeler, 'all_background') else None
                if bg is not None and np.any(trust):
                    dat_flat = Modeler.all_data[Modeler.roi_id == i_roi]
                    trust_flat = Modeler.all_trusted[Modeler.roi_id == i_roi]
                    dat_sum = float((dat_flat - bg)[trust_flat].sum())
                    mod_flat = (Modeler.best_model + Modeler.all_background)[Modeler.roi_id == i_roi] if Modeler.best_model is not None else None
                    mod_sum = float((mod_flat - bg)[trust_flat].sum()) if mod_flat is not None else np.nan
                else:
                    dat_sum = mod_sum = np.nan
                # bragg signal fraction
                bragg = bragg_subimg[i_roi] if bragg_subimg and bragg_subimg[0] is not None else None
                bragg_max = float(bragg[trust].max()) if bragg is not None and np.any(trust) else np.nan
                rows.append({
                    'roi_id': i_roi,
                    'h': hkl[0], 'k': hkl[1], 'l': hkl[2],
                    'h_asu': hkl_asu[0], 'k_asu': hkl_asu[1], 'l_asu': hkl_asu[2],
                    'd_spacing': d_spacing,
                    'panel': pid,
                    'cent_x': cent_x, 'cent_y': cent_y,
                    'det_x_mm': det_x_mm, 'det_y_mm': det_y_mm,
                    'sigma_z': sigmaZ,
                    'n_trusted': ntrust,
                    'n_pixels': dat.size,
                    'tilt_a': ta, 'tilt_b': tb, 'tilt_c': tc,
                    'bg_resid_var': bg_resid_var,
                    'dat_max': dat_max, 'mod_max': mod_max,
                    'dat_sum': dat_sum, 'mod_sum': mod_sum,
                    'bragg_max': bragg_max,
                })
            spot_df = pandas.DataFrame(rows)
            spot_df.to_csv(diag_path, index=False, float_format='%.4f')

            # Also append a one-line shot summary to a rank-level file
            summary_path = os.path.join(rank_diag_outdir, "%s_rank%d_shot_summary.csv" % (Modeler.params.tag, rank))
            sigz_vals = spot_df['sigma_z'].dropna()
            from simtbx.diffBragg.utils import is_outlier
            n_outlier_sigz = int(is_outlier(sigz_vals.values).sum()) if len(sigz_vals) >= 3 else 0
            shot_summary = {
                'shot': i_shot,
                'exp_name': basename,
                'n_rois': len(rows),
                'G': G_val,
                'B': B_val,
                'rotX': rot_vals[0], 'rotY': rot_vals[1], 'rotZ': rot_vals[2],
                'Na': Na, 'Nb': Nb, 'Nc': Nc,
            }
            if chol_names[0] in Modeler.P:
                for cn, lv in zip(chol_names, L_vals):
                    shot_summary[cn] = lv
            shot_summary.update({
                'uc_a': uc_a, 'uc_b': uc_b, 'uc_c': uc_c,
                'uc_al': uc_al, 'uc_be': uc_be, 'uc_ga': uc_ga,
                'sigz_mean': sigz_vals.mean() if len(sigz_vals) else np.nan,
                'sigz_median': sigz_vals.median() if len(sigz_vals) else np.nan,
                'sigz_max': sigz_vals.max() if len(sigz_vals) else np.nan,
                'frac_sigz_lt2': (sigz_vals < 2).mean() if len(sigz_vals) else np.nan,
                'frac_sigz_lt5': (sigz_vals < 5).mean() if len(sigz_vals) else np.nan,
                'n_outlier_sigz': n_outlier_sigz,
            })
            sum_df = pandas.DataFrame([shot_summary])
            write_header = not os.path.exists(summary_path)
            sum_df.to_csv(summary_path, mode='a', header=write_header, index=False, float_format='%.4f')

        return shot_df


def convolve_model_with_psf(model_pix, J, mod, SIM, PSF=None, psf_args=None,
        roi_id_slices=None, roi_id_unique=None):
    if not SIM.use_psf:
        return model_pix, J
    if PSF is None:
        PSF = SIM.PSF
        assert PSF is not None
    if psf_args is None:
        psf_args = SIM.psf_args
        assert psf_args is not None
    if roi_id_slices is None:
        roi_id_slices = mod.roi_id_slices
    if roi_id_unique is None:
        roi_id_unique = mod.roi_id_unique

    coords = mod.pan_fast_slow.as_numpy_array()
    pid = coords[0::3]
    fid = coords[1::3]
    sid = coords[2::3]

    ref_xpos = []  # Jacobian index (J[xpos]) for the refined  parameters that arent roi scale factors
    if J is not None:
        for name in mod.P:
            if name.startswith("scale_roi") or name.startswith("Fhkl_"):
                continue
            p = mod.P[name]
            if p.refine:
                ref_xpos.append( p.xpos)

    for i in roi_id_unique:
        roi_p = mod.P["scale_roi%d" % i]
        for slc in roi_id_slices[i]:
            pvals = pid[slc]
            fvals = fid[slc]
            svals = sid[slc]
            f0 = fvals.min()
            s0 = svals.min()
            f1 = fvals.max()
            s1 = svals.max()
            fdim = int(f1-f0+1)
            sdim = int(s1-s0+1)
            img = model_pix[slc].reshape((sdim, fdim))
            img = psf.convolve_with_psf(img, psf=PSF, **psf_args)
            model_pix[slc] = img.ravel()
            if roi_p.refine and J is not None:
                deriv_img = J[roi_p.xpos, slc].reshape((sdim, fdim))
                deriv_img = psf.convolve_with_psf(deriv_img, psf=PSF, **psf_args)
                J[roi_p.xpos, slc] = deriv_img.ravel()

            for xpos in ref_xpos: # if J is None, then ref_xpos should be empty!
                deriv_img = J[xpos, slc].reshape((sdim, fdim))
                deriv_img = psf.convolve_with_psf(deriv_img, psf=PSF, **psf_args)
                J[xpos, slc] = deriv_img.ravel()

    return model_pix, J


def print_params(Mod, x):
    """
    :param Mod: Data modeler
    :param x: refinement parameters
    """
    val_s = ""
    for p in Mod.P.values():
        if p.name.startswith("Fhkl_"):
            continue
        if p.refine:
            xval = x[p.xpos]
            val = p.get_val(xval)
            name = p.name
            if name == "detz_shift":
                val = val * 1e3
                name = p.name + "_mm"
            if name == "gonio_angle":
                name = p.name + "_deg"
            if "Rot" in name:
                val = val*180 / np.pi
                name = p.name +"_deg"
            val_s += "%s=%.4f, " % (name, val)
    MAIN_LOGGER.debug(val_s)


def model(x, Mod, SIM,  compute_grad=True, dont_rescale_gradient=False, update_spectrum=False,
          update_Fhkl_scales=True):

    if Mod.P.refining_detector:
        if not hasattr(SIM, "panel_group_from_id"):
            SIM.panel_group_from_id = {0:0}
            SIM.panel_groups_refined = {0}
            SIM.panel_reference_from_id = {0: deepcopy(SIM.detector[0].get_origin())}
            Mod.group_id_slices = {0: [slice(0, len(Mod.all_data), 1)]}
            Mod.unique_panel_group_ids = {0}
        update_detector(x, Mod.P, SIM)
    if Mod.params.logging.parameters:
        print_params(Mod, x)

    pfs = Mod.pan_fast_slow

    if update_spectrum:
        # update the photon energy spectrum for this shot
        SIM.beam.spectrum = Mod.nanoBragg_beam_spectrum
        SIM.D.xray_beams = SIM.beam.xray_beams
        # update Fhkl channels
        if Mod.Fhkl_channel_ids is not None:
            SIM.D.update_Fhkl_channels(Mod.Fhkl_channel_ids)

    if SIM.refining_Fhkl and update_Fhkl_scales:  # once per iteration
        nscales = SIM.Num_ASU*SIM.num_Fhkl_channels
        current_Fhkl_xvals = x[-nscales:]
        if SIM.Fhkl_linear:
            SIM.Fhkl_amplitudes = np.sqrt(SIM.Fhkl_scales_init) + SIM.Fhkl_sigmas * (current_Fhkl_xvals - 1)
            SIM.Fhkl_scales = SIM.Fhkl_amplitudes ** 2
        else:
            SIM.Fhkl_scales = SIM.Fhkl_scales_init * np.exp(SIM.Fhkl_sigmas * (current_Fhkl_xvals - 1))
        SIM.D.update_Fhkl_scale_factors(SIM.Fhkl_scales, SIM.num_Fhkl_channels)

    # get the unit cell variables
    nucell = len(Mod.ucell_man.variables)
    ucell_params = [Mod.P["ucell%d" % i_uc] for i_uc in range(nucell)]
    ucell_xpos = [p.xpos for p in ucell_params]
    unitcell_var_reparam = [x[xpos] for xpos in ucell_xpos]
    unitcell_variables = [ucell_params[i].get_val(xval) for i, xval in enumerate(unitcell_var_reparam)]
    Mod.ucell_man.variables = unitcell_variables
    Bmatrix = Mod.ucell_man.B_recipspace
    SIM.D.Bmatrix = Bmatrix
    if compute_grad:
        for i_ucell in range(len(unitcell_variables)):
            SIM.D.set_ucell_derivative_matrix(
                i_ucell + UCELL_ID_OFFSET,
                Mod.ucell_man.derivative_matrices[i_ucell])

    # update the mosaicity here
    eta_params = [Mod.P["eta_abc%d" % i_eta] for i_eta in range(3)]
    if SIM.umat_maker is not None:
        # we are modeling mosaic spread
        eta_abc = [p.get_val(x[p.xpos]) for p in eta_params]
        if not SIM.D.has_anisotropic_mosaic_spread:
            eta_abc = eta_abc[0]
        SIM.update_umats_for_refinement(eta_abc)

#   detector parameters
    DetZ = Mod.P["detz_shift"]
    x_shiftZ = x[DetZ.xpos]
    shiftZ = DetZ.get_val(x_shiftZ)
    SIM.D.shift_origin_z(SIM.detector, shiftZ)

#   goniometer parameters
    GonioAng = Mod.P["gonio_angle"]
    if GonioAng.refine:
        x_gonio_angle = x[GonioAng.xpos]
        gonio_angle = GonioAng.get_val(x_gonio_angle)
        gonio_ax = SIM.D.spindle_axis
        utils.update_SIM_with_gonio(SIM, delta_phi=gonio_angle, num_phi_steps=Mod.params.simulator.gonio.phi_steps,
                                    spindle_axis=gonio_ax)

    if Mod.P["lambda_offset"].refine:
        p0 = Mod.P["lambda_offset"]
        p1 = Mod.P["lambda_scale"]
        lambda_coef = p0.get_val(x[p0.xpos]), p1.get_val(x[p1.xpos])
        SIM.D.lambda_coefficients = lambda_coef

    # Mosaic block
    Nabc_params = [Mod.P["Nabc%d" % (i_n,)] for i_n in range(3)]
    Ndef_params = [Mod.P["Ndef%d" % (i_n,)] for i_n in range(3)]

    use_cholesky = getattr(Mod, 'use_cholesky_Nabc', False)
    chol_params = None
    if use_cholesky:
        chol_names = ["chol_L11", "chol_L21", "chol_L22", "chol_L31", "chol_L32", "chol_L33"]
        chol_params = [Mod.P[n] for n in chol_names]
        L11, L21, L22, L31, L32, L33 = [p.get_val(x[p.xpos]) for p in chol_params]
        # NABC = L^T * L (positive definite)
        Na = L11*L11
        Nb = L21*L21 + L22*L22
        Nc = L31*L31 + L32*L32 + L33*L33
        Nd = L11*L21
        Ne = L21*L31 + L22*L32
        Nf = L11*L31
        SIM.D.set_ncells_values(tuple([Na, Nb, Nc]))
        SIM.D.Ncells_def = Nd, Ne, Nf
    else:
        Na, Nb, Nc = [n_param.get_val(x[n_param.xpos]) for n_param in Nabc_params]
        if SIM.D.isotropic_ncells:
            Nb = Na
            Nc = Na
        SIM.D.set_ncells_values(tuple([Na, Nb, Nc]))

        Nd, Ne, Nf = [n_param.get_val(x[n_param.xpos]) for n_param in Ndef_params]
        if SIM.D.isotropic_ncells:
            Ne = Nd
            Nf = Nd
        SIM.D.Ncells_def = Nd, Ne, Nf

    # per-image B-factor
    Bfac_param = Mod.P["Bfactor"]
    Bfac_val = Bfac_param.get_val(x[Bfac_param.xpos])
    SIM.D.Bfactor_image = Bfac_val

    # anisotropic B-factor
    Baniso_params = [Mod.P["Baniso%d" % i] for i in range(6)]
    if Baniso_params[0].refine:
        Baniso_vals = tuple(p.get_val(x[p.xpos]) for p in Baniso_params)
        SIM.D.Bfactor_aniso = Baniso_vals

    # diffuse signals
    if SIM.D.use_diffuse:
        diffuse_params_lookup = {}
        iso_flags = {'gamma':SIM.isotropic_diffuse_gamma, 'sigma':SIM.isotropic_diffuse_sigma}
        for diff_type in ['gamma', 'sigma']:
            diff_params = [Mod.P["diffuse_%s%d" % (diff_type,i_gam)] for i_gam in range(3)]
            diffuse_params_lookup[diff_type] = diff_params
            diff_vals = []
            for i_diff, param in enumerate(diff_params):
                val = param.get_val(x[param.xpos])
                if iso_flags[diff_type]:
                    diff_vals = [val]*3
                    break
                else:
                    diff_vals.append(val)
            if diff_type == "gamma":
                SIM.D.diffuse_gamma = tuple(diff_vals)
            else:
                SIM.D.diffuse_sigma = tuple(diff_vals)

    npix = int(len(pfs) / 3)
    nparam = len(x)
    J = None
    if compute_grad:
        # This should be all params save the Fhkl params
        nfhkl = SIM.Num_ASU*SIM.num_Fhkl_channels if SIM.refining_Fhkl else 0
        J = np.zeros((nparam - nfhkl, npix))  # gradients

    model_pix = None
    #TODO check roiScales mode and if its broken, git rid of it!
    model_pix_noRoi = None

    # extract the scale factors per ROI, these might correspond to structure factor intensity scale factors, and quite possibly might result in overfits!
    if not Mod.params.fix.perRoiScale or Mod.params.use_perRoiScale:
        perRoiParams = [Mod.P["scale_roi%d" % roi_id] for roi_id in Mod.roi_id_unique]
        perRoiScaleFactors = [p.get_val(x[p.xpos]) for p in perRoiParams]
        if not isinstance(Mod.roiScalesPerPix, np.ndarray):
            Mod.roiScalesPerPix = np.ones(npix)
        for i_roi, roi_id in enumerate(Mod.roi_id_unique):
            slc = Mod.roi_id_slices[roi_id][0]
            Mod.roiScalesPerPix[slc] = perRoiScaleFactors[i_roi]

    for i_xtal in range(Mod.num_xtals):

        if hasattr(Mod, "Umatrices"):  # reflects new change for modeling multi-crystal experiments
            SIM.D.Umatrix = Mod.Umatrices[i_xtal]

        RotXYZ_params = [Mod.P["RotXYZ%d_xtal%d" % (i_rot, i_xtal)] for i_rot in range(3)]
        rotX,rotY,rotZ = [rot_param.get_val(x[rot_param.xpos]) for rot_param in RotXYZ_params]

        ## update parameters:
        # TODO: if not refining Umat, assert these are 0 , and dont set them here
        SIM.D.set_value(ROTX_ID, rotX)
        SIM.D.set_value(ROTY_ID, rotY)
        SIM.D.set_value(ROTZ_ID, rotZ)

        if Mod.params.symmetrize_Flatt:
            RXYZU = hopper_io.diffBragg_Umat(rotX, rotY, rotZ, SIM.D.Umatrix)
            Cryst = deepcopy(SIM.crystal.dxtbx_crystal)
            A = RXYZU * Mod.ucell_man.B_realspace
            A_recip = A.inverse().transpose()
            Cryst.set_A(A_recip)
            symbol = SIM.crystal.space_group_info.type().lookup_symbol()
            SIM.D.set_mosaic_blocks_sym(Cryst, symbol , Mod.params.simulator.crystal.num_mosaicity_samples,
                                        refining_eta=not Mod.params.fix.eta_abc)

        G = Mod.P["G_xtal%d" % i_xtal]
        scale = G.get_val(x[G.xpos])

        SIM.D.add_diffBragg_spots(pfs)

        pix_noRoiScale = SIM.D.raw_pixels_roi[:npix]
        pix_noRoiScale = pix_noRoiScale.as_numpy_array()

        pix = pix_noRoiScale * Mod.roiScalesPerPix

        if model_pix is None:
            model_pix = scale*pix
            model_pix_noRoi = scale*pix_noRoiScale
        else:
            model_pix += scale*pix
            model_pix_noRoi += scale*pix_noRoiScale

        if compute_grad:
            if G.refine:
                scale_grad = pix  # TODO double check multi crystal case
                scale_grad = G.get_deriv(x[G.xpos], scale_grad)
                J[G.xpos] += scale_grad

            if RotXYZ_params[0].refine:
                for i_rot in range(3):
                    rot_grad = scale * SIM.D.get_derivative_pixels(ROTXYZ_IDS[i_rot]).as_numpy_array()[:npix]
                    rot_p = RotXYZ_params[i_rot]
                    rot_grad = rot_p.get_deriv(x[rot_p.xpos], rot_grad)
                    J[rot_p.xpos] += rot_grad

            if use_cholesky and chol_params is not None and chol_params[0].refine:
                # Cholesky chain-rule: dI/dLij from kernel's dI/dNa..dI/dNf
                Nabc_grads = SIM.D.get_ncells_derivative_pixels()
                Ndef_grads = SIM.D.get_ncells_def_derivative_pixels()
                dI_dNa = scale * Nabc_grads[0][:npix].as_numpy_array()
                dI_dNb = scale * Nabc_grads[1][:npix].as_numpy_array()
                dI_dNc = scale * Nabc_grads[2][:npix].as_numpy_array()
                dI_dNd = scale * Ndef_grads[0][:npix].as_numpy_array()
                dI_dNe = scale * Ndef_grads[1][:npix].as_numpy_array()
                dI_dNf = scale * Ndef_grads[2][:npix].as_numpy_array()
                # Chain rule: NABC = L^T*L where L is lower triangular
                # Na=L11^2, Nb=L21^2+L22^2, Nc=L31^2+L32^2+L33^2
                # Nd=L11*L21, Ne=L21*L31+L22*L32, Nf=L11*L31
                dI_dL = [
                    dI_dNa * 2*L11 + dI_dNd * L21 + dI_dNf * L31,     # dI/dL11
                    dI_dNd * L11 + dI_dNb * 2*L21 + dI_dNe * L31,     # dI/dL21
                    dI_dNb * 2*L22 + dI_dNe * L32,                      # dI/dL22
                    dI_dNf * L11 + dI_dNe * L21 + dI_dNc * 2*L31,     # dI/dL31
                    dI_dNe * L22 + dI_dNc * 2*L32,                      # dI/dL32
                    dI_dNc * 2*L33,                                      # dI/dL33
                ]
                for i_chol in range(6):
                    p = chol_params[i_chol]
                    chol_grad = p.get_deriv(x[p.xpos], dI_dL[i_chol])
                    J[p.xpos] += chol_grad
            else:
                if Nabc_params[0].refine:
                    Nabc_grads = SIM.D.get_ncells_derivative_pixels()
                    for i_n in range(3):
                        N_grad = scale*(Nabc_grads[i_n][:npix].as_numpy_array())
                        p = Nabc_params[i_n]
                        N_grad = p.get_deriv(x[p.xpos], N_grad)
                        J[p.xpos] += N_grad
                        if SIM.D.isotropic_ncells:
                            break

                if Ndef_params[0].refine:
                    Ndef_grads = SIM.D.get_ncells_def_derivative_pixels()
                    for i_n in range(3):
                        N_grad = scale * (Ndef_grads[i_n][:npix].as_numpy_array())
                        p = Ndef_params[i_n]
                        N_grad = p.get_deriv(x[p.xpos], N_grad)
                        J[p.xpos] += N_grad

            # per-image B-factor gradient
            if Bfac_param.refine:
                Bfac_grad = scale * SIM.D.get_Bfactor_derivative_pixels().as_numpy_array()[:npix]
                Bfac_grad = Bfac_param.get_deriv(x[Bfac_param.xpos], Bfac_grad)
                J[Bfac_param.xpos] += Bfac_grad

            # anisotropic B-factor gradients
            if Baniso_params[0].refine:
                Baniso_grads = SIM.D.get_Bfactor_aniso_derivative_pixels()
                for i_ba in range(6):
                    ba_grad = scale * Baniso_grads[i_ba][:npix].as_numpy_array()
                    p = Baniso_params[i_ba]
                    ba_grad = p.get_deriv(x[p.xpos], ba_grad)
                    J[p.xpos] += ba_grad

            if SIM.D.use_diffuse:
                for t in ['gamma','sigma']:
                    diffuse_grads = getattr(SIM.D, "get_diffuse_%s_derivative_pixels" % t)()
                    if diffuse_params_lookup[t][0].refine:
                        for i_diff in range(3):
                            diff_grad = scale*(diffuse_grads[i_diff][:npix].as_numpy_array())
                            p = diffuse_params_lookup[t][i_diff]
                            diff_grad = p.get_deriv(x[p.xpos], diff_grad)
                            J[p.xpos] += diff_grad

            if eta_params[0].refine:
                if SIM.D.has_anisotropic_mosaic_spread:
                    eta_derivs = SIM.D.get_aniso_eta_deriv_pixels()
                else:
                    eta_derivs = [SIM.D.get_derivative_pixels(ETA_ID)]
                num_eta = 3 if SIM.D.has_anisotropic_mosaic_spread else 1
                for i_eta in range(num_eta):
                    p = eta_params[i_eta]
                    eta_grad = scale * (eta_derivs[i_eta][:npix].as_numpy_array())
                    eta_grad = p.get_deriv(x[p.xpos], eta_grad)
                    J[p.xpos] += eta_grad

            if ucell_params[0].refine:
                for i_ucell in range(nucell):
                    p = ucell_params[i_ucell]
                    deriv = scale*SIM.D.get_derivative_pixels(UCELL_ID_OFFSET+i_ucell).as_numpy_array()[:npix]
                    deriv = p.get_deriv(x[p.xpos], deriv)
                    J[p.xpos] += deriv

            if DetZ.refine:
                # Note: do I need to scale the derivs here ? I think so ...
                d = SIM.D.get_derivative_pixels(DETZ_ID).as_numpy_array()[:npix]
                d = DetZ.get_deriv(x[DetZ.xpos], d)
                J[DetZ.xpos] += d

            if GonioAng.refine:
                # Note: do I need to scale the derivs here ? I think so ...
                d = scale*SIM.D.get_gonio_angle_derivative_pixels()[0].as_numpy_array()[:npix]
                d = GonioAng.get_deriv(x[GonioAng.xpos], d)
                J[GonioAng.xpos] += d

            if Mod.P["lambda_offset"].refine:
                lambda_derivs = SIM.D.get_lambda_derivative_pixels()
                lambda_param_names = "lambda_offset", "lambda_scale"
                for d,name in zip(lambda_derivs, lambda_param_names):
                    p = Mod.P[name]
                    # Note: do I need to scale the derivs here ? I think so ...
                    d = scale*d.as_numpy_array()[:npix]
                    d = p.get_deriv(x[p.xpos], d)
                    J[p.xpos] += d

    if (not Mod.params.fix.perRoiScale or Mod.params.use_perRoiScale) and compute_grad:
        if compute_grad:
            for p in perRoiParams:
                roi_id = int(p.name.split("scale_roi")[1])
                slc = Mod.roi_id_slices[roi_id][0]
                if dont_rescale_gradient:
                    d = model_pix_noRoi[slc]
                else:
                    d = p.get_deriv(x[p.xpos], model_pix_noRoi[slc])
                J[p.xpos, slc] += d

    if compute_grad and Mod.P.refining_detector:
        det_Jac = detector_model_derivs(Mod, Mod.P, SIM, x, reduce=False, scale=scale)
        for pname in det_Jac:
            p = Mod.P[pname]
            J[p.xpos] = det_Jac[pname]

    return model_pix, J


def look_at_x(x, Mod):
    for name, p in Mod.P.items():
        if name.startswith("scale_roi") and not p.refine:
            continue
        if name.startswith("Fhkl_"):
            continue
        val = p.get_val(x[p.xpos])
        print("%s: %f" % (name, val))


def get_param_from_x(x, Mod, i_xtal=0, as_dict=False):
    G = Mod.P['G_xtal%d' %i_xtal]
    scale = G.get_val(x[G.xpos])

    RotXYZ = [Mod.P["RotXYZ%d_xtal%d" % (i, i_xtal)] for i in range(3)]
    rotX, rotY, rotZ = [r.get_val(x[r.xpos]) for r in RotXYZ]

    use_cholesky = getattr(Mod, 'use_cholesky_Nabc', False)
    if use_cholesky and "chol_L11" in Mod.P:
        chol_names = ["chol_L11", "chol_L21", "chol_L22", "chol_L31", "chol_L32", "chol_L33"]
        chol_vals = [Mod.P[n].get_val(x[Mod.P[n].xpos]) for n in chol_names]
        L11, L21, L22, L31, L32, L33 = chol_vals
        Na = L11*L11
        Nb = L21*L21 + L22*L22
        Nc = L31*L31 + L32*L32 + L33*L33
        Nd = L11*L21
        Ne = L21*L31 + L22*L32
        Nf = L11*L31
    else:
        Nabc = [Mod.P["Nabc%d" % (i, )] for i in range(3)]
        Na, Nb, Nc = [p.get_val(x[p.xpos]) for p in Nabc]

        Ndef = [Mod.P["Ndef%d" % (i, )] for i in range(3)]
        Nd, Ne, Nf = [p.get_val(x[p.xpos]) for p in Ndef]

    diff_gam_abc = [Mod.P["diffuse_gamma%d" % i] for i in range(3)]
    diff_gam_a, diff_gam_b, diff_gam_c = [p.get_val(x[p.xpos]) for p in diff_gam_abc]

    diff_sig_abc = [Mod.P["diffuse_sigma%d" % i] for i in range(3)]
    diff_sig_a, diff_sig_b, diff_sig_c = [p.get_val(x[p.xpos]) for p in diff_sig_abc]

    nucell = len(Mod.ucell_man.variables)
    ucell_p = [Mod.P["ucell%d" % i] for i in range(nucell)]
    ucell_var = [p.get_val(x[p.xpos]) for p in ucell_p]
    Mod.ucell_man.variables = ucell_var
    a,b,c,al,be,ga = Mod.ucell_man.unit_cell_parameters

    DetZ = Mod.P["detz_shift"]
    detz = DetZ.get_val(x[DetZ.xpos])

    GonioAng = Mod.P["gonio_angle"]
    gonio_angle = GonioAng.get_val(x[GonioAng.xpos])

    Bfac_p = Mod.P["Bfactor"]
    Bfactor = Bfac_p.get_val(x[Bfac_p.xpos])

    if as_dict:
        vals = scale, rotX, rotY, rotZ, Na, Nb, Nc, Nd, Ne, Nf, diff_gam_a, diff_gam_b, diff_gam_c, diff_sig_a, diff_sig_b, diff_sig_c, a,b,c,al,be,ga, detz, gonio_angle, Bfactor
        keys = 'scale', 'rotX', 'rotY', 'rotZ', 'Na', 'Nb', 'Nc', 'Nd', 'Ne', 'Nf', 'diff_gam_a', 'diff_gam_b', 'diff_gam_c', 'diff_sig_a', 'diff_sig_b', 'diff_sig_c', 'a','b','c','al','be','ga', 'detz', 'gonio_angle', 'Bfactor'
        param_dict = dict(zip(keys, vals))
        if use_cholesky and "chol_L11" in Mod.P:
            param_dict['cholesky'] = chol_vals
        return param_dict
    else:
        return scale, rotX, rotY, rotZ, Na, Nb, Nc, Nd, Ne, Nf, diff_gam_a, diff_gam_b, diff_gam_c, diff_sig_a, diff_sig_b, diff_sig_c, a,b,c,al,be,ga, detz, gonio_angle, Bfactor


class TargetFunc:
    def __init__(self, SIM, niter_per_J=1, profile=False):
        self.t_per_iter = []
        self.niter_per_J = niter_per_J
        self.prev_iter_vals = None
        self.global_x = []
        self.percent_change_of_converged = 0.1
        self.all_x = []
        self.terminate_after_n_converged_iterations = None
        self.vary = None #boolean numpy array specifying which params to refine
        self.x0 = None  # 1d array of parameters (should be numpy array, same length as vary)
        self.old_J = None
        self.old_model = None
        self.delta_x = None
        self.iteration = 0
        self.minima = []
        self.all_converged_params = 0
        self.SIM = SIM
        self.all_f = []  # store the target functionals here, 1 per iteration
        self.all_sigZ = []  # store the overall z-score sigmas here, 1 per iteration
        self.all_hop_id = []
        self.hop_iter = 0
        self.lowest_x = None
        self.lowest_f = np.inf

    def at_minimum(self, x, f, accept):
        self.iteration = 0
        self.all_x = []
        self.x0[self.vary] = x
        #look_at_x(self.x0,self)
        self.hop_iter += 1
        self.minima.append((f,self.x0,accept))
        self.lowest_x = x

    def jac(self, x, *args):
        if self.g is not None:
            return self.g[self.vary]

    def __call__(self, x, *args, **kwargs):
        self.x0[self.vary] = x
        if self.all_x:
            self.delta_x = self.x0 - self.all_x[-1]
        update_terms = None
        if not self.iteration % (self.niter_per_J) == 0:
            update_terms = (self.delta_x, self.old_J, self.old_model)
        self.all_x.append(self.x0)

        mod, SIM, compute_grad = args
        f, g, modelpix, J, sigZ, debug_s, zscore_perpix = target_func(self.x0, update_terms, mod, SIM, compute_grad,
                                                                      return_all_zscores=True)
        mod.all_zscore = zscore_perpix

        # filter during refinement?
        if mod.params.filter_during_refinement.enable and self.iteration > 0:
            if self.iteration % mod.params.filter_during_refinement.after_n == 0:
                mod.filter_pixels(thresh=mod.params.filter_during_refinement.threshold)


        self.t_per_iter.append(time.time())
        if len(self.t_per_iter) > 2:
            ave_t_per_it = np.mean([t2-t1 for t2,t1 in zip(self.t_per_iter[1:], self.t_per_iter[:-1])])
        else:
            ave_t_per_it = 0

        debug_s = "Hop=%d |it=%d | t/it=%.4fs" % (self.hop_iter, self.iteration, ave_t_per_it) + debug_s
        MAIN_LOGGER.debug(debug_s)
        self.all_f.append(f)
        self.all_sigZ.append(sigZ)
        self.all_hop_id.append(self.hop_iter)
        self.old_model = modelpix
        self.old_J = J
        self.iteration += 1
        self.g = g
        return f


def target_func(x, udpate_terms, mod, SIM, compute_grad=True, return_all_zscores=False):
    pfs = mod.pan_fast_slow
    data = mod.all_data
    sigma_rdout = mod.all_sigma_rdout
    trusted = mod.all_trusted
    background = mod.all_background
    params = mod.params
    use_cholesky = getattr(mod, 'use_cholesky_Nabc', False)
    if udpate_terms is not None:
        # if approximating the gradients, then fix the parameter refinment managers in diffBragg
        # so we dont waste time computing them
        _compute_grad = False
        SIM.D.fix(NCELLS_ID)
        SIM.D.fix(NCELLS_ID_OFFDIAG)
        for db_id in PAN_OFS_IDS + PAN_XYZ_IDS:
            SIM.D.fix(db_id)
        SIM.D.fix(ROTX_ID)
        SIM.D.fix(ROTY_ID)
        SIM.D.fix(ROTZ_ID)
        for lam_id in LAMBDA_IDS:
            SIM.D.fix(lam_id)
        for i_ucell in range(len(mod.ucell_man.variables)):
            SIM.D.fix(UCELL_ID_OFFSET + i_ucell)
        SIM.D.fix(DETZ_ID)
        SIM.D.fix(ETA_ID)
        SIM.D.fix(DIFFUSE_ID)
        SIM.D.fix(GONIO_ANGLE_ID)
        SIM.D.fix(BFACTOR_ID)
        SIM.D.fix(BFACTOR_ANISO_ID)
    elif compute_grad:
        # actually compute the gradients
        _compute_grad = True
        if use_cholesky and "chol_L11" in mod.P and mod.P["chol_L11"].refine:
            # Cholesky mode: need both diagonal and off-diagonal kernel gradients
            SIM.D.let_loose(NCELLS_ID)
            SIM.D.let_loose(NCELLS_ID_OFFDIAG)
        else:
            if mod.P["Nabc0"].refine:
                SIM.D.let_loose(NCELLS_ID)
            if mod.P["Ndef0"].refine:
                SIM.D.let_loose(NCELLS_ID_OFFDIAG)
        for db_id, name in zip(PAN_OFS_IDS + PAN_XYZ_IDS, ["RotOrth", "RotFast", "RotSlow", "ShiftX", "ShiftY", "ShiftZ"]):
            pname = f"group0_{name}"
            if pname in mod.P and mod.P[pname].refine:
                SIM.D.let_loose(db_id)
        if mod.P["RotXYZ0_xtal0"].refine:
            SIM.D.let_loose(ROTX_ID)
            SIM.D.let_loose(ROTY_ID)
            SIM.D.let_loose(ROTZ_ID)
        if mod.P["ucell0"].refine:
            for i_ucell in range(len(mod.ucell_man.variables)):
                SIM.D.let_loose(UCELL_ID_OFFSET + i_ucell)
        if mod.P["detz_shift"].refine:
            SIM.D.let_loose(DETZ_ID)
        if mod.P["gonio_angle"].refine:
            SIM.D.let_loose(GONIO_ANGLE_ID)
        if mod.P["eta_abc0"].refine:
            SIM.D.let_loose(ETA_ID)
        if mod.P["lambda_offset"].refine:
            for lam_id in LAMBDA_IDS:
                SIM.D.let_loose(lam_id)
        if mod.P["Bfactor"].refine:
            SIM.D.let_loose(BFACTOR_ID)
        if mod.P["Baniso0"].refine:
            SIM.D.let_loose(BFACTOR_ANISO_ID)
    else:
        _compute_grad = False
    model_bragg, Jac = model(x, mod, SIM, compute_grad=_compute_grad)

    if udpate_terms is not None:
        # try a Broyden update ?
        # https://people.duke.edu/~hpgavin/ce281/lm.pdf  equation 19
        delta_x, prev_J, prev_model_bragg = udpate_terms
        if prev_J is not None:
            delta_y = model_bragg - prev_model_bragg

            delta_J = (delta_y - np.dot(prev_J.T, delta_x))
            delta_J /= np.dot(delta_x,delta_x)
            Jac = prev_J + delta_J
    # Jac has shape of num_param x num_pix

    model_pix = model_bragg + background

    if SIM.use_psf:
        model_pix, J = convolve_model_with_psf(model_pix, Jac, mod, SIM)

    resid = data - model_pix

    # data contributions to target function
    V_model = model_pix + sigma_rdout**2
    use_fixed_V = params.correct_Fhkl_gradient_bias and SIM.refining_Fhkl and hasattr(mod, 'V_fixed')
    V = mod.V_fixed if use_fixed_V else V_model
    # TODO:what if V is allowed to be negative? The logarithm/sqrt will explore below
    # TODO ignore overflow encountered here ?
    resid_square = resid**2
    if use_fixed_V:
        # Pure weighted chi-squared with FIXED variance from initial model.
        # Fixed V eliminates both the log(V) bias and the resid^2/V^2 bias
        # that arise when V depends on the parameters being optimized.
        fLogLike = (.5 * resid_square / V)
    else:
        fLogLike = (.5*(np.log(2*np.pi*V) + resid_square / V))
    if params.roi.allow_overlapping_spots:
        fLogLike /= mod.all_freq
    fLogLike = fLogLike[trusted].sum()   # negative log Likelihood target

    # width of z-score should decrease as refinement proceeds
    zscore_per = resid/np.sqrt(V_model)
    zscore_sigma = np.std(zscore_per[trusted])

    restraint_terms = {}
    if params.use_restraints:
        restraint_s = ""
        for name in mod.non_fhkl_params:
            p = mod.P[name]
            if p.beta is not None:
                val = p.get_restraint_val(x[p.xpos])
                restraint_terms[name] = val
                restraint_s += "%s " % name
        if restraint_s != "":
            MAIN_LOGGER.debug("Restrained vars (non-Fhkl related): %s"%restraint_s )

        if params.centers.Nvol is not None:
            na,nb,nc = SIM.D.Ncells_abc_aniso
            nd,ne,nf = SIM.D.Ncells_def
            Nmat = [na, nd, nf,
                    nd, nb, ne,
                    nf, ne, nc]
            Nmat = np.reshape(Nmat, (3,3))
            Nvol = np.linalg.det(Nmat)
            del_Nvol = params.centers.Nvol - Nvol
            fN_vol = .5*del_Nvol**2/params.betas.Nvol
            restraint_terms["Nvol"] = fN_vol
    if params.betas.Fhkl is not None:  # (experimental)
        fhkl_grad_channels = {}
        for i_chan in range(SIM.num_Fhkl_channels):
            fhkl_restraint_f, fhkl_restraint_grad = SIM.D.Fhkl_restraint_data(i_chan, params.betas.Fhkl, params.use_geometric_mean_Fhkl)
            fhkl_grad_channels[i_chan] = fhkl_restraint_grad
            restraint_terms["Fhkl_chan%d"% i_chan] = fhkl_restraint_f

    # Fhkl restraint toward targets (fixed or resolution-bin averages)
    snr_restraint_delta = None
    if SIM.refining_Fhkl and hasattr(SIM, 'Fhkl_restraint_weights') and SIM.Fhkl_restraint_weights is not None:
        current_scales = SIM.Fhkl_scales
        if hasattr(SIM, 'Fhkl_restraint_targets') and SIM.Fhkl_restraint_targets is not None:
            targets = SIM.Fhkl_restraint_targets
        else:
            bin_avg = np.zeros(SIM.Fhkl_reso_n_bins)
            for b in range(SIM.Fhkl_reso_n_bins):
                in_bin = SIM.Fhkl_reso_bins == b
                if in_bin.any():
                    bin_avg[b] = current_scales[in_bin].mean()
            targets = bin_avg[SIM.Fhkl_reso_bins]
        snr_restraint_delta = current_scales - targets
        restraint_terms["Fhkl_snr"] = 0.5 * np.sum(SIM.Fhkl_restraint_weights * snr_restraint_delta**2)

    # Wilson prior on Fhkl intensities: -log P_Wilson(I) where I = scale * |F_mtz|^2
    wilson_grad = None
    if SIM.refining_Fhkl and hasattr(SIM, 'wilson_sigma') and SIM.wilson_sigma is not None:
        I_current = SIM.Fhkl_scales * SIM.wilson_F_sq  # current intensity per HKL
        is_cent = SIM.is_centric_full
        is_acent = ~is_cent
        sigma = SIM.wilson_sigma
        # Acentric: -log P = I/Sigma
        wilson_f = np.zeros(len(I_current))
        wilson_f[is_acent] = I_current[is_acent] / sigma[is_acent]
        # Centric: -log P = I/(2*Sigma) - 0.5*log(I)  (only for I > 0)
        cent_pos = is_cent & (I_current > 1e-30)
        wilson_f[cent_pos] = (I_current[cent_pos] / (2 * sigma[cent_pos])
                              - 0.5 * np.log(I_current[cent_pos]))
        restraint_terms["Wilson"] = wilson_f.sum()
        # Gradient w.r.t. scale: d/ds [-log P]
        wilson_grad = np.zeros(len(I_current))
        wilson_grad[is_acent] = SIM.wilson_F_sq[is_acent] / sigma[is_acent]
        wilson_grad[cent_pos] = (SIM.wilson_F_sq[cent_pos] / (2 * sigma[cent_pos])
                                 - 0.5 / SIM.Fhkl_scales[cent_pos])

#   accumulate target function
    f_restraints = 0
    if restraint_terms:
        f_restraints = np.sum(list(restraint_terms.values()))
    f = f_restraints + fLogLike

    restraint_debug_s = "LogLike: %.1f%%; " % (fLogLike / f *100.)
    for name, val in restraint_terms.items():
        if val > 0:
            frac_total = val / f *100.
            restraint_debug_s += "%s: %.2f%%; " % (name, frac_total)

    # fractions of the target function
    g = None  # gradient vector
    gnorm = -1  # norm of gradient vector
    if compute_grad:
        if use_fixed_V:
            # Exact gradient of 0.5*resid^2/V_fixed w.r.t. model m (V_fixed is constant):
            # df/dm = -resid/V_fixed
            common_grad_term_all = -resid / V
        else:
            common_grad_term_all = (0.5 /V * (1-2*resid - resid_square / V))
        if params.roi.allow_overlapping_spots:
            common_grad_term_all /= mod.all_freq
        common_grad_term = common_grad_term_all[trusted]

        g = np.zeros(Jac.shape[0])
        for name in mod.non_fhkl_params:
            p = mod.P[name]
            Jac_p = Jac[p.xpos]
            g[p.xpos] += (Jac_p[trusted] * common_grad_term).sum()

        if not params.fix.perRoiScale:
            for name in mod.scale_roi_names:
                p = mod.P[name]
                Jac_p = Jac[p.xpos]
                roi_id = int(p.name.split("scale_roi")[1])
                slc = mod.roi_id_slices[roi_id][0]
                common_grad_slc = common_grad_term_all[slc]
                Jac_slc = Jac_p[slc]
                trusted_slc = trusted[slc]
                g_accum = (Jac_slc * common_grad_slc)[trusted_slc].sum()
                g[p.xpos] += g_accum

        # trusted pixels portion of Jacobian
        #  TODO: determine if this following method of summing over g is optimal in certain scenarios
        #Jac_t = Jac[:,trusted]
        # gradient vector
        #g = np.array([np.sum(common_grad_term*Jac_t[param_idx]) for param_idx in range(Jac_t.shape[0])])

        if params.use_restraints:
            # update gradients according to restraints
            for name in mod.non_fhkl_params:
                p = mod.P[name]
                if p.beta is not None:
                    g[p.xpos] += p.get_restraint_deriv(x[p.xpos])

            if not params.fix.perRoiScale:  # deprecated ?
                for name in mod.scale_roi_names:
                    p = mod.P[name]
                    if p.beta is not None:
                        g[p.xpos] += p.get_restraint_deriv(x[p.xpos])

            if params.betas.Nvol is not None:
                Nmat_inv = np.linalg.inv(Nmat)
                dVol_dN_vals = []
                for i_N in range(6):
                    if i_N ==0 :
                        dN = [1,0,0,
                              0,0,0,
                              0,0,0]
                    elif i_N == 1:
                        dN = [0,0,0,
                              0,1,0,
                              0,0,0]
                    elif i_N == 2:
                        dN = [0,0,0,
                              0,0,0,
                              0,0,1]
                    elif i_N == 3:
                        dN = [0,1,0,
                              1,0,0,
                              0,0,0]
                    elif i_N == 4:
                        dN = [0,0,0,
                              0,0,1,
                              0,1,0]
                    else:
                        dN = [0,0,1,
                              0,0,0,
                              1,0,0]
                    if i_N < 3:
                        p = mod.P["Nabc%d" % i_N]
                    else:
                        p = mod.P["Ndef%d" % (i_N-3)]
                    dN = np.reshape(dN, (3,3))
                    dVol_dN = Nvol * np.trace(np.dot(Nmat_inv, dN))
                    dVol_dN_vals.append( dVol_dN)
                    gterm = -del_Nvol / params.betas.Nvol * dVol_dN
                    g[p.xpos] += p.get_deriv(x[p.xpos], gterm)

        if SIM.refining_Fhkl:
            spot_scale_p = mod.P["G_xtal0"]
            G = spot_scale_p.get_val(x[spot_scale_p.xpos])
            fhkl_grad = SIM.D.add_Fhkl_gradients(pfs, resid, V, trusted,
                                                 mod.all_freq, SIM.num_Fhkl_channels, G)

            if params.correct_Fhkl_gradient_bias:
                # The C++ kernel gradient includes a log(V) bias term (the "1" in Gterm=1-2u-u^2/V).
                # Subtract it by computing the gradient at zero residual (where only the bias survives).
                fhkl_bias = SIM.D.add_Fhkl_gradients(pfs, np.zeros_like(resid), V, trusted,
                                                      mod.all_freq, SIM.num_Fhkl_channels, G)
                fhkl_grad -= fhkl_bias

            if params.betas.Fhkl is not None:
                for i_chan in range(SIM.num_Fhkl_channels):
                    restraint_contribution_to_grad = fhkl_grad_channels[i_chan]
                    fhkl_slice = slice(i_chan*SIM.Num_ASU, (i_chan+1)*SIM.Num_ASU, 1)
                    np.add.at(fhkl_grad, fhkl_slice, restraint_contribution_to_grad)

            # SNR-adaptive restraint gradient: d/ds [0.5 * w * (s - target)^2] = w * (s - target)
            if snr_restraint_delta is not None:
                fhkl_grad += SIM.Fhkl_restraint_weights * snr_restraint_delta

            # Wilson prior gradient (in scale-space, before chain rule)
            if wilson_grad is not None:
                fhkl_grad += wilson_grad

            if SIM.Fhkl_linear:
                fhkl_grad *= 2 * SIM.Fhkl_amplitudes * SIM.Fhkl_sigmas
            else:
                fhkl_grad *= SIM.Fhkl_scales * SIM.Fhkl_sigmas

            g = np.append(g, fhkl_grad)


        gnorm = np.linalg.norm(g)

    debug_s = "F=%10.7g sigZ=%10.7g (Fracs of F: %s), |g|=%10.7g" \
              % (f, zscore_sigma, restraint_debug_s, gnorm)

    return_data = f, g, model_bragg, Jac, zscore_sigma, debug_s
    if return_all_zscores:
        return_data += (zscore_per,)
    return return_data


def refine(exp, ref, params, spec=None, gpu_device=None, return_modeler=False, best=None, free_mem=True,
           diffBragg_intensity=False):
    if gpu_device is None:
        gpu_device = 0
    params.simulator.spectrum.filename = spec
    Modeler = DataModeler(params)
    if params.load_data_from_refls:
        Modeler.GatherFromReflectionTable(exp, ref, sg_symbol=params.space_group)
    else:
        assert Modeler.GatherFromExperiment(exp, ref, sg_symbol=params.space_group)

    SIM = get_simulator_for_data_modelers(Modeler)
    Modeler.set_parameters_for_experiment(best=best)
    SIM.D.device_Id = gpu_device

    # Set refinement flags before any GPU kernel call (auto-sigma, bias-correct)
    # so the GPU allocates memory for all needed gradients upfront
    Modeler.set_diffBragg_refinement_flags(SIM)

    nparam = len(Modeler.P)
    if SIM.refining_Fhkl:
        nparam += SIM.Num_ASU*SIM.num_Fhkl_channels
    x0 = [1] * nparam

    if SIM.refining_Fhkl and params.correct_Fhkl_gradient_bias:
        # Precompute fixed variance from initial model for unbiased WLS objective.
        # Using V_fixed (constant w.r.t. parameters) eliminates the resid^2/V^2 bias
        # that arises when V = model + sigma^2 depends on the parameters being optimized.
        model_init, _ = model(np.ones(nparam), Modeler, SIM, compute_grad=False)
        Modeler.V_fixed = model_init + Modeler.all_background + Modeler.all_sigma_rdout**2

    if SIM.refining_Fhkl:
        print("Fhkl parameterization: %s" % ("linear" if SIM.Fhkl_linear else "exponential"))

    # Check if any non-Fhkl parameters are being refined (G, Nabc, etc.)
    has_free_non_fhkl = any(
        p.refine for name, p in Modeler.P.items()
        if not name.startswith("scale_roi") and not name.startswith("Fhkl_"))

    if SIM.refining_Fhkl and params.auto_Fhkl_sigma:
        if has_free_non_fhkl:
            # Two-pass strategy: when G and Fhkl are jointly refined (ill-conditioned
            # due to G*F^2 degeneracy), first converge with uniform sigma (which L-BFGS-B
            # handles well), then compute auto-sigma at the converged point and do a
            # fine-tuning pass. Computing auto-sigma before convergence gives wrong
            # curvatures because the operating point (G, Fhkl) is far from the solution.
            print("Auto-sigma pass 1: uniform sigma, converging G+Fhkl jointly...")
            x_pass1 = Modeler.Minimize(x0, SIM)

            print("Computing per-Fhkl sigmas from curvatures at converged operating point...")
            _compute_fhkl_sigmas_from_curvatures(Modeler, SIM, x0=x_pass1)

            # Remap Fhkl x values to preserve actual scale factors under new sigma.
            # scale = init * exp(sigma * (x-1)), so to keep the same scale:
            #   x_new = 1 + (old_sigma / new_sigma) * (x_old - 1)
            nscales = SIM.Num_ASU * SIM.num_Fhkl_channels
            old_sigma = float(params.sigmas.Fhkl)
            x_pass1 = np.array(x_pass1, dtype=np.float64)
            x_fhkl = x_pass1[-nscales:]
            new_sigmas = SIM.Fhkl_sigmas
            has_signal = new_sigmas > 0
            x_fhkl[has_signal] = 1.0 + (old_sigma / new_sigmas[has_signal]) * (x_fhkl[has_signal] - 1.0)
            x_pass1[-nscales:] = x_fhkl
            print("Remapped %d Fhkl x-values to preserve scale factors under new sigma" % has_signal.sum())

            # Recompute V_fixed at converged operating point if bias correction is on
            if params.correct_Fhkl_gradient_bias:
                model_pass1, _ = model(x_pass1, Modeler, SIM, compute_grad=False)
                Modeler.V_fixed = model_pass1 + Modeler.all_background + Modeler.all_sigma_rdout**2

            print("Auto-sigma pass 2: per-Fhkl sigma, fine-tuning...")
            x0 = list(x_pass1)
        else:
            print("Computing per-Fhkl sigmas from curvatures...")
            _compute_fhkl_sigmas_from_curvatures(Modeler, SIM)
    else:
        if SIM.refining_Fhkl:
            print("Using uniform Fhkl sigma=%.4g (auto_Fhkl_sigma=%s)" % (params.sigmas.Fhkl, params.auto_Fhkl_sigma))

    if SIM.refining_Fhkl and params.Fhkl_wilson_prior:
        _setup_wilson_prior(Modeler, SIM)

    if SIM.refining_Fhkl and params.Fhkl_two_stage_threshold is not None:
        x = _two_stage_fhkl_refine(Modeler, SIM, params, x0)
    else:
        if SIM.refining_Fhkl and params.Fhkl_restraint_snr_sigma is not None:
            _setup_fhkl_snr_restraint(Modeler, SIM)
        x = Modeler.Minimize(x0, SIM)
    compute_grad = params.method=="L-BFGS-B"
    Modeler.best_model, best_Jac = model(x, Modeler, SIM, compute_grad=compute_grad, dont_rescale_gradient=True)
    Modeler.best_model_includes_background = False
    try:
        sigz, niter, _ = Modeler.get_best_hop()
    except Exception:
        sigz = niter = None
        pass

    new_crystal = update_crystal_from_x(Modeler, SIM, x)
    new_exp = deepcopy(Modeler.E)
    new_exp.crystal = new_crystal

    try:
        new_exp.beam.set_wavelength(SIM.dxtbx_spec.get_weighted_wavelength())
    except Exception: pass
    # if we strip the thickness from the detector, then update it here:
    #new_exp.detector. shift Z mm
    new_det = update_detector_from_x(Modeler, SIM, x)
    new_exp.detector = new_det

    if diffBragg_intensity:
        assert best_Jac is not None
        new_refl = get_new_xycalcs(Modeler, new_exp, x=x, SIM=SIM, Jac=best_Jac)
    else:
        new_refl = get_new_xycalcs(Modeler, new_exp, x=x)

    if niter is not None:
        new_refl["shot_niter"] = flex.double(len(new_refl), niter)
        new_refl["shot_sigz"] = flex.double(len(new_refl), sigz)

    if free_mem:
        Modeler.clean_up(SIM)

    if return_modeler:
        return new_exp, new_refl, Modeler, SIM, x

    else:
        return new_exp, new_refl


def update_detector_from_x(Mod, SIM, x):
    scale, rotX, rotY, rotZ, Na, Nb, Nc, _,_,_,_,_,_,_,_,_,a, b, c, al, be, ga, detz_shift, _,_ = get_param_from_x(x, Mod)
    detz_shift_mm = detz_shift*1e3
    det = SIM.detector
    det = utils.shift_panelZ(det, detz_shift_mm)
    return det


def new_cryst_from_rotXYZ_and_ucell(rotXYZ, ucparam, orig_crystal):
    """

    :param rotXYZ: tuple of rotation angles about princple axes (radians)
        Crystal will be rotated by rotX about (-1,0,0 ), rotY about (0,-1,0) and rotZ about
        (0,0,-1) . This is the convention used by diffBragg
    :param ucparam: unit cell parameter tuple (a,b,c, alpha, beta, gamma) in Angstrom,degrees)
    :param orig_crystal: dxtbx.model.Crystal object which will be copied and adjusted according to ucell and rotXYZ
    :return: new dxtbx.model.Crystal object
    """
    rotX, rotY, rotZ = rotXYZ
    xax = col((-1, 0, 0))
    yax = col((0, -1, 0))
    zax = col((0, 0, -1))
    ## update parameters:
    RX = xax.axis_and_angle_as_r3_rotation_matrix(rotX, deg=False)
    RY = yax.axis_and_angle_as_r3_rotation_matrix(rotY, deg=False)
    RZ = zax.axis_and_angle_as_r3_rotation_matrix(rotZ, deg=False)
    M = RX * RY * RZ
    U = M * sqr(orig_crystal.get_U())
    new_C = deepcopy(orig_crystal)
    new_C.set_U(U)

    ucman = utils.manager_from_params(ucparam)
    new_C.set_B(ucman.B_recipspace)
    return new_C


def update_crystal_from_x(Mod, SIM, x):
    """
    :param SIM: sim_data instance containing a nanoBragg crystal object
    :param x: parameters returned by hopper_utils (instance of simtbx.diffBragg.refiners.parameters.Parameters()
    :return: a new dxtbx.model.Crystal object with updated unit cell and orientation matrix
    """
    scale, rotX, rotY, rotZ, Na, Nb, Nc, _,_,_,_,_,_,_,_,_,a, b, c, al, be, ga, detz_shift, _,_ = get_param_from_x(x, Mod)
    ucparam = a, b, c, al, be, ga
    return new_cryst_from_rotXYZ_and_ucell((rotX,rotY,rotZ), ucparam, SIM.crystal.dxtbx_crystal)


def get_new_xycalcs(Modeler, new_exp, old_refl_tag="dials", x=None, Jac=None, SIM=None):
    """

    :param Modeler: data modeler instance after refinement
    :param new_exp: post-refinement dxtbx experiment obj
    :param old_refl_tag: existing columns will be renamed with this tag as a prefix
    :param x: refinement parameter values *scaled
    :param Jac: the modeler Jacobian
    :param SIM: sim_data.SimData instance used by the refiner
    :return: refl table with xyzcalcs derived from the modeling results
    """
    if isinstance(Modeler.all_sigma_rdout, np.ndarray):
        data_subimg, model_subimg, trust_subimg, bragg_subimg, _ = Modeler.get_data_model_pairs()
    else:
        data_subimg, model_subimg, trust_subimg, bragg_subimg = Modeler.get_data_model_pairs()
    new_refls = deepcopy(Modeler.refls)

    reflkeys = list(new_refls.keys())
    if "xyzcal.px" in reflkeys:
        new_refls['%s.xyzcal.px' % old_refl_tag] = deepcopy(new_refls['xyzcal.px'])
    if "xyzcal.mm" in reflkeys:
        new_refls['%s.xyzcal.mm' % old_refl_tag] = deepcopy(new_refls['xyzcal.mm'])
    if "xyzobs.mm.value" in list(new_refls.keys()):
        new_refls['%s.xyzobs.mm.value' % old_refl_tag] = deepcopy(new_refls['xyzobs.mm.value'])

    new_xycalcs = flex.vec3_double(len(Modeler.refls), (np.nan, np.nan, np.nan))
    new_xycalcs_mm = flex.vec3_double(len(Modeler.refls), (np.nan, np.nan, np.nan))
    new_xyobs_mm = flex.vec3_double(len(Modeler.refls), (np.nan, np.nan, np.nan))
    per_refl_scales = flex.double(len(Modeler.refls), 1)
    # for storing diffBragg intensity and variance
    flex_varI = flex.double(len(Modeler.refls),0)
    flex_I = flex.double(len(Modeler.refls),0)
    flex_sigZ = flex.double(len(Modeler.refls), 1)
    flex_meanZ = flex.double(len(Modeler.refls), 0)
    flex_ccModel = flex.double(len(Modeler.refls),0)
    # for filtering reflections based on the diffBragg intensity/variance
    sel2 = flex.bool(len(Modeler.refls), True)

    variance_s = Fp1_map = None
    if Jac is not None:
        variance_s = get_variance_s(Modeler, Jac)
    if SIM is not None:
        Fp1 = SIM.D.Fhkl #crystal.miller_array
        Fp1_map = {h:amp for h,amp in zip(Fp1.indices(), Fp1.data())}

    xyzobs_px_val = new_refls["xyzobs.px.value"]
    pids = new_refls["panel"]
    for i_roi in range(len(bragg_subimg)):

        ref_idx = Modeler.refls_idx[i_roi]

        #assert ref_idx==i_roi
        if np.any(bragg_subimg[i_roi] > 0):
            I = bragg_subimg[i_roi]
            assert np.all(I>=0)
            Y, X = np.indices(bragg_subimg[i_roi].shape)
            x1, _, y1, _ = Modeler.rois[i_roi]

            com_x, com_y, _ = xyzobs_px_val[ref_idx]
            com_x = int(com_x - x1 - 0.5)
            com_y = int(com_y - y1 - 0.5)
            # make sure at least some signal is at the centroid! otherwise this is likely a neighboring spot
            try:
                if I[com_y, com_x] == 0:
                    continue
            except IndexError:
                continue

            X += x1
            Y += y1
            Isum = I.sum()
            xcom = (X * I).sum() / Isum + .5
            ycom = (Y * I).sum() / Isum + .5
            com = xcom, ycom, 0

            pid = Modeler.pids[i_roi]
            assert pid == pids[ref_idx]
            panel = new_exp.detector[int(pid)]
            xmm, ymm = panel.pixel_to_millimeter((xcom, ycom))
            com_mm = xmm, ymm, 0

            #xobs, yobs, _ = xyzobs_new_refls[ref_idx]["xyzobs.px.value"]
            xobs_mm, yobs_mm = panel.pixel_to_millimeter((com_x, com_y))
            obs_com_mm = xobs_mm, yobs_mm, 0

            new_xycalcs[ref_idx] = com
            new_xycalcs_mm[ref_idx] = com_mm
            new_xyobs_mm[ref_idx] = obs_com_mm

    if isinstance(new_refls, pandas.DataFrame):
        new_refls["xyzcal.px"] = list(map(tuple, new_xycalcs))
        new_refls["xyzcal.mm"] = list(map(tuple, new_xycalcs_mm))
        new_refls["xyzobs.mm.value"] = list(map(tuple, new_xyobs_mm))
    else:
        new_refls["xyzcal.px"] = new_xycalcs
        new_refls["xyzcal.mm"] = new_xycalcs_mm
        new_refls["xyzobs.mm.value"] = new_xyobs_mm

    if Modeler.params.filter_unpredicted_refls_in_output:
        sel = [not np.isnan(x) for x,_,_ in new_refls['xyzcal.px']]
        nbefore = len(new_refls)
        if isinstance(new_refls, pandas.DataFrame):
            new_refls = new_refls.loc[sel]
        else:
            new_refls = new_refls.select(flex.bool(sel))
        nafter = len(new_refls)
        MAIN_LOGGER.info("Filtered %d / %d reflections which did not show peaks in model" % (nbefore-nafter, nbefore))

    return new_refls


def get_mosaicity_from_x(x, Mod, SIM):
    """
    :param x: refinement parameters
    :param SIM: simulator used during refinement
    :return: float or 3-tuple, depending on whether mosaic spread was modeled isotropically
    """
    eta_params = [Mod.P["eta_abc%d"%i] for i in range(3)]
    eta_abc = [p.get_val(x[p.xpos]) for p in eta_params]
    if not SIM.D.has_anisotropic_mosaic_spread:
        eta_abc = [eta_abc[0]]*3
    return eta_abc


def generate_gauss_spec(central_en=9500, fwhm=10, res=1, nchan=20, total_flux=1e12, as_spectrum=True):
    """
    In cases where raw experimental data do not contain X-ray spectra, one can generate a gaussian spectra
    and use that as part of the diffBragg model.

    :param central_en: nominal beam energy in eV
    :param fwhm: fwhm of desired Gaussian spectrum in eV
    :param res: energy resolution of spectrum in eV
    :param nchan: number of energy channels in spectrum
    :param total_flux: total number of photons across whole spectrum
    :param as_spectrum: return as a spectrum object, a suitable attribute of nanoBragg_beam
    :return: either a nanoBragg_beam.NBbeam spectrum, or a tuple of numpy arrays spec_energies, spec_weights
    """
    sig = fwhm / 2.35482
    min_en = central_en - nchan / 2 * res
    max_en = central_en + nchan / 2 * res
    ens = np.linspace(min_en, max_en, nchan)
    wt = np.exp(-(ens - central_en) ** 2 / 2 / sig ** 2)
    wt /= wt.sum()
    wt *= total_flux
    if as_spectrum:
        wvs = utils.ENERGY_CONV / ens  # wavelengths
        spec = list(zip(list(wvs), list(wt)))
        return spec
    else:
        return ens, wt

def downsamp_spec_from_params(params, expt=None, imgset=None, i_img=0):
    """

    :param params:  hopper phil params extracted
    :param expt: a dxtbx experiment (optional)
    :param imgset: an dxtbx imageset (optional)
    :param i_img: index of the image in the imageset (only matters if imgset is not None)
    :return: dxtbx spectrum with parameters applied
    """
    if expt is not None:
        dxtbx_spec = expt.imageset.get_spectrum(0)
        starting_wave = expt.beam.get_wavelength()
    else:
        assert imgset is not None
        dxtbx_spec = imgset.get_spectrum(i_img)
        starting_wave = imgset.get_beam(i_img).get_wavelength()

    spec_en = dxtbx_spec.get_energies_eV()
    spec_wt = dxtbx_spec.get_weights()
    if params.downsamp_spec.skip:
        spec_wave = utils.ENERGY_CONV / spec_en.as_numpy_array()
        stride=params.simulator.spectrum.stride
        spec_wave = spec_wave[::stride]
        spec_wt = spec_wt[::stride]
        spectrum = list(zip(spec_wave, spec_wt))
    else:
        spec_en = dxtbx_spec.get_energies_eV()
        spec_wt = dxtbx_spec.get_weights()
        # ---- downsample the spectrum
        method2_param = {"filt_freq": params.downsamp_spec.filt_freq,
                         "filt_order": params.downsamp_spec.filt_order,
                         "tail": params.downsamp_spec.tail,
                         "delta_en": params.downsamp_spec.delta_en}
        downsamp_en, downsamp_wt = downsample_spectrum(spec_en.as_numpy_array(),
                                                       spec_wt.as_numpy_array(),
                                                       method=2, method2_param=method2_param)

        stride = params.simulator.spectrum.stride
        if stride > len(downsamp_en) or stride == 0:
            raise ValueError("Incorrect value for pinkstride")
        downsamp_en = downsamp_en[::stride]
        downsamp_wt = downsamp_wt[::stride]
        tot_fl = params.simulator.total_flux
        if tot_fl is not None:
            downsamp_wt = downsamp_wt / sum(downsamp_wt) * tot_fl

        downsamp_wave = utils.ENERGY_CONV / downsamp_en
        spectrum = list(zip(downsamp_wave, downsamp_wt))
    # the nanoBragg beam has an xray_beams property that is used internally in diffBragg
    waves, specs = map(np.array, zip(*spectrum))
    ave_wave = sum(waves*specs) / sum(specs)
    MAIN_LOGGER.debug("Starting wavelength=%f. Spectrum ave wavelength=%f" % (starting_wave, ave_wave))
    if expt is not None:
        expt.beam.set_wavelength(ave_wave)
        MAIN_LOGGER.debug("Shifting expt wavelength from %f to %f" % (starting_wave, ave_wave))
    MAIN_LOGGER.debug("USING %d ENERGY CHANNELS" % len(spectrum))
    return spectrum


# set the X-ray spectra for this shot
def downsamp_spec(SIM, params, expt, return_and_dont_set=False):
    SIM.dxtbx_spec = expt.imageset.get_spectrum(0)
    spec_en = SIM.dxtbx_spec.get_energies_eV()
    spec_wt = SIM.dxtbx_spec.get_weights()
    if params.downsamp_spec.skip:
        spec_wave = utils.ENERGY_CONV / spec_en.as_numpy_array()
        stride = params.simulator.spectrum.stride
        spec_wave = spec_wave[::stride]
        spec_wt = spec_wt[::stride]
        tot_fl = params.simulator.total_flux
        if tot_fl is not None:
            spec_wt = spec_wt / sum(spec_wt) * tot_fl
        SIM.beam.spectrum = list(zip(spec_wave, spec_wt))
    else:
        spec_en = SIM.dxtbx_spec.get_energies_eV()
        spec_wt = SIM.dxtbx_spec.get_weights()
        # ---- downsample the spectrum
        method2_param = {"filt_freq": params.downsamp_spec.filt_freq,
                         "filt_order": params.downsamp_spec.filt_order,
                         "tail": params.downsamp_spec.tail,
                         "delta_en": params.downsamp_spec.delta_en}
        downsamp_en, downsamp_wt = downsample_spectrum(spec_en.as_numpy_array(),
                                                       spec_wt.as_numpy_array(),
                                                       method=2, method2_param=method2_param)

        stride = params.simulator.spectrum.stride
        if stride > len(downsamp_en) or stride == 0:
            raise ValueError("Incorrect value for pinkstride")
        downsamp_en = downsamp_en[::stride]
        downsamp_wt = downsamp_wt[::stride]
        tot_fl = params.simulator.total_flux
        if tot_fl is not None:
            downsamp_wt = downsamp_wt / sum(downsamp_wt) * tot_fl

        downsamp_wave = utils.ENERGY_CONV / downsamp_en
        SIM.beam.spectrum = list(zip(downsamp_wave, downsamp_wt))
    # the nanoBragg beam has an xray_beams property that is used internally in diffBragg
    starting_wave = expt.beam.get_wavelength()
    waves, specs = map(np.array, zip(*SIM.beam.spectrum))
    ave_wave = sum(waves*specs) / sum(specs)
    expt.beam.set_wavelength(ave_wave)
    MAIN_LOGGER.debug("Shifting wavelength from %f to %f" % (starting_wave, ave_wave))
    MAIN_LOGGER.debug("Using %d energy channels" % len(SIM.beam.spectrum))
    if return_and_dont_set:
        return SIM.beam.spectrum
    else:
        SIM.D.xray_beams = SIM.beam.xray_beams


def set_gauss_spec(SIM=None, params=None, E=None):
    """
    Generates a gaussian spectrum for the experiment E and attach it to SIM
    in a way that diffBragg understands

    This updates the xray_beams property of SIM.D and the spectrum property
    of SIM.beam

    :param SIM: simulator object, instance of nanoBragg/sim_data
    :param params: phil params , expected scope is defined in diffBragg/phil.py
    :param E: dxtbx Experiment
    :return: None
    """
    if SIM is not None and not hasattr(SIM, "D"):
        raise AttributeError("Cannot set the spectrum until diffBragg has been instantiated!")
    spec_mu = utils.ENERGY_CONV / E.beam.get_wavelength()
    spec_res = params.simulator.spectrum.gauss_spec.res
    spec_nchan = params.simulator.spectrum.gauss_spec.nchannels
    spec_fwhm = params.simulator.spectrum.gauss_spec.fwhm
    spec_ens, spec_wts = generate_gauss_spec(central_en=spec_mu, fwhm=spec_fwhm, res=spec_res,
                                             nchan=spec_nchan, as_spectrum=False)
    spec_wvs = utils.ENERGY_CONV / spec_ens
    nanoBragg_beam_spec = list(zip(spec_wvs, spec_wts))
    if SIM is not None:
        SIM.dxtbx_spec = Spectrum(spec_ens, spec_wts)
        SIM.beam.spectrum = nanoBragg_beam_spec
        SIM.D.xray_beams = SIM.beam.xray_beams
    else:
        return nanoBragg_beam_spec


def sanity_test_input_lines(input_lines):
    for line in input_lines:
        line_items = line.strip().split()
        if len(line_items) not in [2, 3, 4]:
            raise IOError("Input line %s is not formatted properly" % line)
        for item in line_items:
            if os.path.isfile(item) and not os.path.exists(item):
                raise FileNotFoundError("File %s does not exist" % item)


def full_img_pfs(img_sh):
    """shape in numpy style"""
    Panel_inds, Slow_inds, Fast_inds = map(np.ravel, np.indices(img_sh) )
    pfs_coords = np.vstack([Panel_inds, Fast_inds, Slow_inds]).T
    pfs_coords_flattened = pfs_coords.ravel()
    pfs_full = flex.size_t( np.ascontiguousarray(pfs_coords_flattened))
    return pfs_full


def print_profile(stats, timed_methods):
    for method in stats.timings.keys():
        filename, header_ln, name = method
        if name not in timed_methods:
            continue
        info = stats.timings[method]
        PROFILE_LOGGER.warning("\n")
        PROFILE_LOGGER.warning("FILE: %s" % filename)
        if not info:
            PROFILE_LOGGER.warning("<><><><><><><><><><><><><><><><><><><><><><><>")
            PROFILE_LOGGER.warning("METHOD %s : Not profiled because never called" % (name))
            PROFILE_LOGGER.warning("<><><><><><><><><><><><><><><><><><><><><><><>")
            continue
        unit = stats.unit

        line_nums, ncalls, timespent = zip(*info)
        fp = open(filename, 'r').readlines()
        total_time = sum(timespent)
        header_line = fp[header_ln-1][:-1]
        PROFILE_LOGGER.warning(header_line)
        PROFILE_LOGGER.warning("TOTAL FUNCTION TIME: %f ms" % (total_time*unit*1e3))
        PROFILE_LOGGER.warning("<><><><><><><><><><><><><><><><><><><><><><><>")
        PROFILE_LOGGER.warning("%5s%14s%9s%10s" % ("Line#", "Time", "%Time", "Line" ))
        PROFILE_LOGGER.warning("%5s%14s%9s%10s" % ("", "(ms)", "", ""))
        PROFILE_LOGGER.warning("<><><><><><><><><><><><><><><><><><><><><><><>")
        for i_l, l in enumerate(line_nums):
            frac_t = timespent[i_l] / total_time * 100.
            line = fp[l-1][:-1]
            PROFILE_LOGGER.warning("%5d%14.2f%9.2f%s" % (l, timespent[i_l]*unit*1e3, frac_t, line))


def get_lam0_lam1_from_pandas(df):
    lam0 = df.lam0.values[0]
    lam1 = df.lam1.values[0]
    assert not np.isnan(lam0)
    assert not np.isnan(lam1)
    assert lam0 != -1
    assert lam1 != -1
    return lam0, lam1


def estimate_spot_scale(Modeler, SIM):
    """
    Estimate a reasonable init.G by comparing data signal to a forward model at G=1.

    The forward model at G=1 already reflects the no_Nabc_scale flag, so the ratio
    data_sum/model_sum gives the correct G directly.

    Returns the estimated G value.
    """
    x0 = np.ones(len(Modeler.P))
    model_pix, _ = model(x0, Modeler, SIM, compute_grad=False)
    # model_pix is Bragg-only at G=1; data signal = data - background
    data_signal = Modeler.all_data - Modeler.all_background
    trusted = Modeler.all_trusted

    model_sum = model_pix[trusted].sum()
    data_sum = data_signal[trusted].sum()

    # Diagnostics: understand why model might be zero
    model_max = model_pix.max()
    model_nonzero = np.count_nonzero(model_pix > 1e-10)
    MAIN_LOGGER.info("estimate_spot_scale: model_max=%.4g, model_nonzero_pixels=%d/%d, "
                     "data_sum=%.4g, model_sum=%.4g"
                     % (model_max, model_nonzero, len(model_pix), data_sum, model_sum))
    # Per-ROI breakdown: check a few ROIs for signal
    MAIN_LOGGER.info("  Total ROIs: %d, first few HKLs: %s" %
                     (len(Modeler.rois), str(Modeler.Hi[:5]) if Modeler.Hi else "none"))
    # Check structure factors
    if hasattr(SIM, 'crystal') and hasattr(SIM.crystal, 'miller_array') and SIM.crystal.miller_array is not None:
        F = np.array(SIM.crystal.miller_array.data())
        MAIN_LOGGER.info("  Structure factors: n=%d, min=%.4g, max=%.4g, mean=%.4g"
                         % (len(F), F.min(), F.max(), F.mean()))
    # Check beam
    if hasattr(SIM, 'beam') and hasattr(SIM.beam, 'xray_beams'):
        beams = SIM.beam.xray_beams
        MAIN_LOGGER.info("  Beam: %d energy channels, wavelength=%.6g (raw), spectrum[0]=(%s), unit_s0=%s" %
                         (len(beams), beams[0].get_wavelength(),
                          str(SIM.beam.spectrum[0]),
                          str(tuple(SIM.beam.unit_s0))))
    # Check gonio
    MAIN_LOGGER.info("  Gonio: phi_deg=%.4f, osc_deg=%.4f, phisteps=%d, axis=%s" %
                     (SIM.D.phi_deg, SIM.D.osc_deg, SIM.D.phisteps,
                      str(tuple(SIM.D.spindle_axis))))
    # Check crystal
    MAIN_LOGGER.info("  Crystal Amatrix: %s" % str(tuple(SIM.D.Amatrix)[:9]))
    # Check what the engine actually sees for beam
    engine_beams = SIM.D.xray_beams
    if len(engine_beams) > 0:
        MAIN_LOGGER.info("  Engine beam[0]: wavelength=%.6g, flux=%.6g" %
                         (engine_beams[0].get_wavelength(), engine_beams[0].get_flux()))
    # Also check detector
    MAIN_LOGGER.info("  Detector: %d panels, panel0 origin=%s" %
                     (len(SIM.detector), str(SIM.detector[0].get_origin())))

    if model_sum <= 0:
        MAIN_LOGGER.warning("estimate_spot_scale: model sum <= 0, cannot auto-estimate G")
        return None

    G_est = data_sum / model_sum

    MAIN_LOGGER.info("Auto-estimated init.G = %.4g (data_sum=%.4g, model_sum=%.4g)"
                     % (G_est, data_sum, model_sum))
    return G_est


def get_simulator_for_data_modelers(data_modeler):
    self = data_modeler
    SIM = utils.simulator_for_refinement(self.E, self.params)

    if self.params.use_diffuse_models:
        if self.params.symmetrize_diffuse:
            assert self.params.space_group is not None
            SIM.D.laue_group_num = utils.get_laue_group_number(
                self.params.space_group)  # TODO this can also be retrieved from crystal model if params.space_group is None
            MAIN_LOGGER.debug("Set laue group number: %d (for diffuse models)" % SIM.D.laue_group_num)
        if self.params.diffuse_stencil_size > 0:
            SIM.D.stencil_size = self.params.diffuse_stencil_size
            MAIN_LOGGER.debug("Set diffuse stencil size: %d" % SIM.D.stencil_size)
        if self.params.diffuse_orientation == 1:
            ori = (1,0,0,0,1,0,0,0,1)
        else:
            a = 1/np.sqrt(2)
            ori = a, a, 0.0, a, a, 0.0, 0.0, 0.0, 1.0
        SIM.D.set_rotate_principal_axes(ori)
    SIM.D.gamma_miller_units = self.params.gamma_miller_units
    SIM.isotropic_diffuse_gamma = self.params.isotropic.diffuse_gamma
    SIM.isotropic_diffuse_sigma = self.params.isotropic.diffuse_sigma
    if self.params.record_device_timings:
        SIM.D.record_timings = True

    # TODO: use data_modeler.set_spectrum instead
    if self.params.spectrum_from_imageset:
        downsamp_spec(SIM, self.params, self.E)
    elif self.params.gen_gauss_spec:
        set_gauss_spec(SIM, self.params, self.E)
    data_modeler.nanoBragg_beam_spectrum = SIM.beam.spectrum

    # TODO: verify how slow always using lambda coefs is
    # TODO: ensure lam0/lam1 are not -1 and not np.nan
    SIM.D.use_lambda_coefficients = True
    SIM.D.lambda_coefficients = tuple(self.params.init.spec)
    _set_Fhkl_refinement_flags(self.params, SIM)
    data_modeler.set_Fhkl_channels(SIM)  # if doing ensemble refinement, do this on all modelers!

    return SIM


def _set_Fhkl_refinement_flags(params, SIM):

    SIM.refining_Fhkl = False
    SIM.Num_ASU = 0
    SIM.num_Fhkl_channels = 1
    SIM.Fhkl_channel_bounds = [0, np.inf]
    SIM.centric_flags = None
    if not params.fix.Fhkl:
        if params.Fhkl_channel_bounds is not None:
            assert params.Fhkl_channel_bounds == sorted(params.Fhkl_channel_bounds)
            SIM.Fhkl_channel_bounds = [0] + params.Fhkl_channel_bounds + [np.inf]

        SIM.num_Fhkl_channels = len(SIM.Fhkl_channel_bounds) - 1
        asu_map = SIM.D.get_ASUid_map()
        SIM.asu_map_int = {tuple(map(int, k.split(','))): v for k, v in asu_map.items()}

        num_unique_hkl = len(asu_map)
        SIM.Fhkl_scales_init = np.ones(num_unique_hkl * SIM.num_Fhkl_channels)
        SIM.Fhkl_sigmas = np.full(num_unique_hkl * SIM.num_Fhkl_channels, float(params.sigmas.Fhkl))
        SIM.Fhkl_linear = params.Fhkl_linear_parameterization
        SIM.refining_Fhkl = True
        SIM.Num_ASU = num_unique_hkl  # TODO replace with diffBragg property
        if params.betas.Fhkl is not None or params.betas.Finit is not None:
            MAIN_LOGGER.debug("Restraining to average Fhkl with %d bins" % params.Fhkl_dspace_bins)
            SIM.set_dspace_binning(params.Fhkl_dspace_bins)
        if params.betas.Friedel is not None:
            MAIN_LOGGER.debug("Restraining Friedel pairs")
            SIM.D.prep_Friedel_restraints()

        if SIM.num_Fhkl_channels > 1:
            assert params.space_group is not None
            sym = crystal.symmetry(SIM.D.unit_cell_tuple, params.space_group)
            hkl_flex = flex.miller_index(list(SIM.asu_map_int.keys()))
            mset = miller.set(sym, hkl_flex, True)
            cent = mset.centric_flags()
            cent_map = {h: flag for h, flag in zip(cent.indices(), cent.data())}
            SIM.is_centric = np.zeros(SIM.Num_ASU, bool)
            for h, idx in SIM.asu_map_int.items():
                is_centric = cent_map[h]
                SIM.is_centric[idx] = is_centric

            SIM.where_is_centric = np.where(SIM.is_centric)[0]


def _compute_fhkl_sigmas_from_curvatures(Modeler, SIM, x0=None):
    """Compute per-Fhkl sigma from diagonal Hessian at a given operating point.

    Runs a forward model at x0 (default: all ones), computes exact diagonal Hessian via
    add_Fhkl_gradients(errors=True), sets SIM.Fhkl_sigmas = 1/sqrt(|H_ii|).

    If x0 is provided (e.g. from a warm-up pass), the Hessian is computed at that
    operating point rather than the initial point, giving more accurate curvatures
    when non-Fhkl parameters (G, Nabc) have been pre-converged.
    """
    nscales = SIM.Num_ASU * SIM.num_Fhkl_channels
    nparam = len(Modeler.P) + nscales
    if x0 is None:
        x0 = np.ones(nparam)
    else:
        x0 = np.array(x0, dtype=np.float64)
        if len(x0) < nparam:
            # x0 has only non-Fhkl params; append Fhkl params at x=1
            x0 = np.concatenate([x0, np.ones(nscales)])

    # Always evaluate model fresh at the given operating point
    model_bragg, _ = model(x0, Modeler, SIM, compute_grad=False)
    model_pix = model_bragg + Modeler.all_background
    resid = Modeler.all_data - model_pix
    V = model_pix + Modeler.all_sigma_rdout ** 2

    G = Modeler.P["G_xtal0"].get_val(x0[Modeler.P["G_xtal0"].xpos])

    # Diagonal Hessian from C++ kernel
    hessian_diag = SIM.D.add_Fhkl_gradients(
        Modeler.pan_fast_slow, resid, V, Modeler.all_trusted, Modeler.all_freq,
        SIM.num_Fhkl_channels, G, errors=True)

    # sigma_i = 1/sqrt(|H_ii|), with floor for unobserved reflections
    # For squared-amplitude (linear) parameterization, ds/dx = 2*a*sigma at x=1 (a=1),
    # so H_xx = 4*sigma^2*H_ss. To get H_xx=1: sigma = 1/(2*sqrt(H_ss)).
    abs_hess = np.abs(hessian_diag)
    has_signal = abs_hess > 1e-12
    sigmas = np.full(nscales, float(Modeler.params.sigmas.Fhkl))  # default for no-signal
    raw_sigmas = np.full(nscales, 0.0)
    raw_sigmas[has_signal] = 1.0 / np.sqrt(abs_hess[has_signal])
    if SIM.Fhkl_linear:
        raw_sigmas[has_signal] *= 0.5  # correct for ds/dx = 2*a*sigma chain rule
    sigmas[has_signal] = raw_sigmas[has_signal]

    # Rescale: preserve relative curvature ratios, but match default sigma magnitude.
    # This ensures Fhkl gradients remain comparable to other parameters (Nabc, G, etc.)
    # when jointly refined — without this, auto-sigma makes Fhkl gradients too small
    # and the optimizer converges on non-Fhkl params before Fhkl can move.
    if has_signal.any():
        hs = abs_hess[has_signal]
        rs = raw_sigmas[has_signal]
        print("Fhkl Hessian (observed): min=%.4g, max=%.4g, median=%.4g"
              % (hs.min(), hs.max(), np.median(hs)))
        print("Fhkl raw sigma (before rescale): min=%.4g, max=%.4g, median=%.4g"
              % (rs.min(), rs.max(), np.median(rs)))
        median_observed = float(np.median(sigmas[has_signal]))
        default_sigma = float(Modeler.params.sigmas.Fhkl)
        scale_factor = default_sigma / median_observed
        print("Fhkl sigma rescale: factor=%.4g (default=%.4g / median=%.4g)"
              % (scale_factor, default_sigma, median_observed))
        sigmas[has_signal] *= scale_factor
        print("Fhkl sigma after rescale (observed): min=%.4g, max=%.4g, median=%.4g"
              % (sigmas[has_signal].min(), sigmas[has_signal].max(), np.median(sigmas[has_signal])))

    sigmas = np.clip(sigmas, 1e-4, 100)

    SIM.Fhkl_sigmas = sigmas
    SIM.Fhkl_hessian_diag = hessian_diag  # store for SNR-adaptive restraints
    print("Fhkl auto-sigma (final): min=%.4g, max=%.4g, median=%.4g, num_with_signal=%d/%d"
          % (sigmas.min(), sigmas.max(), np.median(sigmas), has_signal.sum(), nscales))
    MAIN_LOGGER.info("Fhkl auto-sigma: min=%.4g, max=%.4g, median=%.4g"
                     % (sigmas.min(), sigmas.max(), np.median(sigmas)))


def _setup_wilson_prior(Modeler, SIM):
    """Set up Wilson statistics prior for Fhkl refinement.

    Computes Sigma(d) per resolution bin (expected intensity under Wilson distribution)
    and centric flags. These are used in target_func to add a Bayesian prior:
      acentric: -log P(I) = I/Sigma
      centric:  -log P(I) = I/(2*Sigma) - 0.5*log(I)
    """
    from cctbx import uctbx, crystal, miller
    from cctbx.array_family import flex

    nscales = SIM.Num_ASU * SIM.num_Fhkl_channels

    # Get centric flags (recompute if not already set)
    if not hasattr(SIM, 'is_centric') or SIM.is_centric is None:
        sym = crystal.symmetry(SIM.D.unit_cell_tuple, Modeler.params.space_group)
        hkl_flex = flex.miller_index(list(SIM.asu_map_int.keys()))
        mset = miller.set(sym, hkl_flex, True)
        cent = mset.centric_flags()
        cent_map = {h: flag for h, flag in zip(cent.indices(), cent.data())}
        SIM.is_centric = np.zeros(SIM.Num_ASU, bool)
        for h, idx in SIM.asu_map_int.items():
            SIM.is_centric[idx] = cent_map[h]

    # Get |F|^2 from the MTZ (these are the perturbed/input structure factors)
    Fp1 = SIM.D.Fhkl
    Fp1_map = {h: amp for h, amp in zip(Fp1.indices(), Fp1.data())}
    F_sq = np.zeros(SIM.Num_ASU)  # |F_mtz|^2 per ASU index
    idx_to_asu = {idx: asu for asu, idx in SIM.asu_map_int.items()}
    for idx in range(SIM.Num_ASU):
        if idx in idx_to_asu:
            hkl = idx_to_asu[idx]
            if hkl in Fp1_map:
                F_sq[idx] = Fp1_map[hkl] ** 2

    # Compute d-spacings for resolution binning
    uc = uctbx.unit_cell(SIM.D.unit_cell_tuple)
    d_spacings = np.zeros(SIM.Num_ASU)
    for idx in range(SIM.Num_ASU):
        if idx in idx_to_asu:
            d_spacings[idx] = uc.d(idx_to_asu[idx])

    # Compute Sigma(d) = <|F|^2> per resolution bin
    n_bins = Modeler.params.Fhkl_restraint_n_bins
    valid = d_spacings > 0
    if valid.sum() < n_bins:
        n_bins = max(1, valid.sum() // 3)
    d_valid = d_spacings[valid]
    bin_edges = np.percentile(np.sort(d_valid), np.linspace(0, 100, n_bins + 1))
    bin_edges[0] -= 1e-6
    bin_edges[-1] += 1e-6
    bin_assignments = np.digitize(d_spacings, bin_edges) - 1
    bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)

    # Sigma = mean |F|^2 per bin (Wilson expectation)
    sigma_wilson = np.ones(n_bins)
    for b in range(n_bins):
        in_bin = (bin_assignments == b) & valid & (F_sq > 0)
        if in_bin.sum() > 0:
            sigma_wilson[b] = F_sq[in_bin].mean()

    # Store per-HKL: Sigma(d) for this HKL, and F_sq
    wilson_sigma = sigma_wilson[bin_assignments]  # Sigma(d) per HKL

    # Handle multi-channel case
    if SIM.num_Fhkl_channels > 1:
        wilson_sigma = np.tile(wilson_sigma, SIM.num_Fhkl_channels)
        F_sq = np.tile(F_sq, SIM.num_Fhkl_channels)
        SIM.is_centric_full = np.tile(SIM.is_centric, SIM.num_Fhkl_channels)
    else:
        SIM.is_centric_full = SIM.is_centric

    SIM.wilson_sigma = wilson_sigma
    SIM.wilson_F_sq = F_sq

    n_cent = SIM.is_centric_full.sum()
    print("Wilson prior: %d bins, %d centric / %d acentric HKLs"
          % (n_bins, n_cent, nscales - n_cent))
    print("Wilson Sigma per bin: %s" % ", ".join(["%.1f" % s for s in sigma_wilson]))


def _setup_fhkl_snr_restraint(Modeler, SIM):
    """Set up per-Fhkl restraints toward resolution-bin averages, weighted by SNR.

    Low-SNR reflections (small Hessian) get strong restraints toward their
    resolution-bin average, preventing noise-driven overfitting. High-SNR
    reflections are free to refine.
    """
    from cctbx import uctbx

    nscales = SIM.Num_ASU * SIM.num_Fhkl_channels
    sigma = Modeler.params.Fhkl_restraint_snr_sigma
    n_bins = Modeler.params.Fhkl_restraint_n_bins

    # Compute Hessian if not already done (e.g. auto_Fhkl_sigma was not enabled)
    if not hasattr(SIM, 'Fhkl_hessian_diag'):
        nparam = len(Modeler.P) + nscales
        if hasattr(Modeler, 'V_fixed'):
            V = Modeler.V_fixed
            resid = Modeler.all_data - (V - Modeler.all_sigma_rdout ** 2)
        else:
            x0 = np.ones(nparam)
            model_bragg, _ = model(x0, Modeler, SIM, compute_grad=False)
            model_pix = model_bragg + Modeler.all_background
            resid = Modeler.all_data - model_pix
            V = model_pix + Modeler.all_sigma_rdout ** 2
        G = Modeler.P["G_xtal0"].get_val(1.0)
        SIM.Fhkl_hessian_diag = SIM.D.add_Fhkl_gradients(
            Modeler.pan_fast_slow, resid, V, Modeler.all_trusted, Modeler.all_freq,
            SIM.num_Fhkl_channels, G, errors=True)

    abs_hess = np.abs(SIM.Fhkl_hessian_diag)
    has_signal = abs_hess > 1e-12

    if has_signal.sum() < 2:
        MAIN_LOGGER.warning("Too few signal-bearing HKLs for SNR restraint (%d)" % has_signal.sum())
        SIM.Fhkl_restraint_weights = None
        return

    # Compute d-spacings for all ASU HKLs
    uc = uctbx.unit_cell(SIM.D.unit_cell_tuple)
    idx_to_asu = {idx: asu for asu, idx in SIM.asu_map_int.items()}
    d_spacings = np.zeros(SIM.Num_ASU)
    for idx in range(SIM.Num_ASU):
        if idx in idx_to_asu:
            d_spacings[idx] = uc.d(idx_to_asu[idx])

    # Extend to full nscales for multi-channel
    if SIM.num_Fhkl_channels > 1:
        d_full = np.tile(d_spacings, SIM.num_Fhkl_channels)
    else:
        d_full = d_spacings

    # Create resolution bins from signal-bearing HKLs
    d_signal = d_full[has_signal]
    actual_n_bins = min(n_bins, max(1, len(d_signal) // 2))
    bin_edges = np.percentile(np.sort(d_signal), np.linspace(0, 100, actual_n_bins + 1))
    bin_edges[0] -= 1e-6
    bin_edges[-1] += 1e-6

    # Assign each HKL to a bin
    bin_assignments = np.digitize(d_full, bin_edges) - 1
    bin_assignments = np.clip(bin_assignments, 0, actual_n_bins - 1)

    # Per-HKL restraint weight in the scale of the likelihood.
    # The likelihood Hessian for HKL k is |H_kk|. The restraint penalty is
    #   f_restraint_k = 0.5 * w_k * (s_k - target_k)^2
    # We want the restraint to be comparable to the data-driven curvature, but
    # stronger for low-SNR reflections. Parameterize by sigma (allowed scale deviation):
    #   w_k = |H_kk| / sigma^2      for all HKLs with signal
    # For a median HKL: w = H_median/sigma^2. For low-SNR: H_low/sigma^2 (but H_low is
    # small, so the restraint is relatively stronger vs. the data term for that HKL).
    # Actually, we want low-SNR to be MORE restrained relative to their data term, so
    # we use the inverse SNR weighting:
    #   w_k = H_median^2 / (sigma^2 * |H_kk|)
    # This means: for median HKL, w = H_median/sigma^2 (matches data curvature at sigma=1).
    # For low-SNR HKL (small H_kk), w >> H_kk, so restraint dominates.
    # For high-SNR HKL (large H_kk), w << H_kk, so data dominates.
    H_median = np.median(abs_hess[has_signal])
    weights = np.zeros(nscales)
    weights[has_signal] = H_median**2 / (sigma**2 * abs_hess[has_signal])
    # No-signal HKLs: very strong restraint (lock to target)
    weights[~has_signal] = H_median**2 / (sigma**2 * 1e-12)
    # Clip to prevent numerical issues
    max_weight = H_median**2 / (sigma**2 * 1e-12)
    weights = np.clip(weights, 0, max_weight)

    SIM.Fhkl_restraint_weights = weights
    SIM.Fhkl_reso_bins = bin_assignments
    SIM.Fhkl_reso_n_bins = actual_n_bins

    print("Fhkl SNR restraint: %d bins, sigma=%.4g, H_median=%.4g, %d/%d with signal"
          % (actual_n_bins, sigma, H_median, has_signal.sum(), nscales))
    # Show weight range for signal-bearing HKLs
    w_sig = weights[has_signal]
    print("  weights (signal): min=%.4g, max=%.4g, median=%.4g"
          % (w_sig.min(), w_sig.max(), np.median(w_sig)))


def _two_stage_fhkl_refine(Modeler, SIM, params, x0):
    """Two-stage Fhkl refinement: strong HKLs first, then weak with restraints.

    Stage 1: Freeze weak HKLs (sigma=0), refine strong.
    Stage 2: Fix strong at stage 1 values, refine weak restrained to
             resolution-bin averages from stage 1.
    """
    from cctbx import uctbx

    nscales = SIM.Num_ASU * SIM.num_Fhkl_channels
    nparam = len(Modeler.P) + nscales
    threshold = params.Fhkl_two_stage_threshold

    # Ensure we have Hessian
    if not hasattr(SIM, 'Fhkl_hessian_diag'):
        _compute_fhkl_sigmas_from_curvatures(Modeler, SIM)

    abs_hess = np.abs(SIM.Fhkl_hessian_diag)
    has_signal = abs_hess > 1e-12

    if has_signal.sum() < 2:
        MAIN_LOGGER.warning("Too few signal-bearing HKLs for two-stage refinement")
        return Modeler.Minimize(x0, SIM)

    H_median = np.median(abs_hess[has_signal])
    is_weak = abs_hess < threshold * H_median
    is_weak[~has_signal] = True  # no-signal HKLs are always weak
    is_strong = ~is_weak

    n_weak = is_weak.sum()
    n_strong = is_strong.sum()
    n_weak_signal = (is_weak & has_signal).sum()

    print("Two-stage Fhkl: %d strong, %d weak (%d with signal), cutoff=%.2f"
          % (n_strong, n_weak, n_weak_signal, threshold * H_median))

    # Save original state
    original_sigmas = SIM.Fhkl_sigmas.copy()
    original_init = SIM.Fhkl_scales_init.copy()

    # === STAGE 1: refine strong HKLs only ===
    SIM.Fhkl_sigmas[is_weak] = 0  # freeze weak
    print("=== Stage 1: refining %d strong HKLs ===" % is_strong.sum())
    x_stage1 = Modeler.Minimize(x0, SIM)

    # Apply stage 1 to get optimized scales
    model(x_stage1, Modeler, SIM, compute_grad=False)
    stage1_scales = SIM.Fhkl_scales.copy()

    # Compute resolution bins from strong HKLs
    uc = uctbx.unit_cell(SIM.D.unit_cell_tuple)
    idx_to_asu = {idx: asu for asu, idx in SIM.asu_map_int.items()}
    d_spacings = np.zeros(SIM.Num_ASU)
    for idx in range(SIM.Num_ASU):
        if idx in idx_to_asu:
            d_spacings[idx] = uc.d(idx_to_asu[idx])

    if SIM.num_Fhkl_channels > 1:
        d_full = np.tile(d_spacings, SIM.num_Fhkl_channels)
    else:
        d_full = d_spacings

    n_bins = params.Fhkl_restraint_n_bins
    d_strong = d_full[is_strong & (d_full > 0)]
    if len(d_strong) == 0:
        MAIN_LOGGER.warning("No strong HKLs with valid d-spacing for stage 2")
        return x_stage1

    # At least 5 strong HKLs per bin for robust averages
    actual_n_bins = min(n_bins, max(1, len(d_strong) // 5))
    bin_edges = np.percentile(np.sort(d_strong), np.linspace(0, 100, actual_n_bins + 1))
    bin_edges[0] -= 1e-6
    bin_edges[-1] += 1e-6
    bin_assignments = np.digitize(d_full, bin_edges) - 1
    bin_assignments = np.clip(bin_assignments, 0, actual_n_bins - 1)

    # Bin averages from strong HKLs only
    bin_avg = np.ones(actual_n_bins)
    for b in range(actual_n_bins):
        in_bin_strong = (bin_assignments == b) & is_strong
        if in_bin_strong.any():
            bin_avg[b] = stage1_scales[in_bin_strong].mean()
    targets = bin_avg[bin_assignments]

    # Overall mean correction from strong HKLs (fallback for no-signal)
    overall_mean = stage1_scales[is_strong].mean()
    print("Stage 1 done. %d bins, overall mean=%.4f, bin averages: %s"
          % (actual_n_bins, overall_mean,
             ", ".join(["%.4f" % v for v in bin_avg])))

    # === STAGE 2: refine weak HKLs restrained to bin averages ===
    # Update init: strong at optimized values, weak-with-signal at bin average
    new_init = stage1_scales.copy()
    new_init[is_weak & has_signal] = targets[is_weak & has_signal]
    # No-signal HKLs: keep at original init (no data to guide them)
    new_init[is_weak & ~has_signal] = original_init[is_weak & ~has_signal]
    SIM.Fhkl_scales_init = new_init

    # Fix strong (sigma=0), restore weak-with-signal sigmas, freeze no-signal
    SIM.Fhkl_sigmas = original_sigmas.copy()
    SIM.Fhkl_sigmas[is_strong] = 0
    SIM.Fhkl_sigmas[is_weak & ~has_signal] = 0  # no data → freeze

    # Restraint: pull weak-with-signal HKLs toward bin averages from strong
    # Weight = H_median so restraint dominates for low-SNR HKLs
    weights = np.zeros(nscales)
    weights[is_weak & has_signal] = H_median
    # no-signal HKLs: frozen (sigma=0), no restraint needed
    SIM.Fhkl_restraint_weights = weights
    SIM.Fhkl_restraint_targets = targets  # fixed targets from stage 1

    # Build x0 for stage 2: keep non-Fhkl params from stage 1, Fhkl at 1.0
    nP = len(Modeler.P)
    x0_stage2 = np.ones(nparam)
    x0_stage2[:nP] = x_stage1[:nP]

    print("=== Stage 2: refining %d weak HKLs with restraints ===" % n_weak)
    x_stage2 = Modeler.Minimize(x0_stage2, SIM)

    return x_stage2


def get_variance_s(Mod, Jac):
    """
    Gets the variance associated with per-reflection scale factors fit using diffBragg
    The Jacobian passed to this method should be computed using the hopper_utils.model method with dont_rescale_gradient=True
    with the optimized parameters (e.g., after minimization)
    :param Mod: data modeler
    :param Jac: Jacobian at the minimum
    :return: the diagonal of the inverse Hessian
    """
    model_pix = Mod.best_model
    if not Mod.best_model_includes_background:
        model_pix += Mod.all_background

    u = Mod.all_data - model_pix  # residuals, named "u" in notes

    sigma_rdout = Mod.params.refiner.sigma_r / Mod.params.refiner.adu_per_photon
    v = model_pix + sigma_rdout**2
    one_by_v = 1/v
    G = 1-2*u - u*u*one_by_v
    coef = one_by_v*(one_by_v*G - 2  - 2*u*one_by_v -u*u*one_by_v*one_by_v)

    coef_t = coef[Mod.all_trusted]
    Jac_t = Jac[:,Mod.all_trusted]
    # if we are only optimizing Fhkl, then the Hess is diagonal matrix
    diag_Hess = -.5*np.sum(coef_t*(Jac_t)**2, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        variance_s = 1/diag_Hess
    return variance_s


def split_line(line):
    line_fields = line.strip().split()
    num_fields = len(line_fields)
    assert num_fields in [2, 3, 4]
    exp, ref = line_fields[:2]
    spec = None
    exp_idx = 0
    if num_fields == 3:
        try:
            exp_idx = int(line_fields[2])
        except ValueError:
            spec = line_fields[2]
            exp_idx = 0
    elif num_fields == 4:
        assert os.path.isfile(line_fields[2])
        spec = line_fields[2]
        exp_idx = int(line_fields[3])
    return exp, ref, exp_idx, spec
