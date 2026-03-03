from __future__ import division, print_function
import time
from copy import deepcopy
import os
import numpy as np
import pandas
import glob
from pylab import plt
from scipy.optimize import basinhopping
import logging
MAIN_LOGGER = logging.getLogger("diffBragg.main")

from libtbx.mpi4py import MPI
COMM = MPI.COMM_WORLD
from dials.array_family import flex
from dxtbx.model import Experiment, ExperimentList
from dxtbx.model import Detector, Panel
from simtbx.diffBragg.hopper_io import single_expt_pandas
from simtbx.diffBragg import utils, hopper_utils, ensemble_refine_launcher
from simtbx.diffBragg.refiners.parameters import RangedParameter, Parameters
from simtbx.diffBragg import psf
from simtbx.diffBragg.prep_stage2_input import prep_dataframe
from simtbx.diffBragg.geom_utils import (
    convolve_model_with_psf,
    detector_model_derivs, PAN_OFS_IDS, PAN_XYZ_IDS, update_detector)
from cctbx import miller


DEG_TO_PI = np.pi / 180.


def get_dist_from_R(R):
    """ returns prediction offset, R is reflection table"""
    x, y, _ = R['xyzobs.px.value'].parts()
    x2, y2, _ = R['xyzcal.px'].parts()
    dist = np.sqrt((x - x2) ** 2 + (y - y2) ** 2)
    return dist


class BeamParameters:
    def __init__(self, phil_params, data_modelers):
        self.parameters = []
        # initialize as the median of all lam0, lam1 values
        all_lam0 = []
        all_lam1 = []
        for i_m in data_modelers:
            m = data_modelers[i_m]
            spec0, spec1 = m.PAR.spec_coef
            lam0, lam1 = spec0.init, spec1.init
            all_lam0.append(lam0)
            all_lam1.append(lam1)
        all_lam0 = COMM.reduce(all_lam0)
        all_lam1 = COMM.reduce(all_lam1)
        global_lam0 = global_lam1 = None
        if COMM.rank==0:
            global_lam0 = np.median(all_lam0)
            global_lam1 = np.median(all_lam1)
        global_lam0 = COMM.bcast(global_lam0)
        global_lam1 = COMM.bcast(global_lam1)

        for i_p, init_val in enumerate((global_lam0, global_lam1)):
            cent = phil_params.centers.spec
            if cent is not None:
                cent = cent[i_p]
            beta = phil_params.betas.spec
            if beta is not None:
                beta = cent[i_p]
            p = RangedParameter(name="lambda%d" % i_p,
                                init=init_val,
                                sigma=phil_params.sigmas.spec[i_p],
                                minval=phil_params.mins.spec[i_p],
                                maxval=phil_params.maxs.spec[i_p],
                                fix=phil_params.fix.spec,
                                center=cent,
                                beta=beta,
                                is_global=True)
            self.parameters.append(p)

class SourceIParameter:
    def __init__(self):
        self.name = None
        self.xpos = None
        self.scale_idx = None
        self.value = None
        self.fix = True
        self.is_global = True
        self.scale = None


class SourceIParameters:
    def __init__(self, params, SIM):
        self.parameters = []
        for i_source in range(len(SIM.beam.xray_beams)):
            p = RangedParameter(name="sourceI_%d" % i_source,
                                init=1,
                                sigma=params.geometry.sigma_sourceI,  # TODO
                                minval=0,
                                maxval=100,
                                fix=not SIM.refining_sourceI, center=None, beta=None, is_global=True)
            #p = SourceIParameter()
            #p.scale_idx =  i_source
            #p.fix = False
            #p.name = "sourceI_%d" % i_source
            self.parameters.append(p)

class FhklParameter:
    def __init__(self):
        self.name = None
        self.xpos = None
        self.scale_idx = None
        self.value = None
        self.fix = True
        self.is_global = True
        self.amp = None
        self.scale = None

class FhklParameters:
    def __init__(self, params, SIM, hiasu):
        self.parameters = []
        for h in hiasu.to_idx:
            assert h in SIM.asu_map_int
            for i_chan in range(SIM.num_Fhkl_channels):
                asu_idx = SIM.asu_map_int[h]
                #is_centric = SIM.is_centric[asu_idx]
                #if is_centric and i_chan > 0: # only refine one energy channel Fhkl if its centric
                #    continue
                p = RangedParameter(name="Fhkl_%d,%d,%d_channel%d" % (h+ (i_chan,)),
                                init=1,
                                sigma=params.sigmas.Fhkl,  # TODO
                                minval=0,
                                maxval=1e7,
                                fix=params.fix.Fhkl, center=None, beta=None, is_global=True)

                #p = FhklParameter()
                p.scale_idx = SIM.asu_map_int[h] + i_chan*SIM.Num_ASU
                #p.fix = False
                #if is_centric:
                #    p.name = "Fhkl_%d,%d,%d_centric" % h
                #else:
                #p.name = "Fhkl_%d,%d,%d_channel%d" % (h+ (i_chan,))
                self.parameters.append(p)


class GoniometerParameters:
    """
    Goniometer axis refinement using spherical parameterization.

    The rotation axis is a unit vector with 2 degrees of freedom.
    We parameterize using spherical angles (theta, phi):
        - theta: polar angle from z-axis [0, π]
        - phi: azimuthal angle from x-axis [0, 2π]
        - axis = (sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))

    This is a global parameter - all crystals share the same goniometer axis.
    """

    @staticmethod
    def cartesian_to_spherical(axis):
        """Convert Cartesian unit vector to spherical angles (theta, phi)."""
        x, y, z = axis
        # Normalize just in case
        norm = np.sqrt(x*x + y*y + z*z)
        if norm == 0:
            return 0.0, 0.0  # Degenerate case
        x, y, z = x/norm, y/norm, z/norm

        # theta: polar angle from z-axis
        theta = np.arccos(np.clip(z, -1.0, 1.0))

        # phi: azimuthal angle from x-axis
        phi = np.arctan2(y, x)
        if phi < 0:
            phi += 2*np.pi

        return theta, phi

    @staticmethod
    def spherical_to_cartesian(theta, phi):
        """Convert spherical angles to Cartesian unit vector."""
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return (x, y, z)

    def __init__(self, phil_params, SIM=None):
        self.parameters = []

        # Get initial axis from phil params (authoritative source)
        # This supports both stills and rotation experiments uniformly
        # If SIM is provided and gonio was initialized, use that; otherwise use phil
        if SIM is not None and hasattr(SIM, 'D') and SIM.D.spindle_axis is not None:
            init_axis = SIM.D.spindle_axis
        else:
            init_axis = tuple(phil_params.simulator.gonio.axis)
        theta_init, phi_init = self.cartesian_to_spherical(init_axis)

        beta = phil_params.geometry.betas.gonio_axis

        # Create theta parameter (polar angle)
        theta_param = RangedParameter(
            name="gonio_theta",
            init=theta_init,
            sigma=phil_params.geometry.sigma_gonio_axis,
            minval=0.0,
            maxval=np.pi,
            fix=phil_params.geometry.fix.gonio_axis,
            center=theta_init,
            beta=beta,
            is_global=True)

        # Create phi parameter (azimuthal angle)
        phi_param = RangedParameter(
            name="gonio_phi",
            init=phi_init,
            sigma=phil_params.geometry.sigma_gonio_axis,
            minval=0.0,
            maxval=2*np.pi,
            fix=phil_params.geometry.fix.gonio_axis,
            center=phi_init,
            beta=beta,
            is_global=True)

        self.parameters = [theta_param, phi_param]


class DetectorParameters:

    def __init__(self, phil_params, panel_groups_refined, num_panel_groups):

        self.parameters = []
        GEO = phil_params.geometry
        for i_group in range(num_panel_groups):
            group_has_data = i_group in panel_groups_refined
            if not group_has_data:
                continue
            vary_rots = [not fixed_flag and group_has_data for fixed_flag in GEO.fix.panel_rotations]
            #vary_rots = [True]*3

            o = RangedParameter(name="group%d_RotOrth" % i_group,
                                init=0,
                                sigma=1,
                                minval=GEO.min.panel_rotations[0]*DEG_TO_PI,
                                maxval=GEO.max.panel_rotations[0]*DEG_TO_PI,
                                fix=not vary_rots[0], center=0, beta=GEO.betas.panel_rot[0], is_global=True)

            f = RangedParameter(name="group%d_RotFast" % i_group,
                                init=0,
                                sigma=1,
                                minval=GEO.min.panel_rotations[1]*DEG_TO_PI,
                                maxval=GEO.max.panel_rotations[1]*DEG_TO_PI,
                                fix=not vary_rots[1], center=0, beta=GEO.betas.panel_rot[1],
                                is_global=True)

            s = RangedParameter(name="group%d_RotSlow" % i_group,
                                init=0,
                                sigma=1,
                                minval=GEO.min.panel_rotations[2]*DEG_TO_PI,
                                maxval=GEO.max.panel_rotations[2]*DEG_TO_PI,
                                fix=not vary_rots[2], center=0, beta=GEO.betas.panel_rot[2],
                                is_global=True)

            vary_shifts = [not fixed_flag and group_has_data for fixed_flag in GEO.fix.panel_translations]
            #vary_shifts = [True]*3
            x = RangedParameter(name="group%d_ShiftX" % i_group, init=0,
                                sigma=1,
                                minval=GEO.min.panel_translations[0]*1e-3, maxval=GEO.max.panel_translations[0]*1e-3,
                                fix=not vary_shifts[0], center=0, beta=GEO.betas.panel_xyz[0],
                                is_global=True)
            y = RangedParameter(name="group%d_ShiftY" % i_group, init=0,
                                sigma=1,
                                minval=GEO.min.panel_translations[1]*1e-3, maxval=GEO.max.panel_translations[1]*1e-3,
                                fix=not vary_shifts[1], center=0, beta=GEO.betas.panel_xyz[1],
                                is_global=True)
            z = RangedParameter(name="group%d_ShiftZ" % i_group, init=0,
                                sigma=1,
                                minval=GEO.min.panel_translations[2]*1e-3, maxval=GEO.max.panel_translations[2]*1e-3,
                                fix=not vary_shifts[2], center=0, beta=GEO.betas.panel_xyz[2],
                                is_global=True)

            self.parameters += [o, f, s, x, y, z]



class CrystalParameters:

    def __init__(self, phil_params, data_modelers):
        self.phil = phil_params
        self.parameters = []
        self.alias_pairs = []  # list of (alias_name, canonical_name) string tuples
        if phil_params.geometry.shared_crystal:
            self._init_shared(data_modelers)
        else:
            self._init_per_shot(data_modelers)

    def _init_per_shot(self, data_modelers):
        """Original per-shot crystal parameters (independent RotXYZ/ucell per shot)."""
        for i_shot in data_modelers:
            Mod = data_modelers[i_shot]

            # set the per-spot scale factors, per pixel...
            Mod.set_slices("roi_id")
            Mod.per_roi_scales_per_pix = np.ones_like(Mod.all_data)
            has_scale_factor = "scale_factor" in list(Mod.refls[0].keys())
            for roi_id, ref_idx in enumerate(Mod.refls_idx):
                init_scale = 1.0
                if has_scale_factor:
                    init_scale = float(Mod.refls[ref_idx]["scale_factor"])
                    if init_scale != 1.0:
                        slcs = Mod.roi_id_slices[roi_id]
                        assert len(slcs) == 1
                        Mod.per_roi_scales_per_pix[slcs[0]] = init_scale

                p = RangedParameter(name="rank%d_shot%d_scale_roi%d" % (COMM.rank, i_shot, roi_id),
                                    minval=0, maxval=1e12, fix=self.phil.geometry.fix.perRoiScale,
                                    center=init_scale, beta=1e12, init=init_scale)
                self.parameters.append(p)

            for i_N in range(3):
                p = Mod.PAR.Nabc[i_N]
                ref_p = RangedParameter(name="rank%d_shot%d_Nabc%d" % (COMM.rank, i_shot, i_N),
                                        minval=p.minval, maxval=p.maxval, fix=self.phil.geometry.fix.Nabc, init=p.init,
                                        center=p.center, beta=p.beta)
                self.parameters.append(ref_p)

            for i_N in range(3):
                p = Mod.PAR.Ndef[i_N]
                ref_p = RangedParameter(name="rank%d_shot%d_Ndef%d" % (COMM.rank, i_shot, i_N),
                                        minval=p.minval, maxval=p.maxval, fix=self.phil.geometry.fix.Ndef, init=p.init,
                                        center=p.center, beta=p.beta)
                self.parameters.append(ref_p)

            for i_eta in range(3):
                p = Mod.PAR.eta[i_eta]
                ref_p = RangedParameter(name="rank%d_shot%d_eta%d" % (COMM.rank, i_shot, i_eta),
                                        minval=p.minval, maxval=p.maxval, fix=self.phil.geometry.fix.eta_abc, init=p.init,
                                        center=p.center, beta=p.beta)
                self.parameters.append(ref_p)

            for i_rot in range(3):
                p = Mod.PAR.RotXYZ_params[i_rot]
                ref_p = RangedParameter(name="rank%d_shot%d_RotXYZ%d" % (COMM.rank, i_shot, i_rot),
                                        minval=p.minval, maxval=p.maxval, fix=self.phil.geometry.fix.RotXYZ[i_rot], init=p.init,
                                        center=p.center, beta=p.beta)
                self.parameters.append(ref_p)

            p = Mod.PAR.Scale
            ref_p = RangedParameter(name="rank%d_shot%d_Scale" % (COMM.rank, i_shot),
                                    minval=p.minval, maxval=p.maxval, fix=self.phil.geometry.fix.G, init=p.init,
                                    center=p.center, beta=p.beta)
            self.parameters.append(ref_p)

            for i_uc in range(len(Mod.PAR.ucell)):
                p = Mod.PAR.ucell[i_uc]
                ref_p = RangedParameter(name="rank%d_shot%d_Ucell%d" % (COMM.rank, i_shot, i_uc),
                                        minval=p.minval, maxval=p.maxval, fix=self.phil.geometry.fix.ucell, init=p.init,
                                        center=p.center, beta=p.beta)
                self.parameters.append(ref_p)

            # Per-shot Bfactor (fixed by default, carries the hopper-refined value)
            bfac_p = Mod.PAR.B
            ref_p = RangedParameter(name="rank%d_shot%d_Bfactor" % (COMM.rank, i_shot),
                                    minval=bfac_p.minval, maxval=bfac_p.maxval, fix=True,
                                    init=bfac_p.init, center=bfac_p.center, beta=bfac_p.beta)
            self.parameters.append(ref_p)

            # Per-shot Bfactor_aniso (6 components, fixed by default)
            if hasattr(Mod.PAR, 'Baniso') and Mod.PAR.Baniso is not None:
                for i_ba in range(6):
                    ba_p = Mod.PAR.Baniso[i_ba]
                    ref_p = RangedParameter(name="rank%d_shot%d_Baniso%d" % (COMM.rank, i_shot, i_ba),
                                            minval=ba_p.minval, maxval=ba_p.maxval, fix=True,
                                            init=ba_p.init, center=ba_p.center, beta=ba_p.beta)
                    self.parameters.append(ref_p)

            # Per-shot diffuse scattering params (fixed by default)
            if hasattr(Mod.PAR, 'diffuse_gamma') and Mod.PAR.diffuse_gamma is not None:
                for i_d in range(3):
                    dg_p = Mod.PAR.diffuse_gamma[i_d]
                    ref_p = RangedParameter(name="rank%d_shot%d_diffuse_gamma%d" % (COMM.rank, i_shot, i_d),
                                            minval=dg_p.minval, maxval=dg_p.maxval, fix=True,
                                            init=dg_p.init, center=dg_p.center, beta=dg_p.beta)
                    self.parameters.append(ref_p)
                    ds_p = Mod.PAR.diffuse_sigma[i_d]
                    ref_p = RangedParameter(name="rank%d_shot%d_diffuse_sigma%d" % (COMM.rank, i_shot, i_d),
                                            minval=ds_p.minval, maxval=ds_p.maxval, fix=True,
                                            init=ds_p.init, center=ds_p.center, beta=ds_p.beta)
                    self.parameters.append(ref_p)

    def _init_shared(self, data_modelers):
        """Shared crystal: single RotXYZ + ucell for all shots (median init)."""
        # First pass: collect per-shot RotXYZ and ucell inits
        all_rot_inits = [[], [], []]  # 3 x n_shots
        all_uc_inits = None  # will be n_uc x n_shots
        n_uc = None
        rot_template = [None, None, None]  # store one exemplar for bounds/sigma/beta
        uc_templates = None
        for i_shot in data_modelers:
            Mod = data_modelers[i_shot]
            for i_rot in range(3):
                p = Mod.PAR.RotXYZ_params[i_rot]
                all_rot_inits[i_rot].append(p.init)
                if rot_template[i_rot] is None:
                    rot_template[i_rot] = p
            if n_uc is None:
                n_uc = len(Mod.PAR.ucell)
                all_uc_inits = [[] for _ in range(n_uc)]
                uc_templates = [None] * n_uc
            for i_uc in range(n_uc):
                p = Mod.PAR.ucell[i_uc]
                all_uc_inits[i_uc].append(p.init)
                if uc_templates[i_uc] is None:
                    uc_templates[i_uc] = p

        # Compute local medians (global median happens via reduce/bcast in geom_min)
        rot_medians = [float(np.median(v)) for v in all_rot_inits]
        uc_medians = [float(np.median(v)) for v in all_uc_inits]

        # Create canonical shared parameters
        shared_rot_params = []
        for i_rot in range(3):
            tp = rot_template[i_rot]
            ref_p = RangedParameter(name="shared_RotXYZ%d" % i_rot,
                                    minval=tp.minval, maxval=tp.maxval,
                                    fix=self.phil.geometry.fix.RotXYZ[i_rot],
                                    init=rot_medians[i_rot],
                                    center=rot_medians[i_rot],
                                    beta=tp.beta, is_global=True)
            shared_rot_params.append(ref_p)
            self.parameters.append(ref_p)

        shared_uc_params = []
        for i_uc in range(n_uc):
            tp = uc_templates[i_uc]
            ref_p = RangedParameter(name="shared_Ucell%d" % i_uc,
                                    minval=tp.minval, maxval=tp.maxval,
                                    fix=self.phil.geometry.fix.ucell,
                                    init=uc_medians[i_uc],
                                    center=uc_medians[i_uc],
                                    beta=tp.beta, is_global=True)
            shared_uc_params.append(ref_p)
            self.parameters.append(ref_p)

        # Second pass: create per-shot params (non-crystal) and alias pairs
        for i_shot in data_modelers:
            Mod = data_modelers[i_shot]

            # per-ROI scale factors (per-shot, as before)
            Mod.set_slices("roi_id")
            Mod.per_roi_scales_per_pix = np.ones_like(Mod.all_data)
            has_scale_factor = "scale_factor" in list(Mod.refls[0].keys())
            for roi_id, ref_idx in enumerate(Mod.refls_idx):
                init_scale = 1.0
                if has_scale_factor:
                    init_scale = float(Mod.refls[ref_idx]["scale_factor"])
                    if init_scale != 1.0:
                        slcs = Mod.roi_id_slices[roi_id]
                        assert len(slcs) == 1
                        Mod.per_roi_scales_per_pix[slcs[0]] = init_scale
                p = RangedParameter(name="rank%d_shot%d_scale_roi%d" % (COMM.rank, i_shot, roi_id),
                                    minval=0, maxval=1e12, fix=self.phil.geometry.fix.perRoiScale,
                                    center=init_scale, beta=1e12, init=init_scale)
                self.parameters.append(p)

            for i_N in range(3):
                p = Mod.PAR.Nabc[i_N]
                ref_p = RangedParameter(name="rank%d_shot%d_Nabc%d" % (COMM.rank, i_shot, i_N),
                                        minval=p.minval, maxval=p.maxval, fix=self.phil.geometry.fix.Nabc, init=p.init,
                                        center=p.center, beta=p.beta)
                self.parameters.append(ref_p)

            for i_N in range(3):
                p = Mod.PAR.Ndef[i_N]
                ref_p = RangedParameter(name="rank%d_shot%d_Ndef%d" % (COMM.rank, i_shot, i_N),
                                        minval=p.minval, maxval=p.maxval, fix=self.phil.geometry.fix.Ndef, init=p.init,
                                        center=p.center, beta=p.beta)
                self.parameters.append(ref_p)

            for i_eta in range(3):
                p = Mod.PAR.eta[i_eta]
                ref_p = RangedParameter(name="rank%d_shot%d_eta%d" % (COMM.rank, i_shot, i_eta),
                                        minval=p.minval, maxval=p.maxval, fix=self.phil.geometry.fix.eta_abc, init=p.init,
                                        center=p.center, beta=p.beta)
                self.parameters.append(ref_p)

            # RotXYZ: alias to shared params
            for i_rot in range(3):
                alias_name = "rank%d_shot%d_RotXYZ%d" % (COMM.rank, i_shot, i_rot)
                canon_name = "shared_RotXYZ%d" % i_rot
                self.alias_pairs.append((alias_name, canon_name))

            p = Mod.PAR.Scale
            ref_p = RangedParameter(name="rank%d_shot%d_Scale" % (COMM.rank, i_shot),
                                    minval=p.minval, maxval=p.maxval, fix=self.phil.geometry.fix.G, init=p.init,
                                    center=p.center, beta=p.beta)
            self.parameters.append(ref_p)

            # ucell: alias to shared params
            for i_uc in range(n_uc):
                alias_name = "rank%d_shot%d_Ucell%d" % (COMM.rank, i_shot, i_uc)
                canon_name = "shared_Ucell%d" % i_uc
                self.alias_pairs.append((alias_name, canon_name))

            # Per-shot Bfactor (fixed by default, carries the hopper-refined value)
            bfac_p = Mod.PAR.B
            ref_p = RangedParameter(name="rank%d_shot%d_Bfactor" % (COMM.rank, i_shot),
                                    minval=bfac_p.minval, maxval=bfac_p.maxval, fix=True,
                                    init=bfac_p.init, center=bfac_p.center, beta=bfac_p.beta)
            self.parameters.append(ref_p)

            # Per-shot Bfactor_aniso (6 components, fixed by default)
            if hasattr(Mod.PAR, 'Baniso') and Mod.PAR.Baniso is not None:
                for i_ba in range(6):
                    ba_p = Mod.PAR.Baniso[i_ba]
                    ref_p = RangedParameter(name="rank%d_shot%d_Baniso%d" % (COMM.rank, i_shot, i_ba),
                                            minval=ba_p.minval, maxval=ba_p.maxval, fix=True,
                                            init=ba_p.init, center=ba_p.center, beta=ba_p.beta)
                    self.parameters.append(ref_p)

            # Per-shot diffuse scattering params (fixed by default)
            if hasattr(Mod.PAR, 'diffuse_gamma') and Mod.PAR.diffuse_gamma is not None:
                for i_d in range(3):
                    dg_p = Mod.PAR.diffuse_gamma[i_d]
                    ref_p = RangedParameter(name="rank%d_shot%d_diffuse_gamma%d" % (COMM.rank, i_shot, i_d),
                                            minval=dg_p.minval, maxval=dg_p.maxval, fix=True,
                                            init=dg_p.init, center=dg_p.center, beta=dg_p.beta)
                    self.parameters.append(ref_p)
                    ds_p = Mod.PAR.diffuse_sigma[i_d]
                    ref_p = RangedParameter(name="rank%d_shot%d_diffuse_sigma%d" % (COMM.rank, i_shot, i_d),
                                            minval=ds_p.minval, maxval=ds_p.maxval, fix=True,
                                            init=ds_p.init, center=ds_p.center, beta=ds_p.beta)
                    self.parameters.append(ref_p)


def hkl_vary_flags(SIM):
    num_fhkl_param = SIM.Num_ASU*SIM.num_Fhkl_channels
    fhkl_vary = np.ones(num_fhkl_param, int)

    #if SIM.num_Fhkl_channels > 1:
    #    assert SIM.is_centric is not None
    #    for i_chan in range(1, SIM.num_Fhkl_channels):
    #        channel_slc = slice(i_chan*SIM.Num_ASU, (i_chan+1) *SIM.Num_ASU, 1)
    #        np.subtract.at(fhkl_vary, channel_slc, SIM.is_centric.astype(int))
    return fhkl_vary.astype(bool)

class TakeStep:
    def __init__(self, paramList):
        self.LP = paramList
        self.stepsize=0.5

    def __call__(self, x):
        #new_x = np.random.normal(x, scale=self.stepsize)
        new_x = np.random.uniform(x-self.stepsize, x+self.stepsize)
        for name in self.LP:
            if name.startswith("Fhkl"):
                p = self.LP[name]
                new_x[p.xpos] = x[p.xpos]
        return new_x

class Target:
    def __init__(self, ref_params, save_state_freq=500, overwrite_state=True, plot=False):
        """

        :param ref_params: instance of refinement Parameters (LMP in code below)
        :param save_state_freq: how often to save all models (will be overwritten each time)
        """
        num_params = ref_params.n_params
        self.vary = np.zeros(num_params).astype(bool)
        for name in ref_params:
            if not ref_params.is_canonical(name):
                continue
            p = ref_params[name]
            self.vary[p.xpos] = not p.fix
        self.x0 = np.ones(num_params)
        self.g = None
        self.ref_params = ref_params
        self.iternum = 0
        self.all_times = []
        self.save_state_freq = save_state_freq
        self.overwrite_state = overwrite_state
        self.med_offsets = [] # median prediction offsets(new number gets added everytime write_output_files is called)
        self.med_iternums = []
        self.first_sigZ = None
        self.first_resid = None
        self.sigmaZ = None
        self.plot = plot and COMM.rank==0
        if self.plot:
            self.fig = plt.figure()
            self.ax = plt.gca()
            plt.draw()
            plt.pause(0.1)

    def _get_panel_group_ids(self):
        """Extract sorted list of unique panel group IDs from ref_params."""
        ids = set()
        for pname in self.ref_params:
            if pname.startswith("group") and "_" in pname:
                try:
                    gid = int(pname.split("_")[0].replace("group", ""))
                    ids.add(gid)
                except ValueError:
                    pass
        return sorted(ids)

    def _print_panel_stats(self, x0, prefix="\t"):
        """Print per-panel rotation/shift statistics across all panel groups."""
        group_ids = self._get_panel_group_ids()
        n_groups = len(group_ids)
        if n_groups == 0:
            return

        # For single-panel, print values directly (existing behavior)
        if n_groups == 1:
            gid = group_ids[0]
            det_params_str = ""
            for suffix in ["RotOrth", "RotFast", "RotSlow", "ShiftX", "ShiftY", "ShiftZ"]:
                pname = "group%d_%s" % (gid, suffix)
                if pname in self.ref_params:
                    p = self.ref_params[pname]
                    val = p.get_val(x0[p.xpos])
                    if "Rot" in suffix:
                        val = val * 180.0 / np.pi
                        unit = "deg"
                    else:
                        val = val * 1000.0
                        unit = "mm"
                    det_params_str += "%s=%.4f%s " % (suffix, val, unit)
            if det_params_str:
                print("%sDetector: %s" % (prefix, det_params_str), flush=True)
            return

        # Multi-panel: collect per-group values, print aggregate stats
        rot_names = ["RotOrth", "RotFast", "RotSlow"]
        shift_names = ["ShiftX", "ShiftY", "ShiftZ"]
        rot_vals = {n: [] for n in rot_names}
        shift_vals = {n: [] for n in shift_names}

        for gid in group_ids:
            for suffix in rot_names:
                pname = "group%d_%s" % (gid, suffix)
                if pname in self.ref_params:
                    p = self.ref_params[pname]
                    val = p.get_val(x0[p.xpos]) * 180.0 / np.pi  # rad -> deg
                    rot_vals[suffix].append(val)
            for suffix in shift_names:
                pname = "group%d_%s" % (gid, suffix)
                if pname in self.ref_params:
                    p = self.ref_params[pname]
                    val = p.get_val(x0[p.xpos]) * 1000.0  # m -> mm
                    shift_vals[suffix].append(val)

        print("%sPanel geometry (%d groups):" % (prefix, n_groups), flush=True)
        for suffix in rot_names:
            v = np.array(rot_vals[suffix])
            if len(v):
                print("%s  %s (deg): mean=%.4f  std=%.4f  min=%.4f  max=%.4f"
                      % (prefix, suffix, v.mean(), v.std(), v.min(), v.max()), flush=True)
        for suffix in shift_names:
            v = np.array(shift_vals[suffix])
            if len(v):
                print("%s  %s  (mm): mean=%.4f  std=%.4f  min=%.4f  max=%.4f"
                      % (prefix, suffix, v.mean(), v.std(), v.min(), v.max()), flush=True)

    def __call__(self, x, *args, **kwargs):
        self.iternum += 1
        t = time.time()
        self.x0[self.vary] = x
        #time_per_iter = (time.time()-self.tstart) / self.iternum

        f, self.g, self.sigmaZ = target_and_grad(self.x0, self.ref_params, iternum=self.iternum, *args, **kwargs)
        t = time.time()-t
        if self.first_sigZ is None:
            self.first_sigZ = self.sigmaZ
            self.first_resid = f
        if COMM.rank==0:
            self.all_times.append(t)
            time_per_iter = np.mean(self.all_times)
            pred_offset_str = ", ".join(map(lambda x: "%.4f" %x, self.med_offsets))
            print("Iteration %d:\n\tResid=%f, sigmaZ %f, t-per-iter=%.4f sec, pred_offsets=%s"
                  % (self.iternum, f, self.sigmaZ, time_per_iter, pred_offset_str), flush=True)

            # Print detector geometry parameters (single or multi-panel)
            self._print_panel_stats(self.x0)

            # Print RotXYZ statistics across all shots (canonical only to avoid alias duplicates)
            rotxyz_vals = []
            for pname in self.ref_params:
                if "RotXYZ" in pname and self.ref_params.is_canonical(pname):
                    p = self.ref_params[pname]
                    if not p.fix:
                        val = p.get_val(self.x0[p.xpos])
                        rotxyz_vals.append(val * 180.0 / np.pi)  # rad to deg
            if rotxyz_vals:
                rotxyz_vals = np.array(rotxyz_vals)
                print("\tRotXYZ: mean=%.4f deg, std=%.4f deg, min=%.4f deg, max=%.4f deg"
                      % (np.mean(rotxyz_vals), np.std(rotxyz_vals),
                         np.min(rotxyz_vals), np.max(rotxyz_vals)), flush=True)

            # Print goniometer axis if being refined
            if "gonio_theta" in self.ref_params and not self.ref_params["gonio_theta"].fix:
                gt = self.ref_params["gonio_theta"]
                gp = self.ref_params["gonio_phi"]
                theta_val = gt.get_val(self.x0[gt.xpos])
                phi_val = gp.get_val(self.x0[gp.xpos])
                axis = GoniometerParameters.spherical_to_cartesian(theta_val, phi_val)
                print("\tGonio axis: (%.6f, %.6f, %.6f)  theta=%.4f deg, phi=%.4f deg"
                      % (axis[0], axis[1], axis[2],
                         np.degrees(theta_val), np.degrees(phi_val)), flush=True)
        if self.iternum % self.save_state_freq==0 and self.iternum >0:
            if not self.overwrite_state:
                params = args[-1]  # phil params
                temp_pandas_dir = params.outdir
                params.outdir=params.outdir + "-iter%d" % self.iternum
            med_offset = write_output_files(self.x0, self.ref_params, iternum=self.iternum, *args, **kwargs)
            self.med_offsets.append(med_offset)
            self.med_iternums.append(self.iternum)
            if self.plot:
                self.ax.clear()
                self.ax.plot(self.med_iternums, self.med_offsets)
                self.ax.set_ylabel("median |xobs-xcal| (pixels)")
                self.ax.set_xlabel("iteration #")
                plt.draw()
                plt.pause(0.01)
            if not self.overwrite_state:
                params.outdir=temp_pandas_dir
        return f

    def jac(self, x, *args):
        if self.g is not None:
            return self.g[self.vary]

    def at_min_callback(self, x, f, accept):
        if COMM.rank==0:
            print("Final Iteration %d:\n\tResid=%f, sigmaZ %f" % (self.iternum, f, self.sigmaZ))
            if self.first_sigZ is not None:
                delta = self.sigmaZ - self.first_sigZ
                print("\tsigmaZ: first=%.6f, last=%.6f, delta=%.6f" % (
                    self.first_sigZ, self.sigmaZ, delta), flush=True)

            # Print final detector geometry parameters (single or multi-panel)
            self._print_panel_stats(self.x0)

            # Print final RotXYZ statistics (canonical only)
            rotxyz_vals = []
            for pname in self.ref_params:
                if "RotXYZ" in pname and self.ref_params.is_canonical(pname):
                    p = self.ref_params[pname]
                    if not p.fix:
                        val = p.get_val(self.x0[p.xpos])
                        rotxyz_vals.append(val * 180.0 / np.pi)
            if rotxyz_vals:
                rotxyz_vals = np.array(rotxyz_vals)
                print("\tRotXYZ: mean=%.4f deg, std=%.4f deg, min=%.4f deg, max=%.4f deg"
                      % (np.mean(rotxyz_vals), np.std(rotxyz_vals),
                         np.min(rotxyz_vals), np.max(rotxyz_vals)), flush=True)


def model(x, ref_params, i_shot, Modeler, SIM, return_bragg_model=False):
    """

    :param x: rescaled parameter array (global)
    :param ref_params: simtbx.diffBragg.refiners.parameters.Parameters() instance
    :param i_shot: shot index for this data model,
        the simtbx.diffBragg.refiners.parameters.RangerParameter objs stored in ref_params
        have names which include i_shot
    :param Modeler: DataModeler for i_shot
    :param SIM: instance of sim_data.SimData
    :param return_bragg_model: if true, bypass the latter half of the method and return the Bragg scattering model
    :return: either the Bragg scattering model (if return_model), or else a 3-tuple of
        (float, dict of float, float)
        (negative log likelihood, gradient of negative log likelihood, average sigmaZ for the shot)
    """

    rotX = ref_params["rank%d_shot%d_RotXYZ%d" % (COMM.rank, i_shot, 0)]
    rotY = ref_params["rank%d_shot%d_RotXYZ%d" % (COMM.rank, i_shot, 1)]
    rotZ = ref_params["rank%d_shot%d_RotXYZ%d" % (COMM.rank, i_shot, 2)]
    Na = ref_params["rank%d_shot%d_Nabc%d" % (COMM.rank, i_shot, 0)]
    Nb = ref_params["rank%d_shot%d_Nabc%d" % (COMM.rank, i_shot, 1)]
    Nc = ref_params["rank%d_shot%d_Nabc%d" % (COMM.rank, i_shot, 2)]
    Nd = ref_params["rank%d_shot%d_Ndef%d" % (COMM.rank, i_shot, 0)]
    Ne = ref_params["rank%d_shot%d_Ndef%d" % (COMM.rank, i_shot, 1)]
    Nf = ref_params["rank%d_shot%d_Ndef%d" % (COMM.rank, i_shot, 2)]
    eta_a = ref_params["rank%d_shot%d_eta%d" % (COMM.rank, i_shot, 0)]
    eta_b = ref_params["rank%d_shot%d_eta%d" % (COMM.rank, i_shot, 1)]
    eta_c = ref_params["rank%d_shot%d_eta%d" % (COMM.rank, i_shot, 2)]
    G = ref_params["rank%d_shot%d_Scale" % (COMM.rank, i_shot)]
    num_uc_p = len(Modeler.ucell_man.variables)
    ucell_pars = [ref_params["rank%d_shot%d_Ucell%d" % (COMM.rank, i_shot, i_uc)] for i_uc in range(num_uc_p)]
    lam0 = ref_params["lambda0"]
    lam1 = ref_params["lambda1"]

    if SIM.refining_Fhkl and SIM.update_Fhkl_scales:
        current_Fhkl_xvals = np.ones_like(SIM.Fhkl_scales_init)
        for name in ref_params:
            if name.startswith("Fhkl"):
                p = ref_params[name]
                #current_Fhkl_xvals[p.scale_idx] = x[p.xpos]
                current_Fhkl_xvals[p.scale_idx] = p.get_val(x[p.xpos])
        #SIM.Fhkl_scales = SIM.Fhkl_scales_init * np.exp(Modeler.params.sigmas.Fhkl *(current_Fhkl_xvals-1))
        SIM.Fhkl_scales = current_Fhkl_xvals
        SIM.D.update_Fhkl_scale_factors(SIM.Fhkl_scales, SIM.num_Fhkl_channels)

    # update the rotational mosaicity here
    # update the mosaicity here
    eta_params = [eta_a, eta_b, eta_c]
    if SIM.umat_maker is not None:
        # we are modeling mosaic spread
        eta_abc = [p.get_val(x[p.xpos]) for p in eta_params]
        #if not SIM.D.has_anisotropic_mosaic_spread:
        #    eta_abc = eta_abc[0]
        SIM.update_umats_for_refinement(eta_abc)

    # update the photon energy spectrum for this shot
    SIM.beam.spectrum = Modeler.spectra
    SIM.D.xray_beams = SIM.beam.xray_beams
    # TODO make faster later...
    if SIM.refining_sourceI and SIM.update_sourceI_scales:
        sourceI_scales = []
        for i_source in range(len(SIM.beam.spectrum)):
            p = ref_params[f"sourceI_{i_source}"]
            #sourceI_scale = x[p.xpos]
            sourceI_scale = p.get_val(x[p.xpos])
            sourceI_scales.append(sourceI_scale)
        SIM.sourceI_scales = np.array(sourceI_scales)
        #sigma_sourceI= Modeler.params.geometry.sigma_sourceI
        #sourceI_init_scale=1
        #SIM.sourceI_scales = sourceI_init_scale*np.exp(sigma_sourceI*(SIM.sourceI_scales-1))
        #if any(np.isinf(sI) for sI in SIM.sourceI_scales):
        #    from IPython import embed;embed()
        SIM.D.update_sourceI_scale_factors(SIM.sourceI_scales)

    # update the lambda coeff
    lambda_coef = lam0.get_val(x[lam0.xpos]), lam1.get_val(x[lam1.xpos])
    SIM.D.lambda_coefficients = lambda_coef

    # update gonio axis (check if being refined)
    if "gonio_theta" in ref_params and "gonio_phi" in ref_params:
        gonio_theta = ref_params["gonio_theta"]
        gonio_phi = ref_params["gonio_phi"]
        if not gonio_theta.fix:
            # Goniometer axis is being refined - update from parameters
            theta = gonio_theta.get_val(x[gonio_theta.xpos])
            phi = gonio_phi.get_val(x[gonio_phi.xpos])
            spindle_axis = GoniometerParameters.spherical_to_cartesian(theta, phi)
        else:
            # Fixed axis - use existing value
            spindle_axis = SIM.D.spindle_axis
    else:
        # No goniometer parameters (backward compatibility)
        spindle_axis = SIM.D.spindle_axis

    utils.update_SIM_with_gonio(SIM, delta_phi=Modeler.osc_deg,
                                num_phi_steps=Modeler.phisteps, spindle_axis=spindle_axis)
    # Set per-frame starting phi for rotation data
    if hasattr(Modeler, 'phi_deg') and Modeler.phi_deg is not None:
        SIM.D.phi_deg = Modeler.phi_deg

    # update the Bmatrix
    Modeler.ucell_man.variables = [p.get_val(x[p.xpos]) for p in ucell_pars]
    Bmatrix = Modeler.ucell_man.B_recipspace
    SIM.D.Bmatrix = Bmatrix
    for i_ucell in range(len(ucell_pars)):
        SIM.D.set_ucell_derivative_matrix(
            i_ucell + hopper_utils.UCELL_ID_OFFSET,
            Modeler.ucell_man.derivative_matrices[i_ucell])

    # update the Umat rotation matrix and the RotXYZ perturbation
    SIM.D.Umatrix = Modeler.PAR.Umatrix
    SIM.D.set_value(hopper_utils.ROTX_ID, rotX.get_val(x[rotX.xpos]))
    SIM.D.set_value(hopper_utils.ROTY_ID, rotY.get_val(x[rotY.xpos]))
    SIM.D.set_value(hopper_utils.ROTZ_ID, rotZ.get_val(x[rotZ.xpos]))

    # update the mosaic block size
    SIM.D.set_ncells_values((Na.get_val(x[Na.xpos]),
                             Nb.get_val(x[Nb.xpos]),
                             Nc.get_val(x[Nc.xpos])))
    SIM.D.Ncells_def = (Nd.get_val(x[Nd.xpos]),
                        Ne.get_val(x[Ne.xpos]),
                        Nf.get_val(x[Nf.xpos]))

    # Set per-image Bfactor (from hopper refinement, fixed during geometry)
    bfac_name = "rank%d_shot%d_Bfactor" % (COMM.rank, i_shot)
    if bfac_name in ref_params:
        Bfac = ref_params[bfac_name]
        SIM.D.Bfactor_image = Bfac.get_val(x[Bfac.xpos])

    # Set per-image Bfactor_aniso (if present)
    baniso_name = "rank%d_shot%d_Baniso0" % (COMM.rank, i_shot)
    if baniso_name in ref_params:
        baniso_vals = []
        for i_ba in range(6):
            p = ref_params["rank%d_shot%d_Baniso%d" % (COMM.rank, i_shot, i_ba)]
            baniso_vals.append(p.get_val(x[p.xpos]))
        if any(v != 0 for v in baniso_vals):
            SIM.D.Bfactor_aniso = tuple(baniso_vals)

    # Set per-image diffuse scattering params (if present)
    dgamma_name = "rank%d_shot%d_diffuse_gamma0" % (COMM.rank, i_shot)
    if dgamma_name in ref_params:
        diff_gamma = []
        diff_sigma = []
        for i_d in range(3):
            dg_p = ref_params["rank%d_shot%d_diffuse_gamma%d" % (COMM.rank, i_shot, i_d)]
            ds_p = ref_params["rank%d_shot%d_diffuse_sigma%d" % (COMM.rank, i_shot, i_d)]
            diff_gamma.append(dg_p.get_val(x[dg_p.xpos]))
            diff_sigma.append(ds_p.get_val(x[ds_p.xpos]))
        SIM.D.diffuse_gamma = tuple(diff_gamma)
        SIM.D.diffuse_sigma = tuple(diff_sigma)

    npix = int(len(Modeler.pan_fast_slow)/3.)

    # calculate the forward Bragg scattering and gradients
    SIM.D.add_diffBragg_spots(Modeler.pan_fast_slow)

    # set the scale factors per ROI
    perRoiScaleFactors = {}
    for roi_id, ref_idx in enumerate(Modeler.refls_idx):
        p = ref_params["rank%d_shot%d_scale_roi%d" % (COMM.rank, i_shot, roi_id )]
        slc = Modeler.roi_id_slices[roi_id][0]  # Note, there's always just one slice for roi_id
        if not p.refine:
            break
        scale_fac = p.get_val(x[p.xpos])
        Modeler.per_roi_scales_per_pix[slc] = scale_fac
        perRoiScaleFactors[roi_id] = (scale_fac, p)

    bragg_no_scale = (SIM.D.raw_pixels_roi[:npix]).as_numpy_array()

    # get the per-shot scale factor
    scale = G.get_val(x[G.xpos])

    #combined the per-shot scale factor with the per-roi scale factors
    all_bragg_scales = scale*Modeler.per_roi_scales_per_pix

    # scale the bragg scattering
    bragg = all_bragg_scales*bragg_no_scale
    if return_bragg_model:
        return bragg

    # this is the total forward model:
    model_pix = bragg + Modeler.all_background
    if SIM.use_psf:
        model_pix = convolve_model_with_psf(model_pix, SIM,  Modeler.pan_fast_slow, roi_id_slices=Modeler.roi_id_slices, roi_id_unique=Modeler.roi_id_unique)

    # compute the negative log Likelihood
    resid = (Modeler.all_data - model_pix)
    resid_square = resid ** 2
    V = model_pix + Modeler.nominal_sigma_rdout ** 2
    neg_LL = (.5*(np.log(2*np.pi*V) + resid_square / V))[Modeler.all_trusted].sum()

    # compute the z-score sigma as a diagnostic
    Modeler.all_zscore =  resid/np.sqrt(V)  # important to set all_zscore so we can filter.. 
    zscore_sigma = np.std(Modeler.all_zscore[Modeler.all_trusted])

    # store the gradients
    J = {}
    # this term is a common factor in all of the gradients
    common_grad_term = (0.5 / V * (1 - 2 * resid - resid_square / V))

    if perRoiScaleFactors:
        # the gradient in this case is the bragg scattering, scaled by only the total shot scale (G in the literature)
        bragg_no_roi = bragg_no_scale*scale

        for roi_id in perRoiScaleFactors:
            scale_fac, p = perRoiScaleFactors[roi_id]
            slc = Modeler.roi_id_slices[roi_id][0]  # theres just one slice for each roi_id
            d = p.get_deriv(x[p.xpos], bragg_no_roi[slc])

            if SIM.use_psf:
                x1,x2,y1,y2 = Modeler.rois[roi_id]
                sdim, fdim = y2-y1, x2-x1
                d_img = d.reshape((sdim, fdim))
                d_img = psf.convolve_with_psf(d_img, psf=SIM.PSF, **SIM.psf_args)
                d = d_img.ravel()

            d_trusted = Modeler.all_trusted[slc]
            common_term_slc = common_grad_term[slc]
            J[p.name] = (common_term_slc*d)[d_trusted].sum()

    # scale factor gradients
    conv_args = {"SIM": SIM, "pan_fast_slow": Modeler.pan_fast_slow, "roi_id_slices": Modeler.roi_id_slices, "roi_id_unique": Modeler.roi_id_unique}
    if not G.fix:
        bragg_no_roi_scale = bragg_no_scale*Modeler.per_roi_scales_per_pix
        scale_grad = G.get_deriv(x[G.xpos], bragg_no_roi_scale)
        scale_grad = convolve_model_with_psf(scale_grad, **conv_args)
        J[G.name] = (common_grad_term*scale_grad)[Modeler.all_trusted].sum()

    # Umat gradients
    for i_rot, rot in enumerate([rotX, rotY, rotZ]):
        if not rot.fix:
            rot_db_id = hopper_utils.ROTXYZ_IDS[i_rot]
            rot_grad = scale*SIM.D.get_derivative_pixels(rot_db_id).as_numpy_array()[:npix]
            rot_grad = rot.get_deriv(x[rot.xpos], rot_grad)
            rot_grad = convolve_model_with_psf(rot_grad, **conv_args)
            J[rot.name] = (common_grad_term*rot_grad)[Modeler.all_trusted].sum()

    # mosaic block size gradients
    if not Na.fix:
        Nabc_grad = SIM.D.get_ncells_derivative_pixels()
        for i_N, N in enumerate([Na, Nb, Nc]):
            N_grad = scale*(Nabc_grad[i_N][:npix].as_numpy_array())
            N_grad = N.get_deriv(x[N.xpos], N_grad)
            N_grad = convolve_model_with_psf(N_grad, **conv_args)
            J[N.name] = (common_grad_term*N_grad)[Modeler.all_trusted].sum()

    if not Nd.fix:
        Ndef_grad = SIM.D.get_ncells_def_derivative_pixels()
        for i_N, N in enumerate([Nf, Ne, Nf]):
            N_grad = scale*(Ndef_grad[i_N][:npix].as_numpy_array())
            N_grad = N.get_deriv(x[N.xpos], N_grad)
            N_grad = convolve_model_with_psf(N_grad, **conv_args)
            J[N.name] = (common_grad_term*N_grad)[Modeler.all_trusted].sum()

    if not eta_a.fix:
        if SIM.D.has_anisotropic_mosaic_spread:
            eta_abc_derivs = SIM.D.get_aniso_eta_deriv_pixels()
        else:
            eta_abc_derivs = [SIM.D.get_derivative_pixels(hopper_utils.ETA_ID)]
        for i_eta, eta in enumerate(eta_params):
            eta_grad = scale*(eta_abc_derivs[i_eta][:npix].as_numpy_array())
            eta_grad = eta.get_deriv(x[eta.xpos], eta_grad)
            eta_grad = convolve_model_with_psf(eta_grad, **conv_args)
            J[eta.name] = (common_grad_term*eta_grad)[Modeler.all_trusted].sum()
            if not SIM.D.has_anisotropic_mosaic_spread:
                break

    # unit cell gradients
    if not ucell_pars[0].fix:
        for i_ucell, uc_p in enumerate(ucell_pars):
            d = scale*SIM.D.get_derivative_pixels(hopper_utils.UCELL_ID_OFFSET+i_ucell).as_numpy_array()[:npix]
            d = uc_p.get_deriv(x[uc_p.xpos], d)
            d = convolve_model_with_psf(d, **conv_args)
            J[ucell_pars[i_ucell].name] = (common_grad_term*d)[Modeler.all_trusted].sum()

    if not lam0.fix:
        lambda_derivs = SIM.D.get_lambda_derivative_pixels()
        lam_params = lam0, lam1
        for d, pr in zip(lambda_derivs, lam_params):
            d = d.as_numpy_array()[:npix]
            d = pr.get_deriv(x[pr.xpos], d)
            d = convolve_model_with_psf(d, **conv_args)
            J[pr.name] = (common_grad_term*d)[Modeler.all_trusted].sum()

    # detector model gradients
    det_Jac = detector_model_derivs(Modeler, ref_params, SIM, x,
                                    scale=scale, common_grad_term=common_grad_term, conv_args=conv_args)
    for key in det_Jac:
        J[key] = det_Jac[key]

    # goniometer axis gradients
    if "gonio_theta" in ref_params and not ref_params["gonio_theta"].fix:
        gonio_derivs = SIM.D.get_gonio_axis_derivative_pixels()
        for i_gonio, gonio_name in enumerate(["gonio_theta", "gonio_phi"]):
            gp = ref_params[gonio_name]
            d = scale * gonio_derivs[i_gonio].as_numpy_array()[:npix]
            d = gp.get_deriv(x[gp.xpos], d)
            d = convolve_model_with_psf(d, **conv_args)
            J[gp.name] = (common_grad_term * d)[Modeler.all_trusted].sum()

    #detector_derivs = []
    #for diffbragg_parameter_id in PAN_OFS_IDS+PAN_XYZ_IDS:
    #    try:
    #        d = SIM.D.get_derivative_pixels(diffbragg_parameter_id).as_numpy_array()[:npix]
    #        d = convolve_model_with_psf(d, **conv_args)
    #        d = common_grad_term*scale*d
    #    except ValueError:
    #        d = None
    #    detector_derivs.append(d)
    #names = "RotOrth", "RotFast", "RotSlow", "ShiftX", "ShiftY", "ShiftZ"
    #for group_id in Modeler.unique_panel_group_ids:
    #    for name in names:
    #        J["group%d_%s" % (group_id, name)] = 0
    #    for pixel_rng in Modeler.group_id_slices[group_id]:
    #        trusted_pixels = Modeler.all_trusted[pixel_rng]
    #        for i_name, name in enumerate(names):
    #            par_name = "group%d_%s" % (group_id, name)
    #            det_param = ref_params[par_name]
    #            if det_param.fix:
    #                continue
    #            pixderivs = detector_derivs[i_name][pixel_rng][trusted_pixels]
    #            pixderivs = det_param.get_deriv(x[det_param.xpos], pixderivs)
    #            J[par_name] += pixderivs.sum()

    if SIM.refining_sourceI:
        Gscale = G.get_val(x[G.xpos])
        sourceI_grad = SIM.D.add_sourceI_gradients(Modeler.pan_fast_slow, resid, V, Modeler.all_trusted,
                                             Modeler.all_freq, Gscale)
        if SIM.update_sourceI_scales and COMM.rank==0:
            print("sourceI_scale stats: %.4f %.4f %.4f %.4f"
                  %(SIM.sourceI_scales.mean(), SIM.sourceI_scales.max(), SIM.sourceI_scales.min(), SIM.sourceI_scales.std()))
        #print("sourceI_scales_states",SIM.sourceI_scales.mean(), SIM.sourceI_scales.max(), SIM.sourceI_scales.min(), SIM.sourceI_scales.std())
        #sigma_sourceI= Modeler.params.geometry.sigma_sourceI
        for i_source in range(len(sourceI_grad)):
            name =f"sourceI_{i_source}"
            p = ref_params[name]
            p_g = p.get_deriv(x[p.xpos], sourceI_grad[i_source])
            if name in J:
                J[name] += p_g
            else:
                J[name] = p_g

        #sourceI_grad *= SIM.sourceI_scales * sigma_sourceI
        #for name in ref_params:
        #    if name.startswith("sourceI"):
        #        p = ref_params[name]
        #        if name in J:
        #            J[name] += sourceI_grad[p.scale_idx]
        #        else:
        #            J[name] = sourceI_grad[p.scale_idx]

    if SIM.refining_Fhkl and not Modeler.params.method=="Nelder-Mead":#  and Modeler.params.method=="L-BFGS-B":
        Gscale = G.get_val(x[G.xpos])
        fhkl_grad = SIM.D.add_Fhkl_gradients(Modeler.pan_fast_slow, resid, V, Modeler.all_trusted,
                                             Modeler.all_freq, SIM.num_Fhkl_channels, Gscale)
        # TODO: fix these restraints:
        #if params.betas.Fhkl is not None:
        #    for i_chan in range(SIM.num_Fhkl_channels):
        #        restraint_contribution_to_grad = fhkl_grad_channels[i_chan]
        #        fhkl_slice = slice(i_chan * SIM.Num_ASU, (i_chan + 1) * SIM.Num_ASU, 1)
        #        np.add.at(fhkl_grad, fhkl_slice, restraint_contribution_to_grad)

        #fhkl_grad *= SIM.Fhkl_scales * Modeler.params.sigmas.Fhkl  # sigma is always 1 for now..
        
        for name in ref_params:
            if name.startswith("Fhkl"):
                p = ref_params[name]
                p_g = p.get_deriv(x[p.xpos], fhkl_grad[p.scale_idx])
                if name in J:
                    #J[name] += fhkl_grad[p.scale_idx]
                    J[name] += p_g 
                else:
                    #J[name] = fhkl_grad[p.scale_idx]
                    J[name] = p_g

    return neg_LL, J, model_pix, zscore_sigma


def set_group_id_slices(Modeler, group_id_from_panel_id):
    """finds the boundaries for each panel group ID in the 1-D array of per-shot data
    Modeler: DataModeler instance with loaded data
    group_id_from_panel_id : dict where key is panel id and value is group id
    """
    Modeler.all_group_id = [group_id_from_panel_id[pid] for pid in Modeler.all_pid]
    splitter = np.where(np.diff(Modeler.all_group_id) != 0)[0]+1
    npix = len(Modeler.all_data)
    slices = [slice(V[0], V[-1]+1, 1) for V in np.split(np.arange(npix), splitter)]
    group_ids = [V[0] for V in np.split(np.array(Modeler.all_group_id), splitter)]
    group_id_slices = {}
    for i_group, slc in zip(group_ids, slices):
        if i_group not in group_id_slices:
            group_id_slices[i_group] = [slc]
        else:
            group_id_slices[i_group].append(slc)
    Modeler.unique_panel_group_ids = set(Modeler.all_group_id)
    logging.debug("Modeler has data on %d unique panel groups" % (len(Modeler.unique_panel_group_ids)))
    Modeler.group_id_slices = group_id_slices





def target_and_grad(x, ref_params, data_modelers, SIM, params, iternum):
    """
    Returns the target functional and the gradients
    :param x: float array of parameter values as seen by scipt.optimize (rescaled)
    :param ref_params: refinement parameter objects (diffBragg.refiners.parameters.Parameters() )
    :param data_modelers: dict of data modelers (one per experiment)
    :param SIM: sim_data instance
    :param params: phil parameters
    :return: 2-tuple, target and gradients
    """
    target_functional = 0
    grad = np.zeros(len(x))

    save_name = params.geometry.optimized_detector_name
    update_detector(x, ref_params, SIM, save_name, rank=COMM.rank)

    all_shot_sigZ = []
    if SIM.refining_Fhkl:
        #TODO add this boolean to the __init__ of nanoBragg/sim_data.SimData
        SIM.update_Fhkl_scales=True  # toggle this to on before iterating over modelers
    if SIM.refining_sourceI:
        SIM.update_sourceI_scales = True
    for i_shot in data_modelers:
        Modeler = data_modelers[i_shot]

        neg_LL, neg_LL_grad, model_pix, per_shot_sigZ = model(x, ref_params, i_shot, Modeler, SIM)
        # toggle these to off after first modeler (only needs to be done once, as its global)
        SIM.update_Fhkl_scales = False
        SIM.update_sourceI_scales = False
        all_shot_sigZ.append(per_shot_sigZ)

        # filter during refinement?
        if Modeler.params.filter_during_refinement.enable and iternum > 0:
            if iternum % Modeler.params.filter_during_refinement.after_n == 0:
                Modeler.filter_pixels(thresh=Modeler.params.filter_during_refinement.threshold)

        # accumulate the target functional for this rank/shot
        target_functional += neg_LL

        if params.use_restraints:
            for name in ref_params:
                if name.startswith("Fhkl"):
                    continue
                if not ref_params.is_canonical(name):
                    continue
                par = ref_params[name]
                if not par.is_global and not par.fix and par.beta is not None:
                    val = par.get_restraint_val(x[par.xpos])
                    target_functional += val

        # accumulate the gradients for this rank/shot
        for name in ref_params:
            if name in neg_LL_grad:
                par = ref_params[name]
                grad[par.xpos] += neg_LL_grad[name]
                # for restraints only update the per-shot restraint gradients here
                if params.use_restraints and not par.is_global and not par.fix and par.beta is not None:
                    if ref_params.is_canonical(name):
                        grad[par.xpos] += par.get_restraint_deriv(x[par.xpos])

    # sum the target functional and the gradients across all ranks
    target_functional = COMM.bcast(COMM.reduce(target_functional))
    grad = COMM.bcast(COMM.reduce(grad))

    if params.use_restraints and params.geometry.betas.close_distances is not None:
        target_functional += np.std(SIM.D.close_distances) / params.geometry.betas.close_distances
    #if SIM.refining_sourceI:
    #    target_functional += np.std(SIM.update_sourceI_scales)

    ## add in the detector parameter restraints
    if params.use_restraints:
        for name in ref_params:
            if name.startswith("Fhkl"):
                continue
            if not ref_params.is_canonical(name):
                continue
            par = ref_params[name]
            if par.is_global and not par.fix and par.beta is not None:
                target_functional += par.get_restraint_val(x[par.xpos])
                grad[par.xpos] += par.get_restraint_deriv(x[par.xpos])
    # TODO Fhkl restraints

    all_shot_sigZ = COMM.reduce(all_shot_sigZ)
    if COMM.rank == 0:
        all_shot_sigZ = np.median(all_shot_sigZ)
    all_shot_sigZ = COMM.bcast(all_shot_sigZ)

    return target_functional, grad, all_shot_sigZ


def geom_min(params):
    """
    :param params: phil parameters (simtbx/diffBragg/phil.py)
    """

    launcher = ensemble_refine_launcher.RefineLauncher(params)
    if params.geometry.input_pkl is not None:
        df = pandas.read_pickle(params.geometry.input_pkl)
    else:
        assert params.geometry.input_pkl_glob is not None
        fnames = glob.glob(params.geometry.input_pkl_glob)
        dfs = []
        for i_f, f in enumerate(fnames):
            if i_f % COMM.size != COMM.rank:
                continue
            if COMM.rank==0:
                print("Loaing hopper pkl %d / %d" %(i_f+1, len(fnames)), flush=True)
            df_i = pandas.read_pickle(f)
            dfs.append(df_i)
        dfs = COMM.reduce(dfs)
        if COMM.rank==0:
            df = pandas.concat(dfs)
        else:
            df = None
        df = COMM.bcast(df)

    if params.skip is not None:
        df = df.iloc[params.skip:]
    if params.max_process is not None:
        df = df.iloc[:params.max_process]

    pdir = params.outdir
    assert pdir is not None, "provide a pandas_dir where output files will be generated"
    params.geometry.optimized_detector_name = os.path.join(pdir, os.path.basename(params.geometry.optimized_detector_name))
    if COMM.rank==0:
        if not os.path.exists(pdir):
            os.makedirs(pdir)
    if COMM.rank == 0:
        print("Will optimize using %d experiments" %len(df))

    main_logger = logging.getLogger("diffBragg.main")
    if not main_logger.handlers:
        from simtbx.diffBragg import mpi_logger
        mpi_logger.setup_logging_from_params(params)
    df.reset_index(drop=True, inplace=True)
    #if "geom_exp" not in df:
    #    exps,refs, exp_idxs = [],[],[]
    #    for line in df.hopper_line:
    #        exp, ref, exp_idx, spec = hopper_utils.split_line(line)
    #        exps.append(exp)
    #        refs.append(ref)
    #        exp_idxs.append(exp_idx)
    #    df["geom_exp"] = exps
    #    df["geom_exp_idx"] = exp_idxs
    #    df["geom_ref"] = refs
    df, work_distribution = prep_dataframe(df, res_ranges_string=params.refiner.res_ranges, refls_key="geom_ref",
                                           exp_idx_key="geom_exp_idx", exp_key="geom_exp")
    launcher.load_inputs(df, refls_key="geom_ref", exp_key="geom_exp",
                         exp_idx_key="geom_exp_idx",
                         work_distribution=work_distribution)


    for i_shot in launcher.Modelers:
        Modeler = launcher.Modelers[i_shot]
        set_group_id_slices(Modeler, launcher.panel_group_from_id)
        if launcher.SIM.refining_Fhkl:
            Modeler.set_Fhkl_channels(launcher.SIM, set_in_diffBragg=False)

    # If optimize_goniometer is set, override fix.gonio_axis
    if params.geometry.optimize_goniometer:
        params.geometry.fix.gonio_axis = False

    # Load optimized gonio axis from reference_geom if available (for macro-cycling)
    if params.refiner.reference_geom is not None and not params.geometry.fix.gonio_axis:
        ref_El = ExperimentList.from_file(params.refiner.reference_geom, check_format=False)
        if len(ref_El) > 0 and ref_El[0].goniometer is not None:
            ref_axis = list(ref_El[0].goniometer.get_rotation_axis())
            params.simulator.gonio.axis = ref_axis
            if COMM.rank == 0:
                print("Loaded gonio axis from reference_geom: (%.6f, %.6f, %.6f)" % tuple(ref_axis))

    # Initialize gonio from phil params before creating GoniometerParameters
    # This ensures SIM.D.spindle_axis is set from phil (authoritative source)
    utils.update_SIM_with_gonio(launcher.SIM, params)

    # same on every rank:
    det_params = DetectorParameters(params, launcher.panel_groups_refined, launcher.n_panel_groups)

    beam_params = BeamParameters(params, launcher.Modelers)

    # Goniometer axis parameters (global)
    gonio_params = GoniometerParameters(params, SIM=launcher.SIM)

    # different on each rank
    crystal_params = CrystalParameters(params,launcher.Modelers)
    crystal_params.parameters = COMM.bcast(COMM.reduce(crystal_params.parameters))
    crystal_params.alias_pairs = COMM.bcast(COMM.reduce(crystal_params.alias_pairs))

    LMP = Parameters()
    for p in crystal_params.parameters + det_params.parameters + beam_params.parameters + gonio_params.parameters:
        LMP.add(p)

    # Register aliases (shared_crystal mode: per-shot RotXYZ/ucell keys -> shared params)
    for alias_name, canon_name in crystal_params.alias_pairs:
        LMP.add_alias(alias_name, LMP[canon_name])

    if launcher.SIM.refining_Fhkl:
        fhkl_params = FhklParameters(params, launcher.SIM, launcher.hiasu)
        print("ADDING %d FHKL parameters!" % len(fhkl_params.parameters))
        for p in fhkl_params.parameters:
            LMP.add(p)

    # Print geometry refinement settings for verification
    if COMM.rank == 0:
        print("\n" + "="*80)
        print("GEOMETRY REFINEMENT SETTINGS:")
        print("="*80)
        if params.geometry.shared_crystal:
            print("Shared crystal: RotXYZ + ucell (%d aliases)" % len(crystal_params.alias_pairs))
        print("Restraints enabled: %s" % params.use_restraints)
        print("Crystal params fixed: G=%s, Nabc=%s, RotXYZ=%s, ucell=%s, eta=%s" %
              (params.geometry.fix.G, params.geometry.fix.Nabc, params.geometry.fix.RotXYZ,
               params.geometry.fix.ucell, params.geometry.fix.eta_abc))
        print("\nDetector parameters:")
        print("  Rotations - fix: %s, bounds: [%.2f, %.2f] deg" %
              (params.geometry.fix.panel_rotations,
               params.geometry.min.panel_rotations[0], params.geometry.max.panel_rotations[0]))
        print("  Translations - fix: %s, bounds: [%.2f, %.2f] mm" %
              (params.geometry.fix.panel_translations,
               params.geometry.min.panel_translations[0], params.geometry.max.panel_translations[0]))
        if not params.geometry.fix.gonio_axis:
            gonio_theta_p = LMP["gonio_theta"]
            gonio_phi_p = LMP["gonio_phi"]
            print("\nGoniometer axis:")
            print("  fix: %s" % params.geometry.fix.gonio_axis)
            print("  theta init: %.6f rad (%.2f deg)" % (gonio_theta_p.init, np.degrees(gonio_theta_p.init)))
            print("  phi init: %.6f rad (%.2f deg)" % (gonio_phi_p.init, np.degrees(gonio_phi_p.init)))
            print("  sigma: %.4f" % params.geometry.sigma_gonio_axis)
            if params.geometry.betas.gonio_axis is not None:
                print("  beta: %.2e" % params.geometry.betas.gonio_axis)
        if params.use_restraints:
            print("\nRestraint betas:")
            print("  panel_rot: %s" % str(params.geometry.betas.panel_rot))
            print("  panel_xyz: %s" % str(params.geometry.betas.panel_xyz))
            if params.geometry.betas.gonio_axis is not None:
                print("  gonio_axis: %.2e" % params.geometry.betas.gonio_axis)

        # Print detector parameter details
        for pname in ["group0_RotOrth", "group0_RotFast", "group0_RotSlow",
                      "group0_ShiftX", "group0_ShiftY", "group0_ShiftZ"]:
            if pname in LMP:
                p = LMP[pname]
                print("  %s: sigma=%.2f, beta=%.2e, fixed=%s" %
                      (pname, p.sigma, p.beta if p.beta is not None else 0, p.fix))
        print("="*80 + "\n", flush=True)

    launcher.SIM.refining_sourceI = not params.geometry.fix.sourceI
    if launcher.SIM.refining_sourceI:
        sourceI_params = SourceIParameters(params, launcher.SIM)
        print("ADDING %d sourceI parameters!" % len(sourceI_params.parameters))
        for p in sourceI_params.parameters:
            LMP.add(p)
    # use spectrum coefficients
    launcher.SIM.D.use_lambda_coefficients = True
    launcher.SIM.D.lambda_coefficients = LMP["lambda0"].init, LMP["lambda1"].init

    # attached some objects to SIM for convenience
    launcher.SIM.panel_reference_from_id = launcher.panel_reference_from_id
    launcher.SIM.panel_group_from_id = launcher.panel_group_from_id
    launcher.SIM.panel_groups_refined = launcher.panel_groups_refined

    # State snapshots at geometry start (for propagation verification)
    from simtbx.diffBragg.diffbragg_state import (
        should_snapshot, capture_geometry_state, write_state_snapshot,
        compare_states, load_state_snapshot)
    x0_snap = np.ones(LMP.n_params)  # initial x (all 1s)
    for i_shot in launcher.Modelers:
        if should_snapshot(params, i_shot, COMM.rank):
            _state = capture_geometry_state(
                x0_snap, LMP, i_shot, launcher.Modelers[i_shot], launcher.SIM,
                "geom_start", COMM.rank)
            write_state_snapshot(_state, params.outdir, "geom_start_rank%d_shot%d" % (COMM.rank, i_shot))

            # Compare with hopper_final snapshot if it exists.
            # Shot distribution differs between hopper and geometry (different
            # MPI load-balancing), so match by core exper_name (strip
            # stage prefixes, geometry suffixes, and trailing indices).
            import glob as _glob
            import re as _re
            def _core_name(n):
                """Extract core experiment ID for cross-stage matching."""
                if n is None:
                    return None
                n = os.path.basename(str(n))
                n = _re.sub(r'\.expt$', '', n)
                n = _re.sub(r'_geomData_mp$', '', n)  # geometry cycle1 suffix
                n = _re.sub(r'^stage\d+_', '', n)      # cycler stage prefix
                n = _re.sub(r'_\d+$', '', n)            # trailing index
                return n
            _snap_dir = os.path.join(params.outdir, "state_snapshots")
            _pattern = os.path.join(_snap_dir, "hopper_final_rank*_shot*.json")
            _matches = sorted(_glob.glob(_pattern))
            _geom_core = _core_name(_state.get("meta", {}).get("exper_name"))
            hopper_state = None
            for _mf in _matches:
                _candidate = load_state_snapshot(
                    params.outdir,
                    os.path.splitext(os.path.basename(_mf))[0])
                if _candidate is None:
                    continue
                _hop_core = _core_name(_candidate.get("meta", {}).get("exper_name"))
                if _geom_core and _hop_core and _geom_core == _hop_core:
                    hopper_state = _candidate
                    break
            if hopper_state is not None:
                compare_states(hopper_state, _state,
                               label="hopper_final -> geom_start (%s)" % _geom_core)
                # Print hopper's final sigma Z for handoff tracking
                hdiag = hopper_state.get("diagnostics", {})
                if hdiag.get("last_sigZ") is not None:
                    print("  Hopper final sigZ=%.6f (resid=%.6g, %d iters)" % (
                        hdiag["last_sigZ"], hdiag.get("last_resid", 0),
                        hdiag.get("n_iterations", 0)), flush=True)

    # set the GPU device
    launcher.SIM.D.device_Id = COMM.rank % params.refiner.num_devices
    npx_str = "(rnk%d, dev%d): %d pix" %(COMM.rank, launcher.SIM.D.device_Id, launcher.NPIX_TO_ALLOC)
    npx_str = COMM.gather(npx_str)
    if COMM.rank==0:
        print("How many pixels each rank will allocate for on its device:")
        print("; ".join(npx_str))
    launcher.SIM.D.Npix_to_allocate = launcher.NPIX_TO_ALLOC

    # configure diffBragg instance for gradient computation
    # Use geometry.fix.* flags (not top-level fix.*) for crystal params during geometry refinement
    for i_rot in range(3):
        if not params.geometry.fix.RotXYZ[i_rot]:
            launcher.SIM.D.refine(hopper_utils.ROTXYZ_IDS[i_rot])
    if not params.geometry.fix.sourceI:
        launcher.SIM.D.refine(hopper_utils.LAMBDA_IDS[0])
        launcher.SIM.D.refine(hopper_utils.LAMBDA_IDS[1])
    if not params.geometry.fix.eta_abc:
        launcher.SIM.D.refine(hopper_utils.ETA_ID)
    if not params.geometry.fix.Nabc:
        launcher.SIM.D.refine(hopper_utils.NCELLS_ID)
    if not params.geometry.fix.Ndef:
        launcher.SIM.D.refine(hopper_utils.NCELLS_ID_OFFDIAG)
    if not params.geometry.fix.ucell:
        for i_ucell in range(launcher.SIM.num_ucell_param):
            launcher.SIM.D.refine(hopper_utils.UCELL_ID_OFFSET + i_ucell)
    for i, diffbragg_id in enumerate(PAN_OFS_IDS):
        if not params.geometry.fix.panel_rotations[i]:
            launcher.SIM.D.refine(diffbragg_id)
    if not params.geometry.fix.gonio_axis:
        launcher.SIM.D.refine(hopper_utils.GONIO_THETA_ID)
        launcher.SIM.D.refine(hopper_utils.GONIO_PHI_ID)

    for i, diffbragg_id in enumerate(PAN_XYZ_IDS):
        if not params.geometry.fix.panel_translations[i]:
            launcher.SIM.D.refine(diffbragg_id)

    # do a barrel roll!
    target = Target(LMP, save_state_freq=params.geometry.save_state_freq, overwrite_state=params.geometry.save_state_overwrite)
    fcn_args = (launcher.Modelers, launcher.SIM, params)

    # Evaluate initial sigZ before optimization and update geom_start snapshots
    if should_snapshot(params, 0, 0):  # any snapshot requested at all
        _init_f, _, _init_sigZ = target_and_grad(
            target.x0, LMP, launcher.Modelers, launcher.SIM, params, iternum=0)
        # target_and_grad already does reduce+bcast, all ranks have same values
        if COMM.rank == 0:
            print("Geometry initial sigZ=%.6f, resid=%.6g" % (_init_sigZ, _init_f), flush=True)
        # Update geom_start snapshots with initial sigZ
        _snap_dir = os.path.join(params.outdir, "state_snapshots")
        for i_shot in launcher.Modelers:
            if should_snapshot(params, i_shot, COMM.rank):
                _snap_name = "geom_start_rank%d_shot%d" % (COMM.rank, i_shot)
                _snap_path = os.path.join(_snap_dir, _snap_name + ".json")
                if os.path.exists(_snap_path):
                    import json as _json
                    with open(_snap_path) as _fh:
                        _sdata = _json.load(_fh)
                    _sdata["diagnostics"] = {"initial_sigZ": _init_sigZ, "initial_resid": _init_f}
                    with open(_snap_path, "w") as _fh:
                        _json.dump(_sdata, _fh, indent=2)

    lbfgs_kws = {"jac": target.jac,
                 "method": "L-BFGS-B",
                 "args": fcn_args,
                 "options":  {"ftol": params.ftol, "gtol": 1e-10, "maxfun":1e5, "maxiter":params.lbfgs_maxiter}}

    result = basinhopping(target, target.x0[target.vary],
                          niter=params.niter,
                          minimizer_kwargs=lbfgs_kws,
                          T=params.temp,
                          take_step=TakeStep,
                          callback=target.at_min_callback,
                          disp=False,
                          stepsize=params.stepsize)

    target.x0[target.vary] = result.x
    Xopt = target.x0  # optimized, rescaled parameters

    # Print final optimized geometry parameters
    if COMM.rank == 0:
        print("\n" + "="*80)
        print("GEOMETRY REFINEMENT COMPLETE:")
        print("="*80)
        print("Final optimization result:")
        print("  Success: %s" % result.message if hasattr(result, 'message') else 'N/A')
        print("  Final residual: %.2f" % result.fun if hasattr(result, 'fun') else 'N/A')
        print("\nFinal detector parameters:")
        target._print_panel_stats(Xopt, prefix="  ")

        # Geometric interpretation (single-panel only)
        group_ids = target._get_panel_group_ids()
        if len(group_ids) == 1:
            gid = group_ids[0]

            def _get_val(suffix, scale):
                pname = "group%d_%s" % (gid, suffix)
                if pname in LMP:
                    return LMP[pname].get_val(Xopt[LMP[pname].xpos]) * scale
                return 0.0

            rot_orth_deg = _get_val("RotOrth", 180.0 / np.pi)
            rot_fast_deg = _get_val("RotFast", 180.0 / np.pi)
            rot_slow_deg = _get_val("RotSlow", 180.0 / np.pi)
            shift_x_mm = _get_val("ShiftX", 1000.0)
            shift_y_mm = _get_val("ShiftY", 1000.0)

            print("\n  Geometric interpretation:")
            total_tilt = np.sqrt(rot_fast_deg**2 + rot_slow_deg**2)
            print("    Total detector tilt: %.4f deg" % total_tilt)
            print("    Pitch (RotFast): %.4f deg" % rot_fast_deg)
            print("    Yaw (RotSlow): %.4f deg" % rot_slow_deg)
            print("    Roll (RotOrth): %.4f deg" % rot_orth_deg)
            beam_shift_mag = np.sqrt(shift_x_mm**2 + shift_y_mm**2)
            print("\n  Beam center shift:")
            print("    X shift: %.4f mm (%.2f pixels @ 0.075mm/pix)" % (shift_x_mm, shift_x_mm/0.075))
            print("    Y shift: %.4f mm (%.2f pixels @ 0.075mm/pix)" % (shift_y_mm, shift_y_mm/0.075))
            print("    Total shift: %.4f mm (%.2f pixels)" % (beam_shift_mag, beam_shift_mag/0.075))

        # Goniometer axis results
        if "gonio_theta" in LMP and not LMP["gonio_theta"].fix:
            gt = LMP["gonio_theta"]
            gp = LMP["gonio_phi"]
            theta_init = gt.init
            phi_init = gp.init
            theta_final = gt.get_val(Xopt[gt.xpos])
            phi_final = gp.get_val(Xopt[gp.xpos])
            axis_init = GoniometerParameters.spherical_to_cartesian(theta_init, phi_init)
            axis_final = GoniometerParameters.spherical_to_cartesian(theta_final, phi_final)
            # Angular change between initial and final axis
            dot = sum(a*b for a, b in zip(axis_init, axis_final))
            dot = min(1.0, max(-1.0, dot))
            angle_change_deg = np.degrees(np.arccos(dot))
            print("\nGoniometer axis refinement:")
            print("  Initial: (%.6f, %.6f, %.6f)  theta=%.4f deg, phi=%.4f deg"
                  % (axis_init[0], axis_init[1], axis_init[2],
                     np.degrees(theta_init), np.degrees(phi_init)))
            print("  Final:   (%.6f, %.6f, %.6f)  theta=%.4f deg, phi=%.4f deg"
                  % (axis_final[0], axis_final[1], axis_final[2],
                     np.degrees(theta_final), np.degrees(phi_final)))
            print("  Axis change: %.4f deg" % angle_change_deg)
        print("="*80 + "\n", flush=True)

    if params.geometry.optimized_results_tag is not None:
        write_output_files(Xopt, LMP, launcher.Modelers, launcher.SIM, params)

    # Write geom_end state snapshots with sigma Z diagnostics
    # (done here in geom_min where target/result are in scope)
    from simtbx.diffBragg.diffbragg_state import (
        should_snapshot as _should_snap2, capture_geometry_state as _cap_geom2,
        write_state_snapshot as _write_snap2)
    for i_shot in launcher.Modelers:
        if _should_snap2(params, i_shot, COMM.rank):
            _state = _cap_geom2(Xopt, LMP, i_shot, launcher.Modelers[i_shot],
                                launcher.SIM, "geom_end", COMM.rank)
            if hasattr(target, 'first_sigZ') and target.first_sigZ is not None:
                _state["diagnostics"] = {
                    "first_sigZ": target.first_sigZ,
                    "first_resid": target.first_resid,
                    "last_sigZ": target.sigmaZ,
                    "last_resid": float(result.fun) if hasattr(result, 'fun') else None,
                    "n_iterations": target.iternum,
                }
            _write_snap2(_state, params.outdir, "geom_end_rank%d_shot%d" % (COMM.rank, i_shot))

    if COMM.rank == 0:
        save_opt_det(params, target.x0, target.ref_params, launcher.SIM)
        save_opt_sourceI(params, SIM=launcher.SIM)


def save_opt_sourceI(params, SIM, tag="current"):
    if not hasattr(SIM, "sourceI_scales"):
        return
    nsource = len(SIM.beam.xray_beams)
    assert len(SIM.sourceI_scales) == nsource
    sourceI_orig, sourceI_scaled = [],[]
    waves = []
    for i_source in range(nsource):
        #p = LMP[f"sourceI_{i_source}"]
        scale = SIM.sourceI_scales[i_source]
        sourceI = SIM.beam.xray_beams[i_source].get_flux()
        scaled_sourceI = scale * sourceI
        wave = SIM.beam.xray_beams[i_source].get_wavelength() * 1e10
        waves.append(wave)
        sourceI_orig.append(sourceI)
        sourceI_scaled.append(scaled_sourceI)
    sourceI_dir = os.path.join(params.outdir, "sourceI")
    if not os.path.exists(sourceI_dir):
        os.makedirs(sourceI_dir)
    opt_spec_name = os.path.join( sourceI_dir, "iter%s.lam" % tag)
    utils.save_spectra_file(spec_file=opt_spec_name, wavelengths=waves,weights=sourceI_scaled)

    orig_spec_name = os.path.join(sourceI_dir, "iter%s_orig.lam" % tag)
    if not os.path.exists(orig_spec_name):
        utils.save_spectra_file(spec_file=orig_spec_name, wavelengths=waves, weights=sourceI_orig)


def save_opt_Fhkl(params, LMP, SIM, tag='current'):
    assert hasattr(SIM, "Fhkl_scales")
    assert len(SIM.Fhkl_scales) == SIM.Num_ASU*SIM.num_Fhkl_channels
    orig_amps = {h:val for h,val in zip(SIM.crystal.miller_array.indices(), SIM.crystal.miller_array.data())}
    for i_chan in range(SIM.num_Fhkl_channels):
        scaled_amps = flex.double()
        hkl_inds = flex.miller_index()
        for name in LMP:
            if name.startswith("Fhkl"):
                if "channel" in name and "channel%d"%i_chan not in name:
                    continue
                p = LMP[name]
                hkl = tuple(map(int, name.split("_")[1].split(",")))
                assert hkl in orig_amps
                scale = SIM.Fhkl_scales[p.scale_idx]
                scaled_amp = scale*orig_amps[hkl]
                scaled_amps.append(scaled_amp)
                hkl_inds.append(hkl)
        mtz_dir = os.path.join(params.outdir, "Fhkl")
        if not os.path.exists(mtz_dir):
            os.makedirs(mtz_dir)
        mtz_name = os.path.join(mtz_dir, "chan%d_iter%s.mtz" % (i_chan, tag))
        sym = SIM.crystal.miller_array.crystal_symmetry()
        mset = miller.set(sym, hkl_inds, True)
        ma = miller.array(mset, data=scaled_amps).set_observation_type_xray_amplitude()
        ma.as_mtz_dataset(column_root_label="F").mtz_object().write(mtz_name)


def write_output_files(Xopt, LMP, Modelers, SIM, params, iternum=None):
    """
    Writes refl and exper files for each experiment modeled during
    the ensemble refiner
    :param Xopt: float array of optimized rescaled parameter values
    :param LMP: simtbx.diffBragg.refiners.parameters.Parameters() object
    :param Modelers: data modelers (launcher.Modleers
    :param SIM: instance of sim_data (launcher.SIM)
    :param params: phil params, simtbx.diffBragg.phil.py
    """
    opt_det = get_optimized_detector(Xopt, LMP, SIM)
    if COMM.rank==0:
        temp = params.geometry.optimized_detector_name
        params.geometry.optimized_detector_name = os.path.splitext(temp)[0] + "_current.expt"
        if iternum is not None:
            params.geometry.optimized_detector_name = os.path.splitext(temp)[0] + "_%d.expt" % iternum
        save_opt_det(params, Xopt, LMP, SIM)
        params.geometry.optimized_detector_name = temp

        if SIM.refining_Fhkl:
            save_opt_Fhkl(params, LMP, SIM, tag=str(iternum) if iternum is not None else 'current')

        if SIM.refining_sourceI:
            save_opt_sourceI(params, SIM, tag=str(iternum) if iternum is not None else 'current')

    if params.outdir is not None and COMM.rank == 0:
        if not os.path.exists(params.outdir):
            os.makedirs(params.outdir)
        if params.debug_mode:
            refdir = os.path.join(params.outdir, "refls")
            expdir = os.path.join(params.outdir, "expts")
            moddir = os.path.join(params.outdir, "modelers")
            for dname in [refdir, expdir, moddir]:
                if not os.path.exists(dname):
                    os.makedirs(dname)

    COMM.barrier()
    lam0_lam1 = []
    for i_lam in [0,1]:
        lam_p = LMP["lambda%d"% i_lam]
        val = lam_p.get_val(Xopt[lam_p.xpos])
        lam0_lam1.append(val)

    all_shot_pred_offsets = []
    all_dfs = []
    for i_shot in Modelers:
        Modeler = Modelers[i_shot]
        # these are in simtbx.diffBragg.refiners.parameters.RangedParameter objects
        rotX = LMP["rank%d_shot%d_RotXYZ%d" % (COMM.rank, i_shot, 0)]
        rotY = LMP["rank%d_shot%d_RotXYZ%d" % (COMM.rank, i_shot, 1)]
        rotZ = LMP["rank%d_shot%d_RotXYZ%d" % (COMM.rank, i_shot, 2)]
        num_uc_p = len(Modeler.ucell_man.variables)
        ucell_pars = [LMP["rank%d_shot%d_Ucell%d" % (COMM.rank, i_shot, i_uc)] for i_uc in range(num_uc_p)]

        # convert rotation angles back to radians (thats what the parameters.RangedParamter.get_val method does)
        rotXYZ = rotX.get_val(Xopt[rotX.xpos]), \
            rotY.get_val(Xopt[rotY.xpos]), \
            rotZ.get_val(Xopt[rotZ.xpos])

        # ucell_man is an instance of
        # simtbx.diffBragg.refiners.crystal_systems.manager.Manager()
        # (for the correct xtal system)
        Modeler.ucell_man.variables = [p.get_val(Xopt[p.xpos]) for p in ucell_pars]
        ucpar = Modeler.ucell_man.unit_cell_parameters

        new_crystal = hopper_utils.new_cryst_from_rotXYZ_and_ucell(rotXYZ, ucpar, Modeler.E.crystal)
        new_exp = deepcopy(Modeler.E)
        new_exp.crystal = new_crystal
        wave, wt = map(np.array, zip(*Modeler.spectra))
        ave_wave = (wave*wt).sum()/wt.sum()
        new_exp.beam.set_wavelength(ave_wave)
        new_exp.detector = opt_det

        if SIM.refining_Fhkl:
            SIM.update_Fhkl_scales = True
        if SIM.refining_sourceI:
            SIM.update_sourceI_scales = True
        Modeler.best_model = model(Xopt, LMP, i_shot, Modeler, SIM, return_bragg_model=True)
        Modeler.best_model_includes_background = False

        # store the updated per-roi scale factors in the new refl table
        roi_scale_factor = flex.double(len(Modeler.refls), 1)
        for roi_id in Modeler.roi_id_unique:
            p = LMP["rank%d_shot%d_scale_roi%d" % (COMM.rank, i_shot, roi_id)]
            scale_fac = p.get_val(Xopt[p.xpos])
            test_refl_idx = Modeler.refls_idx[roi_id]
            slc = Modeler.roi_id_slices[roi_id][0]
            roi_refl_ids = Modeler.all_refls_idx[slc]
            # NOTE, just a sanity check:
            assert len(np.unique(roi_refl_ids))==1, "unique refl ids"
            refl_idx = roi_refl_ids[0]
            assert test_refl_idx==refl_idx
            roi_scale_factor[refl_idx] = scale_fac
        Modeler.refls["scale_factor"] = roi_scale_factor

        # get the new refls
        new_refl = hopper_utils.get_new_xycalcs(Modeler, new_exp, old_refl_tag="before_geom_ref")

        new_refl_fname, refl_ext = os.path.splitext(Modeler.refl_name)
        new_refl_fname = "rank%d_%s_%s%s" % (COMM.rank, os.path.basename(new_refl_fname), params.geometry.optimized_results_tag, refl_ext)
        if not new_refl_fname.endswith(".refl"):
            new_refl_fname += ".refl"
        new_refl_fname = os.path.join(params.outdir,"refls",  new_refl_fname)
        if params.debug_mode:
            new_refl.as_file(new_refl_fname)
        shot_pred_offsets = get_dist_from_R(new_refl)
        all_shot_pred_offsets += list(shot_pred_offsets)

        new_expt_fname, expt_ext = os.path.splitext(Modeler.exper_name)
        new_expt_fname = "rank%d_%s_%s%s" % (COMM.rank, os.path.basename(new_expt_fname), params.geometry.optimized_results_tag, expt_ext)

        if not new_expt_fname.endswith(".expt"):
            new_expt_fname += ".expt"

        new_expt_fname = os.path.join(params.outdir,"expts", new_expt_fname)
        new_exp_lst = ExperimentList()
        new_exp_lst.append(new_exp)
        if params.debug_mode:
            new_exp_lst.as_file(new_expt_fname)

        if params.outdir is not None:
            a,b,c,al,be,ga = ucpar
            ncells_p = [LMP["rank%d_shot%d_Nabc%d" % (COMM.rank, i_shot, i)] for i in range(3)]
            ncells_def_p = [LMP["rank%d_shot%d_Ndef%d" % (COMM.rank, i_shot, i)] for i in range(3)]
            Na,Nb,Nc = [p.get_val(Xopt[p.xpos]) for p in ncells_p]
            Nd,Ne,Nf = [p.get_val(Xopt[p.xpos]) for p in ncells_def_p]

            eta_p = [LMP["rank%d_shot%d_eta%d" % (COMM.rank, i_shot, i)] for i in range(3)]
            eta_abc = tuple([p.get_val(Xopt[p.xpos]) for p in eta_p])

            scale_p = LMP["rank%d_shot%d_Scale" %(COMM.rank, i_shot)]
            scale = scale_p.get_val(Xopt[scale_p.xpos])

            # Extract Bfactor from geometry params (if present)
            bfac_name = "rank%d_shot%d_Bfactor" % (COMM.rank, i_shot)
            Bfactor_val = None
            if bfac_name in LMP:
                bp = LMP[bfac_name]
                Bfactor_val = bp.get_val(Xopt[bp.xpos])

            # Extract Bfactor_aniso (if present)
            Bfactor_aniso_val = None
            baniso_name = "rank%d_shot%d_Baniso0" % (COMM.rank, i_shot)
            if baniso_name in LMP:
                Bfactor_aniso_val = tuple(
                    LMP["rank%d_shot%d_Baniso%d" % (COMM.rank, i_shot, i)].get_val(
                        Xopt[LMP["rank%d_shot%d_Baniso%d" % (COMM.rank, i_shot, i)].xpos])
                    for i in range(6))

            # Extract diffuse params (if present)
            dgamma_name = "rank%d_shot%d_diffuse_gamma0" % (COMM.rank, i_shot)
            diff_gamma_val = (np.nan, np.nan, np.nan)
            diff_sigma_val = (np.nan, np.nan, np.nan)
            if dgamma_name in LMP:
                diff_gamma_val = tuple(
                    LMP["rank%d_shot%d_diffuse_gamma%d" % (COMM.rank, i_shot, i)].get_val(
                        Xopt[LMP["rank%d_shot%d_diffuse_gamma%d" % (COMM.rank, i_shot, i)].xpos])
                    for i in range(3))
                diff_sigma_val = tuple(
                    LMP["rank%d_shot%d_diffuse_sigma%d" % (COMM.rank, i_shot, i)].get_val(
                        Xopt[LMP["rank%d_shot%d_diffuse_sigma%d" % (COMM.rank, i_shot, i)].xpos])
                    for i in range(3))

            _,fluxes = zip(*SIM.beam.spectrum)
            df= single_expt_pandas(xtal_scale=scale, Amat=new_crystal.get_A(),
                                   ncells_abc=(Na, Nb, Nc), ncells_def=(Nd, Ne, Nf),
                                   eta_abc=eta_abc,
                                   diff_gamma=diff_gamma_val,
                                   diff_sigma=diff_sigma_val,
                                   detz_shift=0,
                                   use_diffuse=params.use_diffuse_models,
                                   gamma_miller_units=params.gamma_miller_units,
                                   eta=np.nan,
                                   rotXYZ=tuple(rotXYZ),
                                   ucell_p = (a,b,c,al,be,ga),
                                   ucell_p_init=(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),
                                   lam0_lam1 = lam0_lam1,
                                   spec_file=Modeler.spec_name,
                                   spec_stride=params.simulator.spectrum.stride,
                                   flux=sum(fluxes), beamsize_mm=SIM.beam.size_mm,
                                   orig_exp_name=Modeler.exper_name,
                                   opt_exp_name=os.path.abspath(new_expt_fname),
                                   spec_from_imageset=params.spectrum_from_imageset,
                                   oversample=SIM.D.oversample,
                                   opt_det=params.opt_det, stg1_refls=Modeler.refl_name, stg1_img_path=None,
                                   Bfactor=Bfactor_val, Bfactor_aniso=Bfactor_aniso_val)

            all_dfs.append(df)

            # optionally save the modeler file
            if params.debug_mode:
                mod_name = os.path.splitext(os.path.basename(new_expt_fname))[0] + ".npy"
                mod_name = os.path.join(params.outdir, "modelers", mod_name)
                np.save(mod_name, Modeler)

    rank_df = pandas.concat(all_dfs)
    pandas_name = os.path.join(params.outdir, "models_rank%d.pkl" % COMM.rank)
    rank_df.to_pickle(pandas_name)

    all_shot_pred_offsets = COMM.reduce(all_shot_pred_offsets)
    if COMM.rank==0:
        median_pred_offset = np.median(all_shot_pred_offsets)
    else:
        median_pred_offset = None
    median_pred_offset = COMM.bcast(median_pred_offset)

    return median_pred_offset


def save_opt_det(phil_params, x, ref_params, SIM):
    opt_det = get_optimized_detector(x, ref_params, SIM)
    El = ExperimentList()
    E = Experiment()
    E.detector = opt_det
    # Save optimized goniometer axis if it was refined
    if "gonio_theta" in ref_params and not ref_params["gonio_theta"].fix:
        from dxtbx.model import Goniometer
        gt = ref_params["gonio_theta"]
        gp = ref_params["gonio_phi"]
        theta_opt = gt.get_val(x[gt.xpos])
        phi_opt = gp.get_val(x[gp.xpos])
        axis_opt = GoniometerParameters.spherical_to_cartesian(theta_opt, phi_opt)
        E.goniometer = Goniometer(axis_opt)
        print("Saved optimized gonio axis: (%.6f, %.6f, %.6f)" % axis_opt)
    El.append(E)
    El.as_file(phil_params.geometry.optimized_detector_name)
    print("Saved detector model to %s" % phil_params.geometry.optimized_detector_name )


def get_optimized_detector(x, ref_params, SIM):
    new_det = Detector()
    for pid in range(len(SIM.detector)):
        panel = SIM.detector[pid]
        panel_dict = panel.to_dict()
        group_id = SIM.panel_group_from_id[pid]
        if group_id in SIM.panel_groups_refined:

            Oang_p = ref_params["group%d_RotOrth" % group_id]
            Fang_p = ref_params["group%d_RotFast" % group_id]
            Sang_p = ref_params["group%d_RotSlow" % group_id]
            Xdist_p = ref_params["group%d_ShiftX" % group_id]
            Ydist_p = ref_params["group%d_ShiftY" % group_id]
            Zdist_p = ref_params["group%d_ShiftZ" % group_id]

            Oang = Oang_p.get_val(x[Oang_p.xpos])
            Fang = Fang_p.get_val(x[Fang_p.xpos])
            Sang = Sang_p.get_val(x[Sang_p.xpos])
            Xdist = Xdist_p.get_val(x[Xdist_p.xpos])
            Ydist = Ydist_p.get_val(x[Ydist_p.xpos])
            Zdist = Zdist_p.get_val(x[Zdist_p.xpos])

            origin_of_rotation = SIM.panel_reference_from_id[pid]
            SIM.D.reference_origin = origin_of_rotation
            SIM.D.update_dxtbx_geoms(SIM.detector, SIM.beam.nanoBragg_constructor_beam, pid,
                                     Oang, Fang, Sang, Xdist, Ydist, Zdist,
                                     force=False)
            fdet = SIM.D.fdet_vector
            sdet = SIM.D.sdet_vector
            origin = SIM.D.get_origin()
        else:
            fdet = panel.get_fast_axis()
            sdet = panel.get_slow_axis()
            origin = panel.get_origin()
        panel_dict["fast_axis"] = fdet
        panel_dict["slow_axis"] = sdet
        panel_dict["origin"] = origin

        new_det.add_panel(Panel.from_dict(panel_dict))

    return new_det
