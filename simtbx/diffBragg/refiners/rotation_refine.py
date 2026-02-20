"""
Multi-image rotation crystallography refinement module.

Refines a single crystal model against multiple rotation images simultaneously.
Crystal orientation (RotXYZ), unit cell, and mosaic properties are shared across
all frames, while intensity scale and B-factor are per-image.

Modeled on geometry.py's multi-image refinement pattern.
"""
from __future__ import division, print_function
import time
import numpy as np
from scipy.optimize import minimize
import logging

from mpi4py import MPI
COMM = MPI.COMM_WORLD

from simtbx.diffBragg import utils, hopper_utils
from simtbx.diffBragg.refiners.parameters import RangedParameter, Parameters
from simtbx.diffBragg.geom_utils import (
    convolve_model_with_psf,
    detector_model_derivs, PAN_OFS_IDS, PAN_XYZ_IDS, update_detector)

MAIN_LOGGER = logging.getLogger("diffBragg.main")


class RotationCrystalParameters:
    """Creates shared crystal params + per-shot scale/Bfactor params
    for multi-frame rotation refinement.

    Shared (global) parameters:
      - RotXYZ{0,1,2}: crystal orientation perturbations
      - chol_L{ij}: Cholesky decomposition of mosaic block size (6 params)
      - eta{0,1,2}: mosaic spread
      - Ucell{0..N}: unit cell
    Per-shot parameters:
      - shot{i}_Scale: intensity scale factor
      - shot{i}_Bfactor: B-factor (currently fixed)
    """

    def __init__(self, phil_params, data_modelers, first_modeler_key=None):
        self.phil = phil_params
        self.parameters = []

        # Use the first modeler's PAR values as initial crystal parameters
        if first_modeler_key is None:
            first_modeler_key = next(iter(data_modelers))
        Mod0 = data_modelers[first_modeler_key]

        # --- Shared crystal orientation params ---
        # fix.RotXYZ is a list [fix_x, fix_y, fix_z], index per axis
        fix_RotXYZ = self.phil.fix.RotXYZ
        for i_rot in range(3):
            p = Mod0.PAR.RotXYZ_params[i_rot]
            ref_p = RangedParameter(
                name="RotXYZ%d" % i_rot,
                minval=p.minval, maxval=p.maxval,
                fix=fix_RotXYZ[i_rot],
                init=p.init, sigma=p.sigma,
                center=p.center, beta=p.beta,
                is_global=True)
            self.parameters.append(ref_p)

        # --- Shared Cholesky mosaic block params ---
        use_cholesky = getattr(Mod0, 'use_cholesky_Nabc', False)
        if use_cholesky and hasattr(Mod0.PAR, '__getitem__'):
            # Use the cholesky params from the first modeler
            chol_names = ["chol_L11", "chol_L21", "chol_L22",
                          "chol_L31", "chol_L32", "chol_L33"]
            try:
                for cname in chol_names:
                    p = Mod0.PAR[cname]
                    ref_p = RangedParameter(
                        name=cname,
                        minval=p.minval, maxval=p.maxval,
                        fix=self.phil.fix.Nabc,
                        init=p.init, sigma=p.sigma,
                        center=p.center, beta=p.beta,
                        is_global=True)
                    self.parameters.append(ref_p)
            except (KeyError, AttributeError):
                use_cholesky = False

        if not use_cholesky:
            # Initialize Cholesky from diagonal Nabc values
            Na_init = Mod0.PAR.Nabc[0].init
            Nb_init = Mod0.PAR.Nabc[1].init
            Nc_init = Mod0.PAR.Nabc[2].init
            chol_init = [np.sqrt(max(Na_init, 0.01)), 0,
                         np.sqrt(max(Nb_init, 0.01)),
                         0, 0,
                         np.sqrt(max(Nc_init, 0.01))]
            chol_names = ["chol_L11", "chol_L21", "chol_L22",
                          "chol_L31", "chol_L32", "chol_L33"]
            for ii in range(6):
                ref_p = RangedParameter(
                    name=chol_names[ii],
                    minval=-100, maxval=100,
                    fix=self.phil.fix.Nabc,
                    init=chol_init[ii], sigma=1,
                    center=None, beta=None,
                    is_global=True)
                self.parameters.append(ref_p)

        # --- Shared eta (mosaic spread) params ---
        for i_eta in range(3):
            p = Mod0.PAR.eta[i_eta]
            ref_p = RangedParameter(
                name="eta%d" % i_eta,
                minval=p.minval, maxval=p.maxval,
                fix=True,  # Fixed initially for POC
                init=p.init, sigma=p.sigma,
                center=p.center, beta=p.beta,
                is_global=True)
            self.parameters.append(ref_p)

        # --- Shared unit cell params ---
        num_uc = len(Mod0.PAR.ucell)
        for i_uc in range(num_uc):
            p = Mod0.PAR.ucell[i_uc]
            ref_p = RangedParameter(
                name="Ucell%d" % i_uc,
                minval=p.minval, maxval=p.maxval,
                fix=True,  # Fixed initially for POC
                init=p.init, sigma=p.sigma,
                center=p.center, beta=p.beta,
                is_global=True)
            self.parameters.append(ref_p)

        # --- Per-shot Scale and Bfactor params ---
        for i_shot in data_modelers:
            Mod = data_modelers[i_shot]
            p = Mod.PAR.Scale
            ref_p = RangedParameter(
                name="shot%d_Scale" % i_shot,
                minval=p.minval, maxval=p.maxval,
                fix=self.phil.fix.G,
                init=p.init, sigma=p.sigma,
                center=p.center, beta=p.beta,
                is_global=False)
            self.parameters.append(ref_p)

            # B-factor per shot (fixed for POC)
            ref_p = RangedParameter(
                name="shot%d_Bfactor" % i_shot,
                minval=-100, maxval=200,
                fix=True,
                init=0, sigma=1,
                center=None, beta=None,
                is_global=False)
            self.parameters.append(ref_p)


def rotation_model(x, ref_params, i_shot, Modeler, SIM, phi_deg_for_shot,
                   return_bragg_model=False):
    """Forward model for one rotation frame with shared crystal parameters.

    :param x: rescaled parameter array (global)
    :param ref_params: Parameters() instance
    :param i_shot: shot index
    :param Modeler: DataModeler for i_shot
    :param SIM: SimData instance
    :param phi_deg_for_shot: starting phi angle (degrees) for this frame
    :param return_bragg_model: if True, return just the Bragg model
    :return: (neg_LL, gradient_dict, model_pixels, zscore_sigma) or bragg model
    """
    # --- Extract shared crystal parameters ---
    rotX = ref_params["RotXYZ0"]
    rotY = ref_params["RotXYZ1"]
    rotZ = ref_params["RotXYZ2"]

    chol_names = ["chol_L11", "chol_L21", "chol_L22",
                  "chol_L31", "chol_L32", "chol_L33"]
    chol_params = [ref_params[n] for n in chol_names]

    eta_params = [ref_params["eta%d" % i] for i in range(3)]

    num_uc_p = len(Modeler.ucell_man.variables)
    ucell_pars = [ref_params["Ucell%d" % i_uc] for i_uc in range(num_uc_p)]

    # Per-shot params
    G = ref_params["shot%d_Scale" % i_shot]
    Bfac = ref_params["shot%d_Bfactor" % i_shot]

    # --- Update mosaicity (eta) ---
    if SIM.umat_maker is not None:
        eta_abc = [p.get_val(x[p.xpos]) for p in eta_params]
        SIM.update_umats_for_refinement(eta_abc)

    # --- Update spectrum ---
    SIM.beam.spectrum = Modeler.spectra
    SIM.D.xray_beams = SIM.beam.xray_beams

    # --- Update goniometer with per-frame phi angle ---
    utils.update_SIM_with_gonio(SIM, delta_phi=Modeler.osc_deg,
                                num_phi_steps=Modeler.phisteps,
                                spindle_axis=SIM.D.spindle_axis)
    SIM.D.phi_deg = phi_deg_for_shot  # Override starting angle per frame

    # --- Update unit cell (B matrix) ---
    Modeler.ucell_man.variables = [p.get_val(x[p.xpos]) for p in ucell_pars]
    Bmatrix = Modeler.ucell_man.B_recipspace
    SIM.D.Bmatrix = Bmatrix
    for i_ucell in range(len(ucell_pars)):
        SIM.D.set_ucell_derivative_matrix(
            i_ucell + hopper_utils.UCELL_ID_OFFSET,
            Modeler.ucell_man.derivative_matrices[i_ucell])

    # --- Update crystal orientation (Umat + RotXYZ) ---
    SIM.D.Umatrix = Modeler.PAR.Umatrix
    SIM.D.set_value(hopper_utils.ROTX_ID, rotX.get_val(x[rotX.xpos]))
    SIM.D.set_value(hopper_utils.ROTY_ID, rotY.get_val(x[rotY.xpos]))
    SIM.D.set_value(hopper_utils.ROTZ_ID, rotZ.get_val(x[rotZ.xpos]))

    # --- Cholesky mosaic block: convert L params to NABC ---
    L11, L21, L22, L31, L32, L33 = [p.get_val(x[p.xpos]) for p in chol_params]
    Na = L11 * L11
    Nb = L21 * L21 + L22 * L22
    Nc = L31 * L31 + L32 * L32 + L33 * L33
    Nd = L11 * L21
    Ne = L21 * L31 + L22 * L32
    Nf = L11 * L31
    SIM.D.set_ncells_values((Na, Nb, Nc))
    SIM.D.Ncells_def = (Nd, Ne, Nf)

    # --- Per-image B-factor ---
    Bfac_val = Bfac.get_val(x[Bfac.xpos])
    SIM.D.Bfactor_image = Bfac_val

    # --- Compute forward model ---
    npix = int(len(Modeler.pan_fast_slow) / 3.)
    SIM.D.add_diffBragg_spots(Modeler.pan_fast_slow)

    bragg_no_scale = SIM.D.raw_pixels_roi[:npix].as_numpy_array()
    scale = G.get_val(x[G.xpos])
    bragg = bragg_no_scale * scale

    if return_bragg_model:
        return bragg

    # --- Total forward model ---
    model_pix = bragg + Modeler.all_background
    if SIM.use_psf:
        model_pix = convolve_model_with_psf(
            model_pix, SIM, Modeler.pan_fast_slow,
            roi_id_slices=Modeler.roi_id_slices,
            roi_id_unique=Modeler.roi_id_unique)

    # --- Negative log-likelihood ---
    resid = Modeler.all_data - model_pix
    resid_square = resid ** 2
    V = model_pix + Modeler.nominal_sigma_rdout ** 2
    neg_LL = (0.5 * (np.log(2 * np.pi * V) + resid_square / V))[Modeler.all_trusted].sum()

    # --- Diagnostic z-score ---
    Modeler.all_zscore = resid / np.sqrt(V)
    zscore_sigma = np.std(Modeler.all_zscore[Modeler.all_trusted])

    # --- Gradient computation ---
    J = {}
    common_grad_term = 0.5 / V * (1 - 2 * resid - resid_square / V)
    conv_args = {
        "SIM": SIM, "pan_fast_slow": Modeler.pan_fast_slow,
        "roi_id_slices": Modeler.roi_id_slices,
        "roi_id_unique": Modeler.roi_id_unique
    }

    # Scale factor gradient
    if not G.fix:
        scale_grad = G.get_deriv(x[G.xpos], bragg_no_scale)
        scale_grad = convolve_model_with_psf(scale_grad, **conv_args)
        J[G.name] = (common_grad_term * scale_grad)[Modeler.all_trusted].sum()

    # B-factor gradient
    if not Bfac.fix:
        Bfac_grad = scale * SIM.D.get_Bfactor_derivative_pixels().as_numpy_array()[:npix]
        Bfac_grad = Bfac.get_deriv(x[Bfac.xpos], Bfac_grad)
        Bfac_grad = convolve_model_with_psf(Bfac_grad, **conv_args)
        J[Bfac.name] = (common_grad_term * Bfac_grad)[Modeler.all_trusted].sum()

    # RotXYZ gradients
    for i_rot, rot in enumerate([rotX, rotY, rotZ]):
        if not rot.fix:
            rot_db_id = hopper_utils.ROTXYZ_IDS[i_rot]
            rot_grad = scale * SIM.D.get_derivative_pixels(rot_db_id).as_numpy_array()[:npix]
            rot_grad = rot.get_deriv(x[rot.xpos], rot_grad)
            rot_grad = convolve_model_with_psf(rot_grad, **conv_args)
            J[rot.name] = (common_grad_term * rot_grad)[Modeler.all_trusted].sum()

    # Cholesky mosaic block gradients (chain rule from kernel dI/dNa..dI/dNf)
    if not chol_params[0].fix:
        Nabc_grads = SIM.D.get_ncells_derivative_pixels()
        Ndef_grads = SIM.D.get_ncells_def_derivative_pixels()
        dI_dNa = scale * Nabc_grads[0][:npix].as_numpy_array()
        dI_dNb = scale * Nabc_grads[1][:npix].as_numpy_array()
        dI_dNc = scale * Nabc_grads[2][:npix].as_numpy_array()
        dI_dNd = scale * Ndef_grads[0][:npix].as_numpy_array()
        dI_dNe = scale * Ndef_grads[1][:npix].as_numpy_array()
        dI_dNf = scale * Ndef_grads[2][:npix].as_numpy_array()

        # Chain rule: NABC = L^T*L
        # Na=L11^2, Nb=L21^2+L22^2, Nc=L31^2+L32^2+L33^2
        # Nd=L11*L21, Ne=L21*L31+L22*L32, Nf=L11*L31
        dI_dL = [
            dI_dNa * 2 * L11 + dI_dNd * L21 + dI_dNf * L31,     # dI/dL11
            dI_dNd * L11 + dI_dNb * 2 * L21 + dI_dNe * L31,     # dI/dL21
            dI_dNb * 2 * L22 + dI_dNe * L32,                      # dI/dL22
            dI_dNf * L11 + dI_dNe * L21 + dI_dNc * 2 * L31,     # dI/dL31
            dI_dNe * L22 + dI_dNc * 2 * L32,                      # dI/dL32
            dI_dNc * 2 * L33,                                      # dI/dL33
        ]
        for i_chol in range(6):
            p = chol_params[i_chol]
            chol_grad = p.get_deriv(x[p.xpos], dI_dL[i_chol])
            chol_grad = convolve_model_with_psf(chol_grad, **conv_args)
            J[p.name] = (common_grad_term * chol_grad)[Modeler.all_trusted].sum()

    # eta (mosaic spread) gradients
    if not eta_params[0].fix:
        if SIM.D.has_anisotropic_mosaic_spread:
            eta_abc_derivs = SIM.D.get_aniso_eta_deriv_pixels()
        else:
            eta_abc_derivs = [SIM.D.get_derivative_pixels(hopper_utils.ETA_ID)]
        for i_eta, eta in enumerate(eta_params):
            eta_grad = scale * eta_abc_derivs[i_eta][:npix].as_numpy_array()
            eta_grad = eta.get_deriv(x[eta.xpos], eta_grad)
            eta_grad = convolve_model_with_psf(eta_grad, **conv_args)
            J[eta.name] = (common_grad_term * eta_grad)[Modeler.all_trusted].sum()
            if not SIM.D.has_anisotropic_mosaic_spread:
                break

    # Unit cell gradients
    if not ucell_pars[0].fix:
        for i_ucell, uc_p in enumerate(ucell_pars):
            d = scale * SIM.D.get_derivative_pixels(
                hopper_utils.UCELL_ID_OFFSET + i_ucell).as_numpy_array()[:npix]
            d = uc_p.get_deriv(x[uc_p.xpos], d)
            d = convolve_model_with_psf(d, **conv_args)
            J[uc_p.name] = (common_grad_term * d)[Modeler.all_trusted].sum()

    # Detector gradients (if detector params are present)
    try:
        det_Jac = detector_model_derivs(
            Modeler, ref_params, SIM, x,
            scale=scale, common_grad_term=common_grad_term, conv_args=conv_args)
        for key in det_Jac:
            J[key] = det_Jac[key]
    except (KeyError, AttributeError):
        pass  # No detector params in this refinement

    return neg_LL, J, model_pix, zscore_sigma


def rotation_target_and_grad(x, ref_params, data_modelers, SIM, phi_angles,
                             use_restraints=False, iternum=0):
    """Accumulates target functional and gradients across all rotation frames.

    :param x: parameter array
    :param ref_params: Parameters() instance
    :param data_modelers: dict of DataModeler, keyed by shot index
    :param SIM: SimData instance
    :param phi_angles: dict mapping shot index -> phi_deg starting angle
    :param use_restraints: whether to apply parameter restraints
    :param iternum: iteration number
    :return: (target_functional, gradient_array, median_sigmaZ)
    """
    target_functional = 0
    grad = np.zeros(len(x))

    # Update detector geometry (once before shot loop)
    try:
        update_detector(x, ref_params, SIM, save=None, rank=COMM.rank)
    except (KeyError, AttributeError):
        pass  # No detector params

    all_shot_sigZ = []
    for i_shot in data_modelers:
        Modeler = data_modelers[i_shot]
        phi_deg = phi_angles[i_shot]

        neg_LL, neg_LL_grad, model_pix, per_shot_sigZ = rotation_model(
            x, ref_params, i_shot, Modeler, SIM, phi_deg)
        all_shot_sigZ.append(per_shot_sigZ)

        # Accumulate target functional
        target_functional += neg_LL

        # Per-shot restraints
        if use_restraints:
            for name in ref_params:
                par = ref_params[name]
                if not par.is_global and not par.fix and par.beta is not None:
                    target_functional += par.get_restraint_val(x[par.xpos])

        # Accumulate gradients
        for name in ref_params:
            if name in neg_LL_grad:
                par = ref_params[name]
                grad[par.xpos] += neg_LL_grad[name]
                if use_restraints and not par.is_global and not par.fix and par.beta is not None:
                    grad[par.xpos] += par.get_restraint_deriv(x[par.xpos])

    # MPI reduction
    target_functional = COMM.bcast(COMM.reduce(target_functional))
    grad = COMM.bcast(COMM.reduce(grad))

    # Global restraints
    if use_restraints:
        for name in ref_params:
            par = ref_params[name]
            if par.is_global and not par.fix and par.beta is not None:
                target_functional += par.get_restraint_val(x[par.xpos])
                grad[par.xpos] += par.get_restraint_deriv(x[par.xpos])

    all_shot_sigZ = COMM.reduce(all_shot_sigZ)
    if COMM.rank == 0:
        all_shot_sigZ = np.median(all_shot_sigZ)

    return target_functional, grad, all_shot_sigZ


class RotationTarget:
    """L-BFGS-B wrapper for rotation refinement target function."""

    def __init__(self, ref_params, phi_angles):
        num_params = len(ref_params)
        self.vary = np.zeros(num_params).astype(bool)
        for p in ref_params.values():
            self.vary[p.xpos] = not p.fix
        self.x0 = np.ones(num_params)
        self.g = None
        self.ref_params = ref_params
        self.phi_angles = phi_angles
        self.iternum = 0
        self.all_times = []

    def __call__(self, x, *args, **kwargs):
        self.iternum += 1
        t = time.time()
        self.x0[self.vary] = x

        data_modelers, SIM, use_restraints = args

        f, self.g, self.sigmaZ = rotation_target_and_grad(
            self.x0, self.ref_params, data_modelers, SIM,
            self.phi_angles, use_restraints=use_restraints,
            iternum=self.iternum)

        t = time.time() - t
        if COMM.rank == 0:
            self.all_times.append(t)
            time_per_iter = np.mean(self.all_times)
            print("Iteration %d: Resid=%f, sigmaZ=%f, t=%.4f sec"
                  % (self.iternum, f, self.sigmaZ, time_per_iter), flush=True)
        return f

    def jac(self, x, *args):
        if self.g is not None:
            return self.g[self.vary]


def rotation_refine(data_modelers, SIM, phil_params, phi_angles,
                    use_restraints=False, max_iter=100, ftol=1e-10):
    """Entry point for multi-frame rotation refinement.

    :param data_modelers: dict of DataModeler instances keyed by shot index
    :param SIM: SimData instance (shared)
    :param phil_params: phil parameters
    :param phi_angles: dict mapping shot index -> phi starting angle (degrees)
    :param use_restraints: whether to apply restraints
    :param max_iter: maximum L-BFGS-B iterations
    :param ftol: function tolerance for convergence
    :return: (optimized_x, ref_params, target)
    """
    # Build parameter set
    crystal_params = RotationCrystalParameters(phil_params, data_modelers)
    LMP = Parameters()
    for p in crystal_params.parameters:
        LMP.add(p)

    # Configure diffBragg for gradient computation
    # fix.RotXYZ is a list [fix_x, fix_y, fix_z]
    for i_rot in range(3):
        if not phil_params.fix.RotXYZ[i_rot]:
            SIM.D.refine(hopper_utils.ROTXYZ_IDS[i_rot])
    if not phil_params.fix.Nabc:
        SIM.D.refine(hopper_utils.NCELLS_ID)
        SIM.D.refine(hopper_utils.NCELLS_ID_OFFDIAG)
    if not phil_params.fix.eta_abc:
        SIM.D.refine(hopper_utils.ETA_ID)
    # Get number of ucell params from the first modeler's ucell_man
    first_mod_key = next(iter(data_modelers))
    num_ucell_param = len(data_modelers[first_mod_key].ucell_man.variables)
    if not phil_params.fix.ucell:
        for i_ucell in range(num_ucell_param):
            SIM.D.refine(hopper_utils.UCELL_ID_OFFSET + i_ucell)

    # Allocate pixels for diffBragg
    max_npix = max(int(len(data_modelers[k].pan_fast_slow) / 3.)
                   for k in data_modelers)
    SIM.D.Npix_to_allocate = max_npix

    # Set up target and optimizer
    target = RotationTarget(LMP, phi_angles)
    fcn_args = (data_modelers, SIM, use_restraints)

    result = minimize(
        target, target.x0[target.vary],
        jac=target.jac,
        method="L-BFGS-B",
        args=fcn_args,
        options={"ftol": ftol, "gtol": 1e-10,
                 "maxfun": int(1e5), "maxiter": max_iter})

    target.x0[target.vary] = result.x
    Xopt = target.x0

    if COMM.rank == 0:
        print("Refinement converged: %s (niter=%d, nfev=%d)"
              % (result.message, result.nit, result.nfev), flush=True)

    return Xopt, LMP, target
