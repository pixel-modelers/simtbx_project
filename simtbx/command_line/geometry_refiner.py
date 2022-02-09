from __future__ import print_function, division

# LIBTBX_SET_DISPATCHER_NAME diffBragg.geometry_refiner

import numpy as np
from simtbx.command_line.hopper import single_expt_pandas
from copy import deepcopy
import os
from libtbx.mpi4py import  MPI
COMM = MPI.COMM_WORLD

import logging
MAIN_LOGGER = logging.getLogger("diffBragg.main")

from dxtbx.model import Experiment, ExperimentList
#from simtbx.diffBragg.command_line.hopper import save_to_pandas
from dxtbx.model import Detector, Panel
from simtbx.diffBragg import hopper_utils, ensemble_refine_launcher
from simtbx.diffBragg.refiners.parameters import RangedParameter, Parameters
import pandas
from scipy.optimize import basinhopping

# diffBragg internal indices for derivative manager
ROTXYZ_ID = hopper_utils.ROTXYZ_IDS
PAN_O_ID = 14
PAN_F_ID = 17
PAN_S_ID = 18
PAN_X_ID = 15
PAN_Y_ID = 16
PAN_Z_ID = 10
PAN_OFS_IDS = PAN_O_ID, PAN_F_ID, PAN_S_ID
PAN_XYZ_IDS = PAN_X_ID, PAN_Y_ID, PAN_Z_ID

DEG_TO_PI = np.pi/180.


class DetectorParameters:

    def __init__(self, phil_params, panel_groups_refined, num_panel_groups ):

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
                                sigma=1,  # TODO
                                minval=GEO.min.panel_rotations[0]*DEG_TO_PI,
                                maxval=GEO.max.panel_rotations[0]*DEG_TO_PI,
                                fix=not vary_rots[0], center=0, beta=GEO.betas.panel_rot[0], is_global=True)

            f = RangedParameter(name="group%d_RotFast" % i_group,
                                init=0,
                                sigma=1,  # TODO
                                minval=GEO.min.panel_rotations[1]*DEG_TO_PI,
                                maxval=GEO.max.panel_rotations[1]*DEG_TO_PI,
                                fix=not vary_rots[1], center=0, beta=GEO.betas.panel_rot[1],
                                is_global=True)

            s = RangedParameter(name="group%d_RotSlow" % i_group,
                                init=0,
                                sigma=1,  # TODO
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
        for i_shot in data_modelers:
            Mod = data_modelers[i_shot]

            for i_N in range(3):
                p = Mod.PAR.Nabc[i_N]
                ref_p = RangedParameter(name="rank%d_shot%d_Nabc%d" % (COMM.rank, i_shot, i_N),
                                          minval=p.minval, maxval=p.maxval, fix=self.phil.fix.Nabc, init=p.init,
                                          center=p.center, beta=p.beta)
                self.parameters.append(ref_p)

            for i_rot in range(3):
                p = Mod.PAR.RotXYZ_params[i_rot]
                ref_p = RangedParameter(name="rank%d_shot%d_RotXYZ%d" % (COMM.rank, i_shot, i_rot),
                                          minval=p.minval, maxval=p.maxval, fix=self.phil.fix.RotXYZ, init=p.init,
                                          center=p.center, beta=p.beta)
                self.parameters.append(ref_p)

            p = Mod.PAR.Scale
            ref_p = RangedParameter(name="rank%d_shot%d_Scale" % (COMM.rank, i_shot),
                                      minval=p.minval, maxval=p.maxval, fix=self.phil.fix.G, init=p.init,
                                      center=p.center, beta=p.beta)
            self.parameters.append(ref_p)

            for i_uc in range(len(Mod.PAR.ucell)):
                p = Mod.PAR.ucell[i_uc]
                ref_p = RangedParameter(name="rank%d_shot%d_Ucell%d" % (COMM.rank, i_shot, i_uc),
                                          minval=p.minval, maxval=p.maxval, fix=self.phil.fix.ucell, init=p.init,
                                          center=p.center, beta=p.beta)
                self.parameters.append(ref_p)


class Target:
    def __init__(self, ref_params):
        num_params = len(ref_params)
        self.vary = np.zeros(num_params).astype(bool)
        for p in ref_params.values():
            self.vary[p.xpos] = not p.fix
        self.x0 = np.ones(num_params)
        self.g = None
        self.ref_params = ref_params
        self.iternum = 0

    def __call__(self, x, *args, **kwargs):
        self.iternum += 1
        self.x0[self.vary] = x

        f, self.g, self.sigmaZ = target_and_grad(self.x0, self.ref_params, *args, **kwargs)
        if COMM.rank==0:
            print("Iteration %d:\n\tResid=%f, sigmaZ %f" % (self.iternum, f, self.sigmaZ))
        return f

    def jac(self, x, *args):
        if self.g is not None:
            return self.g[self.vary]

    def at_min_callback(self, x, f, accept):
        if COMM.rank==0:
            print("Final Iteration %d:\n\tResid=%f, sigmaZ %f" % (self.iternum, f, self.sigmaZ))


def model(x, ref_params, i_shot, Modeler, SIM, return_model=False):
    """

    :param x: rescaled parameter array (global)
    :param ref_params: simtbx.diffBragg.refiners.parameters.Parameters() instance
    :param i_shot: shot index for this data model,
        the simtbx.diffBragg.refiners.parameters.RangerParameter objs stored in ref_params
        have names which include i_shot
    :param Modeler: DataModeler for i_shot
    :param SIM: instance of sim_data.SimData
    :param return_model: if true, bypass the latter half of the method and return the Bragg scattering model
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
    G = ref_params["rank%d_shot%d_Scale" % (COMM.rank, i_shot)]
    num_uc_p = len(Modeler.ucell_man.variables)
    ucell_pars = [ref_params["rank%d_shot%d_Ucell%d" % (COMM.rank, i_shot, i_uc)] for i_uc in range(num_uc_p)]

    # update the photon energy spectrum for this shot
    SIM.beam.spectrum = Modeler.spectra
    SIM.D.xray_beams = SIM.beam.xray_beams

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

    npix = int(len(Modeler.pan_fast_slow)/3.)

    # calculate the forward Bragg scattering and gradients
    SIM.D.add_diffBragg_spots(Modeler.pan_fast_slow)
    bragg_no_scale = (SIM.D.raw_pixels_roi[:npix]).as_numpy_array()

    # apply the per-shot scale factor
    scale = G.get_val(x[G.xpos])
    bragg = scale*bragg_no_scale

    model_pix = bragg + Modeler.all_background
    if return_model:
        return model_pix

    # compute the negative log Likelihood
    resid = (Modeler.all_data - model_pix)
    resid_square = resid ** 2
    V = model_pix + Modeler.sigma_rdout ** 2
    neg_LL = (.5*(np.log(2*np.pi*V) + resid_square / V))[Modeler.all_trusted].sum()

    # compute the z-score sigma as a diagnostic
    zscore_sigma = np.std((resid / np.sqrt(V))[Modeler.all_trusted])

    # store the gradients
    J = {}
    # this term is a common factor in all of the gradients
    common_grad_term = (0.5 / V * (1 - 2 * resid - resid_square / V))

    # scale factor gradients
    if not G.fix:
        scale_grad = G.get_deriv(x[G.xpos], bragg_no_scale)
        J[G.name] = (common_grad_term*scale_grad)[Modeler.all_trusted].sum()

    # Umat gradients
    for i_rot, rot in enumerate([rotX, rotY, rotZ]):
        if not rot.fix:
            rot_db_id = ROTXYZ_ID[i_rot]
            rot_grad = scale*SIM.D.get_derivative_pixels(rot_db_id).as_numpy_array()[:npix]
            rot_grad = rot.get_deriv(x[rot.xpos], rot_grad)
            J[rot.name] = (common_grad_term*rot_grad)[Modeler.all_trusted].sum()

    # mosaic block size gradients
    if not Na.fix:
        Nabc_grad = SIM.D.get_ncells_derivative_pixels()
        for i_N, N in enumerate([Na, Nb, Nc]):
            N_grad = scale*(Nabc_grad[i_N][:npix].as_numpy_array())
            N_grad = N.get_deriv(x[N.xpos], N_grad)
            J[N.name] = (common_grad_term*N_grad)[Modeler.all_trusted].sum()

    # unit cell gradients
    if not ucell_pars[0].fix:
        for i_ucell, uc_p in enumerate(ucell_pars):
            d = scale*SIM.D.get_derivative_pixels(hopper_utils.UCELL_ID_OFFSET+i_ucell).as_numpy_array()[:npix]
            d = uc_p.get_deriv(x[uc_p.xpos], d)
            J[ucell_pars[i_ucell].name] = (common_grad_term*d)[Modeler.all_trusted].sum()

    # detector model gradients
    detector_derivs = []
    for diffbragg_parameter_id in PAN_OFS_IDS+PAN_XYZ_IDS:
        try:
            d = common_grad_term*scale*(SIM.D.get_derivative_pixels(diffbragg_parameter_id).as_numpy_array()[:npix])
        except ValueError:
            d = None
        detector_derivs.append(d)
    names = "RotOrth", "RotFast", "RotSlow", "ShiftX", "ShiftY", "ShiftZ"
    for group_id in Modeler.unique_panel_group_ids:
        for name in names:
            J["group%d_%s" % (group_id, name)] = 0
        for pixel_rng in Modeler.group_id_slices[group_id]:
            trusted_pixels = Modeler.all_trusted[pixel_rng]
            for i_name, name in enumerate(names):
                par_name = "group%d_%s" % (group_id, name)
                det_param = ref_params[par_name]
                if det_param.fix:
                    continue
                pixderivs = detector_derivs[i_name][pixel_rng][trusted_pixels]
                pixderivs = det_param.get_deriv(x[det_param.xpos], pixderivs)
                J[par_name] += pixderivs.sum()

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


def update_detector(x, ref_params, SIM):
    """
    Update the internal geometry of the diffBragg instance
    :param x: refinement parameters as seen by scipy.optimize (e.g. rescaled floats)
    :param ref_params: diffBragg.refiners.Parameters (dict of RangedParameters)
    :param SIM: SIM instance (instance of nanoBragg.sim_data.SimData)
    """
    det = SIM.detector
    for pid in range(len(det)):
        group_id = SIM.panel_group_from_id[pid]
        if group_id not in SIM.panel_groups_refined:
            continue
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
        SIM.D.update_dxtbx_geoms(det, SIM.beam.nanoBragg_constructor_beam, pid,
                                  Oang, Fang, Sang, Xdist, Ydist, Zdist,
                                  force=False)


def target_and_grad(x, ref_params, data_modelers, SIM, params):
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

    update_detector(x, ref_params, SIM)

    all_shot_sigZ = []
    for i_shot in data_modelers:
        Modeler = data_modelers[i_shot]

        neg_LL, neg_LL_grad, model_pix, per_shot_sigZ = model(x, ref_params, i_shot, Modeler, SIM)
        all_shot_sigZ.append(per_shot_sigZ)

        # accumulate the target functional for this rank/shot
        target_functional += neg_LL

        if params.use_restraints:
            for name in ref_params:
                par = ref_params[name]
                if not par.is_global and not par.fix:
                    val = par.get_restraint_val(x[par.xpos])
                    target_functional += val

        # accumulate the gradients for this rank/shot
        for name in ref_params:
            if name in neg_LL_grad:
                par = ref_params[name]
                grad[par.xpos] += neg_LL_grad[name]
                # for restraints only update the per-shot restraint gradients here
                if params.use_restraints and not par.is_global and not par.fix:
                    grad[par.xpos] += par.get_restraint_deriv(x[par.xpos])

    # sum the target functional and the gradients across all ranks
    target_functional = COMM.bcast(COMM.reduce(target_functional))
    grad = COMM.bcast(COMM.reduce(grad))

    if params.use_restraints and params.geometry.betas.close_distances is not None:
        target_functional += np.std(SIM.D.close_distances) / params.geometry.betas.close_distances

    ## add in the detector parameter restraints
    if params.use_restraints:
        for name in ref_params:
            par = ref_params[name]
            if par.is_global and not par.fix:
                target_functional += par.get_restraint_val(x[par.xpos])
                grad[par.xpos] += par.get_restraint_deriv(x[par.xpos])

    all_shot_sigZ = COMM.reduce(all_shot_sigZ)
    if COMM.rank == 0:
        all_shot_sigZ = np.median(all_shot_sigZ)

    return target_functional, grad, all_shot_sigZ


def geom_min(params):
    """
    :param params: phil parameters (simtbx/diffBragg/phil.py)
    """

    launcher = ensemble_refine_launcher.RefineLauncher(params)
    df = pandas.read_pickle(params.geometry.input_pkl)
    if params.skip is not None:
        df = df.iloc[params.skip:]
    if params.geometry.first_n is not None:
        df = df.iloc[:params.geometry.first_n]
    #if len(set(df.detz_shift_mm.values))> 1:
    #    s = ""
    #    if COMM.rank==0:
    #        s = "Can only do ensemble refinement when input experiments all have the same detector distance"
    #        s +="\n Suggested protocol: do stage 1 while fixing detz, then do ensemble refinement to get the optimized detector model"
    #        s += "\n Then do another round of stage 1 with optimized detector model, and allow detz shift to refine per shot"
    #    raise NotImplementedError(s)
    if COMM.rank == 0:
        print("Will optimize using %d experiments" %len(df))
    launcher.load_inputs(df, refls_key=params.geometry.refls_key)

    # same on every rank:
    det_params = DetectorParameters(params, launcher.panel_groups_refined, launcher.n_panel_groups)

    # different on each rank
    crystal_params = CrystalParameters(params,launcher.Modelers)
    crystal_params.parameters = COMM.bcast(COMM.reduce(crystal_params.parameters))

    LMP = Parameters()
    for p in crystal_params.parameters + det_params.parameters:
        LMP.add(p)

    for i_shot in launcher.Modelers:
        Modeler = launcher.Modelers[i_shot]
        set_group_id_slices(Modeler, launcher.panel_group_from_id)

    # attached some objects to SIM for convenience
    launcher.SIM.panel_reference_from_id = launcher.panel_reference_from_id
    launcher.SIM.panel_group_from_id = launcher.panel_group_from_id
    launcher.SIM.panel_groups_refined = launcher.panel_groups_refined

    # set the GPU device
    launcher.SIM.D.device_Id = COMM.rank % params.refiner.num_devices
    npx_str = "(rnk%d, dev%d): %d pix" %(COMM.rank, launcher.SIM.D.device_Id, launcher.NPIX_TO_ALLOC)
    npx_str = COMM.gather(npx_str)
    if COMM.rank==0:
        print("How many pixels each rank will allocate for on its device:")
        print("; ".join(npx_str))
    launcher.SIM.D.Npix_to_allocate = launcher.NPIX_TO_ALLOC

    # configure diffBragg instance for gradient computation
    if not params.fix.RotXYZ:
        for i_rot in range(3):
            launcher.SIM.D.refine(ROTXYZ_ID[i_rot])
    if not params.fix.Nabc:
        launcher.SIM.D.refine(hopper_utils.NCELLS_ID)
    if not params.fix.ucell:
        for i_ucell in range(launcher.SIM.num_ucell_param):
            launcher.SIM.D.refine(hopper_utils.UCELL_ID_OFFSET + i_ucell)
    for i, diffbragg_id in enumerate(PAN_OFS_IDS):
        if not params.geometry.fix.panel_rotations[i]:
            launcher.SIM.D.refine(diffbragg_id)

    for i, diffbragg_id in enumerate(PAN_XYZ_IDS):
        if not params.geometry.fix.panel_translations[i]:
            launcher.SIM.D.refine(diffbragg_id)

    # do a barrel roll!
    target = Target(LMP)
    fcn_args = (launcher.Modelers, launcher.SIM, params)
    lbfgs_kws = {"jac": target.jac,
                 "method": "L-BFGS-B",
                 "args": fcn_args,
                 "options":  {"ftol": params.ftol, "gtol": 1e-10, "maxfun":1e5, "maxiter":params.lbfgs_maxiter}}

    result = basinhopping(target, target.x0[target.vary],
                 niter=params.niter,
                 minimizer_kwargs=lbfgs_kws,
                 T=params.temp,
                 callback=target.at_min_callback,
                 disp=False,
                 stepsize=params.stepsize)

    target.x0[target.vary] = result.x
    Xopt = target.x0  # optimized, rescaled parameters

    if params.geometry.optimized_results_tag is not None:
        write_output_files(params, Xopt, LMP, launcher)

    if COMM.rank == 0:
        save_opt_det(params, target.x0, target.ref_params, launcher.SIM)


def write_output_files(params, Xopt, LMP, launcher):
    """
    Writes refl and exper files for each experiment modeled during
    the ensemble refiner
    :param params: phil params, simtbx.diffBragg.phil.py
    :param Xopt: float array of optimized rescaled parameter values
    :param LMP: simtbx.diffBragg.refiners.parameters.Parameters() object
    :param launcher: launcher object , instance of simtbx.diffBragg.ensemble_refine_launcher.RefineLauncher
    """
    opt_det = get_optimized_detector(Xopt, LMP, launcher.SIM)

    if params.geometry.pandas_dir is not None and COMM.rank == 0:
        if not os.path.exists(params.geometry.pandas_dir):
            os.makedirs(params.geometry.pandas_dir)

    for i_shot in launcher.Modelers:
        Modeler = launcher.Modelers[i_shot]
        # these are in simtbx.diffBragg.refiners.parameters.RangedParameter objects
        rotX = LMP["rank%d_shot%d_RotXYZ%d" % (COMM.rank, i_shot, 0)]
        rotY = LMP["rank%d_shot%d_RotXYZ%d" % (COMM.rank, i_shot, 1)]
        rotZ = LMP["rank%d_shot%d_RotXYZ%d" % (COMM.rank, i_shot, 2)]
        num_uc_p = len(Modeler.ucell_man.variables)
        ucell_pars = [LMP["rank%d_shot%d_Ucell%d" % (COMM.rank, i_shot, i_uc)] for i_uc in range(num_uc_p)]

        # convert rotation angles back to radians (thats what the parameters.RangedParamter.get_val method does)
        rotXYZ = rotX.get_val(Xopt[rotX.xpos]),\
                 rotY.get_val(Xopt[rotY.xpos]),\
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

        Modeler.best_model = model(Xopt, LMP, i_shot, Modeler, launcher.SIM, return_model=True)
        Modeler.best_model_includes_background = True
        new_refl = hopper_utils.get_new_xycalcs(Modeler, new_exp, old_refl_tag="before_geom_ref")

        new_refl_fname, refl_ext = os.path.splitext(Modeler.refl_name)
        new_refl_fname = "%s_%s%s" % (new_refl_fname, params.geometry.optimized_results_tag, refl_ext)
        if not new_refl_fname.endswith(".refl"):
            new_refl_fname += ".refl"
        new_refl.as_file(new_refl_fname)

        new_expt_fname, expt_ext = os.path.splitext(Modeler.exper_name)
        new_expt_fname = "%s_%s%s" % (new_expt_fname, params.geometry.optimized_results_tag, expt_ext)
        if not new_expt_fname.endswith(".expt"):
            new_expt_fname += ".expt"
        new_exp_lst = ExperimentList()
        new_exp_lst.append(new_exp)
        new_exp_lst.as_file(new_expt_fname)

        if params.geometry.pandas_dir is not None:
            a,b,c,al,be,ga = ucpar
            ncells_p = [LMP["rank%d_shot%d_Nabc%d" % (COMM.rank, i_shot, i)] for i in range(3)]
            Na,Nb,Nc = [p.get_val(Xopt[p.xpos]) for p in ncells_p]
            scale_p = LMP["rank%d_shot%d_Scale" %(COMM.rank, i_shot)]
            scale = scale_p.get_val(Xopt[scale_p.xpos])

            eta_a = eta_b = eta_c = np.nan
            df= single_expt_pandas(xtal_scale=scale, Amat=new_crystal.get_A(),
                ncells_abc=(Na, Nb, Nc), ncells_def=(0,0,0),
                eta_abc=(eta_a, eta_b, eta_c),
                diff_gamma=(np.nan, np.nan, np.nan),
                diff_sigma=(np.nan, np.nan, np.nan),
                detz_shift=0,
                use_diffuse=params.use_diffuse_models,
                gamma_miller_units=params.gamma_miller_units,
                eta=np.nan,
                rotXYZ=tuple(rotXYZ),
                ucell_p = (a,b,c,al,be,ga),
                ucell_p_init=(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),
                lam0_lam1 = (np.nan, np.nan),
                spec_file=Modeler.spec_name,
                spec_stride=params.simulator.spectrum.stride,
                flux=launcher.SIM.D.flux, beamsize_mm=launcher.SIM.beam.size_mm,
                orig_exp_name=Modeler.exper_name,
                opt_exp_name=os.path.abspath(new_expt_fname),
                spec_from_imageset=params.spectrum_from_imageset,
                oversample=launcher.SIM.D.oversample,
                opt_det=params.opt_det, stg1_refls=Modeler.refl_name, stg1_img_path=None)
            pandas_name = os.path.splitext(os.path.basename(new_expt_fname))[0] + ".pkl"
            pandas_name = os.path.join(params.geometry.pandas_dir, pandas_name)
            df.to_pickle(pandas_name)
            modeler_name = pandas_name.replace(".pkl", ".npy")
            np.save(modeler_name, Modeler)

        print("Wrote files %s and %s" % (new_refl_fname, new_expt_fname))


def save_opt_det(phil_params, x, ref_params, SIM):
    opt_det = get_optimized_detector(x, ref_params, SIM)
    El = ExperimentList()
    E = Experiment()
    E.detector = opt_det
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


if __name__ == "__main__":
    from argparse import ArgumentParser
    from libtbx.phil import parse
    from simtbx.diffBragg.phil import philz, hopper_phil

    parser = ArgumentParser()
    parser.add_argument("--phil", type=str, required=True, help="path to a phil string")
    parser.add_argument("--cmdlinePhil", nargs="+", default=None, type=str, help="command line phil params")
    progargs = parser.parse_args()

    phil_scope = parse(philz+hopper_phil)
    arg_interp = phil_scope.command_line_argument_interpreter(home_scope="")

    phil_file = open(progargs.phil, "r").read()
    user_phil = parse(phil_file)
    phil_sources = [user_phil]

    if progargs.cmdlinePhil is not None:
        command_line_phils = [arg_interp.process(phil) for phil in progargs.cmdlinePhil]
        phil_sources += command_line_phils

    working_phil, unused = phil_scope.fetch(sources=phil_sources, track_unused_definitions=True)
    for loc in unused:
        print("WARNING: unused phil:", loc)

    params = working_phil.extract()
    geom_min(params)
