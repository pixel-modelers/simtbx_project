from __future__ import division
import pandas
from dxtbx.model import Experiment, ExperimentList
from copy import deepcopy
import logging
from simtbx.diffBragg import hopper_utils, utils
from scitbx.matrix import sqr , col
import os
import numpy as np


def _apply_detz_shift(detector, detz_shift_m):
    """
    Apply a per-image detector z-shift (in meters) to a dxtbx detector.
    Returns a new detector with shifted panel origins.
    The shift is along each panel's normal direction.
    """
    from dxtbx.model import Detector, Panel
    new_det = Detector()
    for panel in detector:
        pdict = panel.to_dict()
        origin = list(panel.get_origin())
        normal = panel.get_normal()
        # detz_shift is in meters, dxtbx origins are in mm
        shift_mm = detz_shift_m * 1e3
        origin[0] += normal[0] * shift_mm
        origin[1] += normal[1] * shift_mm
        origin[2] += normal[2] * shift_mm
        pdict["origin"] = tuple(origin)
        new_det.add_panel(Panel.from_dict(pdict))
    return new_det


def save_expt_refl_file(filename, expts, refls, specs=None, check_exists=False, indices=None):
    """
    Save an input file for bg_and_probOri (the EMC initializer script)
    expt and refl names will be given absolute paths
    :param filename: input expt_refl name to be written (passable to script bg_and_probOri.py)
    :param expts: list of experiments
    :param refls: list of reflection tables
    :param specs: optional list of spectrum .lam files
    :param check_exists: ensure files actually exist
    :param indices: experiment indices if multiple images per experiment
    :return:
    """
    if specs is None:
        specs = [None]*len(expts)
    if indices is None:
        indices = [None] *len(expts)
    with open(filename, "w") as o:
        for expt, refl, spec, idx in zip(expts, refls, specs, indices):
            expt = os.path.abspath(expt)
            refl = os.path.abspath(refl)
            if spec is not None:
                spec = os.path.abspath(spec)
            if check_exists:
                assert os.path.exists(expt)
                assert os.path.exists(refl)
                if spec is not None:
                    assert os.path.exists(spec)
            if spec is None:
                spec = ""
            else:
                spec = " %s" %spec
            if idx is None:
                idx = ""
            else:
                idx = " %d" % idx
            o.write("%s %s%s%s\n" % (expt, refl, spec, idx))


def make_rank_outdir(root, subfolder, rank=0):
    rank_imgs_outdir = os.path.join(root, subfolder, "rank%d" % rank)
    if not os.path.exists(rank_imgs_outdir):
        os.makedirs(rank_imgs_outdir)
    return rank_imgs_outdir


def diffBragg_Umat(rotX, rotY, rotZ, U):
    xax = col((-1, 0, 0))
    yax = col((0, -1, 0))
    zax = col((0, 0, -1))
    ## update parameters:
    RX = xax.axis_and_angle_as_r3_rotation_matrix(rotX, deg=False)
    RY = yax.axis_and_angle_as_r3_rotation_matrix(rotY, deg=False)
    RZ = zax.axis_and_angle_as_r3_rotation_matrix(rotZ, deg=False)
    M = RX * RY * RZ
    U = M * sqr(U)
    return U


def save_to_pandas(x, Mod, SIM, orig_exp_name, params, expt, rank_exp_idx, stg1_refls, stg1_img_path=None,
                   rank=0, write_expt=True, write_pandas=True, exp_idx=0):
    """

    :param x: the optiized parameters used by hopper (output of Minimize)
    :param Mod: the instance of the hopper_utils.DataModeler that was used by hopper
    :param SIM: the instance of nanoBragg/sim_data.SimData that was used by hopper
    :param orig_exp_name: the name of the experiment list that was input to hopper
    :param params: the diffBragg hopper parameters
    :param expt: the data modeler experiment
    :param rank_exp_idx: order this shot was processed by this MPI rank #TODO rename this
    :param stg1_refls: path to the refls that were input to hopper
    :param stg1_img_path: leave as None, no longer used
    :param rank: MPI rank
    :param write_expt: whether to write the single shot experiment
    :param write_pandas: whether to write the single shot dataframe
    :param exp_idx: the index of the experiment within the experiment list (orig_exp_name)
    :return: the single shot dataframe
    """
    LOGGER = logging.getLogger("refine")
    opt_exp_path = None
    basename = os.path.splitext(os.path.basename(orig_exp_name))[0]
    if write_expt:
        rank_exper_outdir = make_rank_outdir(params.outdir, "expers",rank)
        opt_exp_path = os.path.join(rank_exper_outdir, "%s_%s_%d.expt" % (params.tag, basename, rank_exp_idx))

    scale, rotX, rotY, rotZ, Na, Nb, Nc, Nd, Ne, Nf,\
        diff_gam_a, diff_gam_b, diff_gam_c, diff_sig_a, \
        diff_sig_b, diff_sig_c, a,b,c,al,be,ga,detz_shift, gonio_angle, Bfactor = \
        hopper_utils.get_param_from_x(x, Mod)

    scale_p = Mod.P["G_xtal0"]
    scale_init = scale_p.init

    Nabc_init = []
    for i in [0,1,2]:
        p = Mod.P["Nabc%d" % i]
        Nabc_init.append(p.init)
    Nabc_init = tuple(Nabc_init)

    if params.isotropic.diffuse_gamma:
        diff_gam_b = diff_gam_c = diff_gam_a
    if params.isotropic.diffuse_sigma:
        diff_sig_b = diff_sig_c = diff_sig_a

    if params.simulator.crystal.has_isotropic_ncells:
        Nb = Nc = Na

    eta_a, eta_b, eta_c = hopper_utils.get_mosaicity_from_x(x, Mod, SIM)
    a_init, b_init, c_init, al_init, be_init, ga_init = SIM.crystal.dxtbx_crystal.get_unit_cell().parameters()

    U = diffBragg_Umat(rotX, rotY, rotZ, SIM.crystal.dxtbx_crystal.get_U())
    new_cryst = deepcopy(SIM.crystal.dxtbx_crystal)
    new_cryst.set_U(U)

    ucparam = a, b, c, al, be, ga
    ucman = utils.manager_from_params(ucparam)
    new_cryst.set_B(ucman.B_recipspace)

    Amat = new_cryst.get_A()
    other_Umats = []
    other_spotscales = []
    if Mod.num_xtals > 1:
        for i_xtal in range(1,Mod.num_xtals,1):
            par = hopper_utils.get_param_from_x(x, Mod, i_xtal=i_xtal, as_dict=True)
            scale_xt = par['scale']
            rotX_xt = par['rotX']
            rotY_xt = par['rotY']
            rotZ_xt = par['rotZ']
            U_xt = diffBragg_Umat(rotX_xt, rotY_xt, rotZ_xt, SIM.Umatrices[i_xtal])
            other_Umats.append(U_xt)
            other_spotscales.append(scale_xt)

    eta = [0]
    lam_coefs = [0], [1]
    if hasattr(Mod, "P"):
        names = "lambda_offset", "lambda_scale"
        if names[0] in Mod.P and names[1] in Mod.P:
            lam_coefs = []
            for name in names:
                if name in Mod.P:
                    p = Mod.P[name]
                    val = p.get_val(x[p.xpos])
                    lam_coefs.append([val])
            lam_coefs = tuple(lam_coefs)

    new_expt = Experiment()
    new_expt.crystal = new_cryst
    new_expt.detector = expt.detector
    if Mod.P.refining_detector:
        from simtbx.diffBragg.refiners.geometry import get_optimized_detector
        optD = get_optimized_detector(x, Mod.P, SIM)
        new_expt.detector = optD
    elif detz_shift != 0:
        # Bake per-image detz_shift into the detector so geometry sees correct distance
        new_expt.detector = _apply_detz_shift(expt.detector, detz_shift)
    new_expt.beam = expt.beam
    new_expt.identifier = expt.identifier
    new_expt.imageset = expt.imageset
    # expt.detector = refiner.get_optimized_detector()
    new_exp_list = ExperimentList()
    new_exp_list.append(new_expt)
    if write_expt:
        new_exp_list.as_file(opt_exp_path)
        LOGGER.debug("saved opt_exp %s with wavelength %f" % (opt_exp_path, expt.beam.get_wavelength()))
    _,flux_vals = zip(*SIM.beam.spectrum)

    # Extract Bfactor_aniso values if refined
    Bfactor_aniso = None
    if hasattr(Mod, 'P') and "Baniso0" in Mod.P:
        Baniso_params = [Mod.P["Baniso%d" % i] for i in range(6)]
        Bfactor_aniso = tuple(p.get_val(x[p.xpos]) for p in Baniso_params)

    # Extract per-ROI scale factors keyed by ASU HKL for MTZ update propagation
    perRoiScale = None
    if not params.fix.perRoiScale or params.use_perRoiScale:
        if hasattr(Mod, 'roi_id_unique') and hasattr(Mod, 'Hi_asu'):
            perRoiScale = {}
            for roi_id in Mod.roi_id_unique:
                pname = "scale_roi%d" % roi_id
                if pname in Mod.P:
                    p = Mod.P[pname]
                    scale_val = float(p.get_val(x[p.xpos]))
                    slc = Mod.roi_id_slices[roi_id][0]
                    hkl = tuple(Mod.hi_asu_perpix[slc.start])
                    perRoiScale[hkl] = scale_val

    df = single_expt_pandas(xtal_scale=scale, Amat=Amat,
        ncells_abc=(Na, Nb, Nc), ncells_def=(Nd,Ne,Nf),
        eta_abc=(eta_a, eta_b, eta_c),
        diff_gamma=(diff_gam_a, diff_gam_b, diff_gam_c),
        diff_sigma=(diff_sig_a, diff_sig_b, diff_sig_c),
        detz_shift=detz_shift,
        use_diffuse=params.use_diffuse_models,
        gamma_miller_units=params.gamma_miller_units,
        eta=eta,
        rotXYZ=(rotX, rotY, rotZ),
        ucell_p = (a,b,c,al,be,ga),
        ucell_p_init=(a_init, b_init, c_init, al_init, be_init, ga_init),
        lam0_lam1 = lam_coefs,
        spec_file=params.simulator.spectrum.filename,
        spec_stride=params.simulator.spectrum.stride,
        flux=sum(flux_vals), beamsize_mm=SIM.beam.size_mm,
        orig_exp_name=orig_exp_name, opt_exp_name=opt_exp_path,
        spec_from_imageset=params.spectrum_from_imageset,
        oversample=params.simulator.oversample,
        opt_det=params.opt_det, stg1_refls=stg1_refls,
        stg1_img_path=stg1_img_path,
        ncells_init=Nabc_init, spot_scales_init=scale_init,
        other_Umats = other_Umats, other_spotscales = other_spotscales,
        num_mosaicity_samples=params.simulator.crystal.num_mosaicity_samples,
                            gonio_angle=gonio_angle, Bfactor=Bfactor,
                            Bfactor_aniso=Bfactor_aniso,
                            perRoiScale=perRoiScale)

    df["gonio_axis"] = [SIM.D.spindle_axis]

    if Mod.P.refining_detector:
        for name in ["RotOrth", "RotFast", "RotSlow", "ShiftX", "ShiftY", "ShiftZ"]:
            p = Mod.P["group0_%s" %name]
            val = p.get_val(x[p.xpos])
            df[p.name] = val

    df['exp_idx'] = exp_idx

    if hasattr(Mod, "sigz"):
        df['sigz'] = [Mod.sigz]
    if hasattr(Mod, "niter"):
        df['niter'] = [Mod.niter]
    if hasattr(Mod, "nidx"):
        df['nidx'] = [Mod.nidx]
    df['phi_deg'] = SIM.D.phi_deg
    df['osc_deg'] = SIM.D.osc_deg
    df['phisteps'] = SIM.D.phisteps
    #df["gonio_ax"] = SIM.D.spindle_axis
    if write_pandas:
        rank_pandas_outdir = make_rank_outdir(params.outdir, "pandas",rank)
        pandas_path = os.path.join(rank_pandas_outdir, "%s_%s_%d.pkl" % (params.tag, basename, rank_exp_idx))
        df.to_pickle(pandas_path)
    return df


def single_expt_pandas(xtal_scale, Amat, ncells_abc, ncells_def, eta_abc,
                       diff_gamma, diff_sigma, detz_shift, use_diffuse, gamma_miller_units, eta,
                       rotXYZ, ucell_p, ucell_p_init, lam0_lam1,
                       spec_file, spec_stride,flux, beamsize_mm,
                       orig_exp_name, opt_exp_name, spec_from_imageset, oversample,
                       opt_det, stg1_refls, stg1_img_path, ncells_init=None, spot_scales_init = None,
                       other_Umats=None, other_spotscales=None, num_mosaicity_samples=None, gonio_angle=None, Bfactor=None,
                       Bfactor_aniso=None, perRoiScale=None):
    """

    :param xtal_scale:
    :param Amat:
    :param ncells_abc:
    :param ncells_def:
    :param eta_abc:
    :param diff_gamma:
    :param diff_sigma:
    :param detz_shift:
    :param use_diffuse:
    :param gamma_miller_units:
    :param eta:
    :param rotXYZ:
    :param ucell_p:
    :param ucell_p_init:
    :param lam0_lam1:
    :param spec_file:
    :param spec_stride:
    :param flux:
    :param beamsize_mm:
    :param orig_exp_name:
    :param opt_exp_name:
    :param spec_from_imageset:
    :param oversample:
    :param opt_det:
    :param stg1_refls:
    :param stg1_img_path:
    :param num_mosaicity_samples:
    :param gonio_angle:
    :return:
    """
    if other_Umats is None:
        other_Umats = []
    if other_spotscales is None:
        other_spotscales = []
    if ncells_init is None:
        ncells_init = np.nan, np.nan, np.nan
    if spot_scales_init is None:
        spot_scales_init = np.nan
    a,b,c,al,be,ga = ucell_p
    a_init, b_init, c_init, al_init, be_init, ga_init = ucell_p_init
    lam0,lam1 = lam0_lam1
    df = pandas.DataFrame({
        "spot_scales": [xtal_scale], "Amats": [Amat], "ncells": [ncells_abc],
        "spot_scales_init": [spot_scales_init],
        "ncells_init": [ncells_init],
        "eta_abc": [eta_abc],
        "detz_shift_mm": [detz_shift * 1e3],
        "ncells_def": [ncells_def],
        "diffuse_gamma": [diff_gamma],
        "diffuse_sigma": [diff_sigma],
        "fp_fdp_shift": [np.nan],
        "use_diffuse_models": [use_diffuse],
        "gamma_miller_units": [gamma_miller_units],
        "eta": eta,
        "rotX": rotXYZ[0],
        "rotY": rotXYZ[1],
        "rotZ": rotXYZ[2],
        "a": a, "b": b, "c": c, "al": al, "be": be, "ga": ga,
        "a_init": a_init, "b_init": b_init, "c_init": c_init, "al_init": al_init,
        "lam0": lam0, "lam1": lam1,
        "be_init": be_init, "ga_init": ga_init})
    if gonio_angle is not None:
        df["gonio_angle"] = gonio_angle
    if Bfactor is not None:
        df["Bfactor"] = Bfactor
    if Bfactor_aniso is not None:
        df["Bfactor_aniso"] = [Bfactor_aniso]
    if perRoiScale is not None:
        df["perRoiScale"] = [perRoiScale]
    if spec_file is not None:
        spec_file = os.path.abspath(spec_file)
    df["spectrum_filename"] = spec_file
    df["spectrum_stride"] = spec_stride
    if other_spotscales:
        df["other_spotscales"] = [tuple(other_spotscales)]
    if other_Umats:
        df["other_Umats"] = [tuple(map(tuple, other_Umats))]
    if num_mosaicity_samples is not None:
        df['num_mosaicity_samples'] = num_mosaicity_samples

    df["total_flux"] = flux
    df["beamsize_mm"] = beamsize_mm
    df["exp_name"] = os.path.abspath(orig_exp_name)

    if opt_exp_name is not None:
        opt_exp_name = os.path.abspath(opt_exp_name)
    df["opt_exp_name"] = opt_exp_name
    df["spectrum_from_imageset"] = spec_from_imageset
    df["oversample"] = oversample
    if opt_det is not None:
        df["opt_det"] = opt_det
    df["stage1_refls"] = stg1_refls
    df["stage1_output_img"] = stg1_img_path
    return df


def update_mtz_with_roi_scales(input_mtz_path, output_mtz_path, pandas_paths,
                                mtz_column=None, damping=1.0):
    """
    Update MTZ structure factors using per-ROI scale corrections from refinement.

    Reads perRoiScale dicts {(h,k,l): scale} from pandas pkls, computes median
    correction per HKL across all shots, and applies:
        F_corrected = F_orig * (median_scale ** (0.5 * damping))

    This folds per-ROI scale corrections into the reference Fhkl so the next
    refinement cycle starts with perRoiScale ~1.0 (self-consistent).

    Args:
        input_mtz_path: path to original MTZ file
        output_mtz_path: path for corrected MTZ output
        pandas_paths: list of pandas pkl paths containing perRoiScale dicts
        mtz_column: column label for reading original MTZ (default: "fobs(+)fobs(-)")
        damping: fraction of correction to apply (0=none, 1=full)

    Returns:
        dict with keys: n_corrected, n_unique_hkl, n_total_mtz, median_correction
    """
    # Collect all HKL -> scale pairs from all shots
    all_hkl_scales = {}  # (h,k,l) -> [scale1, scale2, ...]
    for pkl_path in pandas_paths:
        df = pandas.read_pickle(pkl_path)
        for _, row in df.iterrows():
            if "perRoiScale" not in row or row["perRoiScale"] is None:
                continue
            scales_dict = row["perRoiScale"]
            if not isinstance(scales_dict, dict):
                continue
            for hkl, scale in scales_dict.items():
                hkl = tuple(int(h) for h in hkl)
                if hkl not in all_hkl_scales:
                    all_hkl_scales[hkl] = []
                all_hkl_scales[hkl].append(scale)

    if not all_hkl_scales:
        return {"n_corrected": 0, "n_unique_hkl": 0, "n_total_mtz": 0,
                "median_correction": 1.0}

    # Compute median scale per HKL
    corrections = {}
    for hkl, scales in all_hkl_scales.items():
        corrections[hkl] = np.median(scales)

    median_correction = np.median(list(corrections.values()))

    # Read original MTZ (returns amplitude array)
    original_ma = utils.open_mtz(input_mtz_path, mtz_column)

    # KEEP anomalous data - do NOT merge Bijvoet mates (preserves anomalous signal)
    # if original_ma.anomalous_flag():
    #     original_ma = original_ma.average_bijvoet_mates()

    # Apply corrections to amplitudes: F_new = F_old * (scale ^ (0.5*damping))
    # For anomalous data, Friedel mates can have different corrections (true anomalous signal)
    indices = original_ma.indices()
    data = original_ma.data().deep_copy()
    n_corrected = 0
    for i in range(len(indices)):
        hkl = tuple(indices[i])
        if hkl in corrections:
            corr = corrections[hkl]
            if corr > 0:
                data[i] = data[i] * (corr ** (0.5 * damping))
                n_corrected += 1

    # Create corrected miller array and write MTZ
    corrected_ma = original_ma.customized_copy(data=data)
    mtz_dataset = corrected_ma.as_mtz_dataset(column_root_label="FP")
    mtz_dataset.mtz_object().write(output_mtz_path)

    # Determine output column name (anomalous: "FP(+),SIGFP(+),FP(-),SIGFP(-)", merged: "FP,SIGFP")
    is_anomalous = corrected_ma.anomalous_flag()
    output_column = "FP(+),SIGFP(+),FP(-),SIGFP(-)" if is_anomalous else "FP,SIGFP"

    return {"n_corrected": n_corrected, "n_unique_hkl": len(corrections),
            "n_total_mtz": len(indices), "median_correction": median_correction,
            "is_anomalous": is_anomalous, "mtz_column": output_column}
