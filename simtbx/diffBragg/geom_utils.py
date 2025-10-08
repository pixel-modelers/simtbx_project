
import time
import numpy as np
from dxtbx.model import Panel, Detector, Experiment, ExperimentList
from simtbx.diffBragg import psf
from copy import copy

# diffBragg internal indices for derivative manager
ROTXYZ_ID = 0, 1, 2
PAN_O_ID = 14
PAN_F_ID = 17
PAN_S_ID = 18
PAN_X_ID = 15
PAN_Y_ID = 16
PAN_Z_ID = 10
PAN_OFS_IDS = PAN_O_ID, PAN_F_ID, PAN_S_ID
PAN_XYZ_IDS = PAN_X_ID, PAN_Y_ID, PAN_Z_ID

def convolve_model_with_psf(model_pix, SIM, pan_fast_slow, roi_id_slices, roi_id_unique):

    if not SIM.use_psf:
        return model_pix
    PSF = SIM.PSF
    psf_args = SIM.psf_args

    coords = pan_fast_slow.as_numpy_array()
    fid = coords[1::3]
    sid = coords[2::3]

    for i in roi_id_unique:
        for slc in roi_id_slices[i]:
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

    return model_pix

def detector_model_derivs(Modeler, ref_params, SIM, x, reduce=True, scale=1, common_grad_term=1, conv_args=None):
    detector_derivs = []
    Jac = {}
    npix = int(len(Modeler.pan_fast_slow)/3.)
    for diffbragg_parameter_id in PAN_OFS_IDS+PAN_XYZ_IDS:
        try:
            d = SIM.D.get_derivative_pixels(diffbragg_parameter_id).as_numpy_array()[:npix]
            if conv_args is not None:
                d = convolve_model_with_psf(d, **conv_args)
            d = common_grad_term*scale*d
        except ValueError:
            d = None
        detector_derivs.append(d)
    names = "RotOrth", "RotFast", "RotSlow", "ShiftX", "ShiftY", "ShiftZ"
    for group_id in Modeler.unique_panel_group_ids:
        for name in names:
            if reduce:
                Jac["group%d_%s" % (group_id, name)] = 0
            else:
                Jac["group%d_%s" % (group_id, name)] = np.zeros(npix)

        for pixel_rng in Modeler.group_id_slices[group_id]:
            trusted_pixels = Modeler.all_trusted[pixel_rng]
            for i_name, name in enumerate(names):
                par_name = "group%d_%s" % (group_id, name)
                det_param = ref_params[par_name]
                if det_param.fix:
                    continue
                if reduce:
                    pixderivs = detector_derivs[i_name][pixel_rng][trusted_pixels]
                    pixderivs = det_param.get_deriv(x[det_param.xpos], pixderivs)
                    Jac[par_name] += pixderivs.sum()
                else:
                    pixderivs = detector_derivs[i_name][pixel_rng]
                    pixderivs = det_param.get_deriv(x[det_param.xpos], pixderivs)
                    Jac[par_name] += pixderivs
    else:
        return Jac


def update_detector(x, ref_params, SIM, save=None, rank=0, force=False):
    """
    Update the internal geometry of the diffBragg instance
    :param x: refinement parameters as seen by scipy.optimize (e.g. rescaled floats)
    :param ref_params: diffBragg.refiners.Parameters (dict of RangedParameters)
    :param SIM: SIM instance (instance of nanoBragg.sim_data.SimData)
    :param save: optional name to save the detector
    """
    det = SIM.detector
    # NOTE: update_dxtbx_geoms does not preserve spindle_axis!
    spindle_vec = copy(SIM.D.spindle_axis)
    if save is not None:
        new_det = Detector()
    for pid in range(len(det)):
        panel = det[pid]
        panel_dict = panel.to_dict()

        group_id = SIM.panel_group_from_id[pid]
        if group_id not in SIM.panel_groups_refined:
            fdet = panel.get_fast_axis()
            sdet = panel.get_slow_axis()
            origin = panel.get_origin()
        else:

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
                                     force=force)
            fdet = SIM.D.fdet_vector
            sdet = SIM.D.sdet_vector
            origin = SIM.D.get_origin()

        if save is not None:
            panel_dict["fast_axis"] = fdet
            panel_dict["slow_axis"] = sdet
            panel_dict["origin"] = origin
            new_det.add_panel(Panel.from_dict(panel_dict))
    SIM.D.spindle_axis = spindle_vec

    if save is not None and rank==0:
        t = time.time()
        El = ExperimentList()
        E = Experiment()
        E.detector = new_det
        El.append(E)
        El.as_file(save)
        t = time.time()-t
        #print("Saved detector model to %s (took %.4f sec)" % (save, t), flush=True )
