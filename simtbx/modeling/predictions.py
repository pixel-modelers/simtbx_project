from __future__ import division

from simtbx.modeling.forward_models import model_spots_from_pandas
from simtbx.diffBragg.utils import refls_to_hkl, refls_from_sims
from dials.array_family import flex
from scipy.spatial import cKDTree, distance
from dials.algorithms.shoebox import MaskCode
SIGNAL_MASK = MaskCode.Valid + MaskCode.Foreground
import numpy as np
from numpy import logical_or as logi_or
from numpy import logical_and as logi_and
from dxtbx.model import ExperimentList
from numpy import logical_not as logi_not


def get_predicted_from_pandas(df, params, strong, eid, device_Id=0, spectrum_override=None):
    """
    :param df: pandas dataframe, stage1_df attribute of simtbx.command_line.hopper_process.HopperProcess
    :param params: instance of diffBragg/phil.py phil params
    :param strong: strong (observed) reflections
    :param eid: experiment identifier
    :param device_Id: GPU device Id for simulating forward model
    :param spectrum_override: the X-ray spectra to use during prediction
    :return: predicted reflections table , to be passed along to dials.integrate functions
    """
    mtz_file = mtz_col = None
    defaultF = params.predictions.default_Famplitude
    if params.predictions.use_diffBragg_mtz:
        mtz_file = params.simulator.structure_factors.mtz_name
        mtz_col = params.simulator.structure_factors.mtz_column
        defaultF = 0
    # returns the images and the experiment including any pre-modeling modifications (e.g. thinning out the detector)
    panel_images, expt = model_spots_from_pandas(
        df,
        oversample_override=params.predictions.oversample_override,
        Ncells_abc_override=params.predictions.Nabc_override,
        pink_stride_override=params.predictions.pink_stride_override,
        spectrum_override=spectrum_override,
        defaultF=defaultF,
        device_Id=device_Id,
        mtz_file=mtz_file, mtz_col=mtz_col,
        d_max=params.predictions.resolution_range[1],
        d_min=params.predictions.resolution_range[0],
        symbol_override=params.predictions.symbol_override,
        force_no_detector_thickness=params.simulator.detector.force_zero_thickness,
        use_exascale_api=params.predictions.method == "exascale",
        use_db=params.predictions.method == "diffbragg")

    predictions = refls_from_sims(panel_images, expt.detector, expt.beam, thresh=params.predictions.threshold,
                                  max_spot_size=1000)
    print("Found %d Bragg peak predictions above the threshold" %len(predictions))

    # TODO: pulled these from comparing to a normal stills_process prediction table, not sure what they imply
    # TODO: multiple experiments per shot
    predictions['flags'] = flex.size_t(len(predictions), 1)
    predictions['id'] = flex.int(len(predictions), 0)
    predictions['entering'] = flex.bool(len(predictions), False)
    predictions['delpsical.rad'] = flex.double(len(predictions), 0)
    if eid:
        predictions.experiment_identifiers()[0] = eid

    El = ExperimentList()
    El.append(expt)
    predictions.centroid_px_to_mm(El)
    predictions.map_centroids_to_reciprocal_space(El)

    strong.centroid_px_to_mm(El)
    strong.map_centroids_to_reciprocal_space(El)

    refls_to_hkl(predictions, expt.detector, expt.beam, expt.crystal, update_table=True)
    predictions['xyzcal.px'] = predictions['xyzobs.px.value']
    predictions['xyzcal.mm'] = predictions['xyzobs.mm.value']
    predictions["num_pixels"] = numpix = predictions["shoebox"].count_mask_values(SIGNAL_MASK)
    predictions['scatter'] = predictions["intensity.sum.value"] / flex.double(np.array(numpix, np.float64))

    # separate out the weak from the strong
    label_weak_predictions(predictions, strong)
    n_weak = sum(predictions["is_weak"])
    n_pred = len(predictions)
    n_strong = n_pred - n_weak
    print("%d / %d predicted refls are near strongs" % (n_strong, n_pred))

    label_weak_spots_for_integration(params.predictions.weak_fraction, predictions)
    print("Will use %d spots for integration" % sum(predictions["is_for_integration"]))
    predictions = predictions.select(predictions["is_for_integration"])

    return predictions, panel_images


def label_weak_predictions(predictions, strong, q_cutoff=0.005):
    """
    :param predictions: model reflection table
    :param strong: strong observed spots (reflection table)
    :param q_cutoff: distance in RLP space - arbitrary, increasing will bring in more candidates, 0.005 seems reasonable
    """
    strong_tree = cKDTree(strong['rlp'])
    predicted_tree = cKDTree(predictions['rlp'])

    # for each strong refl, find all predictions within q_cutoff of the strong rlp
    pred_idx_candidates = strong_tree.query_ball_tree(predicted_tree, q_cutoff)

    is_weak = flex.bool(len(predictions), True)
    xyz_obs = [(-1,-1,-1)]*len(predictions)
    for i_idx, cands in enumerate(pred_idx_candidates):
        if not cands:
            continue
        if len(cands) == 1:
            # if 1 spot is within q_cutoff , then its the closest
            pred_idx = cands[0]
        else:
            # in this case there are multiple predictions near the strong refl, we choose the closest one
            dists = []
            for c in cands:
                d = distance.euclidean(strong_tree.data[i_idx], predicted_tree.data[c])
                dists.append(d)
            pred_idx = cands[np.argmin(dists)]
        is_weak[pred_idx] = False
        xyz_obs[pred_idx] = strong["xyzobs.px.value"][i_idx]
    predictions["is_weak"] = is_weak
    predictions["orig.xyzobs.px"] = flex.vec3_double(xyz_obs)


def label_weak_spots_for_integration(fraction, predictions, num_res_bins=10):
    """
    :param fraction: fraction of reflections to label as "integratable" within each resolution bin
    :param predictions: dials.flex.reflections_table
    :param num_res_bins: number of resolution bins
    """
    res = 1. / np.linalg.norm(predictions["rlp"], axis=1)
    res_sort = np.sort(res)
    res_bins = [rb[0]-1e-6 for rb in np.array_split( res_sort, num_res_bins)] + [res_sort[-1]+1e-6]
    res_bin_assigments = np.digitize(res, res_bins)
    is_weak_but_integratable = np.zeros(len(predictions)).astype(np.bool)
    for i_res in range(1, num_res_bins+1):
        # grab weak spots in this res bin
        is_weak_and_in_bin = logi_and(res_bin_assigments == i_res, predictions["is_weak"])
        refls_in_bin = predictions.select(flex.bool(is_weak_and_in_bin))

        # determine which weak spots are closer to the Ewald sphere, based on the scatter value
        signal_cutoff_in_bin = np.percentile(refls_in_bin['scatter'], (1.-fraction)*100)
        is_above_cutoff = predictions["scatter"] > signal_cutoff_in_bin

        is_integratable = logi_and(is_weak_and_in_bin, is_above_cutoff)
        is_weak_but_integratable[is_integratable] = True
    predictions['is_for_integration'] = flex.bool(logi_or(logi_not(predictions["is_weak"]), is_weak_but_integratable))


def normalize_by_partiality(refls, model, default_F=1, gain=1):
    """
    :param refls: integrated refls output by the integrator.integrate() method in hopper_process / stills_process
    :param model: prediction intensities, output by the model_spots_from_pandas method
    :param default_F: value of the default structure factor used in the predictive model
    :param gain: detector ADU to photon factor
    :return: updated reflection table
    """
    nref = len(refls)
    F2 = default_F**2
    partials = flex.double()
    new_Isum = flex.double()
    new_Ivar = flex.double()
    for i_ref in range(nref):
        refl = refls[i_ref]
        sb = refl['shoebox']
        pid = refl['panel']
        mask = sb.mask.as_numpy_array()[0]
        data = sb.data.as_numpy_array()[0] / gain
        x1,x2,y1,y2,_,_ = sb.bbox
        was_integrated = mask == SIGNAL_MASK
        Y, X = np.where(was_integrated)
        Y += y1
        X += x1
        par = model[pid, Y, X] / F2
        good = par > 0
        corrected = (data[was_integrated] / par)[good]

        par_sum = par.sum()
        data_sum = corrected.sum()
        data_var = corrected.std()**2
        partials.append(par_sum)
        new_Isum.append(data_sum)
        new_Ivar.append(data_var)
    refls['dials.intensity.sum.value'] = refls['intensity.sum.value']
    refls['dials.intensity.sum.variance'] = refls['intensity.sum.variance']
    refls['diffBragg_partials'] = partials
    refls['intensity.sum.value'] = new_Isum
    refls['intensity.sum.variance'] = new_Ivar
    return refls
