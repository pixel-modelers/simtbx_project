#!/usr/bin/env python
"""
DEPRECATED: This module is not used by hopper_cycler. Prediction expansion is
handled in-memory by DataModeler via the prediction_expansion.expand_rois phil
parameter. The --expand-after flag in hopper_cycler.py is the correct approach.
This file is retained for reference only and may be removed in a future release.

---
Original docstring:
Predict new reflections from a refined hopper model and merge with observed.

After a hopper refinement cycle, this script:
  1. Loads refined pandas output from a cycle output directory
  2. For each shot: reconstructs the refined crystal model, runs a forward
     simulation over the full detector, and extracts predicted reflections
  3. Filters predictions using a selectable strategy
  4. Merges new predictions with original observed reflections
  5. Saves expanded .refl files for the next refinement cycle

Filter modes (--filter-mode):
  none            Keep all predictions above the simulation threshold
  intensity       Global intensity percentile (keep top weak_fraction)
  scatter_binned  Resolution-binned scatter percentile (predictions.py approach)
  signal          Data-based I/sqrt(bg) at predicted pixel locations

Use --compare-filters to run all filter modes and dump per-shot statistics
for comparison (useful with known ground-truth simulations).

Usage:
  hopper_predict.py <cycle_outdir> [options]
"""
from __future__ import division, print_function
import sys
import os
import glob
import argparse
import numpy as np
from copy import deepcopy

from dxtbx.model.experiment_list import ExperimentListFactory
from dxtbx.model import ExperimentList
from dials.array_family import flex
from dials.algorithms.shoebox import MaskCode
SIGNAL_MASK = MaskCode.Valid + MaskCode.Foreground

import logging
LOGGER = logging.getLogger("diffBragg.predict")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict reflections from refined hopper model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("cycle_outdir", help="Output directory from a hopper cycle")
    parser.add_argument("--threshold", type=float, default=1e-3,
                        help="Peak intensity threshold for prediction extraction")
    parser.add_argument("--d-min", type=float, default=None,
                        help="Resolution limit (Angstrom) for predictions")
    parser.add_argument("--weak-fraction", type=float, default=0.5,
                        help="Fraction of weak predictions to include (0-1)")
    parser.add_argument("--filter-mode", default="none",
                        choices=["none", "intensity", "scatter_binned", "signal", "score"],
                        help="Prediction filter strategy (default: none)")
    parser.add_argument("--signal-threshold", type=float, default=2.0,
                        help="I/sqrt(bg) threshold for signal filter mode")
    parser.add_argument("--score-state-file", default=None,
                        help="Path to score_trainer .net file for score filter mode")
    parser.add_argument("--score-cutoff", type=float, default=0.5,
                        help="Score threshold for score filter mode (0-1)")
    parser.add_argument("--expand-dir", default="expanded",
                        help="Output subdirectory name for expanded .refl files")
    parser.add_argument("--q-cutoff", type=float, default=0.005,
                        help="Reciprocal-space distance for deduplication (1/Angstrom)")
    parser.add_argument("--input-file", default=None,
                        help="Path to input.txt (auto-detected from cycle_outdir if not given)")
    parser.add_argument("--shoebox-size", type=int, default=10,
                        help="Shoebox size for signal validation ROI")
    parser.add_argument("--compare-filters", action="store_true",
                        help="Run all filter modes and dump comparison stats to CSV")
    parser.add_argument("--save-dials", action="store_true",
                        help="Save DIALS-compatible .expt + .refl for viewing in dials.image_viewer")
    parser.add_argument("--device-id", type=int, default=0,
                        help="GPU device ID for simulation (default: 0)")
    parser.add_argument("--phil", default=None,
                        help="Path to hopper phil file (auto-detected from cycle outdir)")
    parser.add_argument("--signal-check", action="store_true",
                        help="Check raw pixel data at unmatched prediction sites for signal")
    # Kept for backward compat with cycler interface
    parser.add_argument("--signal-filter", action="store_true",
                        help="(Deprecated) Equivalent to --filter-mode=signal")
    return parser.parse_args()


def _load_phil_params(cycle_outdir, phil_arg=None):
    """Load hopper phil params for simulator settings.

    Tries (in order):
      1. Explicit --phil path
      2. hopper.phil in cycle_outdir or its parent
      3. Any *.phil file in cycle_outdir
    Returns a phil params object or None if no phil file found.
    """
    from simtbx.diffBragg import phil as diffBragg_phil

    if phil_arg and os.path.isfile(phil_arg):
        phil_path = phil_arg
    else:
        candidates = [
            os.path.join(cycle_outdir, "hopper.phil"),
            os.path.join(os.path.dirname(cycle_outdir), "hopper.phil"),
        ]
        candidates += sorted(glob.glob(os.path.join(cycle_outdir, "*.phil")))
        phil_path = next((p for p in candidates if os.path.isfile(p)), None)

    if phil_path is None:
        return None

    from libtbx.phil import parse
    user_phil = parse(open(phil_path).read())
    working = diffBragg_phil.phil_scope.fetch(source=user_phil)
    params = working.extract()
    print("Loaded phil from: %s" % phil_path)
    return params


# ===========================================================================
# Data loading
# ===========================================================================

def load_pandas_from_outdir(cycle_outdir):
    """Load all refined pandas DataFrames from a hopper cycle output directory."""
    import pandas
    pd_dir = os.path.join(cycle_outdir, "pandas")
    if not os.path.isdir(pd_dir):
        raise FileNotFoundError("No pandas directory found at %s" % pd_dir)

    # Prefer top-level hopper result pkls (per-shot refined parameters with
    # hopper_line, all refined params).  Fall back to rank*/*.pkl (geometry
    # stage output, typically only 1 combined row).
    pkl_files = sorted(glob.glob(os.path.join(pd_dir, "hopper_results_*.pkl")))
    if not pkl_files:
        pkl_files = sorted(glob.glob(os.path.join(pd_dir, "rank*", "*.pkl")))
    if not pkl_files:
        pkl_files = sorted(glob.glob(os.path.join(pd_dir, "*.pkl")))
    if not pkl_files:
        raise FileNotFoundError("No pickle files found in %s" % pd_dir)

    dfs = []
    for f in pkl_files:
        try:
            df = pandas.read_pickle(f)
            dfs.append(df)
        except Exception as e:
            LOGGER.warning("Skipping %s: %s" % (f, e))

    if not dfs:
        raise ValueError("No valid DataFrames loaded from %s" % pd_dir)

    all_df = pandas.concat(dfs, ignore_index=True)
    print("Loaded %d refined shots from %s" % (len(all_df), pd_dir))
    return all_df


def _load_optimized_detector(cycle_outdir):
    """Load the geometry-optimized detector from a cycle output directory.

    Looks for diffBragg_detector.expt in the cycle outdir (written by geometry
    optimization). Returns a dxtbx Detector or None if not found.
    """
    det_path = os.path.join(cycle_outdir, "diffBragg_detector.expt")
    if not os.path.isfile(det_path):
        # Also check pandas subdirectory
        det_path = os.path.join(cycle_outdir, "pandas", "diffBragg_detector.expt")
    if not os.path.isfile(det_path):
        return None
    try:
        det_El = ExperimentListFactory.from_json_file(det_path, check_format=False)
        if len(det_El) > 0 and det_El[0].detector is not None:
            return det_El[0].detector
    except Exception as e:
        LOGGER.warning("Failed to load optimized detector from %s: %s" % (det_path, e))
    return None


def get_input_lines(cycle_outdir, input_file=None, pandas_df=None):
    """Find and parse the input lines for this cycle.

    Tries, in order:
      1. Explicit --input-file path
      2. input.txt in cycle_outdir or its parent
      3. hopper_line column from the pandas DataFrame (fallback)
    """
    if input_file is not None:
        lines = open(input_file, "r").readlines()
        return [l.strip() for l in lines if l.strip()]

    for candidate in [
        os.path.join(cycle_outdir, "input.txt"),
        os.path.join(os.path.dirname(cycle_outdir), "input.txt"),
    ]:
        if os.path.isfile(candidate):
            lines = open(candidate, "r").readlines()
            return [l.strip() for l in lines if l.strip()]

    # Fallback: extract from pandas hopper_line or exp_name+stage1_refls columns
    if pandas_df is not None:
        if "hopper_line" in pandas_df.columns:
            lines = [str(l).strip() for l in pandas_df.hopper_line.values
                     if str(l).strip()]
            if lines:
                print("Using hopper_line from pandas as input (%d lines)" % len(lines))
                return lines

        if "exp_name" in pandas_df.columns and "stage1_refls" in pandas_df.columns:
            lines = []
            for _, row in pandas_df.iterrows():
                exp = str(row.exp_name).strip()
                refl = str(row.stage1_refls).strip()
                if exp and refl and exp != "nan" and refl != "nan":
                    exp_idx = int(row.exp_idx) if "exp_idx" in pandas_df.columns else 0
                    if exp_idx > 0:
                        lines.append("%s %s %d" % (exp, refl, exp_idx))
                    else:
                        lines.append("%s %s" % (exp, refl))
            if lines:
                print("Using exp_name+stage1_refls from pandas as input (%d lines)"
                      % len(lines))
                return lines

    raise FileNotFoundError(
        "Cannot find input.txt in %s. Provide --input-file explicitly." % cycle_outdir)


# ===========================================================================
# Simulation
# ===========================================================================

def simulate_and_predict(df_row, expt, detector, beam, threshold=1e-3, d_min=None,
                         cuda=False, device_id=0, phil_params=None):
    """Run forward simulation from refined parameters and extract predicted spots.

    Uses model_spots_from_pandas for full parameter support (B-factor, aniso B,
    spectrum, diffuse, eta, gonio) and GPU acceleration.

    Follows the same phil propagation as predictions.get_predicted_from_pandas()
    to ensure identical simulator settings (polarization, detector thickness,
    oversample, etc.).

    Returns (refls, panel_imgs, sim_expt) where sim_expt is the experiment
    used during simulation (has the correct detector model for centroids).
    """
    import pandas
    from simtbx.diffBragg import utils
    from simtbx.diffBragg.utils import refls_from_sims
    from simtbx.modeling.forward_models import model_spots_from_pandas

    # Build a single-row DataFrame for model_spots_from_pandas
    row_dict = df_row.to_dict() if hasattr(df_row, 'to_dict') else dict(df_row)
    df = pandas.DataFrame([row_dict])

    # Use the optimized detector if provided (override the one in opt_exp_name)
    det_override = None
    if detector is not expt.detector:
        import tempfile
        tmp_det = tempfile.NamedTemporaryFile(suffix=".expt", delete=False)
        tmp_det_path = tmp_det.name
        tmp_det.close()
        det_El = ExperimentList()
        from dxtbx.model import Experiment as DxtbxExpt
        det_e = DxtbxExpt()
        det_e.detector = detector
        det_El.append(det_e)
        det_El.as_file(tmp_det_path)
        det_override = tmp_det_path

    # Extract phil settings (matching predictions.get_predicted_from_pandas)
    force_zero_thick = False
    det_thicksteps = None
    oversample_override = None
    nopolar_val = False  # apply polarization (matches hopper and integrate.py)

    if phil_params is not None:
        force_zero_thick = phil_params.simulator.detector.force_zero_thickness
        if hasattr(phil_params, 'predictions'):
            if phil_params.predictions.thicksteps_override is not None:
                det_thicksteps = phil_params.predictions.thicksteps_override
            if phil_params.predictions.oversample_override is not None:
                oversample_override = phil_params.predictions.oversample_override

    try:
        model_out = model_spots_from_pandas(
            df, use_db=True, cuda=cuda, device_Id=device_id,
            d_min=d_min or 1.0, d_max=999,
            defaultF=1e3, quiet=True,
            nopolar=nopolar_val,
            no_Nabc_scale=True,
            force_no_detector_thickness=force_zero_thick,
            det_thicksteps=det_thicksteps,
            oversample_override=oversample_override,
            detector_override=det_override,
            return_sim=True)
    finally:
        if det_override is not None:
            try:
                os.unlink(det_override)
            except OSError:
                pass

    # model_out with return_sim=True: ((panel_imgs, SIM), expt)
    (panel_imgs_raw, SIM), sim_expt = model_out
    panel_imgs = panel_imgs_raw if isinstance(panel_imgs_raw, np.ndarray) else np.array(panel_imgs_raw)

    # Use refls_from_sims (DIALS pixel_list_to_reflection_table) for centroid
    # extraction — same as predictions.get_predicted_from_pandas and integrate.py
    refls = refls_from_sims(panel_imgs, sim_expt.detector, sim_expt.beam,
                            thresh=threshold, max_spot_size=1000)

    # Resolution filter
    if d_min is not None and len(refls) > 0:
        pred_key = 'xyzobs.px.value'
        if pred_key in refls:
            keep = flex.bool(len(refls), True)
            s0 = np.array(beam.get_s0())  # magnitude = 1/wavelength
            for i_ref in range(len(refls)):
                fast, slow, z = refls[pred_key][i_ref]
                pid = int(refls['panel'][i_ref]) if 'panel' in refls else 0
                lab = sim_expt.detector[pid].get_pixel_lab_coord((fast, slow))
                lab_arr = np.array(lab)
                s1 = lab_arr / np.linalg.norm(lab_arr) / beam.get_wavelength()
                d_spacing = 1.0 / max(np.linalg.norm(s1 - s0), 1e-10)
                if d_spacing < d_min:
                    keep[i_ref] = False
            refls = refls.select(keep)

    return refls, panel_imgs, sim_expt


def _extract_peaks(panel_imgs, detector, beam, threshold=1e-3,
                    d_min=None, min_conn=3, max_conn=1000):
    """Extract predicted peaks from simulated panel images.

    Uses the same centroid convention as hopper: intensity-weighted center of
    mass of the Bragg signal with +0.5 pixel-center offset.  This matches
    how hopper computes xyzcal.px (hopper_utils.py:1738-1755).
    """
    from scipy.ndimage import measurements

    xyzobs = flex.vec3_double()
    panels = flex.size_t()
    intensities = flex.double()
    bboxes = flex.int6()

    for i_pan in range(len(detector)):
        img = panel_imgs[i_pan]
        mask = img > threshold
        if not np.any(mask):
            continue

        lab_img, n_labels = measurements.label(mask)
        if n_labels == 0:
            continue

        obs = measurements.find_objects(lab_img)
        for i_lab, (sy, sx) in enumerate(obs):
            lab_idx = i_lab + 1
            nconn = int(np.sum(lab_img[sy, sx] == lab_idx))
            if nconn < min_conn or nconn > max_conn:
                continue

            roi = img[sy, sx]
            roi_mask = lab_img[sy, sx] == lab_idx
            I = roi * roi_mask

            Isum = float(I.sum())
            if Isum <= 0:
                continue

            # CoM within ROI, then offset to panel coords (+0.5 for pixel center)
            Y, X = np.indices(roi.shape)
            xcom = float((X * I).sum() / Isum) + sx.start + 0.5
            ycom = float((Y * I).sum() / Isum) + sy.start + 0.5

            # Resolution filter
            if d_min is not None:
                lab = detector[i_pan].get_pixel_lab_coord((xcom, ycom))
                s1 = np.array(lab) / np.linalg.norm(lab) / beam.get_wavelength()
                s0 = np.array(beam.get_s0())
                d_spacing = 1.0 / max(np.linalg.norm(s1 - s0), 1e-10)
                if d_spacing < d_min:
                    continue

            xyzobs.append((xcom, ycom, 0))
            panels.append(i_pan)
            intensities.append(Isum)
            fi, si = int(round(xcom)), int(round(ycom))
            bboxes.append((max(0, fi - 5), fi + 6, max(0, si - 5), si + 6, 0, 1))

    refls = flex.reflection_table()
    refls['xyzobs.px.value'] = xyzobs
    refls['panel'] = panels
    refls['intensity.sum.value'] = intensities
    refls['bbox'] = bboxes
    return refls


# ===========================================================================
# Reciprocal-space preparation
# ===========================================================================

def prepare_for_labeling(predictions, expt):
    """Add rlp, scatter, and other columns needed by the labeling filters.

    Maps predictions to reciprocal space and computes per-spot scatter
    (intensity / num_pixels in signal mask).
    """
    El = ExperimentList()
    El.append(expt)

    predictions['id'] = flex.int(len(predictions), 0)
    predictions['entering'] = flex.bool(len(predictions), False)
    predictions['flags'] = flex.size_t(len(predictions), 1)
    predictions['xyzcal.px'] = predictions['xyzobs.px.value']

    predictions.centroid_px_to_mm(El)
    predictions['xyzcal.mm'] = predictions['xyzobs.mm.value']
    predictions.map_centroids_to_reciprocal_space(El)

    if 'shoebox' in predictions:
        numpix = predictions['shoebox'].count_mask_values(SIGNAL_MASK)
        safe_numpix = flex.double(np.maximum(np.array(numpix, dtype=np.float64), 1.0))
        predictions['num_pixels'] = numpix
        predictions['scatter'] = predictions['intensity.sum.value'] / safe_numpix
    elif 'intensity.sum.value' in predictions:
        predictions['scatter'] = predictions['intensity.sum.value']

    return predictions


def prepare_observed_for_labeling(observed, expt):
    """Add rlp column to observed reflections if not present."""
    obs = deepcopy(observed)
    El = ExperimentList()
    El.append(expt)

    if 'id' not in obs or len(set(obs['id'])) == 0:
        obs['id'] = flex.int(len(obs), 0)
    if 'entering' not in obs:
        obs['entering'] = flex.bool(len(obs), False)

    if 'xyzobs.mm.value' not in obs and 'xyzobs.px.value' in obs:
        obs.centroid_px_to_mm(El)
    if 'rlp' not in obs:
        obs.map_centroids_to_reciprocal_space(El)

    return obs


def compute_resolutions(predictions):
    """Compute d-spacing for each prediction from rlp column. Returns numpy array."""
    if 'rlp' not in predictions:
        return None
    rlp = np.array(predictions['rlp'])
    q_mag = np.linalg.norm(rlp, axis=1)
    q_mag = np.maximum(q_mag, 1e-10)
    return 1.0 / q_mag


# ===========================================================================
# Filter functions — each returns a boolean keep mask (numpy array)
# ===========================================================================

def filter_none(predictions, observed, args, expt=None):
    """No filter — keep all predictions."""
    return np.ones(len(predictions), dtype=bool)


def filter_intensity(predictions, observed, args, expt=None):
    """Global intensity percentile: keep top weak_fraction of predictions."""
    if args.weak_fraction >= 1.0 or len(predictions) == 0:
        return np.ones(len(predictions), dtype=bool)
    if 'intensity.sum.value' not in predictions:
        return np.ones(len(predictions), dtype=bool)

    intensities = predictions['intensity.sum.value'].as_numpy_array()
    cutoff_idx = int(len(intensities) * (1.0 - args.weak_fraction))
    if cutoff_idx <= 0:
        return np.ones(len(predictions), dtype=bool)
    sorted_i = np.sort(intensities)
    i_cutoff = sorted_i[cutoff_idx]
    return intensities >= i_cutoff


def filter_scatter_binned(predictions, observed, args, expt=None):
    """Resolution-binned scatter percentile (predictions.py approach).

    Within each resolution shell, keep strong predictions (matching observed)
    plus the top weak_fraction of weak predictions ranked by scatter
    (intensity per pixel).
    """
    try:
        from simtbx.modeling.predictions import (
            label_weak_predictions, label_weak_spots_for_integration)
    except ImportError:
        LOGGER.warning("predictions module not available, falling back to intensity filter")
        return filter_intensity(predictions, observed, args, expt)

    if 'rlp' not in predictions:
        LOGGER.warning("No rlp column in predictions, falling back to intensity filter")
        return filter_intensity(predictions, observed, args, expt)

    # Make a working copy so we don't mutate the originals
    pred_copy = deepcopy(predictions)
    obs_copy = deepcopy(observed)

    label_weak_predictions(pred_copy, obs_copy,
                           q_cutoff=args.q_cutoff, col="rlp")
    pred_copy['is_strong'] = flex.bool(np.logical_not(pred_copy['is_weak']))
    pred_copy['refl_idx'] = flex.int(np.arange(len(pred_copy)))

    label_weak_spots_for_integration(args.weak_fraction, pred_copy)

    # Return: keep anything tagged for integration
    return np.array(pred_copy['is_for_integration'])


def filter_signal(predictions, observed, args, expt=None):
    """Data-based: check actual pixel data at predicted locations."""
    if expt is None or len(predictions) == 0:
        return np.ones(len(predictions), dtype=bool)

    try:
        iset = expt.imageset
        raw_data = iset.get_raw_data(0)
    except Exception:
        LOGGER.warning("Cannot read image data for signal filter, keeping all")
        return np.ones(len(predictions), dtype=bool)

    imgs = np.array([panel.as_numpy_array().astype(float) for panel in raw_data])
    detector = expt.detector
    keep = np.zeros(len(predictions), dtype=bool)
    half = args.shoebox_size // 2

    for i_ref in range(len(predictions)):
        try:
            x, y, z = predictions['xyzobs.px.value'][i_ref]
            pid = int(predictions['panel'][i_ref]) if 'panel' in predictions else 0
            img = imgs[pid]
            nslow, nfast = img.shape

            i_slow = int(round(y))
            i_fast = int(round(x))
            s0 = max(0, i_slow - half)
            s1_idx = min(nslow, i_slow + half)
            f0 = max(0, i_fast - half)
            f1 = min(nfast, i_fast + half)

            roi = img[s0:s1_idx, f0:f1]
            if roi.size < 4:
                continue

            edge_pixels = np.concatenate([
                roi[0, :], roi[-1, :], roi[:, 0], roi[:, -1]
            ])
            bg = np.median(edge_pixels)

            # Central 4x4 region
            center_s = max(0, half - 2 - s0 + max(0, i_slow - half))
            center_e = min(roi.shape[0], center_s + 4)
            center_f = max(0, half - 2 - f0 + max(0, i_fast - half))
            center_fe = min(roi.shape[1], center_f + 4)

            if center_e <= center_s or center_fe <= center_f:
                continue

            center_roi = roi[center_s:center_e, center_f:center_fe]
            signal = np.sum(center_roi - bg)
            noise = np.sqrt(max(1, np.sum(np.abs(bg)) * center_roi.size))

            if signal / noise > args.signal_threshold:
                keep[i_ref] = True
        except Exception:
            continue

    return keep


def filter_score(predictions, observed, args, expt=None):
    """Score-trainer CNN: evaluate data vs model at each predicted location.

    Uses a trained LeNet model that takes [data_roi, model_roi] as 2-channel
    input and outputs a score in [0,1]. The prediction-filter variant is
    trained to be offset-tolerant: it scores high when there's real signal
    anywhere in the data ROI, even if offset from the model centroid.

    Requires:
      - args._panel_imgs: model panel images from simulate_and_predict()
      - args._expt: experiment with imageset for reading actual data
      - args.score_state_file: path to .net weights (or None for default)
      - args.score_cutoff: threshold score (default 0.5)
      - args.shoebox_size: ROI size (default 10)
    """
    if len(predictions) == 0:
        return np.ones(0, dtype=bool)

    # Load the scorer
    try:
        from score_trainer import roi_check
        checker = roi_check.roiCheck(state_file=getattr(args, 'score_state_file', None))
    except ImportError:
        LOGGER.warning("score_trainer not available, keeping all predictions")
        return np.ones(len(predictions), dtype=bool)

    # Get model images and real data
    model_imgs = getattr(args, '_panel_imgs', None)
    real_expt = expt
    if model_imgs is None or real_expt is None:
        LOGGER.warning("No model images or experiment for score filter, keeping all")
        return np.ones(len(predictions), dtype=bool)

    try:
        iset = real_expt.imageset
        raw_data = iset.get_raw_data(0)
        data_imgs = np.array([p.as_numpy_array().astype(float) for p in raw_data])
    except Exception as e:
        LOGGER.warning("Cannot read image data for score filter: %s" % e)
        return np.ones(len(predictions), dtype=bool)

    sb = getattr(args, 'shoebox_size', 10)
    half = sb // 2
    cutoff = getattr(args, 'score_cutoff', 0.5)

    # Batch extract ROIs for efficiency
    dat_rois = []
    mod_rois = []
    valid_indices = []

    for i_ref in range(len(predictions)):
        try:
            x, y, z = predictions['xyzobs.px.value'][i_ref]
            pid = int(predictions['panel'][i_ref]) if 'panel' in predictions else 0

            i_slow = int(round(y))
            i_fast = int(round(x))
            y0 = i_slow - half
            x0 = i_fast - half
            y1 = y0 + sb
            x1 = x0 + sb

            data_img = data_imgs[pid]
            model_img = model_imgs[pid]

            if (y0 < 0 or x0 < 0 or
                y1 > data_img.shape[0] or x1 > data_img.shape[1]):
                continue

            dat_roi = data_img[y0:y1, x0:x1].astype(np.float32)
            mod_roi = model_img[y0:y1, x0:x1].astype(np.float32)

            if dat_roi.shape != (sb, sb) or mod_roi.shape != (sb, sb):
                continue

            dat_rois.append(dat_roi)
            mod_rois.append(mod_roi)
            valid_indices.append(i_ref)
        except Exception:
            continue

    # Score in batch
    keep = np.zeros(len(predictions), dtype=bool)
    if dat_rois:
        scores = checker.score(dat_rois, mod_rois)
        if isinstance(scores, (int, float)):
            scores = [scores]
        for idx, score in zip(valid_indices, scores):
            if score >= cutoff:
                keep[idx] = True

    return keep


FILTER_FUNCS = {
    "none": filter_none,
    "intensity": filter_intensity,
    "scatter_binned": filter_scatter_binned,
    "signal": filter_signal,
    "score": filter_score,
}


# ===========================================================================
# Alignment diagnostic
# ===========================================================================

def _refls_to_lab_xy(refls, detector):
    """Convert reflection centroids to lab-frame (x, y) in mm.

    Works with both monolithic and multi-panel detectors.
    Returns array of shape (n_refls, 2).
    """
    key = 'xyzobs.px.value' if 'xyzobs.px.value' in refls else 'xyzcal.px'
    if key not in refls:
        return np.empty((0, 2))
    coords = []
    for i_ref in range(len(refls)):
        fast, slow, z = refls[key][i_ref]
        pid = int(refls['panel'][i_ref]) if 'panel' in refls else 0
        panel = detector[pid]
        lab = panel.get_pixel_lab_coord((fast, slow))
        coords.append((lab[0], lab[1]))
    return np.array(coords) if coords else np.empty((0, 2))


def check_alignment(observed, predicted, obs_detector=None, pred_detector=None,
                    pixel_cutoff=10.0, shot_idx=None, **kwargs):
    """Check centroid alignment between observed and predicted reflections.

    Compares in lab-frame mm to avoid any pixel-convention issues from
    remapping between different detector models.  Offsets are converted to
    approximate pixels using the obs_detector pixel size for display.

    Parameters
    ----------
    obs_detector : dxtbx Detector for observed reflections
    pred_detector : dxtbx Detector for predicted reflections
    pixel_cutoff : matching radius in pixels (default: 10)
    shot_idx : shot index for display (optional)

    Returns dict with:
      offsets_px : array of matched offsets (approx pixels)
      offsets_mm : array of matched offsets (mm)
      dx_mm, dy_mm : arrays of signed (lab x, lab y) offsets in mm
      n_matched, n_obs, n_pred, n_unmatched_obs, n_unmatched_pred
    """
    from scipy.spatial import cKDTree

    # Backward compat
    if obs_detector is None:
        obs_detector = kwargs.get('detector')
    if pred_detector is None:
        pred_detector = kwargs.get('detector', obs_detector)

    result = dict(offsets_px=np.array([]), offsets_mm=np.array([]),
                  dx_mm=np.array([]), dy_mm=np.array([]),
                  n_matched=0, n_obs=len(observed),
                  n_pred=len(predicted), n_unmatched_obs=len(observed),
                  n_unmatched_pred=len(predicted))

    label = "Shot %d" % shot_idx if shot_idx is not None else "Alignment"

    if obs_detector is None or pred_detector is None:
        print("    %s: skipped (no detector model)" % label)
        return result
    if len(observed) == 0 or len(predicted) == 0:
        print("    %s: skipped (%d obs, %d pred)" % (label, len(observed), len(predicted)))
        return result

    obs_key = 'xyzobs.px.value' if 'xyzobs.px.value' in observed else 'xyzcal.px'
    pred_key = 'xyzobs.px.value' if 'xyzobs.px.value' in predicted else 'xyzcal.px'
    if obs_key not in observed or pred_key not in predicted:
        return result

    pixel_size = obs_detector[0].get_pixel_size()[0]  # mm per pixel

    # Convert observed centroids to lab-frame (x, y) in mm
    obs_lab = []
    for i_ref in range(len(observed)):
        fast, slow, z = observed[obs_key][i_ref]
        pid = int(observed['panel'][i_ref]) if 'panel' in observed else 0
        lab = obs_detector[pid].get_pixel_lab_coord((fast, slow))
        obs_lab.append((lab[0], lab[1]))
    obs_lab = np.array(obs_lab) if obs_lab else np.empty((0, 2))

    # Convert predicted centroids to lab-frame (x, y) in mm
    pred_lab = []
    for i_ref in range(len(predicted)):
        fast, slow, z = predicted[pred_key][i_ref]
        pid = int(predicted['panel'][i_ref]) if 'panel' in predicted else 0
        lab = pred_detector[pid].get_pixel_lab_coord((fast, slow))
        pred_lab.append((lab[0], lab[1]))
    pred_lab = np.array(pred_lab) if pred_lab else np.empty((0, 2))

    if len(obs_lab) == 0 or len(pred_lab) == 0:
        print("    %s: no centroids to compare" % label)
        return result

    # Match in lab-frame mm
    mm_cutoff = pixel_cutoff * pixel_size
    pred_tree = cKDTree(pred_lab)

    offsets_mm = []
    dx_mm = []
    dy_mm = []
    matched_pred = set()
    n_unmatched_obs = 0

    for i_obs in range(len(obs_lab)):
        dist, idx = pred_tree.query(obs_lab[i_obs])
        if dist < mm_cutoff:
            offsets_mm.append(dist)
            dx_mm.append(pred_lab[idx, 0] - obs_lab[i_obs, 0])
            dy_mm.append(pred_lab[idx, 1] - obs_lab[i_obs, 1])
            matched_pred.add(idx)
        else:
            n_unmatched_obs += 1

    offsets_mm = np.array(offsets_mm)
    dx_mm = np.array(dx_mm)
    dy_mm = np.array(dy_mm)
    offsets_px = offsets_mm / pixel_size
    n_matched = len(offsets_mm)
    n_unmatched_pred = len(predicted) - len(matched_pred)

    result.update(offsets_px=offsets_px, offsets_mm=offsets_mm,
                  dx_mm=dx_mm, dy_mm=dy_mm,
                  n_matched=n_matched, n_unmatched_obs=n_unmatched_obs,
                  n_unmatched_pred=n_unmatched_pred)

    if n_matched > 0:
        dx_px = dx_mm / pixel_size
        dy_px = dy_mm / pixel_size
        print("    %s: %d/%d observed matched to predictions (%.1f%%)"
              % (label, n_matched, len(observed),
                 100.0 * n_matched / len(observed)))
        print("      Offset: mean=%.2f px  median=%.2f px  max=%.2f px"
              % (offsets_px.mean(), np.median(offsets_px), offsets_px.max()))
        print("      Bias:   dx=%.2f  dy=%.2f px  (pred - obs, lab frame, mean)"
              % (dx_px.mean(), dy_px.mean()))
        n_gt2 = int(np.sum(offsets_px > 2))
        if n_gt2 > 0:
            print("      >2px: %d (%.0f%%)" % (n_gt2, 100 * n_gt2 / n_matched))
        if n_unmatched_pred > 0:
            print("      Unmatched predictions: %d" % n_unmatched_pred)
        if np.median(offsets_px) > 2.0:
            print("      WARNING: Large median offset -- check phil/detector!")
    else:
        print("    %s: no observed reflections matched predictions" % label)

    return result


def check_alignment_via_model_centroids(observed, panel_imgs, obs_detector,
                                        pred_detector, shot_idx=None,
                                        roi_half=10):
    """Check alignment by computing model centroids at observed spot locations.

    This replicates geometry.py's get_new_xycalcs approach: for each observed
    spot, extract an ROI from the simulated panel image, compute intensity-
    weighted centroid (with +0.5 pixel-center convention), and compare to the
    observed centroid.  Both centroids are on pred_detector pixel coordinates.

    This avoids any DIALS spot-finding convention issues.
    """
    obs_key = 'xyzobs.px.value' if 'xyzobs.px.value' in observed else 'xyzcal.px'
    if obs_key not in observed:
        return dict(offsets_px=np.array([]), n_matched=0)

    label = "Shot %d" % shot_idx if shot_idx is not None else "Alignment"
    offsets = []
    dx_fast = []
    dy_slow = []

    for i_ref in range(len(observed)):
        obs_fast, obs_slow, z = observed[obs_key][i_ref]
        obs_pid = int(observed['panel'][i_ref]) if 'panel' in observed else 0

        # Remap observed centroid onto pred_detector via lab frame
        lab = obs_detector[obs_pid].get_pixel_lab_coord((obs_fast, obs_slow))
        lab_arr = np.array(lab)
        s1 = tuple(lab_arr / np.linalg.norm(lab_arr))

        pred_fast = pred_slow = None
        pred_pid = None
        for dp in range(len(pred_detector)):
            try:
                mm = pred_detector[dp].get_ray_intersection(s1)
                pf, ps = pred_detector[dp].millimeter_to_pixel(mm)
                sz = pred_detector[dp].get_image_size()
                if -0.5 <= pf < sz[0] + 0.5 and -0.5 <= ps < sz[1] + 0.5:
                    pred_fast, pred_slow = pf, ps
                    pred_pid = dp
                    break
            except Exception:
                continue

        if pred_pid is None or pred_pid >= len(panel_imgs):
            continue

        img = panel_imgs[pred_pid]
        nslow, nfast = img.shape

        # ROI around the observed spot position on the pred detector
        i_fast = int(round(pred_fast))
        i_slow = int(round(pred_slow))
        f0 = max(0, i_fast - roi_half)
        f1 = min(nfast, i_fast + roi_half)
        s0 = max(0, i_slow - roi_half)
        s1_idx = min(nslow, i_slow + roi_half)

        roi = img[s0:s1_idx, f0:f1]
        if roi.size < 4 or roi.sum() <= 0:
            continue

        # Check that model has signal at the observed centroid
        local_f = int(pred_fast - f0 - 0.5)
        local_s = int(pred_slow - s0 - 0.5)
        local_f = max(0, min(local_f, roi.shape[1] - 1))
        local_s = max(0, min(local_s, roi.shape[0] - 1))
        if roi[local_s, local_f] <= 0:
            continue

        # Intensity-weighted centroid (matching get_new_xycalcs convention)
        I = roi.clip(min=0)
        Y, X = np.indices(roi.shape)
        X = X + f0   # panel coords
        Y = Y + s0
        Isum = I.sum()
        if Isum <= 0:
            continue
        xcom = float((X * I).sum() / Isum) + 0.5  # +0.5 pixel center
        ycom = float((Y * I).sum() / Isum) + 0.5

        # Offset between model centroid and observed position (both on pred_detector)
        df = xcom - pred_fast
        ds = ycom - pred_slow
        dist = np.sqrt(df**2 + ds**2)
        offsets.append(dist)
        dx_fast.append(df)
        dy_slow.append(ds)

    offsets = np.array(offsets)
    dx_fast = np.array(dx_fast)
    dy_slow = np.array(dy_slow)
    n_matched = len(offsets)

    if n_matched > 0:
        print("    %s (model CoM): %d/%d spots have model signal"
              % (label, n_matched, len(observed)))
        print("      Offset: mean=%.2f px  median=%.2f px  max=%.2f px"
              % (offsets.mean(), np.median(offsets), offsets.max()))
        print("      Bias:   dfast=%.2f  dslow=%.2f  (model CoM - obs, mean)"
              % (dx_fast.mean(), dy_slow.mean()))
    else:
        print("    %s (model CoM): no spots with model signal" % label)

    return dict(offsets_px=offsets, dx_fast=dx_fast, dy_slow=dy_slow,
                n_matched=n_matched)


def check_signal_at_predictions(predicted, pred_detector, expt,
                                matched_pred_set=None,
                                roi_size=10, signal_threshold=2.0):
    """Check raw pixel data at unmatched prediction sites for actual signal.

    For predictions NOT in matched_pred_set, extract an ROI from the raw
    image and compute SNR.

    Returns (n_with_signal, n_checked).
    """
    if len(predicted) == 0:
        return 0, 0

    try:
        iset = expt.imageset
        raw_data = iset.get_raw_data(0)
        imgs = np.array([panel.as_numpy_array().astype(float) for panel in raw_data])
    except Exception as e:
        LOGGER.warning("Cannot read image data for signal check: %s" % e)
        return 0, 0

    pred_key = 'xyzobs.px.value' if 'xyzobs.px.value' in predicted else 'xyzcal.px'
    if pred_key not in predicted:
        return 0, 0

    # Determine which predictions are unmatched
    if matched_pred_set is None:
        check_indices = list(range(len(predicted)))
    else:
        check_indices = [i for i in range(len(predicted))
                         if i not in matched_pred_set]

    half = roi_size // 2
    n_with_signal = 0
    n_checked = 0

    # Remap prediction centroids from pred_detector to obs detector for pixel lookup
    obs_detector = expt.detector
    for i_ref in check_indices:
        try:
            fast, slow, z = predicted[pred_key][i_ref]
            pid = int(predicted['panel'][i_ref]) if 'panel' in predicted else 0

            # If detectors differ, remap via ray intersection
            if pred_detector is not obs_detector and len(obs_detector) != len(pred_detector):
                lab = pred_detector[pid].get_pixel_lab_coord((fast, slow))
                lab_arr = np.array(lab)
                s1 = tuple(lab_arr / np.linalg.norm(lab_arr))
                found = False
                for dp in range(len(obs_detector)):
                    try:
                        mm = obs_detector[dp].get_ray_intersection(s1)
                        fast, slow = obs_detector[dp].millimeter_to_pixel(mm)
                        pid = dp
                        found = True
                        break
                    except Exception:
                        continue
                if not found:
                    continue

            if pid >= len(imgs):
                continue
            img = imgs[pid]
            nslow, nfast = img.shape
            i_slow = int(round(slow))
            i_fast = int(round(fast))

            s0 = max(0, i_slow - half)
            s1_idx = min(nslow, i_slow + half)
            f0 = max(0, i_fast - half)
            f1 = min(nfast, i_fast + half)

            roi = img[s0:s1_idx, f0:f1]
            if roi.size < 9:
                continue

            n_checked += 1
            bg = np.median(roi)
            signal = roi.sum() - bg * roi.size
            noise = np.sqrt(bg * roi.size) if bg > 0 else 1.0
            if noise > 0 and signal / noise > signal_threshold:
                n_with_signal += 1
        except Exception:
            continue

    return n_with_signal, n_checked


# ===========================================================================
# Merging
# ===========================================================================

def _convert_refls_to_detector(refls, src_detector, dst_detector):
    """Convert reflection centroids from one detector model to another.

    Maps (panel, fast, slow) through lab frame via ray intersection to the
    destination detector's panel/pixel coordinates. Useful for converting
    multi-panel predictions to monolithic format (or vice versa).

    Returns a new reflection table with updated panel/centroid columns.
    """
    key = 'xyzobs.px.value' if 'xyzobs.px.value' in refls else 'xyzcal.px'
    if key not in refls:
        return refls

    new_refls = deepcopy(refls)
    new_panels = flex.size_t(len(refls), 0)
    new_xyz = flex.vec3_double(len(refls), (0, 0, 0))
    dropped = []

    for i_ref in range(len(refls)):
        fast, slow, z = refls[key][i_ref]
        pid = int(refls['panel'][i_ref]) if 'panel' in refls else 0
        src_panel = src_detector[pid]
        lab = src_panel.get_pixel_lab_coord((fast, slow))

        # Compute ray direction from sample (origin) to this lab point
        lab_arr = np.array(lab)
        s1 = tuple(lab_arr / np.linalg.norm(lab_arr))

        # Find which destination panel this ray hits
        best_pid = -1
        best_fast = best_slow = 0.0
        for dp in range(len(dst_detector)):
            try:
                dst_panel = dst_detector[dp]
                mm = dst_panel.get_ray_intersection(s1)
                new_f, new_s = dst_panel.millimeter_to_pixel(mm)
                sz = dst_panel.get_image_size()
                if -0.5 <= new_f < sz[0] + 0.5 and -0.5 <= new_s < sz[1] + 0.5:
                    best_pid = dp
                    best_fast = new_f
                    best_slow = new_s
                    break
            except Exception:
                continue

        if best_pid < 0:
            dropped.append(i_ref)
        else:
            new_panels[i_ref] = best_pid
            new_xyz[i_ref] = (best_fast, best_slow, z)

    if dropped:
        keep = flex.bool(len(refls), True)
        for idx in dropped:
            keep[idx] = False
        new_refls = new_refls.select(keep)
        new_panels = new_panels.select(keep)
        new_xyz = new_xyz.select(keep)

    new_refls['panel'] = new_panels
    new_refls[key] = new_xyz
    if dropped:
        print("    Detector conversion: %d/%d refls mapped, %d dropped (off detector)"
              % (len(new_refls), len(refls), len(dropped)))
    return new_refls


def _safe_concat(observed, new_preds):
    """Concatenate observed and predicted reflection tables safely.

    Strips columns from new_preds that don't exist in observed (or have
    incompatible types) to avoid boost::bad_get errors from extend().
    Adds essential missing columns to new_preds with default values.
    """
    # Essential columns that hopper needs from .refl files
    essential = {'xyzobs.px.value', 'panel', 'bbox', 'intensity.sum.value'}

    # Keep only columns that exist in observed OR are essential
    obs_keys = set(observed.keys())
    pred_keys = set(new_preds.keys())

    # Remove columns from predicted that don't exist in observed
    # (these would cause extend to fail or add unwanted data)
    for k in pred_keys - obs_keys:
        if k not in essential:
            del new_preds[k]

    # For columns in observed but not in predicted, we don't need to add them —
    # extend() handles missing columns by filling with defaults.
    # But 'shoebox' column is problematic — remove from predicted if types differ.
    if 'shoebox' in new_preds and 'shoebox' in observed:
        del new_preds['shoebox']

    # Ensure panel column types match (observed uses size_t, predicted may use int)
    if 'panel' in new_preds and 'panel' in observed:
        if type(new_preds['panel']) != type(observed['panel']):
            new_preds['panel'] = flex.size_t(list(new_preds['panel']))

    try:
        return flex.reflection_table.concat([observed, new_preds])
    except RuntimeError:
        # If concat still fails, strip predicted to bare minimum
        minimal = flex.reflection_table()
        for k in essential:
            if k in new_preds:
                minimal[k] = new_preds[k]
        return flex.reflection_table.concat([observed, minimal])


def merge_reflections(observed, predicted, q_cutoff=0.005,
                      obs_detector=None, pred_detector=None, beam=None, **kwargs):
    """Merge predicted reflections with observed, deduplicating in lab frame.

    Handles monolithic vs multi-panel detector differences transparently.
    """
    from scipy.spatial import cKDTree

    # Backward compat
    if obs_detector is None:
        obs_detector = kwargs.get('detector')
    if pred_detector is None:
        pred_detector = kwargs.get('detector', obs_detector)

    if len(predicted) == 0:
        return observed

    # Use lab-frame deduplication when we have detector models
    if obs_detector is not None and pred_detector is not None:
        obs_lab = _refls_to_lab_xy(observed, obs_detector)
        pred_lab = _refls_to_lab_xy(predicted, pred_detector)

        mm_cutoff = 0.3  # ~4 pixels at 0.075mm pixel size
        if beam is not None:
            pixel_size = obs_detector[0].get_pixel_size()[0]
            d_det = obs_detector[0].get_distance()
            wavelength = beam.get_wavelength()
            mm_cutoff = max(0.15, q_cutoff * d_det * wavelength)

        if len(obs_lab) > 0 and len(pred_lab) > 0:
            obs_tree = cKDTree(obs_lab)
            keep = flex.bool(len(predicted), True)
            for i_ref in range(len(predicted)):
                dist, _ = obs_tree.query(pred_lab[i_ref])
                if dist < mm_cutoff:
                    keep[i_ref] = False

            # Convert predictions to observed detector convention before merge
            new_preds = predicted.select(keep)
            if len(obs_detector) != len(pred_detector):
                new_preds = _convert_refls_to_detector(
                    new_preds, pred_detector, obs_detector)

            n_kept = keep.count(True)
            n_total = len(predicted)
            print("    Dedup: %d passed filter, %d new (not near observed), %d removed"
                  % (n_total, n_kept, n_total - n_kept))

            if n_kept == 0:
                return observed
            return _safe_concat(observed, new_preds)
        else:
            # No centroids — just concat
            if len(obs_detector) != len(pred_detector):
                predicted = _convert_refls_to_detector(
                    predicted, pred_detector, obs_detector)
            return _safe_concat(observed, predicted)

    # Fallback: pixel-based dedup (same detector convention assumed)
    print("    WARNING: No detector model for lab-frame dedup, using pixel fallback")
    centroid_key = 'xyzobs.px.value' if 'xyzobs.px.value' in observed else 'xyzcal.px'
    if centroid_key not in observed:
        return _safe_concat(observed, predicted)

    obs_coords = []
    for i_ref in range(len(observed)):
        x, y, z = observed[centroid_key][i_ref]
        obs_coords.append((x, y))
    obs_tree = cKDTree(np.array(obs_coords)) if obs_coords else None

    pred_key = 'xyzobs.px.value' if 'xyzobs.px.value' in predicted else 'xyzcal.px'
    keep = flex.bool(len(predicted), True)
    if obs_tree is not None and pred_key in predicted:
        for i_ref in range(len(predicted)):
            x, y, z = predicted[pred_key][i_ref]
            dist, _ = obs_tree.query([x, y])
            if dist < 3.0:
                keep[i_ref] = False

    new_preds = predicted.select(keep)
    n_kept = keep.count(True)
    print("    Dedup: %d passed filter, %d new, %d removed"
          % (len(predicted), n_kept, len(predicted) - n_kept))
    if n_kept == 0:
        return observed
    return _safe_concat(observed, new_preds)


# ===========================================================================
# Per-shot statistics (for --compare-filters)
# ===========================================================================

def compute_filter_stats(predictions, observed, args, expt, resolutions=None):
    """Run all filter modes and return a dict of stats for comparison."""
    stats = {}
    stats['n_predictions'] = len(predictions)
    stats['n_observed'] = len(observed)

    for mode_name, filter_func in FILTER_FUNCS.items():
        keep = filter_func(predictions, observed, args, expt=expt)
        n_kept = int(np.sum(keep))
        stats['%s_kept' % mode_name] = n_kept
        stats['%s_frac' % mode_name] = n_kept / max(1, len(predictions))

        # Resolution breakdown (3 bins: low, mid, high)
        if resolutions is not None and len(resolutions) > 0:
            res_sort = np.sort(resolutions)
            n3 = len(res_sort) // 3
            if n3 > 0:
                d_low = res_sort[n3]  # boundary between high-res and mid-res
                d_high = res_sort[2 * n3]
                for label, sel in [
                    ("hires", resolutions < d_low),
                    ("midres", (resolutions >= d_low) & (resolutions < d_high)),
                    ("lores", resolutions >= d_high)
                ]:
                    n_bin = int(np.sum(sel))
                    n_kept_bin = int(np.sum(keep[sel]))
                    stats['%s_%s_kept' % (mode_name, label)] = n_kept_bin
                    stats['%s_%s_total' % (mode_name, label)] = n_bin

    return stats


# ===========================================================================
# Main predict-and-expand
# ===========================================================================

def predict_and_expand(cycle_outdir, args):
    """Main entry point: predict, filter, merge, save expanded reflections.

    Returns the path to the expanded output directory.
    """
    import pandas

    # Handle deprecated --signal-filter flag
    if getattr(args, 'signal_filter', False) and args.filter_mode == "none":
        args.filter_mode = "signal"

    all_df = load_pandas_from_outdir(cycle_outdir)
    input_lines = get_input_lines(cycle_outdir, args.input_file, pandas_df=all_df)

    # Load phil params for simulator settings
    phil_params = _load_phil_params(cycle_outdir, getattr(args, 'phil', None))

    expand_dir = os.path.join(cycle_outdir, args.expand_dir)
    os.makedirs(expand_dir, exist_ok=True)

    filter_func = FILTER_FUNCS[args.filter_mode]

    new_input_lines = []
    n_expanded = 0
    n_new_total = 0
    all_offsets = []  # aggregate alignment offsets across shots
    all_stats = []  # for --compare-filters
    dials_expts = ExperimentList()  # for --save-dials
    dials_refls = flex.reflection_table()  # for --save-dials

    print("\nFilter mode: %s" % args.filter_mode)
    if args.filter_mode in ("intensity", "scatter_binned"):
        print("Weak fraction: %.2f" % args.weak_fraction)
    if args.filter_mode == "signal":
        print("Signal threshold: %.1f" % args.signal_threshold)
    if args.filter_mode == "score":
        print("Score cutoff: %.2f" % getattr(args, 'score_cutoff', 0.5))
        sf = getattr(args, 'score_state_file', None)
        print("Score model: %s" % (sf if sf else "default"))
    print()

    # Load optimized detector if available (from geometry refinement)
    opt_detector = _load_optimized_detector(cycle_outdir)
    if opt_detector is not None:
        print("Using optimized detector from %s (%d panels)"
              % (cycle_outdir, len(opt_detector)))

    for i_shot, line in enumerate(input_lines):
        from simtbx.diffBragg.hopper_utils import split_line
        exp_path, ref_path, exp_idx, spec = split_line(line)

        # Find matching refined model
        match = all_df[all_df.opt_exp_name.apply(
            lambda x: os.path.splitext(os.path.basename(str(x)))[0]
        ).str.contains(os.path.splitext(os.path.basename(exp_path))[0])]

        if len(match) == 0:
            print("  Shot %d: no refined model found, keeping original" % i_shot)
            new_input_lines.append(line)
            continue

        row = match.iloc[0]

        El = ExperimentListFactory.from_json_file(exp_path, check_format=True)
        expt = El[exp_idx]
        beam = expt.beam
        # Use optimized detector if available, otherwise original
        detector = opt_detector if opt_detector is not None else expt.detector

        refls = flex.reflection_table.from_file(ref_path)

        # Simulate (uses GPU if DIFFBRAGG_USE_CUDA is set)
        use_cuda = os.environ.get("DIFFBRAGG_USE_CUDA") is not None
        dev_id = getattr(args, 'device_id', 0) or 0
        print("  Shot %d: simulating predictions%s..." %
              (i_shot, " (GPU %d)" % dev_id if use_cuda else " (CPU)"))
        predictions, panel_imgs, sim_expt = simulate_and_predict(
            row, expt, detector, beam,
            threshold=args.threshold, d_min=args.d_min,
            cuda=use_cuda, device_id=dev_id,
            phil_params=phil_params)

        if len(predictions) == 0:
            print("  Shot %d: no predictions, keeping original (%d refls)"
                  % (i_shot, len(refls)))
            new_input_lines.append(line)
            continue

        # Predictions are on sim_expt.detector (the geometry-optimized detector)
        pred_det = sim_expt.detector

        # Check alignment: lab-frame comparison (refls_from_sims centroids)
        obs_det = expt.detector  # original detector from the experiment file
        align_result = check_alignment(refls, predictions,
                                       obs_detector=obs_det,
                                       pred_detector=pred_det,
                                       shot_idx=i_shot)
        if len(align_result['offsets_px']) > 0:
            all_offsets.append(align_result['offsets_px'])

        # Check alignment: model CoM at observed locations (geometry.py method)
        com_result = check_alignment_via_model_centroids(
            refls, panel_imgs, obs_det, pred_det, shot_idx=i_shot)
        if len(com_result.get('offsets_px', [])) > 0:
            all_offsets_com = getattr(args, '_all_offsets_com', [])
            all_offsets_com.append(com_result['offsets_px'])
            args._all_offsets_com = all_offsets_com

        # Optional: check raw data for signal at unmatched prediction sites
        if getattr(args, 'signal_check', False) and align_result['n_unmatched_pred'] > 0:
            # Build set of matched prediction indices for exclusion
            # (we don't have per-index tracking from check_alignment, so check all)
            n_sig, n_chk = check_signal_at_predictions(
                predictions, pred_det, expt,
                roi_size=args.shoebox_size,
                signal_threshold=args.signal_threshold)
            if n_chk > 0:
                print("      Signal check: %d/%d predictions have signal (SNR>%.1f)"
                      % (n_sig, n_chk, args.signal_threshold))

        # Attach model images to args for score filter
        args._panel_imgs = panel_imgs
        args._expt = expt

        # Prepare for reciprocal-space filtering
        # (scatter_binned needs rlp+scatter; others don't need it but it's cheap)
        # Use sim_expt for predictions (on sim detector) and expt for observed
        needs_rlp = args.filter_mode == "scatter_binned" or args.compare_filters
        if needs_rlp:
            try:
                predictions = prepare_for_labeling(predictions, sim_expt)
                obs_labeled = prepare_observed_for_labeling(refls, expt)
            except Exception as e:
                LOGGER.warning("Failed to prepare for labeling: %s" % e)
                obs_labeled = refls
                needs_rlp = False
        else:
            obs_labeled = refls

        # Comparison mode: run all filters and record stats
        resolutions = None
        if args.compare_filters:
            if 'rlp' in predictions:
                resolutions = compute_resolutions(predictions)
            shot_stats = compute_filter_stats(
                predictions, obs_labeled, args, expt, resolutions)
            shot_stats['shot'] = i_shot
            shot_stats['exp_path'] = exp_path
            all_stats.append(shot_stats)

        # Apply the selected filter
        if needs_rlp:
            keep_mask = filter_func(predictions, obs_labeled, args, expt=expt)
        else:
            keep_mask = filter_func(predictions, refls, args, expt=expt)
        filtered_preds = predictions.select(flex.bool(keep_mask))

        n_pred = len(predictions)
        n_kept = len(filtered_preds)
        print("  Shot %d: %d predictions, %d pass %s filter"
              % (i_shot, n_pred, n_kept, args.filter_mode))

        # Merge with observed (dedup removes predictions near existing observed)
        # Uses lab-frame coordinates to handle monolithic vs multi-panel
        n_orig = len(refls)
        if n_kept > 0:
            merged = merge_reflections(refls, filtered_preds,
                                       q_cutoff=args.q_cutoff,
                                       obs_detector=obs_det,
                                       pred_detector=pred_det,
                                       beam=beam)
        else:
            merged = refls
        n_new = len(merged) - n_orig
        n_new_total += n_new
        if n_new > 0:
            n_expanded += 1

        # Save expanded reflections
        basename = os.path.splitext(os.path.basename(ref_path))[0]
        expanded_refl_path = os.path.join(expand_dir, basename + ".refl")
        merged.as_file(expanded_refl_path)

        if spec is not None:
            new_line = "%s %s %s %d" % (exp_path, expanded_refl_path, spec, exp_idx)
        elif exp_idx > 0:
            new_line = "%s %s %d" % (exp_path, expanded_refl_path, exp_idx)
        else:
            new_line = "%s %s" % (exp_path, expanded_refl_path)
        new_input_lines.append(new_line)

        print("  Shot %d: %d original + %d new = %d total reflections"
              % (i_shot, n_orig, n_new, len(merged)))

        # Accumulate for DIALS output
        if getattr(args, 'save_dials', False) and n_kept > 0:
            from dxtbx.model import Experiment
            dials_e = Experiment()
            dials_e.detector = pred_det
            dials_e.beam = beam
            dials_e.crystal = expt.crystal
            if expt.imageset is not None:
                dials_e.imageset = expt.imageset
            dials_expts.append(dials_e)
            exp_id = len(dials_expts) - 1
            # Tag predictions with experiment id and 'predicted' flag
            shot_preds = deepcopy(filtered_preds)
            shot_preds['id'] = flex.int(len(shot_preds), exp_id)
            if 'is_predicted' not in shot_preds:
                shot_preds['is_predicted'] = flex.bool(len(shot_preds), True)
            dials_refls.extend(shot_preds)

    # Write new input file
    new_input_path = os.path.join(expand_dir, "input.txt")
    with open(new_input_path, "w") as f:
        for line in new_input_lines:
            f.write(line + "\n")

    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    print("Filter mode:     %s" % args.filter_mode)
    print("Shots processed: %d" % len(input_lines))
    print("Shots expanded:  %d" % n_expanded)
    print("Total new refls: %d" % n_new_total)
    print("Expanded input:  %s" % new_input_path)
    if all_offsets:
        combined = np.concatenate(all_offsets)
        print("Alignment (lab):  mean=%.2f px  median=%.2f px  max=%.2f px  (%d matched)"
              % (combined.mean(), np.median(combined), combined.max(), len(combined)))
    all_offsets_com = getattr(args, '_all_offsets_com', [])
    if all_offsets_com:
        combined_com = np.concatenate(all_offsets_com)
        print("Alignment (CoM):  mean=%.2f px  median=%.2f px  max=%.2f px  (%d matched)"
              % (combined_com.mean(), np.median(combined_com), combined_com.max(),
                 len(combined_com)))
    print("="*60)

    # Write comparison stats
    if args.compare_filters and all_stats:
        stats_path = os.path.join(expand_dir, "filter_comparison.csv")
        _write_comparison_stats(all_stats, stats_path)

    # Write DIALS-compatible output for visualization
    if getattr(args, 'save_dials', False) and len(dials_refls) > 0:
        dials_expt_path = os.path.join(expand_dir, "predictions.expt")
        dials_refl_path = os.path.join(expand_dir, "predictions.refl")
        dials_expts.as_file(dials_expt_path)
        dials_refls.as_file(dials_refl_path)
        print("\nDIALS output for visualization:")
        print("  dials.image_viewer %s %s" % (dials_expt_path, dials_refl_path))

    return expand_dir


def _write_comparison_stats(all_stats, stats_path):
    """Write filter comparison CSV and print summary table."""
    import pandas
    df = pandas.DataFrame(all_stats)
    df.to_csv(stats_path, index=False)
    print("\nFilter comparison saved to: %s" % stats_path)

    # Print summary
    print("\n%-18s %8s %8s %8s %8s" % ("Filter", "Mean", "Median", "Min", "Max"))
    print("-" * 62)
    for mode in FILTER_FUNCS:
        col = "%s_kept" % mode
        if col in df:
            vals = df[col]
            print("%-18s %8.1f %8.1f %8d %8d"
                  % (mode, vals.mean(), vals.median(), vals.min(), vals.max()))

    # Resolution breakdown if available
    has_res = any(("%s_hires_kept" % m) in df.columns for m in FILTER_FUNCS)
    if has_res:
        print("\nPer-resolution-bin kept fractions:")
        print("%-18s %10s %10s %10s" % ("Filter", "Hi-res", "Mid-res", "Lo-res"))
        print("-" * 50)
        for mode in FILTER_FUNCS:
            fracs = []
            for label in ("hires", "midres", "lores"):
                kept_col = "%s_%s_kept" % (mode, label)
                total_col = "%s_%s_total" % (mode, label)
                if kept_col in df and total_col in df:
                    total = df[total_col].sum()
                    kept = df[kept_col].sum()
                    fracs.append("%.1f%%" % (100 * kept / max(1, total)))
                else:
                    fracs.append("N/A")
            print("%-18s %10s %10s %10s" % (mode, fracs[0], fracs[1], fracs[2]))


def main():
    args = parse_args()
    predict_and_expand(args.cycle_outdir, args)


if __name__ == "__main__":
    main()
