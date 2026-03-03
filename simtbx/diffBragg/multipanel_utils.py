"""
Utilities for converting monolithic single-panel detector data to multi-panel format
for per-panel geometry refinement.

Uses split_eiger_16M_to_panels() from easyBragg/split_to_panels.py for panel detection
and multi-panel detector construction. Builds reflection conversion logic on top.
"""
from __future__ import division, print_function
import os
import logging
import numpy as np

logger = logging.getLogger("diffBragg.multipanel")


def get_panel_defs_from_raw(raw_image, mono_detector, gap_value=-1):
    """
    Detect sub-panels in a monolithic raw image and build a multi-panel detector.

    :param raw_image: 2D numpy array, raw monolithic detector image
    :param mono_detector: dxtbx Detector for the monolithic image
    :param gap_value: sentinel value for gap pixels (default -1 for EIGER)
    :return: (panel_id_map, n_panels, region_slices, panels, new_detector)
        panel_id_map: int array same shape as raw_image, -1 for gap, 0..N-1 for panel ID
        n_panels: number of panels found
        region_slices: list of (sY, sX) slice pairs for each panel
        panels: list of trimmed panel images
        new_detector: dxtbx Detector with N sub-panels
    """
    from simtbx.diffBragg.utils import split_eiger_16M_to_panels

    regions, n_panels, region_slices, panels, new_detector = \
        split_eiger_16M_to_panels(raw_image, detector=mono_detector)

    # Build panel_id_map: regions is 1-indexed (0 = gap), convert to 0-indexed (-1 = gap)
    panel_id_map = regions.astype(np.int32) - 1

    return panel_id_map, n_panels, region_slices, panels, new_detector


def convert_refl_to_multipanel(mono_refl_path, region_slices, n_panels,
                                panel_id_map, new_detector, output_path):
    """
    Convert a monolithic gathered reflection table to multi-panel format.

    For each reflection:
    1. Look up sub-panel from centroid position
    2. Convert bounding box to panel-local coordinates
    3. Clip/crop shoebox data to panel boundaries
    4. Update panel ID and centroid

    :param mono_refl_path: path to monolithic .refl file
    :param region_slices: list of (sY, sX) slices from split_eiger_16M_to_panels
    :param n_panels: number of panels
    :param panel_id_map: 2D array mapping pixels to panel IDs (-1 = gap)
    :param new_detector: multi-panel dxtbx Detector
    :param output_path: where to write converted .refl
    :return: (n_converted, n_dropped)
    """
    from dials.array_family import flex as dials_flex
    from dials.model.data import Shoebox

    mono_refls = dials_flex.reflection_table.from_file(mono_refl_path)
    n_total = len(mono_refls)
    if n_total == 0:
        mono_refls.as_file(output_path)
        return 0, 0

    # Precompute panel offsets and sizes
    panel_offsets = []  # (col_off, row_off) for each panel
    panel_sizes = []    # (nfast, nslow) for each panel
    for pid in range(n_panels):
        sY, sX = region_slices[pid]
        # EIGER padding: 514-row regions have 1-pixel border padding
        pad_eiger = (sY.stop - sY.start) == 514
        if pad_eiger:
            col_off = sX.start + 1
            row_off = sY.start + 1
        else:
            col_off = sX.start
            row_off = sY.start
        panel_offsets.append((col_off, row_off))
        nfast, nslow = new_detector[pid].get_image_size()
        panel_sizes.append((nfast, nslow))

    new_refls = dials_flex.reflection_table()
    new_shoeboxes = []
    converted = 0
    dropped = 0

    for i_ref in range(n_total):
        ref = mono_refls[i_ref: i_ref + 1]
        xyz = mono_refls[i_ref]["xyzobs.px.value"]
        cx, cy, cz = xyz

        # Look up panel from centroid
        iy, ix = int(round(cy)), int(round(cx))
        map_h, map_w = panel_id_map.shape
        if iy < 0 or iy >= map_h or ix < 0 or ix >= map_w:
            dropped += 1
            continue
        pid = int(panel_id_map[iy, ix])
        if pid < 0:
            dropped += 1
            continue

        col_off, row_off = panel_offsets[pid]
        nfast, nslow = panel_sizes[pid]

        # Get monolithic bbox
        sb = mono_refls[i_ref]["shoebox"]
        x1, x2, y1, y2, z1, z2 = sb.bbox
        assert z1 == 0 and z2 == 1

        # Convert to panel-local coordinates
        lx1 = x1 - col_off
        lx2 = x2 - col_off
        ly1 = y1 - row_off
        ly2 = y2 - row_off

        # Clip to panel boundaries
        cx1 = max(0, lx1)
        cx2 = min(nfast, lx2)
        cy1 = max(0, ly1)
        cy2 = min(nslow, ly2)

        if cx2 <= cx1 or cy2 <= cy1:
            dropped += 1
            continue

        # Crop shoebox data arrays
        # Offsets into original shoebox data
        sx1 = cx1 - lx1
        sx2 = sx1 + (cx2 - cx1)
        sy1 = cy1 - ly1
        sy2 = sy1 + (cy2 - cy1)

        old_data = sb.data.as_numpy_array()     # shape (1, ydim, xdim)
        old_bg = sb.background.as_numpy_array()
        old_mask = sb.mask.as_numpy_array()

        new_data = old_data[0:1, sy1:sy2, sx1:sx2].copy()
        new_bg = old_bg[0:1, sy1:sy2, sx1:sx2].copy()
        new_mask = old_mask[0:1, sy1:sy2, sx1:sx2].copy()

        if new_data.size == 0:
            dropped += 1
            continue

        # Build new shoebox
        new_sb = Shoebox((cx1, cx2, cy1, cy2, 0, 1))
        new_sb.allocate()
        new_sb.data = dials_flex.float(np.ascontiguousarray(new_data))
        new_sb.background = dials_flex.float(np.ascontiguousarray(new_bg))
        new_sb.mask = dials_flex.int(np.ascontiguousarray(new_mask))

        # Update reflection: panel, centroid
        new_refls.extend(ref)
        idx = len(new_refls) - 1
        new_refls["panel"][idx] = pid
        new_refls["xyzobs.px.value"][idx] = (cx - col_off, cy - row_off, cz)
        new_shoeboxes.append(new_sb)
        converted += 1

    new_refls["shoebox"] = dials_flex.shoebox(new_shoeboxes)
    new_refls["id"] = dials_flex.int(len(new_refls), 0)
    new_refls.as_file(output_path)

    return converted, dropped


def compute_panel_offsets_from_detector(mono_detector, multi_detector):
    """
    Compute monolithic pixel offsets for each panel in a multi-panel detector,
    using the panel origins relative to the monolithic detector origin.

    No raw image needed — purely geometric computation from detector models.

    :param mono_detector: single-panel dxtbx Detector (monolithic)
    :param multi_detector: multi-panel dxtbx Detector
    :return: list of (col_off, row_off, nfast, nslow) per panel
    """
    mono_orig = np.array(mono_detector[0].get_origin())
    fast = np.array(mono_detector[0].get_fast_axis())
    slow = np.array(mono_detector[0].get_slow_axis())
    pixsize = mono_detector[0].get_pixel_size()[0]

    offsets = []
    for pid in range(len(multi_detector)):
        pan_orig = np.array(multi_detector[pid].get_origin())
        delta = pan_orig - mono_orig
        col_off = int(round(np.dot(delta, fast) / pixsize))
        row_off = int(round(np.dot(delta, slow) / pixsize))
        nfast, nslow = multi_detector[pid].get_image_size()
        offsets.append((col_off, row_off, nfast, nslow))
    return offsets


def build_panel_id_map_from_detector(mono_detector, multi_detector):
    """
    Build a panel_id_map array from detector geometry (no raw image needed).

    :param mono_detector: single-panel monolithic detector
    :param multi_detector: multi-panel detector
    :return: 2D int array, shape = monolithic image size, -1 for unmapped, 0..N-1 for panels
    """
    mono_nfast, mono_nslow = mono_detector[0].get_image_size()
    panel_id_map = np.full((mono_nslow, mono_nfast), -1, dtype=np.int32)
    offsets = compute_panel_offsets_from_detector(mono_detector, multi_detector)
    for pid, (col_off, row_off, nfast, nslow) in enumerate(offsets):
        panel_id_map[row_off:row_off + nslow, col_off:col_off + nfast] = pid
    return panel_id_map


def is_multipanel_reference(reference_detector, input_detector):
    """
    Detect if a reference detector is multi-panel while input is monolithic.

    :param reference_detector: dxtbx Detector from reference_geom
    :param input_detector: dxtbx Detector from input experiment
    :return: True if conversion is needed
    """
    n_ref = len(reference_detector)
    n_inp = len(input_detector)
    if n_inp != 1 or n_ref <= 1:
        return False
    # Check for known multi-panel counts
    if n_ref in {32, 60}:
        return True
    # Also accept any multi-panel reference if panel sizes are plausible sub-panels
    nfast, nslow = reference_detector[0].get_image_size()
    if nfast <= 1030 and nslow <= 512:
        return True
    return False


def remap_modeler_to_multipanel(Modeler, mono_detector, multi_detector):
    """
    Remap a DataModeler's gathered data from monolithic to multi-panel coordinates.

    Called AFTER GatherFromExperiment + data_to_one_dim with monolithic data, when
    the detector is being swapped to a multi-panel reference geometry.

    The pixel data (all_data, all_background, all_trusted, all_sigmas) stays the same —
    same pixels, just remapped coordinates. We update:
    - self.rois: panel-local bounding boxes
    - self.pids: correct panel IDs
    - self.all_fast, self.all_slow, self.all_pid: per-pixel coords
    - self.pan_fast_slow: the (pid, fast, slow) array used by diffBragg
    - self.E.detector: the multi-panel detector

    :param Modeler: DataModeler instance (post data_to_one_dim)
    :param mono_detector: original monolithic detector
    :param multi_detector: multi-panel reference detector
    :return: number of remapped ROIs
    """
    from scitbx.array_family import flex

    offsets = compute_panel_offsets_from_detector(mono_detector, multi_detector)
    panel_id_map = build_panel_id_map_from_detector(mono_detector, multi_detector)

    n_remapped = 0
    for i_roi in range(len(Modeler.rois)):
        x1, x2, y1, y2 = Modeler.rois[i_roi]

        # Use the actual observed centroid (intensity-weighted) for panel lookup,
        # not the bbox midpoint which can land on inter-module gaps
        if hasattr(Modeler, 'refls_idx') and hasattr(Modeler, 'refls') \
                and i_roi < len(Modeler.refls_idx):
            i_ref_for_centroid = Modeler.refls_idx[i_roi]
            cx_f, cy_f, _ = Modeler.refls[i_ref_for_centroid]["xyzobs.px.value"]
            cx, cy = int(round(cx_f)), int(round(cy_f))
        else:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

        pid = int(panel_id_map[cy, cx])
        if pid < 0:
            logger.warning("ROI %d centroid (%d,%d) on gap pixel, marking as invalid"
                           % (i_roi, cx, cy))
            Modeler.pids[i_roi] = -1
            continue

        col_off, row_off, nfast, nslow = offsets[pid]

        # Panel-local ROI
        lx1 = x1 - col_off
        lx2 = x2 - col_off
        ly1 = y1 - row_off
        ly2 = y2 - row_off

        Modeler.rois[i_roi] = (lx1, lx2, ly1, ly2)
        Modeler.pids[i_roi] = pid

        # Also update Modeler.refls so dump_gathered_to_refl produces consistent data
        if hasattr(Modeler, 'refls_idx') and hasattr(Modeler, 'refls') \
                and i_roi < len(Modeler.refls_idx):
            i_ref = Modeler.refls_idx[i_roi]
            Modeler.refls["panel"][i_ref] = pid
            if "xyzobs.px.value" in Modeler.refls:
                cx_orig, cy_orig, cz = Modeler.refls[i_ref]["xyzobs.px.value"]
                Modeler.refls["xyzobs.px.value"][i_ref] = (cx_orig - col_off, cy_orig - row_off, cz)

        n_remapped += 1

    # Remap per-pixel arrays using the panel_id_map
    mono_fast = Modeler.all_fast.astype(int)
    mono_slow = Modeler.all_slow.astype(int)

    new_pid = np.zeros(len(mono_fast), dtype=int)
    new_fast = mono_fast.copy()
    new_slow = mono_slow.copy()

    for ipx in range(len(mono_fast)):
        fx, sy = int(mono_fast[ipx]), int(mono_slow[ipx])
        pid = int(panel_id_map[sy, fx])
        if pid < 0:
            continue
        col_off, row_off, _, _ = offsets[pid]
        new_pid[ipx] = pid
        new_fast[ipx] = fx - col_off
        new_slow[ipx] = sy - row_off

    Modeler.all_fast = new_fast.astype(Modeler.all_fast.dtype)
    Modeler.all_slow = new_slow.astype(Modeler.all_slow.dtype)
    Modeler.all_pid = new_pid

    # Rebuild pan_fast_slow (the array diffBragg uses)
    pan_fast_slow = np.ascontiguousarray(
        np.vstack([new_pid, new_fast, new_slow]).T.ravel())
    Modeler.pan_fast_slow = flex.size_t(pan_fast_slow)

    # Swap detector
    Modeler.E.detector = multi_detector

    logger.info("Remapped %d/%d ROIs to %d-panel detector"
                % (n_remapped, len(Modeler.rois), len(multi_detector)))

    return n_remapped


def setup_panel_group_from_id(SIM, n_panels):
    """
    Set up SIM.panel_group_from_id for a multi-panel detector where all panels
    belong to group 0 (rigid body). Used for per-shot refinement with multi-panel data.

    :param SIM: sim_data instance
    :param n_panels: number of panels
    """
    from copy import deepcopy
    SIM.panel_group_from_id = {pid: 0 for pid in range(n_panels)}
    SIM.panel_groups_refined = {0}
    SIM.panel_reference_from_id = {}
    for pid in range(n_panels):
        SIM.panel_reference_from_id[pid] = deepcopy(SIM.detector[pid].get_origin())


def remap_image_data_to_multipanel(img_data_mono, mono_detector, multi_detector):
    """
    Remap a monolithic image array (1, nslow, nfast) to multi-panel format
    (npan, nslow_panel, nfast_panel).

    :param img_data_mono: numpy array shape (1, nslow_mono, nfast_mono)
    :param mono_detector: monolithic dxtbx Detector
    :param multi_detector: multi-panel dxtbx Detector
    :return: numpy array shape (npan, nslow_panel, nfast_panel)
    """
    offsets = compute_panel_offsets_from_detector(mono_detector, multi_detector)
    n_panels = len(multi_detector)
    nfast_p, nslow_p = multi_detector[0].get_image_size()

    img_mp = np.zeros((n_panels, nslow_p, nfast_p), dtype=img_data_mono.dtype)
    for pid, (col_off, row_off, nfast, nslow) in enumerate(offsets):
        img_mp[pid] = img_data_mono[0, row_off:row_off + nslow, col_off:col_off + nfast]

    return img_mp


def generate_panel_group_file(n_panels, output_path, grouping="per_panel"):
    """
    Write a panel group file for geometry refinement.

    :param n_panels: number of panels
    :param output_path: path to write the file
    :param grouping: "per_panel" (each panel independent) or "all_one" (rigid body)
    """
    with open(output_path, "w") as f:
        for pid in range(n_panels):
            group_id = pid if grouping == "per_panel" else 0
            f.write("%d %d\n" % (pid, group_id))
    logger.info("Wrote panel group file (%s, %d panels): %s" % (grouping, n_panels, output_path))


def convert_gathers_to_multipanel(params, df):
    """
    Top-level orchestration: convert all gathered monolithic reflections to multi-panel format.

    1. Load raw image to detect panel layout
    2. Build multi-panel detector
    3. Convert each gathered .refl to multi-panel coords
    4. Create multi-panel experiments
    5. Generate panel group file
    6. Update df and params to point at converted files

    :param params: phil params (geometry.multipanel must be True)
    :param df: pandas DataFrame with geom_ref, geom_exp columns
    :return: updated df
    """
    from dxtbx.model import ExperimentList, Experiment

    outdir = os.path.join(params.outdir, "multipanel")
    os.makedirs(outdir, exist_ok=True)

    # --- Check if gathers are already multi-panel (cycle 2+) ---
    first_geom_exp = df["geom_exp"].iloc[0]
    el_check = ExperimentList.from_file(first_geom_exp, check_format=False)
    if len(el_check[0].detector) > 1:
        n_panels = len(el_check[0].detector)
        logger.info("Gathers already in multi-panel format (%d panels), "
                     "skipping reflection conversion" % n_panels)

        # Save detector file for reference_geom
        det_expt_path = os.path.join(outdir, "multipanel_detector.expt")
        det_el = ExperimentList()
        det_el.append(Experiment(detector=el_check[0].detector))
        det_el.as_file(det_expt_path)

        # Generate panel group file
        grouping = params.geometry.multipanel_grouping
        pg_path = os.path.join(outdir, "panel_groups.txt")
        generate_panel_group_file(n_panels, pg_path, grouping=grouping)

        # Update params — df paths already point to multi-panel data
        params.refiner.panel_group_file = pg_path
        params.refiner.reference_geom = det_expt_path

        return df

    # --- Step 1: get a raw image for panel detection ---
    if params.geometry.multipanel_raw_image is not None:
        import dxtbx
        raw_path = params.geometry.multipanel_raw_image
        logger.info("Loading raw image for panel detection: %s" % raw_path)
        img = dxtbx.load(raw_path)
        raw = img.get_raw_data().as_numpy_array()
        mono_det = img.get_detector()
    else:
        # Load detector from geom_exp (optimized), raw image from original exp_name
        first_geom_exp = df["geom_exp"].iloc[0]
        el = ExperimentList.from_file(first_geom_exp, check_format=False)
        mono_det = el[0].detector
        # geom_exp was saved without imageset — load raw from original experiment
        first_orig_exp = df["exp_name"].iloc[0]
        logger.info("Loading raw image from original experiment: %s" % first_orig_exp)
        orig_el = ExperimentList.from_file(first_orig_exp, check_format=True)
        raw = orig_el[0].imageset.get_raw_data(0)[0].as_numpy_array()

    gap_value = params.geometry.multipanel_gap_value

    # --- Step 2: detect panels and build multi-panel detector ---
    logger.info("Detecting panels from raw image (gap_value=%.1f)..." % gap_value)
    panel_id_map, n_panels, region_slices, panels, new_detector = \
        get_panel_defs_from_raw(raw, mono_det, gap_value=gap_value)
    logger.info("Found %d panels" % n_panels)

    # Save multi-panel detector
    det_expt_path = os.path.join(outdir, "multipanel_detector.expt")
    det_el = ExperimentList()
    det_el.append(Experiment(detector=new_detector))
    det_el.as_file(det_expt_path)
    logger.info("Saved multi-panel detector: %s" % det_expt_path)

    # --- Step 3: generate panel group file ---
    grouping = params.geometry.multipanel_grouping
    pg_path = os.path.join(outdir, "panel_groups.txt")
    generate_panel_group_file(n_panels, pg_path, grouping=grouping)

    # --- Step 4: convert each gathered .refl and experiment ---
    total_converted = 0
    total_dropped = 0

    new_geom_refs = []
    new_geom_exps = []
    new_geom_exp_idxs = []

    for i_row in range(len(df)):
        mono_refl_path = df["geom_ref"].iloc[i_row]
        mono_exp_path = df["geom_exp"].iloc[i_row]

        basename = os.path.splitext(os.path.basename(mono_refl_path))[0]

        # Convert reflections
        mp_refl_path = os.path.join(outdir, basename + "_mp.refl")
        n_conv, n_drop = convert_refl_to_multipanel(
            mono_refl_path, region_slices, n_panels,
            panel_id_map, new_detector, mp_refl_path)
        total_converted += n_conv
        total_dropped += n_drop

        # Create multi-panel experiment (replace detector)
        el = ExperimentList.from_file(mono_exp_path, check_format=False)
        el[0].detector = new_detector
        mp_exp_path = os.path.join(outdir, basename + "_mp.expt")
        el.as_file(mp_exp_path)

        new_geom_refs.append(os.path.abspath(mp_refl_path))
        new_geom_exps.append(os.path.abspath(mp_exp_path))
        new_geom_exp_idxs.append(0)

    logger.info("Converted %d reflections, dropped %d (%.1f%%)"
                % (total_converted, total_dropped,
                   100.0 * total_dropped / max(1, total_converted + total_dropped)))

    # --- Step 5: update df ---
    df["geom_ref"] = new_geom_refs
    df["geom_exp"] = new_geom_exps
    df["geom_exp_idx"] = new_geom_exp_idxs

    # --- Step 6: update params ---
    params.refiner.panel_group_file = pg_path
    params.refiner.reference_geom = det_expt_path

    return df
