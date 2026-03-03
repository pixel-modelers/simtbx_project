"""
Test multipanel_utils: panel detection, reflection conversion, panel group file,
and GatherFromReflectionTable roundtrip with multi-panel data.

Usage:
  conda run -n simtbx python simtbx/diffBragg/tests/tst_multipanel_utils.py --verbose
"""
from __future__ import division, print_function
import numpy as np
import os
import tempfile
import shutil

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--verbose", action='store_true')
args = parser.parse_args()

VERBOSE = args.verbose


# ============================================================
# Helpers to build synthetic data
# ============================================================

def make_eiger16m_raw(seed=42):
    """Create synthetic EIGER 16M raw image: 8x4 modules of 514x1030 with gap pixels = -1.

    Returns (raw, module_positions) where module_positions[i] = (row_start, col_start).
    """
    rng = np.random.RandomState(seed)
    module_slow, module_fast = 514, 1030
    n_rows, n_cols = 8, 4
    gap = 2

    total_slow = n_rows * module_slow + (n_rows - 1) * gap
    total_fast = n_cols * module_fast + (n_cols - 1) * gap

    raw = np.full((total_slow, total_fast), -1.0, dtype=np.float64)
    module_positions = []

    for ir in range(n_rows):
        for ic in range(n_cols):
            r0 = ir * (module_slow + gap)
            c0 = ic * (module_fast + gap)
            raw[r0:r0 + module_slow, c0:c0 + module_fast] = rng.uniform(10, 1000,
                                                                         (module_slow, module_fast))
            module_positions.append((r0, c0))

    return raw, module_positions


def make_mono_detector(img_shape, pixsize_mm=0.075, distance_mm=150.0):
    """Create a single-panel monolithic detector."""
    from dxtbx.model import Panel, Detector

    nslow, nfast = img_shape
    d = Detector()
    pan_dict = {
        'fast_axis': (1.0, 0.0, 0.0),
        'slow_axis': (0.0, -1.0, 0.0),
        'origin': (-nfast * pixsize_mm / 2.0, nslow * pixsize_mm / 2.0, -distance_mm),
        'pixel_size': (pixsize_mm, pixsize_mm),
        'image_size': (nfast, nslow),
        'trusted_range': (-2.0, 1e7),
        'type': 'SENSOR_PAD',
        'name': 'Panel0',
        'mask': [],
        'thickness': 0.0,
        'material': '',
        'mu': 0.0,
        'gain': 1.0,
        'pedestal': 0.0,
        'identifier': '',
        'raw_image_offset': (0, 0),
    }
    d.add_panel(Panel.from_dict(pan_dict))
    return d


def make_mono_refls(positions, bbox_half=5):
    """Create synthetic monolithic reflection table.

    :param positions: list of (cx, cy) centroid pixel positions (monolithic coords)
    :param bbox_half: half-width of bounding box
    :return: dials reflection table
    """
    from dials.array_family import flex
    from dials.model.data import Shoebox

    R = flex.reflection_table()
    shoeboxes = []

    for i, (cx, cy) in enumerate(positions):
        x1 = int(cx) - bbox_half
        x2 = int(cx) + bbox_half
        y1 = int(cy) - bbox_half
        y2 = int(cy) + bbox_half

        xdim = x2 - x1
        ydim = y2 - y1

        # Each reflection gets a unique data value (i+1) so we can verify roundtrip
        data = np.full((1, ydim, xdim), float(i + 1), dtype=np.float32)
        bg = np.full((1, ydim, xdim), 0.5, dtype=np.float32)
        mask = np.ones((1, ydim, xdim), dtype=np.int32)

        sb = Shoebox((x1, x2, y1, y2, 0, 1))
        sb.allocate()
        sb.data = flex.float(np.ascontiguousarray(data))
        sb.background = flex.float(np.ascontiguousarray(bg))
        sb.mask = flex.int(np.ascontiguousarray(mask))
        shoeboxes.append(sb)

    n = len(positions)
    R['panel'] = flex.int(n, 0)
    R['id'] = flex.int(n, 0)
    R['shoebox'] = flex.shoebox(shoeboxes)
    R['xyzobs.px.value'] = flex.vec3_double([(cx, cy, 0.0) for cx, cy in positions])
    R['miller_index'] = flex.miller_index([(i + 1, i + 1, i + 1) for i in range(n)])

    return R


# ============================================================
# Test 1: Panel detection
# ============================================================

def test_panel_detection():
    """Verify get_panel_defs_from_raw finds 32 panels and builds correct panel_id_map."""
    from simtbx.diffBragg.multipanel_utils import get_panel_defs_from_raw

    raw, module_positions = make_eiger16m_raw()
    mono_det = make_mono_detector(raw.shape)

    panel_id_map, n_panels, region_slices, panels, new_detector = \
        get_panel_defs_from_raw(raw, mono_det)

    # Should find 32 panels
    assert n_panels == 32, "Expected 32 panels, got %d" % n_panels
    assert len(new_detector) == 32
    assert len(panels) == 32
    assert len(region_slices) == 32

    # panel_id_map shape matches raw image
    assert panel_id_map.shape == raw.shape

    # Gap pixels should be -1
    gap_mask = raw == -1
    assert np.all(panel_id_map[gap_mask] == -1), "Gap pixels should map to -1"

    # Non-gap pixels should be 0..31
    non_gap_mask = raw != -1
    non_gap_ids = panel_id_map[non_gap_mask]
    assert non_gap_ids.min() >= 0
    assert non_gap_ids.max() <= 31
    assert set(non_gap_ids) == set(range(32))

    # Each trimmed panel should be 512x1028
    for i, panel_img in enumerate(panels):
        assert panel_img.shape == (512, 1028), \
            "Panel %d shape %s != (512, 1028)" % (i, panel_img.shape)

    # Each detector panel should have image_size (1028, 512)
    for i in range(32):
        nfast, nslow = new_detector[i].get_image_size()
        assert (nfast, nslow) == (1028, 512), \
            "Panel %d image_size (%d,%d) != (1028,512)" % (i, nfast, nslow)

    # Verify panel origins differ from monolithic origin
    mono_orig = mono_det[0].get_origin()
    for i in range(32):
        pan_orig = new_detector[i].get_origin()
        if i > 0:
            # At least some panels should have different origins
            assert pan_orig != mono_orig or i == 0

    if VERBOSE:
        print("  Found %d panels, panel_id_map shape %s" % (n_panels, panel_id_map.shape))
        print("  Gap pixels: %d, panel pixels: %d" % (gap_mask.sum(), non_gap_mask.sum()))
        for i in range(min(4, n_panels)):
            orig = new_detector[i].get_origin()
            print("  Panel %d origin: (%.4f, %.4f, %.4f)" % (i, orig[0], orig[1], orig[2]))

    print("  test_panel_detection: OK")
    return panel_id_map, n_panels, region_slices, panels, new_detector


# ============================================================
# Test 2: Reflection conversion
# ============================================================

def test_refl_conversion(panel_id_map, n_panels, region_slices, new_detector, module_positions):
    """Test convert_refl_to_multipanel: coordinate transform, data preservation, gap dropping."""
    from simtbx.diffBragg.multipanel_utils import convert_refl_to_multipanel

    tmpdir = tempfile.mkdtemp(prefix="tst_mp_refl_")
    try:
        _test_refl_conversion_impl(tmpdir, panel_id_map, n_panels, region_slices,
                                   new_detector, module_positions)
    finally:
        shutil.rmtree(tmpdir)


def _test_refl_conversion_impl(tmpdir, panel_id_map, n_panels, region_slices,
                                new_detector, module_positions):
    from dials.array_family import flex
    from simtbx.diffBragg.multipanel_utils import convert_refl_to_multipanel

    bbox_half = 5

    # Place reflections at known positions:
    # (a) Center of panel 0 (should convert cleanly)
    # (b) Center of panel 15 (middle of detector)
    # (c) Center of panel 31 (last panel)
    # (d) On a gap pixel (should be dropped)
    # (e) Near panel edge (bbox may need clipping)

    positions = []
    expected_panels = []  # which panel each reflection should end up in (-1 = dropped)

    # For each test case, place centroid in known location
    for panel_idx in [0, 15, 31]:
        sY, sX = region_slices[panel_idx]
        # EIGER padding: centroid at center of trimmed panel
        pad = (sY.stop - sY.start) == 514
        if pad:
            cy = sY.start + 1 + 256  # middle of 512 active rows
            cx = sX.start + 1 + 514  # middle of 1028 active cols
        else:
            cy = sY.start + 256
            cx = sX.start + 514
        positions.append((float(cx), float(cy)))
        expected_panels.append(panel_idx)

    # (d) Gap pixel — find a gap pixel
    gap_ys, gap_xs = np.where(panel_id_map == -1)
    # Pick a gap pixel that's far from edges so bbox fits in image
    gap_idx = len(gap_ys) // 2
    gap_cy, gap_cx = float(gap_ys[gap_idx]), float(gap_xs[gap_idx])
    positions.append((gap_cx, gap_cy))
    expected_panels.append(-1)  # should be dropped

    # (e) Near edge of panel 0 — bbox will extend past panel boundary
    sY, sX = region_slices[0]
    pad = (sY.stop - sY.start) == 514
    if pad:
        # Place centroid 2 pixels from top-left of active area
        # So bbox extends beyond panel into gap/padding
        edge_cy = float(sY.start + 1 + 2)
        edge_cx = float(sX.start + 1 + 2)
    else:
        edge_cy = float(sY.start + 2)
        edge_cx = float(sX.start + 2)
    positions.append((edge_cx, edge_cy))
    expected_panels.append(0)  # should still be panel 0, just clipped

    mono_refls = make_mono_refls(positions, bbox_half=bbox_half)
    mono_path = os.path.join(tmpdir, "mono.refl")
    mono_refls.as_file(mono_path)

    mp_path = os.path.join(tmpdir, "mp.refl")
    n_converted, n_dropped = convert_refl_to_multipanel(
        mono_path, region_slices, n_panels,
        panel_id_map, new_detector, mp_path)

    if VERBOSE:
        print("  Converted: %d, dropped: %d" % (n_converted, n_dropped))

    # Should drop exactly 1 (the gap reflection)
    assert n_dropped == 1, "Expected 1 dropped (gap), got %d" % n_dropped
    assert n_converted == 4, "Expected 4 converted, got %d" % n_converted

    # Load converted reflections and verify
    mp_refls = flex.reflection_table.from_file(mp_path)
    assert len(mp_refls) == 4

    # Check the cleanly-converted reflections (first 3)
    for i_mp, (i_orig, expected_pid) in enumerate(
            [(j, expected_panels[j]) for j in range(len(expected_panels)) if expected_panels[j] >= 0]):
        pid = mp_refls[i_mp]['panel']
        assert pid == expected_pid, \
            "Reflection %d: expected panel %d, got %d" % (i_mp, expected_pid, pid)

        # Check centroid was transformed
        cx_orig, cy_orig = positions[i_orig]
        cx_mp, cy_mp, cz_mp = mp_refls[i_mp]['xyzobs.px.value']

        sY, sX = region_slices[expected_pid]
        pad = (sY.stop - sY.start) == 514
        col_off = sX.start + 1 if pad else sX.start
        row_off = sY.start + 1 if pad else sY.start

        assert abs(cx_mp - (cx_orig - col_off)) < 0.01, \
            "Reflection %d: cx_mp=%.1f, expected %.1f" % (i_mp, cx_mp, cx_orig - col_off)
        assert abs(cy_mp - (cy_orig - row_off)) < 0.01, \
            "Reflection %d: cy_mp=%.1f, expected %.1f" % (i_mp, cy_mp, cy_orig - row_off)

        # Check centroid is within panel bounds
        nfast, nslow = new_detector[expected_pid].get_image_size()
        assert 0 <= cx_mp < nfast, "cx_mp=%.1f out of panel fast range [0,%d)" % (cx_mp, nfast)
        assert 0 <= cy_mp < nslow, "cy_mp=%.1f out of panel slow range [0,%d)" % (cy_mp, nslow)

        # Check shoebox data for non-edge reflections (first 3 have unique values i+1)
        sb = mp_refls[i_mp]['shoebox']
        data = sb.data.as_numpy_array()
        if i_orig < 3:
            # Center reflections: bbox should not be clipped, data value = i_orig + 1
            expected_val = float(i_orig + 1)
            assert np.allclose(data, expected_val), \
                "Reflection %d: data value %.1f != expected %.1f" % (i_mp, data.mean(), expected_val)

        # Check bbox is within panel
        x1, x2, y1, y2, z1, z2 = sb.bbox
        assert x1 >= 0, "bbox x1=%d < 0" % x1
        assert y1 >= 0, "bbox y1=%d < 0" % y1
        assert x2 <= nfast, "bbox x2=%d > nfast=%d" % (x2, nfast)
        assert y2 <= nslow, "bbox y2=%d > nslow=%d" % (y2, nslow)

    # The edge reflection (last one) should have a clipped bbox
    edge_sb = mp_refls[3]['shoebox']
    edge_x1, edge_x2, edge_y1, edge_y2, _, _ = edge_sb.bbox
    edge_data = edge_sb.data.as_numpy_array()
    # Original bbox was 10x10, clipped bbox should be smaller
    clipped_xdim = edge_x2 - edge_x1
    clipped_ydim = edge_y2 - edge_y1
    assert clipped_xdim <= 2 * bbox_half, "Edge reflection should be clipped in x"
    assert clipped_ydim <= 2 * bbox_half, "Edge reflection should be clipped in y"
    if VERBOSE:
        print("  Edge reflection bbox: (%d,%d,%d,%d), size %dx%d (orig %dx%d)" %
              (edge_x1, edge_x2, edge_y1, edge_y2, clipped_xdim, clipped_ydim,
               2 * bbox_half, 2 * bbox_half))

    print("  test_refl_conversion: OK")


# ============================================================
# Test 3: Panel group file
# ============================================================

def test_panel_group_file():
    """Test generate_panel_group_file for per_panel and all_one groupings."""
    from simtbx.diffBragg.multipanel_utils import generate_panel_group_file

    tmpdir = tempfile.mkdtemp(prefix="tst_mp_pg_")
    try:
        # per_panel grouping
        pg_path = os.path.join(tmpdir, "per_panel.txt")
        generate_panel_group_file(32, pg_path, grouping="per_panel")
        lines = open(pg_path).readlines()
        assert len(lines) == 32
        for i, line in enumerate(lines):
            pid, gid = map(int, line.strip().split())
            assert pid == i
            assert gid == i, "per_panel: panel %d should have group %d, got %d" % (i, i, gid)

        # all_one grouping
        pg_path2 = os.path.join(tmpdir, "all_one.txt")
        generate_panel_group_file(32, pg_path2, grouping="all_one")
        lines2 = open(pg_path2).readlines()
        assert len(lines2) == 32
        for i, line in enumerate(lines2):
            pid, gid = map(int, line.strip().split())
            assert pid == i
            assert gid == 0, "all_one: panel %d should have group 0, got %d" % (i, gid)

    finally:
        shutil.rmtree(tmpdir)

    print("  test_panel_group_file: OK")


# ============================================================
# Test 4: GatherFromReflectionTable roundtrip
# ============================================================

def test_gather_roundtrip(panel_id_map, n_panels, region_slices, new_detector, module_positions):
    """Test that converted multi-panel reflections can be loaded via GatherFromReflectionTable
    and that pixel data matches the original monolithic data."""
    from dials.array_family import flex
    from dxtbx.model import Experiment, ExperimentList, Crystal, Beam
    from cctbx import uctbx
    from scitbx.matrix import sqr
    from simtbx.diffBragg.multipanel_utils import convert_refl_to_multipanel
    from simtbx.diffBragg import hopper_utils

    tmpdir = tempfile.mkdtemp(prefix="tst_mp_gather_")
    try:
        _test_gather_roundtrip_impl(tmpdir, panel_id_map, n_panels, region_slices,
                                    new_detector, module_positions)
    finally:
        shutil.rmtree(tmpdir)


def _test_gather_roundtrip_impl(tmpdir, panel_id_map, n_panels, region_slices,
                                 new_detector, module_positions):
    from dials.array_family import flex
    from dxtbx.model import Experiment, ExperimentList, Crystal, Beam
    from cctbx import uctbx
    from scitbx.matrix import sqr
    from simtbx.diffBragg.multipanel_utils import convert_refl_to_multipanel
    from simtbx.diffBragg import hopper_utils

    bbox_half = 4

    # Create a crystal and beam for the experiment
    ucell = (55, 65, 75, 90, 90, 90)
    symbol = "P1"
    a_real, b_real, c_real = sqr(
        uctbx.unit_cell(ucell).orthogonalization_matrix()
    ).transpose().as_list_of_lists()
    crystal = Crystal(a_real, b_real, c_real, symbol)
    beam = Beam(s0=(0, 0, -1.0 / 1.0))  # 1 Angstrom wavelength

    # Place 3 reflections in different panels (well inside panel boundaries)
    positions = []
    for panel_idx in [0, 10, 20]:
        sY, sX = region_slices[panel_idx]
        pad = (sY.stop - sY.start) == 514
        if pad:
            cy = sY.start + 1 + 256
            cx = sX.start + 1 + 514
        else:
            cy = sY.start + 256
            cx = sX.start + 514
        positions.append((float(cx), float(cy)))

    mono_refls = make_mono_refls(positions, bbox_half=bbox_half)

    # Save monolithic reflections
    mono_refl_path = os.path.join(tmpdir, "mono.refl")
    mono_refls.as_file(mono_refl_path)

    # Store original shoebox data for comparison
    orig_data = {}
    for i_ref in range(len(mono_refls)):
        sb = mono_refls[i_ref]['shoebox']
        orig_data[i_ref] = sb.data.as_numpy_array().copy()

    # Convert to multi-panel
    mp_refl_path = os.path.join(tmpdir, "mp.refl")
    n_conv, n_drop = convert_refl_to_multipanel(
        mono_refl_path, region_slices, n_panels,
        panel_id_map, new_detector, mp_refl_path)
    assert n_conv == 3 and n_drop == 0

    # Add rlp column (needed by GatherFromReflectionTable)
    mp_refls = flex.reflection_table.from_file(mp_refl_path)
    A_inv = np.array(crystal.get_A()).reshape((3, 3))
    hkl = np.array(mp_refls["miller_index"]).astype(float)
    rlp = hkl.dot(A_inv.T)
    mp_refls["rlp"] = flex.vec3_double(tuple(map(tuple, rlp)))
    mp_refls.as_file(mp_refl_path)

    # Create multi-panel experiment
    exp = Experiment(detector=new_detector, beam=beam, crystal=crystal)
    exp_list = ExperimentList()
    exp_list.append(exp)
    mp_exp_path = os.path.join(tmpdir, "mp.expt")
    exp_list.as_file(mp_exp_path)

    # Load via GatherFromReflectionTable
    # Need the full phil (hopper_phil + simulator/refiner/roi/predictions)
    from iotbx.phil import parse
    from simtbx.diffBragg.phil import hopper_phil, philz
    full_phil_scope = parse(hopper_phil + philz)
    params = full_phil_scope.extract()
    Modeler = hopper_utils.DataModeler(params)
    gathered = Modeler.GatherFromReflectionTable(mp_exp_path, mp_refl_path, sg_symbol="P1")

    assert gathered, "GatherFromReflectionTable returned False"
    assert len(Modeler.rois) == 3, "Expected 3 ROIs, got %d" % len(Modeler.rois)

    # Verify that the loaded pixel data matches original
    for i_roi in range(len(Modeler.rois)):
        roi_sel = Modeler.roi_id == i_roi
        roi_pixels = Modeler.all_data[roi_sel]

        x1, x2, y1, y2 = Modeler.rois[i_roi]
        roi_shape = (y2 - y1, x2 - x1)
        roi_img = roi_pixels.reshape(roi_shape)

        # The original data was filled with (i_ref + 1)
        expected_val = float(i_roi + 1)
        assert np.allclose(roi_img, expected_val), \
            "ROI %d: data value %.1f != expected %.1f" % (i_roi, roi_img.mean(), expected_val)

        # Verify panel assignment
        pid = Modeler.pids[i_roi]
        assert pid in [0, 10, 20], "ROI %d: unexpected panel %d" % (i_roi, pid)

    if VERBOSE:
        print("  Loaded %d ROIs via GatherFromReflectionTable" % len(Modeler.rois))
        for i_roi in range(len(Modeler.rois)):
            x1, x2, y1, y2 = Modeler.rois[i_roi]
            print("  ROI %d: panel=%d, bbox=(%d,%d,%d,%d), data_val=%.0f" %
                  (i_roi, Modeler.pids[i_roi], x1, x2, y1, y2,
                   Modeler.all_data[Modeler.roi_id == i_roi].mean()))

    print("  test_gather_roundtrip: OK")


# ============================================================
# Test 5: Empty reflection table
# ============================================================

def test_empty_refls():
    """Verify convert_refl_to_multipanel handles empty reflection table."""
    from dials.array_family import flex
    from simtbx.diffBragg.multipanel_utils import convert_refl_to_multipanel

    raw, module_positions = make_eiger16m_raw()
    mono_det = make_mono_detector(raw.shape)

    from simtbx.diffBragg.multipanel_utils import get_panel_defs_from_raw
    panel_id_map, n_panels, region_slices, panels, new_detector = \
        get_panel_defs_from_raw(raw, mono_det)

    tmpdir = tempfile.mkdtemp(prefix="tst_mp_empty_")
    try:
        empty = flex.reflection_table()
        empty_path = os.path.join(tmpdir, "empty.refl")
        empty.as_file(empty_path)

        out_path = os.path.join(tmpdir, "empty_mp.refl")
        n_conv, n_drop = convert_refl_to_multipanel(
            empty_path, region_slices, n_panels,
            panel_id_map, new_detector, out_path)
        assert n_conv == 0 and n_drop == 0

        loaded = flex.reflection_table.from_file(out_path)
        assert len(loaded) == 0
    finally:
        shutil.rmtree(tmpdir)

    print("  test_empty_refls: OK")


# ============================================================
# Test 6: Origin calculation sanity check
# ============================================================

def test_panel_origins(new_detector, region_slices, mono_det):
    """Verify that multi-panel origins are geometrically consistent with monolithic detector."""

    mono_orig = np.array(mono_det[0].get_origin())
    pixsize = mono_det[0].get_pixel_size()[0]
    fast = np.array(mono_det[0].get_fast_axis())
    slow = np.array(mono_det[0].get_slow_axis())

    for pid in range(len(new_detector)):
        sY, sX = region_slices[pid]
        pad = (sY.stop - sY.start) == 514

        if pad:
            col_off = sX.start + 1
            row_off = sY.start + 1
        else:
            col_off = sX.start
            row_off = sY.start

        expected_orig = mono_orig + fast * (pixsize * col_off) + slow * (pixsize * row_off)
        actual_orig = np.array(new_detector[pid].get_origin())

        assert np.allclose(expected_orig, actual_orig, atol=1e-6), \
            "Panel %d origin mismatch: expected %s, got %s" % (pid, expected_orig, actual_orig)

    if VERBOSE:
        print("  Verified origins for %d panels" % len(new_detector))
    print("  test_panel_origins: OK")


# ============================================================
# Run all tests
# ============================================================

print("Testing multipanel_utils...")

# Test 1
panel_id_map, n_panels, region_slices, panels, new_detector = test_panel_detection()

# Reconstruct for reuse
raw, module_positions = make_eiger16m_raw()
mono_det = make_mono_detector(raw.shape)

# Test 2
test_refl_conversion(panel_id_map, n_panels, region_slices, new_detector, module_positions)

# Test 3
test_panel_group_file()

# Test 4
test_gather_roundtrip(panel_id_map, n_panels, region_slices, new_detector, module_positions)

# Test 5
test_empty_refls()

# Test 6
test_panel_origins(new_detector, region_slices, mono_det)

print("All multipanel_utils tests passed. OK")
