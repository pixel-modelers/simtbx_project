"""
DiffBragg state capture and comparison utilities.

Captures the complete diffBragg parameter state at stage boundaries
(hopper, perRoi_finish, geometry start/end) to verify faithful
parameter propagation between stages.

Gated by phil: logging.state_snapshots = True
"""
from __future__ import division
import os
import json
import time
import numpy as np
import logging

LOGGER = logging.getLogger("diffBragg.state")


def _to_json_safe(val):
    """Convert numpy/tuple/matrix values to JSON-serializable types."""
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (np.floating, np.float64, np.float32)):
        return float(val)
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    if isinstance(val, tuple):
        return list(val)
    if hasattr(val, 'elems'):  # scitbx.matrix.sqr
        return list(val.elems)
    return val


def capture_hopper_state(x, Mod, SIM, params, i_shot, stage_name, rank=0):
    """
    Capture complete diffBragg state from hopper refinement parameters.

    Parameters
    ----------
    x : array-like
        Optimized parameter vector
    Mod : DataModeler
        The hopper DataModeler instance
    SIM : SimData
        The nanoBragg simulation data
    params : phil
        Phil parameters
    i_shot : int
        Shot index
    stage_name : str
        e.g. "hopper_main", "hopper_perRoi", "hopper_final"
    rank : int
        MPI rank

    Returns
    -------
    dict : complete state snapshot
    """
    from simtbx.diffBragg import hopper_utils

    state = {"meta": {
        "stage_name": stage_name,
        "rank": rank,
        "i_shot": i_shot,
        "timestamp": time.time(),
    }}

    # Extract crystal parameters from x via Mod.P
    try:
        pdict = hopper_utils.get_param_from_x(x, Mod, as_dict=True)
        state["crystal"] = {
            "RotXYZ": [pdict["rotX"], pdict["rotY"], pdict["rotZ"]],
            "Nabc": [pdict["Na"], pdict["Nb"], pdict["Nc"]],
            "Ndef": [pdict["Nd"], pdict["Ne"], pdict["Nf"]],
            "Bfactor_image": pdict["Bfactor"],
            "ucell": [pdict["a"], pdict["b"], pdict["c"],
                      pdict["al"], pdict["be"], pdict["ga"]],
        }
        if "cholesky" in pdict:
            state["crystal"]["cholesky"] = pdict["cholesky"]

        state["scale"] = {"G": pdict["scale"]}
        state["diffuse"] = {
            "gamma": [pdict["diff_gam_a"], pdict["diff_gam_b"], pdict["diff_gam_c"]],
            "sigma": [pdict["diff_sig_a"], pdict["diff_sig_b"], pdict["diff_sig_c"]],
        }
        state["detector"] = {"detz_shift_m": pdict["detz"]}
        state["gonio"] = {"gonio_angle": pdict["gonio_angle"]}
    except Exception as e:
        LOGGER.warning("Failed to extract params from x: %s" % e)
        state["crystal"] = {}
        state["scale"] = {}

    # Extract eta from SIM
    try:
        eta_abc = hopper_utils.get_mosaicity_from_x(x, Mod, SIM)
        state["crystal"]["eta_abc"] = list(eta_abc)
    except Exception:
        pass

    # Bfactor_aniso (if available)
    try:
        if hasattr(SIM.D, 'Bfactor_aniso'):
            baniso = SIM.D.Bfactor_aniso
            if baniso is not None:
                state["crystal"]["Bfactor_aniso"] = list(baniso)
    except Exception:
        pass

    # Umatrix and Bmatrix from SIM
    try:
        state["crystal"]["Umatrix"] = list(SIM.D.Umatrix)
        state["crystal"]["Bmatrix"] = list(SIM.D.Bmatrix)
    except Exception:
        pass

    # Per-ROI scale factors
    try:
        roi_scales = {}
        for roi_id in range(len(Mod.rois)):
            pname = "scale_roi%d" % roi_id
            if pname in Mod.P:
                p = Mod.P[pname]
                roi_scales[str(roi_id)] = p.get_val(x[p.xpos])
        if roi_scales:
            state["scale"]["per_roi"] = roi_scales
    except Exception:
        pass

    # Beam
    try:
        state["beam"] = {
            "lambda_coefficients": list(SIM.D.lambda_coefficients) if hasattr(SIM.D, 'lambda_coefficients') else None,
        }
    except Exception:
        pass

    # Gonio from SIM.D
    try:
        state["gonio"]["spindle_axis"] = list(SIM.D.spindle_axis) if SIM.D.spindle_axis is not None else None
        state["gonio"]["phi_deg"] = SIM.D.phi_deg
        state["gonio"]["osc_deg"] = SIM.D.osc_deg
        state["gonio"]["phisteps"] = SIM.D.phisteps
    except Exception:
        pass

    # Amatrix (crystal orientation matrix)
    # Note: Mod.E.crystal holds the INITIAL crystal (pre-refinement).
    # We also compute the REFINED Amat with RotXYZ baked in and refined ucell,
    # which is what actually gets serialized via save_to_pandas.
    try:
        state["crystal"]["Amat_initial"] = list(Mod.E.crystal.get_A())
    except Exception:
        pass
    try:
        from simtbx.diffBragg.hopper_utils import update_crystal_from_x
        refined_crystal = update_crystal_from_x(Mod, SIM, x)
        state["crystal"]["Amat"] = list(refined_crystal.get_A())
    except Exception:
        # Fallback to initial if update fails
        if "Amat_initial" in state.get("crystal", {}):
            state["crystal"]["Amat"] = state["crystal"]["Amat_initial"]

    # Global image identifier (for cross-stage matching)
    try:
        state["meta"]["exper_name"] = getattr(Mod, "exper_name", None)
    except Exception:
        pass

    return state


def capture_geometry_state(x, ref_params, i_shot, Modeler, SIM, stage_name, rank=0):
    """
    Capture diffBragg state from geometry refinement parameters.

    Parameters
    ----------
    x : array-like
        Optimized (rescaled) parameter vector
    ref_params : Parameters
        Geometry refinement parameter collection
    i_shot : int
        Shot index in Modelers dict
    Modeler : DataModeler
        The DataModeler for this shot
    SIM : SimData
        The simulation data
    stage_name : str
        e.g. "geom_start", "geom_end"
    rank : int
        MPI rank

    Returns
    -------
    dict : complete state snapshot
    """
    from simtbx.diffBragg import hopper_utils

    state = {"meta": {
        "stage_name": stage_name,
        "rank": rank,
        "i_shot": i_shot,
        "timestamp": time.time(),
    }}

    prefix = "rank%d_shot%d" % (rank, i_shot)

    # Crystal params
    crystal = {}
    try:
        rotXYZ = []
        for i in range(3):
            p = ref_params["%s_RotXYZ%d" % (prefix, i)]
            rotXYZ.append(p.get_val(x[p.xpos]))
        crystal["RotXYZ"] = rotXYZ

        Nabc = []
        for i in range(3):
            p = ref_params["%s_Nabc%d" % (prefix, i)]
            Nabc.append(p.get_val(x[p.xpos]))
        crystal["Nabc"] = Nabc

        Ndef = []
        for i in range(3):
            p = ref_params["%s_Ndef%d" % (prefix, i)]
            Ndef.append(p.get_val(x[p.xpos]))
        crystal["Ndef"] = Ndef

        eta = []
        for i in range(3):
            p = ref_params["%s_eta%d" % (prefix, i)]
            eta.append(p.get_val(x[p.xpos]))
        crystal["eta_abc"] = eta

        G_p = ref_params["%s_Scale" % prefix]
        ucell = []
        num_uc = len(Modeler.ucell_man.variables)
        for i in range(num_uc):
            p = ref_params["%s_Ucell%d" % (prefix, i)]
            ucell.append(p.get_val(x[p.xpos]))
        # Convert ucell manager variables to a,b,c,al,be,ga
        Modeler.ucell_man.variables = ucell
        crystal["ucell"] = list(Modeler.ucell_man.unit_cell_parameters)
    except KeyError as e:
        LOGGER.debug("Missing geometry crystal param: %s" % e)

    # Bfactor (if present in geometry params)
    try:
        bfac_p = ref_params["%s_Bfactor" % prefix]
        crystal["Bfactor_image"] = bfac_p.get_val(x[bfac_p.xpos])
    except KeyError:
        crystal["Bfactor_image"] = 0.0  # geometry doesn't have it (GAP 1)

    # Bfactor_aniso (if present in geometry params)
    try:
        baniso = []
        for i in range(6):
            p = ref_params["%s_Baniso%d" % (prefix, i)]
            baniso.append(p.get_val(x[p.xpos]))
        crystal["Bfactor_aniso"] = baniso
    except KeyError:
        pass

    state["crystal"] = crystal

    # Scale
    try:
        G_p = ref_params["%s_Scale" % prefix]
        state["scale"] = {"G": G_p.get_val(x[G_p.xpos])}
    except KeyError:
        state["scale"] = {}

    # Per-ROI scale factors
    try:
        roi_scales = {}
        roi_id = 0
        while True:
            pname = "%s_scale_roi%d" % (prefix, roi_id)
            if pname not in ref_params:
                break
            p = ref_params[pname]
            roi_scales[str(roi_id)] = p.get_val(x[p.xpos])
            roi_id += 1
        if roi_scales:
            state["scale"]["per_roi"] = roi_scales
    except Exception:
        pass

    # Diffuse params
    try:
        diff_gamma = []
        diff_sigma = []
        for i in range(3):
            pname_g = "%s_diffuse_gamma%d" % (prefix, i)
            pname_s = "%s_diffuse_sigma%d" % (prefix, i)
            if pname_g in ref_params:
                diff_gamma.append(ref_params[pname_g].get_val(x[ref_params[pname_g].xpos]))
                diff_sigma.append(ref_params[pname_s].get_val(x[ref_params[pname_s].xpos]))
        if diff_gamma:
            state["diffuse"] = {"gamma": diff_gamma, "sigma": diff_sigma}
    except Exception:
        pass

    # Detector
    try:
        det_params = {}
        for name in ref_params:
            if name.startswith("group"):
                p = ref_params[name]
                val = p.get_val(x[p.xpos])
                if "Rot" in name:
                    val = val * 180.0 / np.pi  # to degrees
                elif "Shift" in name:
                    val = val * 1000.0  # to mm
                det_params[name] = val
        if det_params:
            state["detector"] = det_params
    except Exception:
        state["detector"] = {}

    # Beam
    try:
        lam0 = ref_params["lambda0"]
        lam1 = ref_params["lambda1"]
        state["beam"] = {
            "lambda0": lam0.get_val(x[lam0.xpos]),
            "lambda1": lam1.get_val(x[lam1.xpos]),
        }
    except KeyError:
        pass

    # Goniometer
    try:
        gonio = {}
        if "gonio_theta" in ref_params:
            gt = ref_params["gonio_theta"]
            gp = ref_params["gonio_phi"]
            gonio["theta"] = gt.get_val(x[gt.xpos])
            gonio["phi"] = gp.get_val(x[gp.xpos])
            from simtbx.diffBragg.refiners.geometry import GoniometerParameters
            gonio["axis"] = list(GoniometerParameters.spherical_to_cartesian(gonio["theta"], gonio["phi"]))
        gonio["spindle_axis"] = list(SIM.D.spindle_axis) if SIM.D.spindle_axis is not None else None
        gonio["phi_deg"] = SIM.D.phi_deg if hasattr(SIM.D, 'phi_deg') else None
        gonio["osc_deg"] = SIM.D.osc_deg if hasattr(SIM.D, 'osc_deg') else None
        state["gonio"] = gonio
    except Exception:
        pass

    # Umatrix / Bmatrix from PAR
    try:
        state["crystal"]["Umatrix"] = list(Modeler.PAR.Umatrix.elems)
        state["crystal"]["Bmatrix"] = list(Modeler.PAR.Bmatrix.elems)
    except Exception:
        pass

    # Global image identifier (for cross-stage matching)
    try:
        state["meta"]["exper_name"] = getattr(Modeler, "exper_name", None)
    except Exception:
        pass

    return state


def write_state_snapshot(state, outdir, tag):
    """
    Write a single state dict to a JSON file.

    Parameters
    ----------
    state : dict
        State dictionary from capture_*_state
    outdir : str
        Output directory (will be created if needed)
    tag : str
        Filename tag, e.g. "hopper_main_rank0_shot0"
    """
    snap_dir = os.path.join(outdir, "state_snapshots")
    if not os.path.exists(snap_dir):
        os.makedirs(snap_dir)

    fname = os.path.join(snap_dir, "%s.json" % tag)

    # Convert all values to JSON-safe types
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return _to_json_safe(obj)

    safe_state = convert(state)
    with open(fname, "w") as f:
        json.dump(safe_state, f, indent=2)
    LOGGER.debug("Wrote state snapshot: %s" % fname)
    return fname


def compare_states(state_before, state_after, label="", atol=1e-8, rtol=1e-6):
    """
    Compare two state dicts and report differences.

    Parameters
    ----------
    state_before : dict
        State snapshot from earlier stage
    state_after : dict
        State snapshot from later stage
    label : str
        Label for printout
    atol : float
        Absolute tolerance for float comparison
    rtol : float
        Relative tolerance for float comparison

    Returns
    -------
    list of str : description of each difference found
    """
    diffs = []
    _compare_recursive(state_before, state_after, "", diffs, atol, rtol)

    if diffs:
        header = "STATE COMPARISON: %s (%d differences)" % (label, len(diffs))
        print("=" * len(header))
        print(header)
        print("=" * len(header))
        for d in diffs:
            print("  %s" % d)
        print("")
    else:
        print("STATE COMPARISON: %s — no differences" % label)

    return diffs


def _compare_recursive(a, b, path, diffs, atol, rtol):
    """Recursively compare nested dicts/lists."""
    if path.endswith("meta"):
        return  # skip meta (timestamps, stage names)

    if isinstance(a, dict) and isinstance(b, dict):
        all_keys = set(list(a.keys()) + list(b.keys()))
        for k in sorted(all_keys):
            if k == "meta":
                continue
            if k not in a:
                diffs.append("%s.%s: MISSING in before (present in after)" % (path, k))
            elif k not in b:
                diffs.append("%s.%s: MISSING in after (present in before)" % (path, k))
            else:
                _compare_recursive(a[k], b[k], "%s.%s" % (path, k), diffs, atol, rtol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append("%s: length mismatch %d vs %d" % (path, len(a), len(b)))
            return
        for i, (va, vb) in enumerate(zip(a, b)):
            _compare_recursive(va, vb, "%s[%d]" % (path, i), diffs, atol, rtol)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not _close(a, b, atol, rtol):
            diffs.append("%s: %.8g -> %.8g (delta=%.4g)" % (path, a, b, b - a))
    elif a != b:
        diffs.append("%s: %s -> %s" % (path, repr(a), repr(b)))


def _close(a, b, atol, rtol):
    """Check if two numbers are close."""
    if a == b:
        return True
    diff = abs(a - b)
    return diff <= atol or diff <= rtol * max(abs(a), abs(b))


def should_snapshot(params, i_shot, rank=0):
    """Check if this shot should be snapshotted."""
    if not getattr(params.logging, 'state_snapshots', False):
        return False
    max_shots = getattr(params.logging, 'state_snapshot_shots', 3)
    return i_shot < max_shots


def load_state_snapshot(outdir, tag):
    """Load a state snapshot from JSON."""
    fname = os.path.join(outdir, "state_snapshots", "%s.json" % tag)
    if not os.path.exists(fname):
        return None
    with open(fname, "r") as f:
        return json.load(f)
