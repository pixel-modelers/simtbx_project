#!/usr/bin/env python
"""
Macro-cycling wrapper for hopper refinement + geometry optimization.

Runs alternating cycles of:
  1. Hopper per-image refinement
  2. Geometry optimization
  3. Update detector → repeat
  4. (Optional) Predict new reflections from refined model and expand

Usage:
  hopper_cycler.py input.txt [hopper_args] geometry.optimize=True geometry.macro_cycles=N
  hopper_cycler.py input.txt --mpi-cmd "mpirun -n 64" geometry.macro_cycles=3

Cycler-specific arguments:
  geometry.macro_cycles=N          Number of refinement cycles (default: 1)
  geometry.cycler_outdir=DIR       Parent output directory (creates DIR/cycle1, DIR/cycle2, etc.)
  --mpi-cmd "COMMAND"              MPI launcher (e.g., "mpirun -n 64", "srun -n 128")
  --mpi-cmd="COMMAND"              (equals format also supported)

Prediction arguments:
  --predict-after=N                Predict after cycle N (expand refls for remaining cycles)
  --predict-threshold=1e-3         Peak intensity threshold for model prediction
  --predict-d-min=None             Resolution limit for predictions (Angstrom)
  --predict-weak-fraction=0.5      Fraction of weak predictions to include (0-1)
  --predict-filter-mode=MODE       Filter: none, intensity, scatter_binned, signal
  --predict-signal-filter          (Deprecated) Equivalent to --predict-filter-mode=signal
  --predict-signal-threshold=2.0   I/sqrt(bg) threshold for signal filter mode
  --predict-compare-filters        Run all filter modes and dump comparison CSV
"""
from __future__ import division, print_function
import sys
import os
import subprocess
import argparse
import shlex
import glob as _glob
from simtbx.diffBragg import utils


def parse_cycles_from_args(args):
    """Extract geometry.macro_cycles from command line args.

    Accepts both full and abbreviated forms (phil auto-completion):
      geometry.macro_cycles=3
      macro_cycles=3
    """
    cycles = 1
    for arg in args:
        if arg.startswith("geometry.macro_cycles=") or arg.startswith("macro_cycles="):
            cycles = int(arg.split("=")[1])
            break
    return cycles


def parse_cycler_outdir(args):
    """Extract geometry.cycler_outdir if present.

    Accepts both full and abbreviated forms (phil auto-completion):
      geometry.cycler_outdir=DIR
      cycler_outdir=DIR
    """
    for arg in args:
        if arg.startswith("geometry.cycler_outdir=") or arg.startswith("cycler_outdir="):
            return arg.split("=", 1)[1]
    return None


def parse_mpi_cmd(args):
    """Extract --mpi-cmd if present.

    Handles both formats:
      --mpi-cmd="mpirun -n 12"
      --mpi-cmd "mpirun -n 12"
    """
    for i, arg in enumerate(args):
        if arg.startswith("--mpi-cmd="):
            return arg.split("=", 1)[1]
        elif arg == "--mpi-cmd" and i + 1 < len(args):
            return args[i + 1]
    return None


def parse_predict_args(args):
    """Extract prediction-specific arguments."""
    predict_after = None
    predict_threshold = 1e-3
    predict_d_min = None
    predict_weak_fraction = 0.5
    predict_filter_mode = "none"
    predict_signal_threshold = 2.0
    predict_compare = False
    predict_score_state_file = None
    predict_score_cutoff = 0.5

    for i, arg in enumerate(args):
        if arg.startswith("--predict-after="):
            predict_after = int(arg.split("=", 1)[1])
        elif arg.startswith("--predict-threshold="):
            predict_threshold = float(arg.split("=", 1)[1])
        elif arg.startswith("--predict-d-min="):
            val = arg.split("=", 1)[1]
            predict_d_min = float(val) if val.lower() != "none" else None
        elif arg.startswith("--predict-weak-fraction="):
            predict_weak_fraction = float(arg.split("=", 1)[1])
        elif arg.startswith("--predict-filter-mode="):
            predict_filter_mode = arg.split("=", 1)[1]
        elif arg == "--predict-signal-filter":
            predict_filter_mode = "signal"  # backward compat
        elif arg.startswith("--predict-signal-threshold="):
            predict_signal_threshold = float(arg.split("=", 1)[1])
        elif arg == "--predict-compare-filters":
            predict_compare = True
        elif arg.startswith("--predict-score-state-file="):
            predict_score_state_file = arg.split("=", 1)[1]
        elif arg.startswith("--predict-score-cutoff="):
            predict_score_cutoff = float(arg.split("=", 1)[1])

    return {
        'predict_after': predict_after,
        'threshold': predict_threshold,
        'd_min': predict_d_min,
        'weak_fraction': predict_weak_fraction,
        'filter_mode': predict_filter_mode,
        'signal_threshold': predict_signal_threshold,
        'compare_filters': predict_compare,
        'score_state_file': predict_score_state_file,
        'score_cutoff': predict_score_cutoff,
    }


def remove_cycler_args(args):
    """Remove cycler-specific arguments (managed by cycler, not passed to hopper).

    Removes both full and abbreviated forms (phil auto-completion):
      geometry.macro_cycles=N / macro_cycles=N
      geometry.cycler_outdir=DIR / cycler_outdir=DIR
    Also removes prediction arguments.
    """
    new_args = []
    skip_next = False

    predict_prefixes = (
        "--predict-after=", "--predict-threshold=", "--predict-d-min=",
        "--predict-weak-fraction=", "--predict-filter-mode=",
        "--predict-signal-filter", "--predict-signal-threshold=",
        "--predict-compare-filters",
        "--predict-score-state-file=", "--predict-score-cutoff=",
    )

    for i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue

        # Remove =value format (full and abbreviated forms)
        if (arg.startswith("geometry.macro_cycles=") or
            arg.startswith("macro_cycles=") or
            arg.startswith("geometry.cycler_outdir=") or
            arg.startswith("cycler_outdir=") or
            arg.startswith("--mpi-cmd=")):
            continue

        # Remove prediction args
        if any(arg.startswith(p) for p in predict_prefixes):
            continue

        # Remove --flag value format (space-separated)
        if arg == "--mpi-cmd":
            skip_next = True  # Skip the next argument (the value)
            continue

        new_args.append(arg)

    return new_args


def update_outdir(args, cycle_num, cycler_outdir=None):
    """Update outdir parameter for the current cycle.

    If cycler_outdir is set: uses cycler_outdir/cycle1, cycler_outdir/cycle2, etc.
    Otherwise: uses outdir_cycle1, outdir_cycle2, etc. (backward compatible)
    """
    new_args = []
    outdir_found = False
    base_outdir = None

    # Remove any existing outdir parameter
    for arg in args:
        if not arg.startswith("outdir="):
            new_args.append(arg)
        else:
            base_outdir = arg.split("=", 1)[1]
            outdir_found = True

    # Determine output directory for this cycle
    if cycler_outdir is not None:
        # Use cycler_outdir/cycleN structure
        new_outdir = os.path.join(cycler_outdir, "cycle%d" % cycle_num)
        base_outdir = cycler_outdir
    else:
        # Backward compatible: outdir_cycleN structure
        if base_outdir is None:
            base_outdir = "hopper_output"
        new_outdir = "%s_cycle%d" % (base_outdir, cycle_num)

    new_args.append("outdir=%s" % new_outdir)
    return new_args, base_outdir


def add_reference_geom(args, detector_path):
    """Add or update refiner.reference_geom parameter."""
    new_args = []
    for arg in args:
        # Skip existing reference_geom
        if not arg.startswith("refiner.reference_geom="):
            new_args.append(arg)
    # Add new reference
    new_args.append("refiner.reference_geom=%s" % detector_path)
    return new_args


def add_best_pickle(args, pkl_glob):
    """Add or update best_pickle parameter to warm-start from previous cycle."""
    new_args = []
    for arg in args:
        if not arg.startswith("best_pickle="):
            new_args.append(arg)
    new_args.append("best_pickle=%s" % pkl_glob)
    return new_args


def _find_mtz_path(args):
    """Extract simulator.structure_factors.mtz_name from hopper args or .phil files."""
    # First check direct args
    for arg in args:
        if "structure_factors.mtz_name=" in arg:
            return arg.split("=", 1)[1]
    # Fall back: search inside .phil files referenced in args
    for arg in args:
        if arg.endswith(".phil") and os.path.isfile(arg):
            try:
                with open(arg) as f:
                    for line in f:
                        line = line.strip()
                        if "mtz_name" in line and "=" in line:
                            val = line.split("=", 1)[1].strip().strip('"').strip("'")
                            if val:
                                return val
            except Exception:
                pass
    return None


def _find_mtz_column(args):
    """Extract simulator.structure_factors.mtz_column from hopper args or .phil files."""
    # First check direct args
    for arg in args:
        if "structure_factors.mtz_column=" in arg:
            return arg.split("=", 1)[1]
    # Fall back: search inside .phil files referenced in args
    for arg in args:
        if arg.endswith(".phil") and os.path.isfile(arg):
            try:
                with open(arg) as f:
                    for line in f:
                        line = line.strip()
                        if "mtz_column" in line and "=" in line:
                            val = line.split("=", 1)[1].strip().strip('"').strip("'")
                            if val:
                                return val
            except Exception:
                pass
    return None


def update_mtz_in_args(args, mtz_path, mtz_column="FP,SIGFP"):
    """Update MTZ path and column in hopper args for the next cycle.

    Default column label "FP,SIGFP" matches the label_string() produced by
    as_mtz_dataset(column_root_label="FP") — open_mtz() does exact matching.
    """
    new_args = []
    for arg in args:
        if "structure_factors.mtz_name=" not in arg and "structure_factors.mtz_column=" not in arg:
            new_args.append(arg)
    new_args.append("simulator.structure_factors.mtz_name=%s" % mtz_path)
    new_args.append("simulator.structure_factors.mtz_column=%s" % mtz_column)
    return new_args


def run_prediction(cycle_outdir, predict_params, input_file=None):
    """Run prediction on a completed cycle and return path to expanded directory.

    Uses hopper_predict to simulate from refined model, extract new predictions,
    merge with observed reflections, and write expanded .refl files.
    """
    from simtbx.command_line.hopper_predict import predict_and_expand

    class PredictArgs:
        pass

    args = PredictArgs()
    args.threshold = predict_params['threshold']
    args.d_min = predict_params['d_min']
    args.weak_fraction = predict_params['weak_fraction']
    args.filter_mode = predict_params['filter_mode']
    args.signal_filter = (predict_params['filter_mode'] == 'signal')
    args.signal_threshold = predict_params['signal_threshold']
    args.compare_filters = predict_params.get('compare_filters', False)
    args.score_state_file = predict_params.get('score_state_file', None)
    args.score_cutoff = predict_params.get('score_cutoff', 0.5)
    args.expand_dir = "expanded"
    args.q_cutoff = 0.005
    args.input_file = input_file
    args.shoebox_size = 10
    args.phil = predict_params.get('phil', None)
    args.signal_check = predict_params.get('signal_check', False)

    print("\n" + "="*80)
    print("PREDICTION STEP")
    print("="*80)
    print("Cycle output: %s" % cycle_outdir)
    print("Threshold: %g" % args.threshold)
    if args.d_min is not None:
        print("Resolution limit: %.2f A" % args.d_min)
    print("Weak fraction: %.2f" % args.weak_fraction)
    print("Filter mode: %s" % args.filter_mode)

    expand_dir = predict_and_expand(cycle_outdir, args)
    return expand_dir


def update_input_file_path(args, new_input_path):
    """Replace the input file (first positional argument) with expanded version."""
    new_args = list(args)
    # The first argument that looks like a file (not a phil param) is the input file
    for i, arg in enumerate(new_args):
        if "=" not in arg and not arg.startswith("--") and os.path.isfile(arg):
            new_args[i] = new_input_path
            break
    else:
        # If no file found, also check if first arg could be the input file reference
        # even if file doesn't exist yet (for the expanded path)
        for i, arg in enumerate(new_args):
            if "=" not in arg and not arg.startswith("--"):
                new_args[i] = new_input_path
                break
    return new_args


def run_hopper_cycle(args, cycle_num, reference_geom=None, best_pickle=None,
                     cycler_outdir=None, mpi_cmd=None):
    """Run a single hopper cycle."""
    print("\n" + "="*80)
    print("MACRO-CYCLE %d" % cycle_num)
    print("="*80)

    # Update output directory
    cycle_args, base_outdir = update_outdir(args, cycle_num, cycler_outdir)

    # Add reference geometry if provided
    if reference_geom is not None:
        print("Using reference geometry: %s" % reference_geom)
        cycle_args = add_reference_geom(cycle_args, reference_geom)

    # Warm-start from previous cycle's refined per-shot parameters
    if best_pickle is not None:
        print("Warm-starting from previous cycle: %s" % best_pickle)
        cycle_args = add_best_pickle(cycle_args, best_pickle)

    # Build command with optional MPI prefix
    if mpi_cmd is not None:
        mpi_tokens = shlex.split(mpi_cmd)
        cmd = mpi_tokens + ["hopper.py"] + cycle_args
    else:
        cmd = ["hopper.py"] + cycle_args

    print("\nRunning: %s\n" % " ".join(cmd))

    # Run hopper
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("\nERROR: hopper.py failed with exit code %d" % result.returncode)
        sys.exit(result.returncode)

    # Check for optimized detector
    if cycler_outdir is not None:
        outdir = os.path.join(cycler_outdir, "cycle%d" % cycle_num)
    else:
        outdir = "%s_cycle%d" % (base_outdir, cycle_num)
    detector_path = os.path.join(outdir, "diffBragg_detector.expt")

    if os.path.exists(detector_path):
        print("\nOptimized detector saved to: %s" % detector_path)
        return detector_path
    else:
        print("\nWARNING: No optimized detector found at %s" % detector_path)
        print("Was geometry.optimize=True set?")
        return None


def _get_cycle_outdir(cycler_outdir, base_outdir, cycle_num):
    """Determine the output directory path for a given cycle number."""
    if cycler_outdir is not None:
        return os.path.join(cycler_outdir, "cycle%d" % cycle_num)
    else:
        return "%s_cycle%d" % (base_outdir, cycle_num)


def _find_input_file(args):
    """Find the input file path from the hopper args (first file-like argument).

    Skips .phil files (parameter files, not input.txt).
    Also tries to read exp_ref_spec_file from .phil if that's the only file found.
    """
    phil_path = None
    for arg in args:
        if "=" not in arg and not arg.startswith("--") and os.path.isfile(arg):
            if arg.endswith(".phil"):
                phil_path = arg
                continue
            return arg

    # If only a .phil was found, try to extract exp_ref_spec_file from it
    if phil_path is not None:
        try:
            with open(phil_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("exp_ref_spec_file"):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if os.path.isfile(val):
                            return val
        except Exception:
            pass

    return None


def _print_cycle_summary(hopper_args, n_cycles, cycler_outdir):
    """Read hopper_summary.json and geometry_summary.json from each cycle and print evolution tables."""
    import json

    hopper_summaries = []
    geom_summaries = []
    for cycle in range(1, n_cycles + 1):
        cycle_outdir = _get_cycle_outdir(cycler_outdir, None, cycle)
        if cycler_outdir is None:
            _, base_outdir = update_outdir(hopper_args, cycle, cycler_outdir)
            cycle_outdir = _get_cycle_outdir(None, base_outdir, cycle)

        hopper_path = os.path.join(cycle_outdir, "hopper_summary.json")
        if os.path.exists(hopper_path):
            with open(hopper_path) as fh:
                hopper_summaries.append((cycle, json.load(fh)))

        geom_path = os.path.join(cycle_outdir, "geometry_summary.json")
        if os.path.exists(geom_path):
            with open(geom_path) as fh:
                geom_summaries.append((cycle, json.load(fh)))

    # Hopper (per-shot) evolution table
    if hopper_summaries:
        print("\nPer-shot hopper evolution:")
        print("-" * 80)
        print("  Cycle | N_shots | pred_offset (init -> final)  | sigZ (init -> final)")
        print("-" * 80)
        for cycle, h in hopper_summaries:
            n = h.get("n_shots", 0)
            sz_i = h.get("median_init_sigZ")
            sz_f = h.get("median_final_sigZ")
            po_i = h.get("median_init_pred_offset")
            po_f = h.get("median_final_pred_offset")
            sz_delta = (sz_f - sz_i) if (sz_i is not None and sz_f is not None) else None
            po_delta = (po_f - po_i) if (po_i is not None and po_f is not None) else None

            sz_str = "%.4f -> %.4f (%+.4f)" % (sz_i, sz_f, sz_delta) if sz_delta is not None else "N/A"
            po_str = "%.4f -> %.4f (%+.4f)" % (po_i, po_f, po_delta) if po_delta is not None else "N/A"
            print("  %5d | %7d | %-28s | %s" % (cycle, n, po_str, sz_str))
        print("-" * 80)

    # Geometry evolution table
    if geom_summaries:
        print("\nGeometry refinement evolution:")
        print("-" * 90)
        print("  Cycle | pred_offset (init -> final)  | sigZ (init -> final)       | Resid (init -> final)")
        print("-" * 90)
        for cycle, s in geom_summaries:
            po_i = s.get("initial_pred_offset")
            po_f = s.get("final_pred_offset")
            sz_i = s.get("initial_sigZ")
            sz_f = s.get("final_sigZ")
            r_i = s.get("initial_resid")
            r_f = s.get("final_resid")
            po_delta = (po_f - po_i) if (po_i is not None and po_f is not None) else None
            sz_delta = (sz_f - sz_i) if (sz_i is not None and sz_f is not None) else None
            r_delta = (r_f - r_i) if (r_i is not None and r_f is not None) else None

            po_str = "%.4f -> %.4f (%+.4f)" % (po_i, po_f, po_delta) if po_delta is not None else "N/A"
            sz_str = "%.4f -> %.4f (%+.4f)" % (sz_i, sz_f, sz_delta) if sz_delta is not None else "N/A"
            r_str = "%.0f -> %.0f (%+.0f)" % (r_i, r_f, r_delta) if r_delta is not None else "N/A"
            print("  %5d | %-28s | %-26s | %s" % (cycle, po_str, sz_str, r_str))
        print("-" * 90)

    # Full linear track: hopper_init -> hopper_final -> geom_init -> geom_final for each cycle
    if hopper_summaries or geom_summaries:
        geom_by_cycle = {c: g for c, g in geom_summaries}
        hopper_by_cycle = {c: h for c, h in hopper_summaries}
        all_cycles = sorted(set(c for c, _ in hopper_summaries) | set(c for c, _ in geom_summaries))

        print("\nFull pipeline track (pred_offset / sigZ):")
        print("-" * 72)
        print("  %-28s | %10s | %10s" % ("Stage", "pred_off", "sigZ"))
        print("-" * 72)
        for cycle in all_cycles:
            h = hopper_by_cycle.get(cycle)
            g = geom_by_cycle.get(cycle)
            if h:
                po_i = h.get("median_init_pred_offset")
                sz_i = h.get("median_init_sigZ")
                po_f = h.get("median_final_pred_offset")
                sz_f = h.get("median_final_sigZ")
                po_i_s = "%.4f" % po_i if po_i is not None else "N/A"
                sz_i_s = "%.4f" % sz_i if sz_i is not None else "N/A"
                po_f_s = "%.4f" % po_f if po_f is not None else "N/A"
                sz_f_s = "%.4f" % sz_f if sz_f is not None else "N/A"
                print("  %-28s | %10s | %10s" % ("cycle %d hopper init" % cycle, po_i_s, sz_i_s))
                print("  %-28s | %10s | %10s" % ("cycle %d hopper final" % cycle, po_f_s, sz_f_s))
            if g:
                po_i = g.get("initial_pred_offset")
                sz_i = g.get("initial_sigZ")
                po_f = g.get("final_pred_offset")
                sz_f = g.get("final_sigZ")
                po_i_s = "%.4f" % po_i if po_i is not None else "N/A"
                sz_i_s = "%.4f" % sz_i if sz_i is not None else "N/A"
                po_f_s = "%.4f" % po_f if po_f is not None else "N/A"
                sz_f_s = "%.4f" % sz_f if sz_f is not None else "N/A"
                print("  %-28s | %10s | %10s" % ("cycle %d geometry init" % cycle, po_i_s, sz_i_s))
                print("  %-28s | %10s | %10s" % ("cycle %d geometry final" % cycle, po_f_s, sz_f_s))
        print("-" * 72)

    # Per-shot sigZ tracking across stages
    if hopper_summaries or geom_summaries:
        # Collect per-shot sigZ at each stage boundary, keyed by shot_id
        # stages: list of (stage_label, {shot_id: sigZ})
        stages = []
        for cycle in all_cycles:
            h = hopper_by_cycle.get(cycle)
            g = geom_by_cycle.get(cycle)
            if h and h.get("per_shot"):
                init_map = {s["shot_id"]: s["init_sigZ"] for s in h["per_shot"] if s["init_sigZ"] is not None}
                final_map = {s["shot_id"]: s["final_sigZ"] for s in h["per_shot"] if s["final_sigZ"] is not None}
                stages.append(("c%d_h_i" % cycle, init_map))
                stages.append(("c%d_h_f" % cycle, final_map))
            if g and g.get("per_shot_init"):
                init_map = {s["shot_id"]: s["sigZ"] for s in g["per_shot_init"]}
                final_map = {s["shot_id"]: s["sigZ"] for s in g.get("per_shot_final", [])}
                stages.append(("c%d_g_i" % cycle, init_map))
                stages.append(("c%d_g_f" % cycle, final_map))

        if stages:
            # Get all shot_ids (sorted for consistent display)
            all_shot_ids = sorted(set(sid for _, smap in stages for sid in smap))
            if all_shot_ids:
                # Truncate shot_id for display (take last 12 chars)
                def short_id(s):
                    return s[-16:] if len(s) > 16 else s

                stage_labels = [lbl for lbl, _ in stages]
                col_w = max(8, max(len(l) for l in stage_labels))
                id_w = min(16, max(len(short_id(s)) for s in all_shot_ids))

                print("\nPer-shot sigZ tracking:")
                header = "  %-*s" % (id_w, "shot")
                for lbl in stage_labels:
                    header += " | %*s" % (col_w, lbl)
                print("-" * len(header))
                print(header)
                print("-" * len(header))
                for sid in all_shot_ids:
                    row = "  %-*s" % (id_w, short_id(sid))
                    for _, smap in stages:
                        val = smap.get(sid)
                        row += " | %*s" % (col_w, "%.4f" % val if val is not None else "-")
                    print(row)
                print("-" * len(header))

        # Show n_rois / n_trusted differences at hopper->geometry transitions
        # Collect per-shot n_rois and n_trusted at each stage
        roi_stages = []   # (label, {shot_id: n_rois})
        trust_stages = []  # (label, {shot_id: n_trusted})
        for cycle in all_cycles:
            h = hopper_by_cycle.get(cycle)
            g = geom_by_cycle.get(cycle)
            if h and h.get("per_shot"):
                roi_stages.append(("c%d_hop" % cycle,
                    {s["shot_id"]: s.get("n_rois") for s in h["per_shot"]}))
                trust_stages.append(("c%d_hop" % cycle,
                    {s["shot_id"]: s.get("n_trusted") for s in h["per_shot"]}))
            if g and g.get("per_shot_init"):
                roi_stages.append(("c%d_geo" % cycle,
                    {s["shot_id"]: s.get("n_rois") for s in g["per_shot_init"]}))
                trust_stages.append(("c%d_geo" % cycle,
                    {s["shot_id"]: s.get("n_trusted") for s in g["per_shot_init"]}))

        if len(roi_stages) >= 2:
            print("\nPer-shot n_rois / n_trusted at each stage:")
            rlabels = [l for l, _ in roi_stages]
            col_w = 14
            header2 = "  %-*s" % (id_w, "shot")
            for lbl in rlabels:
                header2 += " | %*s" % (col_w, lbl)
            print("-" * len(header2))
            print(header2)
            print("-" * len(header2))
            for sid in all_shot_ids:
                row = "  %-*s" % (id_w, short_id(sid))
                for (_, rmap), (_, tmap) in zip(roi_stages, trust_stages):
                    nr = rmap.get(sid)
                    nt = tmap.get(sid)
                    nr_s = "%d" % nr if nr is not None else "?"
                    nt_s = "%d" % nt if nt is not None else "?"
                    row += " | %*s" % (col_w, "%s/%s" % (nr_s, nt_s))
                print(row)
            print("-" * len(header2))

    elif geom_summaries:
        # Fallback: geometry-only overall summary
        first = geom_summaries[0][1]
        last = geom_summaries[-1][1]
        if first.get("initial_pred_offset") is not None and last.get("final_pred_offset") is not None:
            total_po = last["final_pred_offset"] - first["initial_pred_offset"]
            print("  Overall pred_offset: %.4f -> %.4f (%+.4f pixels)"
                  % (first["initial_pred_offset"], last["final_pred_offset"], total_po))
        if first.get("initial_sigZ") is not None and last.get("final_sigZ") is not None:
            total_sz = last["final_sigZ"] - first["initial_sigZ"]
            print("  Overall sigZ:        %.4f -> %.4f (%+.4f)"
                  % (first["initial_sigZ"], last["final_sigZ"], total_sz))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # Parse arguments
    hopper_args = sys.argv[1:]

    # Extract cycler-specific parameters
    n_cycles = parse_cycles_from_args(hopper_args)
    cycler_outdir = parse_cycler_outdir(hopper_args)
    mpi_cmd = parse_mpi_cmd(hopper_args)
    predict_params = parse_predict_args(hopper_args)

    # Remove cycler args (hopper doesn't know about them)
    hopper_args = remove_cycler_args(hopper_args)

    # Ensure geometry optimization is enabled
    if not any("geometry.optimize=True" in arg for arg in hopper_args):
        print("\nWARNING: Adding geometry.optimize=True (required for macro-cycling)")
        hopper_args.append("geometry.optimize=True")

    print("\n" + "="*80)
    print("HOPPER MACRO-CYCLING")
    print("="*80)
    print("Number of cycles: %d" % n_cycles)
    if cycler_outdir is not None:
        print("Cycler output directory: %s" % cycler_outdir)
    if mpi_cmd is not None:
        print("MPI command: %s" % mpi_cmd)
    if predict_params['predict_after'] is not None:
        print("Predict after cycle: %d" % predict_params['predict_after'])
    print("Base arguments: %s" % " ".join(hopper_args))
    print("="*80)

    # Run cycles
    reference_geom = None
    best_pickle = None
    final_detector = None
    predict_after = predict_params['predict_after']
    original_input_file = _find_input_file(hopper_args)

    for cycle in range(1, n_cycles + 1):
        detector_path = run_hopper_cycle(hopper_args, cycle, reference_geom, best_pickle,
                                         cycler_outdir, mpi_cmd)
        final_detector = detector_path  # Track the last detector produced

        # Update reference for next cycle
        if detector_path is not None and cycle < n_cycles:
            reference_geom = detector_path
        elif detector_path is None and cycle < n_cycles:
            print("\nERROR: Cannot continue to cycle %d without optimized detector" % (cycle + 1))
            sys.exit(1)

        # Build best_pickle glob for next cycle's warm-start
        # Prefer geometry output pandas (latest state with geometry corrections baked in),
        # fall back to hopper pandas if geometry didn't run or didn't produce output.
        if cycle < n_cycles:
            cycle_outdir = _get_cycle_outdir(cycler_outdir, None, cycle)
            if cycler_outdir is None:
                _, base_outdir = update_outdir(hopper_args, cycle, cycler_outdir)
                cycle_outdir = _get_cycle_outdir(None, base_outdir, cycle)
            geom_pkl_glob = os.path.join(cycle_outdir, "models_rank*.pkl")
            hopper_pkl_glob = os.path.join(cycle_outdir, "pandas", "hopper_results_rank*_chunk*.pkl")
            if _glob.glob(geom_pkl_glob):
                best_pickle = geom_pkl_glob
                print("Will warm-start next cycle from geometry output: %s" % best_pickle)
            elif _glob.glob(hopper_pkl_glob):
                best_pickle = hopper_pkl_glob
                print("Will warm-start next cycle from hopper output: %s" % best_pickle)
            else:
                print("WARNING: No pandas pickles found — next cycle starts fresh")
                best_pickle = None

            # Update MTZ with per-ROI scale corrections (fold into Fhkl for next cycle)
            mtz_path = _find_mtz_path(hopper_args)
            if mtz_path is not None:
                # Prefer geometry output pkls (latest state), fall back to hopper pkls
                geom_pkls = sorted(_glob.glob(os.path.join(cycle_outdir, "models_rank*.pkl")))
                if not geom_pkls:
                    geom_pkls = sorted(_glob.glob(hopper_pkl_glob)) if _glob.glob(hopper_pkl_glob) else []
                if geom_pkls:
                    updated_mtz = os.path.join(cycle_outdir, "optimized_Fhkl.mtz")
                    mtz_col = _find_mtz_column(hopper_args)
                    try:
                        from simtbx.diffBragg.hopper_io import update_mtz_with_roi_scales
                        result = update_mtz_with_roi_scales(
                            mtz_path, updated_mtz, geom_pkls, mtz_column=mtz_col)
                        if result["n_corrected"] > 0:
                            anom_flag = " (anomalous)" if result.get("is_anomalous", False) else ""
                            print("MTZ updated: %d/%d HKLs corrected, median scale=%.4f -> %s%s"
                                  % (result["n_corrected"], result["n_total_mtz"],
                                     result["median_correction"], updated_mtz, anom_flag))
                            # Use the column name from the result (handles anomalous vs merged)
                            output_col = result.get("mtz_column", "FP,SIGFP")
                            hopper_args = update_mtz_in_args(hopper_args, updated_mtz, mtz_column=output_col)
                        else:
                            print("MTZ update: no perRoiScale corrections found, keeping original MTZ")
                    except Exception as e:
                        import traceback
                        print("WARNING: MTZ update failed: %s" % e)
                        traceback.print_exc()
                        print("Continuing with previous MTZ")

        # Prediction hook: after the specified cycle, predict and expand reflections
        if predict_after is not None and cycle == predict_after and cycle < n_cycles:
            # Determine cycle output directory
            # Parse base_outdir for path construction
            _, base_outdir = update_outdir(hopper_args, cycle, cycler_outdir)
            cycle_outdir_path = _get_cycle_outdir(cycler_outdir, base_outdir, cycle)

            try:
                expand_dir = run_prediction(
                    cycle_outdir_path, predict_params,
                    input_file=original_input_file)

                # Update hopper args to use expanded reflections
                expanded_input = os.path.join(expand_dir, "input.txt")
                if os.path.isfile(expanded_input):
                    hopper_args = update_input_file_path(hopper_args, expanded_input)
                    print("\nUpdated input file to: %s" % expanded_input)
                else:
                    print("\nWARNING: Expanded input.txt not found at %s" % expanded_input)
            except Exception as e:
                import traceback
                print("\nWARNING: Prediction failed: %s" % e)
                traceback.print_exc()
                print("Continuing with original reflections")

    # Collect and print geometry refinement summary
    print("\n" + "="*80)
    print("MACRO-CYCLING COMPLETE - %d cycles finished" % n_cycles)
    print("="*80)
    if final_detector is not None:
        print("Final optimized detector: %s" % final_detector)

    _print_cycle_summary(hopper_args, n_cycles, cycler_outdir)
    print()


if __name__ == "__main__":
    main()
