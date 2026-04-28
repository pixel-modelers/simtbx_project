#!/usr/bin/env python
"""
Macro-cycling wrapper for hopper refinement + geometry optimization.

Runs alternating cycles of:
  1. Hopper per-image refinement
  2. Geometry optimization
  3. Update detector → repeat
  4. (Optional) Expand ROIs using in-memory DataModeler predictions

Usage:
  hopper_cycler.py input.txt [hopper_args] geometry.optimize=True geometry.macro_cycles=N
  hopper_cycler.py input.txt --mpi-cmd "mpirun -n 64" geometry.macro_cycles=3

Cycler-specific arguments:
  geometry.macro_cycles=N          Number of refinement cycles (default: 1)
  geometry.cycler_outdir=DIR       Parent output directory (creates DIR/cycle1, DIR/cycle2, etc.)
  --mpi-cmd "COMMAND"              MPI launcher (e.g., "mpirun -n 64", "srun -n 128")
  --mpi-cmd="COMMAND"              (equals format also supported)

In-place expansion arguments (uses in-memory DataModeler prediction expansion):
  --expand-after=N                 Enable expand_rois starting at cycle N
  --expand-threshold=0.01          Spot-finding threshold (fraction of max model intensity)
  --expand-max-new-rois=500        Max new predicted ROIs per shot
"""
from __future__ import division, print_function
import sys
import os
import subprocess
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


def parse_expand_args(args):
    """Extract --expand-* arguments for in-place prediction expansion."""
    expand_after = None
    expand_threshold = 0.01
    expand_max_new_rois = 500

    for arg in args:
        if arg.startswith("--expand-after="):
            expand_after = int(arg.split("=", 1)[1])
        elif arg.startswith("--expand-threshold="):
            expand_threshold = float(arg.split("=", 1)[1])
        elif arg.startswith("--expand-max-new-rois="):
            expand_max_new_rois = int(arg.split("=", 1)[1])

    return {
        'expand_after': expand_after,
        'threshold': expand_threshold,
        'max_new_rois': expand_max_new_rois,
    }


def remove_cycler_args(args):
    """Remove cycler-specific arguments (managed by cycler, not passed to hopper).

    Removes both full and abbreviated forms (phil auto-completion):
      geometry.macro_cycles=N / macro_cycles=N
      geometry.cycler_outdir=DIR / cycler_outdir=DIR
    Also removes expand arguments.
    """
    new_args = []
    skip_next = False

    expand_prefixes = (
        "--expand-after=", "--expand-threshold=", "--expand-max-new-rois=",
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

        # Remove expand args
        if any(arg.startswith(p) for p in expand_prefixes):
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


def generate_expanded_input(cycle_outdir, hopper_args):
    """Scan cycle output for _expanded.refl files and generate an expanded input.txt.

    Reads pandas pickles to map experiment basenames → optimized experiment paths,
    then writes expanded/input.txt with lines: <opt_exp_path> <refl_path> [spec]

    Uses _expanded.refl where available, falls back to regular .refl for shots
    that had no new predictions (so no shots are silently dropped).

    Returns path to new input.txt or None if no refls found.
    """
    import pandas

    refls_dir = os.path.join(cycle_outdir, "refls")

    # Find all .refl files, prefer _expanded over regular for each shot
    all_refls = sorted(_glob.glob(os.path.join(refls_dir, "rank*", "*.refl")))
    if not all_refls:
        print("  No .refl files found in %s" % refls_dir)
        return None

    # Build per-basename mapping: prefer _expanded.refl, fall back to regular
    # Key: experiment basename (e.g. "frame_000003"), Value: refl path
    basename_to_refl = {}
    for refl_path in all_refls:
        refl_basename = os.path.basename(refl_path)
        is_expanded = refl_basename.endswith("_expanded.refl")
        # Match against known basenames (done below after building basename_to_info)
        # For now store by full path, keyed by refl path
        if is_expanded:
            # Always prefer expanded
            basename_to_refl[refl_path] = ('expanded', refl_path)
        else:
            basename_to_refl[refl_path] = ('regular', refl_path)

    # Build mapping: basename → (opt_exp_path, spec) from pandas pickles
    pkl_globs = [
        os.path.join(cycle_outdir, "models_rank*.pkl"),
        os.path.join(cycle_outdir, "pandas", "hopper_results_rank*_chunk*.pkl"),
    ]
    basename_to_info = {}
    for pkl_glob in pkl_globs:
        for pkl_path in sorted(_glob.glob(pkl_glob)):
            try:
                df = pandas.read_pickle(pkl_path)
            except Exception:
                continue
            if 'opt_exp_name' not in df.columns:
                continue
            for _, row in df.iterrows():
                exp_name = row.get('exp_name', '')
                opt_exp = row.get('opt_exp_name', '')
                spec = row.get('spectrum_filename', '')
                if exp_name and opt_exp:
                    bname = os.path.basename(os.path.splitext(exp_name)[0])
                    basename_to_info[bname] = (opt_exp, spec if spec and str(spec) != 'nan' else '')
        if basename_to_info:
            break  # prefer geometry pkls

    # Also read the original input file for spec info as fallback
    original_input = _find_input_file(hopper_args)
    orig_input_map = {}
    if original_input and os.path.isfile(original_input):
        with open(original_input) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    bname = os.path.basename(os.path.splitext(parts[0])[0])
                    spec = parts[2] if len(parts) >= 3 else ''
                    orig_input_map[bname] = (parts[0], spec)

    # For each experiment basename, pick the best available refl file:
    # prefer _expanded.refl, fall back to regular .refl
    bname_to_best_refl = {}
    for refl_path in all_refls:
        refl_basename = os.path.basename(refl_path)
        is_expanded = refl_basename.endswith("_expanded.refl")
        for bname in list(basename_to_info.keys()) + list(orig_input_map.keys()):
            if bname in refl_basename:
                if bname not in bname_to_best_refl or is_expanded:
                    bname_to_best_refl[bname] = refl_path
                break

    if not bname_to_best_refl:
        print("  WARNING: Found %d .refl files but could not match any to experiments" % len(all_refls))
        return None

    # Write expanded/input.txt
    expand_dir = os.path.join(cycle_outdir, "expanded")
    os.makedirs(expand_dir, exist_ok=True)
    expanded_input = os.path.join(expand_dir, "input.txt")

    n_expanded = 0
    n_regular = 0
    with open(expanded_input, "w") as fout:
        for bname, refl_path in sorted(bname_to_best_refl.items()):
            # Find experiment path
            if bname in basename_to_info:
                exp_path, spec = basename_to_info[bname]
            elif bname in orig_input_map:
                exp_path, spec = orig_input_map[bname]
            else:
                continue

            abs_refl = os.path.abspath(refl_path)
            line = "%s %s" % (exp_path, abs_refl)
            if spec:
                line += " %s" % spec
            fout.write(line + "\n")
            if refl_path.endswith("_expanded.refl"):
                n_expanded += 1
            else:
                n_regular += 1

    n_total = n_expanded + n_regular
    if n_total == 0:
        print("  WARNING: No shots matched for expanded input")
        return None

    print("  Generated expanded input: %s (%d shots: %d expanded, %d regular fallback)"
          % (expanded_input, n_total, n_expanded, n_regular))
    return expanded_input


def generate_fresh_input(cycle_outdir, hopper_args):
    """Generate input.txt from cycle output when no _expanded.refl files exist.

    Uses opt_exp_name from pandas pickles + regular .refl files from cycle output.
    This ensures warm-start matching works in the next cycle (opt_exp_name paths
    match the input experiment paths).

    Returns path to new input.txt or None if unable to generate.
    """
    import pandas

    # Build mapping: basename → (opt_exp_path, spec) from pandas pickles
    pkl_globs = [
        os.path.join(cycle_outdir, "models_rank*.pkl"),
        os.path.join(cycle_outdir, "pandas", "hopper_results_rank*_chunk*.pkl"),
    ]
    basename_to_info = {}
    for pkl_glob in pkl_globs:
        for pkl_path in sorted(_glob.glob(pkl_glob)):
            try:
                df = pandas.read_pickle(pkl_path)
            except Exception:
                continue
            if 'opt_exp_name' not in df.columns:
                continue
            for _, row in df.iterrows():
                exp_name = row.get('exp_name', '')
                opt_exp = row.get('opt_exp_name', '')
                spec = row.get('spectrum_filename', '')
                if exp_name and opt_exp:
                    bname = os.path.basename(os.path.splitext(exp_name)[0])
                    basename_to_info[bname] = (opt_exp, spec if spec and str(spec) != 'nan' else '')
        if basename_to_info:
            break  # prefer geometry pkls

    if not basename_to_info:
        print("  No pandas pickles with opt_exp_name found in %s" % cycle_outdir)
        return None

    # Find regular (non-expanded) .refl files in refls/rank*/
    refls_dir = os.path.join(cycle_outdir, "refls")
    all_refls = sorted(_glob.glob(os.path.join(refls_dir, "rank*", "*.refl")))
    regular_refls = [r for r in all_refls if not r.endswith("_expanded.refl")]
    if not regular_refls:
        print("  No regular .refl files found in %s" % refls_dir)
        return None

    # Write fresh/input.txt
    fresh_dir = os.path.join(cycle_outdir, "fresh")
    os.makedirs(fresh_dir, exist_ok=True)
    fresh_input = os.path.join(fresh_dir, "input.txt")

    n_matched = 0
    with open(fresh_input, "w") as fout:
        for refl_path in regular_refls:
            refl_basename = os.path.basename(refl_path)
            matched_exp = None
            matched_spec = ''
            for bname, (exp_path, spec) in basename_to_info.items():
                if bname in refl_basename:
                    matched_exp = exp_path
                    matched_spec = spec
                    break
            if matched_exp is None:
                continue

            abs_refl = os.path.abspath(refl_path)
            line = "%s %s" % (matched_exp, abs_refl)
            if matched_spec:
                line += " %s" % matched_spec
            fout.write(line + "\n")
            n_matched += 1

    if n_matched == 0:
        print("  WARNING: Found %d regular .refl files but could not match any to experiments"
              % len(regular_refls))
        return None

    print("  Generated fresh input: %s (%d shots)" % (fresh_input, n_matched))
    return fresh_input


def update_exp_ref_spec_file(args, new_input_path):
    """Add/update exp_ref_spec_file phil param to point to expanded input.txt.

    This is the correct way to override the input file when a .phil file is the
    primary positional argument — replacing the .phil path would break hopper's
    argument parsing.
    """
    new_args = [a for a in args if not a.startswith("exp_ref_spec_file=")]
    new_args.append("exp_ref_spec_file=%s" % new_input_path)
    return new_args


def add_expand_phil_args(args, expand_params, load_from_refls=False):
    """Add/update prediction_expansion phil params in hopper args.

    Appends prediction_expansion.expand_rois=True, threshold, max_new_rois.
    If load_from_refls=True, also adds load_data_from_refls=True.
    """
    # Strip any existing expand/load phil params to avoid duplicates
    expand_phil_prefixes = (
        "prediction_expansion.expand_rois=",
        "prediction_expansion.threshold=",
        "prediction_expansion.max_new_rois=",
        "load_data_from_refls=",
    )
    new_args = [a for a in args if not any(a.startswith(p) for p in expand_phil_prefixes)]

    new_args.append("prediction_expansion.expand_rois=True")
    new_args.append("prediction_expansion.threshold=%g" % expand_params['threshold'])
    new_args.append("prediction_expansion.max_new_rois=%d" % expand_params['max_new_rois'])
    if load_from_refls:
        new_args.append("load_data_from_refls=True")

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


def _canonical_shot_id(shot_id):
    """Canonical shot identifier for matching across cycles.

    Since hopper.py now stores the original experiment basename as shot_id
    (propagated via orig_exp_name through warm-start), this is normally
    a pass-through. The regex fallback handles legacy summaries where
    expanded names (rank7_stage1_frame_000005_5_None.expt) leaked through.
    """
    import re
    # If it looks like an expanded name (contains 'stage1_'), extract the
    # original frame portion as fallback for pre-fix summaries.
    if 'stage1_' in shot_id:
        m = re.search(r'stage1_(.*?)_\d+_\w+\.expt', shot_id)
        if m:
            return m.group(1) + '.expt'
    return shot_id


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
                init_map = {_canonical_shot_id(s["shot_id"]): s["init_sigZ"] for s in h["per_shot"] if s["init_sigZ"] is not None}
                final_map = {_canonical_shot_id(s["shot_id"]): s["final_sigZ"] for s in h["per_shot"] if s["final_sigZ"] is not None}
                stages.append(("c%d_h_i" % cycle, init_map))
                stages.append(("c%d_h_f" % cycle, final_map))
            if g and g.get("per_shot_init"):
                init_map = {_canonical_shot_id(s["shot_id"]): s["sigZ"] for s in g["per_shot_init"]}
                final_map = {_canonical_shot_id(s["shot_id"]): s["sigZ"] for s in g.get("per_shot_final", [])}
                stages.append(("c%d_g_i" % cycle, init_map))
                stages.append(("c%d_g_f" % cycle, final_map))

        if stages:
            # Get all shot_ids (sorted for consistent display)
            all_shot_ids = sorted(set(sid for _, smap in stages for sid in smap))
            if all_shot_ids:
                # Truncate shot_id for display
                def short_id(s):
                    return s[-20:] if len(s) > 20 else s

                stage_labels = [lbl for lbl, _ in stages]
                col_w = max(8, max(len(l) for l in stage_labels))
                id_w = max(16, min(20, max(len(short_id(s)) for s in all_shot_ids)))

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

        # Per-shot sigZ split: original vs predicted ROIs (only if any cycle has predictions)
        _any_pred = False
        for cycle in all_cycles:
            h = hopper_by_cycle.get(cycle)
            if h and h.get("per_shot"):
                if any(s.get("sigZ_orig") is not None or s.get("init_sigZ_orig") is not None
                       for s in h["per_shot"]):
                    _any_pred = True
                    break
        if _any_pred:
            # Collect init and final split per cycle
            # Each entry: (label, {shot_id: (init_orig, init_pred, final_orig, final_pred)})
            split_stages = []
            for cycle in all_cycles:
                h = hopper_by_cycle.get(cycle)
                if h and h.get("per_shot"):
                    smap = {}
                    for s in h["per_shot"]:
                        sid = _canonical_shot_id(s["shot_id"])
                        smap[sid] = (s.get("init_sigZ_orig"), s.get("init_sigZ_pred"),
                                     s.get("sigZ_orig"), s.get("sigZ_pred"))
                    split_stages.append(("c%d" % cycle, smap))
            if split_stages:
                all_shot_ids_s = sorted(set(sid for _, smap in split_stages for sid in smap))
                id_w_s = max(16, min(20, max(len(short_id(s)) for s in all_shot_ids_s)))
                print("\nPer-shot sigZ split (original / predicted ROIs):")
                header_s = "  %-*s" % (id_w_s, "shot")
                for lbl, _ in split_stages:
                    header_s += " | %10s" % (lbl + "_i_or")
                    header_s += " | %10s" % (lbl + "_i_pr")
                    header_s += " | %10s" % (lbl + "_f_or")
                    header_s += " | %10s" % (lbl + "_f_pr")
                print("-" * len(header_s))
                print(header_s)
                print("-" * len(header_s))
                for sid in all_shot_ids_s:
                    row_s = "  %-*s" % (id_w_s, short_id(sid))
                    for _, smap in split_stages:
                        vals = smap.get(sid, (None, None, None, None))
                        for v in vals:
                            row_s += " | %10s" % ("%.4f" % v if v is not None else "-")
                    print(row_s)
                print("-" * len(header_s))

        # Show n_rois / n_trusted / %pred tracking across all stages
        # Each entry: (label, is_hopper, {canon_id: (n_rois, n_trusted, n_predicted)})
        roi_stages = []
        for cycle in all_cycles:
            h = hopper_by_cycle.get(cycle)
            g = geom_by_cycle.get(cycle)
            if h and h.get("per_shot"):
                roi_stages.append(("c%d_h" % cycle, True,
                    {_canonical_shot_id(s["shot_id"]):
                     (s.get("n_rois"), s.get("n_trusted"), s.get("n_predicted", 0))
                     for s in h["per_shot"]}))
            if g and g.get("per_shot_final"):
                roi_stages.append(("c%d_g" % cycle, False,
                    {_canonical_shot_id(s["shot_id"]):
                     (s.get("n_rois"), s.get("n_trusted"), 0)
                     for s in g["per_shot_final"]}))

        if roi_stages:
            print("\nPer-shot n_rois / n_trusted (%%pred) tracking:")
            rlabels = [l for l, _, _ in roi_stages]
            col_w = 16
            header2 = "  %-*s" % (id_w, "shot")
            for lbl in rlabels:
                header2 += " | %*s" % (col_w, lbl)
            print("-" * len(header2))
            print(header2)
            print("-" * len(header2))
            for sid in all_shot_ids:
                row = "  %-*s" % (id_w, short_id(sid))
                for _, is_hop, smap in roi_stages:
                    vals = smap.get(sid)
                    if vals is not None:
                        nr, nt, np_ = vals
                        if nr is not None:
                            if is_hop:
                                pct = 100.0 * np_ / nr if nr > 0 else 0
                                cell = "%d/%d (%2.1f%%)" % (nr, nt or 0, pct)
                            else:
                                cell = "%d/%d" % (nr, nt or 0)
                        else:
                            cell = "-"
                    else:
                        cell = "-"
                    row += " | %*s" % (col_w, cell)
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
    expand_params = parse_expand_args(hopper_args)

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
    expand_after = expand_params['expand_after']
    if expand_after is not None:
        print("Expand ROIs after cycle: %d (threshold=%g, max_new=%d)"
              % (expand_after, expand_params['threshold'], expand_params['max_new_rois']))
    print("Base arguments: %s" % " ".join(hopper_args))
    print("="*80)

    # Run cycles
    reference_geom = None
    best_pickle = None
    final_detector = None

    for cycle in range(1, n_cycles + 1):
        # --- Prediction expansion: add phil args for this cycle ---
        cycle_hopper_args = list(hopper_args)
        if expand_after is not None and cycle >= expand_after:
            load_from_refls = (cycle > expand_after)
            cycle_hopper_args = add_expand_phil_args(
                cycle_hopper_args, expand_params, load_from_refls=load_from_refls)
            if load_from_refls:
                print("Expansion: cycle %d will load from expanded .refl (load_data_from_refls=True)" % cycle)
            else:
                print("Expansion: cycle %d will predict new ROIs (expand_rois=True)" % cycle)

        detector_path = run_hopper_cycle(cycle_hopper_args, cycle, reference_geom, best_pickle,
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

        # In-place expansion hook: collect _expanded.refl files and generate new input.txt
        if expand_after is not None and cycle >= expand_after and cycle < n_cycles:
            _, base_outdir_e = update_outdir(hopper_args, cycle, cycler_outdir)
            cycle_outdir_e = _get_cycle_outdir(cycler_outdir, base_outdir_e, cycle)
            try:
                expanded_input = generate_expanded_input(cycle_outdir_e, hopper_args)
                if expanded_input is not None:
                    hopper_args = update_exp_ref_spec_file(hopper_args, expanded_input)
                    print("Updated input file for next cycle: %s" % expanded_input)
                else:
                    # No expanded refls → generate fresh input from cycle output
                    # so warm-start matching works (opt_exp_name paths match)
                    fresh_input = generate_fresh_input(cycle_outdir_e, hopper_args)
                    if fresh_input is not None:
                        hopper_args = update_exp_ref_spec_file(hopper_args, fresh_input)
                        print("Updated input file for next cycle (from regular refls): %s" % fresh_input)
            except Exception as e:
                import traceback
                print("WARNING: Expansion input generation failed: %s" % e)
                traceback.print_exc()
                print("Continuing with previous input file")

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
