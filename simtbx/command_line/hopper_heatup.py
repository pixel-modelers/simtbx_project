#!/usr/bin/env python
"""
Automated hopper warm-up: find optimal refinement sigmas.

Runs hopper on a small subset of frames with different sigma values
for each parameter, scoring each trial by median sigZ. Builds up
parameter-by-parameter so each stage inherits the best settings from
the previous stage.

Stages (standard 2-pass pipeline):
  Pass 1 (ballpark): Quick sigma sweep with inherited model params
    1. G estimation: forward model at G=1, compute data/model ratio
    2. G sigma sweep: fix everything else, sweep sigmas.G
    3. RotXYZ sigma sweep: best G sigma, sweep sigmas.RotXYZ
    4. Nabc/cholesky sigma sweep: sweep sigmas.Nabc or sigmas.cholesky
    5. B sigma sweep: sweep sigmas.B
  Model tuning: Sweep phisteps + sigma_r using Pass 1 sigmas
  Pass 2 (refined): Re-sweep all sigmas with tuned model params

  Use --no-tune-model to skip the 2-pass pipeline (single pass only).

Deep-dive mode (--deep-dive): after sigma sweeps, does per-component
sigma tuning for multi-component parameters (cholesky[6], ucell[6],
Baniso[6]). RotXYZ uses scalar sigma by default (use --deep-dive-params
RotXYZ to override).

Each trial runs hopper on N sample frames (default 3, evenly spaced).
Scoring: median sigZ from spot diagnostics (lower = better fit).

Output: optimized phil file with best sigmas + a report.

Usage:
  # Basic: sweep sigmas for an existing phil
  python hopper_heatup.py base.phil --n-frames 3

  # With custom spec file (overrides spec in phil)
  python hopper_heatup.py base.phil --spec exp_ref_spec.txt --n-frames 5

  # Skip stages you don't need
  python hopper_heatup.py base.phil --stages G,RotXYZ,Nabc

  # Custom sigma sweep values
  python hopper_heatup.py base.phil --sigmas-G "0.01,0.1,1,5"

  # Use GPU
  python hopper_heatup.py base.phil --cuda

  # Deep-dive: per-component sigma tuning after standard stages
  python hopper_heatup.py base.phil --deep-dive

  # Deep-dive only specific params (RotXYZ excluded by default)
  python hopper_heatup.py base.phil --deep-dive --deep-dive-params cholesky,RotXYZ

  # Custom deep-dive multipliers (relative to scalar best)
  python hopper_heatup.py base.phil --deep-dive --deep-dive-multipliers "0.2,0.5,1,2,5"

  # Multi-GPU parallel (N GPUs, M trials per GPU)
  python hopper_heatup.py base.phil --n-gpus 4 --procs-per-gpu 2 --deep-dive

  # MPI parallel (all workers execute trials, rank 0 coordinates)
  mpirun -n 8 python hopper_heatup.py base.phil --mpi --cuda --deep-dive
"""

from __future__ import division, print_function

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

# MPI support (optional)
try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False

# Defaults for sigma sweeps (kept in [0.01, 100] range)
DEFAULT_SIGMAS = {
    'G':        [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    'RotXYZ':   [0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    'Nabc':     [0.01, 0.1, 0.5, 1.0, 5.0],
    'cholesky': [0.01, 0.1, 0.5, 1.0, 5.0],
    'B':        [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    'ucell':    [0.01, 0.1, 0.5, 1.0, 5.0],
}

# Consolidated sigma groups (for --use-consolidated-sigmas mode)
# Maps group name -> list of individual parameter names it controls
CONSOLIDATED_SIGMA_GROUPS = {
    'ucell_lengths': ['ucell_a', 'ucell_b', 'ucell_c'],
    'ucell_angles':  ['ucell_al', 'ucell_be', 'ucell_ga'],
    'Nabc_group':    ['Nabc'],  # Already a group, but included for consistency
    'RotXYZ_group':  ['RotXYZ'],  # Already a group, but included for consistency
}

# Default sigmas for consolidated groups
DEFAULT_CONSOLIDATED_SIGMAS = {
    'ucell_lengths': [0.01, 0.1, 0.5, 1.0, 5.0],
    'ucell_angles':  [0.001, 0.01, 0.05, 0.1, 0.5],  # Smaller range for angles (degrees)
    'Nabc_group':    [0.01, 0.1, 0.5, 1.0, 5.0],
    'RotXYZ_group':  [0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
}

# Which params to fix at each stage (True = fixed)
STAGE_FIXES = {
    'G':      {'G': False, 'RotXYZ': True,  'Nabc': True,  'B': True,  'ucell': True},
    'G+B':    {'G': False, 'RotXYZ': True,  'Nabc': True,  'B': False, 'ucell': True},
    'RotXYZ': {'G': False, 'RotXYZ': False, 'Nabc': True,  'B': True,  'ucell': True},
    'Nabc':   {'G': False, 'RotXYZ': False, 'Nabc': False, 'B': True,  'ucell': True},
    'B':      {'G': False, 'RotXYZ': False, 'Nabc': False, 'B': False, 'ucell': True},
    'ucell':  {'G': False, 'RotXYZ': False, 'Nabc': False, 'B': False, 'ucell': False},
}

ALL_STAGES = ['G', 'RotXYZ', 'Nabc', 'B']  # ucell optional

# Deep-dive: per-component parameters
# Maps param name -> (phil_key, n_components, component_names)
DEEP_DIVE_PARAMS = {
    'cholesky': ('cholesky', 6, ['L11', 'L21', 'L22', 'L31', 'L32', 'L33']),
    'Nabc':     ('Nabc', 3, ['Na', 'Nb', 'Nc']),
    'RotXYZ':   ('RotXYZ', 3, ['RotX', 'RotY', 'RotZ']),
    'ucell':    ('ucell', 6, ['a', 'b', 'c', 'al', 'be', 'ga']),
    'B':        ('B', 1, ['B']),
    'Baniso':   ('Baniso', 6, ['b11', 'b22', 'b33', 'b12', 'b13', 'b23']),
}

# Default deep-dive sweep multipliers (relative to the scalar best)
# We test fractions and multiples of the scalar best sigma
DEEP_DIVE_MULTIPLIERS = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]

# --- Phase 2: Restraint & model tuning ---
# Default sweep values for betas (restraint tightness)
# betas are variances for the restraint penalty: smaller = tighter restraint
# None means no restraint (the default). We sweep from tight to loose.
DEFAULT_BETAS = {
    'G':        [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
    'B':        [1e-2, 1e-1, 1.0, 10.0, 100.0],
    'Nabc':     [1e-2, 1e-1, 1.0, 10.0, 100.0],
    'cholesky': [1e-2, 1e-1, 1.0, 10.0, 100.0],
    'RotXYZ':   [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
}
# Add 'ucell' to DEFAULT_BETAS (same range as cholesky by default)
DEFAULT_BETAS['ucell'] = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]

# Default sweep values for model parameters
DEFAULT_PHISTEPS = [1, 2, 5, 10, 20, 30, 50, 75, 100]
DEFAULT_SIGMA_R = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

# Tunable model/restraint params and their phil paths
TUNE_PARAMS = {
    'phisteps':  'simulator.gonio.phi_steps',
    'sigma_r':   'refiner.sigma_r',
    'betas.G':   'betas.G',
    'betas.B':   'betas.B',
    'betas.Nabc': 'betas.Nabc',
    'betas.cholesky': 'betas.cholesky',
    'betas.RotXYZ': 'betas.RotXYZ',
    'betas.ucell': 'betas.ucell',
}


def generate_beta_sweep(tune_name, baseline_vars=None, refined_params=None,
                         n_points=7, decades_below=3, decades_above=4):
    """Generate a variance-scaled beta sweep centered on the parameter's actual variance.

    The beta at which spread_ratio ~ 0.5 is typically near the parameter's natural
    variance. We sweep from several decades below (tight) to above (loose).

    Args:
        tune_name: e.g. 'betas.G'
        baseline_vars: dict from unrestrained baseline (col -> variance), preferred
        refined_params: dict from final refinement (has var_G, etc.), fallback
        n_points: number of sweep points
        decades_below: decades below center to sweep (tight end)
        decades_above: decades above center to sweep (loose end)

    Returns: list of beta values (log-spaced), or DEFAULT_BETAS fallback.
    """
    binfo = BETA_PARAM_MAP.get(tune_name)
    if binfo is None:
        short = tune_name.split('.', 1)[1] if '.' in tune_name else tune_name
        return list(DEFAULT_BETAS.get(short, [1e-2, 1e-1, 1.0, 10.0, 100.0]))

    # Get center variance (baseline preferred, refined fallback)
    center_var = None
    if baseline_vars is not None:
        vals = [baseline_vars.get(c) for c in binfo['csv_cols']
                if c in baseline_vars and baseline_vars[c] > 0]
        if vals:
            center_var = float(np.mean(vals))

    if center_var is None and refined_params is not None:
        var_val = refined_params.get(binfo['var_key'])
        if var_val is not None:
            if isinstance(var_val, (tuple, list)):
                pos = [v for v in var_val if v > 0]
                center_var = float(np.mean(pos)) if pos else None
            elif var_val > 0:
                center_var = float(var_val)

    if center_var is None or center_var <= 0:
        # No variance info — fall back to DEFAULT_BETAS
        short = tune_name.split('.', 1)[1] if '.' in tune_name else tune_name
        return list(DEFAULT_BETAS.get(short, [1e-2, 1e-1, 1.0, 10.0, 100.0]))

    # Generate log-spaced sweep centered on the variance
    log_center = np.log10(center_var)
    lo = log_center - decades_below
    hi = log_center + decades_above
    sweep = list(np.logspace(lo, hi, n_points))

    return sweep


def expand_consolidated_sigmas(sigma_sweeps, use_consolidated=False):
    """Expand consolidated sigma groups to individual parameters.

    Args:
        sigma_sweeps: dict mapping param name -> list of sigma values
        use_consolidated: if True, interpret keys as consolidated groups

    Returns:
        dict mapping individual param names -> sigma values

    Example:
        Input:  {'ucell_lengths': [0.1, 1.0]}
        Output: {'ucell_a': [0.1, 1.0], 'ucell_b': [0.1, 1.0], 'ucell_c': [0.1, 1.0]}
    """
    if not use_consolidated:
        # Pass through unchanged for backward compatibility
        return sigma_sweeps

    expanded = {}
    for group_name, sigma_vals in sigma_sweeps.items():
        if group_name in CONSOLIDATED_SIGMA_GROUPS:
            # Expand group to individual parameters
            param_names = CONSOLIDATED_SIGMA_GROUPS[group_name]
            for param_name in param_names:
                expanded[param_name] = sigma_vals
        else:
            # Not a consolidated group, pass through
            expanded[group_name] = sigma_vals

    return expanded


def densify_sweep(values, n_workers, clamp_min=0.01, clamp_max=100.0):
    """Add log-spaced points between existing sweep values to fill n_workers.

    If len(values) >= n_workers, returns values unchanged.
    Otherwise, interpolates in log-space to reach n_workers points,
    clamped to [clamp_min, clamp_max].
    """
    if len(values) >= n_workers or n_workers <= 1:
        return values
    import numpy as np
    log_vals = np.log10(values)
    log_dense = np.linspace(log_vals[0], log_vals[-1], n_workers)
    dense = sorted(set([max(clamp_min, min(clamp_max, round(10**v, 6))) for v in log_dense]))
    # Ensure original values are included
    for v in values:
        if v not in dense:
            dense.append(v)
    return sorted(set(dense))[:n_workers]


def parse_phil_file(phil_path):
    """Read a phil file and return the raw text."""
    with open(phil_path) as f:
        return f.read()


def sample_frames(spec_path, n_frames, n_nearby=1):
    """Sample n_frames phi positions evenly from a spec file.

    For each sampled position, also include n_nearby frames on each side
    (total per position = 2*n_nearby + 1). This gives better rocking curve
    coverage since reflections span multiple frames.

    Args:
        spec_path: path to exp_ref_spec.txt
        n_frames: number of phi positions to sample
        n_nearby: number of neighboring frames on each side (default 1 = triplets)

    Returns:
        list of spec lines (n_frames * (2*n_nearby+1) total, deduplicated)
    """
    with open(spec_path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

    n_total = len(lines)
    if n_total <= n_frames:
        return lines

    # Evenly space across the full scan
    step = n_total / float(n_frames)
    center_indices = [int(round(i * step)) for i in range(n_frames)]
    # Clamp to valid range
    center_indices = [min(i, n_total - 1) for i in center_indices]

    if n_nearby == 0:
        return [lines[i] for i in center_indices]

    # Expand each center to include neighbors
    all_indices = []
    for ci in center_indices:
        for offset in range(-n_nearby, n_nearby + 1):
            idx = ci + offset
            if 0 <= idx < n_total:
                all_indices.append(idx)

    # Deduplicate while preserving order
    seen = set()
    unique_indices = []
    for idx in all_indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)

    return [lines[i] for i in unique_indices]


def write_sample_spec(sample_lines, tmpdir):
    """Write sampled spec lines to a temp file."""
    spec_path = os.path.join(tmpdir, "sample_spec.txt")
    with open(spec_path, 'w') as f:
        for line in sample_lines:
            f.write(line + '\n')
    return spec_path


def generate_trial_phil(base_phil, stage, sigma_value, best_sigmas,
                        use_cholesky=False, max_calls=None,
                        num_devices=None):
    """Generate a phil string for one trial (in-process refinement).

    Starts from the base phil, then overrides:
    - fix flags for the current stage
    - sigma for the parameter being swept
    - best sigmas from previous stages
    - num_devices (when using --n-gpus, force 1 device per trial)

    Does NOT set exp_ref_spec_file, outdir, or save_spot_diagnostics
    since refinement runs in-process via hopper_utils.refine().
    """
    lines = [base_phil, '\n# --- hopper_heatup trial ---\n']

    # Disable disk output (scoring is done in-process)
    lines.append('save_spot_diagnostics = False')

    # Suppress verbose per-image logging (Cholesky bounds, etc.)
    lines.append('logging.rank0_level = low')

    # Override num_devices for multi-GPU parallel trials
    if num_devices is not None:
        lines.append('refiner { num_devices = %d }' % num_devices)

    # Quick refinement settings
    if max_calls is not None:
        lines.append('lbfgs_maxiter = %d' % max_calls)
        lines.append('method = "L-BFGS-B"')

    # Fix flags for this stage
    fixes = STAGE_FIXES[stage]
    lines.append('fix {')
    lines.append('  G = %s' % str(fixes['G']))
    if fixes['RotXYZ']:
        lines.append('  RotXYZ = 1,1,1')
    else:
        lines.append('  RotXYZ = 0,0,0')
    lines.append('  Nabc = %s' % str(fixes['Nabc']))
    lines.append('  B = %s' % str(fixes['B']))
    lines.append('  ucell = %s' % str(fixes['ucell']))
    lines.append('}')

    # Sigma overrides: best from previous stages + current sweep value
    lines.append('sigmas {')
    for param, val in best_sigmas.items():
        if param == 'RotXYZ':
            lines.append('  RotXYZ = %.6g,%.6g,%.6g' % (val, val, val))
        elif param == 'Nabc':
            lines.append('  Nabc = %.6g,%.6g,%.6g' % (val, val, val))
        elif param == 'cholesky':
            lines.append('  cholesky = %.6g,%.6g,%.6g,%.6g,%.6g,%.6g' %
                          (val, val, val, val, val, val))
        elif param == 'ucell':
            lines.append('  ucell = %.6g,%.6g,%.6g,%.6g,%.6g,%.6g' %
                          (val, val, val, val, val, val))
        else:
            lines.append('  %s = %.6g' % (param, val))

    # Current sweep value
    param_name = stage
    if stage == 'Nabc' and use_cholesky:
        param_name = 'cholesky'

    if param_name == 'RotXYZ':
        lines.append('  RotXYZ = %.6g,%.6g,%.6g' % (sigma_value, sigma_value, sigma_value))
    elif param_name == 'Nabc':
        lines.append('  Nabc = %.6g,%.6g,%.6g' % (sigma_value, sigma_value, sigma_value))
    elif param_name == 'cholesky':
        lines.append('  cholesky = %.6g,%.6g,%.6g,%.6g,%.6g,%.6g' %
                      (sigma_value, sigma_value, sigma_value, sigma_value,
                       sigma_value, sigma_value))
    elif param_name == 'ucell':
        lines.append('  ucell = %.6g,%.6g,%.6g,%.6g,%.6g,%.6g' %
                      (sigma_value, sigma_value, sigma_value, sigma_value,
                       sigma_value, sigma_value))
    elif param_name == 'G+B':
        # Combined G+B sweep: sigma_value is sigma_G; sigma_B added by caller
        lines.append('  G = %.6g' % sigma_value)
    else:
        lines.append('  %s = %.6g' % (param_name, sigma_value))
    lines.append('}')

    return '\n'.join(lines) + '\n'


def run_hopper_trial(phil_str, tmpdir, trial_name, use_cuda=False, timeout=300,
                     gpu_id=None):
    """DEPRECATED: Use _run_one_image_trial() instead, which runs refinement
    in-process via hopper_utils.refine() without subprocess spawning.

    Run hopper on one trial and return the score.

    Returns dict with: sigZ_median, sigZ_mean, n_rois, n_outliers, converged, time_sec
    """
    phil_path = os.path.join(tmpdir, '%s.phil' % trial_name)
    with open(phil_path, 'w') as f:
        f.write(phil_str)

    # Clean stale spot_diagnostics from previous runs to avoid
    # appending to old CSVs with different column counts or frame counts
    trial_outdir = os.path.join(tmpdir, trial_name)
    stale_diag = os.path.join(trial_outdir, 'spot_diagnostics')
    if os.path.isdir(stale_diag):
        shutil.rmtree(stale_diag)

    env = os.environ.copy()
    # Strip MPI environment so hopper.py subprocess doesn't try to join
    # the parent MPI communicator (would deadlock)
    for key in list(env.keys()):
        if key.startswith(('OMPI_', 'PMI_', 'PMIX_', 'MPI_', 'SLURM_MPI',
                           'I_MPI_', 'MPICH_', 'HYDRA_')):
            del env[key]
    if use_cuda:
        env['DIFFBRAGG_USE_CUDA'] = '1'
    if gpu_id is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    cmd = ['hopper.py', phil_path]
    t0 = time.time()

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env)
        elapsed = time.time() - t0
        converged = result.returncode == 0
    except subprocess.TimeoutExpired:
        elapsed = timeout
        converged = False
        return {
            'sigZ_median': float('inf'),
            'sigZ_mean': float('inf'),
            'n_rois': 0,
            'n_outliers': 0,
            'param_vars': {},
            'converged': False,
            'time_sec': elapsed,
            'error': 'timeout',
        }

    if not converged:
        # Try to parse whatever output we got
        stderr_tail = result.stderr[-500:] if result.stderr else ''
        return {
            'sigZ_median': float('inf'),
            'sigZ_mean': float('inf'),
            'n_rois': 0,
            'n_outliers': 0,
            'param_vars': {},
            'converged': False,
            'time_sec': elapsed,
            'error': stderr_tail,
        }

    # Parse results from spot diagnostics
    return parse_trial_results(tmpdir, trial_name, elapsed)


def parse_trial_results(tmpdir, trial_name, elapsed):
    """DEPRECATED: Scoring is now done in-process via extract_image_score().
    Parse spot diagnostics from a hopper run."""
    import numpy as np

    # Find the outdir used by this trial
    phil_path = os.path.join(tmpdir, '%s.phil' % trial_name)
    outdir = None
    with open(phil_path) as f:
        for line in f:
            if 'outdir' in line and '=' in line and '#' not in line.split('=')[0]:
                outdir = line.split('=', 1)[1].strip().strip('"').strip("'")

    if outdir is None:
        return {
            'sigZ_median': float('inf'),
            'sigZ_mean': float('inf'),
            'n_rois': 0,
            'n_outliers': 0,
            'param_vars': {},
            'converged': True,
            'time_sec': elapsed,
            'error': 'no outdir found in phil',
        }

    # Look for shot summary CSVs
    patterns = [
        os.path.join(outdir, 'spot_diagnostics', 'rank*', '*_shot_summary.csv'),
        os.path.join(outdir, 'spot_diagnostics', '*_shot_summary.csv'),
    ]

    csv_files = []
    for pat in patterns:
        csv_files.extend(glob.glob(pat))

    if not csv_files:
        # Fall back to per-spot CSVs
        spot_patterns = [
            os.path.join(outdir, 'spot_diagnostics', 'rank*', '*_spots.csv'),
            os.path.join(outdir, 'spot_diagnostics', '*_spots.csv'),
        ]
        spot_files = []
        for pat in spot_patterns:
            spot_files.extend(glob.glob(pat))

        if not spot_files:
            return {
                'sigZ_median': float('inf'),
                'sigZ_mean': float('inf'),
                'n_rois': 0,
                'n_outliers': 0,
                'param_vars': {},
                'converged': True,
                'time_sec': elapsed,
                'error': 'no diagnostic CSVs found in %s' % outdir,
            }

        # Parse per-spot CSVs
        sigz_vals = []
        for f in spot_files:
            try:
                import csv
                with open(f) as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        if 'sigma_z' in row:
                            try:
                                sigz_vals.append(float(row['sigma_z']))
                            except (ValueError, TypeError):
                                pass
            except Exception:
                pass

        if not sigz_vals:
            return {
                'sigZ_median': float('inf'),
                'sigZ_mean': float('inf'),
                'n_rois': 0,
                'n_outliers': 0,
                'param_vars': {},
                'converged': True,
                'time_sec': elapsed,
                'error': 'no valid sigZ values in spot CSVs',
            }

        from simtbx.diffBragg.utils import is_outlier
        sigz_arr = np.array(sigz_vals)
        n_outliers = int(is_outlier(sigz_arr).sum()) if len(sigz_arr) >= 3 else 0
        return {
            'sigZ_median': float(np.median(sigz_vals)),
            'sigZ_mean': float(np.mean(sigz_vals)),
            'n_rois': len(sigz_vals),
            'n_outliers': n_outliers,
            'param_vars': {},  # per-spot CSVs don't have parameter columns
            'converged': True,
            'time_sec': elapsed,
            'error': None,
        }

    # Parse shot summary CSVs
    try:
        import csv
        all_sigz_med = []
        all_sigz_mean = []
        all_nrois = []
        all_n_outliers = []
        all_spearman = []

        # Per-frame parameter values for spread measurement
        _param_cols = ['G', 'B', 'rotX', 'rotY', 'rotZ', 'Na', 'Nb', 'Nc',
                       'chol_L11', 'chol_L21', 'chol_L22',
                       'chol_L31', 'chol_L32', 'chol_L33',
                       'uc_a', 'uc_b', 'uc_c', 'uc_al', 'uc_be', 'uc_ga']
        all_param_vals = {c: [] for c in _param_cols}

        for f in csv_files:
            with open(f) as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    if 'sigz_median' in row:
                        try:
                            all_sigz_med.append(float(row['sigz_median']))
                        except (ValueError, TypeError):
                            pass
                    if 'sigz_mean' in row:
                        try:
                            all_sigz_mean.append(float(row['sigz_mean']))
                        except (ValueError, TypeError):
                            pass
                    if 'n_rois' in row:
                        try:
                            all_nrois.append(int(row['n_rois']))
                        except (ValueError, TypeError):
                            pass
                    if 'n_outlier_sigz' in row:
                        try:
                            all_n_outliers.append(int(row['n_outlier_sigz']))
                        except (ValueError, TypeError):
                            pass
                    if 'spearman_r_median' in row:
                        try:
                            v = float(row['spearman_r_median'])
                            if np.isfinite(v):
                                all_spearman.append(v)
                        except (ValueError, TypeError):
                            pass
                    # Collect per-frame parameter values
                    for col in _param_cols:
                        if col in row:
                            try:
                                v = float(row[col])
                                if np.isfinite(v):
                                    all_param_vals[col].append(v)
                            except (ValueError, TypeError):
                                pass

        # Compute per-parameter variance across frames (robust MAD-based)
        # MAD estimator: var ≈ (1.4826 * median(|x - median(x)|))^2
        # Much more stable than np.var with few frames (resistant to outliers)
        param_vars = {}
        for col, vals in all_param_vals.items():
            if len(vals) >= 2:
                arr = np.array(vals)
                med = np.median(arr)
                mad = np.median(np.abs(arr - med))
                param_vars[col] = float((1.4826 * mad) ** 2)

        if not all_sigz_med:
            return {
                'sigZ_median': float('inf'),
                'sigZ_mean': float('inf'),
                'n_rois': sum(all_nrois) if all_nrois else 0,
                'n_outliers': sum(all_n_outliers) if all_n_outliers else 0,
                'param_vars': param_vars,
                'converged': True,
                'time_sec': elapsed,
                'error': 'no sigz_median in shot summaries',
            }

        return {
            'sigZ_median': float(np.median(all_sigz_med)),
            'sigZ_mean': float(np.mean(all_sigz_mean)) if all_sigz_mean else float('inf'),
            'spearman_r': float(np.median(all_spearman)) if all_spearman else float('nan'),
            'n_rois': sum(all_nrois) if all_nrois else 0,
            'n_outliers': sum(all_n_outliers) if all_n_outliers else 0,
            'param_vars': param_vars,
            'converged': True,
            'time_sec': elapsed,
            'error': None,
        }
    except Exception as e:
        return {
            'sigZ_median': float('inf'),
            'sigZ_mean': float('inf'),
            'n_rois': 0,
            'n_outliers': 0,
            'param_vars': {},
            'converged': True,
            'time_sec': elapsed,
            'error': str(e),
        }


def estimate_G_parallel(base_phil, sample_lines, use_cuda=False,
                         n_parallel=1, n_gpus=None,
                         procs_per_gpu=1, mpi_comm=None, num_devices=None):
    """Estimate init.G by running parallel trials across orders of magnitude.

    Runs G-only refinement at G = 10^0, 10^1, ..., 10^15 in parallel.
    Each trial fixes everything except G and runs a few L-BFGS-B iterations.
    The best G is the one with lowest median sigZ.

    Disables auto_G so our explicit init.G / bounds are respected.
    sample_lines: list of spec file lines (one per image)
    """
    import numpy as np

    # Span 16 orders of magnitude: G = 1 to 1e15
    g_values = [10.0 ** i for i in range(16)]

    n_images = len(sample_lines)
    print("  Searching %d G values x %d images = %d work units..." %
          (len(g_values), n_images, len(g_values) * n_images))

    # Build per-sigma trial configs
    trial_args = []
    for g_val in g_values:
        # Set bounds to +/- 2 orders of magnitude around init.G
        # so the ranged parameterization has a sensible range
        g_min = max(1e-10, g_val * 0.01)
        g_max = g_val * 100.0

        lines = [base_phil, '\n# G search trial\n']
        lines.append('save_spot_diagnostics = False')
        lines.append('logging.rank0_level = low')
        lines.append('init.G = %.6e' % g_val)
        lines.append('init.auto_G = False')
        lines.append('mins.G = %.6e' % g_min)
        lines.append('maxs.G = %.6e' % g_max)
        lines.append('lbfgs_maxiter = 10')
        lines.append('method = "L-BFGS-B"')
        if num_devices is not None:
            lines.append('refiner { num_devices = %d }' % num_devices)
        lines.append('fix {')
        lines.append('  G = False')
        lines.append('  RotXYZ = 1,1,1')
        lines.append('  Nabc = True')
        lines.append('  B = True')
        lines.append('  ucell = True')
        lines.append('}')
        phil_str = '\n'.join(lines) + '\n'

        label = 'G=%.0e' % g_val
        trial_args.append((phil_str, use_cuda, g_val, label))

    # Flatten to per-(G_val, image) work units
    flat_trials = _flatten_trials(trial_args, sample_lines)
    flat_trials = _assign_gpus(flat_trials, n_gpus, procs_per_gpu)

    # Run all in parallel
    raw_results = _run_trials_batch(flat_trials, n_parallel=n_parallel,
                                     mpi_comm=mpi_comm)

    # Aggregate per-image results back to per-G_val
    agg_results = aggregate_image_results(raw_results, n_images)

    # Find the best G (lowest sigZ).
    # Use <= so that when scores are tied across a flat plateau, we pick
    # the HIGHEST G value. This gives better bounds centering since the
    # true G is near the top of the plateau (too-high init.G causes sigZ
    # to rise, too-low converges to the same value).
    sorted_results = sorted(agg_results, key=lambda x: x[0])
    best_G = None
    best_score = float('inf')
    for g_val, label, score in sorted_results:
        if score['converged'] and score['error'] is None:
            trial_score = _score_trial(score)
            if trial_score <= best_score:
                best_score = trial_score
                best_G = g_val

    print("\n  %-12s  %10s  %10s  %6s  %5s  %s" %
          ('init.G', 'sigZ_med', 'sigZ_mean', 'n_roi', 'n_out', 'status'))
    print("  " + "-" * 63)
    for g_val, label, score in sorted_results:
        if score['converged'] and score['error'] is None:
            marker = ' <-- best' if g_val == best_G else ''
            sigz_str = '%.3f' % score['sigZ_median']
            mean_str = '%.3f' % score['sigZ_mean']
            print("  %-12.2e  %10s  %10s  %6d  %5d  OK%s" %
                  (g_val, sigz_str, mean_str, score['n_rois'],
                   score.get('n_outliers', 0), marker))
        else:
            print("  %-12.2e  %10s  %10s  %6d  %5d  FAIL" %
                  (g_val, 'inf', 'inf', 0, 0))

    if best_G is not None:
        print("\n  Best init.G = %.4e (score = %.3f)" % (best_G, best_score))
    else:
        print("\n  WARNING: No valid G found across all orders of magnitude!")

    return best_G


def _run_one_trial(args_tuple):
    """DEPRECATED: Use _run_one_image_trial() instead.
    Worker function for parallel trial execution via subprocess.

    args_tuple: (phil_str, tmpdir, trial_name, use_cuda, timeout, sigma_val, label, gpu_id)
    Returns: (sigma_val, label, score_dict)
    """
    if len(args_tuple) == 8:
        phil_str, tmpdir, trial_name, use_cuda, timeout, sigma_val, label, gpu_id = args_tuple
    else:
        phil_str, tmpdir, trial_name, use_cuda, timeout, sigma_val, label = args_tuple
        gpu_id = None
    score = run_hopper_trial(phil_str, tmpdir, trial_name,
                             use_cuda=use_cuda, timeout=timeout, gpu_id=gpu_id)
    return (sigma_val, label, score)


def _parse_phil_str(phil_str):
    """Parse a phil string into a params object.

    Uses the hopper phil scope (hopper_phil + philz).
    """
    from libtbx.phil import parse
    from simtbx.diffBragg.phil import philz as base_philz, hopper_phil
    scope = parse(hopper_phil + base_philz)
    user = parse(phil_str)
    working = scope.fetch(source=user)
    return working.extract()


def _run_one_image_trial(args_tuple):
    """Worker function for in-process per-image trial execution.

    args_tuple: (phil_str, spec_line, use_cuda, sigma_val, label, gpu_id)
    Returns: (sigma_val, label, per_image_score_dict)
    """
    phil_str, spec_line, use_cuda, sigma_val, label, gpu_id = args_tuple
    t0 = time.time()
    try:
        if use_cuda:
            os.environ['DIFFBRAGG_USE_CUDA'] = '1'

        from simtbx.diffBragg import hopper_utils

        params = _parse_phil_str(phil_str)
        exp, ref, exp_idx, spec = hopper_utils.split_line(spec_line)

        # Override params for in-process mode
        params.outdir = None  # no disk output

        gpu_device = gpu_id if gpu_id is not None else 0
        _, _, Modeler, SIM, x = hopper_utils.refine(
            exp, ref, params, spec=spec, gpu_device=gpu_device,
            return_modeler=True, free_mem=False)

        score = hopper_utils.extract_image_score(Modeler, SIM, x, params)
        score['time_sec'] = time.time() - t0

        Modeler.clean_up(SIM)
        return (sigma_val, label, score)

    except Exception as e:
        import traceback
        return (sigma_val, label, {
            'sigz_median': float('inf'),
            'sigz_mean': float('inf'),
            'sigz_vals': [],
            'n_rois': 0,
            'n_outlier_sigz': 0,
            'spearman_r_median': float('nan'),
            'spearman_vals': [],
            'checker_median': float('nan'),
            'checker_vals': [],
            'param_vals': {},
            'converged': False,
            'time_sec': time.time() - t0,
            'error': traceback.format_exc()[-200:],
        })


def _flatten_trials(trial_args, spec_lines):
    """Expand per-sigma trial list to per-(sigma, image) work units.

    Input trial_args: list of (phil_str, use_cuda, sigma_val, label)
        — note NO spec_line or gpu_id yet
    Input spec_lines: list of spec file lines (one per image)

    Output: list of (phil_str, spec_line, use_cuda, sigma_val, label, gpu_id=None)
        — one per (sigma x image), with gpu_id to be assigned by _assign_gpus

    Work units are ordered expensive-first (reversed trial_args order) so that
    costly trials (e.g. high phisteps) get dispatched to workers early,
    avoiding a long tail where one worker is still grinding on an expensive
    trial while others sit idle.
    """
    flat = []
    for phil_str, use_cuda, sigma_val, label in reversed(trial_args):
        for spec_line in spec_lines:
            flat.append((phil_str, spec_line, use_cuda, sigma_val, label, None))
    return flat


def aggregate_image_results(image_results, n_images):
    """Aggregate per-image results into per-sigma score dicts.

    Groups results by (sigma_val, label), computes aggregate stats
    matching the output format of parse_trial_results().

    image_results: list of (sigma_val, label, per_image_score_dict)
    n_images: expected number of images per sigma value

    Returns: list of (sigma_val, label, aggregated_score_dict)
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for sigma_val, label, score in image_results:
        groups[(sigma_val, label)].append(score)

    aggregated = []
    _param_cols = ['scale', 'Bfactor', 'rotX', 'rotY', 'rotZ', 'Na', 'Nb', 'Nc',
                   'a', 'b', 'c', 'al', 'be', 'ga']
    # Also include cholesky if present
    _chol_cols = ['cholesky']

    for (sigma_val, label), scores in groups.items():
        # Filter out errored images
        good = [s for s in scores if s.get('error') is None and s.get('converged', False)]

        if not good:
            aggregated.append((sigma_val, label, {
                'sigZ_median': float('inf'),
                'sigZ_mean': float('inf'),
                'spearman_r': float('nan'),
                'checker_median': float('nan'),
                'checker_mean': float('nan'),
                'n_rois': 0,
                'n_outliers': 0,
                'param_vars': {},
                'converged': False,
                'time_sec': max((s['time_sec'] for s in scores), default=0),
                'error': 'all images failed',
            }))
            continue

        # Aggregate sigZ across images by pooling raw per-ROI values
        # This matches the baseline behavior (median of ALL ROIs across all images)
        # rather than median-of-per-image-medians which is too coarse with few images
        pooled_sigz = []
        pooled_spearman = []
        pooled_checker = []
        for s in good:
            pooled_sigz.extend(s.get('sigz_vals', []))
            pooled_spearman.extend(s.get('spearman_vals', []))
            pooled_checker.extend(s.get('checker_vals', []))
        # Filter non-finite
        pooled_sigz = [v for v in pooled_sigz if np.isfinite(v)]
        pooled_spearman = [v for v in pooled_spearman if np.isfinite(v)]
        pooled_checker = [v for v in pooled_checker if np.isfinite(v)]

        # n_rois and n_outliers are summed across images
        total_rois = sum(s['n_rois'] for s in good)
        total_outliers = sum(s.get('n_outlier_sigz', 0) for s in good)

        # Compute per-parameter variance across images (robust MAD-based)
        param_vars = {}
        # Map from get_param_from_x keys to CSV-compatible names for downstream
        _key_map = {'scale': 'G', 'Bfactor': 'B',
                     'rotX': 'rotX', 'rotY': 'rotY', 'rotZ': 'rotZ',
                     'Na': 'Na', 'Nb': 'Nb', 'Nc': 'Nc',
                     'a': 'uc_a', 'b': 'uc_b', 'c': 'uc_c',
                     'al': 'uc_al', 'be': 'uc_be', 'ga': 'uc_ga'}
        for src_key, dst_key in _key_map.items():
            vals = []
            for s in good:
                pv = s.get('param_vals', {})
                if src_key in pv and np.isfinite(pv[src_key]):
                    vals.append(pv[src_key])
            if len(vals) >= 2:
                arr = np.array(vals)
                med = np.median(arr)
                mad = np.median(np.abs(arr - med))
                param_vars[dst_key] = float((1.4826 * mad) ** 2)

        # Cholesky components
        for s in good:
            pv = s.get('param_vals', {})
            if 'cholesky' in pv:
                chol = pv['cholesky']
                chol_names = ['chol_L11', 'chol_L21', 'chol_L22',
                              'chol_L31', 'chol_L32', 'chol_L33']
                for i, cn in enumerate(chol_names):
                    if cn not in param_vars:
                        param_vars[cn] = 0.0
                break  # Just check if any image has cholesky
        # Now compute cholesky variances
        for i, cn in enumerate(['chol_L11', 'chol_L21', 'chol_L22',
                                 'chol_L31', 'chol_L32', 'chol_L33']):
            vals = []
            for s in good:
                pv = s.get('param_vals', {})
                if 'cholesky' in pv and len(pv['cholesky']) > i:
                    v = pv['cholesky'][i]
                    if np.isfinite(v):
                        vals.append(v)
            if len(vals) >= 2:
                arr = np.array(vals)
                med = np.median(arr)
                mad = np.median(np.abs(arr - med))
                param_vars[cn] = float((1.4826 * mad) ** 2)

        aggregated.append((sigma_val, label, {
            'sigZ_median': float(np.median(pooled_sigz)) if pooled_sigz else float('inf'),
            'sigZ_mean': float(np.mean(pooled_sigz)) if pooled_sigz else float('inf'),
            'spearman_r': float(np.median(pooled_spearman)) if pooled_spearman else float('nan'),
            'checker_median': float(np.median(pooled_checker)) if pooled_checker else float('nan'),
            'checker_mean': float(np.mean(pooled_checker)) if pooled_checker else float('nan'),
            'n_rois': total_rois,
            'n_outliers': total_outliers,
            'param_vars': param_vars,
            'converged': True,
            'time_sec': max(s['time_sec'] for s in scores),
            'error': None,
        }))

    return aggregated


def _assign_gpus(trial_args, n_gpus, procs_per_gpu):
    """Assign GPU IDs to trial args using round-robin across GPUs.

    Each GPU can run procs_per_gpu concurrent trials. Total parallelism
    is n_gpus * procs_per_gpu.

    Replaces the gpu_id field (last element) in each trial tuple.
    Tuple format: (phil_str, spec_line, use_cuda, sigma_val, label, gpu_id)
    """
    if n_gpus is None or n_gpus <= 0:
        return trial_args  # gpu_id already None from _flatten_trials

    result = []
    for i, ta in enumerate(trial_args):
        gpu_id = i % n_gpus
        result.append(ta[:-1] + (gpu_id,))
    return result


def _run_trials_batch(trial_args, n_parallel=1, mpi_comm=None):
    """Run a batch of trials with the best available parallelism.

    trial_args: list of (phil_str, spec_line, use_cuda, sigma_val, label, gpu_id)
    Returns: list of (sigma_val, label, score_dict) in submission order.
    """
    if mpi_comm is not None and mpi_comm.Get_size() > 1:
        # MPI mode: rank 0 distributes, workers execute
        raw = run_trials_mpi(trial_args, mpi_comm)
        return raw  # already list of (sigma_val, label, score)

    if n_parallel <= 1:
        # Sequential
        results = []
        for i, ta in enumerate(trial_args):
            label = ta[4]  # label is index 4 in new tuple
            sigma_val, label, score = _run_one_image_trial(ta)
            print("  %s:" % label, end='')
            _print_trial_result(score)
            results.append((sigma_val, label, score))
        return results

    # ThreadPoolExecutor parallel
    results = [None] * len(trial_args)
    with ThreadPoolExecutor(max_workers=n_parallel) as executor:
        future_to_idx = {}
        for idx, ta in enumerate(trial_args):
            f = executor.submit(_run_one_image_trial, ta)
            future_to_idx[f] = idx
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            sigma_val, label, score = future.result()
            results[idx] = (sigma_val, label, score)
            print("  %s:" % label, end='')
            _print_trial_result(score)
    return results


def run_sigma_sweep(base_phil, stage, sigma_values, best_sigmas, sample_lines,
                    use_cholesky=False, use_cuda=False, max_calls=50,
                    n_parallel=1, n_gpus=None, procs_per_gpu=1,
                    mpi_comm=None, num_devices=None, densify=1):
    """Sweep sigma values for one parameter and return results.

    Runs in-process refinement for each (sigma, image) pair, then
    aggregates per-image results into per-sigma scores.

    sample_lines: list of spec file lines (one per image)
    densify: integer multiplier for sweep resolution (1=base, 2=2x finer, etc.)
    """
    n_images = len(sample_lines)
    # Densify sweep: interpolate between base values for finer resolution
    n_target_sigmas = max(len(sigma_values), len(sigma_values) * densify)
    if n_target_sigmas > len(sigma_values):
        sigma_values = densify_sweep(sigma_values, n_target_sigmas)
        print("  Densified %dx to %d sweep values (%d images/sigma, %d total work units)" %
              (densify, len(sigma_values), n_images, len(sigma_values) * n_images))
    # Deduplicate values that would collide in trial naming
    seen = set()
    unique_vals = []
    for v in sigma_values:
        key = '%.4e' % v
        if key not in seen:
            seen.add(key)
            unique_vals.append(v)
    sigma_values = unique_vals

    n_images = len(sample_lines)
    print("  %d sigma values x %d images = %d work units" %
          (len(sigma_values), n_images, len(sigma_values) * n_images))

    # Build per-sigma trial configs (without spec_line)
    trial_args = []
    for sigma_val in sigma_values:
        phil_str = generate_trial_phil(
            base_phil, stage, sigma_val, best_sigmas,
            use_cholesky=use_cholesky, max_calls=max_calls,
            num_devices=num_devices)

        label = 'sigmas.%s=%.2e' % (stage, sigma_val)
        trial_args.append((phil_str, use_cuda, sigma_val, label))

    # Flatten to per-(sigma, image) work units
    flat_trials = _flatten_trials(trial_args, sample_lines)

    # Assign GPUs
    flat_trials = _assign_gpus(flat_trials, n_gpus, procs_per_gpu)

    # Run all
    raw_results = _run_trials_batch(flat_trials, n_parallel=n_parallel,
                                     mpi_comm=mpi_comm)

    # Aggregate per-image results back to per-sigma
    agg_results = aggregate_image_results(raw_results, n_images)

    results = []
    for sigma_val, label, score in agg_results:
        results.append({'sigma': sigma_val, 'stage': stage, **score})
    results.sort(key=lambda r: r['sigma'])

    return results


def run_GB_sigma_sweep(base_phil, sigma_G_values, sigma_B_values, best_sigmas,
                       sample_lines, use_cholesky=False, use_cuda=False,
                       max_calls=50, n_parallel=1, n_gpus=None,
                       procs_per_gpu=1, mpi_comm=None, num_devices=None):
    """Combinatorial G x B sigma sweep. Both G and B are free simultaneously.

    Instead of sweeping G alone (useless — 1D optimizer always converges) then
    B alone, we sweep all (sigma_G, sigma_B) pairs so the relative step sizes
    between G and B actually matter. Total trials = len(G) * len(B).

    sample_lines: list of spec file lines (one per image)
    Returns: list of dicts with 'sigma_G', 'sigma_B', 'sigZ_median', etc.
    """
    import itertools

    # Build grid of all (sigma_G, sigma_B) pairs
    grid = list(itertools.product(sigma_G_values, sigma_B_values))

    n_images = len(sample_lines)
    print("  %d G x B pairs x %d images = %d work units" %
          (len(grid), n_images, len(grid) * n_images))

    trial_args = []
    for sigma_G, sigma_B in grid:
        # Build phil: G+B stage fixes (both free, RotXYZ/Nabc/ucell fixed)
        phil_str = generate_trial_phil(
            base_phil, 'G+B', sigma_G, best_sigmas,
            use_cholesky=use_cholesky, max_calls=max_calls,
            num_devices=num_devices)
        # Override sigma_B (generate_trial_phil set sigma_G via the sweep value)
        phil_str += 'sigmas {\n  B = %.6g\n}\n' % sigma_B

        label = 'G=%.2e,B=%.2e' % (sigma_G, sigma_B)
        trial_args.append((phil_str, use_cuda, sigma_G, label))

    # Flatten to per-(sigma_pair, image) work units
    flat_trials = _flatten_trials(trial_args, sample_lines)
    flat_trials = _assign_gpus(flat_trials, n_gpus, procs_per_gpu)

    raw_results = _run_trials_batch(flat_trials, n_parallel=n_parallel,
                                     mpi_comm=mpi_comm)

    # Aggregate per-image results back to per-sigma_pair
    agg_results = aggregate_image_results(raw_results, n_images)

    # Map back to grid indices by label
    label_to_grid = {('G=%.2e,B=%.2e' % (g, b)): (g, b) for g, b in grid}

    results = []
    for sigma_val, label, score in agg_results:
        sigma_G, sigma_B = label_to_grid.get(label, (sigma_val, 0))
        results.append({
            'sigma_G': sigma_G,
            'sigma_B': sigma_B,
            'stage': 'G+B',
            **score,
        })

    return results


def print_GB_summary(results):
    """Print summary table for a combinatorial G x B sweep."""
    # Check if any result has CHECKER scores
    has_checker = any(r.get('checker_median') is not None and
                      not np.isnan(r.get('checker_median', float('nan')))
                      for r in results)

    if has_checker:
        print("\n  %-10s  %-10s  %10s  %10s  %9s  %6s  %5s  %5s  %s" %
              ('sigma_G', 'sigma_B', 'sigZ_med', 'sigZ_mean', 'CHECKER', 'n_roi', 'n_out',
               'time', 'status'))
        print("  " + "-" * 88)
    else:
        print("\n  %-10s  %-10s  %10s  %10s  %6s  %5s  %5s  %s" %
              ('sigma_G', 'sigma_B', 'sigZ_med', 'sigZ_mean', 'n_roi', 'n_out',
               'time', 'status'))
        print("  " + "-" * 78)

    valid = [r for r in results if r['converged'] and r.get('error') is None
             and r['sigZ_median'] < 1e6]
    best = min(valid, key=_score_trial) if valid else None

    for r in sorted(results, key=lambda r: (r['sigma_G'], r['sigma_B'])):
        status = 'OK' if r['converged'] and r.get('error') is None else 'FAIL'
        sigz_med = '%.3f' % r['sigZ_median'] if r['sigZ_median'] < 1e6 else 'inf'
        sigz_mean = '%.3f' % r['sigZ_mean'] if r['sigZ_mean'] < 1e6 else 'inf'
        marker = ' <-- best' if r is best else ''

        if has_checker:
            checker_str = '%.4f' % r.get('checker_median', float('nan')) if not np.isnan(r.get('checker_median', float('nan'))) else 'n/a'
            print("  %-10.2e  %-10.2e  %10s  %10s  %9s  %6d  %5d  %5.1f  %s%s" %
                  (r['sigma_G'], r['sigma_B'], sigz_med, sigz_mean, checker_str,
                   r['n_rois'], r.get('n_outliers', 0), r['time_sec'],
                   status, marker))
        else:
            print("  %-10.2e  %-10.2e  %10s  %10s  %6d  %5d  %5.1f  %s%s" %
                  (r['sigma_G'], r['sigma_B'], sigz_med, sigz_mean,
                   r['n_rois'], r.get('n_outliers', 0), r['time_sec'],
                   status, marker))


def _print_trial_result(score):
    """Print a single trial's result.

    Handles both per-image keys (sigz_median, n_outlier_sigz) from
    extract_image_score() and aggregate keys (sigZ_median, n_outliers)
    from aggregate_image_results().
    """
    if score.get('converged', False) and score.get('error') is None:
        sigz_med = score.get('sigZ_median', score.get('sigz_median', float('inf')))
        sigz_mean = score.get('sigZ_mean', score.get('sigz_mean', float('inf')))
        n_rois = score.get('n_rois', 0)
        n_out = score.get('n_outliers', score.get('n_outlier_sigz', 0))
        checker_med = score.get('checker_median', float('nan'))

        if not np.isnan(checker_med):
            print(" sigZ_med=%.3f sigZ_mean=%.3f CHECKER=%.4f n_rois=%d n_out=%d (%.1fs)" %
                  (sigz_med, sigz_mean, checker_med, n_rois, n_out, score['time_sec']))
        else:
            print(" sigZ_med=%.3f sigZ_mean=%.3f n_rois=%d n_out=%d (%.1fs)" %
                  (sigz_med, sigz_mean, n_rois, n_out, score['time_sec']))
    else:
        print(" FAILED: %s (%.1fs)" %
              (score.get('error', 'unknown')[:60], score['time_sec']))


def run_trials_mpi(trial_args, comm):
    """Run a batch of trials using MPI manager-worker pattern.

    Rank 0 distributes trial_args to workers, collects results.
    Workers are persistent (they don't stop after this batch).

    trial_args: list of tuples (phil_str, spec_line, use_cuda,
                                sigma_val, label, gpu_id)
    Returns: list of (sigma_val, label, score_dict) in original order
    """
    size = comm.Get_size()
    n_workers = size - 1

    if n_workers == 0:
        # No workers, run sequentially on rank 0
        results = []
        for ta in trial_args:
            results.append(_run_one_image_trial(ta))
        return results

    results = [None] * len(trial_args)
    next_idx = 0
    busy_workers = set()

    # Send initial work to workers
    for worker in range(1, min(size, len(trial_args) + 1)):
        if next_idx < len(trial_args):
            comm.send(('work', next_idx, trial_args[next_idx]), dest=worker)
            busy_workers.add(worker)
            next_idx += 1

    # Collect results and distribute remaining work
    n_done = 0
    n_total = len(trial_args)
    while n_done < n_total:
        status = MPI.Status()
        msg = comm.recv(source=MPI.ANY_SOURCE, status=status)
        worker = status.Get_source()
        _, idx, result = msg  # ('done', idx, result_tuple)
        results[idx] = result
        n_done += 1

        sigma_val, label, score = result
        print("  [rank %d] %s:" % (worker, label), end='')
        _print_trial_result(score)

        if next_idx < len(trial_args):
            comm.send(('work', next_idx, trial_args[next_idx]), dest=worker)
            next_idx += 1
        else:
            busy_workers.discard(worker)
            comm.send(('idle',), dest=worker)

    # Idle remaining workers
    for worker in busy_workers:
        comm.send(('idle',), dest=worker)

    return results


def mpi_worker_loop(comm):
    """Persistent worker loop for MPI mode. Called on ranks > 0.

    Waits for messages from rank 0:
      ('work', idx, trial_tuple) -> execute and return result
      ('idle',) -> wait for next message
      ('shutdown',) -> exit

    trial_tuple: (phil_str, spec_line, use_cuda, sigma_val, label, gpu_id)
    """
    while True:
        msg = comm.recv(source=0)
        cmd = msg[0]
        if cmd == 'shutdown':
            break
        elif cmd == 'work':
            _, idx, trial_tuple = msg
            result = _run_one_image_trial(trial_tuple)
            comm.send(('done', idx, result), dest=0)
        elif cmd == 'idle':
            continue  # just loop back and wait


def _score_trial(r):
    """Score a trial result. Lower = better.
    Uses sigZ_median as base, penalizes outlier ROIs (MAD-based).
    Each 10% of outlier ROIs adds 50% to the score.
    """
    med = r['sigZ_median']
    if med <= 0 or med >= 1e6:
        return float('inf')
    n_rois = r.get('n_rois', 0)
    n_outliers = r.get('n_outliers', 0)
    outlier_frac = n_outliers / n_rois if n_rois > 0 else 0
    penalty = 1.0 + 5.0 * outlier_frac  # 10% outliers -> 1.5x, 20% -> 2.0x
    return med * penalty


def _score_trial_spearman(r):
    """Score for sigma_r tuning: maximize Spearman rank correlation.

    sigma_r controls the noise model (var = model + sigma_r²), which directly
    scales sigma_Z. Minimizing sigma_Z is circular — it just picks the largest
    sigma_r. Instead, Spearman R measures model-data shape agreement independent
    of sigma_r. Higher = better, so we return negative for min().
    """
    rho = r.get('spearman_r', float('nan'))
    if not np.isfinite(rho):
        return float('inf')
    n_rois = r.get('n_rois', 0)
    n_outliers = r.get('n_outliers', 0)
    outlier_frac = n_outliers / n_rois if n_rois > 0 else 0
    penalty = 1.0 + 5.0 * outlier_frac
    return -rho * (1.0 / penalty)  # negate: min(-rho) = max(rho)


def print_stage_summary(stage, results):
    """Print a summary table for one stage's sweep."""
    # Check if any result has CHECKER scores
    has_checker = any(r.get('checker_median') is not None and
                      not np.isnan(r.get('checker_median', float('nan')))
                      for r in results)

    if has_checker:
        print("\n  %-12s  %10s  %10s  %9s  %6s  %5s  %5s  %s" %
              ('sigma', 'sigZ_med', 'sigZ_mean', 'CHECKER', 'n_roi', 'n_out', 'time', 'status'))
        print("  " + "-" * 80)
    else:
        print("\n  %-12s  %10s  %10s  %6s  %5s  %5s  %s" %
              ('sigma', 'sigZ_med', 'sigZ_mean', 'n_roi', 'n_out', 'time', 'status'))
        print("  " + "-" * 68)

    best = find_best_sigma(results, return_result=True)
    for r in results:
        status = 'OK' if r['converged'] and r['error'] is None else 'FAIL'
        sigz_med = '%.3f' % r['sigZ_median'] if r['sigZ_median'] < 1e6 else 'inf'
        sigz_mean = '%.3f' % r['sigZ_mean'] if r['sigZ_mean'] < 1e6 else 'inf'
        marker = ' <-- best' if best is not None and r is best else ''

        if has_checker:
            checker_str = '%.4f' % r.get('checker_median', float('nan')) if not np.isnan(r.get('checker_median', float('nan'))) else 'n/a'
            print("  %-12.2e  %10s  %10s  %9s  %6d  %5d  %5.1f  %s%s" %
                  (r['sigma'], sigz_med, sigz_mean, checker_str, r['n_rois'],
                   r.get('n_outliers', 0), r['time_sec'],
                   status, marker))
        else:
            print("  %-12.2e  %10s  %10s  %6d  %5d  %5.1f  %s%s" %
                  (r['sigma'], sigz_med, sigz_mean, r['n_rois'],
                   r.get('n_outliers', 0), r['time_sec'],
                   status, marker))


def find_best_sigma(results, return_result=False):
    """Find the sigma with best score (sigZ_median penalized by outlier fraction)."""
    valid = [r for r in results if r['converged'] and r['error'] is None
             and r['sigZ_median'] < 1e6]
    if not valid:
        return None
    best = min(valid, key=_score_trial)
    return best if return_result else best['sigma']


def generate_deep_dive_trial_phil(base_phil, param_name, component_idx,
                                   sigma_value, component_sigmas, best_sigmas,
                                   use_cholesky=False,
                                   max_calls=None, num_devices=None):
    """Generate a phil for a deep-dive trial where one component's sigma varies.

    component_sigmas: list of current best per-component sigmas for this param
    sigma_value: the value to try for component_idx
    best_sigmas: dict of scalar best sigmas from standard stages (for other params)
    """
    lines = [base_phil, '\n# --- hopper_heatup deep-dive trial ---\n']

    lines.append('save_spot_diagnostics = False')
    lines.append('logging.rank0_level = low')

    if num_devices is not None:
        lines.append('refiner { num_devices = %d }' % num_devices)

    if max_calls is not None:
        lines.append('lbfgs_maxiter = %d' % max_calls)
        lines.append('method = "L-BFGS-B"')

    # For deep-dive: free everything that was free in the final standard stage
    lines.append('fix {')
    lines.append('  G = False')
    lines.append('  RotXYZ = 0,0,0')
    lines.append('  Nabc = False')
    lines.append('  B = False')
    lines.append('  ucell = True')  # ucell still fixed by default
    lines.append('}')

    # Build the sigma vector for this param with one component varied
    trial_sigmas = list(component_sigmas)
    trial_sigmas[component_idx] = sigma_value

    # Write all sigmas
    lines.append('sigmas {')

    # First, write the best scalar sigmas for other params
    for p, val in best_sigmas.items():
        # Skip the param we're deep-diving on
        if p == param_name:
            continue
        # Also skip if this is the cholesky/Nabc pair
        if param_name == 'cholesky' and p == 'Nabc':
            continue
        if param_name == 'Nabc' and p == 'cholesky':
            continue
        if isinstance(val, list):
            # Already a per-component list from previous deep-dive
            if p == 'RotXYZ':
                lines.append('  RotXYZ = %s' % ','.join(['%.6g' % v for v in val[:3]]))
            elif p in ('cholesky', 'Baniso', 'ucell'):
                lines.append('  %s = %s' % (p, ','.join(['%.6g' % v for v in val[:6]])))
            elif p == 'Nabc':
                lines.append('  Nabc = %s' % ','.join(['%.6g' % v for v in val[:3]]))
            else:
                lines.append('  %s = %.6g' % (p, val[0] if val else 1.0))
        else:
            if p == 'RotXYZ':
                lines.append('  RotXYZ = %.6g,%.6g,%.6g' % (val, val, val))
            elif p == 'Nabc' and not use_cholesky:
                lines.append('  Nabc = %.6g,%.6g,%.6g' % (val, val, val))
            elif p == 'cholesky':
                lines.append('  cholesky = %.6g,%.6g,%.6g,%.6g,%.6g,%.6g' %
                              (val, val, val, val, val, val))
            elif p == 'ucell':
                lines.append('  ucell = %.6g,%.6g,%.6g,%.6g,%.6g,%.6g' %
                              (val, val, val, val, val, val))
            elif p == 'Baniso':
                lines.append('  Baniso = %.6g,%.6g,%.6g,%.6g,%.6g,%.6g' %
                              (val, val, val, val, val, val))
            else:
                lines.append('  %s = %.6g' % (p, val))

    # Write the deep-dive param with per-component sigmas
    if param_name == 'RotXYZ':
        lines.append('  RotXYZ = %s' % ','.join(['%.6g' % v for v in trial_sigmas[:3]]))
    elif param_name == 'Nabc':
        lines.append('  Nabc = %s' % ','.join(['%.6g' % v for v in trial_sigmas[:3]]))
    elif param_name == 'cholesky':
        lines.append('  cholesky = %s' % ','.join(['%.6g' % v for v in trial_sigmas[:6]]))
    elif param_name == 'ucell':
        lines.append('  ucell = %s' % ','.join(['%.6g' % v for v in trial_sigmas[:6]]))
    elif param_name == 'Baniso':
        lines.append('  Baniso = %s' % ','.join(['%.6g' % v for v in trial_sigmas[:6]]))
    elif param_name == 'B':
        lines.append('  B = %.6g' % trial_sigmas[0])
    lines.append('}')

    return '\n'.join(lines) + '\n'


def run_deep_dive_sweep(base_phil, param_name, n_components, comp_names,
                         scalar_best, best_sigmas, sample_lines,
                         multipliers, use_cholesky=False, use_cuda=False,
                         max_calls=50, n_parallel=1,
                         n_gpus=None, procs_per_gpu=1, mpi_comm=None,
                         num_devices=None, densify=1):
    """Coordinate-descent sweep of per-component sigmas for one parameter.

    Starts from scalar_best for all components, then sweeps each component
    individually while holding others at their current best.

    sample_lines: list of spec file lines (one per image)
    Returns: list of per-component best sigmas
    """
    # Initialize all components to the scalar best
    component_sigmas = [scalar_best] * n_components
    n_images = len(sample_lines)

    print("  Starting from uniform sigma = %.2e for all %d components" %
          (scalar_best, n_components))

    for comp_idx in range(n_components):
        comp_name = comp_names[comp_idx]
        sweep_values = [max(0.01, min(100.0, scalar_best * m)) for m in multipliers]
        # Remove duplicates and sort
        sweep_values = sorted(set(sweep_values))
        # Densify: interpolate between base values for finer resolution
        n_target = max(len(sweep_values), len(sweep_values) * densify)
        if n_target > len(sweep_values):
            sweep_values = densify_sweep(sweep_values, n_target)
        # Deduplicate values that would collide in trial naming (%.4e precision)
        seen = set()
        unique_vals = []
        for v in sweep_values:
            key = '%.4e' % v
            if key not in seen:
                seen.add(key)
                unique_vals.append(v)
        sweep_values = unique_vals

        print("\n  --- Component %d/%d: %s ---" % (comp_idx + 1, n_components, comp_name))
        print("  Sweep values (%d) x %d images:" % (len(sweep_values), n_images))

        # Build per-sigma trial configs for this component
        trial_args = []
        for sigma_val in sweep_values:
            phil_str = generate_deep_dive_trial_phil(
                base_phil, param_name, comp_idx, sigma_val,
                component_sigmas, best_sigmas,
                use_cholesky=use_cholesky, max_calls=max_calls,
                num_devices=num_devices)

            label = '%s=%.2e' % (comp_name, sigma_val)
            trial_args.append((phil_str, use_cuda, sigma_val, label))

        # Flatten to per-(sigma, image) work units
        flat_trials = _flatten_trials(trial_args, sample_lines)
        flat_trials = _assign_gpus(flat_trials, n_gpus, procs_per_gpu)

        raw_results = _run_trials_batch(flat_trials, n_parallel=n_parallel,
                                         mpi_comm=mpi_comm)

        # Aggregate per-image results back to per-sigma
        agg_results = aggregate_image_results(raw_results, n_images)

        results = []
        for sigma_val, label, score in agg_results:
            results.append({'sigma': sigma_val, 'component': comp_name, **score})
        results.sort(key=lambda r: r['sigma'])

        # Find best for this component
        best_val = find_best_sigma(results)
        if best_val is not None:
            component_sigmas[comp_idx] = best_val
            print("  Best %s sigma = %.2e" % (comp_name, best_val))
        else:
            print("  WARNING: No valid results for %s, keeping %.2e" %
                  (comp_name, component_sigmas[comp_idx]))

    return component_sigmas


def _run_deep_dive_for_param(dd_param, base_phil, best_sigmas, sample_lines,
                              multipliers, use_cholesky, use_cuda, max_calls,
                              n_parallel, n_gpus=None, procs_per_gpu=1, mpi_comm=None,
                              num_devices=None, densify=1):
    """Run deep-dive for one parameter. Designed for parallel execution across params."""
    phil_key, n_comp, comp_names = DEEP_DIVE_PARAMS[dd_param]

    scalar_best = best_sigmas.get(dd_param, 1.0)
    if isinstance(scalar_best, list):
        scalar_best = scalar_best[0]

    print("\n" + "-" * 50)
    print("DEEP-DIVE: %s (%d components: %s)" %
          (dd_param, n_comp, ', '.join(comp_names)))
    print("  Scalar best from standard stage: %.2e" % scalar_best)
    print("-" * 50)

    comp_sigmas = run_deep_dive_sweep(
        base_phil, dd_param, n_comp, comp_names,
        scalar_best, best_sigmas, sample_lines,
        multipliers, use_cholesky=use_cholesky, use_cuda=use_cuda,
        max_calls=max_calls, n_parallel=n_parallel,
        n_gpus=n_gpus, procs_per_gpu=procs_per_gpu, mpi_comm=mpi_comm,
        num_devices=num_devices, densify=densify)

    print("\n  Final %s sigmas:" % dd_param)
    for name, sig in zip(comp_names, comp_sigmas):
        ratio = sig / scalar_best if scalar_best > 0 else float('inf')
        print("    %s: %.4e (%.1fx scalar)" % (name, sig, ratio))

    return dd_param, {
        'comp_names': comp_names,
        'comp_sigmas': comp_sigmas,
        'scalar_best': scalar_best,
    }


def run_final_refinement(base_phil, best_sigmas, sample_lines,
                          use_cholesky=False, use_cuda=False, max_calls=500,
                          n_parallel=1, n_gpus=None,
                          procs_per_gpu=1, mpi_comm=None, num_devices=None,
                          init_G=None, refine_ucell=False):
    """Run a final refinement with optimized sigmas, return median refined params.

    Runs in-process refinement on each sample image and extracts refined
    parameter values directly from the result dicts.

    sample_lines: list of spec file lines (one per image)
    Returns dict with keys: G, B, Nabc (tuple of 3), var_G, etc., or None on failure.
    """
    import numpy as np

    lines = [base_phil, '\n# --- Final refinement with optimized sigmas ---\n']
    lines.append('save_spot_diagnostics = False')
    lines.append('logging.rank0_level = low')

    if num_devices is not None:
        lines.append('refiner { num_devices = %d }' % num_devices)

    if init_G is not None:
        lines.append('init.G = %.6e' % init_G)
        lines.append('init.auto_G = False')

    lines.append('lbfgs_maxiter = %d' % max_calls)
    lines.append('method = "L-BFGS-B"')

    # All parameters free (ucell conditional on refine_ucell)
    lines.append('fix {')
    lines.append('  G = False')
    lines.append('  RotXYZ = 0,0,0')
    lines.append('  Nabc = False')
    lines.append('  B = False')
    lines.append('  ucell = %s' % ('False' if refine_ucell else 'True'))
    lines.append('}')

    # Optimized sigmas
    lines.append('sigmas {')
    for param, val in best_sigmas.items():
        if isinstance(val, list):
            lines.append('  %s = %s' % (param, ','.join(['%.6g' % v for v in val])))
        elif param == 'RotXYZ':
            lines.append('  RotXYZ = %.6g,%.6g,%.6g' % (val, val, val))
        elif param == 'Nabc' and not use_cholesky:
            lines.append('  Nabc = %.6g,%.6g,%.6g' % (val, val, val))
        elif param == 'cholesky':
            lines.append('  cholesky = %.6g,%.6g,%.6g,%.6g,%.6g,%.6g' %
                          (val, val, val, val, val, val))
        elif param == 'ucell':
            lines.append('  ucell = %.6g,%.6g,%.6g,%.6g,%.6g,%.6g' %
                          (val, val, val, val, val, val))
        elif param == 'Baniso':
            lines.append('  Baniso = %.6g,%.6g,%.6g,%.6g,%.6g,%.6g' %
                          (val, val, val, val, val, val))
        else:
            lines.append('  %s = %.6g' % (param, val))
    lines.append('}')

    phil_str = '\n'.join(lines) + '\n'

    n_images = len(sample_lines)
    print("  Final refinement: %d images" % n_images)

    # Run one image per work unit (no sigma sweep — just one phil config)
    trial_args = [(phil_str, use_cuda, None, 'final')]
    flat_trials = _flatten_trials(trial_args, sample_lines)
    flat_trials = _assign_gpus(flat_trials, n_gpus, procs_per_gpu)

    raw_results = _run_trials_batch(flat_trials, n_parallel=n_parallel,
                                     mpi_comm=mpi_comm)

    # Filter successful results
    good_scores = [score for _, _, score in raw_results
                   if score.get('error') is None and score.get('converged', False)]

    if not good_scores:
        print("  WARNING: Final refinement failed: no successful images")
        return None

    # Aggregate: sigZ stats
    agg_results = aggregate_image_results(raw_results, n_images)
    if agg_results:
        _, _, agg_score = agg_results[0]
        print("  Final refinement: sigZ_median=%.3f, n_rois=%d, time=%.1fs" %
              (agg_score['sigZ_median'], agg_score['n_rois'], agg_score['time_sec']))

    # Extract refined parameters from per-image param_vals dicts
    # Weight by n_rois / sigz_median^2 (good fits count more)
    frame_rows = []
    for s in good_scores:
        pv = s.get('param_vals', {})
        if not pv:
            continue
        sigz = s.get('sigz_median', np.nan)
        nrois = s.get('n_rois', 0)
        weight = float(nrois) / float(sigz) ** 2 if (np.isfinite(sigz) and sigz > 0 and nrois > 0) else 0.0
        fr = {'weight': weight}
        # Map from get_param_from_x keys to our internal keys
        if 'scale' in pv and np.isfinite(pv['scale']):
            fr['G'] = float(pv['scale'])
        if 'Bfactor' in pv and np.isfinite(pv['Bfactor']):
            fr['B'] = float(pv['Bfactor'])
        for key in ('Na', 'Nb', 'Nc'):
            if key in pv and np.isfinite(pv[key]):
                fr[key] = float(pv[key])
        if 'cholesky' in pv and len(pv['cholesky']) == 6:
            if all(np.isfinite(v) for v in pv['cholesky']):
                fr['chol'] = tuple(float(v) for v in pv['cholesky'])
        for i, rkey in enumerate(['rotX', 'rotY', 'rotZ']):
            if rkey in pv and np.isfinite(pv[rkey]):
                fr.setdefault('rot_vals', [None, None, None])
                fr['rot_vals'][i] = float(pv[rkey])
        if 'rot_vals' in fr and all(v is not None for v in fr['rot_vals']):
            fr['rot'] = tuple(fr['rot_vals'])
            del fr['rot_vals']
        elif 'rot_vals' in fr:
            del fr['rot_vals']
        uc_keys = ['a', 'b', 'c', 'al', 'be', 'ga']
        uc_vals = [pv.get(k) for k in uc_keys]
        if all(v is not None and np.isfinite(v) for v in uc_vals):
            fr['ucell'] = tuple(float(v) for v in uc_vals)
        frame_rows.append(fr)

    if not frame_rows:
        print("  WARNING: Could not extract refined parameters from results")
        return None

    def _weighted_var(vals, weights, robust=True):
        vals = np.array(vals, dtype=float)
        weights = np.array(weights, dtype=float)
        if len(vals) < 2 or weights.sum() == 0:
            return 0.0
        if robust and len(vals) >= 3:
            median = np.median(vals)
            mad = np.median(np.abs(vals - median))
            if mad > 0:
                modified_z = 0.6745 * np.abs(vals - median) / mad
                keep = modified_z <= 3.5
                if keep.sum() >= 2:
                    vals = vals[keep]
                    weights = weights[keep]
        if len(vals) < 2 or weights.sum() == 0:
            return 0.0
        wmean = np.average(vals, weights=weights)
        return float(np.average((vals - wmean) ** 2, weights=weights))

    def _weighted_median(vals, weights):
        vals = np.array(vals)
        weights = np.array(weights)
        if weights.sum() == 0:
            return float(np.median(vals))
        order = np.argsort(vals)
        vals, weights = vals[order], weights[order]
        cumw = np.cumsum(weights)
        idx = np.searchsorted(cumw, cumw[-1] * 0.5)
        return float(vals[min(idx, len(vals) - 1)])

    def _extract(key):
        vals = [fr[key] for fr in frame_rows if key in fr]
        wts = [fr['weight'] for fr in frame_rows if key in fr]
        return vals, wts

    n_frames = len(frame_rows)
    refined_params = {}

    g_vals, g_wts = _extract('G')
    if g_vals:
        refined_params['G'] = _weighted_median(g_vals, g_wts)
        if len(g_vals) >= 2:
            refined_params['var_G'] = _weighted_var(g_vals, g_wts)

    b_vals, b_wts = _extract('B')
    if b_vals:
        refined_params['B'] = _weighted_median(b_vals, b_wts)
        if len(b_vals) >= 2:
            refined_params['var_B'] = _weighted_var(b_vals, b_wts)

    nabc_frames = [(fr, fr['weight']) for fr in frame_rows
                   if all(k in fr for k in ('Na', 'Nb', 'Nc'))]
    if nabc_frames:
        nabc_vals = np.array([(fr['Na'], fr['Nb'], fr['Nc']) for fr, _ in nabc_frames])
        nabc_wts = [w for _, w in nabc_frames]
        refined_params['Nabc'] = tuple(
            _weighted_median(nabc_vals[:, i], nabc_wts) for i in range(3))
        if len(nabc_frames) >= 2:
            refined_params['var_Nabc'] = tuple(
                _weighted_var(nabc_vals[:, i], nabc_wts) for i in range(3))

    chol_frames = [(fr, fr['weight']) for fr in frame_rows if 'chol' in fr]
    if chol_frames:
        chol_vals = np.array([fr['chol'] for fr, _ in chol_frames])
        chol_wts = [w for _, w in chol_frames]
        refined_params['cholesky'] = tuple(
            _weighted_median(chol_vals[:, i], chol_wts) for i in range(6))
        if len(chol_frames) >= 2:
            refined_params['var_cholesky'] = tuple(
                _weighted_var(chol_vals[:, i], chol_wts) for i in range(6))

    rot_frames = [(fr, fr['weight']) for fr in frame_rows if 'rot' in fr]
    if rot_frames:
        rot_vals = np.array([fr['rot'] for fr, _ in rot_frames])
        rot_wts = [w for _, w in rot_frames]
        if len(rot_frames) >= 2:
            refined_params['var_RotXYZ'] = float(np.mean([
                _weighted_var(rot_vals[:, i], rot_wts) for i in range(3)]))

    ucell_frames = [(fr, fr['weight']) for fr in frame_rows if 'ucell' in fr]
    if ucell_frames:
        ucell_vals = np.array([fr['ucell'] for fr, _ in ucell_frames])
        ucell_wts = [w for _, w in ucell_frames]
        refined_params['ucell'] = tuple(
            _weighted_median(ucell_vals[:, i], ucell_wts) for i in range(6))
        if len(ucell_frames) >= 2:
            refined_params['var_ucell'] = tuple(
                _weighted_var(ucell_vals[:, i], ucell_wts) for i in range(6))

    if refined_params:
        all_wts = [fr['weight'] for fr in frame_rows]
        n_nonzero = sum(1 for w in all_wts if w > 0)
        print("  Refined parameters from %d frames (%d with nonzero weight):" %
              (n_frames, n_nonzero))
        print("  (weighted by n_rois / sigZ_median^2 — good fits count more)")
        if 'G' in refined_params:
            print("    G = %.4e" % refined_params['G'])
            if 'var_G' in refined_params:
                print("      var(G) = %.4e  →  beta_init = %.4e" %
                      (refined_params['var_G'], refined_params['var_G']))
        if 'B' in refined_params:
            print("    B = %.2f" % refined_params['B'])
            if 'var_B' in refined_params:
                print("      var(B) = %.4f  →  beta_init = %.4f" %
                      (refined_params['var_B'], refined_params['var_B']))
        if 'Nabc' in refined_params:
            print("    Nabc = (%.1f, %.1f, %.1f)" % refined_params['Nabc'])
            if 'var_Nabc' in refined_params:
                print("      var(Nabc) = (%.4f, %.4f, %.4f)" % refined_params['var_Nabc'])
        if 'cholesky' in refined_params:
            L = refined_params['cholesky']
            print("    cholesky = (%.3f, %.3f, %.3f, %.3f, %.3f, %.3f)" % L)
            if 'var_cholesky' in refined_params:
                print("      var(chol) = (%s)" %
                      ', '.join('%.4f' % v for v in refined_params['var_cholesky']))
        if 'var_RotXYZ' in refined_params:
            print("    var(RotXYZ) = %.4e  →  beta_init = %.4e" %
                  (refined_params['var_RotXYZ'], refined_params['var_RotXYZ']))
        if 'ucell' in refined_params:
            uc = refined_params['ucell']
            print("    ucell = (%.4f, %.4f, %.4f, %.4f, %.4f, %.4f)" % uc)
            if 'var_ucell' in refined_params:
                print("      var(ucell) = (%s)" %
                      ', '.join('%.4g' % v for v in refined_params['var_ucell']))
    else:
        print("  WARNING: Could not extract refined parameters from results")
        return None

    return refined_params


def build_tuning_base_phil(base_phil, best_sigmas, init_G, refined_params,
                            use_cholesky=False, max_calls=500, refine_ucell=False):
    """Build a phil string for the tuning phase: all params free, best sigmas set,
    refined init values set, centers set from refined params."""
    lines = [base_phil, '\n# --- Tuning phase base phil ---\n']
    lines.append('logging.rank0_level = low')

    # Init values from final refinement
    if refined_params and 'G' in refined_params:
        lines.append('init.G = %.6e' % refined_params['G'])
        lines.append('init.auto_G = False')
    elif init_G is not None:
        lines.append('init.G = %.6e' % init_G)
        lines.append('init.auto_G = False')
    if refined_params and 'B' in refined_params:
        lines.append('init.B = %.4f' % refined_params['B'])
    if refined_params and 'Nabc' in refined_params:
        na, nb, nc = refined_params['Nabc']
        lines.append('init.Nabc = %.2f,%.2f,%.2f' % (na, nb, nc))
    if refined_params and 'cholesky' in refined_params:
        L = refined_params['cholesky']
        lines.append('init.cholesky = %.6f,%.6f,%.6f,%.6f,%.6f,%.6f' % L)
    # Note: no init.ucell phil param — ucell is initialized from the crystal model

    lines.append('lbfgs_maxiter = %d' % max_calls)
    lines.append('method = "L-BFGS-B"')

    # All parameters free (ucell conditional on refine_ucell)
    lines.append('fix {')
    lines.append('  G = False')
    lines.append('  RotXYZ = 0,0,0')
    lines.append('  Nabc = False')
    lines.append('  B = False')
    lines.append('  ucell = %s' % ('False' if refine_ucell else 'True'))
    lines.append('}')

    # Best sigmas
    lines.append('sigmas {')
    for param, val in best_sigmas.items():
        if isinstance(val, list):
            lines.append('  %s = %s' % (param, ','.join(['%.6g' % v for v in val])))
        elif param == 'RotXYZ':
            lines.append('  RotXYZ = %.6g,%.6g,%.6g' % (val, val, val))
        elif param == 'Nabc' and not use_cholesky:
            lines.append('  Nabc = %.6g,%.6g,%.6g' % (val, val, val))
        elif param == 'cholesky':
            lines.append('  cholesky = %.6g,%.6g,%.6g,%.6g,%.6g,%.6g' %
                          (val, val, val, val, val, val))
        elif param == 'ucell':
            lines.append('  ucell = %.6g,%.6g,%.6g,%.6g,%.6g,%.6g' %
                          (val, val, val, val, val, val))
        else:
            lines.append('  %s = %.6g' % (param, val))
    lines.append('}')

    # Enable restraints (defaults to False in phil.py!)
    lines.append('use_restraints = True')

    # Centers and betas from refined params (restraint targets)
    # Both must be set for each param, or hopper raises:
    # "must specify both center and beta for param ..."
    if refined_params:
        lines.append('centers {')
        if 'G' in refined_params:
            lines.append('  G = %.6e' % refined_params['G'])
        if 'B' in refined_params:
            lines.append('  B = %.4f' % refined_params['B'])
        if 'cholesky' in refined_params and use_cholesky:
            L = refined_params['cholesky']
            lines.append('  cholesky = %.4f,%.4f,%.4f,%.4f,%.4f,%.4f' % L)
        elif 'Nabc' in refined_params:
            na, nb, nc = refined_params['Nabc']
            if use_cholesky:
                # Fallback: convert Nabc centers to Cholesky centers (diagonal L)
                L11 = np.sqrt(na)
                L22 = np.sqrt(nb)
                L33 = np.sqrt(nc)
                lines.append('  cholesky = %.4f,0,%.4f,0,0,%.4f' % (L11, L22, L33))
            else:
                lines.append('  Nabc = %.2f,%.2f,%.2f' % (na, nb, nc))
        lines.append('  RotXYZ = 0,0,0')
        if 'ucell' in refined_params:
            uc = refined_params['ucell']
            _uc_names = ['ucell_a', 'ucell_b', 'ucell_c',
                         'ucell_alpha', 'ucell_beta', 'ucell_gamma']
            for name, val in zip(_uc_names, uc):
                lines.append('  %s = %.4f' % (name, val))
        lines.append('}')

        # Default betas (variance from final refinement, or large = loose restraint).
        # Beta sweeps will override individual betas, but non-beta sweeps
        # (phisteps, sigma_r) need valid betas or hopper crashes.
        lines.append('betas {')
        if 'G' in refined_params:
            beta_G = refined_params.get('var_G', refined_params['G'] ** 2)
            lines.append('  G = %.6e' % beta_G)
        if 'B' in refined_params:
            beta_B = refined_params.get('var_B', 1e4)
            lines.append('  B = %.6e' % beta_B)
        if 'cholesky' in refined_params and use_cholesky:
            var_chol = refined_params.get('var_cholesky', None)
            if var_chol:
                beta_chol = float(np.mean(var_chol))
            else:
                beta_chol = 1.0
            lines.append('  cholesky = %.6g,%.6g,%.6g,%.6g,%.6g,%.6g' %
                          ((beta_chol,) * 6))
        elif 'Nabc' in refined_params:
            var_nabc = refined_params.get('var_Nabc', None)
            if var_nabc:
                beta_nabc = float(np.mean(var_nabc))
            else:
                beta_nabc = 100.0
            if use_cholesky:
                lines.append('  cholesky = %.6g,%.6g,%.6g,%.6g,%.6g,%.6g' %
                              ((beta_nabc,) * 6))
            else:
                lines.append('  Nabc = %.6g,%.6g,%.6g' % ((beta_nabc,) * 3))
        beta_rot = refined_params.get('var_RotXYZ', 1e-2)
        lines.append('  RotXYZ = %.6e' % beta_rot)
        if 'ucell' in refined_params:
            _uc_names = ['ucell_a', 'ucell_b', 'ucell_c',
                         'ucell_alpha', 'ucell_beta', 'ucell_gamma']
            var_uc = refined_params.get('var_ucell')
            for i, name in enumerate(_uc_names):
                if var_uc and var_uc[i] > 0:
                    lines.append('  %s = %.6g' % (name, var_uc[i]))
                else:
                    lines.append('  %s = 1e10' % name)
        lines.append('}')

    return '\n'.join(lines) + '\n'


def build_model_sweep_phil(base_phil, best_sigmas, init_G=None,
                           use_cholesky=False, max_calls=100, refine_ucell=False):
    """Build a phil for model param sweeps (phisteps, sigma_r).
    No centers/betas — just best sigmas and all params free.
    Used before deep-dive to find the right model settings."""
    lines = [base_phil, '\n# --- Model parameter sweep ---\n']
    lines.append('logging.rank0_level = low')

    if init_G is not None:
        lines.append('init.G = %.6e' % init_G)
        lines.append('init.auto_G = False')

    lines.append('lbfgs_maxiter = %d' % max_calls)
    lines.append('method = "L-BFGS-B"')

    # All parameters free (ucell conditional on refine_ucell)
    lines.append('fix {')
    lines.append('  G = False')
    lines.append('  RotXYZ = 0,0,0')
    lines.append('  Nabc = False')
    lines.append('  B = False')
    lines.append('  ucell = %s' % ('False' if refine_ucell else 'True'))
    lines.append('}')

    # Best sigmas
    lines.append('sigmas {')
    for param, val in best_sigmas.items():
        if isinstance(val, list):
            lines.append('  %s = %s' % (param, ','.join(['%.6g' % v for v in val])))
        elif param == 'RotXYZ':
            lines.append('  RotXYZ = %.6g,%.6g,%.6g' % (val, val, val))
        elif param == 'Nabc' and not use_cholesky:
            lines.append('  Nabc = %.6g,%.6g,%.6g' % (val, val, val))
        elif param == 'cholesky':
            lines.append('  cholesky = %.6g,%.6g,%.6g,%.6g,%.6g,%.6g' %
                          (val, val, val, val, val, val))
        elif param == 'ucell':
            lines.append('  ucell = %.6g,%.6g,%.6g,%.6g,%.6g,%.6g' %
                          (val, val, val, val, val, val))
        else:
            lines.append('  %s = %.6g' % (param, val))
    lines.append('}')

    return '\n'.join(lines) + '\n'


def run_geometry_refinement_tuning(base_phil, spec_lines, refined_params,
                                   refine_goniometer=False, method="lbfgsb",
                                   max_calls=100, panel_group_file=None,
                                   use_cuda=False, outdir=None, mpi_comm=None):
    """Run ensemble geometry refinement during Phase 2 tuning.

    This refines detector and optionally goniometer using geom_min(), similar
    to how phisteps/sigma_r are tuned in Phase 2, but not a sweep - just a
    single optimization run.

    Args:
        base_phil: Base phil string (from tuning_base)
        spec_lines: List of spec file lines for geometry refinement
        refined_params: Dict of refined parameters from Phase 1 (medians)
        refine_goniometer: Whether to refine goniometer axis
        method: Optimization method (lbfgsb or nelder)
        max_calls: Max iterations
        panel_group_file: Panel grouping file (or None)
        use_cuda: Use GPU
        outdir: Output directory
        mpi_comm: MPI communicator

    Returns:
        Path to optimized detector expt file (or None if failed)
    """
    import os
    from libtbx.phil import parse
    from simtbx.diffBragg import phil as diffBragg_phil
    from simtbx.diffBragg.phil import hopper_phil
    from simtbx.diffBragg.refiners.geometry import geom_min
    from simtbx.diffBragg import hopper_utils

    # Create combined phil scope that includes hopper_phil
    combined_phil_str = hopper_phil + diffBragg_phil.philz
    combined_phil_scope = parse(combined_phil_str)

    if mpi_comm is None:
        try:
            from mpi4py import MPI
            mpi_comm = MPI.COMM_WORLD
        except ImportError:
            # Create mock communicator for non-MPI case
            class MockComm:
                rank = 0
                size = 1
                def barrier(self): pass
                def reduce(self, data, root=0): return data
                def bcast(self, data, root=0): return data
            mpi_comm = MockComm()

    rank = mpi_comm.rank

    if outdir is None:
        outdir = "geom_tuning_output"

    if rank == 0:
        os.makedirs(outdir, exist_ok=True)
        pandas_dir = os.path.join(outdir, "pandas")
        os.makedirs(pandas_dir, exist_ok=True)

    mpi_comm.barrier()
    pandas_dir = os.path.join(outdir, "pandas")

    # Step 1: Generate pandas pickles for geometry refinement
    if rank == 0:
        print("  Step 1: Generating pandas pickles...")

    # Create phil for pickle generation
    geom_phil = base_phil + '\n'
    geom_phil += 'debug_mode = True\n'
    geom_phil += 'save_pandas = True\n'
    geom_phil += 'load_data_from_refls = False\n'
    # Note: outdir should be the parent directory; make_rank_outdir will add "pandas/rank*"
    geom_phil += 'outdir = %s\n' % outdir
    geom_phil += 'logging.rank0_level = low\n'
    geom_phil += 'save_spot_diagnostics = False\n'

    # Fix crystal params (should already be in base_phil, but ensure)
    geom_phil += 'fix.scale = True\n'
    geom_phil += 'fix.Nabc = True\n'
    geom_phil += 'fix.RotXYZ = 0,0,0\n'
    geom_phil += 'fix.B = True\n'
    geom_phil += 'fix.ucell = True\n'

    # Parse phil
    try:
        params = combined_phil_scope.fetch(
            source=parse(geom_phil)
        ).extract()
    except Exception as e:
        if rank == 0:
            print("  ERROR parsing phil: %s" % str(e))
        return None

    # Distribute spec_lines across ranks
    my_lines = [line for i, line in enumerate(spec_lines) if i % mpi_comm.size == rank]

    n_success = 0
    n_fail = 0
    for i_line, line in enumerate(my_lines):
        try:
            if rank == 0:
                print("    DEBUG: Processing line %d: %s" % (i_line, line[:80]))
            exp, ref, exp_idx, spec = hopper_utils.split_line(line)
            if rank == 0:
                print("    DEBUG: exp=%s ref=%s spec=%s" % (exp[:40], ref[:40], spec[:40] if spec else "None"))
            new_exp, new_refl, Modeler, SIM, x = hopper_utils.refine(
                exp, ref, params, spec=spec,
                gpu_device=0 if use_cuda else None,
                return_modeler=True)
            if rank == 0:
                print("    DEBUG: refine() completed, calling save_up()")

            # Set exper_name and refl_name (required for save_up)
            Modeler.exper_name = exp
            Modeler.refl_name = ref
            Modeler.exper_idx = exp_idx

            # Create gathered reflection file for geometry refinement
            # This contains the observed and predicted pixel data from the model
            geom_gathers_dir = os.path.join(outdir, "geom_gathers")
            if rank == 0:
                os.makedirs(geom_gathers_dir, exist_ok=True)
            mpi_comm.barrier()

            basename = os.path.splitext(os.path.basename(exp))[0]
            geom_output_name = "%s_geomData.refl" % basename
            geom_output_name = os.path.join(geom_gathers_dir, geom_output_name)

            # Dump gathered data to reflection file (contains obs/pred pixels from model)
            if rank == 0:
                print("    DEBUG: Creating gathered refl file: %s" % geom_output_name)
            Modeler.dump_gathered_to_refl(geom_output_name, per_roi_scales=None)
            if rank == 0 and os.path.exists(geom_output_name):
                print("    DEBUG: Gathered refl file created successfully, size: %d bytes" % os.path.getsize(geom_output_name))

            # Explicitly call save_up to save pandas pickle
            # Note: must save_expt=True for geometry refinement to work (geom_exp needs valid path)
            shot_df = Modeler.save_up(x, SIM, rank=rank, i_shot=i_line,
                           save_fhkl_data=False,
                           save_modeler_file=False,
                           save_refl=False,
                           save_sim_info=False,
                           save_traces=False,
                           save_pandas=True,
                           save_expt=True)

            # CRITICAL: Set geom_ref to the gathered reflection file (not stage1_refls!)
            # This is what geometry refinement expects
            shot_df["geom_ref"] = os.path.abspath(geom_output_name)
            shot_df["geom_exp"] = shot_df["opt_exp_name"].values[0]
            shot_df["geom_exp_idx"] = 0

            if rank == 0:
                print("    DEBUG: Setting geom_ref to: %s" % shot_df["geom_ref"].values[0])
                print("    DEBUG: Setting geom_exp to: %s" % shot_df["geom_exp"].values[0])

            # Re-save the pickle with the geometry columns
            rank_pandas_outdir = os.path.join(outdir, "pandas", "rank%d" % rank)
            pandas_path = os.path.join(rank_pandas_outdir, "stage1_%s_%d.pkl" % (basename, i_line))
            shot_df.to_pickle(pandas_path)

            if rank == 0:
                print("    DEBUG: Saved pickle to: %s" % pandas_path)

            if rank == 0:
                print("    DEBUG: save_up() completed successfully")
            n_success += 1
        except Exception as e:
            n_fail += 1
            if rank == 0 and n_fail <= 3:  # Print first 3 errors
                import traceback
                print("    WARNING: Failed on image %d" % i_line)
                print("    ERROR TYPE: %s" % type(e).__name__)
                print("    ERROR MESSAGE: %s" % repr(str(e)))
                print("    TRACEBACK:")
                traceback.print_exc()

    # Gather success/fail counts
    all_success = mpi_comm.reduce(n_success, root=0)
    all_fail = mpi_comm.reduce(n_fail, root=0)

    if rank == 0:
        print("    Generated %d pickles (%d failed)" % (all_success, all_fail))

    mpi_comm.barrier()

    # Step 2: Run geometry refinement
    if rank == 0:
        print("\n  Step 2: Running geometry refinement...")
        # Debug: check what pickles exist
        import glob as glob_module
        import subprocess
        pkl_pattern = '%s/rank*/*.pkl' % pandas_dir
        found_pickles = glob_module.glob(pkl_pattern)
        print("  DEBUG: Looking for pickles with pattern: %s" % pkl_pattern)
        print("  DEBUG: Found %d pickle files" % len(found_pickles))
        # Check actual directory structure
        result = subprocess.run(['find', pandas_dir, '-name', '*.pkl'],
                              capture_output=True, text=True)
        if result.stdout:
            print("  DEBUG: Actual pickle files found:\n%s" % result.stdout[:500])
        else:
            print("  DEBUG: No pickle files found with find command")
            # Check if pandas_dir exists and what's in it
            result2 = subprocess.run(['ls', '-laR', pandas_dir],
                                   capture_output=True, text=True)
            print("  DEBUG: pandas_dir contents:\n%s" % result2.stdout[:500])

    # Create geometry refinement phil
    geom_refine_phil = base_phil + '\n'
    geom_refine_phil += 'geometry.input_pkl_glob = %s/rank*/*.pkl\n' % pandas_dir
    geom_refine_phil += 'geometry.optimize = True\n'
    geom_refine_phil += 'geometry.optimize_method = %s\n' % method
    geom_refine_phil += 'geometry.max_calls = %d\n' % max_calls
    geom_refine_phil += 'outdir = %s\n' % outdir

    # CRITICAL: Load data from gathered reflection files (not images!)
    # This is what makes geometry refinement work - it uses the obs/pred pixel data
    # from the gathered refl files created by dump_gathered_to_refl()
    geom_refine_phil += 'refiner.load_data_from_refl = True\n'

    # Goniometer refinement
    if refine_goniometer:
        geom_refine_phil += 'geometry.fix.gonio_axis = False\n'
        geom_refine_phil += 'geometry.sigma_gonio_axis = 0.01\n'
    else:
        geom_refine_phil += 'geometry.fix.gonio_axis = True\n'

    # Panel grouping
    if panel_group_file is not None:
        geom_refine_phil += 'geometry.panel_group_file = %s\n' % panel_group_file

    # Fix crystal parameters during geometry refinement
    geom_refine_phil += 'fix.G = True\n'
    geom_refine_phil += 'fix.Nabc = True\n'
    geom_refine_phil += 'fix.Ndef = True\n'
    geom_refine_phil += 'fix.RotXYZ = 0,0,0\n'  # REFINE crystal orientation during geometry
    geom_refine_phil += 'fix.B = True\n'
    geom_refine_phil += 'fix.ucell = True\n'
    geom_refine_phil += 'fix.eta_abc = True\n'
    geom_refine_phil += 'fix.perRoiScale = True\n'

    # Parse and run
    try:
        params = combined_phil_scope.fetch(
            source=parse(geom_refine_phil)
        ).extract()

        geom_min(params)

        # Check for optimized detector
        optimized_detector = os.path.join(outdir, params.geometry.optimized_detector_name)

        # Broadcast result to all ranks
        if rank == 0:
            if os.path.exists(optimized_detector):
                result = optimized_detector
            else:
                result = None
        else:
            result = None

        result = mpi_comm.bcast(result, root=0)

        if rank == 0:
            if result is not None:
                print("    Geometry refinement complete!")
                print("    Detector: %s" % result)
            else:
                print("    WARNING: Optimized detector not found!")

        return result

    except Exception as e:
        if rank == 0:
            print("  ERROR in geometry refinement: %s" % str(e))
            import traceback
            traceback.print_exc()
        return None


# Mapping from beta param name to CSV columns and unrestricted variance key
BETA_PARAM_MAP = {
    'betas.G': {
        'csv_cols': ['G'],
        'var_key': 'var_G',
    },
    'betas.B': {
        'csv_cols': ['B'],
        'var_key': 'var_B',
    },
    'betas.cholesky': {
        'csv_cols': ['chol_L11', 'chol_L21', 'chol_L22',
                     'chol_L31', 'chol_L32', 'chol_L33'],
        'var_key': 'var_cholesky',
    },
    'betas.Nabc': {
        'csv_cols': ['Na', 'Nb', 'Nc'],
        'var_key': 'var_Nabc',
    },
    'betas.RotXYZ': {
        'csv_cols': ['rotX', 'rotY', 'rotZ'],
        'var_key': 'var_RotXYZ',
    },
    'betas.ucell': {
        'csv_cols': ['uc_a', 'uc_b', 'uc_c', 'uc_al', 'uc_be', 'uc_ga'],
        'var_key': 'var_ucell',
    },
}


def compute_spread_ratio(param_vars, beta_name, refined_params=None, baseline_vars=None):
    """Compute spread_ratio = mean(var(cols)) / unrestricted_var for a beta sweep trial.

    param_vars: dict from parse_trial_results (col -> variance) for the trial
    beta_name: e.g. 'betas.G'
    baseline_vars: dict from unrestrained baseline trial (col -> variance), preferred
    refined_params: dict from run_final_refinement (has var_G, var_B, etc.), fallback

    When baseline_vars is provided, uses same-conditions variance as denominator
    (apples-to-apples comparison). Falls back to refined_params if no baseline.

    Returns spread_ratio (float), or None if data is missing.
    """
    info = BETA_PARAM_MAP.get(beta_name)
    if info is None:
        return None

    # Get denominator: unrestricted variance
    if baseline_vars is not None:
        # Preferred: same-conditions baseline (same frames, ROI fraction, max_calls)
        denom_vars = []
        for col in info['csv_cols']:
            if col in baseline_vars:
                denom_vars.append(baseline_vars[col])
        if not denom_vars:
            return None
        denom = float(np.mean(denom_vars))
    elif refined_params is not None:
        # Fallback: final-refinement variance (different conditions)
        unr_var = refined_params.get(info['var_key'])
        if unr_var is None:
            return None
        if isinstance(unr_var, tuple):
            denom = float(np.mean(unr_var))
        else:
            denom = float(unr_var)
    else:
        return None

    if denom <= 0:
        return None

    # Get trial variance for the relevant columns
    trial_vars = []
    for col in info['csv_cols']:
        if col in param_vars:
            trial_vars.append(param_vars[col])
    if not trial_vars:
        return None

    trial_var = float(np.mean(trial_vars))
    return trial_var / denom


def interpolate_beta_for_spread(beta_spread_pairs, target_spread):
    """Given list of (beta, spread_ratio) pairs, interpolate beta for a target spread.

    Uses log-space interpolation on beta (since betas span orders of magnitude).
    Returns the interpolated beta, or None if extrapolation would be needed.
    """
    # Filter to valid pairs and sort by spread_ratio
    valid = [(b, s) for b, s in beta_spread_pairs if s is not None and s > 0 and b > 0]
    if len(valid) < 2:
        return None

    # Sort by spread_ratio
    valid.sort(key=lambda x: x[1])
    spreads = np.array([s for _, s in valid])
    log_betas = np.array([np.log10(b) for b, _ in valid])

    # Clamp target to observed range (don't extrapolate)
    if target_spread <= spreads[0]:
        return valid[0][0]  # tightest observed
    if target_spread >= spreads[-1]:
        return valid[-1][0]  # loosest observed

    # Interpolate in (spread → log10(beta)) space
    log_beta = float(np.interp(target_spread, spreads, log_betas))
    return 10.0 ** log_beta


def find_elbow_betas(beta_curves):
    """Find optimal per-parameter betas by detecting the elbow in each spread-ratio curve.

    For each parameter, computes d(spread)/d(log_beta) between consecutive points,
    then picks the beta at the top of the steepest positive gradient in the transition
    zone (spread < 0.95). This gives the lightest-touch restraint that still has effect.

    Returns dict mapping param name -> (elbow_beta, spread_at_elbow, gradient, details_str).
    """
    results = {}
    for bname, pairs in beta_curves.items():
        valid = [(b, s) for b, s in pairs if s is not None and s > 0 and b > 0]
        if len(valid) < 3:
            continue

        valid.sort(key=lambda x: x[0])
        betas = np.array([b for b, _ in valid])
        spreads = np.array([s for _, s in valid])
        log_betas = np.log10(betas)

        # Compute gradient d(spread)/d(log_beta) for each interval
        grads = []
        for i in range(len(log_betas) - 1):
            dspread = spreads[i + 1] - spreads[i]
            dlogb = log_betas[i + 1] - log_betas[i]
            if abs(dlogb) < 1e-12:
                continue
            grad = dspread / dlogb
            mid_spread = (spreads[i] + spreads[i + 1]) / 2
            grads.append((grad, mid_spread, i))

        # Filter to positive gradients in the transition zone (below plateau)
        grads_pos = [g for g in grads if g[0] > 0.05 and g[1] < 0.95]
        if not grads_pos:
            continue

        # Find steepest gradient
        best = max(grads_pos, key=lambda g: g[0])
        idx = best[2]

        # Pick the beta at the TOP of the sharp rise (upper end of interval)
        elbow_beta = betas[idx + 1]
        elbow_spread = spreads[idx + 1]
        gradient = best[0]

        short_name = bname.split('.', 1)[1] if '.' in bname else bname
        details = "%.3g->%.3g (grad=%.2f/decade)" % (
            spreads[idx], spreads[idx + 1], gradient)

        results[bname] = (elbow_beta, elbow_spread, gradient, details)

    return results


def _adaptive_beta_search_DISABLED(tune_name, tuning_base, sample_spec, spec_path,
                          tmpdir, baseline_vars, refined_params,
                          n_frames, n_nearby, target_spread=0.5,
                          max_evals=10, max_frame_doublings=2,
                          use_cuda=False, timeout=1800, n_parallel=1,
                          n_gpus=None, procs_per_gpu=1,
                          mpi_comm=None, num_devices=None,
                          densify=1):
    """Adaptive bracketing search for optimal beta (restraint variance).

    Instead of sweeping a predefined range, starts at beta=baseline_variance
    and adaptively brackets the transition zone where spread_ratio crosses the
    target. If estimates are too noisy (spread > 2.0 everywhere), doubles
    n_frames and re-runs baseline for more stable variance estimates.

    Args:
        tune_name: e.g. 'betas.G'
        tuning_base: base phil string for tuning trials
        sample_spec: path to current sample spec file
        spec_path: path to full exp_ref_spec.txt (for re-sampling with more frames)
        tmpdir: working directory
        baseline_vars: dict of param -> variance from unrestrained baseline
        refined_params: dict from final refinement (has var_G, etc.)
        n_frames: current number of phi positions sampled
        n_nearby: neighboring frames per position
        target_spread: target spread_ratio for the elbow (default 0.5)
        max_evals: max number of beta evaluations per attempt (default 10)
        max_frame_doublings: max times to double n_frames (default 2)

    Returns:
        (elbow_beta, elbow_spread, beta_curve, new_baseline_vars, new_n_frames)
        beta_curve is list of (beta, spread_ratio) for all evaluations.
        Returns (None, None, curve, ...) if no elbow found.
    """
    beta_key = tune_name.split('.', 1)[1]
    binfo = BETA_PARAM_MAP.get(tune_name)
    if binfo is None:
        return None, None, [], baseline_vars, n_frames

    # Determine starting beta from baseline variance
    def _get_var(bvars_dict):
        if bvars_dict is None:
            return None
        vals = [bvars_dict.get(c) for c in binfo['csv_cols']
                if c in bvars_dict and bvars_dict[c] > 0]
        return float(np.mean(vals)) if vals else None

    start_beta = _get_var(baseline_vars)
    if start_beta is None or start_beta <= 0:
        # Fallback to refined_params variance
        var_key = 'var_' + beta_key
        if var_key in refined_params:
            var_val = refined_params[var_key]
            start_beta = float(np.mean(var_val)) if isinstance(var_val, (tuple, list)) else float(var_val)
    if start_beta is None or start_beta <= 0:
        print("    WARNING: no variance estimate for %s, cannot search" % tune_name)
        return None, None, [], baseline_vars, n_frames

    current_n_frames = n_frames
    current_baseline_vars = baseline_vars
    current_sample_spec = sample_spec

    for doubling in range(max_frame_doublings + 1):
        print("\n  --- Adaptive beta search: %s (n_frames=%d, start_beta=%.3g, target=%.2f) ---"
              % (tune_name, current_n_frames, start_beta, target_spread))

        curve = []  # (beta, spread_ratio)
        lo_bracket = None  # (beta, spread) with spread < target
        hi_bracket = None  # (beta, spread) with spread > target

        def _eval_beta(beta_val):
            """Run a single trial at the given beta and return spread_ratio."""
            results = run_tuning_sweep(
                tuning_base, tune_name, tune_name, [beta_val],
                current_sample_spec, tmpdir, use_cuda=use_cuda,
                timeout=timeout, n_parallel=n_parallel,
                n_gpus=n_gpus, procs_per_gpu=procs_per_gpu,
                mpi_comm=mpi_comm, num_devices=num_devices,
                densify=1)
            if not results or not results[0].get('converged') or results[0].get('error'):
                return None, results
            spread = compute_spread_ratio(
                results[0].get('param_vars', {}), tune_name,
                refined_params=refined_params,
                baseline_vars=current_baseline_vars)
            return spread, results

        # Step 1: Evaluate at starting beta
        beta = start_beta
        spread, _ = _eval_beta(beta)
        if spread is not None:
            curve.append((beta, spread))
            print("    eval 1: beta=%.3g  spread=%.3f" % (beta, spread))
            if spread <= target_spread:
                lo_bracket = (beta, spread)
            else:
                hi_bracket = (beta, spread)
        else:
            print("    eval 1: beta=%.3g  FAILED" % beta)

        # Step 2: Search for bracket
        n_evals = 1
        step_factor = 10.0  # move 1 decade at a time

        while n_evals < max_evals:
            if lo_bracket is not None and hi_bracket is not None:
                break  # bracketed!

            n_evals += 1

            if lo_bracket is None:
                # Need a point with spread < target → tighten (decrease beta)
                beta = beta / step_factor
                spread, _ = _eval_beta(beta)
                if spread is not None:
                    curve.append((beta, spread))
                    print("    eval %d: beta=%.3g  spread=%.3f  (tightening)" % (n_evals, beta, spread))
                    if spread <= target_spread:
                        lo_bracket = (beta, spread)
                    elif hi_bracket is None or beta > hi_bracket[0]:
                        hi_bracket = (beta, spread)
                else:
                    print("    eval %d: beta=%.3g  FAILED (tightening)" % (n_evals, beta))
            else:
                # Need a point with spread > target → loosen (increase beta)
                beta = beta * step_factor
                spread, _ = _eval_beta(beta)
                if spread is not None:
                    curve.append((beta, spread))
                    print("    eval %d: beta=%.3g  spread=%.3f  (loosening)" % (n_evals, beta, spread))
                    if spread > target_spread:
                        hi_bracket = (beta, spread)
                    elif lo_bracket is None or beta < lo_bracket[0]:
                        lo_bracket = (beta, spread)
                else:
                    print("    eval %d: beta=%.3g  FAILED (loosening)" % (n_evals, beta))

        # Check if all spreads are > 2.0 (noisy baseline)
        valid_spreads = [s for _, s in curve if s is not None]
        if valid_spreads and min(valid_spreads) > 2.0:
            if doubling < max_frame_doublings:
                new_n_frames = current_n_frames * 2
                print("\n    All spreads > 2.0 (noisy baseline) — "
                      "increasing n_frames: %d → %d" % (current_n_frames, new_n_frames))
                current_n_frames = new_n_frames
                # Re-sample frames
                new_sample_lines = sample_frames(spec_path, current_n_frames,
                                                  n_nearby=n_nearby)
                resample_dir = os.path.join(tmpdir, 'resample_%d' % current_n_frames)
                os.makedirs(resample_dir, exist_ok=True)
                current_sample_spec = write_sample_spec(
                    new_sample_lines, resample_dir)

                # Re-run baseline with more frames
                print("    Re-running baseline with %d frames..." % current_n_frames)
                baseline_phil = tuning_base + '\nuse_restraints = False\n'
                baseline_phil += '\nexp_ref_spec_file = "%s"\n' % current_sample_spec
                baseline_results = run_tuning_sweep(
                    baseline_phil, 'baseline_%d' % current_n_frames, 'betas.G', [1.0],
                    current_sample_spec, tmpdir, use_cuda=use_cuda,
                    timeout=timeout, n_parallel=n_parallel,
                    n_gpus=n_gpus, procs_per_gpu=procs_per_gpu,
                    mpi_comm=mpi_comm, num_devices=num_devices,
                    densify=1)
                if baseline_results and baseline_results[0].get('param_vars'):
                    current_baseline_vars = baseline_results[0]['param_vars']
                    bl = baseline_results[0]
                    print("    New baseline: sigZ_med=%.3f n_rois=%d" %
                          (bl['sigZ_median'], bl['n_rois']))
                    # Recompute start_beta from new baseline
                    new_start = _get_var(current_baseline_vars)
                    if new_start and new_start > 0:
                        start_beta = new_start
                        print("    New start_beta=%.3g" % start_beta)
                else:
                    print("    WARNING: re-run baseline produced no param_vars, stopping")
                    break
                # Reset and retry
                beta = start_beta
                continue
            else:
                print("\n    Max frame doublings reached (%d), using best available"
                      % max_frame_doublings)

        # Step 3: Bisect within bracket
        if lo_bracket is not None and hi_bracket is not None:
            print("\n    Bracketed: [%.3g (spread=%.3f), %.3g (spread=%.3f)]"
                  % (lo_bracket[0], lo_bracket[1], hi_bracket[0], hi_bracket[1]))

            lo_b, hi_b = lo_bracket[0], hi_bracket[0]
            if lo_b > hi_b:
                lo_b, hi_b = hi_b, lo_b

            while n_evals < max_evals:
                # Check convergence: within 0.3 decades
                if abs(np.log10(hi_b) - np.log10(lo_b)) < 0.3:
                    break

                n_evals += 1
                mid_log = (np.log10(lo_b) + np.log10(hi_b)) / 2
                beta = 10.0 ** mid_log

                spread, _ = _eval_beta(beta)
                if spread is not None:
                    curve.append((beta, spread))
                    print("    eval %d: beta=%.3g  spread=%.3f  (bisect)" % (n_evals, beta, spread))
                    if spread <= target_spread:
                        lo_b = beta  # lo_bracket side: spread < target
                    else:
                        hi_b = beta
                else:
                    print("    eval %d: beta=%.3g  FAILED (bisect)" % (n_evals, beta))
                    break

            # Pick the beta closest to target
            best_pair = min(curve, key=lambda p: abs(p[1] - target_spread) if p[1] is not None else 999)
            elbow_beta, elbow_spread = best_pair
            print("\n    Result: beta=%.3g  spread=%.3f  (%d evals, %d frames)"
                  % (elbow_beta, elbow_spread, n_evals, current_n_frames))
            return elbow_beta, elbow_spread, curve, current_baseline_vars, current_n_frames

        elif lo_bracket is not None:
            # All evaluations below target — pick the loosest (highest spread below target)
            below = [(b, s) for b, s in curve if s is not None and s <= target_spread]
            if below:
                best = max(below, key=lambda p: p[1])
                print("\n    No upper bracket found. Best below target: beta=%.3g  spread=%.3f"
                      % (best[0], best[1]))
                return best[0], best[1], curve, current_baseline_vars, current_n_frames

        elif hi_bracket is not None:
            # All evaluations above target — pick the tightest (lowest spread)
            best = min(curve, key=lambda p: p[1] if p[1] is not None else 999)
            print("\n    No lower bracket found. Tightest: beta=%.3g  spread=%.3f"
                  % (best[0], best[1]))
            return best[0], best[1], curve, current_baseline_vars, current_n_frames

        # If we get here without a break/continue/return, we exhausted doublings
        break

    # Fallback: return best point from whatever we collected
    if curve:
        best = min(curve, key=lambda p: abs(p[1] - target_spread) if p[1] is not None else 999)
        print("\n    Fallback: beta=%.3g  spread=%.3f" % (best[0], best[1]))
        return best[0], best[1], curve, current_baseline_vars, current_n_frames

    return None, None, curve, current_baseline_vars, current_n_frames


def write_restraint_level_phils(base_phil, best_sigmas, init_G, output_dir,
                                 use_cholesky, refined_params,
                                 beta_curves, target_levels,
                                 tuned_model=None):
    """Write a folder of phil files at different restraint levels.

    base_phil: the base phil string (with sigmas, phisteps, sigma_r)
    beta_curves: dict mapping beta param name -> list of (beta, spread_ratio)
    target_levels: list of target spread ratios (e.g. [1.0, 0.8, 0.5, 0.3, 0.1])
    """
    os.makedirs(output_dir, exist_ok=True)

    beta_param_names = sorted(beta_curves.keys())
    print("\n  Writing restraint-level phils to: %s" % output_dir)
    print("  Target levels: %s" % ', '.join(['%.0f%%' % (t * 100) for t in target_levels]))
    print("  Parameters: %s" % ', '.join(beta_param_names))

    for level_i, target in enumerate(target_levels):
        if target >= 1.0:
            label = 'free'
        else:
            label = 'spread%02d' % int(target * 100)

        fname = 'level_%02d_%s.phil' % (level_i, label)
        fpath = os.path.join(output_dir, fname)

        # For the free level, write phil WITHOUT restraints
        if target >= 1.0:
            write_optimized_phil(base_phil, best_sigmas, init_G, fpath,
                                 use_cholesky=use_cholesky,
                                 refined_params=refined_params,
                                 tuned_params=tuned_model)
            print("    %s  (no restraints)" % fname)
            continue

        # Interpolate betas for each parameter at this target spread
        level_betas = {}
        details = []
        for bname in beta_param_names:
            pairs = beta_curves[bname]
            beta_val = interpolate_beta_for_spread(pairs, target)
            if beta_val is not None:
                level_betas[bname] = beta_val
                details.append('%s=%.3g' % (bname.split('.')[1], beta_val))
            else:
                details.append('%s=N/A' % bname.split('.')[1])

        # Build tuned_params dict with the interpolated betas
        tuned_params = dict(tuned_model or {})
        tuned_params.update(level_betas)

        write_optimized_phil(base_phil, best_sigmas, init_G, fpath,
                             use_cholesky=use_cholesky,
                             refined_params=refined_params,
                             tuned_params=tuned_params)
        print("    %s  (%s)" % (fname, ', '.join(details)))

    print("  Done: %d phil files written" % len(target_levels))


def run_tuning_sweep(tuning_base_phil, param_name, phil_path, sweep_values,
                      sample_lines, use_cuda=False,
                      n_parallel=1, n_gpus=None, procs_per_gpu=1,
                      mpi_comm=None, num_devices=None, densify=1):
    """Sweep a single phil parameter (betas, phisteps, sigma_r) and score by sigZ.

    Unlike sigma sweeps, the base phil already has all sigmas/fixes/inits set.
    We just override one parameter at a time.

    sample_lines: list of spec file lines (one per image)
    densify: integer multiplier for sweep resolution (1=base, 2=2x finer, etc.)
    Returns: list of dicts with 'value', 'sigZ_median', 'sigZ_mean', etc.
    """
    import numpy as np

    is_int_param = param_name == 'phisteps'
    n_images = len(sample_lines)
    n_target = max(len(sweep_values), len(sweep_values) * densify)

    if is_int_param and param_name == 'phisteps' and n_target > len(sweep_values):
        lo, hi = min(sweep_values), max(sweep_values)
        raw = np.geomspace(max(1, lo), hi, n_target)
        generated = sorted(set(int(round(v)) for v in raw))
        generated = sorted(set(generated) | set(int(v) for v in sweep_values))
        if len(generated) > len(sweep_values):
            sweep_values = generated
            print("  Densified %dx to %d phisteps values (geomspace)"
                  % (densify, len(sweep_values)))
            print("  Values: %s" % ', '.join(str(v) for v in sweep_values))
    elif not is_int_param and n_target > len(sweep_values) and len(sweep_values) >= 2:
        clamp_min = min(sweep_values) * 0.5
        clamp_max = max(sweep_values) * 2.0
        sweep_values = densify_sweep(sweep_values, n_target,
                                      clamp_min=clamp_min, clamp_max=clamp_max)
        seen = set()
        unique_vals = []
        for v in sweep_values:
            key = '%.4e' % v
            if key not in seen:
                seen.add(key)
                unique_vals.append(v)
        sweep_values = unique_vals
        print("  Densified %dx to %d sweep values" % (densify, len(sweep_values)))

    n_images = len(sample_lines)
    print("  %d sweep values x %d images = %d work units" %
          (len(sweep_values), n_images, len(sweep_values) * n_images))

    trial_args = []
    for val in sweep_values:
        if is_int_param:
            val = int(val)

        # Build phil: base + override the swept parameter
        phil_lines = [tuning_base_phil]
        phil_lines.append('save_spot_diagnostics = False')
        if num_devices is not None:
            phil_lines.append('refiner { num_devices = %d }' % num_devices)

        # Set the parameter being swept
        parts = phil_path.split('.')
        if len(parts) == 1:
            phil_lines.append('%s = %s' % (phil_path, val))
        elif len(parts) == 2:
            if parts[1] in ('Nabc',):
                phil_lines.append('%s {\n  %s = %.6g,%.6g,%.6g\n}' % (parts[0], parts[1], val, val, val))
            elif parts[1] in ('cholesky',):
                phil_lines.append('%s {\n  %s = %.6g,%.6g,%.6g,%.6g,%.6g,%.6g\n}' %
                                  (parts[0], parts[1], val, val, val, val, val, val))
            elif parts[1] == 'ucell':
                phil_lines.append('%s {\n'
                    '  ucell_a = %.6g\n  ucell_b = %.6g\n  ucell_c = %.6g\n'
                    '  ucell_alpha = %.6g\n  ucell_beta = %.6g\n  ucell_gamma = %.6g\n'
                    '}' % (parts[0], val, val, val, val, val, val))
            else:
                phil_lines.append('%s {\n  %s = %.6g\n}' % (parts[0], parts[1], val))
        elif len(parts) == 3:
            phil_lines.append('%s {\n  %s {\n    %s = %s\n  }\n}' % (parts[0], parts[1], parts[2], val))

        phil_str = '\n'.join(phil_lines) + '\n'
        label = '%s=%.3g' % (param_name, val)
        trial_args.append((phil_str, use_cuda, val, label))

    # Flatten to per-(value, image) work units
    flat_trials = _flatten_trials(trial_args, sample_lines)
    flat_trials = _assign_gpus(flat_trials, n_gpus, procs_per_gpu)
    raw_results = _run_trials_batch(flat_trials, n_parallel=n_parallel,
                                     mpi_comm=mpi_comm)

    # Aggregate per-image results back to per-value
    agg_results = aggregate_image_results(raw_results, n_images)

    # Collect results
    results = []
    for sigma_val, label, score in agg_results:
        results.append({
            'value': sigma_val,
            'label': label,
            'sigZ_median': score['sigZ_median'],
            'sigZ_mean': score['sigZ_mean'],
            'n_rois': score['n_rois'],
            'n_outliers': score.get('n_outliers', 0),
            'param_vars': score.get('param_vars', {}),
            'converged': score['converged'],
            'time_sec': score['time_sec'],
            'error': score.get('error'),
            'spearman_r': score.get('spearman_r', float('nan')),
        })

    return results


def write_optimized_phil(base_phil, best_sigmas, init_G, output_path,
                         use_cholesky=False, refined_params=None,
                         tuned_params=None):
    """Write the final optimized phil with best sigmas and refined init values."""
    lines = [base_phil, '\n# --- Optimized by hopper_heatup.py ---\n']

    # Ensure roi.fraction is cleared (only used during heatup, not production)
    lines.append('roi.fraction = None')

    # Init values from final refinement (or G estimation)
    if refined_params and 'G' in refined_params:
        lines.append('init.G = %.6e' % refined_params['G'])
        lines.append('init.auto_G = False')
    elif init_G is not None:
        lines.append('init.G = %.6e' % init_G)
        lines.append('init.auto_G = False')

    if refined_params and 'B' in refined_params:
        lines.append('init.B = %.4f' % refined_params['B'])

    if refined_params and 'Nabc' in refined_params:
        na, nb, nc = refined_params['Nabc']
        lines.append('init.Nabc = %.2f,%.2f,%.2f' % (na, nb, nc))

    if refined_params and 'cholesky' in refined_params:
        L = refined_params['cholesky']
        lines.append('init.cholesky = %.6f,%.6f,%.6f,%.6f,%.6f,%.6f' % L)
    # Note: no init.ucell phil param — ucell is initialized from the crystal model

    # Optimized sigmas
    lines.append('\n# Optimized sigmas')
    lines.append('sigmas {')
    for param, val in best_sigmas.items():
        if isinstance(val, list):
            # Per-component sigmas from deep-dive
            lines.append('  %s = %s' % (param, ','.join(['%.6g' % v for v in val])))
        elif param == 'RotXYZ':
            lines.append('  RotXYZ = %.6g,%.6g,%.6g' % (val, val, val))
        elif param == 'Nabc' and not use_cholesky:
            lines.append('  Nabc = %.6g,%.6g,%.6g' % (val, val, val))
        elif param == 'cholesky':
            lines.append('  cholesky = %.6g,%.6g,%.6g,%.6g,%.6g,%.6g' %
                          (val, val, val, val, val, val))
        elif param == 'ucell':
            lines.append('  ucell = %.6g,%.6g,%.6g,%.6g,%.6g,%.6g' %
                          (val, val, val, val, val, val))
        elif param == 'Baniso':
            lines.append('  Baniso = %.6g,%.6g,%.6g,%.6g,%.6g,%.6g' %
                          (val, val, val, val, val, val))
        else:
            lines.append('  %s = %.6g' % (param, val))
    lines.append('}')

    # Tuned parameters (betas, centers, phisteps, sigma_r)
    if tuned_params:
        lines.append('\n# Tuned model/restraint parameters')

        # Enable restraints if any betas are set
        betas = {k: v for k, v in tuned_params.items() if k.startswith('betas.')}

        # Ensure every param that will get a center also has a beta.
        # If the elbow finder skipped a param (no good elbow), fall back to
        # the observed variance from the final refinement (spread ≈ 1).
        # This gives a gentle restraint that catches outliers without
        # fighting the data.  If no variance is available, use a huge beta
        # (effectively unrestrained) as a last resort.
        if betas and refined_params:
            if 'betas.G' not in betas and 'G' in refined_params:
                betas['betas.G'] = refined_params.get('var_G', 1e30)
            if 'betas.B' not in betas and 'B' in refined_params:
                betas['betas.B'] = refined_params.get('var_B', 1e10)
            if 'betas.RotXYZ' not in betas:
                betas['betas.RotXYZ'] = refined_params.get('var_RotXYZ', 1e10)
            if 'betas.cholesky' not in betas and ('cholesky' in refined_params or 'Nabc' in refined_params):
                # Use mean variance across cholesky components
                var_chol = refined_params.get('var_cholesky')
                if var_chol is not None:
                    betas['betas.cholesky'] = max(np.mean(var_chol), 1e-20)
                else:
                    betas['betas.cholesky'] = 1e10
            if 'betas.ucell' not in betas and 'ucell' in refined_params:
                # Use mean of nonzero ucell variances (skip fixed angles)
                var_uc = refined_params.get('var_ucell')
                if var_uc is not None:
                    nonzero = [v for v in var_uc if v > 0]
                    betas['betas.ucell'] = max(np.mean(nonzero), 1e-20) if nonzero else 1e10
                else:
                    betas['betas.ucell'] = 1e10

        if betas:
            lines.append('use_restraints = True')
        if betas:
            lines.append('betas {')
            for k, v in sorted(betas.items()):
                param_name = k.split('.', 1)[1]
                if param_name == 'Nabc':
                    lines.append('  Nabc = %.6g,%.6g,%.6g' % (v, v, v))
                elif param_name == 'cholesky':
                    lines.append('  cholesky = %.6g,%.6g,%.6g,%.6g,%.6g,%.6g' %
                                  (v, v, v, v, v, v))
                elif param_name == 'ucell':
                    for uc_name in ['ucell_a', 'ucell_b', 'ucell_c',
                                    'ucell_alpha', 'ucell_beta', 'ucell_gamma']:
                        lines.append('  %s = %.6g' % (uc_name, v))
                else:
                    lines.append('  %s = %.6g' % (param_name, v))
            lines.append('}')

        # Centers (restraint targets) from refined params — only when betas are set,
        # otherwise hopper_utils raises RuntimeError (center without beta)
        if betas and refined_params:
            lines.append('centers {')
            if 'G' in refined_params:
                lines.append('  G = %.6e' % refined_params['G'])
            if 'B' in refined_params:
                lines.append('  B = %.4f' % refined_params['B'])
            if 'cholesky' in refined_params and use_cholesky:
                L = refined_params['cholesky']
                lines.append('  cholesky = %.4f,%.4f,%.4f,%.4f,%.4f,%.4f' % L)
            elif 'Nabc' in refined_params:
                na, nb, nc = refined_params['Nabc']
                if use_cholesky:
                    import numpy as np
                    # Fallback: convert Nabc centers to diagonal Cholesky
                    L11 = np.sqrt(na)
                    L22 = np.sqrt(nb)
                    L33 = np.sqrt(nc)
                    lines.append('  cholesky = %.4f,0,%.4f,0,0,%.4f' % (L11, L22, L33))
                else:
                    lines.append('  Nabc = %.2f,%.2f,%.2f' % (na, nb, nc))
            lines.append('  RotXYZ = 0,0,0')
            if 'ucell' in refined_params:
                uc = refined_params['ucell']
                _uc_names = ['ucell_a', 'ucell_b', 'ucell_c',
                             'ucell_alpha', 'ucell_beta', 'ucell_gamma']
                for name, val in zip(_uc_names, uc):
                    lines.append('  %s = %.4f' % (name, val))
            lines.append('}')

        # Model params
        if 'phisteps' in tuned_params:
            lines.append('simulator.gonio.phi_steps = %d' % int(tuned_params['phisteps']))
        if 'sigma_r' in tuned_params:
            lines.append('refiner.sigma_r = %.4g' % tuned_params['sigma_r'])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description="Automated hopper sigma tuning / warm-up",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Runs hopper on a small frame sample, sweeping sigma values for each
parameter to find optimal refinement scaling. Each stage builds on
the previous, so G is tuned first, then RotXYZ, then Nabc, then B.

Examples:
  # Basic usage with an existing phil:
  python hopper_heatup.py base.phil --n-frames 3

  # Override spec file:
  python hopper_heatup.py base.phil --spec exp_ref_spec.txt --n-frames 5

  # Only tune G and RotXYZ:
  python hopper_heatup.py base.phil --stages G,RotXYZ

  # Custom sweep values for G:
  python hopper_heatup.py base.phil --sigmas-G "0.01,0.1,1,5,10"

  # With GPU:
  python hopper_heatup.py base.phil --cuda
""")
    parser.add_argument("phil", help="Base phil file to start from")
    parser.add_argument("--spec", default=None,
                        help="exp_ref_spec.txt (overrides phil if given)")
    parser.add_argument("--n-frames", type=int, default=3,
                        help="Number of phi positions to sample (default: 3)")
    parser.add_argument("--n-nearby", type=int, default=1,
                        help="Neighboring frames on each side per phi position "
                             "(default: 1, giving triplets). Total frames per "
                             "position = 2*n_nearby+1. Use 0 for single frames.")
    parser.add_argument("--stages", default=None,
                        help="Comma-separated stages to run (default: G,RotXYZ,Nabc,B)")
    parser.add_argument("--n-shuffles", type=int, default=1,
                        help="Number of random stage orderings to try (default: 1). "
                             "G is always first; remaining stages are shuffled. "
                             "Best overall sigma set wins.")
    parser.add_argument("--max-calls", type=int, default=50,
                        help="Max L-BFGS-B iterations per trial (default: 50)")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Timeout per trial in seconds (default: 1800)")
    parser.add_argument("--roi-fraction", type=float, default=None,
                        help="Use only this fraction of ROIs per shot (0-1). "
                             "E.g. 0.1 uses ~10%% of spots, 5-10x speedup. "
                             "Default: None (use all ROIs)")
    parser.add_argument("--cuda", action="store_true",
                        help="Set DIFFBRAGG_USE_CUDA=1")
    parser.add_argument("--tmpdir", default=None,
                        help="Temp directory for trials (default: auto)")
    parser.add_argument("--keep-trials", action="store_true",
                        help="Keep trial output directories (default: clean up)")
    parser.add_argument("-o", "--output", default="optimized.phil",
                        help="Output optimized phil (default: optimized.phil)")
    parser.add_argument("--no-final-refine", action="store_true",
                        help="Skip final refinement pass (default: run final "
                             "refinement to get init values for optimized phil)")
    parser.add_argument("--final-refine-max-calls", type=int, default=500,
                        help="Max L-BFGS-B iterations for final refinement (default: 500)")
    parser.add_argument("--skip-to-tuning", action="store_true",
                        help="Skip sigma sweeps, deep-dive, and final refinement. "
                             "Load state from a previous run's checkpoint JSON "
                             "(written alongside the optimized phil). Jump straight "
                             "to Phase 2 restraint tuning. Requires --tune-restraints.")

    parser.add_argument("--trials-per-worker", type=int, default=None,
                        help="DEPRECATED: use --densify instead.")
    parser.add_argument("--densify", type=int, default=1,
                        help="Sweep density multiplier (default: 1). "
                             "Interpolates between base sweep values for finer "
                             "resolution. E.g. --densify 3 gives 3x more sweep "
                             "points. --densify 5 gives 5x finer grid.")

    # Custom sigma sweep values
    parser.add_argument("--sigmas-G", default=None,
                        help="Comma-sep sigma values for G sweep")
    parser.add_argument("--sigmas-RotXYZ", default=None,
                        help="Comma-sep sigma values for RotXYZ sweep")
    parser.add_argument("--sigmas-Nabc", default=None,
                        help="Comma-sep sigma values for Nabc sweep")
    parser.add_argument("--sigmas-B", default=None,
                        help="Comma-sep sigma values for B sweep")
    parser.add_argument("--sigmas-ucell", default=None,
                        help="Comma-sep sigma values for ucell sweep")

    # G estimation
    parser.add_argument("--skip-g-estimation", action="store_true",
                        help="Skip the G estimation phase")
    parser.add_argument("--init-G", type=float, default=None,
                        help="Set init.G directly (skips G estimation). "
                             "E.g. --init-G 1e9")

    # Deep-dive mode
    parser.add_argument("--deep-dive", action="store_true",
                        help="After standard stages, do per-component sigma "
                             "sweeps for cholesky/Nabc, RotXYZ, ucell, B, Baniso")
    parser.add_argument("--deep-dive-params", default=None,
                        help="Comma-sep params to deep-dive (default: all applicable). "
                             "Options: cholesky,Nabc,RotXYZ,ucell,B,Baniso")
    parser.add_argument("--deep-dive-multipliers", default=None,
                        help="Comma-sep multipliers of scalar best to sweep "
                             "(default: 0.1,0.3,0.5,0.7,1,1.5,2,3,5,10)")
    parser.add_argument("--deep-dive-max-calls", type=int, default=500,
                        help="Max L-BFGS-B iterations for deep-dive trials "
                             "(default: 500, higher than standard stages)")

    # Model tuning (phisteps + sigma_r, before deep-dive)
    parser.add_argument("--tune-model", action="store_true", default=True,
                        help="Sweep phisteps and sigma_r after initial sigma scan, "
                             "then re-sweep sigmas with the tuned model (default: True). "
                             "Use --no-tune-model to disable.")
    parser.add_argument("--no-tune-model", action="store_true",
                        help="Disable automatic model tuning (sigma_r + phisteps). "
                             "Sigmas are swept once with the inherited model params.")
    parser.add_argument("--tune-model-max-calls", type=int, default=None,
                        help="Max L-BFGS-B iterations for model tuning trials "
                             "(default: same as --max-calls)")

    # Tuning phase (restraints + model params)
    parser.add_argument("--tune-restraints", action="store_true",
                        help="After sigma optimization, sweep betas (restraint "
                             "tightness), phisteps, and sigma_r. Requires final "
                             "refinement (provides centers from refined params).")
    parser.add_argument("--tune-params", default=None,
                        help="Comma-sep list of params to tune (default: all). "
                             "Options: geometry,betas.G,betas.B,betas.Nabc,betas.cholesky,"
                             "betas.RotXYZ,phisteps,sigma_r")
    parser.add_argument("--tune-max-calls", type=int, default=500,
                        help="Max L-BFGS-B iterations for tuning trials (default: 500)")
    parser.add_argument("--spread-roi-fraction", type=float, default=None,
                        help="ROI fraction for spread-ratio beta sweeps (default: "
                             "None = use all ROIs for accurate variance estimates). "
                             "Set to match --roi-fraction if you want speed over accuracy.")
    parser.add_argument("--tune-n-frames", type=int, default=None,
                        help="Number of phi positions for model tuning (phisteps, "
                             "sigma_r) and beta sweep trials (default: same as "
                             "--n-frames). More frames give more stable results "
                             "for sigma_r selection and spread-ratio curves.")

    # Geometry refinement (Phase 2)
    parser.add_argument("--tune-geometry", action="store_true",
                        help="Refine detector geometry in Phase 2 before beta sweeps. "
                             "Uses tune_n_frames images. Requires --tune-restraints.")
    parser.add_argument("--tune-goniometer", action="store_true",
                        help="Include goniometer axis in geometry refinement. "
                             "Requires --tune-geometry. Uses spherical parameterization.")
    parser.add_argument("--geometry-method", default="lbfgsb",
                        choices=["lbfgsb", "nelder"],
                        help="Optimization method for geometry refinement (default: lbfgsb). "
                             "Use 'nelder' if C++ derivatives not available.")
    parser.add_argument("--geometry-max-calls", type=int, default=100,
                        help="Max iterations for geometry refinement (default: 100).")
    parser.add_argument("--panel-group-file", default=None,
                        help="Panel grouping file for multi-panel detectors. "
                             "See refiners/geometry.py for format.")

    # Parallelism
    parser.add_argument("--n-parallel", type=int, default=1,
                        help="Number of parallel hopper trials (default: 1). "
                             "Applies to both standard sweeps and deep-dive.")
    parser.add_argument("--n-gpus", type=int, default=None,
                        help="Number of GPUs to distribute trials across. "
                             "Each trial gets CUDA_VISIBLE_DEVICES=<gpu_id> "
                             "via round-robin. Implies --cuda.")
    parser.add_argument("--procs-per-gpu", type=int, default=1,
                        help="Max concurrent trials per GPU (default: 1). "
                             "Total parallelism = n_gpus * procs_per_gpu.")
    parser.add_argument("--mpi", action="store_true",
                        help="Use MPI for trial distribution (mpirun -n N). "
                             "Rank 0 coordinates, ranks 1..N-1 execute trials.")
    parser.add_argument("--logfile", default=None,
                        help="Log file path. All stdout/stderr output (including "
                             "errors and tracebacks) is duplicated to this file. "
                             "Default: <output_basename>_heatup.log")

    args = parser.parse_args()
    if args.no_tune_model:
        args.tune_model = False

    # Backward compat: --trials-per-worker is deprecated
    if args.trials_per_worker is not None:
        print("WARNING: --trials-per-worker is deprecated, use --densify instead. Ignoring.")
        args.trials_per_worker = None

    # Handle MPI (before logging — workers exit here)
    mpi_comm = None
    if args.mpi:
        if not HAS_MPI:
            print("ERROR: --mpi requires mpi4py. Install with: pip install mpi4py")
            sys.exit(1)
        mpi_comm = MPI.COMM_WORLD
        rank = mpi_comm.Get_rank()
        size = mpi_comm.Get_size()
        if rank > 0:
            # Workers enter persistent loop, process work until shutdown
            mpi_worker_loop(mpi_comm)
            sys.exit(0)
        # Rank 0 continues as coordinator
        n_workers = size - 1
        args.n_parallel = size  # MPI size determines parallelism
        # Auto-compute procs_per_gpu from MPI workers / n_gpus
        if args.n_gpus is not None and args.n_gpus > 0 and args.procs_per_gpu == 1:
            args.procs_per_gpu = max(1, n_workers // args.n_gpus)
        print("MPI mode: rank 0 (coordinator) + %d workers" % n_workers)

    # --- Set up log file (tee stdout+stderr to file) ---
    # Only rank 0 reaches here (workers exited above)
    if args.logfile is None:
        logfile_path = args.output.replace('.phil', '_heatup.log')
    else:
        logfile_path = args.logfile
    logdir = os.path.dirname(logfile_path)
    if logdir:
        os.makedirs(logdir, exist_ok=True)
    _logfile = open(logfile_path, 'w')

    class _Tee(object):
        """Duplicate writes to both a stream and a log file, flushing immediately."""
        def __init__(self, stream, logfile):
            self._stream = stream
            self._logfile = logfile
        def write(self, msg):
            self._stream.write(msg)
            self._stream.flush()
            try:
                self._logfile.write(msg)
                self._logfile.flush()
            except (ValueError, OSError):
                pass
        def flush(self):
            self._stream.flush()
            try:
                self._logfile.flush()
            except (ValueError, OSError):
                pass
        def fileno(self):
            return self._stream.fileno()
        def isatty(self):
            return self._stream.isatty()

    sys.stdout = _Tee(sys.__stdout__, _logfile)
    sys.stderr = _Tee(sys.__stderr__, _logfile)
    print("hopper_heatup log: %s" % os.path.abspath(logfile_path))
    print("Command: %s" % ' '.join(sys.argv))

    # Handle GPU parallelism
    # When distributing across GPUs, force 1 device per trial
    # (overrides num_devices in base phil)
    trial_num_devices = None
    if args.n_gpus is not None and args.n_gpus > 0:
        args.cuda = True  # --n-gpus implies --cuda
        trial_num_devices = 1  # each trial uses 1 GPU
        if not args.mpi:
            # Thread mode: compute parallelism from GPU count
            total_parallel = args.n_gpus * args.procs_per_gpu
            if args.n_parallel <= 1:
                args.n_parallel = total_parallel
        print("GPU mode: %d GPUs x %d procs/GPU = %d parallel trials (num_devices=1 per trial)" %
              (args.n_gpus, args.procs_per_gpu, args.n_parallel))

    # Parse stages
    if args.stages:
        stages = [s.strip() for s in args.stages.split(',')]
    else:
        stages = list(ALL_STAGES)

    # Parse custom sigma values
    sigma_sweeps = dict(DEFAULT_SIGMAS)
    for stage in stages:
        custom = getattr(args, 'sigmas_%s' % stage, None)
        if custom:
            sigma_sweeps[stage] = [float(x) for x in custom.split(',')]

    # Read base phil
    base_phil = parse_phil_file(args.phil)

    # Inject ROI subsampling if requested
    if args.roi_fraction is not None:
        base_phil += '\nroi.fraction = %.4f\n' % args.roi_fraction
        print("ROI subsampling: %.0f%% of spots per shot" % (args.roi_fraction * 100))

    # Detect if using cholesky from the phil
    use_cholesky = 'use_cholesky_Nabc' in base_phil and 'True' in base_phil.split('use_cholesky_Nabc')[1].split('\n')[0]

    # Find spec file
    spec_path = args.spec
    if spec_path is None:
        # Try to extract from phil
        for line in base_phil.split('\n'):
            if 'exp_ref_spec_file' in line and '=' in line and '#' not in line.split('=')[0]:
                spec_path = line.split('=', 1)[1].strip().strip('"').strip("'")
                break

    if spec_path is None:
        print("ERROR: No spec file found. Use --spec or include exp_ref_spec_file in phil.")
        sys.exit(1)

    if not os.path.exists(spec_path):
        print("ERROR: Spec file not found: %s" % spec_path)
        sys.exit(1)

    # Sample frames
    sample_lines = sample_frames(spec_path, args.n_frames, n_nearby=args.n_nearby)
    n_per_pos = 2 * args.n_nearby + 1
    print("Sampled %d phi positions x %d nearby = %d frames from %s" %
          (args.n_frames, n_per_pos, len(sample_lines), spec_path))
    for i, line in enumerate(sample_lines):
        # Show just the expt filename
        parts = line.split()
        expt = os.path.basename(parts[0]) if parts else line
        print("  Frame %d: %s" % (i, expt))

    # Create temp directory
    if args.tmpdir:
        tmpdir = args.tmpdir
        os.makedirs(tmpdir, exist_ok=True)
    else:
        tmpdir = tempfile.mkdtemp(prefix='hopper_heatup_')
    print("\nWorking directory: %s" % tmpdir)

    # Write sample spec
    sample_spec = write_sample_spec(sample_lines, tmpdir)

    # Track best sigmas
    best_sigmas = {}
    init_G = None
    tuned_model = {}
    refined_params = None

    # --- Skip to tuning: load checkpoint from previous run ---
    if args.skip_to_tuning:
        import json
        checkpoint_path = args.output.replace('.phil', '_checkpoint.json')
        if not os.path.exists(checkpoint_path):
            print("ERROR: --skip-to-tuning requires checkpoint file: %s" % checkpoint_path)
            print("  Run without --skip-to-tuning first to generate it.")
            sys.exit(1)
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        best_sigmas = checkpoint['best_sigmas']
        # Convert list values back to lists (JSON preserves them)
        for k, v in best_sigmas.items():
            if isinstance(v, list):
                best_sigmas[k] = v  # keep as list (per-component sigmas)
        init_G = checkpoint.get('init_G')
        tuned_model = checkpoint.get('tuned_model', {})
        rp = checkpoint.get('refined_params')
        refined_params = None
        if rp:
            refined_params = {}
            for k, v in rp.items():
                if isinstance(v, list):
                    refined_params[k] = tuple(v)
                else:
                    refined_params[k] = v

        print("\n" + "=" * 60)
        print("SKIP TO TUNING: loaded checkpoint from %s" % checkpoint_path)
        print("=" * 60)
        print("  best_sigmas: %s" % best_sigmas)
        if init_G is not None:
            print("  init_G: %.4e" % init_G)
        if refined_params:
            print("  refined_params keys: %s" % list(refined_params.keys()))
        if tuned_model:
            print("  tuned_model: %s" % tuned_model)

        # Apply tuned model params to base_phil (same as --tune-model does)
        if tuned_model:
            model_lines = ['\n# --- Tuned model parameters (from checkpoint) ---']
            if 'phisteps' in tuned_model:
                model_lines.append('simulator.gonio.phi_steps = %d' %
                                   int(tuned_model['phisteps']))
            if 'sigma_r' in tuned_model:
                model_lines.append('refiner.sigma_r = %.4g' %
                                   tuned_model['sigma_r'])
            base_phil = base_phil + '\n'.join(model_lines) + '\n'

        # Jump straight to Phase 2 (skip G estimation, sigma sweeps,
        # model tuning, deep-dive, and final refinement)

    # --- Phase 0: G estimation ---
    if args.skip_to_tuning:
        pass  # skipped — state loaded from checkpoint
    elif args.init_G is not None:
        init_G = args.init_G
        base_phil += '\ninit.G = %.6e\ninit.auto_G = False\n' % init_G
        print("\n  Using --init-G = %.4e (skipping G estimation)" % init_G)
    elif not args.skip_g_estimation and 'G' in stages:
        print("\n" + "=" * 60)
        print("PHASE 0: G ESTIMATION (parallel search across orders of magnitude)")
        print("=" * 60)
        G_est = estimate_G_parallel(
            base_phil, sample_lines, use_cuda=args.cuda,
            n_parallel=args.n_parallel,
            n_gpus=args.n_gpus, procs_per_gpu=args.procs_per_gpu,
            mpi_comm=mpi_comm, num_devices=trial_num_devices)
        if G_est is not None and G_est > 0:
            init_G = G_est
            # Update base phil with estimated G
            base_phil += '\ninit.G = %.6e\ninit.auto_G = False\n' % init_G

    # --- Sigma sweep stages ---
    # Helper to run one ordering of stages
    def run_sweep_ordering(stage_order, sigma_sweeps, base_phil, best_sigmas_init,
                           label="", densify=1):
        """Run a complete sweep through stages in the given order.
        Returns (best_sigmas, all_results, final_sigZ).
        """
        cur_sigmas = dict(best_sigmas_init)
        cur_results = {}

        # When both G and B are in stage_order, replace them with a combined
        # G+B sweep. Sweeping G alone is useless (1D optimizer always converges
        # to the same G regardless of step size); the sigma ratio between G and
        # B only matters when both are free simultaneously.
        gb_done = False
        effective_stages = []
        if 'G' in stage_order and 'B' in stage_order:
            for s in stage_order:
                if s == 'G':
                    effective_stages.append('G+B')
                elif s == 'B':
                    continue  # already covered by G+B
                else:
                    effective_stages.append(s)
        else:
            effective_stages = list(stage_order)

        for stage in effective_stages:

            # --- Combined G+B sigma sweep ---
            if stage == 'G+B':
                print("\n" + "=" * 60)
                print("%sSTAGE: G+B combinatorial sigma sweep" % label)
                print("=" * 60)

                sigma_G_vals = sigma_sweeps.get('G', DEFAULT_SIGMAS['G'])
                sigma_B_vals = sigma_sweeps.get('B', DEFAULT_SIGMAS['B'])

                print("  Parameters: sigmas.G x sigmas.B (both free)")
                print("  sigma_G values (%d): %s" % (
                    len(sigma_G_vals),
                    ', '.join(['%.2e' % v for v in sigma_G_vals])))
                print("  sigma_B values (%d): %s" % (
                    len(sigma_B_vals),
                    ', '.join(['%.2e' % v for v in sigma_B_vals])))
                print("  Total trials: %d" % (len(sigma_G_vals) * len(sigma_B_vals)))
                print("  Free params: %s" % ', '.join(
                    [k for k, v in STAGE_FIXES['G+B'].items() if not v]))
                print("  Best sigmas so far: %s" % cur_sigmas)
                print()

                results = run_GB_sigma_sweep(
                    base_phil, sigma_G_vals, sigma_B_vals, cur_sigmas,
                    sample_lines, use_cholesky=use_cholesky,
                    use_cuda=args.cuda, max_calls=args.max_calls,
                    n_parallel=args.n_parallel,
                    n_gpus=args.n_gpus, procs_per_gpu=args.procs_per_gpu,
                    mpi_comm=mpi_comm, num_devices=trial_num_devices)

                cur_results['G+B'] = results
                print_GB_summary(results)

                # Find best (sigma_G, sigma_B) pair
                valid = [r for r in results
                         if r['converged'] and r.get('error') is None
                         and r['sigZ_median'] < 1e6]
                if valid:
                    best_r = min(valid, key=_score_trial)
                    cur_sigmas['G'] = best_r['sigma_G']
                    cur_sigmas['B'] = best_r['sigma_B']
                    print("\n  Best sigmas.G = %.2e, sigmas.B = %.2e" %
                          (best_r['sigma_G'], best_r['sigma_B']))
                else:
                    print("\n  WARNING: No valid results for G+B sweep!")
                    cur_sigmas['G'] = 0.01
                    cur_sigmas['B'] = 1.0
                continue

            # --- Standard single-parameter sigma sweep ---
            print("\n" + "=" * 60)
            print("%sSTAGE: %s sigma sweep" % (label, stage))
            print("=" * 60)

            sweep_key = stage
            if stage == 'Nabc' and use_cholesky:
                sweep_key = 'cholesky'

            sweep_values = sigma_sweeps.get(sweep_key, DEFAULT_SIGMAS.get(sweep_key, [1.0]))

            print("  Parameter: sigmas.%s" % sweep_key)
            print("  Sweep values: %s" % ', '.join(['%.2e' % v for v in sweep_values]))
            print("  Free params: %s" % ', '.join(
                [k for k, v in STAGE_FIXES[stage].items() if not v]))
            print("  Best sigmas so far: %s" % cur_sigmas)
            print()

            results = run_sigma_sweep(
                base_phil, stage, sweep_values, cur_sigmas, sample_lines,
                use_cholesky=use_cholesky, use_cuda=args.cuda,
                max_calls=args.max_calls,
                n_parallel=args.n_parallel, n_gpus=args.n_gpus,
                procs_per_gpu=args.procs_per_gpu, mpi_comm=mpi_comm,
                num_devices=trial_num_devices,
                densify=densify)

            cur_results[stage] = results
            print_stage_summary(stage, results)

            best_val = find_best_sigma(results)
            if best_val is not None:
                if stage == 'Nabc' and use_cholesky:
                    cur_sigmas['cholesky'] = best_val
                else:
                    cur_sigmas[stage] = best_val
                print("\n  Best sigmas.%s = %.2e" % (sweep_key, best_val))
            else:
                print("\n  WARNING: No valid results for stage %s!" % stage)
                if stage == 'Nabc' and use_cholesky:
                    cur_sigmas['cholesky'] = 1.0
                else:
                    cur_sigmas[stage] = 1.0

        # Compute final score: run one trial with all best sigmas to get sigZ
        # Use the last stage's best result as the score
        last_stage = effective_stages[-1]
        last_results = cur_results[last_stage]
        valid = [r for r in last_results if r['converged'] and r.get('error') is None
                 and r['sigZ_median'] < 1e6]
        final_sigZ = min(r['sigZ_median'] for r in valid) if valid else float('inf')

        return cur_sigmas, cur_results, final_sigZ

    import random

    all_results = {}
    deep_dive_results = {}

    if args.skip_to_tuning:
        print("\n  SKIP TO TUNING: jumping to Phase 2 restraint tuning")

    if not args.skip_to_tuning:
        n_shuffles = args.n_shuffles
        # G is always first if present; shuffle the rest.
        # When both G and B are in stages, they form a combined G+B sweep
        # (run_sweep_ordering handles this), so B is excluded from shuffling.
        if 'G' in stages:
            fixed_stages = ['G']
            exclude = {'G'}
            if 'B' in stages:
                fixed_stages.append('B')  # included via G+B, not shuffled
                exclude.add('B')
            shuffleable = [s for s in stages if s not in exclude]
        else:
            fixed_stages = []
            shuffleable = list(stages)

        if n_shuffles <= 1:
            # Original behavior: single ordering
            best_sigmas, all_results, _ = run_sweep_ordering(
                stages, sigma_sweeps, base_phil, best_sigmas, label="",
                densify=args.densify)
        else:
            # Multiple shuffled orderings
            best_overall_sigmas = None
            best_overall_sigZ = float('inf')
            best_overall_results = {}
            best_ordering = None

            for shuffle_i in range(n_shuffles):
                ordering = list(fixed_stages) + list(shuffleable)
                if shuffle_i > 0:
                    random.shuffle(shuffleable)
                    ordering = list(fixed_stages) + list(shuffleable)

                label = "[shuffle %d/%d: %s] " % (shuffle_i + 1, n_shuffles,
                                                    '→'.join(ordering))
                print("\n" + "#" * 60)
                print("SHUFFLE %d/%d: %s" % (shuffle_i + 1, n_shuffles, ' → '.join(ordering)))
                print("#" * 60)

                shuf_sigmas, shuf_results, shuf_sigZ = run_sweep_ordering(
                    ordering, sigma_sweeps, base_phil, {}, label=label,
                    densify=args.densify)

                print("\n  Shuffle %d result: sigZ_median = %.3f  sigmas = %s" %
                      (shuffle_i + 1, shuf_sigZ, shuf_sigmas))

                if shuf_sigZ < best_overall_sigZ:
                    best_overall_sigZ = shuf_sigZ
                    best_overall_sigmas = dict(shuf_sigmas)
                    best_overall_results = dict(shuf_results)
                    best_ordering = ordering

            print("\n" + "#" * 60)
            print("BEST ORDERING: %s (sigZ_median = %.3f)" %
                  (' → '.join(best_ordering), best_overall_sigZ))
            print("#" * 60)

            best_sigmas = best_overall_sigmas
            all_results = best_overall_results

    # --- Prepare tune_sample_lines (more frames for model + beta tuning) ---
    tune_sample_lines = sample_lines
    if not args.skip_to_tuning:
        tune_n_frames = args.tune_n_frames or args.n_frames
        if args.tune_n_frames and args.tune_n_frames > args.n_frames:
            print("\n  Resampling for tuning phases: %d phi positions "
                  "(vs %d for sigma sweeps)" % (tune_n_frames, args.n_frames))
            tune_sample_lines = sample_frames(spec_path, tune_n_frames,
                                               n_nearby=args.n_nearby)
            total_frames = len(tune_sample_lines)
            print("  Tune sample: %d frames (%d positions x %d nearby)"
                  % (total_frames, tune_n_frames, 2 * args.n_nearby + 1))

    # --- Model parameter tuning (phisteps + sigma_r) ---
    if not args.skip_to_tuning:
        tuned_model = {}
    if args.tune_model and not args.skip_to_tuning:
        print("\n" + "=" * 60)
        print("MODEL TUNING: phisteps + sigma_r (before deep-dive)")
        print("=" * 60)

        model_max_calls = args.tune_model_max_calls or args.max_calls
        model_base = build_model_sweep_phil(
            base_phil, best_sigmas, init_G=init_G,
            use_cholesky=use_cholesky, max_calls=model_max_calls,
            refine_ucell='ucell' in stages)

        for model_param in ['phisteps', 'sigma_r']:
            sweep_vals = list(DEFAULT_PHISTEPS if model_param == 'phisteps'
                              else DEFAULT_SIGMA_R)
            phil_path = TUNE_PARAMS[model_param]

            print("\n  --- %s ---" % model_param)
            print("  Sweep values: %s" % ', '.join(
                str(int(v)) if model_param == 'phisteps' else '%.3g' % v
                for v in sweep_vals))

            results = run_tuning_sweep(
                model_base, model_param, phil_path, sweep_vals,
                tune_sample_lines, use_cuda=args.cuda,
                n_parallel=args.n_parallel,
                n_gpus=args.n_gpus, procs_per_gpu=args.procs_per_gpu,
                mpi_comm=mpi_comm, num_devices=trial_num_devices,
                densify=args.densify)

            # Print and find best
            # Both phisteps and sigma_r use Spearman R (model-data shape agreement)
            # sigZ is unreliable for model params: phisteps landscape is often flat
            # in sigZ (especially small delta_phi), and sigma_r directly scales sigZ.
            # Fall back to sigZ if Spearman data not available.
            has_spearman = any(np.isfinite(r.get('spearman_r', float('nan')))
                               for r in results if r['converged'] and r['error'] is None)
            use_spearman = has_spearman
            if not has_spearman:
                print("  WARNING: Spearman R not available (rebuild hopper_utils?)")
                print("  Falling back to sigZ scoring for %s" % model_param)
            if use_spearman:
                cost_note = " + cost tiebreaker" if model_param == 'phisteps' else ""
                print("  Scoring: Spearman rank correlation%s" % cost_note)
                print("\n  %-12s  %10s  %10s  %8s  %6s  %5s  %5s  %s" %
                      ('value', 'sigZ_med', 'sigZ_mean', 'spR_med', 'n_roi', 'n_out', 'time', 'status'))
                print("  " + "-" * 78)
            else:
                print("\n  %-12s  %10s  %10s  %6s  %5s  %5s  %s" %
                      ('value', 'sigZ_med', 'sigZ_mean', 'n_roi', 'n_out', 'time', 'status'))
                print("  " + "-" * 68)
            valid_results = [r for r in results
                             if r['converged'] and r['error'] is None
                             and r['sigZ_median'] < 1e6]
            base_scorer = _score_trial_spearman if use_spearman else _score_trial
            if model_param == 'phisteps':
                # For phisteps, prefer speed when scores are similar:
                # add a tiny cost penalty proportional to value so that
                # when Spearman R is flat, the cheapest option wins.
                max_val = max(r['value'] for r in valid_results) if valid_results else 1
                def scorer(r, _base=base_scorer, _max=max_val):
                    return _base(r) + 1e-6 * r['value'] / _max
            else:
                scorer = base_scorer
            best_r = min(valid_results, key=scorer) if valid_results else None
            best_val = best_r['value'] if best_r else None
            for r in results:
                status = 'OK' if r['converged'] and r['error'] is None else 'FAIL'
                sigz_med = '%.3f' % r['sigZ_median'] if r['sigZ_median'] < 1e6 else 'inf'
                sigz_mean = '%.3f' % r['sigZ_mean'] if r['sigZ_mean'] < 1e6 else 'inf'
                marker = ' <-- best' if r is best_r else ''
                if use_spearman:
                    spr = '%.4f' % r.get('spearman_r', float('nan')) if np.isfinite(r.get('spearman_r', float('nan'))) else 'N/A'
                    print("  %-12.3g  %10s  %10s  %8s  %6d  %5d  %5.1f  %s%s" %
                          (r['value'], sigz_med, sigz_mean, spr, r['n_rois'],
                           r.get('n_outliers', 0), r['time_sec'],
                           status, marker))
                    continue
                print("  %-12.3g  %10s  %10s  %6d  %5d  %5.1f  %s%s" %
                      (r['value'], sigz_med, sigz_mean, r['n_rois'],
                       r.get('n_outliers', 0), r['time_sec'],
                       status, marker))

            if best_val is not None:
                tuned_model[model_param] = best_val
                print("\n  Best %s = %s" % (model_param,
                      str(int(best_val)) if model_param == 'phisteps'
                      else '%.4g' % best_val))
            else:
                print("\n  WARNING: No valid results for %s!" % model_param)

        # --- Geometry refinement (if enabled) ---
        if args.tune_geometry:
            print("\n" + "=" * 60)
            print("GEOMETRY REFINEMENT")
            print("=" * 60)
            print("  Images: %d" % len(tune_sample_lines))
            print("  Method: %s" % args.geometry_method)
            print("  Refine goniometer: %s" % args.tune_goniometer)
            print()

            optimized_detector_path = run_geometry_refinement_tuning(
                base_phil=model_base if tuned_model else base_phil,
                spec_lines=tune_sample_lines,
                refined_params={},  # No centers needed for geometry
                refine_goniometer=args.tune_goniometer,
                method=args.geometry_method,
                max_calls=args.geometry_max_calls,
                panel_group_file=args.panel_group_file,
                use_cuda=args.cuda,
                outdir=os.path.join(tmpdir, "geometry"),
                mpi_comm=mpi_comm
            )

            if optimized_detector_path is not None:
                # Update base_phil with optimized detector
                base_phil += '\ngeometry.input_expt = %s\n' % optimized_detector_path
                tuned_model['geometry'] = optimized_detector_path
                print("  Geometry refinement complete!")
                print("  Detector: %s" % optimized_detector_path)
            else:
                print("  WARNING: Geometry refinement failed!")

        if tuned_model:
            # Append tuned model params to base_phil so all subsequent sweeps use them
            model_lines = ['\n# --- Tuned model parameters (from --tune-model) ---']
            if 'phisteps' in tuned_model:
                model_lines.append('simulator.gonio.phi_steps = %d' %
                                   int(tuned_model['phisteps']))
            if 'sigma_r' in tuned_model:
                model_lines.append('refiner.sigma_r = %.4g' %
                                   tuned_model['sigma_r'])
            base_phil = base_phil + '\n'.join(model_lines) + '\n'
            print("\n  Updated base phil with tuned model params")
            print("  Tuned: %s" % ', '.join(
                '%s=%s' % (k, str(int(v)) if k == 'phisteps' else '%.4g' % v)
                for k, v in tuned_model.items()))

            # Re-sweep sigmas with tuned model
            print("\n" + "-" * 60)
            print("RE-SWEEPING SIGMAS with tuned model params")
            print("-" * 60)
            old_sigmas = dict(best_sigmas)
            best_sigmas, all_results, _ = run_sweep_ordering(
                stages, sigma_sweeps, base_phil, {}, label="[re-sweep] ",
                densify=args.densify)

            # Report changes
            print("\n  Sigma changes after model tuning:")
            for param in sorted(set(list(old_sigmas.keys()) + list(best_sigmas.keys()))):
                old = old_sigmas.get(param)
                new = best_sigmas.get(param)
                if old is not None and new is not None:
                    if isinstance(old, (list, tuple)):
                        print("    %s: [per-component, see deep-dive]" % param)
                    else:
                        ratio = new / old if old > 0 else float('inf')
                        print("    sigmas.%s: %.2e → %.2e (%.1fx)" %
                              (param, old, new, ratio))

    # --- Deep-dive: per-component sigma sweeps ---
    if args.deep_dive and not args.skip_to_tuning:
        print("\n" + "=" * 60)
        print("DEEP-DIVE: Per-component sigma sweeps")
        print("=" * 60)

        # Parse multipliers
        if args.deep_dive_multipliers:
            multipliers = sorted([float(x) for x in args.deep_dive_multipliers.split(',')])
        else:
            multipliers = list(DEEP_DIVE_MULTIPLIERS)

        # Determine which params to deep-dive
        if args.deep_dive_params:
            dd_param_names = [s.strip() for s in args.deep_dive_params.split(',')]
        else:
            # Auto-detect based on what was refined in standard stages
            dd_param_names = []
            if use_cholesky:
                dd_param_names.append('cholesky')
            elif 'Nabc' in best_sigmas:
                dd_param_names.append('Nabc')
            # RotXYZ: scalar sigma is sufficient; per-component deep-dive
            # adds noise without meaningful improvement. Use --deep-dive-params
            # RotXYZ to force it if needed.
            if 'B' in best_sigmas:
                dd_param_names.append('B')
            # Baniso only if explicitly requested or if base phil has it
            if 'Baniso' in base_phil and 'fix.Baniso' not in base_phil:
                dd_param_names.append('Baniso')

        # Filter to valid params
        valid_dd_params = []
        for dd_param in dd_param_names:
            if dd_param not in DEEP_DIVE_PARAMS:
                print("  WARNING: Unknown deep-dive param '%s', skipping" % dd_param)
                continue
            n_comp = DEEP_DIVE_PARAMS[dd_param][1]
            if n_comp == 1 and dd_param == 'B':
                print("  Skipping deep-dive for B (scalar, already tuned)")
                continue
            valid_dd_params.append(dd_param)

        print("  Parameters: %s" % ', '.join(valid_dd_params))
        print("  Multipliers: %s" % ', '.join(['%.1f' % m for m in multipliers]))

        if len(valid_dd_params) > 1 and args.n_parallel > 1:
            # Run different parameter deep-dives in parallel
            # Note: with MPI, we can't use ThreadPoolExecutor across params
            # since all ranks share one comm. Instead, run sequentially but
            # each param's trials are parallelized via MPI/threads.
            if mpi_comm is not None:
                print("  MPI mode: running %d params sequentially, trials in parallel" %
                      len(valid_dd_params))
                dd_max = args.deep_dive_max_calls
                for dd_param in valid_dd_params:
                    dd_param, dd_data = _run_deep_dive_for_param(
                        dd_param, base_phil, dict(best_sigmas), sample_lines,
                        multipliers, use_cholesky, args.cuda,
                        dd_max, args.n_parallel,
                        args.n_gpus, args.procs_per_gpu, mpi_comm,
                        trial_num_devices,
                        densify=args.densify)
                    deep_dive_results[dd_param] = dd_data
                    best_sigmas[dd_param] = dd_data['comp_sigmas']
            else:
                print("  Running %d parameter sweeps in parallel..." % len(valid_dd_params))
                dd_max = args.deep_dive_max_calls
                with ThreadPoolExecutor(max_workers=len(valid_dd_params)) as executor:
                    futures = {}
                    for dd_param in valid_dd_params:
                        f = executor.submit(
                            _run_deep_dive_for_param, dd_param, base_phil,
                            dict(best_sigmas), sample_lines, multipliers,
                            use_cholesky, args.cuda, dd_max,
                            args.n_parallel, args.n_gpus, args.procs_per_gpu,
                            None, trial_num_devices,
                            args.densify)
                        futures[f] = dd_param

                    for future in as_completed(futures):
                        dd_param, dd_data = future.result()
                        deep_dive_results[dd_param] = dd_data
                        best_sigmas[dd_param] = dd_data['comp_sigmas']
        else:
            # Sequential: run one parameter at a time
            dd_max = args.deep_dive_max_calls
            for dd_param in valid_dd_params:
                dd_param, dd_data = _run_deep_dive_for_param(
                    dd_param, base_phil, best_sigmas, sample_lines,
                    multipliers, use_cholesky, args.cuda, dd_max,
                    args.n_parallel, args.n_gpus,
                    args.procs_per_gpu, mpi_comm, trial_num_devices,
                    densify=args.densify)
                deep_dive_results[dd_param] = dd_data
                best_sigmas[dd_param] = dd_data['comp_sigmas']

    # --- Final refinement: get init values for optimized phil ---
    if not args.skip_to_tuning:
        refined_params = None
    if not args.no_final_refine and not args.skip_to_tuning:
        print("\n" + "=" * 60)
        print("FINAL REFINEMENT: Refining sample with optimized sigmas")
        print("=" * 60)
        print("  Max calls: %d" % args.final_refine_max_calls)
        print("  Purpose: extract refined G, B, Nabc for init values in optimized phil")

        refined_params = run_final_refinement(
            base_phil, best_sigmas, sample_lines,
            use_cholesky=use_cholesky, use_cuda=args.cuda,
            max_calls=args.final_refine_max_calls,
            n_parallel=args.n_parallel, n_gpus=args.n_gpus,
            procs_per_gpu=args.procs_per_gpu, mpi_comm=mpi_comm,
            num_devices=trial_num_devices, init_G=init_G,
            refine_ucell='ucell' in stages)

        if refined_params:
            print("\n  Final refinement succeeded — init values will be written to phil")
        else:
            print("\n  Final refinement failed — using G estimation only for init values")

    # Write optimized phil (sigmas + init values, no tuned restraints yet)
    if not args.skip_to_tuning:
        write_optimized_phil(base_phil, best_sigmas, init_G, args.output,
                             use_cholesky=use_cholesky, refined_params=refined_params,
                             tuned_params=None)
        print("\n  Wrote optimized phil to: %s" % args.output)

        # Save checkpoint for --skip-to-tuning
        import json
        checkpoint = {
            'best_sigmas': {k: (list(v) if isinstance(v, (list, tuple)) else v)
                            for k, v in best_sigmas.items()},
            'init_G': init_G,
            'refined_params': None,
            'tuned_model': tuned_model,
        }
        if refined_params:
            checkpoint['refined_params'] = {}
            for k, v in refined_params.items():
                if isinstance(v, tuple):
                    checkpoint['refined_params'][k] = list(v)
                else:
                    checkpoint['refined_params'][k] = v
        checkpoint_path = args.output.replace('.phil', '_checkpoint.json')
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        print("  Saved checkpoint to: %s" % checkpoint_path)

    # --- Phase 2: Restraint tuning (spread-ratio based) ---
    tuned_params = {}
    beta_curves = {}  # beta_name -> [(beta, spread_ratio), ...]

    if args.tune_restraints:
        if refined_params is None:
            print("\n  WARNING: --tune-restraints requires final refinement for centers.")
            print("  Skipping tuning phase (run without --no-final-refine).")
        else:
            print("\n" + "=" * 60)
            print("RESTRAINT TUNING (spread-ratio method)")
            print("=" * 60)
            print("  Method: sweep betas, measure parameter spread vs unrestricted")
            print("  Metric: spread_ratio = var(param, with_beta) / var(param, free)")
            print("    ratio=1.0 → no restraint effect, ratio→0 → params pinned to center")

            # Build base phil for tuning (all params free, best sigmas, centers set)
            tuning_base = build_tuning_base_phil(
                base_phil, best_sigmas, init_G, refined_params,
                use_cholesky=use_cholesky, max_calls=args.tune_max_calls,
                refine_ucell='ucell' in stages)

            # Override ROI fraction for spread tuning
            # Default: use all ROIs for accurate variance measurement
            if args.spread_roi_fraction is not None:
                tuning_base += '\nroi.fraction = %.4f\n' % args.spread_roi_fraction
                print("  Spread ROI fraction: %.0f%%" % (args.spread_roi_fraction * 100))
            else:
                tuning_base += '\nroi.fraction = None\n'
                print("  Spread ROI fraction: 100% (all ROIs)")

            # Determine which beta params to sweep
            if args.tune_params:
                tune_param_names = [s.strip() for s in args.tune_params.split(',')]
            else:
                tune_param_names = []
                # Model params (phisteps/sigma_r) — only if not already done
                if not tuned_model:
                    tune_param_names.extend(['phisteps', 'sigma_r'])
                elif 'phisteps' not in tuned_model:
                    tune_param_names.append('phisteps')
                elif 'sigma_r' not in tuned_model:
                    tune_param_names.append('sigma_r')
                # Beta params
                if 'G' in refined_params:
                    tune_param_names.append('betas.G')
                if 'B' in refined_params:
                    tune_param_names.append('betas.B')
                if use_cholesky:
                    tune_param_names.append('betas.cholesky')
                elif 'Nabc' in refined_params:
                    tune_param_names.append('betas.Nabc')
                tune_param_names.append('betas.RotXYZ')
                if 'ucell' in stages and 'ucell' in refined_params:
                    tune_param_names.append('betas.ucell')

            has_beta_params = any(n.startswith('betas.') for n in tune_param_names)

            # tune_sample_lines already set up before model tuning

            # Run unrestrained baseline trial to establish spread denominator
            # This gives apples-to-apples comparison (same frames, ROI fraction, max_calls)
            baseline_vars = {}
            if has_beta_params:
                print("\n  Running unrestrained baseline trial...")
                baseline_phil = tuning_base + '\nuse_restraints = False\n'
                baseline_results = run_tuning_sweep(
                    baseline_phil, 'baseline', 'betas.G', [1.0],
                    tune_sample_lines, use_cuda=args.cuda,
                    n_parallel=args.n_parallel,
                    n_gpus=args.n_gpus, procs_per_gpu=args.procs_per_gpu,
                    mpi_comm=mpi_comm, num_devices=trial_num_devices,
                    densify=1)
                if baseline_results and baseline_results[0].get('param_vars'):
                    baseline_vars = baseline_results[0]['param_vars']
                    bl = baseline_results[0]
                    print("  Baseline: sigZ_med=%.3f n_rois=%d (%.1fs)" %
                          (bl['sigZ_median'], bl['n_rois'], bl['time_sec']))
                    print("\n  Unrestricted parameter spread (baseline trial):")
                    for bname in [n for n in tune_param_names if n.startswith('betas.')]:
                        binfo = BETA_PARAM_MAP.get(bname)
                        if binfo:
                            bvars = [baseline_vars.get(c) for c in binfo['csv_cols']
                                     if c in baseline_vars]
                            if bvars:
                                if len(bvars) == 1:
                                    print("    %s: var = %.4g" % (bname, bvars[0]))
                                else:
                                    print("    %s: var = (%s)" %
                                          (bname, ', '.join('%.4g' % v for v in bvars)))
                else:
                    print("  WARNING: baseline trial produced no param_vars")
                    print("  Falling back to final-refinement variance as denominator")

            print("\n  Parameters to tune: %s" % ', '.join(tune_param_names))
            print("  Max calls per trial: %d" % args.tune_max_calls)

            for tune_name in tune_param_names:
                is_beta = tune_name.startswith('betas.')

                if tune_name == 'phisteps':
                    sweep_vals = list(DEFAULT_PHISTEPS)
                    phil_path = TUNE_PARAMS['phisteps']
                elif tune_name == 'sigma_r':
                    sweep_vals = list(DEFAULT_SIGMA_R)
                    phil_path = TUNE_PARAMS['sigma_r']
                elif is_beta:
                    # Variance-scaled sweep: centered on baseline variance
                    sweep_vals = generate_beta_sweep(
                        tune_name, baseline_vars=baseline_vars,
                        refined_params=refined_params)
                    phil_path = TUNE_PARAMS.get(tune_name, tune_name)
                else:
                    print("  WARNING: Unknown tune param '%s', skipping" % tune_name)
                    continue

                print("\n  --- %s ---" % tune_name)
                print("  Sweep values: %s" % ', '.join(['%.3g' % v for v in sweep_vals]))

                # Beta sweeps and sigma_r use tune_sample_lines (more frames
                # for stable variance / sigma_r estimates); phisteps uses sample_lines
                sweep_lines = tune_sample_lines if (is_beta or tune_name == 'sigma_r') else sample_lines
                results = run_tuning_sweep(
                    tuning_base, tune_name, phil_path, sweep_vals,
                    sweep_lines, use_cuda=args.cuda,
                    n_parallel=args.n_parallel,
                    n_gpus=args.n_gpus, procs_per_gpu=args.procs_per_gpu,
                    mpi_comm=mpi_comm, num_devices=trial_num_devices,
                    densify=args.densify)

                if is_beta:
                    # --- Spread-ratio scoring for beta params ---
                    print("\n  %-12s  %10s  %10s  %6s  %8s  %5s  %s" %
                          ('beta', 'sigZ_med', 'sigZ_mean', 'n_roi', 'spread', 'time', 'status'))
                    print("  " + "-" * 72)
                    curve = []
                    for r in results:
                        status = 'OK' if r['converged'] and r['error'] is None else 'FAIL'
                        spread = compute_spread_ratio(
                            r.get('param_vars', {}), tune_name,
                            refined_params=refined_params,
                            baseline_vars=baseline_vars)
                        if spread is not None:
                            curve.append((r['value'], spread))
                        sigz_med = '%.3f' % r['sigZ_median'] if r['sigZ_median'] < 1e6 else 'inf'
                        sigz_mean = '%.3f' % r['sigZ_mean'] if r['sigZ_mean'] < 1e6 else 'inf'
                        spread_s = '%.4f' % spread if spread is not None else 'N/A'
                        print("  %-12.3g  %10s  %10s  %6d  %8s  %5.1f  %s" %
                              (r['value'], sigz_med, sigz_mean, r['n_rois'],
                               spread_s, r['time_sec'], status))
                    if curve:
                        beta_curves[tune_name] = curve
                        spreads_only = [s for _, s in curve]
                        print("  Spread range: %.4f — %.4f" %
                              (min(spreads_only), max(spreads_only)))
                else:
                    # --- Scoring for model params (phisteps, sigma_r) ---
                    has_sp = any(np.isfinite(r.get('spearman_r', float('nan')))
                                 for r in results if r['converged'] and r['error'] is None)
                    use_sp = (tune_name == 'sigma_r') and has_sp
                    if use_sp:
                        print("\n  %-12s  %10s  %10s  %8s  %6s  %5s  %5s  %s" %
                              ('value', 'sigZ_med', 'sigZ_mean', 'spR_med', 'n_roi', 'n_out', 'time', 'status'))
                        print("  " + "-" * 78)
                    else:
                        print("\n  %-12s  %10s  %10s  %6s  %5s  %5s  %s" %
                              ('value', 'sigZ_med', 'sigZ_mean', 'n_roi', 'n_out', 'time', 'status'))
                        print("  " + "-" * 68)
                    valid_results = [r for r in results
                                     if r['converged'] and r['error'] is None
                                     and r['sigZ_median'] < 1e6]
                    scorer = _score_trial_spearman if use_sp else _score_trial
                    best_r = min(valid_results, key=scorer) if valid_results else None
                    for r in results:
                        status = 'OK' if r['converged'] and r['error'] is None else 'FAIL'
                        sigz_med = '%.3f' % r['sigZ_median'] if r['sigZ_median'] < 1e6 else 'inf'
                        sigz_mean = '%.3f' % r['sigZ_mean'] if r['sigZ_mean'] < 1e6 else 'inf'
                        marker = ' <-- best' if r is best_r else ''
                        if use_sp:
                            spr = '%.4f' % r.get('spearman_r', float('nan')) if np.isfinite(r.get('spearman_r', float('nan'))) else 'N/A'
                            print("  %-12.3g  %10s  %10s  %8s  %6d  %5d  %5.1f  %s%s" %
                                  (r['value'], sigz_med, sigz_mean, spr, r['n_rois'],
                                   r.get('n_outliers', 0), r['time_sec'],
                                   status, marker))
                            continue
                        print("  %-12.3g  %10s  %10s  %6d  %5d  %5.1f  %s%s" %
                              (r['value'], sigz_med, sigz_mean, r['n_rois'],
                               r.get('n_outliers', 0), r['time_sec'],
                               status, marker))
                    if best_r is not None:
                        tuned_params[tune_name] = best_r['value']
                        print("\n  Best %s = %.4g" % (tune_name, best_r['value']))

            # --- Write restraint-level phil folder ---
            if beta_curves:
                target_levels = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1]
                base_name = os.path.splitext(args.output)[0]
                restraint_dir = base_name + '_restraints'
                # Include tuned model params in all phils
                model_params = dict(tuned_model or {})
                model_params.update({k: v for k, v in tuned_params.items()
                                     if not k.startswith('betas.')})
                write_restraint_level_phils(
                    base_phil, best_sigmas, init_G, restraint_dir,
                    use_cholesky, refined_params, beta_curves,
                    target_levels, tuned_model=model_params)

                # --- Auto-elbow: per-parameter optimal betas ---
                elbow_results = find_elbow_betas(beta_curves)
                if elbow_results:
                    print("\n  Auto-elbow per-parameter betas:")
                    elbow_betas = {}
                    for bname, (ebeta, espread, egrad, edetails) in elbow_results.items():
                        short = bname.split('.', 1)[1] if '.' in bname else bname
                        elbow_betas[bname] = ebeta
                        print("    %s: beta=%.4g  spread=%.3f  (%s)" %
                              (short, ebeta, espread, edetails))

                    # Write elbow phil
                    elbow_params = dict(model_params)
                    elbow_params.update(elbow_betas)
                    elbow_path = os.path.join(restraint_dir, 'level_auto_elbow.phil')
                    write_optimized_phil(
                        base_phil, best_sigmas, init_G, elbow_path,
                        use_cholesky=use_cholesky,
                        refined_params=refined_params,
                        tuned_params=elbow_params)
                    print("    Wrote: %s" % elbow_path)
                else:
                    print("\n  WARNING: No elbows found in any beta curve."
                          " Try increasing --n-frames for more stable variance.")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if refined_params:
        print("  Refined init values (median across sample):")
        if 'G' in refined_params:
            print("    init.G = %.4e" % refined_params['G'])
        if 'B' in refined_params:
            print("    init.B = %.2f" % refined_params['B'])
        if 'Nabc' in refined_params:
            print("    init.Nabc = (%.1f, %.1f, %.1f)" % refined_params['Nabc'])
        if 'cholesky' in refined_params:
            L = refined_params['cholesky']
            print("    init.cholesky = (%.3f, %.3f, %.3f, %.3f, %.3f, %.3f)" % L)
        if 'ucell' in refined_params:
            uc = refined_params['ucell']
            print("    init.ucell = (%.4f, %.4f, %.4f, %.4f, %.4f, %.4f)" % uc)
    elif init_G is not None:
        print("  init.G = %.4e (estimated)" % init_G)
    print("  Optimal sigmas:")
    for param, val in best_sigmas.items():
        if isinstance(val, list):
            dd_info = DEEP_DIVE_PARAMS.get(param, (param, len(val), None))
            comp_names = dd_info[2] or ['c%d' % i for i in range(len(val))]
            print("    sigmas.%s = [%s]" % (param, ', '.join(['%.4e' % v for v in val])))
            for name, v in zip(comp_names, val):
                print("      %s: %.4e" % (name, v))
        else:
            print("    sigmas.%s = %.4e" % (param, val))
    if tuned_model:
        print("  Model parameters (from --tune-model):")
        for k, v in tuned_model.items():
            print("    %s = %s" % (k, str(int(v)) if k == 'phisteps' else '%.4g' % v))
    if tuned_params:
        print("  Tuned model parameters:")
        for k, v in tuned_params.items():
            print("    %s = %.4g" % (k, v))
    if beta_curves:
        print("  Restraint tuning (spread-ratio curves):")
        for bname, pairs in beta_curves.items():
            valid = [(b, s) for b, s in pairs if s is not None]
            if valid:
                spreads = [s for _, s in valid]
                print("    %s: spread range %.3f — %.3f (%d points)" %
                      (bname, min(spreads), max(spreads), len(valid)))
        base_name = os.path.splitext(args.output)[0]
        print("  Restraint phil folder: %s_restraints/" % base_name)
        elbow_results = find_elbow_betas(beta_curves)
        if elbow_results:
            print("  Auto-elbow betas:")
            for bname, (ebeta, espread, egrad, edetails) in elbow_results.items():
                short = bname.split('.', 1)[1] if '.' in bname else bname
                print("    %s: %.4g (spread=%.3f)" % (short, ebeta, espread))
            print("  Recommended: %s_restraints/level_auto_elbow.phil" % base_name)
        else:
            print("  No elbows found — try increasing --n-frames")

    print("\n  Optimized phil (no restraints): %s" % args.output)

    # Write report
    report_path = args.output.replace('.phil', '_report.txt')
    with open(report_path, 'w') as f:
        f.write("hopper_heatup report\n")
        f.write("=" * 60 + '\n')
        f.write("Base phil: %s\n" % args.phil)
        f.write("Spec: %s\n" % spec_path)
        f.write("N frames: %d\n" % args.n_frames)
        f.write("Max calls: %d\n" % args.max_calls)
        f.write("Stages: %s\n" % ', '.join(stages))
        f.write("Cholesky: %s\n\n" % use_cholesky)

        if refined_params:
            f.write("Refined init values (median across sample):\n")
            if 'G' in refined_params:
                f.write("  init.G = %.6e\n" % refined_params['G'])
            if 'B' in refined_params:
                f.write("  init.B = %.4f\n" % refined_params['B'])
            if 'Nabc' in refined_params:
                f.write("  init.Nabc = %.2f, %.2f, %.2f\n" % refined_params['Nabc'])
            if 'cholesky' in refined_params:
                f.write("  init.cholesky = %.6f, %.6f, %.6f, %.6f, %.6f, %.6f\n" % refined_params['cholesky'])
            if 'ucell' in refined_params:
                f.write("  init.ucell = %.4f, %.4f, %.4f, %.4f, %.4f, %.4f\n" % refined_params['ucell'])
            f.write("\n")
        elif init_G is not None:
            f.write("Estimated init.G = %.6e\n\n" % init_G)

        for stage, results in all_results.items():
            f.write("Stage: %s\n" % stage)
            if stage == 'G+B':
                f.write("  %-10s  %-10s  %10s  %10s  %6s  %5s  %5s  %s\n" %
                        ('sigma_G', 'sigma_B', 'sigZ_med', 'sigZ_mean',
                         'n_roi', 'n_out', 'time', 'status'))
                f.write("  " + "-" * 78 + '\n')
                for r in sorted(results, key=lambda r: (r['sigma_G'], r['sigma_B'])):
                    status = 'OK' if r['converged'] and r.get('error') is None else 'FAIL'
                    sigz_med = '%.3f' % r['sigZ_median'] if r['sigZ_median'] < 1e6 else 'inf'
                    sigz_mean = '%.3f' % r['sigZ_mean'] if r['sigZ_mean'] < 1e6 else 'inf'
                    f.write("  %-10.2e  %-10.2e  %10s  %10s  %6d  %5d  %5.1f  %s\n" %
                            (r['sigma_G'], r['sigma_B'], sigz_med, sigz_mean,
                             r['n_rois'], r.get('n_outliers', 0),
                             r['time_sec'], status))
                valid = [r for r in results
                         if r['converged'] and r.get('error') is None
                         and r['sigZ_median'] < 1e6]
                if valid:
                    best_r = min(valid, key=_score_trial)
                    f.write("  --> Best: G=%.2e, B=%.2e\n" %
                            (best_r['sigma_G'], best_r['sigma_B']))
            else:
                f.write("  %-12s  %10s  %10s  %6s  %5s  %5s  %s\n" %
                        ('sigma', 'sigZ_med', 'sigZ_mean', 'n_roi', 'n_out', 'time', 'status'))
                f.write("  " + "-" * 68 + '\n')
                for r in results:
                    status = 'OK' if r['converged'] and r.get('error') is None else 'FAIL'
                    sigz_med = '%.3f' % r['sigZ_median'] if r['sigZ_median'] < 1e6 else 'inf'
                    sigz_mean = '%.3f' % r['sigZ_mean'] if r['sigZ_mean'] < 1e6 else 'inf'
                    f.write("  %-12.2e  %10s  %10s  %6d  %5d  %5.1f  %s\n" %
                            (r['sigma'], sigz_med, sigz_mean, r['n_rois'],
                             r.get('n_outliers', 0), r['time_sec'], status))
                best = find_best_sigma(results)
                if best is not None:
                    f.write("  --> Best: %.2e\n" % best)
            f.write("\n")

        # Deep-dive results
        if deep_dive_results:
            f.write("\nDeep-dive per-component results:\n")
            for dd_param, dd_data in deep_dive_results.items():
                f.write("  %s (scalar best: %.2e):\n" % (dd_param, dd_data['scalar_best']))
                for name, sig in zip(dd_data['comp_names'], dd_data['comp_sigmas']):
                    ratio = sig / dd_data['scalar_best'] if dd_data['scalar_best'] > 0 else 0
                    f.write("    %s: %.6e (%.1fx)\n" % (name, sig, ratio))
                f.write("\n")

        f.write("Final optimal sigmas:\n")
        for param, val in best_sigmas.items():
            if isinstance(val, list):
                f.write("  sigmas.%s = %s\n" % (param, ','.join(['%.6e' % v for v in val])))
            else:
                f.write("  sigmas.%s = %.6e\n" % (param, val))

        if tuned_params:
            f.write("\nTuned model parameters:\n")
            for k, v in tuned_params.items():
                f.write("  %s = %.6g\n" % (k, v))

        if beta_curves:
            f.write("\nRestraint tuning (spread-ratio curves):\n")
            for bname, pairs in beta_curves.items():
                f.write("  %s:\n" % bname)
                f.write("    %-12s  %8s\n" % ('beta', 'spread'))
                f.write("    " + "-" * 24 + "\n")
                for beta, spread in sorted(pairs, key=lambda x: x[0]):
                    sr = '%.4f' % spread if spread is not None else 'N/A'
                    f.write("    %-12.3g  %8s\n" % (beta, sr))
                f.write("\n")

            elbow_results = find_elbow_betas(beta_curves)
            if elbow_results:
                f.write("\nAuto-elbow per-parameter betas:\n")
                f.write("  %-12s  %12s  %8s  %s\n" % ('param', 'beta', 'spread', 'transition'))
                f.write("  " + "-" * 60 + "\n")
                for bname, (ebeta, espread, egrad, edetails) in elbow_results.items():
                    short = bname.split('.', 1)[1] if '.' in bname else bname
                    f.write("  %-12s  %12.4g  %8.3f  %s\n" % (short, ebeta, espread, edetails))
                f.write("\n")

    print("  Wrote report to: %s" % report_path)

    # Shutdown MPI workers
    if mpi_comm is not None:
        size = mpi_comm.Get_size()
        for worker in range(1, size):
            mpi_comm.send(('shutdown',), dest=worker)

    # Cleanup
    if not args.keep_trials and args.tmpdir is None:
        print("\n  Cleaning up temp directory: %s" % tmpdir)
        shutil.rmtree(tmpdir, ignore_errors=True)
    else:
        print("\n  Trial outputs kept in: %s" % tmpdir)

    print("\nDone! Run hopper with the optimized phil:")
    print("  mpirun -n <NPROC> hopper.py %s" % args.output)
    if beta_curves:
        base_name = os.path.splitext(args.output)[0]
        print("  Recommended (auto-elbow): %s_restraints/level_auto_elbow.phil" % base_name)
        print("  Or try different restraint levels from: %s_restraints/" % base_name)

    # Close log file
    print("\nLog saved to: %s" % os.path.abspath(logfile_path))
    try:
        _logfile.close()
    except Exception:
        pass


if __name__ == '__main__':
    main()
