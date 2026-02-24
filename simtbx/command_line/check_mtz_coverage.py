#!/usr/bin/env python
"""
Check how well an MTZ covers the observed reflections in expanded data.

Reports the fraction of observed HKLs (from reflection tables) that have
a matching entry in the MTZ, broken down by resolution shell.

Usage:
  python check_mtz_coverage.py iobs_all.mtz expanded/exp_ref_spec.txt \\
      --space-group P22121 --n-bins 10

  # Or check a single .refl file directly:
  python check_mtz_coverage.py iobs_all.mtz --refl frame_000000.refl \\
      --expt frame_000000.expt --space-group P22121
"""

from __future__ import division, print_function

# LIBTBX_SET_DISPATCHER_NAME diffBragg.check_mtz_coverage

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                        description="Check MTZ coverage of observed reflections")
parser.add_argument("mtz", type=str, help="MTZ file with structure factors")
parser.add_argument("exp_ref_spec", type=str, nargs="?", default=None,
                    help="exp_ref_spec file from expand_rotation_to_stills.py")
parser.add_argument("--refl", type=str, default=None,
                    help="Single .refl file to check (instead of exp_ref_spec)")
parser.add_argument("--expt", type=str, default=None,
                    help="Single .expt file (needed with --refl for unit cell)")
parser.add_argument("--mtz-column", type=str, default=None,
                    help="MTZ column label. If None, uses first array found.")
parser.add_argument("--space-group", type=str, default=None,
                    help="Space group for mapping to ASU (e.g., P22121)")
parser.add_argument("--n-bins", type=int, default=10,
                    help="Number of resolution bins")
parser.add_argument("--max-frames", type=int, default=None,
                    help="Max frames to sample from exp_ref_spec (for speed)")
parser.add_argument("--d-min", type=float, default=None,
                    help="High resolution limit (Angstrom)")

args = parser.parse_args()

import os
import numpy as np

from cctbx import miller, crystal, sgtbx, uctbx
from dxtbx.model import ExperimentList
from dials.array_family import flex
from iotbx.reflection_file_reader import any_reflection_file


def load_mtz_hkls(mtz_file, mtz_column=None):
    """Load HKLs from MTZ file, return miller set in ASU."""
    miller_arrays = any_reflection_file(mtz_file).as_miller_arrays()

    if mtz_column is not None:
        ma = None
        for arr in miller_arrays:
            if arr.info().label_string() == mtz_column:
                ma = arr
                break
        if ma is None:
            labels = [arr.info().label_string() for arr in miller_arrays]
            raise ValueError("Column '%s' not found. Available: %s"
                             % (mtz_column, ", ".join(labels)))
    else:
        # Use first array
        ma = miller_arrays[0]
        print("Using MTZ column: %s" % ma.info().label_string())

    # Map to ASU
    ma_asu = ma.map_to_asu()
    return ma_asu


def get_observed_hkls(exp_ref_spec, max_frames=None):
    """Collect all unique HKLs from reflection tables in exp_ref_spec."""
    with open(exp_ref_spec) as f:
        lines = [l.strip() for l in f if l.strip()]

    if max_frames is not None and max_frames < len(lines):
        # Sample evenly
        indices = np.linspace(0, len(lines) - 1, max_frames, dtype=int)
        lines = [lines[i] for i in indices]

    all_hkls = set()
    n_refls_total = 0

    for i, line in enumerate(lines):
        parts = line.split()
        refl_file = parts[1]
        R = flex.reflection_table.from_file(refl_file)
        if 'miller_index' not in R:
            continue

        for h, k, l in R['miller_index']:
            all_hkls.add((h, k, l))
        n_refls_total += len(R)

        if (i + 1) % 200 == 0:
            print("  Read %d / %d frames (%d unique HKLs so far)"
                  % (i + 1, len(lines), len(all_hkls)), flush=True)

    print("Sampled %d frames, %d total reflections, %d unique HKLs"
          % (len(lines), n_refls_total, len(all_hkls)))
    return all_hkls


def get_observed_hkls_single(refl_file):
    """Get HKLs from a single reflection table."""
    R = flex.reflection_table.from_file(refl_file)
    hkls = set()
    for h, k, l in R['miller_index']:
        hkls.add((h, k, l))
    print("Single file: %d reflections, %d unique HKLs" % (len(R), len(hkls)))
    return hkls


def map_hkls_to_asu(hkls, space_group, unit_cell):
    """Map a set of (h,k,l) tuples to ASU using cctbx."""
    sgi = sgtbx.space_group_info(space_group)
    sg = sgi.group()
    sym = crystal.symmetry(unit_cell=unit_cell, space_group=sg)
    ms = miller.set(sym, flex.miller_index(list(hkls)), anomalous_flag=True)
    ms_asu = ms.map_to_asu()
    return ms_asu


def main():
    # Load MTZ
    print("Loading MTZ: %s" % args.mtz)
    ma_mtz = load_mtz_hkls(args.mtz, args.mtz_column)
    mtz_hkls = set(ma_mtz.indices())
    unit_cell = ma_mtz.unit_cell()
    sg = ma_mtz.space_group()
    sg_info = sgtbx.space_group_info(group=sg)

    print("MTZ: %d unique HKLs, space group %s, unit cell %s"
          % (len(mtz_hkls), sg_info, unit_cell))

    # Load observed HKLs
    if args.exp_ref_spec is not None:
        print("\nLoading observed HKLs from exp_ref_spec...")
        obs_hkls_raw = get_observed_hkls(args.exp_ref_spec, args.max_frames)
    elif args.refl is not None:
        obs_hkls_raw = get_observed_hkls_single(args.refl)
    else:
        parser.error("Provide either exp_ref_spec or --refl")

    # Determine space group for ASU mapping
    if args.space_group is not None:
        map_sg = args.space_group
    else:
        map_sg = str(sg_info)
    print("\nMapping observed HKLs to ASU using space group: %s" % map_sg)

    # Get unit cell from MTZ or expt
    if args.expt is not None:
        El = ExperimentList.from_file(args.expt, False)
        uc = El[0].crystal.get_unit_cell().parameters()
    else:
        uc = unit_cell.parameters()

    obs_ms = map_hkls_to_asu(obs_hkls_raw, map_sg, uc)
    obs_hkls_asu = set(obs_ms.indices())
    print("Observed (ASU): %d unique HKLs" % len(obs_hkls_asu))

    # Filter by d_min if requested
    if args.d_min is not None:
        obs_ms = obs_ms.resolution_filter(d_min=args.d_min)
        obs_hkls_asu = set(obs_ms.indices())
        print("After d_min=%.2f filter: %d unique HKLs" % (args.d_min, len(obs_hkls_asu)))

    # Compute coverage
    covered = obs_hkls_asu.intersection(mtz_hkls)
    missing = obs_hkls_asu - mtz_hkls
    n_obs = len(obs_hkls_asu)
    n_covered = len(covered)
    n_missing = len(missing)

    print("\n" + "=" * 60)
    print("COVERAGE SUMMARY")
    print("=" * 60)
    print("  Observed unique HKLs (ASU):  %d" % n_obs)
    print("  Covered by MTZ:              %d  (%.1f%%)" % (n_covered, 100 * n_covered / max(n_obs, 1)))
    print("  Missing from MTZ:            %d  (%.1f%%)" % (n_missing, 100 * n_missing / max(n_obs, 1)))
    print("  MTZ total HKLs:              %d" % len(mtz_hkls))

    # Resolution-binned coverage
    if n_obs > 0:
        # Use the observed miller set for binning
        obs_ms_for_bins = map_hkls_to_asu(obs_hkls_raw, map_sg, uc)
        if args.d_min is not None:
            obs_ms_for_bins = obs_ms_for_bins.resolution_filter(d_min=args.d_min)

        obs_ms_for_bins.setup_binner(n_bins=args.n_bins)
        binner = obs_ms_for_bins.binner()

        print("\n%-5s  %-22s  %7s  %7s  %7s" % (
            "Bin", "Resolution", "Obs", "Covered", "Coverage"))
        print("-" * 60)

        for i_bin in binner.range_used():
            sel = binner.selection(i_bin)
            bin_hkls = set(obs_ms_for_bins.select(sel).indices())
            bin_covered = bin_hkls.intersection(mtz_hkls)
            n_bin = len(bin_hkls)
            n_bin_cov = len(bin_covered)
            d_max_bin, d_min_bin = binner.bin_d_range(i_bin)
            pct = 100 * n_bin_cov / max(n_bin, 1)
            print("%3d    %6.2f - %6.2f A     %5d    %5d    %5.1f%%" % (
                i_bin, d_max_bin, d_min_bin, n_bin, n_bin_cov, pct))

        print("-" * 60)
        print("%-28s  %5d    %5d    %5.1f%%" % (
            "Total", n_obs, n_covered,
            100 * n_covered / max(n_obs, 1)))

    print()
    if n_missing > 0:
        print("RECOMMENDATION: Run diffBragg.completeF to fill %d missing reflections" % n_missing)
        print("  before stage 1 refinement (hopper.py).")
    else:
        print("All observed HKLs are covered by the MTZ. No completion needed.")


if __name__ == '__main__':
    main()
