#!/usr/bin/env python
"""
Macro-cycling wrapper for hopper refinement + geometry optimization.

Runs alternating cycles of:
  1. Hopper per-image refinement
  2. Geometry optimization
  3. Update detector → repeat

Usage:
  hopper_cycler.py input.txt [hopper_args] geometry.optimize=True geometry.macro_cycles=N
  hopper_cycler.py input.txt --mpi-cmd "mpirun -n 64" geometry.macro_cycles=3

Cycler-specific arguments:
  geometry.macro_cycles=N          Number of refinement cycles (default: 1)
  geometry.cycler_outdir=DIR       Parent output directory (creates DIR/cycle1, DIR/cycle2, etc.)
  --mpi-cmd "COMMAND"              MPI launcher (e.g., "mpirun -n 64", "srun -n 128")
  --mpi-cmd="COMMAND"              (equals format also supported)
"""
from __future__ import division, print_function
import sys
import os
import subprocess
import argparse
import shlex
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


def remove_cycler_args(args):
    """Remove cycler-specific arguments (managed by cycler, not passed to hopper).

    Removes both full and abbreviated forms (phil auto-completion):
      geometry.macro_cycles=N / macro_cycles=N
      geometry.cycler_outdir=DIR / cycler_outdir=DIR
    """
    new_args = []
    skip_next = False

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


def run_hopper_cycle(args, cycle_num, reference_geom=None, cycler_outdir=None, mpi_cmd=None):
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
    print("Base arguments: %s" % " ".join(hopper_args))
    print("="*80)

    # Run cycles
    reference_geom = None
    final_detector = None
    for cycle in range(1, n_cycles + 1):
        detector_path = run_hopper_cycle(hopper_args, cycle, reference_geom, cycler_outdir, mpi_cmd)
        final_detector = detector_path  # Track the last detector produced

        # Update reference for next cycle
        if detector_path is not None and cycle < n_cycles:
            reference_geom = detector_path
        elif detector_path is None and cycle < n_cycles:
            print("\nERROR: Cannot continue to cycle %d without optimized detector" % (cycle + 1))
            sys.exit(1)

    print("\n" + "="*80)
    print("MACRO-CYCLING COMPLETE - %d cycles finished" % n_cycles)
    print("="*80)
    if final_detector is not None:
        print("Final optimized detector: %s" % final_detector)
    print()


if __name__ == "__main__":
    main()
