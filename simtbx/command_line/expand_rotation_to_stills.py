#!/usr/bin/env python
"""
Expand a rotation crystallography experiment into per-frame pseudo-stills
for processing through diffBragg.

Takes a DIALS rotation experiment (one crystal, one scan with N frames,
one goniometer) and produces N single-frame experiments with 2D reflections.

Two modes:
  Default:         Bake goniometer rotation into crystal A-matrix.
                   Output experiments have no scan/gonio (pseudo-stills).
                   For use with hopper.py (Approach A).

  --no-bake-gonio: Keep crystal A in the goniometer frame.
                   Output experiments preserve scan/gonio metadata.
                   For use with rotation_hopper.py (Approach B).

Reflections are expanded from 3D (spanning multiple frames) to 2D
(one per frame), preserving the full profile for pixel-level modeling.

Usage:
  mpirun -n 32 python expand_rotation_to_stills.py \\
      integrated.expt integrated.refl --outdir expanded/

  # Or without MPI:
  python expand_rotation_to_stills.py \\
      integrated.expt integrated.refl --outdir expanded/
"""

from __future__ import division, print_function

# LIBTBX_SET_DISPATCHER_NAME diffBragg.expand_rotation

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                        description="Expand rotation data to per-frame pseudo-stills")
parser.add_argument("expt", type=str, help="Input .expt file from DIALS rotation processing")
parser.add_argument("refl", type=str, help="Input .refl file from DIALS rotation processing")
parser.add_argument("--outdir", type=str, required=True, help="Output directory for expanded files")
parser.add_argument("--prefix", type=str, default="frame", help="Prefix for output filenames")
parser.add_argument("--no-bake-gonio", action="store_true",
                    help="Keep crystal A in goniometer frame (for rotation_hopper.py)")
parser.add_argument("--exp-idx", type=int, default=0,
                    help="Experiment index if .expt has multiple experiments")

args = parser.parse_args()

import os
import sys
import numpy as np

from dxtbx.model import ExperimentList, Experiment, Crystal, Scan
from scitbx.matrix import sqr, col
from dials.array_family import flex
from simtbx.diffBragg import hopper_io

from libtbx.mpi4py import MPI
COMM = MPI.COMM_WORLD


def expand_rotation_data(args):
    """Main expansion logic."""

    outdir = args.outdir
    if COMM.rank == 0:
        os.makedirs(outdir, exist_ok=True)
    COMM.barrier()

    # ============================
    # 1. Load experiment (rank 0)
    # ============================
    El = None
    E = None
    scan_info = None  # (start_phi, osc_width, first_frame, last_frame)
    gonio_axis = None

    if COMM.rank == 0:
        El = ExperimentList.from_file(args.expt, False)
        E = El[args.exp_idx]

        if E.scan is None:
            raise ValueError("Experiment has no scan — is this really rotation data?")
        if E.goniometer is None:
            raise ValueError("Experiment has no goniometer — is this really rotation data?")

        start_phi, osc_width = E.scan.get_oscillation()
        first_frame, last_frame = E.scan.get_array_range()
        num_frames = last_frame - first_frame
        gonio_axis = col(E.goniometer.get_rotation_axis())

        has_scan_varying = hasattr(E.crystal, 'get_A_at_scan_point')
        if has_scan_varying:
            try:
                E.crystal.get_A_at_scan_point(0)
            except (RuntimeError, IndexError):
                has_scan_varying = False

        scan_info = {
            'start_phi': start_phi,
            'osc_width': osc_width,
            'first_frame': first_frame,
            'last_frame': last_frame,
            'num_frames': num_frames,
            'gonio_axis': tuple(gonio_axis.elems),
            'has_scan_varying': has_scan_varying,
        }
        print("Rotation dataset: %d frames, %.3f deg oscillation, phi range %.1f-%.1f deg"
              % (num_frames, osc_width, start_phi,
                 start_phi + num_frames * osc_width))
        print("Goniometer axis: %s" % (scan_info['gonio_axis'],))
        print("Scan-varying crystal: %s" % has_scan_varying)
        print("Bake gonio into A: %s" % (not args.no_bake_gonio))

    scan_info = COMM.bcast(scan_info)
    gonio_axis = col(scan_info['gonio_axis'])
    first_frame = scan_info['first_frame']
    last_frame = scan_info['last_frame']
    num_frames = scan_info['num_frames']
    osc_width = scan_info['osc_width']
    start_phi = scan_info['start_phi']
    has_scan_varying = scan_info['has_scan_varying']

    # ============================
    # 2. Build per-frame experiments (rank 0)
    # ============================
    if COMM.rank == 0:
        print("Creating %d per-frame experiments..." % num_frames)
        frame_expt_files = []

        for i_frame in range(num_frames):
            frame_idx = first_frame + i_frame  # scan array index
            phi_deg = start_phi + i_frame * osc_width

            # Get crystal A matrix for this frame
            if has_scan_varying:
                A = sqr(E.crystal.get_A_at_scan_point(i_frame))
            else:
                A = sqr(E.crystal.get_A())

            if not args.no_bake_gonio:
                # Bake goniometer rotation into A: A_lab = R_gonio(phi) * A_crystal
                Agon = gonio_axis.axis_and_angle_as_r3_rotation_matrix(
                    angle=phi_deg, deg=True)
                A = Agon * A

            # Create crystal with this A matrix
            # Extract real-space vectors from A (A = (a* b* c*)^T in reciprocal space)
            # Use set_A to set the full orientation+cell
            new_crystal = Crystal(E.crystal)  # copy
            new_crystal.set_A(A)

            # Create single-frame imageset
            single_iset = E.imageset.as_imageset()[frame_idx - first_frame:
                                                    frame_idx - first_frame + 1]

            # Build the experiment
            new_E = Experiment(
                detector=E.detector,
                beam=E.beam,
                crystal=new_crystal,
                imageset=single_iset,
            )

            if args.no_bake_gonio:
                # Preserve goniometer and create a single-frame scan at correct phi
                new_E.goniometer = E.goniometer
                new_E.scan = Scan((frame_idx, frame_idx), (phi_deg, osc_width))

            new_E.identifier = '%d' % i_frame

            # Save per-frame experiment
            fname = os.path.join(outdir, "%s_%06d.expt" % (args.prefix, i_frame))
            new_El = ExperimentList([new_E])
            new_El.as_file(fname)
            frame_expt_files.append(fname)

            if i_frame % 100 == 0 or i_frame == num_frames - 1:
                print("  experiments: %d / %d" % (i_frame + 1, num_frames), flush=True)

        print("Done creating experiments.")
    else:
        frame_expt_files = None

    frame_expt_files = COMM.bcast(frame_expt_files)
    COMM.barrier()

    # ============================
    # 3. Expand reflections (MPI parallel)
    # ============================
    Rall = None
    bbox_z1 = None
    bbox_z2 = None
    bbox_x1 = None
    bbox_x2 = None
    bbox_y1 = None
    bbox_y2 = None
    refl_data = None

    if COMM.rank == 0:
        print("Loading reflections...")
        Rall = flex.reflection_table.from_file(args.refl)
        n_refl_orig = len(Rall)
        print("  %d reflections loaded" % n_refl_orig)

        # Filter out unindexed reflections (miller_index == (0,0,0))
        if 'miller_index' in Rall:
            hkl = Rall['miller_index']
            is_indexed = flex.bool([h != (0, 0, 0) for h in hkl])
            n_unindexed = n_refl_orig - is_indexed.count(True)
            if n_unindexed > 0:
                Rall = Rall.select(is_indexed)
                print("  Removed %d unindexed reflections (h,k,l = 0,0,0), %d remain"
                      % (n_unindexed, len(Rall)))
        n_refl = len(Rall)

        # Extract bbox arrays for fast frame-to-reflection mapping
        bboxes = Rall['bbox']
        bbox_x1 = np.array([b[0] for b in bboxes])
        bbox_x2 = np.array([b[1] for b in bboxes])
        bbox_y1 = np.array([b[2] for b in bboxes])
        bbox_y2 = np.array([b[3] for b in bboxes])
        bbox_z1 = np.array([b[4] for b in bboxes])
        bbox_z2 = np.array([b[5] for b in bboxes])

        # Build frame → reflection index mapping
        print("Building frame-to-reflection mapping...")
        frame_to_refls = {}
        for i_frame in range(num_frames):
            frame_abs = first_frame + i_frame
            # Select reflections whose z-range includes this frame
            sel = (bbox_z1 <= frame_abs) & (bbox_z2 > frame_abs)
            frame_to_refls[i_frame] = np.where(sel)[0]

        total_expanded = sum(len(v) for v in frame_to_refls.values())
        print("  Total expanded reflections: %d (from %d originals, avg z-span %.1f)"
              % (total_expanded, n_refl, total_expanded / max(n_refl, 1)))

        # Prepare reflection data for broadcast
        # We send the full table + bbox arrays + mapping
        refl_data = {
            'Rall': Rall,
            'bbox_x1': bbox_x1, 'bbox_x2': bbox_x2,
            'bbox_y1': bbox_y1, 'bbox_y2': bbox_y2,
            'frame_to_refls': frame_to_refls,
        }

    refl_data = COMM.bcast(refl_data)
    Rall = refl_data['Rall']
    bbox_x1 = refl_data['bbox_x1']
    bbox_x2 = refl_data['bbox_x2']
    bbox_y1 = refl_data['bbox_y1']
    bbox_y2 = refl_data['bbox_y2']
    frame_to_refls = refl_data['frame_to_refls']

    # Each rank processes a subset of frames
    my_frames = [i for i in range(num_frames) if i % COMM.size == COMM.rank]
    frame_refl_files = [None] * num_frames

    if COMM.rank == 0:
        print("Expanding reflections across %d ranks..." % COMM.size)

    for i_frame in my_frames:
        refl_indices = frame_to_refls[i_frame]

        if len(refl_indices) == 0:
            # No reflections for this frame — write empty table
            new_refls = flex.reflection_table()
            new_refls['id'] = flex.int()
        else:
            # Select reflections for this frame
            sel = flex.bool(len(Rall), False)
            for idx in refl_indices:
                sel[int(idx)] = True
            new_refls = Rall.select(sel)

            # Set 2D bbox: keep x,y from original, set z=(0,1)
            new_bbox = flex.int6(len(new_refls))
            for j in range(len(new_refls)):
                orig_idx = refl_indices[j]
                new_bbox[j] = (int(bbox_x1[orig_idx]), int(bbox_x2[orig_idx]),
                               int(bbox_y1[orig_idx]), int(bbox_y2[orig_idx]),
                               0, 1)
            new_refls['bbox'] = new_bbox

            # Set ids to match experiment
            new_refls['id'] = flex.int(len(new_refls), 0)
            new_refls['imageset_id'] = flex.int(len(new_refls), 0)

        # Reset experiment identifiers
        new_refls.experiment_identifiers()[0] = '%d' % i_frame

        # Remove shoebox column if present (it's 3D and inconsistent with new bbox)
        if 'shoebox' in new_refls:
            del new_refls['shoebox']

        # Save
        refl_fname = os.path.join(outdir, "%s_%06d.refl" % (args.prefix, i_frame))
        new_refls.as_file(refl_fname)
        frame_refl_files[i_frame] = refl_fname

        if COMM.rank == 0 and i_frame % 100 == 0:
            print("  reflections: %d / %d frames" % (i_frame + 1, num_frames), flush=True)

    # Gather refl filenames across ranks
    all_refl_files = COMM.gather(
        {i: frame_refl_files[i] for i in my_frames}, root=0)

    if COMM.rank == 0:
        # Merge the filename dicts
        merged_refl_files = {}
        for d in all_refl_files:
            merged_refl_files.update(d)

        # Build exp_ref_spec file
        expt_list = []
        refl_list = []
        for i_frame in range(num_frames):
            expt_list.append(frame_expt_files[i_frame])
            refl_list.append(merged_refl_files[i_frame])

        spec_file = os.path.join(outdir, "exp_ref_spec.txt")
        hopper_io.save_expt_refl_file(spec_file, expt_list, refl_list,
                                       check_exists=True)
        print("\nDone! Wrote %d frame experiments and reflections to %s"
              % (num_frames, outdir))
        print("exp_ref_spec file: %s" % spec_file)

        # Also write a metadata file with phi angles per frame
        meta_file = os.path.join(outdir, "frame_metadata.txt")
        with open(meta_file, 'w') as f:
            f.write("# frame_idx  phi_start_deg  osc_width_deg\n")
            for i_frame in range(num_frames):
                phi = start_phi + i_frame * osc_width
                f.write("%d  %.6f  %.6f\n" % (i_frame, phi, osc_width))
        print("Metadata file: %s" % meta_file)


if __name__ == '__main__':
    expand_rotation_data(args)
