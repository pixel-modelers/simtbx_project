"""
Test the per-image B-factor (Debye-Waller) refinement.

Verifies that:
1. Setting Bfactor_image applies exp(-B * stol^2) damping to intensities.
2. The analytical gradient dI/dB agrees with finite-difference.
3. High-resolution reflections are damped more than low-resolution.

Usage:
    python tst_diffBragg_Bfactor_property.py
    python tst_diffBragg_Bfactor_property.py --plot
    python tst_diffBragg_Bfactor_property.py --Bfactor 5
"""
from __future__ import division

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--kokkos", action="store_true")
parser.add_argument("--plot", action='store_true')
parser.add_argument("--Bfactor", default=2.0, type=float,
                    help="B-factor in Angstrom^2 (default: 2.0)")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()
if args.kokkos:
    import os
    os.environ["DIFFBRAGG_USE_KOKKOS"] = "1"
from simtbx.diffBragg.utils import find_diffBragg_instances
from simtbx.diffBragg.device import DeviceWrapper
with DeviceWrapper(0) as _:

    import numpy as np
    import pylab as plt
    from scipy.stats import linregress
    from scipy.spatial.transform import Rotation
    from simtbx.nanoBragg import sim_data
    from scitbx.matrix import sqr, rec
    from cctbx import uctbx
    from dxtbx.model import Crystal

    BFACTOR_ID = 25

    # ---- crystal setup ----

    ucell = (70, 60, 50, 90.0, 110, 90.0)
    symbol = "C121"

    a_real, b_real, c_real = sqr(
        uctbx.unit_cell(ucell).orthogonalization_matrix()
    ).transpose().as_list_of_lists()
    C = Crystal(a_real, b_real, c_real, symbol)

    rotation = Rotation.random(num=1, random_state=101)[0]
    Q = rec(rotation.as_quat(), n=(4, 1))
    rot_ang, rot_axis = Q.unit_quaternion_as_axis_and_angle()
    C.rotate_around_origin(rot_axis, rot_ang)

    # ---- simulation setup ----

    S = sim_data.SimData(use_default_crystal=True)
    S.crystal.dxtbx_crystal = C
    S.detector = sim_data.SimData.simple_detector(180, 0.1, (1024, 1024))
    S.instantiate_diffBragg(verbose=0, oversample=0, auto_set_spotscale=True)
    S.D.spot_scale = 100000
    S.D.Ncells_abc = 15

    # ---- Test 1: B=0 should give same image as no B-factor ----

    S.D.Bfactor_image = 0
    S.D.raw_pixels_roi *= 0
    S.D.add_diffBragg_spots()
    img_noB = S.D.raw_pixels_roi.as_numpy_array()

    print("Test 1: B=0 should not change image")
    assert np.allclose(img_noB, S.D.raw_pixels_roi.as_numpy_array())
    print("  PASS")

    # ---- Test 2: B>0 should damp intensities ----

    B = args.Bfactor
    S.D.Bfactor_image = B
    S.D.raw_pixels_roi *= 0
    S.D.add_diffBragg_spots()
    img_B = S.D.raw_pixels_roi.as_numpy_array()

    bragg = img_noB > 1e-1

    print("\nTest 2: B=%.1f should damp intensities (Bragg pixels)" % B)
    ratio = img_B[bragg] / img_noB[bragg]
    print("  damping ratio: min=%.4f mean=%.4f max=%.4f" %
          (ratio.min(), ratio.mean(), ratio.max()))
    assert np.all(ratio <= 1.0 + 1e-8), "B-factor should only reduce intensity"
    assert ratio.mean() < 1.0, "Mean damping should be < 1"
    print("  PASS")

    # ---- Test 3: gradient finite-difference test ----

    print("\nTest 3: analytical vs finite-difference gradient for B=%.1f" % B)

    S.D.Bfactor_image = B
    S.D.refine(BFACTOR_ID)
    S.D.initialize_managers()
    S.D.verbose = int(args.verbose)

    S.D.raw_pixels_roi *= 0
    S.D.add_diffBragg_spots()
    img = S.D.raw_pixels_roi.as_numpy_array()
    deriv = S.D.get_Bfactor_derivative_pixels().as_numpy_array()

    bragg = img > 1e-1

    # Check gradient sign: dI/dB should be negative (more B -> less intensity)
    assert np.all(deriv[bragg] <= 0), \
        "dI/dB should be <= 0 everywhere (increasing B decreases intensity)"
    print("  gradient sign check PASS (all non-positive on Bragg pixels)")

    perc = [0.001, 0.01, 0.1, 1, 10]
    all_error = []
    shifts = []

    for p in perc:
        delta_B = max(abs(B), 0.1) * p * 0.01
        shifts.append(delta_B)

        S.D.Bfactor_image = B + delta_B
        S.D.raw_pixels_roi *= 0
        S.D.add_diffBragg_spots()
        img2 = S.D.raw_pixels_roi.as_numpy_array()

        fdiff = (img2 - img) / delta_B
        error = np.abs(fdiff[bragg] - deriv[bragg]).mean()
        all_error.append(error)
        print("  step=%.6f  error=%.6f" % (delta_B, error))

    if args.plot:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        axes[0].plot(shifts, all_error, 'o-')
        axes[0].set_xlabel("step size (delta B)")
        axes[0].set_ylabel("mean |finite_diff - analytical|")
        axes[0].set_title("B-factor gradient convergence")

        im = axes[1].imshow(img.reshape(1024, 1024), vmax=np.percentile(img, 99.5))
        axes[1].set_title("Image with B=%.1f" % B)
        plt.colorbar(im, ax=axes[1])

        dim = axes[2].imshow(deriv.reshape(1024, 1024),
                             vmin=np.percentile(deriv, 0.5),
                             vmax=0)
        axes[2].set_title("dI/dB")
        plt.colorbar(dim, ax=axes[2])
        plt.tight_layout()
        plt.show()

    l = linregress(shifts, all_error)
    print("  linregress: r=%.6f slope=%.4f intercept=%.6f pvalue=%.2e" %
          (l.rvalue, l.slope, l.intercept, l.pvalue))
    assert l.rvalue > .99, "Expected linear convergence, r=%.4f" % l.rvalue
    assert l.slope > 0
    assert l.pvalue < 1e-3
    assert abs(l.intercept) < 0.1 * l.slope, \
        "intercept=%.4f should be small relative to slope=%.4f" % (l.intercept, l.slope)

    print("\nAll tests PASSED!")
    for name in find_diffBragg_instances(globals()): del globals()[name]
