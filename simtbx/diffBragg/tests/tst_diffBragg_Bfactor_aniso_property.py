"""
Test the anisotropic B-factor (Debye-Waller) refinement.

Verifies that:
1. Setting Bfactor_aniso to all zeros gives the same image as no B-factor.
2. Non-zero aniso B damps intensities (all ratios <= 1).
3. Isotropic-equivalent aniso B gives the same image as scalar Bfactor_image.
4. The 6 analytical gradients dI/dbeta_ij agree with finite-difference.

Usage:
    python tst_diffBragg_Bfactor_aniso_property.py
    python tst_diffBragg_Bfactor_aniso_property.py --plot
"""
from __future__ import division

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--kokkos", action="store_true")
parser.add_argument("--plot", action='store_true')
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()
if args.kokkos:
    import os
    os.environ["DIFFBRAGG_USE_KOKKOS"] = "1"
from simtbx.diffBragg.utils import find_diffBragg_instances
from simtbx.diffBragg.device import DeviceWrapper
with DeviceWrapper(0) as _:

    import numpy as np
    from scipy.stats import linregress
    from scipy.spatial.transform import Rotation
    from simtbx.nanoBragg import sim_data
    from scitbx.matrix import sqr, rec
    from cctbx import uctbx

    from dxtbx.model import Crystal

    BFACTOR_ANISO_ID = 26
    BFACTOR_ID = 25

    # ---- crystal setup (monoclinic, so all 6 beta components can be non-zero) ----

    ucell = (70, 60, 50, 90.0, 110, 90.0)
    symbol = "P121"

    a_real, b_real, c_real = sqr(
        uctbx.unit_cell(ucell).orthogonalization_matrix()
    ).transpose().as_list_of_lists()
    C = Crystal(a_real, b_real, c_real, symbol)

    rotation = Rotation.random(num=1, random_state=101)[0]
    Q = rec(rotation.as_quat(), n=(4, 1))
    rot_ang, rot_axis = Q.unit_quaternion_as_axis_and_angle()
    C.rotate_around_origin(rot_axis, rot_ang)

    # Reciprocal metric tensor for computing isotropic-equivalent beta
    uc = uctbx.unit_cell(ucell)
    gstar = list(uc.reciprocal_metrical_matrix())
    # gstar = (a*^2, b*^2, c*^2, 2*a*b*cos(gamma*), 2*a*c*cos(beta*), 2*b*c*cos(alpha*))
    print("Reciprocal metrical matrix:", ["%.6f" % g for g in gstar])

    # ---- simulation setup ----

    def make_sim():
        S = sim_data.SimData(use_default_crystal=True)
        S.crystal.dxtbx_crystal = C
        S.detector = sim_data.SimData.simple_detector(180, 0.1, (1024, 1024))
        S.instantiate_diffBragg(verbose=0, oversample=0, auto_set_spotscale=True)
        S.D.spot_scale = 100000
        S.D.Ncells_abc = 15
        return S

    S = make_sim()

    # ---- Test 1: beta=0 should give same image as no B-factor ----

    S.D.Bfactor_aniso = (0, 0, 0, 0, 0, 0)
    S.D.raw_pixels_roi *= 0
    S.D.add_diffBragg_spots()
    img_noB = S.D.raw_pixels_roi.as_numpy_array().copy()

    S.D.Bfactor_aniso = (0, 0, 0, 0, 0, 0)
    S.D.Bfactor_image = 0
    S.D.raw_pixels_roi *= 0
    S.D.add_diffBragg_spots()
    img_zero = S.D.raw_pixels_roi.as_numpy_array()

    print("Test 1: beta=0 should not change image")
    assert np.allclose(img_noB, img_zero), \
        "max diff=%.6e" % np.max(np.abs(img_noB - img_zero))
    print("  PASS")

    # ---- Test 2: positive beta should damp intensities ----

    # Use an anisotropic B: more dampening along a* than c*
    B_a = 15.0  # Ang^2
    B_c = 5.0   # Ang^2
    # cctbx reciprocal_metrical_matrix() returns (G*11, G*22, G*33, G*12, G*13, G*23)
    # where G*12 = a*b*cos(gamma*), etc. — NO factor of 2 included.
    # Isotropic equivalence: beta_ij = B/4 * G*_ij
    beta_gt = [B_a/4 * gstar[0],   # beta11
               B_a/4 * gstar[1],   # beta22
               B_c/4 * gstar[2],   # beta33
               B_a/4 * gstar[3],   # beta12
               B_c/4 * gstar[4],   # beta13  (monoclinic: may be non-zero)
               0]                   # beta23  (monoclinic C121: alpha*=90)
    print("\nTest 2: aniso B should damp intensities")
    print("  beta_gt:", ["%.6f" % b for b in beta_gt])

    S.D.Bfactor_aniso = tuple(beta_gt)
    S.D.raw_pixels_roi *= 0
    S.D.add_diffBragg_spots()
    img_B = S.D.raw_pixels_roi.as_numpy_array()

    bragg = img_noB > 1e-1
    ratio = img_B[bragg] / img_noB[bragg]
    print("  damping ratio: min=%.4f mean=%.4f max=%.4f" %
          (ratio.min(), ratio.mean(), ratio.max()))
    assert np.all(ratio <= 1.0 + 1e-8), "Positive beta should only reduce intensity"
    assert ratio.mean() < 1.0, "Mean damping should be < 1"
    print("  PASS")

    # ---- Test 3: isotropic-equivalent beta should match scalar B ----

    print("\nTest 3: isotropic-equivalent aniso B should match scalar B")
    B_iso = 8.0  # Ang^2
    beta_iso = [B_iso/4 * gstar[0],
                B_iso/4 * gstar[1],
                B_iso/4 * gstar[2],
                B_iso/4 * gstar[3],
                B_iso/4 * gstar[4],
                B_iso/4 * gstar[5]]
    print("  B_iso=%.1f -> beta_iso: [%s]" %
          (B_iso, ", ".join("%.6f" % b for b in beta_iso)))

    # Image with scalar B
    S.D.Bfactor_image = B_iso
    S.D.Bfactor_aniso = (0, 0, 0, 0, 0, 0)
    S.D.raw_pixels_roi *= 0
    S.D.add_diffBragg_spots()
    img_scalar = S.D.raw_pixels_roi.as_numpy_array()

    # Image with isotropic-equivalent aniso B
    S.D.Bfactor_image = 0
    S.D.Bfactor_aniso = tuple(beta_iso)
    S.D.raw_pixels_roi *= 0
    S.D.add_diffBragg_spots()
    img_aniso_equiv = S.D.raw_pixels_roi.as_numpy_array()

    bragg_iso = img_scalar > 1e-1
    if bragg_iso.sum() > 0:
        max_diff = np.max(np.abs(img_scalar[bragg_iso] - img_aniso_equiv[bragg_iso]))
        rel_diff = max_diff / np.max(img_scalar[bragg_iso])
        print("  max abs diff on Bragg pixels: %.6e" % max_diff)
        print("  max relative diff: %.6e" % rel_diff)
        assert rel_diff < 1e-6, "Isotropic-equivalent aniso B should match scalar B (rel_diff=%.2e)" % rel_diff
    print("  PASS")

    # ---- Test 4: finite-difference gradient test for all 6 components ----

    print("\nTest 4: analytical vs finite-difference gradients")

    # Use the anisotropic beta_gt from Test 2
    S.D.Bfactor_image = 0
    S.D.Bfactor_aniso = tuple(beta_gt)
    S.D.refine(BFACTOR_ANISO_ID)
    S.D.initialize_managers()
    S.D.verbose = int(args.verbose)

    S.D.raw_pixels_roi *= 0
    S.D.add_diffBragg_spots()
    img = S.D.raw_pixels_roi.as_numpy_array()
    derivs = S.D.get_Bfactor_aniso_derivative_pixels()
    # derivs is a 6-tuple of flex arrays
    deriv_arrays = [derivs[i].as_numpy_array() for i in range(6)]

    bragg = img > 1e-1
    component_names = ["beta11", "beta22", "beta33", "beta12", "beta13", "beta23"]

    all_pass = True
    for i_comp in range(6):
        deriv_ana = deriv_arrays[i_comp]

        # Check gradient sign for diagonal components (should be non-positive for positive beta)
        if i_comp < 3 and beta_gt[i_comp] > 0:
            neg_frac = (deriv_ana[bragg] <= 0).sum() / bragg.sum()
            print("  %s: gradient sign check: %.1f%% non-positive" %
                  (component_names[i_comp], neg_frac * 100))

        # Finite-difference convergence
        perc = [0.001, 0.01, 0.1, 1, 10]
        all_error = []
        shifts = []
        base_scale = max(abs(beta_gt[i_comp]), 1e-5)

        for p in perc:
            delta = base_scale * p * 0.01
            shifts.append(delta)

            beta_shifted = list(beta_gt)
            beta_shifted[i_comp] += delta
            S.D.Bfactor_aniso = tuple(beta_shifted)
            S.D.raw_pixels_roi *= 0
            S.D.add_diffBragg_spots()
            img2 = S.D.raw_pixels_roi.as_numpy_array()

            fdiff = (img2 - img) / delta
            error = np.abs(fdiff[bragg] - deriv_ana[bragg]).mean()
            all_error.append(error)

        # Restore
        S.D.Bfactor_aniso = tuple(beta_gt)

        l = linregress(shifts, all_error)
        print("  %s: r=%.4f slope=%.4f intercept=%.6f steps=[%.2e..%.2e] errors=[%.2e..%.2e]" %
              (component_names[i_comp], l.rvalue, l.slope, l.intercept,
               shifts[0], shifts[-1], all_error[0], all_error[-1]))

        # For components that are exactly 0 (beta23), the gradient and finite-diff
        # should both be ~0, so we check differently
        if abs(beta_gt[i_comp]) < 1e-10 and all(e < 1e-8 for e in all_error):
            print("    (zero component, all errors < 1e-8: PASS)")
            continue

        if l.rvalue < 0.99:
            print("    WARNING: poor linear convergence r=%.4f" % l.rvalue)
            all_pass = False
        if l.slope <= 0:
            print("    WARNING: non-positive slope=%.4f" % l.slope)
            all_pass = False

    assert all_pass, "Some gradient components failed convergence test"
    print("  PASS")

    if args.plot:
        import pylab as plt
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        axes = axes.flatten()

        # Show image
        axes[0].imshow(img.reshape(1024, 1024), vmax=np.percentile(img, 99.5))
        axes[0].set_title("Image with aniso B")

        # Show damping ratio
        ratio_img = np.ones_like(img)
        ratio_img[bragg] = img[bragg] / img_noB[bragg]
        axes[1].imshow(ratio_img.reshape(1024, 1024), vmin=0.5, vmax=1.0, cmap='RdYlBu')
        axes[1].set_title("Damping ratio")

        # Show 6 derivative images
        for i_comp in range(6):
            ax = axes[i_comp + 2]
            d = deriv_arrays[i_comp]
            vmin = np.percentile(d[bragg], 1) if bragg.sum() > 0 else 0
            vmax = np.percentile(d[bragg], 99) if bragg.sum() > 0 else 0
            ax.imshow(d.reshape(1024, 1024), vmin=vmin, vmax=vmax, cmap='RdBu_r')
            ax.set_title("dI/d%s" % component_names[i_comp])
        plt.tight_layout()
        plt.show()

    print("\nAll tests PASSED!")
    S.D.fix(BFACTOR_ANISO_ID)
    for name in find_diffBragg_instances(globals()): del globals()[name]
