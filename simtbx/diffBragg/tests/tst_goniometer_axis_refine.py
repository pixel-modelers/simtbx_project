"""
Test goniometer axis refinement using spherical parameterization.

Tests:
1. Spherical <-> Cartesian conversions
2. Effect of axis perturbations on diffraction pattern
3. Analytical vs finite-difference gradient check (theta and phi)
4. GoniometerParameters class initialization
"""
from __future__ import division, print_function
import numpy as np
import sys
import os

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--plot", action='store_true', help="Show plots")
parser.add_argument("--verbose", action='store_true', help="Verbose output")
args = parser.parse_args()

# Test 1: Spherical parameterization conversions
print("="*80)
print("TEST 1: Spherical <-> Cartesian Conversions")
print("="*80)

from simtbx.diffBragg.refiners.geometry import GoniometerParameters

test_axes = [
    (1, 0, 0),   # X-axis (typical synchrotron)
    (0, 1, 0),   # Y-axis
    (0, 0, 1),   # Z-axis
    (1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)),  # (1,1,1) direction
    (0.5, 0.5, np.sqrt(0.5)),  # Random direction
]

print("\nTesting roundtrip conversion:")
max_error = 0.0
for axis in test_axes:
    # Cartesian -> Spherical -> Cartesian
    theta, phi = GoniometerParameters.cartesian_to_spherical(axis)
    axis_recovered = GoniometerParameters.spherical_to_cartesian(theta, phi)

    error = np.sqrt(sum((a1 - a2)**2 for a1, a2 in zip(axis, axis_recovered)))
    max_error = max(max_error, error)

    if args.verbose:
        print(f"  Input:     {axis}")
        print(f"  θ={theta:.6f}, φ={phi:.6f}")
        print(f"  Recovered: {axis_recovered}")
        print(f"  Error:     {error:.2e}")
        print()

assert max_error < 1e-10, f"Roundtrip error too large: {max_error}"
print(f"✓ Roundtrip conversion passed (max error: {max_error:.2e})")

# Test 2: Axis perturbation effects on diffraction
print("\n" + "="*80)
print("TEST 2: Axis Perturbation Effects")
print("="*80)

from simtbx.nanoBragg import sim_data
from simtbx.diffBragg import utils
from simtbx.diffBragg.device import DeviceWrapper

with DeviceWrapper(0) as _:
    # Setup simulation
    S = sim_data.SimData(use_default_crystal=True)
    det_shape = (512, 512)  # Smaller for faster testing
    S.detector = sim_data.SimData.simple_detector(180, 0.1, det_shape)
    S.instantiate_diffBragg(verbose=0, oversample=0, auto_set_spotscale=True)
    S.D.spot_scale = 100000

    # Rotation settings
    delta_phi = 0.5  # degrees
    nphi_steps = 10

    # Reference axis (1,0,0) typical synchrotron
    axis_ref = (1.0, 0.0, 0.0)
    theta_ref, phi_ref = GoniometerParameters.cartesian_to_spherical(axis_ref)

    # Generate reference image
    utils.update_SIM_with_gonio(S, delta_phi=delta_phi,
                                num_phi_steps=nphi_steps,
                                spindle_axis=axis_ref)
    S.D.add_diffBragg_spots()
    img_ref = S.D.raw_pixels_roi.as_numpy_array()
    S.D.raw_pixels_roi *= 0
    S.D.raw_pixels *= 0

    print(f"\nReference axis: {axis_ref}")
    print(f"  θ = {theta_ref:.6f} rad ({np.degrees(theta_ref):.2f}°)")
    print(f"  φ = {phi_ref:.6f} rad ({np.degrees(phi_ref):.2f}°)")
    print(f"  Reference image sum: {img_ref.sum():.2e}")

    # Test small perturbations in theta and phi
    perturbations = [
        ('theta', 0.01),  # 0.01 radians ~ 0.57 degrees
        ('phi', 0.01),
        ('theta', 0.05),  # 0.05 radians ~ 2.87 degrees
        ('phi', 0.05),
    ]

    print("\nTesting perturbations:")
    for param, delta in perturbations:
        if param == 'theta':
            theta_pert = theta_ref + delta
            phi_pert = phi_ref
        else:  # phi
            theta_pert = theta_ref
            phi_pert = phi_ref + delta

        axis_pert = GoniometerParameters.spherical_to_cartesian(theta_pert, phi_pert)

        utils.update_SIM_with_gonio(S, delta_phi=delta_phi,
                                    num_phi_steps=nphi_steps,
                                    spindle_axis=axis_pert)
        S.D.add_diffBragg_spots()
        img_pert = S.D.raw_pixels_roi.as_numpy_array()
        S.D.raw_pixels_roi *= 0
        S.D.raw_pixels *= 0

        # Measure change in diffraction pattern
        diff = np.abs(img_pert - img_ref)
        rmsd = np.sqrt(np.mean(diff**2))
        rel_change = diff.sum() / img_ref.sum()

        print(f"  Δ{param}={delta:.4f}: RMSD={rmsd:.2e}, rel_change={rel_change:.4f}")

        # Should see measurable change for significant perturbations
        if abs(delta) >= 0.05:
            assert rel_change > 0.001, f"No significant change for Δ{param}={delta}"

print("\n✓ Axis perturbations show expected effects on diffraction")

# Test 3: Analytical vs Finite-Difference Gradient Check
print("\n" + "="*80)
print("TEST 3: Analytical vs Finite-Difference Gradient Check")
print("="*80)

from scipy import stats
from simtbx.nanoBragg.nanoBragg_crystal import NBcrystal
from scitbx.matrix import sqr
from cctbx import uctbx
from dxtbx.model import Crystal

GONIO_THETA_ID = 27
GONIO_PHI_ID = 28

with DeviceWrapper(0) as _:
    # Set up crystal
    ucell = (70, 60, 50, 90.0, 110, 90.0)
    symbol = "C121"
    a_real, b_real, c_real = sqr(
        uctbx.unit_cell(ucell).orthogonalization_matrix()
    ).transpose().as_list_of_lists()
    C = Crystal(a_real, b_real, c_real, symbol)
    nbcryst = NBcrystal()
    nbcryst.dxtbx_crystal = C
    nbcryst.n_mos_domains = 1
    nbcryst.thick_mm = 0.01
    nbcryst.Ncells_abc = (7, 7, 7)

    SIM = sim_data.SimData(use_default_crystal=True)
    SIM.detector = sim_data.SimData.simple_detector(180, 0.1, (512, 512))
    SIM.crystal = nbcryst
    SIM.instantiate_diffBragg(oversample=0, verbose=0, interpolate=0,
                              default_F=1e3, auto_set_spotscale=True)
    D = SIM.D
    D.spot_scale = 100000
    #D.verbose = 2  # show kernel timing

    # Set up rotation (goniometer)
    delta_phi = 0.5  # degrees
    nphi_steps = 10

    # Use an off-axis direction so both theta and phi have non-trivial gradients
    # Avoid pure x-axis where phi derivative vanishes (theta=pi/2, phi=0 is a
    # singularity for phi perturbation since sin(theta) terms cancel)
    axis_ref = (0.5, 0.5, np.sqrt(0.5))  # ~45 deg from z
    axis_ref = tuple(np.array(axis_ref) / np.linalg.norm(axis_ref))
    theta_ref, phi_ref = GoniometerParameters.cartesian_to_spherical(axis_ref)

    print(f"\nReference axis: ({axis_ref[0]:.4f}, {axis_ref[1]:.4f}, {axis_ref[2]:.4f})")
    print(f"  θ = {theta_ref:.6f} rad ({np.degrees(theta_ref):.2f}°)")
    print(f"  φ = {phi_ref:.6f} rad ({np.degrees(phi_ref):.2f}°)")
    print(f"  Kernel: {'GPU' if os.environ.get('DIFFBRAGG_USE_CUDA') else 'CPU'}")

    test3_pass = True

    for param_name, refine_id in [("theta", GONIO_THETA_ID), ("phi", GONIO_PHI_ID)]:
        print(f"\n--- Testing d(image)/d({param_name}) ---")

        # Set spindle axis and gonio params
        utils.update_SIM_with_gonio(SIM, delta_phi=delta_phi,
                                    num_phi_steps=nphi_steps,
                                    spindle_axis=axis_ref)

        # Enable refinement
        D.refine(refine_id)
        D.initialize_managers()

        # Compute baseline image + analytical derivatives
        D.raw_pixels_roi *= 0
        D.raw_pixels *= 0
        D.add_diffBragg_spots()
        if param_name == "theta":  # only print once
            gpu = D.last_kernel_on_GPU
            print(f"  last_kernel_on_GPU: {gpu}")
            D.verbose = 0  # quiet for remaining calls
        img0 = D.raw_pixels_roi.as_numpy_array()

        # Get analytical derivatives (returns tuple: (theta_deriv, phi_deriv))
        derivs = D.get_gonio_axis_derivative_pixels()
        if param_name == "theta":
            deriv_ana = derivs[0].as_numpy_array()
        else:
            deriv_ana = derivs[1].as_numpy_array()

        # Bragg mask: pixels with significant signal
        bragg = img0 > 1

        n_bragg = bragg.sum()
        deriv_max = np.abs(deriv_ana[bragg]).max() if n_bragg > 0 else 0
        print(f"  Bragg pixels: {n_bragg}")
        print(f"  Max |analytical deriv|: {deriv_max:.2e}")
        print(f"  Image sum: {img0.sum():.2e}")

        # Finite-difference at multiple perturbation sizes
        delta_vals = [0.0005 + i*0.0005 for i in range(8)]
        error_vals = []
        delta_h = []

        for delta in delta_vals:
            # Perturb the spherical angle
            if param_name == "theta":
                theta_pert = theta_ref + delta
                phi_pert = phi_ref
            else:
                theta_pert = theta_ref
                phi_pert = phi_ref + delta

            # Convert perturbed angles to Cartesian axis
            axis_pert = GoniometerParameters.spherical_to_cartesian(theta_pert, phi_pert)

            # Update spindle axis and re-simulate
            D.spindle_axis = axis_pert
            D.raw_pixels_roi *= 0
            D.raw_pixels *= 0
            D.add_diffBragg_spots()
            img_pert = D.raw_pixels_roi.as_numpy_array()

            # Forward finite difference
            fdiff = (img_pert - img0) / delta

            # Mean absolute error on Bragg pixels
            error = np.abs(fdiff[bragg] - deriv_ana[bragg]).mean()
            error_vals.append(error)
            delta_h.append(delta)

            if args.verbose:
                print(f"  Δ{param_name}={delta:.4f} rad: "
                      f"mean|fdiff-ana|={error:.4e}")

        # Restore original axis
        D.spindle_axis = axis_ref

        # Turn off refinement for this param
        D.fix(refine_id)

        # Validate: error should scale linearly with step size (first-order convergence)
        l = stats.linregress(delta_h, error_vals)
        print(f"  Linear regression: r={l.rvalue:.6f}, slope={l.slope:.4e}, "
              f"intercept={l.intercept:.4e}, p={l.pvalue:.2e}")

        ok_r = l.rvalue > 0.999
        ok_slope = l.slope > 0
        ok_p = l.pvalue < 1e-4
        ok_intercept = abs(l.intercept) < 0.5 * abs(l.slope * delta_h[0])

        if ok_r and ok_slope and ok_p:
            print(f"  ✓ d(image)/d({param_name}) PASSED "
                  f"(r={l.rvalue:.6f})")
        else:
            print(f"  ✗ d(image)/d({param_name}) FAILED")
            if not ok_r:
                print(f"    r-value {l.rvalue:.6f} < 0.999")
            if not ok_slope:
                print(f"    slope {l.slope:.4e} <= 0")
            if not ok_p:
                print(f"    p-value {l.pvalue:.2e} >= 1e-4")
            test3_pass = False

        if args.plot:
            import pylab as plt
            plt.figure()
            plt.plot(delta_h, error_vals, 'o-')
            plt.xlabel(f"Δ{param_name} (rad)")
            plt.ylabel("Mean |finite_diff - analytical|")
            plt.title(f"Gradient convergence: {param_name}")
            plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
            plt.tight_layout()
            plt.show()

    assert test3_pass, "Gradient check failed — see details above"

print("\n✓ Analytical gradients match finite differences")

# Test 4: Parameter initialization from geometry.py
print("\n" + "="*80)
print("TEST 4: GoniometerParameters Class")
print("="*80)

from simtbx.diffBragg import phil as diffBragg_phil

# Create mock SIM with spindle_axis
class MockD:
    def __init__(self, axis):
        self.spindle_axis = axis

class MockSIM:
    def __init__(self, axis):
        self.D = MockD(axis)

# Test with different initial axes
test_cases = [
    ((1, 0, 0), "X-axis (typical)"),
    ((0, 1, 0), "Y-axis"),
    ((0, 0, 1), "Z-axis"),
]

for axis, desc in test_cases:
    mock_sim = MockSIM(axis)

    # Create phil params
    params = diffBragg_phil.phil_scope.extract()
    params.geometry.fix.gonio_axis = False
    params.geometry.sigma_gonio_axis = 0.01

    # Initialize GoniometerParameters with SIM
    gonio_params = GoniometerParameters(params, SIM=mock_sim)

    assert len(gonio_params.parameters) == 2, "Should have 2 parameters (theta, phi)"

    theta_param = gonio_params.parameters[0]
    phi_param = gonio_params.parameters[1]

    assert theta_param.name == "gonio_theta"
    assert phi_param.name == "gonio_phi"
    assert not theta_param.fix  # Should be free (fix.gonio_axis=False)
    assert not phi_param.fix

    # Check initialization
    theta_init = theta_param.init
    phi_init = phi_param.init
    axis_recovered = GoniometerParameters.spherical_to_cartesian(theta_init, phi_init)

    error = np.sqrt(sum((a1 - a2)**2 for a1, a2 in zip(axis, axis_recovered)))

    print(f"\n  {desc}: {axis}")
    print(f"    Initialized θ={theta_init:.6f}, φ={phi_init:.6f}")
    print(f"    Recovered axis: {axis_recovered}")
    print(f"    Error: {error:.2e}")

    assert error < 1e-10, f"Initialization error for {desc}: {error}"

print("\n✓ GoniometerParameters initialization passed")

# Summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print("✓ Test 1: Spherical <-> Cartesian conversions PASSED")
print("✓ Test 2: Axis perturbation effects PASSED")
print("✓ Test 3: Analytical vs finite-difference gradients PASSED")
print("✓ Test 4: GoniometerParameters class PASSED")
print("\nAll tests passed! OK")
