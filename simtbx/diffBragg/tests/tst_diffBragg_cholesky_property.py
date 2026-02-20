"""
Test the Cholesky reparameterization of NABC = L^T * L.

Verifies that:
1. The chain-rule gradients dI/dLij (computed from kernel dI/dNa..dI/dNf)
   agree with finite-difference gradients.
2. The Cholesky decomposition correctly maps 6 lower-triangular elements
   to a positive-definite NABC matrix.

Usage:
    python tst_diffBragg_cholesky_property.py --idx 0   # test L11
    python tst_diffBragg_cholesky_property.py --idx 3   # test L31
    python tst_diffBragg_cholesky_property.py --plot     # show convergence
"""
from __future__ import division

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--kokkos", action="store_true")
parser.add_argument("--plot", action='store_true')
parser.add_argument("--idx", default=0, type=int, choices=range(6),
                    help="0=L11, 1=L21, 2=L22, 3=L31, 4=L32, 5=L33")
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

    # ---- helpers ----

    def cholesky_to_nabc(L11, L21, L22, L31, L32, L33):
        """NABC = L^T * L  (L lower-triangular)"""
        Na = L11*L11
        Nb = L21*L21 + L22*L22
        Nc = L31*L31 + L32*L32 + L33*L33
        Nd = L11*L21
        Ne = L21*L31 + L22*L32
        Nf = L11*L31
        return Na, Nb, Nc, Nd, Ne, Nf

    def cholesky_chain_rule(dI_dNa, dI_dNb, dI_dNc, dI_dNd, dI_dNe, dI_dNf,
                            L11, L21, L22, L31, L32, L33):
        """dI/dLij via chain rule from kernel derivatives."""
        dI_dL = [
            dI_dNa * 2*L11 + dI_dNd * L21 + dI_dNf * L31,     # dI/dL11
            dI_dNd * L11 + dI_dNb * 2*L21 + dI_dNe * L31,     # dI/dL21
            dI_dNb * 2*L22 + dI_dNe * L32,                      # dI/dL22
            dI_dNf * L11 + dI_dNe * L21 + dI_dNc * 2*L31,     # dI/dL31
            dI_dNe * L22 + dI_dNc * 2*L32,                      # dI/dL32
            dI_dNc * 2*L33,                                      # dI/dL33
        ]
        return dI_dL

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
    S.D.isotropic_ncells = False

    # Enable both diagonal and off-diagonal Ncells kernel gradients
    Ncells_id = 9
    Ncells_def_id = 21
    S.D.refine(Ncells_id)
    S.D.refine(Ncells_def_id)
    S.D.initialize_managers()
    S.D.verbose = int(args.verbose)

    # ---- Cholesky parameters ----
    # L is lower-triangular: L = [[L11, 0, 0], [L21, L22, 0], [L31, L32, L33]]
    L_GT = [4.0, 1.5, 3.5, 0.8, -0.5, 3.2]  # L11, L21, L22, L31, L32, L33
    L11, L21, L22, L31, L32, L33 = L_GT
    Na, Nb, Nc, Nd, Ne, Nf = cholesky_to_nabc(*L_GT)

    chol_names = ["L11", "L21", "L22", "L31", "L32", "L33"]
    print("Cholesky params: %s" % dict(zip(chol_names, L_GT)))
    print("NABC: Na=%.2f Nb=%.2f Nc=%.2f Nd=%.2f Ne=%.2f Nf=%.2f" % (Na, Nb, Nc, Nd, Ne, Nf))

    # Verify positive-definiteness
    Nmat = np.array([[Na, Nd, Nf], [Nd, Nb, Ne], [Nf, Ne, Nc]])
    eigvals = np.linalg.eigvalsh(Nmat)
    assert np.all(eigvals > 0), "NABC matrix should be positive definite, eigenvalues=%s" % eigvals
    print("NABC eigenvalues: %s  (all positive -- good)" % eigvals)

    # ---- compute reference image and analytical gradients ----

    S.D.set_ncells_values((Na, Nb, Nc))
    S.D.Ncells_def = Nd, Ne, Nf
    S.D.raw_pixels_roi *= 0
    S.D.add_diffBragg_spots()
    img = S.D.raw_pixels_roi.as_numpy_array()

    # Get kernel gradients w.r.t. Na..Nf
    Nabc_grads = S.D.get_ncells_derivative_pixels()
    Ndef_grads = S.D.get_ncells_def_derivative_pixels()
    dI_dNa = Nabc_grads[0].as_numpy_array()
    dI_dNb = Nabc_grads[1].as_numpy_array()
    dI_dNc = Nabc_grads[2].as_numpy_array()
    dI_dNd = Ndef_grads[0].as_numpy_array()
    dI_dNe = Ndef_grads[1].as_numpy_array()
    dI_dNf = Ndef_grads[2].as_numpy_array()

    # Chain-rule to get dI/dLij
    dI_dL_all = cholesky_chain_rule(dI_dNa, dI_dNb, dI_dNc, dI_dNd, dI_dNe, dI_dNf,
                                     L11, L21, L22, L31, L32, L33)

    # ---- finite-difference test for selected Cholesky element ----

    i_chol = args.idx
    print("\nTesting Cholesky element %s (idx=%d)" % (chol_names[i_chol], i_chol))

    deriv_analytical = dI_dL_all[i_chol]
    bragg = img > 1e-1

    perc = [0.001, 0.01, 0.1, 1, 10]
    all_error = []
    shifts = []
    L_val = L_GT[i_chol]

    for p in perc:
        delta_L = max(abs(L_val), 1.0) * p * 0.01
        shifts.append(delta_L)

        # perturbed Cholesky
        L_pert = list(L_GT)
        L_pert[i_chol] = L_val + delta_L
        Na2, Nb2, Nc2, Nd2, Ne2, Nf2 = cholesky_to_nabc(*L_pert)

        S.D.set_ncells_values((Na2, Nb2, Nc2))
        S.D.Ncells_def = Nd2, Ne2, Nf2
        S.D.raw_pixels_roi *= 0
        S.D.add_diffBragg_spots()
        img2 = S.D.raw_pixels_roi.as_numpy_array()

        fdiff = (img2 - img) / delta_L
        error = np.abs(fdiff[bragg] - deriv_analytical[bragg]).mean()
        all_error.append(error)
        print("  step=%.6f  error=%.6f" % (delta_L, error))

    if args.plot:
        plt.figure()
        plt.plot(shifts, all_error, 'o-')
        plt.xlabel("step size")
        plt.ylabel("mean |finite_diff - analytical|")
        plt.title("Cholesky %s gradient convergence" % chol_names[i_chol])
        plt.show()

    l = linregress(shifts, all_error)
    print("  linregress: r=%.6f slope=%.4f intercept=%.6f pvalue=%.2e" %
          (l.rvalue, l.slope, l.intercept, l.pvalue))
    assert l.rvalue > .99, "Expected linear convergence, r=%.4f" % l.rvalue
    assert l.slope > 0
    assert l.pvalue < 1e-3
    assert abs(l.intercept) < 0.1 * l.slope, \
        "intercept=%.4f should be small relative to slope=%.4f" % (l.intercept, l.slope)

    print("OK!")
    for name in find_diffBragg_instances(globals()): del globals()[name]
