"""
Test for multi-image rotation crystallography refinement.

Simulates multiple rotation frames, perturbs the crystal, then jointly
refines shared crystal orientation and per-image scale factors.
"""
from __future__ import division
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--kokkos", action="store_true")
parser.add_argument("--perturb", choices=["crystal", "Nabc", "G", "cholesky"],
                    type=str, nargs="+", default=["crystal"])
parser.add_argument("--num-frames", type=int, default=5)
parser.add_argument("--plot", action="store_true")
args = parser.parse_args()
name = "rotation_refine_%s" % "-".join(args.perturb)

import os
if args.kokkos:
    os.environ["DIFFBRAGG_USE_KOKKOS"] = "1"

from simtbx.diffBragg.utils import find_diffBragg_instances
from simtbx.diffBragg.device import DeviceWrapper
with DeviceWrapper(0) as _:

    from dxtbx.model.crystal import Crystal
    from cctbx import uctbx
    from scitbx.matrix import sqr, col
    import numpy as np
    from scipy.spatial.transform import Rotation
    from simtbx.nanoBragg.nanoBragg_crystal import NBcrystal
    from simtbx.nanoBragg.sim_data import SimData
    from simtbx.diffBragg import utils, hopper_utils
    from dxtbx.model import Experiment
    from simtbx.nanoBragg import make_imageset
    from simtbx.diffBragg.phil import hopper_phil, philz
    from libtbx.phil import parse
    from simtbx.diffBragg.stage_two_utils import PAR_from_params
    from scitbx.matrix import rec

    phil_scope = parse(hopper_phil + philz)

    ucell = (55, 65, 75, 90, 95, 90)
    symbol = "P121"
    NUM_FRAMES = args.num_frames
    osc_width = 0.5  # degrees per frame

    # Ground-truth mosaic block parameters
    # For cholesky test, use off-diagonal Ndef to test full matrix recovery
    if "cholesky" in args.perturb:
        NCELLS_GT = (14, 12, 10)
        NDEF_GT = (2.0, 1.5, 1.0)  # off-diagonal: Nd, Ne, Nf
    else:
        NCELLS_GT = (12, 12, 11)
        NDEF_GT = (0, 0, 0)

    # Phi angles for each frame
    phi_angles = {}
    for i in range(NUM_FRAMES):
        phi_angles[i] = i * osc_width

    # ===================================================
    # 1. Create ground-truth crystal with random rotation
    # ===================================================
    rotation = Rotation.random(num=1, random_state=100)[0]
    a_real, b_real, c_real = sqr(
        uctbx.unit_cell(ucell).orthogonalization_matrix()
    ).transpose().as_list_of_lists()
    C = Crystal(a_real, b_real, c_real, symbol)
    # Apply a large random rotation to the crystal
    rot = Rotation.random(num=1, random_state=100)[0]
    Q_quat = rec(rot.as_quat(), n=(4, 1))
    rot_ang, rot_axis = Q_quat.unit_quaternion_as_axis_and_angle()
    C.rotate_around_origin(rot_axis, rot_ang)

    # Create perturbed crystal (what we start refinement from)
    C2 = Crystal(a_real, b_real, c_real, symbol)
    C2.rotate_around_origin(rot_axis, rot_ang)
    np.random.seed(1)
    perturb_rot_axis = np.random.random(3)
    perturb_rot_axis /= np.linalg.norm(perturb_rot_axis)
    perturb_rot_ang = 0.15  # degree perturbation
    C2.rotate_around_origin(col(perturb_rot_axis), perturb_rot_ang)

    # ===================================================
    # 2. Simulate ground-truth rotation frames
    # ===================================================
    nbcryst = NBcrystal()
    nbcryst.dxtbx_crystal = C
    nbcryst.thick_mm = 0.1
    nbcryst.isotropic_ncells = False
    nbcryst.Ncells_abc = NCELLS_GT

    SIM = SimData(use_default_crystal=True)
    detdist = 150
    shape = 513, 512
    SIM.detector = SimData.simple_detector(detdist, 0.1, shape)
    SIM.crystal = nbcryst
    SIM.instantiate_diffBragg(oversample=0, auto_set_spotscale=True)
    SIM.D.default_F = 0
    SIM.D.F000 = 0
    SIM.D.progress_meter = False
    SIM.water_path_mm = 0.005
    SIM.air_path_mm = 0.1
    SIM.add_air = True
    SIM.add_Water = True
    SIM.include_noise = True
    SIM.D.verbose = 0

    # Set off-diagonal Ndef for cholesky ground truth
    if "cholesky" in args.perturb:
        SIM.D.Ncells_def = NDEF_GT

    GT_spot_scale = SIM.D.spot_scale
    phisteps = 50

    print("=" * 60)
    print("Ground truth parameters:")
    print("  Nabc (diagonal): %s" % (NCELLS_GT,))
    print("  Ndef (off-diag): %s" % (NDEF_GT,))
    print("  N matrix:")
    Na_gt, Nb_gt, Nc_gt = NCELLS_GT
    Nd_gt, Ne_gt, Nf_gt = NDEF_GT
    print("    [[%6.2f  %6.2f  %6.2f]" % (Na_gt, Nd_gt, Nf_gt))
    print("     [%6.2f  %6.2f  %6.2f]" % (Nd_gt, Nb_gt, Ne_gt))
    print("     [%6.2f  %6.2f  %6.2f]]" % (Nf_gt, Ne_gt, Nc_gt))
    print("  Scale (G): %.4f" % GT_spot_scale)
    print("=" * 60)

    # Simulate each frame at different phi angles
    frame_images = []
    frame_spots = []
    for i_frame in range(NUM_FRAMES):
        SIM.D.raw_pixels *= 0
        phi_deg = phi_angles[i_frame]

        # Set goniometer for this frame
        utils.update_SIM_with_gonio(SIM, delta_phi=osc_width,
                                    num_phi_steps=phisteps)
        SIM.D.phi_deg = phi_deg

        SIM.D.add_diffBragg_spots()
        spots = SIM.D.raw_pixels.as_numpy_array()
        frame_spots.append(spots.copy())

        SIM._add_background()
        SIM.D.readout_noise_adu = 0
        SIM._add_noise()
        img = SIM.D.raw_pixels.as_numpy_array()
        frame_images.append(img)

    if args.plot:
        import pylab as plt
        fig, axes = plt.subplots(1, NUM_FRAMES, figsize=(4 * NUM_FRAMES, 4))
        if NUM_FRAMES == 1:
            axes = [axes]
        for i, img in enumerate(frame_images):
            axes[i].imshow(img, vmax=100)
            axes[i].set_title("phi=%.1f deg" % phi_angles[i])
        plt.suptitle("Ground truth rotation frames")
        plt.tight_layout()
        plt.show()

    # ===================================================
    # 3. Create DataModelers - one per frame
    # ===================================================
    P = phil_scope.extract()
    P.roi.shoebox_size = 20
    P.relative_tilt = False
    P.roi.fit_tilt = False
    P.roi.pad_shoebox_for_background_estimation = 10
    P.roi.reject_edge_reflections = False
    P.refiner.sigma_r = SIM.D.readout_noise_adu
    P.refiner.adu_per_photon = SIM.D.quantum_gain
    P.simulator.init_scale = 1
    P.simulator.beam.size_mm = SIM.beam.size_mm
    P.simulator.total_flux = SIM.D.flux
    P.use_restraints = False
    P.sigmas.RotXYZ = [1, 1, 1]
    P.ftol = 1e-10

    # Save structure factors for loading
    mtz_name = name + ".mtz"
    SIM.crystal.miller_array.as_mtz_dataset(
        column_root_label="F").mtz_object().write(mtz_name)
    P.simulator.structure_factors.mtz_name = mtz_name
    P.simulator.structure_factors.mtz_column = "F(+),F(-)"

    # Configure perturbation
    if "G" in args.perturb:
        P.init.G = GT_spot_scale * 5
    else:
        P.init.G = GT_spot_scale

    if "Nabc" in args.perturb:
        P.init.Nabc = [20, 20, 20]
    elif "cholesky" in args.perturb:
        # Start from diagonal approximation (no off-diagonal terms)
        # Use the GT diagonal values so the only thing to recover is off-diag
        P.init.Nabc = list(NCELLS_GT)
        P.init.Ndef = [0, 0, 0]
    else:
        P.init.Nabc = list(NCELLS_GT)

    if "crystal" in args.perturb:
        xtal_for_refine = C2
    else:
        xtal_for_refine = C

    # Compute initial Cholesky L from the init Nabc (diagonal only)
    init_Na, init_Nb, init_Nc = P.init.Nabc
    init_Nd, init_Ne, init_Nf = P.init.Ndef
    init_L11 = np.sqrt(max(init_Na, 0.01))
    init_L22 = np.sqrt(max(init_Nb, 0.01))
    init_L33 = np.sqrt(max(init_Nc, 0.01))
    # For diagonal init, off-diag L elements are 0
    init_L21 = init_L31 = init_L32 = 0

    print("\nInitial (perturbed) parameters for refinement:")
    print("  init.Nabc: %s" % (tuple(P.init.Nabc),))
    print("  init.Ndef: %s" % (tuple(P.init.Ndef),))
    print("  N matrix (initial):")
    print("    [[%6.2f  %6.2f  %6.2f]" % (init_Na, init_Nd, init_Nf))
    print("     [%6.2f  %6.2f  %6.2f]" % (init_Nd, init_Nb, init_Ne))
    print("     [%6.2f  %6.2f  %6.2f]]" % (init_Nf, init_Ne, init_Nc))
    print("  Cholesky L (initial):")
    print("    [[%6.3f  %6.3f  %6.3f]" % (init_L11, 0, 0))
    print("     [%6.3f  %6.3f  %6.3f]" % (init_L21, init_L22, 0))
    print("     [%6.3f  %6.3f  %6.3f]]" % (init_L31, init_L32, init_L33))
    print("  Scale (G init): %.4f  (GT: %.4f)" % (P.init.G, GT_spot_scale))
    if "crystal" in args.perturb:
        print("  Crystal: perturbed by %.2f deg" % perturb_rot_ang)
    else:
        print("  Crystal: ground truth (not perturbed)")
    print()

    # Create one DataModeler per frame
    data_modelers = {}
    for i_frame in range(NUM_FRAMES):
        E = Experiment()
        E.crystal = xtal_for_refine
        E.detector = SIM.detector
        E.beam = SIM.D.beam
        E.imageset = make_imageset([frame_images[i_frame]], E.beam, E.detector)

        # Find reflections from the spots-only image
        refls = utils.refls_from_sims(
            [frame_spots[i_frame]], E.detector, E.beam, thresh=18)
        if len(refls) == 0:
            print("Frame %d: no reflections found, skipping" % i_frame)
            continue
        utils.refls_to_q(refls, E.detector, E.beam, update_table=True)
        utils.refls_to_hkl(refls, E.detector, E.beam, E.crystal,
                           update_table=True)

        Modeler = hopper_utils.DataModeler(P)
        assert Modeler.GatherFromExperiment(E, refls, sg_symbol=symbol)

        # Set up PAR (initial parameters) for this modeler
        Modeler.PAR = PAR_from_params(P, E)
        Modeler.ucell_man = utils.manager_from_crystal(E.crystal)

        # Set per-frame attributes needed by rotation_model
        Modeler.spectra = SIM.beam.spectrum
        Modeler.osc_deg = osc_width
        Modeler.phisteps = phisteps

        # Set up ROI slices
        Modeler.set_slices("roi_id")

        data_modelers[i_frame] = Modeler
        print("Frame %d (phi=%.1f deg): %d reflections, %d pixels"
              % (i_frame, phi_angles[i_frame], len(refls),
                 len(Modeler.all_data)))

    assert len(data_modelers) >= 2, \
        "Need at least 2 frames with reflections, got %d" % len(data_modelers)

    # ===================================================
    # 4. Set up SIM for refinement
    # ===================================================
    refine_E = Experiment()
    refine_E.crystal = xtal_for_refine
    refine_E.detector = SIM.detector
    refine_E.beam = SIM.D.beam

    SIM_refine = utils.simulator_for_refinement(refine_E, P)
    SIM_refine.D.device_Id = 0

    # ===================================================
    # 5. Refine jointly
    # ===================================================
    from simtbx.diffBragg.refiners.rotation_refine import rotation_refine

    print("\n" + "=" * 60)
    print("Starting rotation refinement with %d frames" % len(data_modelers))
    print("Perturbations: %s" % ", ".join(args.perturb))
    print("=" * 60)

    Xopt, LMP, target = rotation_refine(
        data_modelers, SIM_refine, P, phi_angles,
        use_restraints=False, max_iter=100, ftol=1e-10)

    # ===================================================
    # 6. Extract and print results
    # ===================================================
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    # Extract optimized RotXYZ
    rotX_p = LMP["RotXYZ0"]
    rotY_p = LMP["RotXYZ1"]
    rotZ_p = LMP["RotXYZ2"]
    rotX = rotX_p.get_val(Xopt[rotX_p.xpos])
    rotY = rotY_p.get_val(Xopt[rotY_p.xpos])
    rotZ = rotZ_p.get_val(Xopt[rotZ_p.xpos])
    print("Optimized RotXYZ (deg): %.6f, %.6f, %.6f"
          % (rotX * 180 / np.pi, rotY * 180 / np.pi, rotZ * 180 / np.pi))

    # Extract optimized Cholesky → N matrix
    L11 = LMP["chol_L11"].get_val(Xopt[LMP["chol_L11"].xpos])
    L21 = LMP["chol_L21"].get_val(Xopt[LMP["chol_L21"].xpos])
    L22 = LMP["chol_L22"].get_val(Xopt[LMP["chol_L22"].xpos])
    L31 = LMP["chol_L31"].get_val(Xopt[LMP["chol_L31"].xpos])
    L32 = LMP["chol_L32"].get_val(Xopt[LMP["chol_L32"].xpos])
    L33 = LMP["chol_L33"].get_val(Xopt[LMP["chol_L33"].xpos])
    Na_opt = L11**2
    Nb_opt = L21**2 + L22**2
    Nc_opt = L31**2 + L32**2 + L33**2
    Nd_opt = L11 * L21
    Ne_opt = L21 * L31 + L22 * L32
    Nf_opt = L11 * L31
    print("\nOptimized Cholesky L:")
    print("    [[%6.3f  %6.3f  %6.3f]" % (L11, 0, 0))
    print("     [%6.3f  %6.3f  %6.3f]" % (L21, L22, 0))
    print("     [%6.3f  %6.3f  %6.3f]]" % (L31, L32, L33))
    print("Optimized N matrix (L^T * L):")
    print("    [[%6.2f  %6.2f  %6.2f]" % (Na_opt, Nd_opt, Nf_opt))
    print("     [%6.2f  %6.2f  %6.2f]" % (Nd_opt, Nb_opt, Ne_opt))
    print("     [%6.2f  %6.2f  %6.2f]]" % (Nf_opt, Ne_opt, Nc_opt))
    print("Ground truth N matrix:")
    print("    [[%6.2f  %6.2f  %6.2f]" % (Na_gt, Nd_gt, Nf_gt))
    print("     [%6.2f  %6.2f  %6.2f]" % (Nd_gt, Nb_gt, Ne_gt))
    print("     [%6.2f  %6.2f  %6.2f]]" % (Nf_gt, Ne_gt, Nc_gt))

    # Extract optimized per-shot scales
    print("\nPer-shot scale factors:")
    for i_shot in sorted(data_modelers.keys()):
        scale_p = LMP["shot%d_Scale" % i_shot]
        scale = scale_p.get_val(Xopt[scale_p.xpos])
        print("  Frame %d: %.4f  (GT: %.4f, init: %.4f)"
              % (i_shot, scale, GT_spot_scale, scale_p.init))

    # ===================================================
    # 7. Validate
    # ===================================================
    print("\n" + "=" * 60)
    print("Validation")
    print("=" * 60)

    # Reconstruct optimized crystal
    ucman = utils.manager_from_crystal(xtal_for_refine)
    ucpar = ucman.unit_cell_parameters
    Copt = hopper_utils.new_cryst_from_rotXYZ_and_ucell(
        (rotX, rotY, rotZ), ucpar, xtal_for_refine)

    # Compare with ground truth
    misset_opt, misset_init = utils.compare_with_ground_truth(
        *C.get_real_space_vectors(),
        dxcryst_models=[Copt, xtal_for_refine],
        symbol=symbol)

    print("Initial misorientation: %.6f deg" % misset_init)
    print("Optimized misorientation: %.6f deg" % misset_opt)

    if "crystal" in args.perturb:
        assert misset_opt < misset_init, \
            "Optimized misset (%.4f) should be less than initial (%.4f)" % (
                misset_opt, misset_init)
        assert misset_opt < 0.01, \
            "Crystal misorientation should be < 0.01 deg, got %.6f" % misset_opt

    # Check per-image scale factors
    if "G" in args.perturb:
        for i_shot in data_modelers:
            scale_p = LMP["shot%d_Scale" % i_shot]
            scale = scale_p.get_val(Xopt[scale_p.xpos])
            perc_diff = abs(GT_spot_scale - scale) / GT_spot_scale * 100
            print("Frame %d scale percent diff: %.2f%%" % (i_shot, perc_diff))
            assert perc_diff < 5, \
                "Scale factor too far from GT: %.2f%%" % perc_diff

    # Check Nabc recovery
    if "Nabc" in args.perturb:
        print("Nabc optimized: (%.2f, %.2f, %.2f) GT: %s"
              % (Na_opt, Nb_opt, Nc_opt, NCELLS_GT))
        assert all(abs(opt - gt) < 2 for opt, gt in
                   zip([Na_opt, Nb_opt, Nc_opt], NCELLS_GT)), \
            "Nabc too far from GT: (%.2f,%.2f,%.2f) vs %s" % (
                Na_opt, Nb_opt, Nc_opt, NCELLS_GT)

    # Check Cholesky (off-diagonal) recovery
    if "cholesky" in args.perturb:
        print("Ndef optimized: (%.2f, %.2f, %.2f) GT: %s"
              % (Nd_opt, Ne_opt, Nf_opt, NDEF_GT))
        # The diagonal should still be close
        assert all(abs(opt - gt) < 2 for opt, gt in
                   zip([Na_opt, Nb_opt, Nc_opt], NCELLS_GT)), \
            "Nabc (diagonal) too far from GT"
        # The off-diagonal should have moved toward GT from zero
        init_ndef_err = sum(abs(gt) for gt in NDEF_GT)  # initial error (started at 0)
        opt_ndef_err = sum(abs(opt - gt) for opt, gt in
                          zip([Nd_opt, Ne_opt, Nf_opt], NDEF_GT))
        print("Ndef initial error: %.2f, optimized error: %.2f" %
              (init_ndef_err, opt_ndef_err))
        assert opt_ndef_err < init_ndef_err, \
            "Off-diagonal Ndef should improve: init_err=%.2f opt_err=%.2f" % (
                init_ndef_err, opt_ndef_err)

    print("\nOK - Rotation refinement test passed!")

    # Cleanup
    del SIM_refine.D
    del SIM.D
    for name_var in find_diffBragg_instances(globals()):
        del globals()[name_var]
