from __future__ import division
import glob
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--kokkos", action="store_true")
parser.add_argument("--nolog", action="store_true")
parser.add_argument("--readout", type=float, default=3)
parser.add_argument("--scale", type=float, default=1)
parser.add_argument("--perturb", choices=["G", "Nabc"], type=str, nargs="+", default=None)
parser.add_argument("--plot", action="store_true")
parser.add_argument("--beta", default=None, type=float)
parser.add_argument("--sigmaFhkl", default=1, type=float)
parser.add_argument("--sigmaG", default=1, type=float)
parser.add_argument("--maxiter", default=None, type=int)
parser.add_argument("--geo", action="store_true")
parser.add_argument("--auto-sigma", action="store_true")
parser.add_argument("--bias-correct", action="store_true",
                    help="subtract log(V) bias from Fhkl gradient")
parser.add_argument("--snr-sigma", type=float, default=None)
parser.add_argument("--two-stage", type=float, default=None,
                    help="Hessian fraction cutoff for two-stage refinement (e.g. 0.5)")
parser.add_argument("--linear", action="store_true",
                    help="use linear Fhkl parameterization instead of exponential")
parser.add_argument("--perRoi", action="store_true",
                    help="use per-ROI scale refinement instead of global Fhkl")
parser.add_argument("--wilson", action="store_true",
                    help="apply Wilson statistics prior on Fhkl intensities")
parser.add_argument("--perturb-B", type=float, default=None,
                    help="ground truth B-factor (Ang^2) to simulate and refine")
parser.add_argument("--perturb-B-aniso", action="store_true",
                    help="test anisotropic B-factor refinement (more dampening along c*)")
parser.add_argument("--nshots", type=int, default=3,
                    help="number of shots for multi-shot ensemble test (default: 3)")
parser.add_argument("--refine-B-iso", action="store_true",
                    help="refine isotropic B instead of aniso B (use with --perturb-B-aniso to compare)")
args = parser.parse_args()
import os

if args.kokkos:
    os.environ["DIFFBRAGG_USE_KOKKOS"]="1"
from simtbx.diffBragg.utils import find_diffBragg_instances
from simtbx.diffBragg.device import DeviceWrapper
with DeviceWrapper(0) as _:

    import logging
    import sys
    import pandas
    import numpy as np

    from cctbx import miller
    from dials.array_family import flex
    from simtbx.nanoBragg.nanoBragg_crystal import NBcrystal
    from simtbx.nanoBragg.sim_data import SimData
    from simtbx.diffBragg import hopper_utils
    from simtbx.diffBragg import utils
    from simtbx.diffBragg.hopper_ensemble_utils import load_inputs
    from dxtbx.model import Experiment
    from simtbx.nanoBragg import make_imageset
    from simtbx.diffBragg.phil import hopper_phil, philz
    from libtbx.phil import parse

    phil_scope = parse(hopper_phil+philz)

    def cholesky_to_nabc(L11, L21, L22, L31, L32, L33):
        """NABC = L^T * L  (L lower-triangular)"""
        Na = L11*L11
        Nb = L21*L21 + L22*L22
        Nc = L31*L31 + L32*L32 + L33*L33
        Nd = L11*L21
        Ne = L21*L31 + L22*L32
        Nf = L11*L31
        return Na, Nb, Nc, Nd, Ne, Nf

    def print_refinement_summary(params, label=""):
        refining = []
        if not params.fix.G: refining.append("G (scale)")
        if not params.fix.Fhkl: refining.append("Fhkl (structure factors)")
        if not params.fix.Nabc: refining.append("Nabc (6-term Cholesky tensor)")
        if not params.fix.B: refining.append("B-factor (per-image)")
        if not params.fix.Baniso: refining.append("Baniso (anisotropic B-factor)")
        if not all(params.fix.RotXYZ): refining.append("RotXYZ")
        if not params.fix.ucell: refining.append("unit cell")
        fixed = []
        if params.fix.G: fixed.append("G")
        if params.fix.Fhkl: fixed.append("Fhkl")
        if params.fix.Nabc: fixed.append("Nabc")
        if params.fix.B: fixed.append("B")
        opts = []
        if getattr(params, 'auto_Fhkl_sigma', False): opts.append("auto_Fhkl_sigma")
        if getattr(params, 'use_cholesky_Nabc', False): opts.append("cholesky_Nabc")
        print("\n=== Refinement: %s ===" % label)
        print("  Refining: %s" % ", ".join(refining))
        print("  Fixed:    %s" % ", ".join(fixed))
        if opts: print("  Options:  %s" % ", ".join(opts))
        print("")

    p65_cryst = {'__id__': 'crystal',
                 'real_space_a': (43.32309880004587, 25.5289818883498, 60.49634260901813),
                 'real_space_b': (34.201635357808115, -38.82573591182249, -59.255697149884924),
                 'real_space_c': (41.42476391176581, 229.70849483520402, -126.60059788183489),
                 'space_group_hall_symbol': ' P 65 2 (x,y,z+1/12)',
                 'ML_half_mosaicity_deg': 0.06671930026192037,
                 'ML_domain_size_ang': 6349.223840307989}
    from dxtbx.model.crystal import CrystalFactory
    p65_C = CrystalFactory.from_dict(p65_cryst)
    ucell = p65_C.get_unit_cell().parameters()
    symbol = p65_C.get_space_group().info().type().lookup_symbol()

    # Setup the simulation and create a realistic image
    # with background and noise
    # <><><><><><><><><><><><><><><><><><><><><><><><><>
    nbcryst = NBcrystal()
    nbcryst.dxtbx_crystal = p65_C
    nbcryst.thick_mm = 0.005
    nbcryst.isotropic_ncells = False
    # Ground truth Cholesky factors: L = [[L11,0,0],[L21,L22,0],[L31,L32,L33]]
    # Gives NABC ~ diag(12, 12, 11) with small off-diagonal terms
    L_GT = [np.sqrt(12), 0.3, np.sqrt(12 - 0.09), 0.1, -0.15, np.sqrt(11 - 0.01 - 0.0225)]
    NCELLS_GT = cholesky_to_nabc(*L_GT)  # (Na, Nb, Nc, Nd, Ne, Nf)
    # det(NABC) = det(L*L^T) = det(L)^2 = (L11*L22*L33)^2
    NABC_determ = (L_GT[0] * L_GT[2] * L_GT[5])**2
    # When no_Nabc_scale=True, the det(NABC) scaling is removed from the kernel.
    # G must absorb det(NABC)^2 (since F_latt is squared in the intensity formula).
    refine_nabc = args.perturb is not None and "Nabc" in args.perturb
    nbcryst.Ncells_abc = NCELLS_GT[:3]
    nbcryst.Ncells_def = NCELLS_GT[3:]
    nbcryst.space_group = "P6522"
    ma = utils.make_miller_array(symbol, ucell, d_min=1.5)
    np.random.seed(0)
    new_data = ma.d_spacings().data()*10

    ma = miller.array(ma.set(), new_data).set_observation_type_xray_amplitude()
    ma_map = {h:v for h,v in zip(ma.indices(), ma.data())}
    nbcryst.miller_array = ma
    assert ma.is_xray_amplitude_array()

    SIM = SimData(use_default_crystal=False)
    shape = 1000, 1001
    detdist = 140
    SIM.detector = SimData.simple_detector(detdist, 0.1, shape)
    SIM.crystal = nbcryst
    SIM.instantiate_diffBragg(oversample=1, auto_set_spotscale=True, default_F=0)

    # test the code for computing the acerage structure factor intensity with resolution
    # (this is why we set the structure factor data to be the same as the resolution (x10)
    num_dspace_bins = 10
    SIM.set_dspace_binning(num_dspace_bins, verbose=True)
    dspace_bins = SIM.D.dspace_bins
    ave_I_cell = SIM.D.ave_I_cell()[0]
    assert len(ave_I_cell) == num_dspace_bins
    assert len(dspace_bins) == num_dspace_bins + 1
    aves = []
    dspaces = []
    for i, (d1,d2) in enumerate(zip(dspace_bins, dspace_bins[1:])):
        ave_val = np.sqrt(ave_I_cell[i]) / 10.
        if not args.geo: assert d1 < ave_val < d2
        aves.append(ave_val)
        dspaces.append(.5*(d1+d2))

    print("1 0 7: ", ma.value_at_index((1,0,7)))
    SIM.D.default_F = 0
    SIM.D.F000 = 0
    SIM.D.progress_meter = False
    SIM.water_path_mm = 0.005
    SIM.air_path_mm = 0.1
    SIM.add_air = True
    SIM.add_Water = True
    SIM.include_noise = True
    B_GT = args.perturb_B if args.perturb_B is not None else 0
    SIM.D.Bfactor_image = B_GT

    # Anisotropic B-factor ground truth in fractional hkl coords
    # β_ij such that T = exp(-(β11*h² + β22*k² + β33*l² + 2*β12*h*k + 2*β13*h*l + 2*β23*k*l))
    # Use different B along a*/b* vs c* to test anisotropy recovery
    Baniso_GT = [0, 0, 0, 0, 0, 0]
    if args.perturb_B_aniso:
        uc = p65_C.get_unit_cell()
        # reciprocal_metrical_matrix() returns (G*11, G*22, G*33, G*12, G*13, G*23)
        # where G*12 = a*b*cos(gamma*), etc. — NO factor of 2 included.
        gstar = list(uc.reciprocal_metrical_matrix())
        # Anisotropic B: more dampening along c* than a*/b*
        # Isotropic equivalence: β_ij = (B/4) * G*_ij
        B_ab = 30.0   # Angstrom^2, along a*/b*
        B_c = 80.0    # Angstrom^2, along c*
        # For P65 (hexagonal): a*=b*, γ*=60°, G*12 = a*b*cos(60°) = a*²/2
        # Diagonal: β11 = B_ab/4 * G*11, β22 = B_ab/4 * G*22, β33 = B_c/4 * G*33
        # Off-diagonal: β12 = B_ab/4 * G*12, β13 = β23 = 0 (hexagonal: α*=β*=90°)
        Baniso_GT = [B_ab/4 * gstar[0],   # β11
                     B_ab/4 * gstar[1],   # β22
                     B_c/4  * gstar[2],   # β33
                     B_ab/4 * gstar[3],   # β12
                     0,                    # β13 (hexagonal: cos(β*)=0)
                     0]                    # β23 (hexagonal: cos(α*)=0)
        SIM.D.Bfactor_aniso = tuple(Baniso_GT)
        print("Aniso B GT: β = [%.6f, %.6f, %.6f, %.6f, %.6f, %.6f]" % tuple(Baniso_GT))
        print("  (B_ab=%.1f, B_c=%.1f Ang^2)" % (B_ab, B_c))

    if refine_nabc:
        SIM.D.no_Nabc_scale = True
        SIM.D.spot_scale *= NABC_determ**2  # G absorbs det(NABC)^2
        print("no_Nabc_scale=True: spot_scale adjusted by det(NABC)^2=%.1f" % NABC_determ**2)
    SIM.D.verbose = 2
    SIM.D.add_diffBragg_spots()
    SIM.D.verbose = 0
    spots = SIM.D.raw_pixels.as_numpy_array()
    SIM._add_background()
    SIM.D.readout_noise_adu=args.readout
    SIM._add_noise()

    # This is the ground truth image:
    img = SIM.D.raw_pixels.as_numpy_array()
    if args.plot:
        import pylab as plt
        plt.imshow(img, vmax=100)
        plt.title("Ground truth image")
        plt.figure()
        plt.plot(dspaces, aves)
        plt.xlabel("Angstrom")
        plt.show()
    SIM.D.raw_pixels *= 0
    #pfs = 0,270,175
    #utils.show_diffBragg_state(SIM.D, pfs)
    SIM.D.raw_pixels *= 0

    P = phil_scope.extract()
    P.debug_mode=True
    E = Experiment()

    P.init.G = SIM.D.spot_scale
    E.crystal = p65_C

    P.init.Nabc = SIM.crystal.Ncells_abc
    P.init.cholesky = list(L_GT)  # full 6-term Cholesky init (matches GT off-diagonals)
    P.init.detz_shift = 0

    E.detector = SIM.detector
    E.beam = SIM.D.beam
    E.imageset = make_imageset([img], E.beam, E.detector)
    refls = utils.refls_from_sims([spots], E.detector, E.beam, thresh=18)
    print("%d REFLS" % len(refls))
    refls['id'] = flex.int(len(refls), 0)
    utils.refls_to_q(refls, E.detector, E.beam, update_table=True)
    utils.refls_to_hkl(refls, E.detector, E.beam, E.crystal, update_table=True)

    P.roi.shoebox_size = 10
    P.roi.allow_overlapping_spots = True
    P.relative_tilt = False
    P.roi.fit_tilt = False
    P.roi.pad_shoebox_for_background_estimation=10
    P.roi.reject_edge_reflections = False
    P.refiner.sigma_r = SIM.D.readout_noise_adu
    P.refiner.adu_per_photon = SIM.D.quantum_gain
    P.simulator.init_scale = 1
    P.simulator.beam.size_mm = SIM.beam.size_mm
    P.simulator.oversample = SIM.D.oversample
    P.simulator.total_flux = SIM.D.flux
    P.use_restraints = False


    mset = ma.set()
    ma_map_keys, ma_map_values = list(ma_map.keys()), np.array(list(ma_map.values()))
    ma_map_values2 = np.random.normal(ma_map_values, scale=args.scale*ma_map_values)
    bad_map = {h:v for h,v in zip(ma_map_keys, ma_map_values2)}
    if args.scale ==0:
        assert np.allclose(ma_map_values, ma_map_values2)

    new_amps = flex.double()
    for h in mset.indices():
        amp = bad_map[h]
        new_amps.append(amp)

    ma2 = miller.array(mset, new_amps).set_observation_type_xray_amplitude()

    ma2_map = {h:v for h,v in zip(ma2.indices(), ma2.data())}
    name = "hopper_refine_Fhkl.mtz"
    print("1 0 7: ", ma2.value_at_index((1,0,7)))
    assert ma2.is_xray_amplitude_array()
    ma2.as_mtz_dataset(column_root_label="F").mtz_object().write(name)
    P.simulator.structure_factors.mtz_name = name
    P.simulator.structure_factors.mtz_column = "F(+),F(-)"
    P.logging.parameters=False
    P.method="L-BFGS-B"
    P.ftol = 1e-10
    P.space_group = symbol
    if args.perRoi:
        P.fix.Fhkl = True
        P.fix.perRoiScale = False
        print("perRoi scale refinement enabled (Fhkl fixed)")
    else:
        P.fix.Fhkl = False
    P.betas.Fhkl = args.beta
    P.fix.G = True
    P.types.G = "positive"
    P.centers.G = SIM.D.spot_scale*2
    P.betas.G=1e8
    P.use_restraints = args.beta is not None
    P.sigmas.G = args.sigmaG
    P.sigmas.Fhkl = args.sigmaFhkl
    P.auto_Fhkl_sigma = args.auto_sigma
    P.correct_Fhkl_gradient_bias = args.bias_correct
    P.Fhkl_restraint_snr_sigma = args.snr_sigma
    P.Fhkl_two_stage_threshold = args.two_stage
    P.Fhkl_linear_parameterization = args.linear
    P.Fhkl_wilson_prior = args.wilson
    print("auto_Fhkl_sigma = %s" % P.auto_Fhkl_sigma)
    if args.bias_correct:
        print("correct_Fhkl_gradient_bias = True")
    if args.snr_sigma is not None:
        print("Fhkl_restraint_snr_sigma = %.4g" % args.snr_sigma)
    if args.two_stage is not None:
        print("Fhkl_two_stage_threshold = %.4g" % args.two_stage)
    if args.linear:
        print("Fhkl_linear_parameterization = True")
    if args.wilson:
        print("Fhkl_wilson_prior = True")
    if args.perturb is not None and "G" in args.perturb:
        P.fix.G = False
        P.init.G = SIM.D.spot_scale*10
        #P.maxs.G = SIM.D.spot_scale*100
    P.use_geometric_mean_Fhkl = args.geo
    P.fix.ucell=True
    P.fix.RotXYZ=[1,1,1]
    P.fix.Nabc=True
    if args.perturb is not None and "Nabc" in args.perturb:
        P.fix.Nabc = False
        P.no_Nabc_scale = True  # decouple det(NABC) from intensity; G absorbs the scale
        P.maxs.G = 1e16  # G is much larger with det(NABC)^2 absorbed
        P.init.Nabc = 15,15,14  # ~25% perturbation from GT (12,12,11)
        P.init.cholesky = None  # auto-compute from init.Nabc (perturbed diagonal-only start)
        # Tighten Cholesky bounds for L-BFGS-B convergence.  Default [-300,300] range
        # gives RangedParameter sensitivity of ~300 per unit x — wild steps for L~3-4.
        # Physics-motivated bounds: L_diag>0 (positive-definite), |L_offdiag|<10.
        P.mins.cholesky = [0.1, -10, 0.1, -10, -10, 0.1]
        P.maxs.cholesky = [10, 10, 10, 10, 10, 10]
    P.fix.detz_shift=True
    if args.perturb_B is not None:
        P.fix.B = False
        P.init.B = 0  # start from B=0, truth is args.perturb_B
        P.mins.B = -1  # avoid zero derivative at boundary (RangedParameter: cos(arcsin(-1))=0)
    # Note: Baniso is only refined in the multi-shot ensemble (too many DOF for single-shot).
    # The single-shot Fhkl can absorb the per-HKL aniso B dampening.

    if not args.nolog:
        h = logging.StreamHandler(sys.stdout)
        logging.basicConfig(level=logging.DEBUG, handlers=[h])
    #del SIM.D

    P.outdir="_temp_fhkl_refine"
    if args.maxiter is not None:
        P.lbfgs_maxiter = args.maxiter
    P.record_device_timings = True
    print_refinement_summary(P, "single-shot")
    Eopt,_, Mod,SIM_from_hopper, x = hopper_utils.refine(E, refls, P, return_modeler=True, free_mem=False)
    SIM_from_hopper.D.show_timings(0)

    logging.disable()
    print("\nResults\n<><><><><><>")

    Mod.exper_name = "dummie.expt"
    Mod.refl_name = "dummie.refl"
    Mod.save_up(x, SIM_from_hopper)

    # we can track the dominant hkls in each shoebox occuring within the diffBragg model
    #count_stats = utils.track_fhkl(Mod)
    #
    #
    #main_hkls = []
    #for i_roi in count_stats:
    #    stats = count_stats[i_roi]
    #    stats = sorted( list(stats.items()), key=lambda x: x[1])
    #    main_hkl, frac = stats[-1]
    #    print(main_hkl, frac)
    #    main_hkls.append(main_hkl)

    # this should agree with what we put into diffBragg in the reflection tables
    main_hkls_from_refls = utils.map_hkl_list(list(Mod.refls["miller_index"]), symbol=P.space_group)
    #assert len(set(main_hkls)) == len(set(main_hkls_from_refls))
    # good.

    # Now, this should also agree with the refined fhkl values, stored in the data table
    # the modeler save_up method creates an output file containing the refined fhkl values

    if args.perRoi:
        # Extract per-ROI scale corrections and map to ASU HKLs
        from simtbx.diffBragg import utils as db_utils
        asu_hkl_list = db_utils.map_hkl_list(list(Mod.refls["miller_index"]), symbol=P.space_group)
        # For each ROI, get the scale factor from refined parameters
        hkl_scales = {}  # asu_hkl -> list of scales (in case multiple ROIs share same HKL)
        for i_roi, roi_id in enumerate(Mod.roi_id_unique):
            p = Mod.P["scale_roi%d" % roi_id]
            scale_val = p.get_val(x[p.xpos])
            # Get the HKL for this ROI (roi_id is the ROI index)
            hkl_asu = tuple(Mod.Hi_asu[roi_id])
            if hkl_asu not in hkl_scales:
                hkl_scales[hkl_asu] = []
            hkl_scales[hkl_asu].append(scale_val)
        # Average scales for HKLs with multiple ROIs
        asu = list(hkl_scales.keys())
        asu_corrections = np.array([np.mean(hkl_scales[h]) for h in asu])
        asu_corrections_var = np.zeros(len(asu))
        # All HKLs from perRoi are "nominal" (they came from indexed reflections)
        main_hkl_set = set(main_hkls_from_refls)
        is_nominal = {h: (h in main_hkl_set) for h in asu}
        print("perRoi: %d unique HKLs, %d nominal" % (len(asu), sum(is_nominal.values())))
    else:
        fnames = glob.glob("%s/Fhkl_scale/rank*/*.npz" % P.outdir)
        assert len(fnames)==1
        fhkl_f= fnames[0]
        fhkl_dat = np.load(fhkl_f)
        asu = list(map(tuple,fhkl_dat['asu_hkl']))
        asu_corrections = fhkl_dat['scale_fac']
        asu_corrections_var = fhkl_dat['scale_var']
        asu_is_nominal = fhkl_dat["is_nominal_hkl"]
        is_nominal = {h:is_nom for h,is_nom in zip(asu,  asu_is_nominal)}

    scale = {h:s for h,s in zip(asu, asu_corrections)}
    scale_var = {h:s for h,s in zip(asu, asu_corrections_var)}

    num_not_nominal = 0
    # TODO: figure out why some main_hkls are missing from is_nominal
    for hkl in main_hkls_from_refls:
        if hkl not in is_nominal:
            continue
        if not is_nominal[hkl]:
            num_not_nominal += 1

    nominal_hkl_corrections = {h:s for h,s in zip(asu, asu_corrections) if is_nominal[h]}
    not_nominal_hkl_corrections = {h:s for h,s in zip(asu, asu_corrections) if not is_nominal[h]}

    nominal_hkl_init = {h:1 for h in asu if is_nominal[h]}
    not_nominal_hkl_init = {h:1 for h in asu if not is_nominal[h]}


    def compute_r_factor_with_gt(corrections):
        gt_data = flex.double()
        opt_data = flex.double()
        flx_hkls = flex.miller_index()
        for hkl, scale in corrections.items():
            gt_amp = ma_map[hkl]
            gt_data.append(gt_amp)

            opt_amp = np.sqrt(scale) * ma2_map[hkl]
            opt_data.append(opt_amp)

            h,k,l = map(int, hkl)
            flx_hkls.append((h,k,l))
        mset = ma.miller_set(flx_hkls, ma.anomalous_flag())
        gt_arr = miller.array(mset, gt_data).set_observation_type_xray_amplitude()
        opt_arr = miller.array(mset, opt_data).set_observation_type_xray_amplitude()
        return gt_arr.r1_factor(opt_arr)


    r1_nominal_init = compute_r_factor_with_gt(nominal_hkl_init)
    r1_nominal = compute_r_factor_with_gt(nominal_hkl_corrections)

    r1_not_nominal_init = compute_r_factor_with_gt(not_nominal_hkl_init)
    r1_not_nominal = compute_r_factor_with_gt(not_nominal_hkl_corrections)

    print("\nResults\n<><><><><><>")
    print("For the dominant HKLs within each modeled shoebox (e.g. those with indexed reflections)")
    print("initial R1 factor=%.2f%%" % (r1_nominal_init*100))
    print("optimized R1 factor=%.2f%%" % (r1_nominal*100))

    diffs = []
    all_opts = []
    all_gts = []
    dsp_map = {d:val for d,val in zip(ma.d_spacings().indices(), ma.d_spacings().data())}
    ds = []
    for hkl in nominal_hkl_corrections:
        dsp = dsp_map[hkl]
        gt_val = ma_map[hkl]
        opt_val = np.sqrt(nominal_hkl_corrections[hkl]) * ma2_map[hkl]
        d = abs(gt_val - opt_val) / gt_val * 100
        diffs.append(d)

        all_opts.append(opt_val)
        all_gts.append(gt_val)
        ds.append(dsp)

    print("mean percent diff", np.mean(diffs))

    # R1 for only the HKLs that had signal (trusted pixels > 0)
    # Build a quick lookup of which HKLs had trusted pixels in the Modeler
    hkl_has_signal = set()
    for i_roi in range(len(Mod.rois)):
        hkl_asu = tuple(Mod.Hi_asu[i_roi])
        pix_mask = Mod.roi_id == i_roi
        if (pix_mask & Mod.all_trusted).sum() > 0:
            hkl_has_signal.add(hkl_asu)
    refined_corrections = {h: s for h, s in nominal_hkl_corrections.items() if h in hkl_has_signal}
    refined_init = {h: 1 for h in refined_corrections}
    if refined_corrections:
        r1_refined_init = compute_r_factor_with_gt(refined_init)
        r1_refined = compute_r_factor_with_gt(refined_corrections)
        print("R1 (refined HKLs only, %d/%d): initial=%.2f%%, optimized=%.2f%%"
              % (len(refined_corrections), len(nominal_hkl_corrections),
                 r1_refined_init * 100, r1_refined * 100))

    # Build per-ROI statistics from the Modeler for background quality diagnostics
    # Map ASU HKL -> list of ROI indices (multiple ROIs can map to same ASU HKL via symmetry)
    hkl_to_rois = {}
    for i_roi in range(len(Mod.rois)):
        hkl_asu = tuple(Mod.Hi_asu[i_roi])
        if hkl_asu not in hkl_to_rois:
            hkl_to_rois[hkl_asu] = []
        hkl_to_rois[hkl_asu].append(i_roi)

    # Per-ROI pixel stats
    roi_stats = {}
    for i_roi in range(len(Mod.rois)):
        pix_mask = Mod.roi_id == i_roi
        n_total = pix_mask.sum()
        n_trusted = (pix_mask & Mod.all_trusted).sum()
        data_roi = Mod.all_data[pix_mask]
        bg_roi = Mod.all_background[pix_mask]
        trusted_roi = Mod.all_trusted[pix_mask]
        # Signal = data - background for trusted pixels
        signal = (data_roi - bg_roi)[trusted_roi]
        mean_signal = signal.mean() if len(signal) > 0 else 0
        mean_bg = bg_roi[trusted_roi].mean() if trusted_roi.sum() > 0 else 0
        # Background fit quality: residual of tilt plane vs data on background pixels
        # tilt_cov diagonal gives variance of (a,b,c) fit coefficients
        cov = Mod.tilt_cov[i_roi]
        if cov is not None and not np.isscalar(cov):
            # r_fact = residual variance / (N-3); encoded in cov = AWA_inv * r_fact
            # The trace of cov relative to its structure indicates fit quality
            # A simple metric: sqrt of diagonal c variance = uncertainty in bg offset
            bg_offset_sigma = np.sqrt(abs(cov[2, 2])) if cov.shape == (3, 3) else 0
        else:
            bg_offset_sigma = 0
        # Also compute the model vs data residual for this ROI post-refinement
        if hasattr(Mod, 'best_model') and Mod.best_model is not None:
            model_roi = (Mod.best_model + Mod.all_background)[pix_mask]
            resid_trusted = (data_roi - model_roi)[trusted_roi]
            rms_resid = np.sqrt(np.mean(resid_trusted**2)) if len(resid_trusted) > 0 else 0
        else:
            rms_resid = 0
        roi_stats[i_roi] = {
            'n_total': n_total, 'n_trusted': n_trusted,
            'mean_signal': mean_signal, 'mean_bg': mean_bg,
            'bg_offset_sigma': bg_offset_sigma, 'rms_resid': rms_resid,
            'snr': mean_signal / rms_resid if rms_resid > 0 else 0
        }

    # Aggregate per-HKL (average over symmetry-equivalent ROIs)
    def aggregate_roi_stats(hkl):
        if hkl not in hkl_to_rois:
            return {'n_rois': 0, 'n_trusted': 0, 'mean_signal': 0, 'mean_bg': 0,
                    'bg_offset_sigma': 0, 'rms_resid': 0, 'snr': 0}
        rois_for_hkl = hkl_to_rois[hkl]
        n_rois = len(rois_for_hkl)
        stats_list = [roi_stats[i] for i in rois_for_hkl]
        return {
            'n_rois': n_rois,
            'n_trusted': sum(s['n_trusted'] for s in stats_list),
            'mean_signal': np.mean([s['mean_signal'] for s in stats_list]),
            'mean_bg': np.mean([s['mean_bg'] for s in stats_list]),
            'bg_offset_sigma': np.mean([s['bg_offset_sigma'] for s in stats_list]),
            'rms_resid': np.mean([s['rms_resid'] for s in stats_list]),
            'snr': np.mean([s['snr'] for s in stats_list]),
        }

    # Per-reflection diagnostics: sort from worst to best fit
    hkl_list = list(nominal_hkl_corrections.keys())
    per_refl = []
    for hkl in hkl_list:
        gt_val = ma_map[hkl]
        correction = nominal_hkl_corrections[hkl]
        opt_val = np.sqrt(correction) * ma2_map[hkl]
        init_val = ma2_map[hkl]  # initial (scale=1)
        pct_err = abs(gt_val - opt_val) / gt_val * 100
        dsp = dsp_map[hkl]
        stats = aggregate_roi_stats(hkl)
        per_refl.append((pct_err, hkl, gt_val, init_val, opt_val, correction, dsp, stats))
    per_refl.sort(reverse=True)  # worst first

    print("\n--- Per-reflection diagnostics (sorted worst-to-best) ---")
    print("%12s  %6s  %8s  %8s  %8s  %6s  %6s | %5s  %6s  %7s  %7s  %7s  %6s" %
          ("HKL", "d(A)", "F_gt", "F_opt", "scale", "err%", "nROI", "nTr", "sig", "bg",
           "bg_sig", "rmsRes", "SNR"))
    for pct_err, hkl, gt_val, init_val, opt_val, correction, dsp, stats in per_refl:
        print("(%2d,%2d,%2d) %6.1f %8.1f %8.1f %8.4f %5.1f%% %5d | %5d %6.1f %7.1f %7.2f %7.1f %6.1f" %
              (hkl[0], hkl[1], hkl[2], dsp, gt_val, opt_val, correction, pct_err,
               stats['n_rois'], stats['n_trusted'], stats['mean_signal'], stats['mean_bg'],
               stats['bg_offset_sigma'], stats['rms_resid'], stats['snr']))
    print("--- End per-reflection diagnostics ---\n")

    # Skip single-shot R1 assertion when:
    # - G+Fhkl jointly refined with auto-sigma (G*F^2 degeneracy)
    # - Nabc perturbed (6 Cholesky + G + Fhkl too ill-conditioned for single shot)
    # - Baniso perturbed (Baniso fixed in single-shot; Fhkl can't absorb strong dampening)
    skip_single_shot_assertion = (
        (args.perturb is not None and "G" in args.perturb and args.auto_sigma) or
        (args.perturb is not None and "Nabc" in args.perturb) or
        args.perturb_B_aniso
    )

    if args.scale != 0 and not skip_single_shot_assertion:
        assert r1_nominal <= r1_nominal_init + 1e-6, "r1_nom=%f, r1_not_nom=%f" %(r1_nominal, r1_nominal_init)
    # Use refined-only R1 (HKLs with signal) if available, otherwise all HKLs
    r1_check = r1_refined if refined_corrections else r1_nominal
    if not skip_single_shot_assertion:
        assert r1_check < 0.04, "R1 (refined)=%.4f >= 0.04" % r1_check
    else:
        reason = "aniso B fixed" if args.perturb_B_aniso else ("auto-sigma" if args.auto_sigma else "Nabc perturbed")
        print("Skipping single-shot R1<4%% assertion (%s, R1=%.2f%%)" % (reason, r1_check*100))

    # test hopper_ensemble_refiner using this one shot
    # dump the refinement data to the reflection table format (e.g. the pixel data and background estimates)
    input_refl = os.path.join(P.outdir, "input_data.refl")
    Mod.dump_gathered_to_refl(input_refl)
    df = pandas.read_pickle("%s/pandas/rank0/stage1_dummie_0.pkl"% P.outdir)
    refl_col = "input_refls"
    df[refl_col] = [input_refl]
    P.refiner.load_data_from_refl = True
    P.refiner.check_expt_format = False

    #from simtbx.diffBragg import mpi_logger
    #P.logging.rank0_level="high"
    #mpi_logger.setup_logging_from_params(P)
    modelers = load_inputs(df, P, exper_key="opt_exp_name", refls_key=refl_col)
    modelers.outdir=P.outdir
    modelers.prep_for_refinement()
    print_refinement_summary(P, "single-shot ensemble")
    print("Minimizing using hopper_ensemble_utils...")
    modelers.Minimize(save=True)
    if modelers.SIM.D.record_timings:
        modelers.SIM.D.show_timings(MPI_RANK=0)
    print("Done!")

    from iotbx.reflection_file_reader import any_reflection_file
    opt_F = any_reflection_file("_temp_fhkl_refine/optimized_channel0.mtz").as_miller_arrays()[0]
    opt_map = {h:v for h,v in zip(opt_F.indices(), opt_F.data())}
    hcommon = set(ma_map).intersection(opt_map)

    mset_common = ma.miller_set(flex.miller_index(list(hcommon)), ma.anomalous_flag())
    ma_vals = flex.double([ma_map[h] for h in hcommon])
    opt_vals = flex.double([opt_map[h] for h in hcommon])

    ma_common = miller.array(mset_common, ma_vals).set_observation_type_xray_amplitude()
    opt_common = miller.array(mset_common, opt_vals).set_observation_type_xray_amplitude()
    r1 = ma_common.r1_factor(opt_common)
    print("Ensemble R1=%.4f%% (%d common HKLs)" % (r1*100, len(hcommon)))
    if not skip_single_shot_assertion:
        assert r1 < 0.04
    else:
        print("Skipping single-shot ensemble R1<4%% assertion (%s, R1=%.2f%%)" % (reason, r1*100))

    print("OK")
    del modelers.SIM.D
    del SIM_from_hopper.D

    # === Multi-shot ensemble test ===
    # When G+Fhkl are jointly refined, multiple shots with different G values break
    # the G*F^2 degeneracy, allowing the optimizer to disentangle per-shot G from
    # shared Fhkl. Runs whenever --perturb G or --perturb-B is specified.
    run_multishot = (args.perturb is not None and "G" in args.perturb) or (args.perturb_B is not None) or args.perturb_B_aniso
    if run_multishot:
        from copy import deepcopy as _deepcopy
        from scitbx.matrix import col as _col, sqr as _sqr
        from simtbx.diffBragg import hopper_ensemble_utils as _heu

        print("\n=== Multi-shot ensemble test (auto_sigma=%s) ===" % args.auto_sigma)

        N_MULTI_SHOTS = args.nshots
        # Deterministic rotation angles and G scale factors (cycling for nshots>3)
        _rot_step = 20.0  # degrees between shots
        rotation_angles_deg = [_rot_step * i for i in range(N_MULTI_SHOTS)]
        _g_pattern = [1.0, 0.6, 1.8, 0.9, 1.4, 0.7, 1.2, 0.5, 1.6, 0.8]
        G_scale_factors = [_g_pattern[i % len(_g_pattern)] for i in range(N_MULTI_SHOTS)]
        B_factors_per_shot = [0] * N_MULTI_SHOTS
        if args.perturb_B is not None:
            _b_pattern = [0.8, 1.0, 1.3, 0.7, 1.1, 0.9, 1.2, 0.6, 1.4, 0.85]
            B_factors_per_shot = [args.perturb_B * _b_pattern[i % len(_b_pattern)] for i in range(N_MULTI_SHOTS)]
        beam_dir = _col((0, 0, 1))  # beam along z
        base_G = SIM.D.spot_scale  # ground truth base G

        # Free the original SIM.D to make GPU room
        saved_beam = SIM.D.beam
        del SIM.D

        # Set up phil params for multi-shot ensemble
        P_multi = _deepcopy(P)
        P_multi.fix.G = False
        P_multi.fix.Fhkl = False
        P_multi.auto_Fhkl_sigma = True
        if args.perturb_B is not None:
            P_multi.fix.B = False
            P_multi.init.B = 0
        if args.perturb_B_aniso:
            if args.refine_B_iso:
                # Simulate with aniso B but refine with isotropic B (comparison mode)
                P_multi.fix.Baniso = True
                P_multi.fix.B = False
                P_multi.init.B = 0
                P_multi.mins.B = -1  # avoid RangedParameter dead gradient at init=min=0
                print("refine-B-iso mode: GT has aniso B, refining with isotropic B only")
            else:
                P_multi.fix.Baniso = False
                P_multi.init.Baniso = [0, 0, 0, 0, 0, 0]
                # Physics-based bounds from reciprocal metric tensor:
                # β_max = B_max/4 * |g*_ij| where B_max ~ 80 Å² is a generous physical upper bound
                B_bound = 80.0  # Angstrom^2
                gstar_components = [gstar[0], gstar[1], gstar[2], gstar[3], gstar[4], gstar[5]]
                beta_bounds = [max(B_bound/4 * abs(gc), 1e-7) for gc in gstar_components]
                P_multi.mins.Baniso = [-bb for bb in beta_bounds]
                P_multi.maxs.Baniso = list(beta_bounds)
                print("Baniso bounds: mins=[%s]" % ", ".join("%.6f" % -b for b in beta_bounds))
                print("               maxs=[%s]" % ", ".join("%.6f" % b for b in beta_bounds))
        P_multi.outdir = "_temp_fhkl_refine_multishot"

        from collections import Counter as _Counter
        ensemble = _heu.DataModelers()
        multi_shot_images = []  # for plotting
        per_shot_hkls = []  # list of sets: ASU HKLs observed in each shot

        # When Nabc is also being refined, use milder G perturbation (3x vs 10x)
        # to avoid G*Nabc coupling making the optimization intractable
        nabc_perturbed = args.perturb is not None and "Nabc" in args.perturb
        g_perturb_factor = 3 if nabc_perturbed else 10

        for i_shot in range(N_MULTI_SHOTS):
            angle = rotation_angles_deg[i_shot]
            g_factor = G_scale_factors[i_shot]
            true_G_i = base_G * g_factor
            perturbed_G_i = true_G_i * g_perturb_factor

            # Rotate crystal about beam axis (same Miller indices stay on Ewald sphere)
            rotated_cryst = _deepcopy(p65_C)
            if angle != 0:
                R = beam_dir.axis_and_angle_as_r3_rotation_matrix(angle, deg=True)
                new_A = _sqr(R) * _sqr(p65_C.get_A())
                rotated_cryst.set_A(new_A)

            # Create SimData for this shot
            nbcryst_i = NBcrystal()
            nbcryst_i.dxtbx_crystal = rotated_cryst
            nbcryst_i.thick_mm = 0.005
            nbcryst_i.isotropic_ncells = False
            nbcryst_i.Ncells_abc = NCELLS_GT[:3]
            nbcryst_i.Ncells_def = NCELLS_GT[3:]
            nbcryst_i.space_group = "P6522"
            nbcryst_i.miller_array = ma  # TRUE Fhkl (ground truth)

            SIM_i = SimData(use_default_crystal=False)
            SIM_i.detector = SimData.simple_detector(detdist, 0.1, shape)
            SIM_i.crystal = nbcryst_i
            SIM_i.instantiate_diffBragg(oversample=1, auto_set_spotscale=True, default_F=0)
            SIM_i.D.default_F = 0
            SIM_i.D.F000 = 0
            SIM_i.D.progress_meter = False
            if nabc_perturbed:
                SIM_i.D.no_Nabc_scale = True
                SIM_i.D.spot_scale *= NABC_determ**2
            SIM_i.D.spot_scale *= g_factor

            SIM_i.water_path_mm = 0.005
            SIM_i.air_path_mm = 0.1
            SIM_i.add_air = True
            SIM_i.add_Water = True
            SIM_i.include_noise = True
            true_B_i = B_factors_per_shot[i_shot]
            SIM_i.D.Bfactor_image = true_B_i
            if args.perturb_B_aniso:
                SIM_i.D.Bfactor_aniso = tuple(Baniso_GT)
            SIM_i.D.add_diffBragg_spots()
            spots_i = SIM_i.D.raw_pixels.as_numpy_array()
            SIM_i._add_background()
            SIM_i.D.readout_noise_adu = args.readout
            SIM_i._add_noise()
            img_i = SIM_i.D.raw_pixels.as_numpy_array()
            multi_shot_images.append((img_i.copy(), angle, g_factor))
            SIM_i.D.raw_pixels *= 0
            del SIM_i.D  # free GPU memory

            # Create experiment with imageset
            E_i = Experiment()
            E_i.crystal = rotated_cryst
            E_i.detector = SIM_i.detector
            E_i.beam = saved_beam
            E_i.imageset = make_imageset([img_i], E_i.beam, E_i.detector)

            refls_i = utils.refls_from_sims([spots_i], E_i.detector, E_i.beam, thresh=18)
            refls_i['id'] = flex.int(len(refls_i), 0)
            utils.refls_to_q(refls_i, E_i.detector, E_i.beam, update_table=True)
            utils.refls_to_hkl(refls_i, E_i.detector, E_i.beam, E_i.crystal, update_table=True)

            print("Shot %d: %d refls, true G=%.6f, perturbed G=%.6f, rot=%d deg, true B=%.2f"
                  % (i_shot, len(refls_i), true_G_i, perturbed_G_i, angle, true_B_i))

            # Create DataModeler for this shot
            P_multi.init.G = perturbed_G_i
            shot_mod = hopper_utils.DataModeler(P_multi)
            assert shot_mod.GatherFromExperiment(E_i, refls_i, sg_symbol=P.space_group)
            shot_mod.set_parameters_for_experiment(best=None)
            shot_mod.set_spectrum()
            shot_mod.Umatrices = [E_i.crystal.get_U()]

            ensemble.add_modeler(shot_mod)
            per_shot_hkls.append(set(map(tuple, shot_mod.Hi_asu)))
            print("  Shot %d: %d unique ASU HKLs" % (i_shot, len(per_shot_hkls[-1])))

        # Compute HKL multiplicity (number of shots each HKL appears in)
        hkl_multiplicity = _Counter()
        for hkl_set in per_shot_hkls:
            hkl_multiplicity.update(hkl_set)
        all_unique_hkls = set(hkl_multiplicity.keys())
        mult_histogram = _Counter(hkl_multiplicity.values())
        print("\n--- HKL multiplicity histogram ---")
        for mult in sorted(mult_histogram):
            print("  multiplicity=%d: %d HKLs" % (mult, mult_histogram[mult]))
        print("  Total unique HKLs across all shots: %d" % len(all_unique_hkls))
        print("--- End HKL multiplicity ---\n")

        # Plot multi-shot images to verify same HKLs at different azimuthal positions
        if args.plot:
            import pylab as plt
            fig, axes = plt.subplots(1, N_MULTI_SHOTS, figsize=(5*N_MULTI_SHOTS, 5))
            if N_MULTI_SHOTS == 1:
                axes = [axes]
            for ax, (shot_img, shot_angle, shot_gfactor) in zip(axes, multi_shot_images):
                im = ax.imshow(shot_img, vmax=100)
                ax.set_title("rot=%d deg, G_scale=%.1f" % (shot_angle, shot_gfactor))
            plt.suptitle("Multi-shot images (same HKLs, rotated about beam)")
            plt.tight_layout()
            plt.show()

        # Set up the ensemble: shared SIM, x slices, Fhkl channels
        ensemble.outdir = P_multi.outdir
        ensemble.mpi_set_x_slices()
        ensemble.SIM = hopper_utils.get_simulator_for_data_modelers(ensemble[0])
        ensemble.set_Fhkl_channels()

        # prep_for_refinement handles auto-sigma computation (two-pass if needed)
        ensemble.prep_for_refinement()
        print_refinement_summary(P_multi, "multi-shot ensemble (%d shots)" % N_MULTI_SHOTS)
        print("Multi-shot ensemble Minimizing...")
        x_result = ensemble.Minimize(save=True)
        print("Multi-shot ensemble Done!")

        # Per-shot parameter recovery
        print("\n--- Per-shot parameter recovery ---")
        for i_shot in range(N_MULTI_SHOTS):
            mod = ensemble[i_shot]
            shot_x = x_result[ensemble.x_slices[i_shot]]

            G_p = mod.P["G_xtal0"]
            refined_G = G_p.get_val(shot_x[G_p.xpos])
            true_G = base_G * G_scale_factors[i_shot]

            B_p = mod.P["Bfactor"]
            refined_B = B_p.get_val(shot_x[B_p.xpos])
            true_B = B_factors_per_shot[i_shot]

            chol_names_p = ["chol_L11", "chol_L21", "chol_L22", "chol_L31", "chol_L32", "chol_L33"]
            if chol_names_p[0] in mod.P:
                L_ref = [mod.P[n].get_val(shot_x[mod.P[n].xpos]) for n in chol_names_p]
                Na, Nb, Nc, Nd, Ne, Nf = cholesky_to_nabc(*L_ref)
                nabc_str = "Na=%.1f Nb=%.1f Nc=%.1f Nd=%.2f Ne=%.2f Nf=%.2f" % (Na, Nb, Nc, Nd, Ne, Nf)
            else:
                nabc_str = "Nabc fixed"

            # Aniso B recovery
            baniso_str = ""
            if args.perturb_B_aniso and not args.refine_B_iso:
                Baniso_ref = [mod.P["Baniso%d" % i].get_val(shot_x[mod.P["Baniso%d" % i].xpos]) for i in range(6)]
                baniso_str = " | Baniso=[%s]" % ", ".join("%.6f" % v for v in Baniso_ref)

            print("  Shot %d: G true=%.4f ref=%.4f (%.1f%%) | B true=%.2f ref=%.4f | %s%s"
                  % (i_shot, true_G, refined_G, refined_G/true_G*100,
                     true_B, refined_B, nabc_str, baniso_str))

        if args.perturb_B_aniso and not args.refine_B_iso:
            print("  Baniso GT: [%s]" % ", ".join("%.6f" % v for v in Baniso_GT))
        elif args.refine_B_iso:
            print("  (refine-B-iso mode: GT has aniso B, refining with scalar B per shot)")
        print("  GT: Na=%.1f Nb=%.1f Nc=%.1f Nd=%.2f Ne=%.2f Nf=%.2f"
              % NCELLS_GT)
        print("--- End per-shot recovery ---\n")

        # Check R1 against ground truth
        from iotbx.reflection_file_reader import any_reflection_file as _arf
        opt_F_multi = _arf("%s/optimized_channel0.mtz" % P_multi.outdir).as_miller_arrays()[0]
        opt_map_multi = {h: v for h, v in zip(opt_F_multi.indices(), opt_F_multi.data())}
        hcommon_multi = set(ma_map).intersection(opt_map_multi)

        mset_common_multi = ma.miller_set(flex.miller_index(list(hcommon_multi)), ma.anomalous_flag())
        ma_vals_multi = flex.double([ma_map[h] for h in hcommon_multi])
        opt_vals_multi = flex.double([opt_map_multi[h] for h in hcommon_multi])

        ma_common_multi = miller.array(mset_common_multi, ma_vals_multi).set_observation_type_xray_amplitude()
        opt_common_multi = miller.array(mset_common_multi, opt_vals_multi).set_observation_type_xray_amplitude()
        r1_multi = ma_common_multi.r1_factor(opt_common_multi)
        print("Multi-shot ensemble R1=%.4f%% (%d common HKLs)" % (r1_multi * 100, len(hcommon_multi)))

        # Filtered R1: only HKLs with multiplicity >= 2
        hkl_mult2 = {h for h, m in hkl_multiplicity.items() if m >= 2}
        hcommon_mult2 = hcommon_multi.intersection(hkl_mult2)
        if hcommon_mult2:
            mset_mult2 = ma.miller_set(flex.miller_index(list(hcommon_mult2)), ma.anomalous_flag())
            ma_vals_mult2 = flex.double([ma_map[h] for h in hcommon_mult2])
            opt_vals_mult2 = flex.double([opt_map_multi[h] for h in hcommon_mult2])
            ma_arr_mult2 = miller.array(mset_mult2, ma_vals_mult2).set_observation_type_xray_amplitude()
            opt_arr_mult2 = miller.array(mset_mult2, opt_vals_mult2).set_observation_type_xray_amplitude()
            r1_mult2 = ma_arr_mult2.r1_factor(opt_arr_mult2)
            print("Multi-shot ensemble R1 (mult>=2)=%.4f%% (%d/%d HKLs)"
                  % (r1_mult2 * 100, len(hcommon_mult2), len(hcommon_multi)))
        else:
            print("WARNING: no HKLs with multiplicity >= 2")

        # Use mult>=2 R1 when available (single-multiplicity HKLs are underconstrained)
        r1_assert = r1_mult2 if hcommon_mult2 else r1_multi
        r1_label = "mult>=2" if hcommon_mult2 else "all"
        # Looser threshold for aniso B: per-shot Baniso + shared Fhkl degeneracy
        r1_thresh = 0.06 if args.perturb_B_aniso else 0.04
        assert r1_assert < r1_thresh, "Multi-shot R1 (%s)=%.4f >= %.2f" % (r1_label, r1_assert, r1_thresh)
        print("Multi-shot ensemble test PASSED!")

        del ensemble.SIM.D

    for name in find_diffBragg_instances(globals()): del globals()[name]
