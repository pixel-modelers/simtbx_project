"""
Test: Hopper <-> Geometry parameter propagation round-trip.

Tests both directions of the hopper/geometry handoff:

FORWARD (hopper -> geometry):
  set_parameters_for_experiment -> construct x -> save_to_pandas -> DataFrame
      -> PAR_from_params -> CrystalParameters -> assert values match

REVERSE (geometry -> hopper):
  set_parameters_for_experiment -> construct x -> save_to_pandas -> DataFrame
      -> new DataModeler -> set_parameters_for_experiment(best=df) -> assert values match

This does NOT run refinement or GPU kernels in the inner loop. It creates
the synthetic data ONCE, then for each of 30 configurations tests both directions.
"""
from __future__ import division
from argparse import ArgumentParser

parser = ArgumentParser(description="Test hopper->geometry parameter propagation")
parser.add_argument("--verbose", "-v", action="store_true")
parser.add_argument("--filter", type=str, default=None,
                    help="Only run configs whose name contains this substring")
parser.add_argument("--kokkos", action="store_true")
parser.add_argument("--forward-only", action="store_true",
                    help="Only run forward (hopper->geometry) tests")
parser.add_argument("--reverse-only", action="store_true",
                    help="Only run reverse (geometry->hopper) tests")
args = parser.parse_args()

import os
import sys

if args.kokkos:
    os.environ["DIFFBRAGG_USE_KOKKOS"] = "1"

from simtbx.diffBragg.device import DeviceWrapper
with DeviceWrapper(0) as _:

    import logging
    import numpy as np
    from copy import deepcopy
    from collections import namedtuple

    from cctbx import miller
    from dials.array_family import flex
    from scitbx.matrix import sqr
    from simtbx.nanoBragg.nanoBragg_crystal import NBcrystal
    from simtbx.nanoBragg.sim_data import SimData
    from simtbx.diffBragg import hopper_utils, utils
    from simtbx.diffBragg.hopper_utils import (
        DataModeler, get_simulator_for_data_modelers, model,
        update_crystal_from_x, get_param_from_x)
    from simtbx.diffBragg.hopper_io import save_to_pandas
    from simtbx.diffBragg.stage_two_utils import PAR_from_params
    from simtbx.diffBragg.refiners.geometry import CrystalParameters
    from simtbx.diffBragg.refiners.parameters import Parameters, RangedParameter
    from dxtbx.model import Experiment
    from simtbx.nanoBragg import make_imageset
    from simtbx.diffBragg.phil import hopper_phil, philz
    from libtbx.phil import parse

    phil_scope = parse(hopper_phil + philz)

    # -------------------------------------------------------------------------
    # Test configuration
    # -------------------------------------------------------------------------
    TestConfig = namedtuple("TestConfig", [
        "name",
        "refine_B",         # bool
        "refine_Baniso",    # bool
        "use_cholesky",     # bool
        "refine_diffuse",   # bool
        "refine_RotXYZ",    # bool
        "refine_ucell",     # bool
        "refine_Nabc",      # bool
        "refine_eta",       # bool
        "is_rotation",      # bool
        "shared_crystal",   # bool
    ])

    def C(name, refine_B=False, refine_Baniso=False, use_cholesky=False,
          refine_diffuse=False, refine_RotXYZ=False, refine_ucell=False,
          refine_Nabc=False, refine_eta=False, is_rotation=False,
          shared_crystal=False):
        return TestConfig(name=name, refine_B=refine_B, refine_Baniso=refine_Baniso,
                          use_cholesky=use_cholesky, refine_diffuse=refine_diffuse,
                          refine_RotXYZ=refine_RotXYZ, refine_ucell=refine_ucell,
                          refine_Nabc=refine_Nabc, refine_eta=refine_eta,
                          is_rotation=is_rotation, shared_crystal=shared_crystal)

    ALL_CONFIGS = [
        # Group 1: Baseline
        C("baseline_all_fixed"),
        C("baseline_RotXYZ", refine_RotXYZ=True),
        C("baseline_ucell", refine_ucell=True),
        C("baseline_Nabc", refine_Nabc=True),
        # Group 2: B-factor variants
        C("B_isotropic", refine_B=True),
        C("B_anisotropic", refine_Baniso=True),
        C("B_iso_plus_Nabc", refine_B=True, refine_Nabc=True),
        C("B_aniso_plus_RotXYZ", refine_Baniso=True, refine_RotXYZ=True),
        # Group 3: Cholesky / Ndef
        C("cholesky_only", refine_Nabc=True, use_cholesky=True),
        C("cholesky_plus_B", refine_Nabc=True, use_cholesky=True, refine_B=True),
        C("cholesky_plus_RotXYZ", refine_Nabc=True, use_cholesky=True, refine_RotXYZ=True),
        C("cholesky_full", refine_Nabc=True, use_cholesky=True, refine_RotXYZ=True,
          refine_ucell=True, refine_B=True),
        # Group 4: Diffuse scattering
        C("diffuse_only", refine_diffuse=True),
        C("diffuse_plus_B", refine_diffuse=True, refine_B=True),
        C("diffuse_plus_Nabc", refine_diffuse=True, refine_Nabc=True),
        C("diffuse_full", refine_diffuse=True, refine_Nabc=True, refine_RotXYZ=True,
          refine_B=True),
        # Group 5: Rotation crystallography
        C("rotation_baseline", is_rotation=True, refine_RotXYZ=True),
        C("rotation_full", is_rotation=True, refine_Nabc=True, refine_ucell=True,
          refine_B=True),
        C("rotation_cholesky", is_rotation=True, refine_Nabc=True, use_cholesky=True,
          refine_B=True),
        C("rotation_diffuse", is_rotation=True, refine_diffuse=True, refine_B=True),
        # Group 6: Multi-parameter combos
        C("combo_RotXYZ_ucell", refine_RotXYZ=True, refine_ucell=True),
        C("combo_RotXYZ_Nabc_B", refine_RotXYZ=True, refine_Nabc=True, refine_B=True),
        C("combo_all_crystal", refine_RotXYZ=True, refine_ucell=True, refine_Nabc=True,
          refine_eta=True, refine_B=True),
        C("combo_kitchen_sink", refine_RotXYZ=True, refine_ucell=True, refine_Nabc=True,
          refine_eta=True, refine_B=True, refine_Baniso=True, refine_diffuse=True,
          use_cholesky=True),
        C("combo_eta_only", refine_eta=True),
        C("combo_Scale_eta_Nabc", refine_eta=True, refine_Nabc=True),
        # Group 7: Shared crystal
        C("shared_crystal_basic", shared_crystal=True, refine_RotXYZ=True, refine_ucell=True),
        C("shared_crystal_plus_B", shared_crystal=True, refine_RotXYZ=True, refine_ucell=True,
          refine_B=True),
        C("shared_crystal_cholesky", shared_crystal=True, refine_RotXYZ=True, refine_ucell=True,
          refine_Nabc=True, use_cholesky=True),
        C("shared_crystal_full", shared_crystal=True, refine_RotXYZ=True, refine_ucell=True,
          refine_Nabc=True, refine_B=True, refine_Baniso=True),
    ]

    # -------------------------------------------------------------------------
    # Synthetic setup (P65 crystal, simple detector)
    # -------------------------------------------------------------------------
    p65_cryst = {
        '__id__': 'crystal',
        'real_space_a': (43.32309880004587, 25.5289818883498, 60.49634260901813),
        'real_space_b': (34.201635357808115, -38.82573591182249, -59.255697149884924),
        'real_space_c': (41.42476391176581, 229.70849483520402, -126.60059788183489),
        'space_group_hall_symbol': ' P 65 2 (x,y,z+1/12)',
        'ML_half_mosaicity_deg': 0.06671930026192037,
        'ML_domain_size_ang': 6349.223840307989,
    }
    from dxtbx.model.crystal import CrystalFactory
    BASE_CRYSTAL = CrystalFactory.from_dict(p65_cryst)
    BASE_UCELL = BASE_CRYSTAL.get_unit_cell().parameters()
    SYMBOL = BASE_CRYSTAL.get_space_group().info().type().lookup_symbol()

    def make_sim_and_data(is_rotation=False, seed=0):
        """Create P65 crystal simulation with spots+background+noise. ONE-TIME setup."""
        np.random.seed(seed)
        cryst = deepcopy(BASE_CRYSTAL)

        nbcryst = NBcrystal()
        nbcryst.dxtbx_crystal = cryst
        nbcryst.thick_mm = 0.005
        nbcryst.isotropic_ncells = False
        nbcryst.Ncells_abc = (12, 12, 11)
        nbcryst.Ncells_def = (0, 0, 0)
        nbcryst.space_group = "P6522"
        ma = utils.make_miller_array(SYMBOL, BASE_UCELL, d_min=1.5)
        np.random.seed(seed)
        new_data = ma.d_spacings().data() * 10
        ma = miller.array(ma.set(), new_data).set_observation_type_xray_amplitude()
        nbcryst.miller_array = ma

        SIM = SimData(use_default_crystal=False)
        shape = 1000, 1001
        SIM.detector = SimData.simple_detector(140, 0.1, shape)
        SIM.crystal = nbcryst

        SIM.instantiate_diffBragg(oversample=1, auto_set_spotscale=True, default_F=0)
        SIM.D.default_F = 0
        SIM.D.F000 = 0
        SIM.D.progress_meter = False
        SIM.D.verbose = 0
        SIM.water_path_mm = 0.005
        SIM.air_path_mm = 0.1
        SIM.add_air = True
        SIM.add_Water = True
        SIM.include_noise = True
        SIM.D.readout_noise_adu = 3

        if is_rotation:
            SIM.D.phi_deg = 0
            SIM.D.osc_deg = 0.1
            SIM.D.phisteps = 2
            SIM.D.spindle_axis = (0, 0, 1)

        SIM.D.Bfactor_image = 0
        SIM.D.add_diffBragg_spots()

        spots = SIM.D.raw_pixels.as_numpy_array()
        SIM._add_background()
        SIM._add_noise()
        img = SIM.D.raw_pixels.as_numpy_array()
        SIM.D.raw_pixels *= 0

        E = Experiment()
        E.crystal = cryst
        E.detector = SIM.detector
        E.beam = SIM.D.beam
        E.imageset = make_imageset([img], E.beam, E.detector)
        refls = utils.refls_from_sims([spots], E.detector, E.beam, thresh=18)
        refls['id'] = flex.int(len(refls), 0)
        utils.refls_to_q(refls, E.detector, E.beam, update_table=True)
        utils.refls_to_hkl(refls, E.detector, E.beam, E.crystal, update_table=True)

        # Write MTZ for hopper's simulator_for_refinement
        mtz_name = os.path.abspath("_propagation_test_seed%d.mtz" % seed)
        ma.as_mtz_dataset(column_root_label="F").mtz_object().write(mtz_name)

        return SIM, E, refls, mtz_name

    # Create data ONCE (two orientations for shared_crystal tests)
    print("Creating synthetic data (stills)...", flush=True)
    SIM_STILL, E_STILL, REFLS_STILL, MTZ_STILL = make_sim_and_data(is_rotation=False, seed=0)
    print("  %d reflections" % len(REFLS_STILL), flush=True)
    print("Creating synthetic data (rotation)...", flush=True)
    SIM_ROT, E_ROT, REFLS_ROT, MTZ_ROT = make_sim_and_data(is_rotation=True, seed=0)
    print("  %d reflections" % len(REFLS_ROT), flush=True)
    # Second orientation for shared crystal tests
    print("Creating synthetic data (stills, seed=42)...", flush=True)
    SIM_STILL2, E_STILL2, REFLS_STILL2, MTZ_STILL2 = make_sim_and_data(is_rotation=False, seed=42)
    print("  %d reflections" % len(REFLS_STILL2), flush=True)

    # Also create one refinement SIM (for save_to_pandas which needs SIM.crystal/beam/D)
    print("Creating refinement SIM...", flush=True)
    _phil_template = phil_scope.extract()
    _phil_template.simulator.structure_factors.mtz_name = MTZ_STILL
    _phil_template.simulator.structure_factors.mtz_column = "F(+),F(-)"
    _phil_template.simulator.beam.size_mm = SIM_STILL.beam.size_mm
    _phil_template.simulator.total_flux = SIM_STILL.D.flux
    _phil_template.simulator.oversample = 1
    _phil_template.space_group = SYMBOL
    _Mod_template = DataModeler(_phil_template)
    _Mod_template.GatherFromExperiment(E_STILL, REFLS_STILL, sg_symbol=SYMBOL)
    REF_SIM = get_simulator_for_data_modelers(_Mod_template)
    print("Setup complete.\n", flush=True)

    # -------------------------------------------------------------------------
    # Phil and Modeler construction (CPU only per config)
    # -------------------------------------------------------------------------
    def make_phil(config, SIM, mtz_name):
        """Create phil params for a given test config."""
        P = phil_scope.extract()
        P.debug_mode = True
        P.simulator.structure_factors.mtz_name = mtz_name
        P.simulator.structure_factors.mtz_column = "F(+),F(-)"
        P.init.G = SIM.D.spot_scale
        P.init.Nabc = SIM.crystal.Ncells_abc
        P.init.Ndef = (0, 0, 0)
        P.init.detz_shift = 0
        P.init.eta_abc = (0.01, 0.01, 0.01) if config.refine_eta else (0, 0, 0)
        P.init.diffuse_gamma = (5, 5, 5)
        P.init.diffuse_sigma = (1, 1, 1)

        P.roi.shoebox_size = 10
        P.roi.allow_overlapping_spots = True
        P.relative_tilt = False
        P.roi.fit_tilt = False
        P.roi.pad_shoebox_for_background_estimation = 10
        P.roi.reject_edge_reflections = False
        P.refiner.sigma_r = SIM.D.readout_noise_adu
        P.refiner.adu_per_photon = SIM.D.quantum_gain
        P.simulator.init_scale = 1
        P.simulator.beam.size_mm = SIM.beam.size_mm
        P.simulator.oversample = SIM.D.oversample
        P.simulator.total_flux = SIM.D.flux
        P.use_restraints = False
        P.logging.parameters = False
        P.method = "L-BFGS-B"
        P.ftol = 1e-10
        P.space_group = SYMBOL

        P.fix.Fhkl = True
        P.fix.G = False  # always refine G
        P.types.G = "positive"
        P.sigmas.G = 1

        # Rotation
        P.fix.RotXYZ = [0, 0, 0] if config.refine_RotXYZ else [1, 1, 1]
        # Unit cell
        P.fix.ucell = not config.refine_ucell
        # Nabc
        P.fix.Nabc = not config.refine_Nabc
        P.fix.Ndef = not config.refine_Nabc
        # Cholesky
        if config.use_cholesky:
            P.use_cholesky_Nabc = True
            P.init.cholesky = None
            P.mins.cholesky = [0.1, -10, 0.1, -10, -10, 0.1]
            P.maxs.cholesky = [10, 10, 10, 10, 10, 10]
        # eta
        P.fix.eta_abc = not config.refine_eta
        if config.refine_eta:
            P.simulator.crystal.num_mosaicity_samples = 10
        else:
            P.simulator.crystal.num_mosaicity_samples = 1
        # B-factor
        P.fix.B = not config.refine_B
        if config.refine_B:
            P.init.B = 0
            P.mins.B = -1
        # Baniso
        P.fix.Baniso = not config.refine_Baniso
        if config.refine_Baniso:
            P.init.Baniso = [0.001] * 6
        # Diffuse
        P.fix.diffuse_gamma = not config.refine_diffuse
        P.fix.diffuse_sigma = not config.refine_diffuse
        P.use_diffuse_models = config.refine_diffuse
        # Rotation mode
        if config.is_rotation:
            P.simulator.gonio.delta_phi = 0.1

        P.fix.detz_shift = True
        P.outdir = "_temp_propagation_test"

        return P

    def build_modeler_and_x(config, E, refls, SIM, mtz_name):
        """Build DataModeler, set parameters, construct a perturbed x vector (CPU only).

        Instead of running L-BFGS-B, we construct x = [1.05, ...] so every parameter
        is slightly off its init value. This exercises the serialization pathway
        without any GPU computation.
        """
        P = make_phil(config, SIM, mtz_name)

        Mod = DataModeler(P)
        Mod.GatherFromExperiment(E, refls, sg_symbol=P.space_group)
        Mod.set_parameters_for_experiment()

        # Perturb all params slightly off init (x=1 is init, x=1.05 is perturbed)
        nparam = len(Mod.P)
        x = np.ones(nparam) * 1.05

        # Build refined experiment with RotXYZ baked into A-matrix
        new_crystal = update_crystal_from_x(Mod, REF_SIM, x)
        Eopt = deepcopy(E)
        Eopt.crystal = new_crystal

        # Serialize via save_to_pandas (uses Mod.P to extract params from x)
        Mod.exper_name = "dummy.expt"
        Mod.refl_name = "dummy.refl"
        os.makedirs(P.outdir, exist_ok=True)
        df = save_to_pandas(x, Mod, REF_SIM, "dummy.expt", P, Mod.E, 0, "dummy.refl",
                            rank=0, write_expt=False, write_pandas=False)

        return Mod, x, df, Eopt, P

    # -------------------------------------------------------------------------
    # Assertion helpers
    # -------------------------------------------------------------------------
    def assert_close(a, b, tol, label):
        diff = abs(a - b)
        if diff > tol:
            raise AssertionError("%s: |%.8g - %.8g| = %.8g > tol %.8g" % (label, a, b, diff, tol))

    def assert_amat_close(amat1, amat2, tol, label):
        for i in range(9):
            assert_close(amat1[i], amat2[i], tol, "%s[%d]" % (label, i))

    # -------------------------------------------------------------------------
    # Single-config test
    # -------------------------------------------------------------------------
    def run_single_test(config, verbose=False):
        if config.shared_crystal:
            run_shared_crystal_test(config, verbose)
            return

        # Pick stills or rotation data
        if config.is_rotation:
            E, refls, SIM, mtz = E_ROT, REFLS_ROT, SIM_ROT, MTZ_ROT
        else:
            E, refls, SIM, mtz = E_STILL, REFLS_STILL, SIM_STILL, MTZ_STILL

        Mod, x, df, Eopt, P = build_modeler_and_x(config, E, refls, SIM, mtz)

        # Extract hopper-side values
        hopper_vals = get_param_from_x(x, Mod, as_dict=True)

        # Deserialize via PAR_from_params
        PAR = PAR_from_params(P, Eopt, best=df)

        # ---- Assert: G (scale) ----
        assert_close(hopper_vals['scale'], PAR.Scale.init, 1e-6, "G")

        # ---- Assert: A-matrix (RotXYZ baked into U) ----
        hopper_Amat = tuple(Eopt.crystal.get_A())
        par_Amat = tuple(PAR.Umatrix * PAR.Bmatrix)
        assert_amat_close(hopper_Amat, par_Amat, 1e-8, "Amat")

        # ---- Assert: Nabc ----
        for i in range(3):
            assert_close(df.ncells.values[0][i], PAR.Nabc[i].init, 1e-6, "Nabc%d" % i)

        # ---- Assert: Ndef ----
        for i in range(3):
            assert_close(df.ncells_def.values[0][i], PAR.Ndef[i].init, 1e-6, "Ndef%d" % i)

        # ---- Assert: eta ----
        for i in range(3):
            assert_close(df.eta_abc.values[0][i], PAR.eta[i].init, 1e-6, "eta%d" % i)

        # ---- Assert: ucell ----
        uc_man = PAR.ucell_man
        par_uc = uc_man.unit_cell_parameters
        hopper_uc = (df.a.values[0], df.b.values[0], df.c.values[0],
                     df.al.values[0], df.be.values[0], df.ga.values[0])
        for i in range(6):
            assert_close(hopper_uc[i], par_uc[i], 1e-4, "ucell[%d]" % i)

        # ---- Assert: detz_shift ----
        assert_close(df.detz_shift_mm.values[0], PAR.detz_shift.init * 1e3, 1e-6, "detz_shift_mm")

        # ---- Assert: B (isotropic) ----
        if "Bfactor" in df.columns:
            assert_close(hopper_vals['Bfactor'], PAR.B.init, 1e-6, "Bfactor")

        # ---- Assert: Baniso ----
        if config.refine_Baniso and PAR.Baniso is not None:
            for i in range(6):
                hopper_ba = Mod.P["Baniso%d" % i].get_val(x[Mod.P["Baniso%d" % i].xpos])
                assert_close(hopper_ba, PAR.Baniso[i].init, 1e-6, "Baniso%d" % i)

        # ---- Assert: diffuse_gamma/sigma ----
        for i in range(3):
            assert_close(df.diffuse_gamma.values[0][i], PAR.diffuse_gamma[i].init, 1e-6,
                         "diffuse_gamma%d" % i)
            assert_close(df.diffuse_sigma.values[0][i], PAR.diffuse_sigma[i].init, 1e-6,
                         "diffuse_sigma%d" % i)

        # ---- Assert: RotXYZ are zero in PAR (baked into Amat) ----
        for i in range(3):
            assert_close(0, PAR.RotXYZ_params[i].init, 1e-10, "PAR.RotXYZ%d==0" % i)

        # ---- Test CrystalParameters construction ----
        class MockModeler:
            pass
        mock = MockModeler()
        mock.PAR = PAR
        mock.all_data = np.ones(10)
        mock.refls = Mod.refls[:1]
        mock.refls_idx = [0]
        mock.roi_id_slices = {0: [slice(0, 10)]}
        mock.roi_id_unique = [0]
        mock.set_slices = lambda s: None
        mock.per_roi_scales_per_pix = np.ones(10)

        P.geometry.shared_crystal = False
        P.geometry.fix.Nabc = True
        P.geometry.fix.Ndef = True
        P.geometry.fix.eta_abc = True
        P.geometry.fix.RotXYZ = [True, True, True]
        P.geometry.fix.G = True
        P.geometry.fix.ucell = True
        P.geometry.fix.perRoiScale = True

        cp = CrystalParameters(P, {0: mock})
        found_scale = False
        found_nabc = [False] * 3
        for p in cp.parameters:
            if "Scale" in p.name:
                assert_close(hopper_vals['scale'], p.init, 1e-6, "CrystalParams.Scale")
                found_scale = True
            for i in range(3):
                if p.name.endswith("Nabc%d" % i):
                    assert_close(df.ncells.values[0][i], p.init, 1e-6,
                                 "CrystalParams.Nabc%d" % i)
                    found_nabc[i] = True
        assert found_scale, "Scale not found in CrystalParameters"
        assert all(found_nabc), "Nabc not found in CrystalParameters"

        if verbose:
            print("    OK", flush=True)

    # -------------------------------------------------------------------------
    # Shared crystal test
    # -------------------------------------------------------------------------
    def run_shared_crystal_test(config, verbose=False):
        E1, refls1, SIM1, mtz1 = E_STILL, REFLS_STILL, SIM_STILL, MTZ_STILL
        E2, refls2, SIM2, mtz2 = E_STILL2, REFLS_STILL2, SIM_STILL2, MTZ_STILL2

        Mod1, x1, df1, Eopt1, P = build_modeler_and_x(config, E1, refls1, SIM1, mtz1)
        Mod2, x2, df2, Eopt2, P2 = build_modeler_and_x(config, E2, refls2, SIM2, mtz2)

        hopper_vals1 = get_param_from_x(x1, Mod1, as_dict=True)
        hopper_vals2 = get_param_from_x(x2, Mod2, as_dict=True)

        PAR1 = PAR_from_params(P, Eopt1, best=df1)
        PAR2 = PAR_from_params(P2, Eopt2, best=df2)

        assert_close(hopper_vals1['scale'], PAR1.Scale.init, 1e-6, "shot1.G")
        assert_close(hopper_vals2['scale'], PAR2.Scale.init, 1e-6, "shot2.G")

        # Build CrystalParameters with shared_crystal=True
        class MockModeler:
            pass
        def make_mock(PAR, refls):
            mock = MockModeler()
            mock.PAR = PAR
            mock.all_data = np.ones(10)
            mock.refls = refls[:1]
            mock.refls_idx = [0]
            mock.roi_id_slices = {0: [slice(0, 10)]}
            mock.roi_id_unique = [0]
            mock.set_slices = lambda s: None
            mock.per_roi_scales_per_pix = np.ones(10)
            return mock

        P.geometry.shared_crystal = True
        P.geometry.fix.Nabc = True
        P.geometry.fix.Ndef = True
        P.geometry.fix.eta_abc = True
        P.geometry.fix.RotXYZ = [True, True, True]
        P.geometry.fix.G = True
        P.geometry.fix.ucell = True
        P.geometry.fix.perRoiScale = True

        data_modelers = {0: make_mock(PAR1, Mod1.refls), 1: make_mock(PAR2, Mod2.refls)}
        cp = CrystalParameters(P, data_modelers)

        # Aliases exist (RotXYZ and ucell are shared)
        assert len(cp.alias_pairs) > 0, "shared_crystal should produce alias pairs"

        # Shared params have median init values
        n_uc = len(PAR1.ucell)
        for i_rot in range(3):
            expected = np.median([PAR1.RotXYZ_params[i_rot].init,
                                  PAR2.RotXYZ_params[i_rot].init])
            found = False
            for p in cp.parameters:
                if p.name == "shared_RotXYZ%d" % i_rot:
                    assert_close(expected, p.init, 1e-8, "shared_RotXYZ%d" % i_rot)
                    found = True
                    break
            assert found, "shared_RotXYZ%d not found" % i_rot

        for i_uc in range(n_uc):
            expected = np.median([PAR1.ucell[i_uc].init, PAR2.ucell[i_uc].init])
            found = False
            for p in cp.parameters:
                if p.name == "shared_Ucell%d" % i_uc:
                    assert_close(expected, p.init, 1e-8, "shared_Ucell%d" % i_uc)
                    found = True
                    break
            assert found, "shared_Ucell%d not found" % i_uc

        # Per-shot Scale still independent
        scales = [p for p in cp.parameters if "Scale" in p.name]
        assert len(scales) == 2, "Expected 2 Scale params, got %d" % len(scales)

        if verbose:
            print("    OK (shared_crystal, %d aliases)" % len(cp.alias_pairs), flush=True)

    # -------------------------------------------------------------------------
    # Reverse test: DataFrame -> set_parameters_for_experiment(best=df)
    # -------------------------------------------------------------------------
    def run_single_reverse_test(config, verbose=False):
        """Test the geometry->hopper path: DataFrame -> set_parameters_for_experiment(best=df).

        After geometry refinement, the next hopper cycle loads refined parameters
        from the DataFrame via set_parameters_for_experiment(best=df). This test
        verifies that all parameters survive that round-trip.
        """
        # Pick stills or rotation data
        if config.is_rotation:
            E, refls, SIM, mtz = E_ROT, REFLS_ROT, SIM_ROT, MTZ_ROT
        else:
            E, refls, SIM, mtz = E_STILL, REFLS_STILL, SIM_STILL, MTZ_STILL

        # Forward pass: build modeler, perturb, serialize to DataFrame
        Mod1, x1, df, Eopt, P = build_modeler_and_x(config, E, refls, SIM, mtz)

        # Reverse pass: create new DataModeler, load from DataFrame
        P2 = make_phil(config, SIM, mtz)
        Mod2 = DataModeler(P2)
        Mod2.GatherFromExperiment(deepcopy(E), refls, sg_symbol=P2.space_group)
        Mod2.set_parameters_for_experiment(best=df)

        # ---- Assert: G (scale) ----
        assert_close(df.spot_scales.values[0], Mod2.P["G_xtal0"].init, 1e-6, "rev.G")

        # ---- Assert: A-matrix (RotXYZ baked in) ----
        saved_Amat = tuple(df.Amats.values[0])
        loaded_Amat = tuple(Mod2.E.crystal.get_A())
        assert_amat_close(saved_Amat, loaded_Amat, 1e-8, "rev.Amat")

        # ---- Assert: RotXYZ are zero (baked into Amat) ----
        for i in range(3):
            assert_close(0, Mod2.P["RotXYZ%d_xtal0" % i].init, 1e-10,
                         "rev.RotXYZ%d==0" % i)

        # ---- Assert: Nabc ----
        for i in range(3):
            assert_close(df.ncells.values[0][i], Mod2.P["Nabc%d" % i].init, 1e-6,
                         "rev.Nabc%d" % i)

        # ---- Assert: Ndef ----
        for i in range(3):
            assert_close(df.ncells_def.values[0][i], Mod2.P["Ndef%d" % i].init, 1e-6,
                         "rev.Ndef%d" % i)

        # ---- Assert: eta ----
        for i in range(3):
            expected = df.eta_abc.values[0][i]
            # eta has a clamp to mins.eta_abc + 0.01 if below mins
            min_eta = P2.mins.eta_abc[i]
            if expected <= min_eta:
                expected = min_eta + 1e-2
            assert_close(expected, Mod2.P["eta_abc%d" % i].init, 1e-6, "rev.eta%d" % i)

        # ---- Assert: ucell ----
        par_uc = Mod2.ucell_man.unit_cell_parameters
        hopper_uc = (df.a.values[0], df.b.values[0], df.c.values[0],
                     df.al.values[0], df.be.values[0], df.ga.values[0])
        for i in range(6):
            assert_close(hopper_uc[i], par_uc[i], 1e-4, "rev.ucell[%d]" % i)

        # ---- Assert: detz_shift ----
        if "detz_shift_mm" in df.columns:
            assert_close(df.detz_shift_mm.values[0] * 1e-3,
                         Mod2.P["detz_shift"].init, 1e-9, "rev.detz_shift")

        # ---- Assert: Bfactor ----
        if "Bfactor" in df.columns and config.refine_B:
            assert_close(df.Bfactor.values[0], Mod2.P["Bfactor"].init, 1e-6,
                         "rev.Bfactor")

        # ---- Assert: Baniso ----
        if config.refine_Baniso and "Bfactor_aniso" in df.columns:
            for i in range(6):
                assert_close(df.Bfactor_aniso.values[0][i],
                             Mod2.P["Baniso%d" % i].init, 1e-6,
                             "rev.Baniso%d" % i)

        # ---- Assert: diffuse_gamma/sigma ----
        if config.refine_diffuse and "diffuse_gamma" in df.columns:
            for i in range(3):
                assert_close(df.diffuse_gamma.values[0][i],
                             Mod2.P["diffuse_gamma%d" % i].init, 1e-6,
                             "rev.diffuse_gamma%d" % i)
                assert_close(df.diffuse_sigma.values[0][i],
                             Mod2.P["diffuse_sigma%d" % i].init, 1e-6,
                             "rev.diffuse_sigma%d" % i)

        if verbose:
            print("    OK", flush=True)

    # -------------------------------------------------------------------------
    # Test runner
    # -------------------------------------------------------------------------
    def _run_test_suite(test_fn, configs, suite_name, verbose=False):
        """Run a suite of tests, return (n_pass, failures)."""
        n_pass = 0
        failures = []
        for i, config in enumerate(configs):
            label = "[%d/%d] %s" % (i + 1, len(configs), config.name)
            try:
                if verbose:
                    print(label, end=" ... ", flush=True)
                test_fn(config, verbose=verbose)
                n_pass += 1
                if not verbose:
                    sys.stdout.write(".")
                    sys.stdout.flush()
            except Exception as e:
                failures.append((config.name, str(e)))
                if verbose:
                    import traceback
                    traceback.print_exc()
                    sys.stdout.flush()
                else:
                    sys.stdout.write("F")
                    sys.stdout.flush()
        if not verbose:
            print()
        return n_pass, failures

    def run_all_tests(verbose=False, name_filter=None,
                      forward_only=False, reverse_only=False):
        configs = ALL_CONFIGS
        if name_filter:
            configs = [c for c in configs if name_filter in c.name]
            if not configs:
                print("No configs match filter '%s'" % name_filter)
                sys.exit(1)

        logging.disable(logging.CRITICAL)

        total_pass = 0
        total_fail = []

        # Forward tests (hopper -> geometry)
        if not reverse_only:
            print("--- Forward (hopper -> geometry) ---", flush=True)
            n, fails = _run_test_suite(run_single_test, configs,
                                       "forward", verbose)
            print("%d/%d forward tests passed" % (n, len(configs)), flush=True)
            total_pass += n
            total_fail.extend([("FWD:" + name, msg) for name, msg in fails])

        # Reverse tests (geometry -> hopper)
        if not forward_only:
            # Skip shared_crystal configs (only meaningful for forward direction)
            rev_configs = [c for c in configs if not c.shared_crystal]
            print("\n--- Reverse (geometry -> hopper) ---", flush=True)
            n, fails = _run_test_suite(run_single_reverse_test, rev_configs,
                                       "reverse", verbose)
            print("%d/%d reverse tests passed" % (n, len(rev_configs)), flush=True)
            total_pass += n
            total_fail.extend([("REV:" + name, msg) for name, msg in fails])

        logging.disable(logging.NOTSET)

        total_count = total_pass + len(total_fail)
        print("\n%d/%d total tests passed" % (total_pass, total_count), flush=True)
        for name, msg in total_fail:
            print("  FAIL: %s: %s" % (name, msg))
        assert not total_fail, "%d tests failed" % len(total_fail)
        print("PASSED")

    run_all_tests(verbose=args.verbose, name_filter=args.filter,
                  forward_only=args.forward_only, reverse_only=args.reverse_only)
