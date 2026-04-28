#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import socket
from copy import deepcopy
import glob
from simtbx.diffBragg import utils, hopper_utils
from dxtbx.model.experiment_list import ExperimentListFactory
import time
from xfel.merging.application.input.file_loader import create_experiment_identifier
from xfel.poly.recompute_mosaic_params import extract_mosaic_parameters_using_lambda_spread
from dials.algorithms.shoebox import MaskCode
import sys
try:
    import pandas
except ImportError:
    print("Please install pandas, libtbx.python -m pip install pandas")
    exit()

try:
    from line_profiler import LineProfiler
except ImportError:
    LineProfiler = None

from simtbx.diffBragg.device import DeviceWrapper

ROTX_ID = 0
ROTY_ID = 1
ROTZ_ID = 2
NCELLS_ID = 9
UCELL_ID_OFFSET = 3
DETZ_ID = 10

# LIBTBX_SET_DISPATCHER_NAME simtbx.diffBragg.hopper
# LIBTBX_SET_DISPATCHER_NAME hopper

import numpy as np
np.seterr(invalid='ignore')
from scipy.stats import pearsonr
import os
from libtbx.mpi4py import MPI

COMM = MPI.COMM_WORLD
# TODO, figure out why next 3 lines are sometimes necessary?!
if not hasattr(COMM, "rank"):
    COMM.rank = 0
    COMM.size = 1
from libtbx.phil import parse

from simtbx.diffBragg import utils
from simtbx.diffBragg.phil import philz
from simtbx.modeling import predictions
from dials.model.data import Shoebox
from dials.array_family import flex
from simtbx.command_line import integrate
from dxtbx.model import ExperimentList

import logging
from simtbx.diffBragg.phil import hopper_phil

philz = hopper_phil + philz
phil_scope = parse(philz)



class Script:
    def __init__(self):
        from dials.util.options import ArgumentParser

        self.params = None
        if COMM.rank == 0:
            self.parser = ArgumentParser(
                usage="",  # stage 1 (per-shot) diffBragg refinement",
                sort_options=True,
                phil=phil_scope,
                read_experiments=True,
                read_reflections=True,
                check_format=False,
                epilog="PyCuties")
            self.params, _ = self.parser.parse_args(show_diff_phil=True)
            assert self.params.outdir is not None
            utils.safe_makedirs(self.params.outdir)
            ts = time.strftime("%Y%m%d-%H%M%S")
            diff_phil_outname = os.path.join(self.params.outdir, "diff_phil_run_at_%s.txt" % ts)
            with open(diff_phil_outname, "w") as o:
                o.write("command line:\n%s\n" % (" ".join(sys.argv)))
                o.write("workding directory: \n%s\n" %os.getcwd())
                o.write("diff phil:\n")
                o.write(self.parser.diff_phil.as_str())
            just_diff_phil_outname = os.path.join(self.params.outdir, "diff.phil")
            with open(just_diff_phil_outname, "w") as o:
                o.write(self.parser.diff_phil.as_str())
        self.params = COMM.bcast(self.params)

        self.dev = COMM.rank % self.params.refiner.num_devices
        logging.info("Rank %d will use device %d on host %s" % (COMM.rank, self.dev, socket.gethostname()))

        if self.params.logging.logname is None:
            self.params.logging.logname = "main_stage1.log"
        if self.params.profile_name is None:
            self.params.profile_name = "prof_stage1.log"
        from simtbx.diffBragg import mpi_logger
        mpi_logger.setup_logging_from_params(self.params)

    def run(self):
        MAIN_LOGGER = logging.getLogger("diffBragg.main")
        assert os.path.exists(self.params.exp_ref_spec_file)
        input_lines = None
        best_models = None
        pd_dir = os.path.join(self.params.outdir, "pandas")
        expt_ref_dir = os.path.join(self.params.outdir, "expers_refls")
        if COMM.rank == 0:
            input_lines = open(self.params.exp_ref_spec_file, "r").readlines()
            if self.params.skip is not None:
                input_lines = input_lines[self.params.skip:]
            if self.params.first_n is not None:
                input_lines = input_lines[:self.params.first_n]
            if self.params.sanity_test_input:
                hopper_utils.sanity_test_input_lines(input_lines)

            if self.params.best_pickle is not None:
                if os.path.isfile(self.params.best_pickle):
                    logging.info("reading pickle %s" % self.params.best_pickle)
                    best_models = pandas.read_pickle(self.params.best_pickle)
                else:
                    logging.info("reading pickles from Glob %s" % self.params.best_pickle)
                    best_pickles = glob.glob(self.params.best_pickle)
                    best_models = pandas.concat([pandas.read_pickle(f) for f in best_pickles])
                    best_models.reset_index(inplace=True, drop=True)

            if self.params.geometry.optimize:
                self.params.dump_gathers = True
                if self.params.gathers_dir is None:
                    self.params.gathers_dir = os.path.join(self.params.outdir, "gathers")

            if self.params.dump_gathers:
                if self.params.gathers_dir is None:
                    raise ValueError("Need to provide a file dir path in order to dump_gathers")
                utils.safe_makedirs(self.params.gathers_dir)

            utils.safe_makedirs(pd_dir)
            utils.safe_makedirs(expt_ref_dir)

        COMM.barrier()
        input_lines = COMM.bcast(input_lines)
        best_models = COMM.bcast(best_models)

        if self.params.ignore_existing:
            exp_names_already =None
            refl_names_already = None
            if COMM.rank==0:
                exp_names_already = {os.path.basename(f) for f in glob.glob("%s/expers/rank*/*.expt" % self.params.outdir)}
                refl_names_already = {os.path.basename(f) for f in glob.glob("%s/refls/rank*/*.refl" % self.params.outdir)}
            exp_names_already = COMM.bcast(exp_names_already)
            refl_names_already = COMM.bcast(refl_names_already)

        exp_gatheredRef_spec = []  # optional list of expt, refls, spectra
        trefs = []
        this_rank_dfs = []  # dataframes storing the modeling results for each shot
        this_rank_hopper_stats = []  # per-shot sigZ and pred_offset tracking
        this_rank_Elints = ExperimentList()
        this_rank_Rints = None
        this_rank_Ridxs = None
        chunk_id = 0
        shots_per_chunk=self.params.shots_per_chunk
        try:
            from score_trainer import roi_check
            CHECKER = roi_check.roiCheck()
        except:
            CHECKER = None
        for i_shot, line in enumerate(input_lines):
            time_to_refine = time.time()
            if i_shot == self.params.max_process:
                break
            if i_shot % COMM.size != COMM.rank:
                continue

            logging.info("COMM.rank %d on shot  %d / %d" % (COMM.rank, i_shot + 1, len(input_lines)))
            exp, ref, exp_idx, spec = hopper_utils.split_line(line)

            if self.params.ignore_existing:
                basename = os.path.splitext(os.path.basename(exp))[0]
                exists = False
                for ii in [i_shot, 0]:
                    opt_exp = "%s_%s_%d_%d.expt" % (self.params.tag, basename, exp_idx, ii)
                    opt_refl = opt_exp.replace(".expt", ".refl")
                    if opt_exp in exp_names_already and opt_refl in refl_names_already:
                        exists = True
                        break
                if exists:
                    print("Found existing!! %d" % i_shot)
                    continue

            best = None
            if best_models is not None:
                if "exp_idx" not in list(best_models):
                    best_models["exp_idx"]= 0
                best = best_models.query("exp_name=='%s'" % os.path.abspath(exp)).query("exp_idx==%d" % exp_idx)

                # Fallback: match via opt_exp_name (for expanded input where exp paths differ)
                if len(best) != 1 and "opt_exp_name" in list(best_models):
                    best = best_models.query("opt_exp_name=='%s'" % os.path.abspath(exp)).query("exp_idx==%d" % exp_idx)

                if len(best) != 1:
                    n_found = len(best)
                    best = None
                    MAIN_LOGGER.info("Expected exactly 1 entry for exp %s in best pickle %s but found %d entries" % (exp, self.params.best_pickle, n_found))
            # Track original experiment name for cross-cycle shot matching.
            # When warm-starting from a pickle where we matched via opt_exp_name,
            # the pickle's exp_name is the original — use it as the canonical ID.
            orig_exp_name = exp
            if best is not None and "exp_name" in list(best):
                _orig = best["exp_name"].values[0]
                if _orig:
                    orig_exp_name = _orig

            self.params.simulator.spectrum.filename = spec
            Modeler = hopper_utils.DataModeler(self.params)
            Modeler.exper_name = exp
            Modeler.orig_exp_name = orig_exp_name
            Modeler.exper_idx = exp_idx
            Modeler.refl_name = ref
            Modeler.rank = COMM.rank
            Modeler.i_shot = i_shot
            # Optional prediction step?
            if self.params.load_data_from_refls:
                gathered = Modeler.GatherFromReflectionTable(exp, ref, sg_symbol=self.params.space_group)
            else:
                gathered = Modeler.GatherFromExperiment(exp, ref,
                                                        remove_duplicate_hkl=self.params.remove_duplicate_hkl,
                                                        sg_symbol=self.params.space_group,
                                                        exp_idx=exp_idx)
            if not gathered:
                logging.warning("No refls in %s; CONTINUE; COMM.rank=%d" % (ref, COMM.rank))
                continue
            MAIN_LOGGER.info("Modeling %s (%d refls)" % (exp, len(Modeler.refls)))
            if self.params.dump_gathers:
                output_name = os.path.splitext(os.path.basename(exp))[0]
                output_name += "_withData.refl"
                output_name = os.path.join(self.params.gathers_dir, output_name)
                Modeler.dump_gathered_to_refl(output_name, do_xyobs_sanity_check=True)  # NOTE do this is modelin strong spots only
                if self.params.test_gathered_file:
                    all_data = Modeler.all_data.copy()
                    all_roi_id = Modeler.roi_id.copy()
                    all_bg = Modeler.all_background.copy()
                    all_trusted = Modeler.all_trusted.copy()
                    all_pids = np.array(Modeler.pids)
                    all_rois = np.array(Modeler.rois)
                    new_Modeler = hopper_utils.DataModeler(self.params)
                    assert new_Modeler.GatherFromReflectionTable(exp, output_name)
                    assert np.allclose(new_Modeler.all_data, all_data)
                    assert np.allclose(new_Modeler.all_background, all_bg)
                    assert np.allclose(new_Modeler.rois, all_rois)
                    assert np.allclose(new_Modeler.pids, all_pids)
                    assert np.allclose(new_Modeler.all_trusted, all_trusted)
                    assert np.allclose(new_Modeler.roi_id, all_roi_id)

                exp_gatheredRef_spec.append((exp, os.path.abspath(output_name), spec))
                if self.params.only_dump_gathers:
                    continue

            _multipanel_remap = False
            if self.params.refiner.reference_geom is not None:
                ref_El = ExperimentListFactory.from_json_file(self.params.refiner.reference_geom, check_format=False)
                ref_detector = ref_El[0].detector
                from simtbx.diffBragg.multipanel_utils import is_multipanel_reference, remap_modeler_to_multipanel
                if is_multipanel_reference(ref_detector, Modeler.E.detector):
                    mono_detector = Modeler.E.detector
                    remap_modeler_to_multipanel(Modeler, mono_detector, ref_detector)
                    _multipanel_remap = True
                else:
                    Modeler.E.detector = ref_detector
                # Load optimized goniometer axis from reference_geom (for macro-cycling)
                if ref_El[0].goniometer is not None:
                    ref_axis = list(ref_El[0].goniometer.get_rotation_axis())
                    self.params.simulator.gonio.axis = ref_axis
                    if i_shot == 0:
                        MAIN_LOGGER.info("Loaded gonio axis from reference_geom: (%.6f, %.6f, %.6f)" % tuple(ref_axis))

            # here we support inputting an experiment list with multiple crystals
            # the first crystal in the exp list is used to instantiate a diffBragg instance,
            # the remaining crystals are added to the sim_data instance for use during hopper_utils modeling
            # best pickle is not supported yet for multiple crystals
            # also, if number of crystals is >1 , then the params.number_of_xtals flag will be overridden
            exp_list = ExperimentListFactory.from_json_file(exp, False)
            xtals = exp_list.crystals()  # TODO: fix as this is broken now that we allow multi image experiments
            if self.params.consider_multicrystal_shots and len(xtals) > 1:
                assert best is None, "cannot pass best pickle if expt list has more than one crystal"
                assert self.params.number_of_xtals==1, "if expt list has more than one xtal, leave number_of_xtals as the default"
                self.params.number_of_xtals = len(xtals)
                MAIN_LOGGER.debug("Found %d xtals with unit cells:" %len(xtals))
                for xtal in xtals:
                    MAIN_LOGGER.debug("%.4f %.4f %.4f %.4f %.4f %.4f" % xtal.get_unit_cell().parameters())
            if self.params.record_device_timings and COMM.rank >0:
                self.params.record_device_timings = False  # only record for rank 0 otherwise there's too much output
            if self.params.simulator.gonio.delta_phi is None:
                self.params.simulator.gonio.delta_phi = self.params.init.gonio_angle
            SIM = hopper_utils.get_simulator_for_data_modelers(Modeler)

            # Set up panel_group_from_id for multi-panel per-shot refinement
            if _multipanel_remap:
                from simtbx.diffBragg.multipanel_utils import setup_panel_group_from_id
                setup_panel_group_from_id(SIM, len(Modeler.E.detector))
                from simtbx.diffBragg.refiners.geometry import set_group_id_slices
                set_group_id_slices(Modeler, SIM.panel_group_from_id)
            Modeler.set_parameters_for_experiment(best)
            MAIN_LOGGER.debug("Set parameters for experiment")
            Modeler.Umatrices = [Modeler.E.crystal.get_U()]

            # TODO: move this to SimulatorFromExperiment
            # TODO: fix multi crystal shot mode
            if best is not None and "other_spotscales" in list(best) and "other_Umats" in list(best):
                Modeler.Umatrices[0] = Modeler.E.get_U()
                assert len(xtals) == len(best.other_spotscales.values[0])+1
                for i_xtal in range(1, len(xtals),1):
                    scale_xt = best.other_spotscales.values[0][i_xtal]
                    Umat_xt = best.other_Umats.values[0][i_xtal]
                    Modeler.Umatrices[i_xtal] = Umat_xt
                    Modeler.P["G_xtal%d" %i_xtal] = scale_xt

            SIM.D.store_ave_wavelength_image = self.params.store_wavelength_images
            if self.params.refiner.verbose is not None and COMM.rank==0:
                SIM.D.verbose = self.params.refiner.verbose
            if self.params.profile:
                SIM.record_timings = True
            if self.params.use_float32:
                Modeler.all_data = Modeler.all_data.astype(np.float32)
                Modeler.all_background = Modeler.all_background.astype(np.float32)

            SIM.D.device_Id = self.dev

            # Auto-estimate G from data/model ratio
            auto_G = getattr(self.params.init, 'auto_G', True)
            fix_G = self.params.fix.G
            MAIN_LOGGER.debug("auto_G=%s, fix.G=%s" % (auto_G, fix_G))
            if auto_G and not fix_G:
                G_est = hopper_utils.estimate_spot_scale(Modeler, SIM)
                if G_est is not None and G_est > 0:
                    if G_est > 1e30:
                        MAIN_LOGGER.error("Auto-G estimate %.4g is unreasonably large (model ~0). "
                                          "Skipping this shot — check beam/crystal/structure factors." % G_est)
                        continue
                    Modeler.P["G_xtal0"].init = G_est
                    MAIN_LOGGER.info("Set G_xtal0.init = %.4g" % G_est)

            nparam = len(Modeler.P)
            if SIM.refining_Fhkl:
                nparam += SIM.Num_ASU*SIM.num_Fhkl_channels
            x0 = [1] * nparam
            tref = time.time()
            MAIN_LOGGER.info("Beginning refinement of shot %d / %d" % (i_shot+1, len(input_lines)))

            # Kernel debug: log full C++ state at hopper init (inside model(), right before add_diffBragg_spots)
            from simtbx.diffBragg.diffbragg_state import should_snapshot
            _kd_label = None
            if should_snapshot(self.params, i_shot, COMM.rank):
                _kd_label = "HOPPER_INIT (rank=%d, i_shot=%d, %s)" % (COMM.rank, i_shot, os.path.basename(exp))
                try:
                    hopper_utils.model(x0, Modeler, SIM, compute_grad=False, kernel_debug=_kd_label)
                except Exception as _e:
                    print("KERNEL DEBUG hopper_init failed: %s" % _e, flush=True)

            # Compute init sigZ split (original vs predicted) before refinement begins
            _init_sigZ_orig = _init_sigZ_pred = None
            try:
                _tf_init = hopper_utils.target_func(
                    x0, None, Modeler, SIM, compute_grad=False, return_all_zscores=True)
                _init_sigZ_all = float(_tf_init[4])
                _init_zs = _tf_init[6]
                _has_pred_init = (hasattr(Modeler, 'is_predicted') and Modeler.is_predicted is not None
                                  and Modeler.is_predicted.any())
                if _has_pred_init:
                    _pred_perpix_init = Modeler.is_predicted[Modeler.roi_id]
                    _trusted_init = Modeler.all_trusted
                    _orig_m = _trusted_init & ~_pred_perpix_init
                    _pred_m = _trusted_init & _pred_perpix_init
                    if _orig_m.any():
                        _init_sigZ_orig = float(np.std(_init_zs[_orig_m]))
                    if _pred_m.any():
                        _init_sigZ_pred = float(np.std(_init_zs[_pred_m]))
            except Exception:
                pass

            try:
                x = Modeler.Minimize(x0, SIM, i_shot=i_shot)
                # Capture initial sigZ from first refinement stage
                # (perRoi_finish will overwrite Modeler.target with a new TargetFunc)
                _stage1_init_sigZ = None
                if hasattr(Modeler, 'target') and Modeler.target is not None and Modeler.target.all_sigZ:
                    _stage1_init_sigZ = float(Modeler.target.all_sigZ[0])
                for i_rep in range(self.params.filter_after_refinement.max_attempts):
                    if not self.params.filter_after_refinement.enable:
                        continue
                    final_sigz = Modeler.target.all_sigZ[-1]
                    niter = len(Modeler.target.all_sigZ)
                    too_few_iter = niter < self.params.filter_after_refinement.min_prev_niter
                    too_high_sigz = final_sigz > self.params.filter_after_refinement.max_prev_sigz
                    if too_few_iter or too_high_sigz:
                        Modeler.filter_pixels(self.params.filter_after_refinement.threshold)
                        x = Modeler.Minimize(x0, SIM, i_shot=i_shot)

                # State snapshot: after main refinement
                from simtbx.diffBragg.diffbragg_state import should_snapshot, capture_hopper_state, write_state_snapshot
                if should_snapshot(self.params, i_shot, COMM.rank):
                    _state = capture_hopper_state(x, Modeler, SIM, self.params, i_shot, "hopper_main", COMM.rank)
                    if hasattr(Modeler, 'target') and Modeler.target is not None and Modeler.target.all_sigZ:
                        _state["diagnostics"] = {
                            "first_sigZ": Modeler.target.all_sigZ[0],
                            "last_sigZ": Modeler.target.all_sigZ[-1],
                            "first_resid": Modeler.target.all_f[0],
                            "last_resid": Modeler.target.all_f[-1],
                            "n_iterations": len(Modeler.target.all_f),
                        }
                    write_state_snapshot(_state, self.params.outdir, "hopper_main_rank%d_shot%d" % (COMM.rank, i_shot))

                if self.params.perRoi_finish:
                    old_params = deepcopy(self.params)
                    old_P = deepcopy(Modeler.P)
                    # fix all of the refinement variables except for perRoiScale
                    for fix_name in dir(self.params.fix):
                        if fix_name.startswith("_"):
                            continue
                        cur_val = getattr(self.params.fix, fix_name, None)
                        if isinstance(cur_val, (list, tuple)):
                            # List-type fix params (RotXYZ, panel_rotations, panel_translations)
                            setattr(self.params.fix, fix_name, [1]*len(cur_val))
                        elif not callable(cur_val):
                            setattr(self.params.fix, fix_name, True)
                    self.params.geometry.fix.panel_rotations=[1,1,1]
                    self.params.geometry.fix.panel_translations=[1,1,1]
                    self.params.fix.perRoiScale = False
                    self.params.niter=0
                    Modeler.params = self.params
                    Modeler.set_parameters_for_experiment(best)
                    new_x = np.array([1.]*len(Modeler.P))
                    for name in old_P:
                        new_p = Modeler.P[name]
                        old_p = old_P[name]
                        new_x[new_p.xpos] = x[old_p.xpos]
                        if not name.startswith("scale_roi"):
                            assert not new_p.refine

                    x = Modeler.Minimize(new_x, SIM, i_shot=i_shot)

                    # State snapshot: after perRoi_finish
                    if should_snapshot(self.params, i_shot, COMM.rank):
                        _state = capture_hopper_state(x, Modeler, SIM, self.params, i_shot, "hopper_perRoi", COMM.rank)
                        write_state_snapshot(_state, self.params.outdir, "hopper_perRoi_rank%d_shot%d" % (COMM.rank, i_shot))

                    # set the roi_scale factors here on the reflection tables:
                    # reset the params
                    self.params = old_params

                    # Filter poor fits
                    if self.params.roi.filter_scores.enable and CHECKER is not None:
                        Modeler.best_model, _ = hopper_utils.model(x, Modeler, SIM, compute_grad=False)
                        Modeler.best_model_includes_background = False
                        num_good = Modeler.filter_bad_scores(CHECKER)
                        if num_good == 0:
                            Modeler.clean_up(SIM)
                            continue

                    # TODO set refined refl scale factors here
                    # Then repeat minimization
                    self.params.use_perRoiScale = True
                    self.params.fix.perRoiScale = True
                    Modeler.params = self.params
                    old_P = deepcopy(Modeler.P)
                    Modeler.set_parameters_for_experiment(best)
                    new_x = np.array([1.] * len(Modeler.P))
                    for name in Modeler.P:
                        new_p = Modeler.P[name]
                        old_p = old_P[name]
                        new_x[new_p.xpos] = x[old_p.xpos]
                    x = Modeler.Minimize(new_x, SIM, i_shot=i_shot)

                    # State snapshot: after final refinement
                    if should_snapshot(self.params, i_shot, COMM.rank):
                        _state = capture_hopper_state(x, Modeler, SIM, self.params, i_shot, "hopper_final", COMM.rank)
                        # Include sigma Z from the refinement trace
                        if hasattr(Modeler, 'target') and Modeler.target is not None and Modeler.target.all_sigZ:
                            _state["diagnostics"] = {
                                "first_sigZ": Modeler.target.all_sigZ[0],
                                "last_sigZ": Modeler.target.all_sigZ[-1],
                                "first_resid": Modeler.target.all_f[0],
                                "last_resid": Modeler.target.all_f[-1],
                                "n_iterations": len(Modeler.target.all_f),
                            }
                        write_state_snapshot(_state, self.params.outdir, "hopper_final_rank%d_shot%d" % (COMM.rank, i_shot))

                    #use_cuda = os.environ["DIFFBRAGG_USE_CUDA"]
                    #del os.environ["DIFFBRAGG_USE_CUDA"]
                    #tracked_fhkl = utils.track_fhkl(Modeler, SIM)
                    #os.environ['DIFFBRAGG_USE_CUDA'] = use_cuda
                    #self.params.fix.perRoiScale = True
                    #Modeler.params.fix.perRoiScale = True

            except StopIteration:
                x = Modeler.target.x0
                _stage1_init_sigZ = None
                if hasattr(Modeler, 'target') and Modeler.target is not None and Modeler.target.all_sigZ:
                    _stage1_init_sigZ = float(Modeler.target.all_sigZ[0])
            tref = time.time()-tref
            sigz = niter = None
            try:
                sigz, niter, _ = Modeler.get_best_hop()
            except Exception:
                pass

            trefs.append(tref)
            print_s = "Finished refinement of shot %d / %d in %.4f sec. (rank mean t/im=%.4f sec.)" \
                        % (i_shot+1, len(input_lines), tref, np.mean(trefs))
            if sigz is not None and niter is not None:
                print_s += " Ran %d iterations. Final sigmaZ = %.1f. %d ROIs." % (niter, sigz, len(Modeler.rois))
            if COMM.rank==0:
                MAIN_LOGGER.info(print_s)
            else:
                MAIN_LOGGER.debug(print_s)
            if self.params.profile:
                SIM.D.show_timings(COMM.rank)

            dbg = self.params.debug_mode
            if dbg and COMM.rank > 0 and self.params.debug_mode_rank0_only:
                dbg = False
            save_refl = dbg or self.params.save_perRoiScale
            save_expt = dbg or self.params.save_perRoiScale or self.params.geometry.optimize
            shot_df = Modeler.save_up(x, SIM, rank=COMM.rank, i_shot=i_shot,
                            save_fhkl_data=dbg, save_refl=save_refl, save_modeler_file=dbg,
                            save_sim_info=dbg, save_pandas=dbg, save_traces=dbg, save_expt=save_expt,
                            checker=CHECKER)

            # Per-shot sigZ and pred_offset tracking
            # Metrics are computed WITHOUT perRoiScale so they are comparable
            # to geometry refinement metrics (which don't have per-ROI scales).
            init_sigZ = _stage1_init_sigZ  # already clean: perRoiScale=1.0 at first eval
            final_sigZ = None

            # Recompute final sigZ without perRoiScale for apples-to-apples comparison
            sigZ_orig = sigZ_pred = None
            try:
                _saved_roiScales = Modeler.roiScalesPerPix
                Modeler.roiScalesPerPix = 1
                _tf_result = hopper_utils.target_func(
                    x, None, Modeler, SIM, compute_grad=False, return_all_zscores=True)
                _noRoi_sigZ = _tf_result[4]
                _zscore_perpix = _tf_result[6]
                final_sigZ = float(_noRoi_sigZ)

                # Split sigZ by original vs predicted ROIs
                _has_pred = (hasattr(Modeler, 'is_predicted') and Modeler.is_predicted is not None
                             and Modeler.is_predicted.any())
                if _has_pred:
                    _pred_perpix = Modeler.is_predicted[Modeler.roi_id]
                    _trusted = Modeler.all_trusted
                    _orig_mask = _trusted & ~_pred_perpix
                    _pred_mask = _trusted & _pred_perpix
                    if _orig_mask.any():
                        sigZ_orig = float(np.std(_zscore_perpix[_orig_mask]))
                    if _pred_mask.any():
                        sigZ_pred = float(np.std(_zscore_perpix[_pred_mask]))

                # Kernel debug: log full C++ state at hopper final (inside model(), right before add_diffBragg_spots)
                if should_snapshot(self.params, i_shot, COMM.rank):
                    _kd_final = "HOPPER_FINAL (rank=%d, i_shot=%d, %s)" % (COMM.rank, i_shot, os.path.basename(exp))
                    try:
                        hopper_utils.model(x, Modeler, SIM, compute_grad=False, kernel_debug=_kd_final)
                    except Exception as _e:
                        print("KERNEL DEBUG hopper_final failed: %s" % _e, flush=True)

                Modeler.roiScalesPerPix = _saved_roiScales
            except Exception:
                # Fallback to last target eval (includes perRoiScale)
                if hasattr(Modeler, 'target') and Modeler.target is not None and Modeler.target.all_sigZ:
                    final_sigZ = float(Modeler.target.all_sigZ[-1])

            init_pred_offset = final_pred_offset = None
            try:
                from simtbx.diffBragg.refiners.geometry import get_dist_from_R

                # Final pred_offset: from refined model (best_model set by save_up)
                _new_crystal = hopper_utils.update_crystal_from_x(Modeler, SIM, x)
                _new_exp = deepcopy(Modeler.E)
                _new_exp.crystal = _new_crystal
                _new_exp.detector = hopper_utils.update_detector_from_x(Modeler, SIM, x)
                _new_refl = hopper_utils.get_new_xycalcs(Modeler, _new_exp, x=x)
                _dists = get_dist_from_R(_new_refl)
                final_pred_offset = float(np.median(_dists))

                # Initial pred_offset: re-evaluate model at x0 (warm-start params)
                _saved_best = Modeler.best_model.copy()
                _init_x = np.array([1.0] * len(x))
                _init_model, _ = hopper_utils.model(_init_x, Modeler, SIM, compute_grad=False)
                Modeler.best_model = _init_model
                Modeler.best_model_includes_background = False
                _init_crystal = hopper_utils.update_crystal_from_x(Modeler, SIM, _init_x)
                _init_exp = deepcopy(Modeler.E)
                _init_exp.crystal = _init_crystal
                _init_exp.detector = hopper_utils.update_detector_from_x(Modeler, SIM, _init_x)
                _init_refl = hopper_utils.get_new_xycalcs(Modeler, _init_exp, x=_init_x)
                _dists = get_dist_from_R(_init_refl)
                init_pred_offset = float(np.median(_dists))
                Modeler.best_model = _saved_best  # restore
            except Exception:
                pass

            shot_df["init_sigZ"] = init_sigZ
            shot_df["final_sigZ"] = final_sigZ
            shot_df["init_pred_offset"] = init_pred_offset
            shot_df["final_pred_offset"] = final_pred_offset
            _n_trusted = int(Modeler.all_trusted.sum()) if hasattr(Modeler, 'all_trusted') else None
            _n_predicted = int(Modeler.is_predicted.sum()) if hasattr(Modeler, 'is_predicted') and Modeler.is_predicted is not None else 0
            _shot_id = os.path.basename(orig_exp_name)
            if exp_idx > 0:
                _shot_id = "%s:%d" % (_shot_id, exp_idx)
            this_rank_hopper_stats.append({
                "shot_id": _shot_id,
                "init_sigZ": init_sigZ, "final_sigZ": final_sigZ,
                "init_sigZ_orig": _init_sigZ_orig, "init_sigZ_pred": _init_sigZ_pred,
                "sigZ_orig": sigZ_orig, "sigZ_pred": sigZ_pred,
                "init_pred_offset": init_pred_offset, "final_pred_offset": final_pred_offset,
                "n_rois": len(Modeler.rois), "n_trusted": _n_trusted,
                "n_predicted": _n_predicted,
            })

            # Dump post-refinement gathered refls for geometry refinement
            if self.params.geometry.optimize:
                geom_gathers_dir = os.path.join(self.params.outdir, "geom_gathers")
                utils.safe_makedirs(geom_gathers_dir)
                geom_output_name = os.path.splitext(os.path.basename(exp))[0] + "_geomData.refl"
                geom_output_name = os.path.join(geom_gathers_dir, geom_output_name)

                per_roi_scales = None
                if hasattr(Modeler, 'P') and (not self.params.fix.perRoiScale or self.params.use_perRoiScale):
                    per_roi_scales = {}
                    for roi_id in range(len(Modeler.rois)):
                        pname = "scale_roi%d" % roi_id
                        if pname in Modeler.P:
                            p = Modeler.P[pname]
                            per_roi_scales[roi_id] = p.get_val(x[p.xpos])

                Modeler.dump_gathered_to_refl(geom_output_name, per_roi_scales=per_roi_scales)

                # State snapshot: geometry gather handoff
                if should_snapshot(self.params, i_shot, COMM.rank):
                    _state = capture_hopper_state(x, Modeler, SIM, self.params, i_shot, "hopper_geomgather", COMM.rank)
                    if per_roi_scales is not None:
                        _state["scale"]["per_roi_geomgather"] = {str(k): v for k, v in per_roi_scales.items()}
                    write_state_snapshot(_state, self.params.outdir, "hopper_geomgather_rank%d_shot%d" % (COMM.rank, i_shot))

                shot_df["geom_ref"] = os.path.abspath(geom_output_name)
                shot_df["geom_exp"] = shot_df["opt_exp_name"]
                shot_df["geom_exp_idx"] = 0

            #if self.params.predictions.integrate_phil is not None:
            do_integrate = self.params.predictions.integrate_phil is not None
            Rstrong = None
            #Modeler.params.predictions.threshold = 10
            #Modeler.params.predictions.use_peak_detection = True
            #Modeler.params.predictions.use_diffBragg_mtz = True
            #pid = Modeler.pids[0]
            #x1, x2, y1, y2 = Modeler.rois[0]
            #fast = int((x2 + x1) / 2)
            #slow = int((y2 + y1) / 2)
            #Modeler.params.predictions.printout_pix = (pid, fast, slow)
            if do_integrate:
                pred_out = predictions.get_predict(Modeler.E,
                                Rstrong, Modeler.params, dev=self.dev, df=shot_df, filter_dupes=True, return_pix=True)
                pred = imgs = None
                if pred_out is None:
                    Modeler.clean_up(SIM)
                    continue
                pred, imgs = pred_out

            pfs = Modeler.pan_fast_slow.as_numpy_array()
            p, f, s = pfs[0::3], pfs[1::3], pfs[2::3]
            if do_integrate:
                predict_model = imgs[p, s, f]
            predict_subimgs = []
            hopper_subimgs = []

            ccs = []
            shoeboxes = []
            scores = []
            Ridx = flex.reflection_table()

            if not hasattr(Modeler, "roi_id_slices"):
                Modeler.set_slices("roi_id")
            for roi, slc in Modeler.roi_id_slices.items():
                hopper_pix = Modeler.best_model[slc[0]]

                x1, x2, y1, y2 = Modeler.rois[int(roi)]
                ydim = y2 - y1
                xdim = x2 - x1

                hopper_subimg = hopper_pix.reshape((ydim, xdim))

                if do_integrate:
                    predict_pix = predict_model[slc[0]]
                    if len(np.unique(predict_pix)) == 1 or len(np.unique(hopper_pix)) == 1:
                        continue
                    cc = pearsonr(predict_pix, hopper_pix)[0]

                    predict_subimg = predict_pix.reshape((ydim, xdim))

                hopper_trust = Modeler.all_trusted[slc[0]].reshape((ydim,xdim))
                if not np.any(hopper_trust):
                    continue

                hopper_subimgs.append(hopper_subimg)
                hopper_bg = Modeler.all_background[slc[0]].reshape((ydim, xdim))
                if do_integrate:
                    ccs.append(cc)
                    predict_subimgs.append(predict_subimg)

                    sb = Shoebox((x1, x2, y1, y2, 0, 1))
                    sb.allocate()
                    sb.data = flex.float(np.ascontiguousarray(hopper_subimg[None]))
                    sb.background = flex.float(np.ascontiguousarray(hopper_bg[None]))

                    dials_mask = np.zeros((ydim, xdim)).astype(np.int32)
                    dials_mask[hopper_trust] = dials_mask[hopper_trust] + MaskCode.Valid
                    braggMask = (hopper_subimg > 10) & hopper_trust
                    dials_mask[braggMask] = dials_mask[braggMask] + MaskCode.Strong
                    bgMask = (hopper_subimg < 1) & hopper_trust
                    dials_mask[bgMask] = dials_mask[bgMask] + MaskCode.Background
                    sb.mask = flex.int(np.ascontiguousarray(dials_mask[None]))
                    shoeboxes.append(sb)

                # TODO: is this correct indexing of the refls ?
                refl_idx = Modeler.all_refls_idx[slc[0]]
                refl_idx = np.unique(refl_idx)
                assert len(refl_idx) == 1
                refl_idx = refl_idx[0]
                refl = Modeler.refls[refl_idx:refl_idx+1]
                # end TODO

                Ridx.extend(refl)
                if CHECKER is not None:
                    data_subimg = Modeler.all_data[slc[0]].reshape((ydim, xdim))
                    score = CHECKER.score(data_subimg, hopper_subimg + hopper_bg)
                    scores.append(score)
            if len(Ridx) == 0:
                Modeler.clean_up(SIM)
                continue

            if scores:
                Ridx["model_score"] = flex.double(scores)
            if shoeboxes and 'shoebox' != Ridx:
                Ridx['shoebox'] = flex.shoebox(shoeboxes)
            Ridx.set_flags(flex.bool(len(Ridx), True), Ridx.flags.indexed)

            Elint = ExperimentList()
            Eref = deepcopy(Modeler.E)
            Eref.crystal.set_A(shot_df.Amats.values[0])
            Eref.detector = SIM.detector
            Elint.append(Eref)
            if do_integrate:
                Elint, Rint = integrate.integrate(self.params.predictions.integrate_phil, Elint, Ridx, pred)
                spot_ev, delpsi_rad, Deff, eta_est = extract_mosaic_parameters_using_lambda_spread(Elint[0], Rint, verbose=False)
                Rint["spot_ev"] = spot_ev
                Rint["delpsical.rad"] = delpsi_rad
                # here is where to do the diffBragg fit of intensity
                if self.params.predictions.fit_intensity_using_diffBragg:
                    old_params = deepcopy(Modeler.params)
                    Modeler.params.roi.centroid = "cal"
                    Modeler.GatherFromExperiment(Modeler.E, Rint, remove_duplicate_hkl=True,
                                                 sg_symbol=Modeler.params.space_group)

                    for fix_name in dir(Modeler.params.fix):
                        if fix_name.startswith("_"):
                            continue
                        setattr(Modeler.params.fix, fix_name, True)
                    Modeler.params.fix.perRoiScale = False

                    # clean up SIM before making prediction SIM
                    if Modeler.params.refiner.debug_pixel_panelfastslow is not None:
                        utils.show_diffBragg_state(SIM.D, Modeler.params.refiner.debug_pixel_panelfastslow)
                    if SIM.D.record_timings:
                        SIM.D.show_timings(COMM.rank)
                    Modeler.clean_up(SIM)

                    SIM2 = hopper_utils.get_simulator_for_data_modelers(Modeler)
                    Modeler.set_parameters_for_experiment(shot_df)
                    new_x = np.array([1.] * len(Modeler.P))
                    Modeler.roiScalesPerPix = 1
                    Fp1 = SIM2.D.Fhkl  # crystal.miller_array
                    Fp1_map = {h: amp for h, amp in zip(Fp1.indices(), Fp1.data())}
                    Modeler.flag_zeroamps_as_untrusted(Fp1_map)
                    x = Modeler.Minimize(new_x, SIM2, i_shot=i_shot)
                    self.params = old_params

                    Modeler.best_model, opt_Jac = hopper_utils.model(x, Modeler, SIM2, compute_grad=True)
                    Modeler.best_model_includes_background = False
                    variance_s = hopper_utils.get_variance_s(Modeler, opt_Jac)


                    Nroi = len(Rint)
                    refl_sel = flex.bool(Nroi, False)
                    db_I = flex.double(Nroi, 0)
                    db_varI = flex.double(Nroi,0)
                    pred_scores = flex.double(Nroi,0)
                    n_not_in_F = 0
                    n_zero = 0
                    n_bad = 0
                    n_out_bound = 0
                    for roi in Modeler.roi_id_unique:
                        slc = Modeler.roi_id_slices[roi]

                        x1,x2,y1,y2 = Modeler.rois[roi]
                        ydim = y2-y1
                        xdim = x2-x1
                        roi_p = Modeler.P["scale_roi%d" % roi]
                        xval = x[roi_p.xpos]
                        scale = roi_p.get_val(xval)
                        test = Modeler.roiScalesPerPix[slc[0]]
                        test = np.unique(test)
                        assert len(test)==1
                        test = test[0]
                        assert scale == test

                        var_s = variance_s[roi_p.xpos]

                        refl_idx = Modeler.all_refls_idx[slc[0]]
                        refl_idx = np.unique(refl_idx)
                        assert len(refl_idx) ==1
                        refl_idx = refl_idx[0]
                        h,k,l = Modeler.refls[int(refl_idx)]['miller_index']
                        assert Modeler.all_nominal_hkl[slc[0]]
                        hkl = int(h), int(k), int(l)
                        if hkl not in Fp1_map:
                            n_not_in_F += 1
                            continue

                        amp = Fp1_map[hkl]
                        if amp==0:
                            n_zero += 1
                            continue
                        I_hkl = amp ** 2
                        var_I = I_hkl ** 2 * var_s
                        if np.isnan(var_I) or np.isinf(var_I):
                            n_bad += 1
                            continue
                        if var_I <= 1e-6 or var_I > 1e16:
                            n_out_bound += 1
                            continue

                        if CHECKER is not None:
                            data_subimg = Modeler.all_data[slc[0]].reshape((ydim, xdim))
                            model_subimg = (Modeler.best_model[slc[0]] + Modeler.all_background[slc[0]]).reshape((ydim,xdim))
                            score = CHECKER.score(data_subimg, model_subimg)
                            pred_scores[refl_idx] = score

                        assert not refl_sel[refl_idx]
                        refl_sel[refl_idx] = True
                        db_I[refl_idx] = I_hkl*scale
                        db_varI[refl_idx] = var_I

                    nrej = n_zero + n_bad + n_not_in_F + n_out_bound
                    MAIN_LOGGER.debug("DB Intensity Fit: Nzero=%d, Nbad=%d, Nmissing=%d, Nout=%d (%d/%d total)"
                                      % (n_zero, n_bad, n_not_in_F, n_out_bound, nrej, len(Modeler.roi_id_unique)))
                    if CHECKER is not None:
                        Rint["pred_scores"] = pred_scores
                    Rint["intensity.diffBragg.value"] = db_I
                    Rint["intensity.diffBragg.variance"] = db_varI
                    Rint = Rint.select(refl_sel)
                    assert not np.any(np.isnan(np.sqrt(Rint['intensity.diffBragg.variance'])))
                    Modeler.clean_up(SIM2)
            else:
                if Modeler.params.refiner.debug_pixel_panelfastslow is not None:
                    # TODO separate diffBragg logger
                    utils.show_diffBragg_state(SIM.D, Modeler.params.refiner.debug_pixel_panelfastslow)

                # TODO verify this works:
                if SIM.D.record_timings:
                    SIM.D.show_timings(COMM.rank)
                Modeler.clean_up(SIM)
                del SIM.D  # TODO: is this necessary ?

            if Modeler.E.identifier is not None:
                ident = Modeler.E.identifier
            else:
                ident = create_experiment_identifier(Modeler.E, Modeler.exper_name, Modeler.exper_idx)

            Rint_id = Ridx_id = len(this_rank_Elints)
            if do_integrate:
                eid_int = Rint.experiment_identifiers()
                for k in eid_int.keys():
                    del eid_int[k]
                eid_int[Rint_id] = ident
                Rint['id'] = flex.int(len(Rint), Rint_id)

            eid_idx = Ridx.experiment_identifiers()
            for k in eid_idx.keys():
                del eid_idx[k]
            eid_idx[Ridx_id] = ident
            Ridx['id'] = flex.int(len(Ridx), Ridx_id)

            Elint[0].identifier = ident

            # verify the prediction is identical to the best model
            shot_df['identifier'] = ident
            shot_df["hopper_time"] = time.time()-time_to_refine
            shot_df["hopper_line"] = line  # exp_ref_spec file line for re-running
            if scores:
                shot_df["scores"] = np.mean(scores)

            this_rank_dfs.append(shot_df)
            this_rank_Elints.extend(Elint)
            if do_integrate:
                if 'shoebox' in Rint:
                    del Rint['shoebox']
                if this_rank_Rints is None:
                    this_rank_Rints = Rint
                else:
                    this_rank_Rints.extend(Rint)
            #if "shoebox" in Ridx:
            #    del Ridx['shoebox']
            #TODO fill in new xyzcal.px column ?
            if this_rank_Ridxs is None:
                this_rank_Ridxs = Ridx
            else:
                this_rank_Ridxs.extend(Ridx)

            if len(this_rank_dfs) == shots_per_chunk:
                save_composite_files(this_rank_dfs, this_rank_Elints, this_rank_Ridxs, this_rank_Rints,
                                     pd_dir, expt_ref_dir, chunk_id)
                chunk_id += 1
                this_rank_dfs = []  # dataframes storing the modeling results for each shot
                this_rank_Elints = ExperimentList()
                this_rank_Rints = None
                this_rank_Ridxs = None

        if self.params.dump_gathers and self.params.gathered_output_file is not None:
            exp_gatheredRef_spec = COMM.reduce(exp_gatheredRef_spec)
            if COMM.rank == 0:
                o = open(self.params.gathered_output_file, "w")
                for e, r, s in exp_gatheredRef_spec:
                    if s is not None:
                        o.write("%s %s %s\n" % (e,r,s))
                    else:
                        o.write("%s %s\n" % (e,r))
                o.close()

        save_composite_files(this_rank_dfs, this_rank_Elints, this_rank_Ridxs, this_rank_Rints,
                             pd_dir, expt_ref_dir, chunk_id)

        # Write hopper summary (sigZ and pred_offset tracking)
        all_hopper_stats = COMM.gather(this_rank_hopper_stats)
        if COMM.rank == 0:
            import json
            flat_stats = [s for rank_stats in all_hopper_stats if rank_stats for s in rank_stats]
            if flat_stats:
                _isZ = [s["init_sigZ"] for s in flat_stats if s["init_sigZ"] is not None]
                _fsZ = [s["final_sigZ"] for s in flat_stats if s["final_sigZ"] is not None]
                _ipo = [s["init_pred_offset"] for s in flat_stats if s["init_pred_offset"] is not None]
                _fpo = [s["final_pred_offset"] for s in flat_stats if s["final_pred_offset"] is not None]
                hopper_summary = {
                    "n_shots": len(flat_stats),
                    "median_init_sigZ": float(np.median(_isZ)) if _isZ else None,
                    "median_final_sigZ": float(np.median(_fsZ)) if _fsZ else None,
                    "median_init_pred_offset": float(np.median(_ipo)) if _ipo else None,
                    "median_final_pred_offset": float(np.median(_fpo)) if _fpo else None,
                    "per_shot": [
                        {"shot_id": s["shot_id"], "n_rois": s["n_rois"],
                         "n_trusted": s.get("n_trusted"),
                         "n_predicted": s.get("n_predicted", 0),
                         "init_sigZ": s["init_sigZ"], "final_sigZ": s["final_sigZ"],
                         "init_sigZ_orig": s.get("init_sigZ_orig"), "init_sigZ_pred": s.get("init_sigZ_pred"),
                         "sigZ_orig": s.get("sigZ_orig"), "sigZ_pred": s.get("sigZ_pred"),
                         "init_pred_offset": s["init_pred_offset"],
                         "final_pred_offset": s["final_pred_offset"]}
                        for s in flat_stats
                    ],
                }
                summary_path = os.path.join(self.params.outdir, "hopper_summary.json")
                with open(summary_path, "w") as fh:
                    json.dump(hopper_summary, fh, indent=2)
                MAIN_LOGGER.info("Hopper summary: %d shots, sigZ %.4f->%.4f, pred_offset %.4f->%.4f"
                                 % (hopper_summary["n_shots"],
                                    hopper_summary["median_init_sigZ"] or 0,
                                    hopper_summary["median_final_sigZ"] or 0,
                                    hopper_summary["median_init_pred_offset"] or 0,
                                    hopper_summary["median_final_pred_offset"] or 0))

        # Phase 2: geometry refinement on the pool of shots
        if self.params.geometry.optimize:
            COMM.barrier()
            if COMM.rank == 0:
                MAIN_LOGGER.info("Starting geometry refinement on %d ranks" % COMM.size)

            # Point geometry input at the pickles we just wrote
            pkl_glob = os.path.join(pd_dir, "hopper_results_rank*_chunk*.pkl")
            self.params.geometry.input_pkl_glob = pkl_glob
            self.params.geometry.input_pkl = None

            # Load data from the gathered refls (exact same data/bg/masks as hopper)
            self.params.refiner.load_data_from_refl = True

            # Crystal params for geometry refinement are controlled by geometry.fix.*
            # Defaults are all fixed (True/1,1,1) - override via phil or command line
            # e.g., geometry.fix.RotXYZ=0,0,0 to refine crystal orientation

            # Enable restraints for geometry refinement if specified
            if self.params.geometry.use_restraints:
                self.params.use_restraints = True

            # --- Multi-panel conversion (rank 0 converts, all ranks use results) ---
            if self.params.geometry.multipanel:
                if COMM.rank == 0:
                    MAIN_LOGGER.info("Converting gathers to multi-panel format...")
                    from simtbx.diffBragg.multipanel_utils import convert_gathers_to_multipanel
                    fnames = sorted(glob.glob(pkl_glob))
                    df = pandas.concat([pandas.read_pickle(f) for f in fnames])
                    df = convert_gathers_to_multipanel(self.params, df)
                    # Re-save the updated pandas pickle
                    mp_pkl = os.path.join(self.params.outdir, "multipanel", "hopper_multipanel.pkl")
                    df.to_pickle(mp_pkl)
                    self.params.geometry.input_pkl = mp_pkl
                    self.params.geometry.input_pkl_glob = None
                COMM.barrier()
                self.params = COMM.bcast(self.params if COMM.rank == 0 else None)

            from simtbx.diffBragg.refiners.geometry import geom_min
            geom_min(self.params)

            if COMM.rank == 0:
                det_name = self.params.geometry.optimized_detector_name
                MAIN_LOGGER.info("Geometry refinement complete. Optimized detector: %s" % det_name)
                from dxtbx.model import ExperimentList as EL_read
                opt_el = EL_read.from_file(det_name, check_format=False)
                opt_det = opt_el[0].detector
                for ipan in range(len(opt_det)):
                    pan = opt_det[ipan]
                    orig = pan.get_origin()
                    MAIN_LOGGER.info("  Panel %d origin: (%.4f, %.4f, %.4f) mm" % (ipan, orig[0], orig[1], orig[2]))

            # Broadcast optimized detector path to all ranks
            det_name = COMM.bcast(det_name if COMM.rank == 0 else None)

            # For macro-cycles: update reference geometry for next cycle
            if hasattr(self.params.geometry, 'macro_cycles') and self.params.geometry.macro_cycles > 1:
                # Set the optimized detector as reference for future refinements
                self.params.refiner.reference_geom = det_name
                if COMM.rank == 0:
                    MAIN_LOGGER.info("Updated reference geometry to: %s for potential next macro-cycle" % det_name)
                    MAIN_LOGGER.info("To use in next hopper run, add: refiner.reference_geom=%s" % det_name)


def save_composite_files(dfs, expts, refls, refls_int, pd_dir, exp_ref_dir, chunk=0):
    df_name = os.path.join(pd_dir, "hopper_results_rank%d_chunk%d.pkl" % (COMM.rank, chunk))
    expt_name = os.path.join(exp_ref_dir, "hopper_rank%d_chunk%d_integrated.expt" % (COMM.rank,chunk))
    int_name = os.path.join(exp_ref_dir, "hopper_rank%d_chunk%d_integrated.refl" % (COMM.rank,chunk))
    idx_name = os.path.join(exp_ref_dir, "hopper_rank%d_chunk%d.refl" % (COMM.rank,chunk))
    if dfs:
        this_rank_dfs = pandas.concat(dfs).reset_index(drop=True)
        this_rank_dfs.to_pickle(df_name)
    if expts:
        expts.as_file(expt_name)
    if refls:
        refls.as_file(idx_name)
    if refls_int:
        refls_int.as_file(int_name)

        #MAIN_LOGGER.info("MPI-Gathering data frames across ranks")
        #all_rank_dfs = COMM.gather(this_rank_dfs)
        #if COMM.rank==0:
        #    all_rank_dfs = pandas.concat(all_rank_dfs)
        #    all_rank_dfs.reset_index(inplace=True, drop=True)
        #    all_df_name = os.path.join(self.params.outdir, "hopper_results.pkl")
        #    all_rank_dfs.to_pickle(all_df_name)


if __name__ == '__main__':
    from dials.util import show_mail_on_error

    with show_mail_on_error():
        script = Script()
        RUN = script.run
        lp = None
        if LineProfiler is not None and script.params.profile:
            lp = LineProfiler()
            lp.add_function(hopper_utils.model)
            lp.add_function(hopper_utils.target_func)
            lp.add_function(RUN)
            lp.add_function(utils.get_roi_background_and_selection_flags)
            lp.add_function(hopper_utils.DataModeler.GatherFromExperiment)
            lp.add_function(utils.simulator_for_refinement)
            lp.add_function(utils.simulator_from_expt_and_params)
            lp.add_function(hopper_utils.get_simulator_for_data_modelers)
            RUN = lp(script.run)
        elif script.params.profile:
            print("Install line_profiler in order to use logging: libtbx.python -m pip install line_profiler")

        with DeviceWrapper(script.dev) as _:
            #with np.errstate(all='raise'):
            try:
                RUN()
            except Exception as err:
                err_file = os.path.join(script.params.outdir, "rank%d_hopper_fail.err" % COMM.rank)
                with open(err_file, "w") as o:
                    from traceback import format_tb
                    _, _, tb = sys.exc_info()
                    tb_s = "".join(format_tb(tb))
                    #tb_s = tb_s.replace("\n", "\nRANK%04d" % COMM.rank)
                    err_s = str(err) + "\n" + tb_s
                    o.write(err_s)
                raise err
        COMM.barrier()

        if lp is not None:
            stats = lp.get_stats()
            hopper_utils.print_profile(stats, ["model", "target_func", "run", "get_roi_background_and_selection_flags", "GatherFromExperiment",
                                               "simulator_for_refinement", "simulator_from_expt_and_params", "get_simulator_for_data_modelers"])
