from libtbx.mpi4py import MPI
COMM = MPI.COMM_WORLD
from simtbx.diffBragg import mpi_logger
import os
import time
import hashlib
import glob
import pandas
from copy import deepcopy
from simtbx.diffBragg import hopper_utils
import numpy as np
import logging
from simtbx.diffBragg import utils
from dials.array_family import flex
from dxtbx.model import ExperimentList
from dxtbx.model.experiment_list import ExperimentListFactory


def create_experiment_identifier(experiment, experiment_file_path, experiment_id):
    """
    from xfel_project;
    Create a hashed experiment identifier based on the experiment file path, experiment index in the file, and experiment features
    """
    exp_identifier_str = os.path.basename(experiment_file_path + \
                                          str(experiment_id) + \
                                          str(experiment.beam) + \
                                          str(experiment.crystal) + \
                                          str(experiment.detector) + \
                                          ''.join([os.path.basename(p) for p in experiment.imageset.paths()]))
    hash_obj = hashlib.md5(exp_identifier_str.encode('utf-8'))
    return hash_obj.hexdigest()


def save_composite_files(dfs, expts, refls, refls_int, pd_dir, exp_ref_dir, chunk=0):
    df_name = os.path.join(pd_dir, "hopper_results_rank%d_chunk%d.pkl" % (COMM.rank, chunk))
    expt_name = os.path.join(exp_ref_dir, "hopper_rank%d_chunk%d.expt" % (COMM.rank,chunk))
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


def run(params, dev):

    #np.seterr(invalid='ignore')

    # TODO, figure out why next 3 lines are sometimes necessary?!
    if not hasattr(COMM, "rank"):
        COMM.rank = 0
        COMM.size = 1

    if params.logging.logname is None:
        params.logging.logname = "main_stage1.log"
    if params.profile_name is None:
        params.profile_name = "prof_stage1.log"
    mpi_logger.setup_logging_from_params(params)

    MAIN_LOGGER = logging.getLogger("diffBragg.main")
    assert os.path.exists(params.exp_ref_spec_file)
    input_lines = None
    best_models = None
    pd_dir = os.path.join(params.outdir, "pandas")
    expt_ref_dir = os.path.join(params.outdir, "expers_refls")
    if COMM.rank == 0:
        input_lines = open(params.exp_ref_spec_file, "r").readlines()
        if params.skip is not None:
            input_lines = input_lines[params.skip:]
        if params.first_n is not None:
            input_lines = input_lines[:params.first_n]
        if params.sanity_test_input:
            hopper_utils.sanity_test_input_lines(input_lines)

        if params.best_pickle is not None:
            logging.info("reading pickle %s" % params.best_pickle)
            best_models = pandas.read_pickle(params.best_pickle)

        utils.safe_makedirs(pd_dir)
        utils.safe_makedirs(expt_ref_dir)

    COMM.barrier()
    input_lines = COMM.bcast(input_lines)
    best_models = COMM.bcast(best_models)

    trefs = []
    this_rank_dfs = []  # dataframes storing the modeling results for each shot
    this_rank_Elints = ExperimentList()
    this_rank_Rints = None
    this_rank_Ridxs = None
    chunk_id = 0
    shots_per_chunk=params.shots_per_chunk
    try:
        from simtbx.diffBragg.roi_score import roi_check
        CHECKER = roi_check.roiCheck(params.roi.filter_scores.state_file)
    except:
        CHECKER = None
    for i_shot, line in enumerate(input_lines):
        time_to_refine = time.time()
        if i_shot == params.max_process:
            break
        if i_shot % COMM.size != COMM.rank:
            continue

        logging.info("COMM.rank %d on shot  %d / %d" % (COMM.rank, i_shot + 1, len(input_lines)))
        exp, ref, exp_idx, spec = hopper_utils.split_line(line)
        best = None
        if best_models is not None:
            if "exp_idx" not in list(best_models):
                best_models["exp_idx"]= 0
            best = best_models.query("exp_name=='%s'" % os.path.abspath(exp)).query("exp_idx==%d" % exp_idx)

            if len(best) != 1:
                raise ValueError("Should be 1 entry for exp %s in best pickle %s" % (exp, params.best_pickle))
        params.simulator.spectrum.filename = spec
        Modeler = hopper_utils.DataModeler(params)
        Modeler.exper_name = exp
        Modeler.exper_idx = exp_idx
        Modeler.refl_name = ref
        Modeler.rank = COMM.rank
        Modeler.i_shot = i_shot
        # Optional prediction step?
        if params.load_data_from_refls:
            gathered = Modeler.GatherFromReflectionTable(exp, ref, sg_symbol=params.space_group)
        else:
            gathered = Modeler.GatherFromExperiment(exp, ref,
                                                    remove_duplicate_hkl=params.remove_duplicate_hkl,
                                                    sg_symbol=params.space_group,
                                                    exp_idx=exp_idx)
        if not gathered:
            logging.warning("No refls in %s; CONTINUE; COMM.rank=%d" % (ref, COMM.rank))
            continue
        MAIN_LOGGER.info("Modeling %s (%d refls)" % (exp, len(Modeler.refls)))

        if params.refiner.reference_geom is not None:
            detector = ExperimentListFactory.from_json_file(params.refiner.reference_geom, check_format=False)[0].detector
            Modeler.E.detector = detector

        # here we support inputting an experiment list with multiple crystals
        # the first crystal in the exp list is used to instantiate a diffBragg instance,
        # the remaining crystals are added to the sim_data instance for use during hopper_utils modeling
        # best pickle is not supported yet for multiple crystals
        # also, if number of crystals is >1 , then the params.number_of_xtals flag will be overridden
        exp_list = ExperimentListFactory.from_json_file(exp, False)
        xtals = exp_list.crystals()  # TODO: fix as this is broken now that we allow multi image experiments
        if params.consider_multicrystal_shots and len(xtals) > 1:
            assert best is None, "cannot pass best pickle if expt list has more than one crystal"
            assert params.number_of_xtals==1, "if expt list has more than one xtal, leave number_of_xtals as the default"
            params.number_of_xtals = len(xtals)
            MAIN_LOGGER.debug("Found %d xtals with unit cells:" %len(xtals))
            for xtal in xtals:
                MAIN_LOGGER.debug("%.4f %.4f %.4f %.4f %.4f %.4f" % xtal.get_unit_cell().parameters())
        if params.record_device_timings and COMM.rank >0:
            params.record_device_timings = False  # only record for rank 0 otherwise there's too much output
        SIM = hopper_utils.get_simulator_for_data_modelers(Modeler)
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

        SIM.D.store_ave_wavelength_image = params.store_wavelength_images
        if params.refiner.verbose is not None and COMM.rank==0:
            SIM.D.verbose = params.refiner.verbose
        if params.profile:
            SIM.record_timings = True
        if params.use_float32:
            Modeler.all_data = Modeler.all_data.astype(np.float32)
            Modeler.all_background = Modeler.all_background.astype(np.float32)

        SIM.D.device_Id = dev

        nparam = len(Modeler.P)
        if SIM.refining_Fhkl:
            nparam += SIM.Num_ASU*SIM.num_Fhkl_channels
        x0 = [1] * nparam
        tref = time.time()
        MAIN_LOGGER.info("Beginning refinement of shot %d / %d" % (i_shot+1, len(input_lines)))
        try:
            x = Modeler.Minimize(x0, SIM, i_shot=i_shot)
            for i_rep in range(params.filter_after_refinement.max_attempts):
                if not params.filter_after_refinement.enable:
                    continue
                final_sigz = Modeler.target.all_sigZ[-1]
                niter = len(Modeler.target.all_sigZ)
                too_few_iter = niter < params.filter_after_refinement.min_prev_niter
                too_high_sigz = final_sigz > params.filter_after_refinement.max_prev_sigz
                if too_few_iter or too_high_sigz:
                    Modeler.filter_pixels(params.filter_after_refinement.threshold)
                    x = Modeler.Minimize(x0, SIM, i_shot=i_shot)

            if params.perRoi_finish:
                old_params = deepcopy(params)
                old_P = deepcopy(Modeler.P)
                # fix all of the refinement variables except for perRoiScale
                for fix_name in dir(params.fix):
                    if fix_name.startswith("_"):
                        continue
                    setattr(params.fix, fix_name, True)
                params.fix.perRoiScale = False
                Modeler.params = params
                Modeler.set_parameters_for_experiment(best)
                new_x = np.array([1.]*len(Modeler.P))
                for name in old_P:
                    new_p = Modeler.P[name]
                    old_p = old_P[name]
                    new_x[new_p.xpos] = x[old_p.xpos]
                    assert not new_p.refine

                x = Modeler.Minimize(new_x, SIM, i_shot=i_shot)

                # reset the params
                params = old_params

                # Filter poor fits
                if params.roi.filter_scores.enable and CHECKER is not None:
                    Modeler.best_model, _ = hopper_utils.model(x, Modeler, SIM, compute_grad=False)
                    Modeler.best_model_includes_background = False
                    num_good = Modeler.filter_bad_scores(CHECKER)
                    if num_good == 0:
                        Modeler.clean_up(SIM)
                        continue

                # Then repeat minimization, fixing roi fits
                Modeler.params = params
                old_P = deepcopy(Modeler.P)
                Modeler.set_parameters_for_experiment(best)
                new_x = np.array([1.] * len(Modeler.P))
                for name in Modeler.P:
                    new_p = Modeler.P[name]
                    old_p = old_P[name]
                    new_x[new_p.xpos] = x[old_p.xpos]
                x = Modeler.Minimize(new_x, SIM, i_shot=i_shot)

        except StopIteration:
            x = Modeler.target.x0
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
            print_s += " Ran %d iterations. Final sigmaZ = %.1f," % (niter, sigz)
        if COMM.rank==0:
            MAIN_LOGGER.info(print_s)
        else:
            MAIN_LOGGER.debug(print_s)
        if params.profile:
            SIM.D.show_timings(COMM.rank)

        dbg = params.debug_mode
        if dbg and COMM.rank > 0 and params.debug_mode_rank0_only:
            dbg = False
        shot_df = Modeler.save_up(x, SIM, rank=COMM.rank, i_shot=i_shot,
                        save_fhkl_data=dbg, save_refl=dbg, save_modeler_file=dbg,
                        save_sim_info=dbg, save_pandas=dbg, save_traces=dbg, save_expt=dbg)

        hopper_subimgs = []
        scores = []
        Ridx = flex.reflection_table()
        for roi, slc in Modeler.roi_id_slices.items():
            hopper_pix = Modeler.best_model[slc[0]]

            x1, x2, y1, y2 = Modeler.rois[int(roi)]
            ydim = y2 - y1
            xdim = x2 - x1

            hopper_subimg = hopper_pix.reshape((ydim, xdim))

            hopper_trust = Modeler.all_trusted[slc[0]].reshape((ydim,xdim))
            if not np.any(hopper_trust):
                continue

            hopper_subimgs.append(hopper_subimg)
            hopper_bg = Modeler.all_background[slc[0]].reshape((ydim, xdim))

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

        Elint = ExperimentList()
        Eref = deepcopy(Modeler.E)
        Eref.crystal.set_A(shot_df.Amats.values[0])
        Eref.detector = SIM.detector
        Elint.append(Eref)

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

    save_composite_files(this_rank_dfs, this_rank_Elints, this_rank_Ridxs, this_rank_Rints,
                         pd_dir, expt_ref_dir, chunk_id)


