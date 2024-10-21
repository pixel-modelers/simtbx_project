
def main():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("philfile", type=str)
    ap.add_argument("cmdlinephil", nargs="+", type=str)
    args = ap.parse_args()
    params = None
    from libtbx.mpi4py import  MPI
    COMM = MPI.COMM_WORLD
    from simtbx.diffBragg import utils
    import time,os,sys

    if COMM.rank == 0:
        params, diff_phil_s = utils.get_extracted_params_from_phil_sources(args.philfile, args.cmdlinephil, return_diff_phil_s=True)
        assert params.outdir is not None
        utils.safe_makedirs(params.outdir)
        ts = time.strftime("%Y%m%d-%H%M%S")
        diff_phil_outname = os.path.join(params.outdir, "diff_phil_run_at_%s.txt" % ts)
        with open(diff_phil_outname, "w") as o:
            o.write("command line:\n%s\n" % (" ".join(sys.argv)))
            o.write("workding directory: \n%s\n" %os.getcwd())
            o.write("diff phil:\n")
            o.write(diff_phil_s)
        just_diff_phil_outname = os.path.join(params.outdir, "diff.phil")
        with open(just_diff_phil_outname, "w") as o:
            o.write(diff_phil_s)
    params = COMM.bcast(params)
    dev = COMM.rank % params.refiner.num_devices

    try:
        from line_profiler import LineProfiler
    except ImportError:
        LineProfiler = None
    lp = None
    from simtbx.diffBragg import hopper_utils
    from simtbx.command_line import run_hopper
    run_ = run_hopper.run
    if LineProfiler is not None and params.profile:
        lp = LineProfiler()
        lp.add_function(hopper_utils.model)
        lp.add_function(hopper_utils.target_func)
        lp.add_function(run_hopper.run)

        lp.add_function(hopper_utils.DataModeler.GatherFromExperiment)
        lp.add_function(utils.simulator_for_refinement)
        lp.add_function(utils.simulator_from_expt_and_params)
        lp.add_function(hopper_utils.get_simulator_for_data_modelers)
        run_= lp(run_hopper.run)
    elif params.profile:
        print("Install line_profiler in order to use logging: libtbx.python -m pip install line_profiler")

    from simtbx.diffBragg.device import DeviceWrapper
    with DeviceWrapper(dev) as _:
        try:
            run_(params, dev)
        except Exception as err:
            err_file = os.path.join(params.outdir, "rank%d_hopper_fail.err" % COMM.rank)
            with open(err_file, "w") as o:
                from traceback import format_tb
                _, _, tb = sys.exc_info()
                tb_s = "".join(format_tb(tb))
                err_s = str(err) + "\n" + tb_s
                o.write(err_s)
            raise err
    COMM.barrier()

    if lp is not None:
        stats = lp.get_stats()
        hopper_utils.print_profile(stats, ["model", "target_func", "run", "get_roi_background_and_selection_flags", "GatherFromExperiment",
                                           "simulator_for_refinement", "simulator_from_expt_and_params", "get_simulator_for_data_modelers"])


if __name__ == '__main__':
    main()
