"""
This test checks the setter and getter for Ncells parameter
"""
from __future__ import division
##from simtbx.kokkos import gpu_instance
#kokkos_run = gpu_instance(deviceId = 0)

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--kokkos", action="store_true")
parser.add_argument("--plot", action='store_true')
parser.add_argument("--curvatures", action='store_true')
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()
if args.kokkos:
    import os
    os.environ["DIFFBRAGG_USE_KOKKOS"]="1"
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

    ucell = (70, 60, 50, 90.0, 110, 90.0)
    symbol = "C121"

    a_real, b_real, c_real = sqr(uctbx.unit_cell(ucell).orthogonalization_matrix()).transpose().as_list_of_lists()
    C = Crystal(a_real, b_real, c_real, symbol)

    # random raotation
    rotation = Rotation.random(num=1, random_state=101)[0]
    Q = rec(rotation.as_quat(), n=(4, 1))
    rot_ang, rot_axis = Q.unit_quaternion_as_axis_and_angle()
    C.rotate_around_origin(rot_axis, rot_ang)

    S = sim_data.SimData(use_default_crystal=True)
    S.crystal.dxtbx_crystal = C
    S.crystal.isotropic_ncells = True
    S.detector = sim_data.SimData.simple_detector(180, 0.1, (1024, 1024))
    S.instantiate_diffBragg(verbose=0, oversample=0)
    assert S.D.isotropic_ncells
    S.D.spot_scale = 100000
    #S.D.oversample = 5

    Ncells_id = 9

    S.D.Ncells_abc = 12
    i = S.D.get_value(Ncells_id)
    S.D.set_value(Ncells_id, 14)
    i2 = S.D.get_value(Ncells_id)
    S.D.Ncells_abc = 80
    i3 = S.D.get_value(Ncells_id)
    S.D.set_value(Ncells_id, 10)

    assert i == 12
    assert i2 == 14
    assert i3 == 80

    print("OK")

    S.D.refine(Ncells_id)
    S.D.initialize_managers()
    S.D.verbose = int(args.verbose)

    for noNabcScale, spot_scale in zip([False, True], [S.D.spot_scale, S.D.spot_scale*20**6]):
        S.D.Ncells_abc = 20
        assert S.D.isotropic_ncells
        S.D.spot_scale = spot_scale
        S.D.no_Nabc_scale = noNabcScale
        #S.D.printout_pixel_fastslow = 10, 10
        S.D.verbose = 1
        #S.D.isotropic_ncells = True
        S.D.add_diffBragg_spots()
        S.D.verbose = 0
        S.D.raw_pixels*=0
        img = S.D.raw_pixels_roi.as_numpy_array()
        deriv = S.D.get_ncells_derivative_pixels()[0].as_numpy_array()
        if args.curvatures:
            deriv2 = S.D.get_ncells_second_derivative_pixels()[0].as_numpy_array()

        N = S.D.get_value(Ncells_id)
        perc = 0.001, 0.01, 0.1, 1, 10

        bragg = img > 1e-1  # select bragg scattering regions

        all_error = []
        all_error2 = []
        shifts = []
        shifts2 = []
        for i_shift, p in enumerate(perc):
            delta_N = N*p*0.001
            #S.D.set_value(Ncells_id, N-delta_N)

            Nabc = [N+delta_N, N+delta_N, N+delta_N]
            shifts.append(delta_N)
            S.D.Ncells_abc_aniso = tuple(Nabc)
            S.D.raw_pixels_roi *= 0
            S.D.add_diffBragg_spots()
            img2 = S.D.raw_pixels_roi.as_numpy_array()

            fdiff = (img2-img) / delta_N
            error = np.abs(fdiff[bragg] - deriv[bragg]).mean()
            all_error.append(error)

            if args.curvatures:
                S.D.set_value(Ncells_id, N - delta_N)
                shifts2.append(delta_N**2)

                S.D.raw_pixels_roi *= 0
                S.D.add_diffBragg_spots()
                img_backward = S.D.raw_pixels.as_numpy_array()

                fdiff2 = (img2 - 2*img + img_backward) / delta_N / delta_N

                error2 = (np.abs(fdiff2[bragg] - deriv2[bragg]) / 1).mean()
                all_error2.append(error2)

            print ("error=%f, step=%f" % (error, delta_N))

        if args.plot:
            plt.plot(shifts, all_error, 'o')
            plt.show()
            if args.curvatures:
                plt.plot(shifts2, all_error2, 'o')
                plt.show()

        l = linregress(shifts, all_error)
        assert l.rvalue > .9999, l  # this is definitely a line!
        assert l.slope > 0
        assert l.pvalue < 1e-6
        assert l.intercept < 0.1*l.slope # line should go through origin
        if args.curvatures:
            l = linregress(shifts2, all_error2)
            assert l.rvalue > .9999  # this is definitely a line!
            assert l.slope > 0
            assert l.pvalue < 1e-6
            assert l.intercept < 0.1*l.slope # line should go through origin

    print("OK!")
    for name in find_diffBragg_instances(globals()): del globals()[name]
