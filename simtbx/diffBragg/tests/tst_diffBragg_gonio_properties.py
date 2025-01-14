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
parser.add_argument("--onlyDiffuse", action='store_true')
parser.add_argument("--sigma", default=[1,1,1], type=float, nargs=3)
parser.add_argument("--gamma", default=[100,100,100], type=float, nargs=3)
parser.add_argument("--idx", type=int, default=0, choices=[0,1,2], help="diffuse parameter index (0,1,2 ->a,b,c)")
parser.add_argument("--stencil", type=int, default=0, help="sets the stencil property in diffBragg (default is 0)")
parser.add_argument("--laue", action="store_true", help="sets the laue group number for the spacegroup")
parser.add_argument("--grad", choices=['sigma','gamma'], default='gamma')
parser.add_argument("--orientation", choices=[0,1], type=int, help="selects orientation for aniso. model", default=0)

import pylab as plt
import os

args = parser.parse_args()
if args.kokkos:
    os.environ["DIFFBRAGG_USE_KOKKOS"] = "1"
from simtbx.diffBragg.utils import find_diffBragg_instances
from simtbx.diffBragg.device import DeviceWrapper
with DeviceWrapper(0) as _:

    from simtbx.nanoBragg import sim_data
    from simtbx.diffBragg import utils

    S = sim_data.SimData(use_default_crystal=True)
    det_shape = (1024,1024)
    S.detector = sim_data.SimData.simple_detector(180, 0.1, det_shape)
    S.instantiate_diffBragg(verbose=0, oversample=0, auto_set_spotscale=True)
    #S.D.record_time = True
    S.D.spot_scale = 100000
    delta_phi_gt = 0.5
    gonio_ax = 1,0,0
    nphi_step=10
    utils.update_SIM_with_gonio(S, delta_phi=delta_phi_gt, num_phi_steps=nphi_step, spindle_axis=gonio_ax)
    gonio_id = 24
    S.D.refine(gonio_id)
    S.D.add_diffBragg_spots()
    img_sh = 1024,1024
    img = S.D.raw_pixels_roi.as_numpy_array()#.reshape((img_sh))
    S.D.raw_pixels_roi *=0
    S.D.raw_pixels *= 0
    # CHeck the goinio axis is moving crystal correctly:
    #import numpy as np
    #utils.update_SIM_with_gonio(S, delta_phi=45, num_phi_steps=1000, spindle_axis=(0,0,1))
    #S.D.add_diffBragg_spots()
    #img2 = S.D.raw_pixels_roi.as_numpy_array().reshape((img_sh))
    ## due to symmetry img2 should have partial concentric rings..
    #S.D.raw_pixels_roi *=0
    #S.D.raw_pixels *= 0
    #utils.update_SIM_with_gonio(S, delta_phi=90, num_phi_steps=1000, spindle_axis=(0,0,1))
    #S.D.add_diffBragg_spots()
    #img3 = S.D.raw_pixels_roi.as_numpy_array().reshape((img_sh))
    # due to symmetry img3 should have conventric rings

    bragg = img > 1e-1  # select bragg scattering regions

    deriv = list(map(lambda x: x.as_numpy_array(), S.D.get_gonio_angle_derivative_pixels()))[0]

    S.D.fix(gonio_id)

    from scipy.stats import linregress
    perc = 0.001, 0.01, 0.1, 1, 10

    all_error = []
    all_error2 = []
    shifts = []
    shifts2 = []
    import numpy as np
    for i_shift, p in enumerate(perc):
        delta_phi_shift = delta_phi_gt*p*0.01
        print(i_shift, delta_phi_shift)
        utils.update_SIM_with_gonio(S, delta_phi=delta_phi_gt + delta_phi_shift, num_phi_steps=nphi_step, spindle_axis=gonio_ax)
        shifts.append(delta_phi_shift)

        S.D.raw_pixels_roi *= 0
        #S.D.printout_pixel_fastslow = 10, 10
        S.D.add_diffBragg_spots()
        img2 = S.D.raw_pixels_roi.as_numpy_array()

        fdiff = (img2 - img) / (delta_phi_shift)
        error = np.abs(fdiff[bragg] - deriv[bragg]).mean()
        all_error.append(error)

        print ("error=%f, step=%f" % (error, delta_phi_shift))

        if args.plot:
            plt.plot(shifts, all_error, 'o')
            plt.show()

    l = linregress(shifts, all_error)
    assert l.rvalue > .9999  # this is definitely a line!
    assert l.slope > 0
    assert l.pvalue < 1e-6
    assert l.intercept < 0.1*l.slope # line should go through origin

    print("OK")
    for name in find_diffBragg_instances(globals()): del globals()[name]
