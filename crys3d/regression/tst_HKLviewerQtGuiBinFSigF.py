# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from crys3d.regression import tests_HKLviewer
from libtbx.test_utils import contains_substring
import os, subprocess

os.environ['PYTHONIOENCODING'] = 'UTF-8'

# With the HKLviewer Qt GUI
# test creating the F/SigF dataset on the fly from iotbx/regression/data/phaser_1.mtz,
# expand data to P! with Friedel pairs, slice with a clip plane at l=9 and only show
# reflections with F/SigF<=1. Then save those to a new file QtGuiLowValueBinFSigF.mtz.
# Then check that this file matches the info in expectedstr

expectedstr = """
Starting job
===============================================================================
1 Miller arrays in this dataset:
 Labels          |       Type      |   λ/Å   |  #HKLs  |               Span              |     min,max data       |     min,max sigmas     |  d_min,d_max/Å   |Anomalous|Sym.uniq.|Data compl.|
  LowValuesFSigF |       Amplitude |       1 |     364 |           (-9, 0, 0), (9, 9, 9) |    0.78407,      11.265|        nan,         nan|     2.5,    8.771|   False |    True |   0.34019 |

===============================================================================
Job complete
"""

def run():
  count = 0
  while True:
    print("running %d" %count)
    # exerciseQtGUI() is unstable and might yield a bogus failure. If so, repeat the test
    # at most maxruns times or until it passes
    if not tests_HKLviewer.runagain(tests_HKLviewer.exerciseQtGUI,
                                    tests_HKLviewer.philstr3 %"QtGuiLowValueBinFSigF.mtz",
                                    tests_HKLviewer.reflections2match3,
                                    "QtGuiBinFSigF"):
      break
    count +=1
    assert(count < tests_HKLviewer.maxruns)

  # Now check that the produced mtz file matches the info in expectedstr
  obj = subprocess.Popen("cctbx.HKLinfo QtGuiLowValueBinFSigF.mtz",
                          shell=True,
                          stdin = subprocess.PIPE,
                          stdout = subprocess.PIPE,
                          stderr = subprocess.STDOUT)
  souterr,err = obj.communicate()
  tests_HKLviewer.Append2LogFile("QtGuiBinFSigFHKLviewer.log", souterr)
  souterr = souterr.decode().replace("\r\n", "\n") # omit \r\n line endings on Windows
  assert (contains_substring( souterr, expectedstr ) )


if __name__ == '__main__':
  run()
