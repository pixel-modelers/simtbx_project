import boost_adaptbx.boost.python as bp
import cctbx.uctbx # possibly implicit
ext = bp.import_ext("simtbx_gpu_ext")
from simtbx_gpu_ext import *

