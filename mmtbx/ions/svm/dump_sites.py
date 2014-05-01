# -*- coding: utf-8; py-indent-offset: 2 -*-
"""
A script to open a model and its data and dump out all information about its
ions sites to a pickle file.

Run this module with:

phenix.python -m mmtbx.ions.svm.dump_sites [args]
"""

from __future__ import division

import os
import sys

from mmtbx import ions
from mmtbx.ions.identify import WATER_RES_NAMES
from mmtbx.ions.environment import ChemicalEnvironment, ScatteringEnvironment
from mmtbx.ions.svm.utils import iterate_sites
import mmtbx.command_line
from mmtbx.command_line.water_screen import master_phil
from libtbx.str_utils import make_header
from libtbx import easy_pickle

def main(args, out = sys.stdout):
  usage_string = """\
mmtbx.dump_sites model.pdb data.mtz [options ...]

Utility to dump information about the properties of water and ion sites in a
model. This properties include local environment, electron density maps, and
atomic properties.
"""
  cmdline = mmtbx.command_line.load_model_and_data(
    args = args,
    master_phil = master_phil(),
    out = out,
    process_pdb_file = True,
    create_fmodel = True,
    prefer_anomalous = True,
    set_wavelength_from_model_header = True,
    set_inelastic_form_factors = "sasaki",
    usage_string = usage_string)

  params = cmdline.params
  params.use_svm = True

  make_header("Inspecting sites", out = out)

  manager = ions.identify.create_manager(
    pdb_hierarchy = cmdline.pdb_hierarchy,
    fmodel = cmdline.fmodel,
    geometry_restraints_manager = cmdline.geometry,
    wavelength = params.input.wavelength,
    params = params,
    verbose = params.debug,
    nproc = params.nproc,
    log = out)

  manager.show_current_scattering_statistics(out = out)

  sites = dump_sites(manager)

  out_name = os.path.splitext(params.input.pdb.file_name[0])[0] + "_sites.pkl"
  print >> out, "Dumping to", out_name
  easy_pickle.dump(out_name, sites)

def dump_sites (manager):
  """
  Iterate over all the ions and waters built into the model and dump out
  information about their properties.
  """

  atoms = iterate_sites(
    manager.pdb_hierarchy,
    res_filter = ions.SUPPORTED + WATER_RES_NAMES,
    split_sites = True)

  # Can't pickle entire AtomProperties because they include references to the
  # Atom object. Instead, gather what properties we want and store them in a
  # second list
  properties = \
    [(
      ChemicalEnvironment(
        atom.i_seq,
        manager.find_nearby_atoms(atom.i_seq, far_distance_cutoff = 3.5),
        manager),
      ScatteringEnvironment(
        atom.i_seq,
        manager,
        fo_density = manager.get_map_gaussian_fit("mFo", atom.i_seq),
        fofc_density = manager.get_map_gaussian_fit("mFo-DFc", atom.i_seq),
        anom_density = manager.get_map_gaussian_fit("anom", atom.i_seq)),
      )
     for atom in atoms]

  return properties

if __name__ == "__main__":
  main(sys.argv[1:])
