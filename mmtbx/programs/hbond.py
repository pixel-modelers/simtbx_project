from __future__ import division, print_function
from libtbx.program_template import ProgramTemplate
import mmtbx.nci.hbond
from libtbx.utils import null_out

# =============================================================================

class Program(ProgramTemplate):

  description = '''
phenix.hbond: tool to find H bonds in an atomic model

Usage example:
  phenix.hbond model.pdb
  '''

  datatypes = ['model', 'phil']

  master_phil_str = mmtbx.nci.hbond.master_phil_str

  # ---------------------------------------------------------------------------
  def validate(self):
    print('Validating inputs', file=self.logger)
    self.data_manager.has_models(raise_sorry=True)

  # ---------------------------------------------------------------------------
  def run(self):
    print('Using model: %s' % self.data_manager.get_default_model_name(),
      file=self.logger)
    model = self.data_manager.get_model()
    model.set_log(log = null_out())
    model.process(make_restraints=True)
    self.results = mmtbx.nci.hbond.find(model = model)
    if self.params.hbond.show_hbonds:
      self.results.show(log = self.logger)
    print("-"*79, file=self.logger)
    self.results.show_summary(log = self.logger)
    prefix=self.params.output.prefix
    if not prefix: prefix='hbond'
    if self.params.hbond.output_pymol_file:
      self.results.as_pymol(prefix=prefix)
    if self.params.hbond.output_restraint_file:
      self.results.as_restraints(file_name='%s.eff' % prefix)
    #
    mmtbx.nci.hbond.stats(model = model, prefix="hbond_stats")

  # ---------------------------------------------------------------------------
  def get_results(self):
    return self.results

# =============================================================================
# end
