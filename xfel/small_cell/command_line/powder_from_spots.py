

# LIBTBX_SET_DISPATCHER_NAME cctbx.xfel.powder_from_spots
from __future__ import division
import logging

from iotbx.phil import parse
from dials.util import log
from dials.util import show_mail_on_error
from dials.util.options import ArgumentParser
from xfel.small_cell.powder_util import Spotfinder_radial_average, Center_scan



logger = logging.getLogger("dials.command_line.powder_from_spots")

help_message = """
Script to synthesize a powder pattern from DIALS spotfinding output
Examples of usage:

$ cctbx.xfel.powder_from_spots all.expt all.refl

$ cctbx.xfel.powder_from_spots all.expt all.refl \
    step_px=1 step_px=.5 step_px=.25 step_px=.125 step_px=.0625 \
    center_scan.d_max=18 center_scan.d_min=16 output.geom_file=xysearch.expt

This computes d-spacings of peak maxima in a reflections file as generated by
dials.find_spots. The results are binned and plotted as a histogram. Plotting
only maxima (and thus discarding the rest of the peak profile) has a large
sharpening effect; the resulting patterns are better than those obtained by
synchrotron powder diffraction.

The input is a single combined .refl file and the corresponding .expt file.
For small-molecule data, 10k to 20k shots are a minimum for good results.
Consider filtering the input by number of spots using the min_... and
max_reflections_per_experiment options of dials.combine_experiments. Min=3
and max=15 are a good starting point for small-molecule samples.

An excellent reference geometry (rmsd <<1 px) is important. A current detector
metrology refined from a protein sample is probably the best approach. Try a
plot with split_panels=True to confirm that the patterns on each panel agree.
In a data set from the MPCCD detector at SACLA we found that the Tau2 and Tau3
tilts (the tilts around the detector fast and slow axes) had to be refined for
each panel.

If ``step_px`` values are provided, grid searches will be performed to locate
the beam center accurately (as in usage example 2). This may greatly improve
the pattern if the detector internal metrology was refined carefully but the
detector subsequently shifted during the experiment.
"""

master_phil = parse(
    """
  n_bins = 3000
    .type = int
    .help = Number of bins in the radial average
  d_max = 20
    .type = float
  d_min = 2
    .type = float
  panel = None
    .type = int
    .help = Only use data from the specified panel
  peak_position = *xyzobs shoebox
    .type = choice
    .help = By default, use the d-spacing of the peak maximum. Shoebox: Use the \
            coordinates of every pixel in the reflection shoebox. This entails \
            intensity-weighted peaks.
  peak_weighting = *unit intensity
    .type = choice
    .help = The histogram may be intensity-weighted, but the results are \
            typically not very good.
  downweight_weak = 0
    .type = float
    .help = Subtract a constant from every intensity. May help filter out \
            impurity peaks.
  split_panels = False
    .type = bool
    .help = Plot a pattern for each detector panel.
  augment = False
    .type = bool
    .help = Plot an additional augmented pattern.
  xyz_offset = 0. 0. 0.
    .type = floats
    .help = origin offset in millimeters
  unit_cell = None
    .type = unit_cell
    .help = Show positions of miller indices from this unit_cell and space \
            group. Not implemented.
  space_group = None
    .type = space_group
    .help = Show positions of miller indices from this unit_cell and space \
            group. Not implemented.
filter {
  enable = False
    .type = bool
    .help = Instead of counts, plot the likelihood that a d-spacing is observed \
            together with a reference peak identified by filter.d_max and \
            filter.d_min.
  d_max = None
    .type = float
    .help = Max resolution of the peak to filter on
  d_min = None
    .type = float
    .help = Min resolution of the peak to filter on
}
output {
  log = dials.powder_from_spots.log
    .type = str
  xy_file = None
    .type = str
  peak_file = None
    .type = str
    .help = Optionally, specify an output file for interactive peak picking in \
            the plot window. Clicking and holding on the plot will bring up a \
            vertical line to help. Releasing the mouse button will add the \
            nearest local maximum to the output file peak_file.
  geom_file = None
    .type = path
    .help = Output a (possibly modified) geometry. For use with center_scan.
  plot_file = None
    .type = path
    .help = Output a powder pattern in image format.
}
center_scan {
  d_min = None
    .type = float
  d_max = None
    .type = float
  step_px = None
    .type = float
    .multiple = True
}
plot {
  interactive = True
    .type = bool
}
"""
)
multi_scan_phil = parse(
    """
  center_scan.step_px=2
  center_scan.step_px=1
  center_scan.step_px=.5
  center_scan.step_px=.25
  center_scan.step_px=.125
  center_scan.step_px=.0625
  center_scan.step_px=.03125
"""
)
phil_scope = master_phil.fetch(multi_scan_phil)






class Script(object):
  def __init__(self):
    usage = "$ cctbx.xfel.powder_from_spots EXPERIMENTS REFLECTIONS [options]"
    self.parser = ArgumentParser(
        usage=usage,
        phil=phil_scope,
        epilog=help_message,
        check_format=False,
        read_reflections=True,
        read_experiments=True,
        )

  def run(self):
    params, options = self.parser.parse_args(show_diff_phil=False)
    assert len(params.input.experiments) == len(params.input.reflections) == 1
    experiments = params.input.experiments[0].data
    reflections = params.input.reflections[0].data

    if params.center_scan.d_min:
      assert params.center_scan.d_max
      cscan = Center_scan(experiments, reflections, params)
      for step in params.center_scan.step_px:
        cscan.search_step(step)
      if params.output.geom_file is not None:
        experiments.as_file(params.output.geom_file)

    averager = Spotfinder_radial_average(experiments, reflections, params)
    averager.calculate()
    averager.plot()


if __name__ == "__main__":
  with show_mail_on_error():
    script = Script()
    script.run()
