#!/usr/bin/env python
"""
Interactive viewer for diffBragg model/data ROIs.

Two modes:
  1. ModelerViewer: View hopper DataModeler .npy files (default)
  2. PredictionViewer: View prediction results from hopper_predict.py

Usage:
  db-look.py <modeler.npy>              # Modeler mode
  db-look.py <hopper_outdir>            # Modeler mode (auto-find .npy)
  db-look.py --predict <cycle_outdir>   # Prediction mode

Prediction mode keybindings:
  LEFT/RIGHT   Navigate pages
  d            Sort by resolution (high-res first)
  i            Sort by intensity (brightest first)
  s            Sort by score (lowest first — worst predictions)
  o            Sort by centroid offset (largest first)
  n            Toggle: show only new predictions (not in observed)
  r            Cycle resolution filter: all → hi-res → mid-res → lo-res → all
  q            Quit
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from argparse import ArgumentParser
import sys
import glob
import os

# --- Configuration ---
ROI_PAIRS_PER_PAGE = 12
GRID_ROWS = 3
IMAGE_COLS = 4

if GRID_ROWS * IMAGE_COLS != ROI_PAIRS_PER_PAGE:
    print("Configuration Error: GRID_ROWS * IMAGE_COLS must equal ROI_PAIRS_PER_PAGE.", file=sys.stderr)
    sys.exit(1)


class ModelerViewer:
    """
    Interactive Matplotlib viewer for diffBragg DataModeler .npy files
    (the outputs saved in the imgs/ subfolder of a hopper outdir).

    Displays Data | Model pairs in a paginated grid, with per-ROI
    metadata: resolution, HKL, sigmaZ, panel/centroid, data/model max.
    """

    def __init__(self, npy_path):
        self.npy_path = npy_path
        self.M = self._load_modeler()
        if self.M is None:
            sys.exit(1)

        self._extract_subimages()

        self.num_rois = len(self.data_imgs)
        self.num_pages = max(1, int(np.ceil(self.num_rois / ROI_PAIRS_PER_PAGE)))
        self.current_page = 0

        plt.ion()
        self.fig = plt.figure(figsize=(10, 5))
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)

        print(f"Loaded {self.num_rois} ROIs from '{self.npy_path}'.")
        print("Use RIGHT/LEFT arrow keys to navigate pages. Press 'q' to quit.")

        self._plot_page()

    def _load_modeler(self):
        try:
            M = np.load(self.npy_path, allow_pickle=True)[()]
            return M
        except FileNotFoundError:
            print(f"Error: File not found at '{self.npy_path}'", file=sys.stderr)
            return None
        except Exception as e:
            print(f"Error loading modeler: {e}", file=sys.stderr)
            return None

    def _extract_subimages(self):
        M = self.M
        sigma_rdout = M.params.refiner.sigma_r / M.params.refiner.adu_per_photon

        has_model = M.best_model is not None
        if has_model:
            result = M.get_data_model_pairs(reorder=True, return_stats=True)
            if isinstance(M.all_sigma_rdout, np.ndarray):
                self.data_imgs, self.model_imgs, self.trusted_imgs, self.bragg_imgs, \
                    self.sigma_rdout_imgs, self.stats = result
            else:
                self.data_imgs, self.model_imgs, self.trusted_imgs, self.bragg_imgs, \
                    self.stats = result
                self.sigma_rdout_imgs = None
        else:
            # No model available — just show the data ROIs
            self.data_imgs = []
            self.model_imgs = []
            self.trusted_imgs = []
            self.bragg_imgs = []
            self.sigma_rdout_imgs = None
            spot_d = []
            spot_hkl = []
            spot_pid_and_cent = []
            for i_roi in range(len(M.rois)):
                x1, x2, y1, y2 = M.rois[i_roi]
                roi_sel = M.roi_id == i_roi
                dat = M.all_data[roi_sel].reshape((y2 - y1, x2 - x1))
                self.data_imgs.append(dat)
                self.model_imgs.append(np.zeros_like(dat))
                if M.all_trusted is not None:
                    self.trusted_imgs.append(
                        M.all_trusted[roi_sel].reshape((y2 - y1, x2 - x1)))
                else:
                    self.trusted_imgs.append(np.ones_like(dat, dtype=bool))
                self.bragg_imgs.append(np.zeros_like(dat))
                pid = M.pids[i_roi]
                cx, cy = (x1 + x2) / 2., (y1 + y2) / 2.
                spot_pid_and_cent.append((pid, cx, cy))
                try:
                    spot_hkl.append(M.Hi[i_roi])
                except (IndexError, AttributeError):
                    spot_hkl.append(None)
                spot_d.append(None)
            self.stats = {"spot_d": spot_d, "spot_hkl": spot_hkl,
                          "spot_pid_and_cent": spot_pid_and_cent,
                          "spot_scale": [1] * len(M.rois)}

        self.sigma_rdout = sigma_rdout
        self.has_model = has_model

    def _plot_page(self):
        self.fig.clf()

        start_idx = self.current_page * ROI_PAIRS_PER_PAGE
        end_idx = min(start_idx + ROI_PAIRS_PER_PAGE, self.num_rois)

        gs = self.fig.add_gridspec(GRID_ROWS, IMAGE_COLS, wspace=0.15, hspace=0.45)
        plt.subplots_adjust(left=0.01, right=0.99, bottom=0.02)

        for i in range(start_idx, end_idx):
            i_page = i - start_idx
            row = i_page // IMAGE_COLS
            col = i_page % IMAGE_COLS

            data_im = self.data_imgs[i]
            model_im = self.model_imgs[i]
            trusted_im = self.trusted_imgs[i]

            # Compute sigmaZ
            sigZ_str = ""
            if self.has_model and trusted_im is not None and trusted_im.sum() > 0:
                Z = (model_im - data_im) / np.sqrt(
                    np.maximum(model_im, 0.1) + self.sigma_rdout ** 2)
                sigZ = Z[trusted_im].std()
                sigZ_str = " sZ=%.1f" % sigZ

            # Build title line
            hkl_str = ""
            res_str = ""
            pfs_str = ""
            try:
                hkl = self.stats["spot_hkl"][i]
                if hkl is not None:
                    hkl_str = "%d,%d,%d" % tuple(hkl)
            except (IndexError, TypeError):
                pass
            try:
                d = self.stats["spot_d"][i]
                if d is not None:
                    res_str = "%.1fA" % d
            except (IndexError, TypeError):
                pass
            try:
                pfs = self.stats["spot_pid_and_cent"][i]
                pfs_str = "p%d" % int(pfs[0])
            except (IndexError, TypeError):
                pass

            # Composite image: [Data | separator | Model]
            height = data_im.shape[0]
            sep = np.full((height, 1), np.nan)
            composite = np.hstack([data_im, sep, model_im])

            ax = self.fig.add_subplot(gs[row, col])

            m = data_im.mean()
            s = data_im.std()
            vmin = m - s
            vmax = m + 3 * s
            if self.has_model:
                mm = model_im.mean()
                ms = model_im.std()
                vmin = min(mm - ms, vmin)
                vmax = max(mm + 3 * ms, vmax)

            ax.imshow(composite, interpolation='none', cmap='gray_r',
                      vmax=vmax, vmin=vmin)

            title_parts = [x for x in [hkl_str, res_str, pfs_str + sigZ_str] if x]
            title = "ROI %d" % i
            if title_parts:
                title += " " + " ".join(title_parts)
            ax.set_title(title, fontsize=7, pad=1)

            # Grid lines at data/model boundary
            xw = data_im.shape[1]
            ax.axvline(x=xw + 0.5, color='yellow', lw=0.5, alpha=0.7)
            ax.set_xticks([])
            ax.set_yticks([])

        basename = os.path.basename(self.npy_path)
        self.fig.suptitle(
            "%s  |  Page %d/%d (ROIs %d-%d)  |  arrows=nav, q=quit"
            % (basename, self.current_page + 1, self.num_pages,
               start_idx, end_idx - 1),
            fontsize=10, fontweight='bold')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _on_key_press(self, event):
        if event.key in ('right', 'up'):
            if self.current_page < self.num_pages - 1:
                self.current_page += 1
                self._plot_page()
        elif event.key in ('left', 'down'):
            if self.current_page > 0:
                self.current_page -= 1
                self._plot_page()
        elif event.key == 'q':
            plt.close(self.fig)
            plt.ioff()

    def show(self):
        plt.show(block=True)


class PredictionViewer:
    """
    Interactive viewer for prediction results from hopper_predict.py.

    Shows Data | Model ROI pairs for predicted reflections, with per-ROI
    metadata: resolution, intensity, centroid offset, prediction score.

    Supports interactive sorting and filtering via keyboard:
      d = sort by resolution    i = sort by intensity
      s = sort by score         o = sort by offset
      n = toggle new-only       r = cycle resolution filter
    """

    SORT_MODES = ['index', 'resolution', 'intensity', 'score', 'offset']
    RES_FILTERS = ['all', 'hires', 'midres', 'lores']

    def __init__(self, cycle_outdir, shot_idx=0, shoebox_size=10):
        self.cycle_outdir = cycle_outdir
        self.shot_idx = shot_idx
        self.sb = shoebox_size

        self._load_prediction_data()

        self.sort_mode = 'index'
        self.res_filter = 'all'
        self.new_only = False
        self._recompute_view()

        plt.ion()
        self.fig = plt.figure(figsize=(12, 7))
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)

        # Reserve space for buttons at bottom
        self.fig.subplots_adjust(bottom=0.12)
        self._create_buttons()

        self._print_help()
        self._plot_page()

    def _load_prediction_data(self):
        """Load prediction output: pandas, expt, refls, model images."""
        import pandas
        from dxtbx.model.experiment_list import ExperimentListFactory
        from dials.array_family import flex

        # Find pandas output
        pd_dir = os.path.join(self.cycle_outdir, "pandas")
        pkl_files = sorted(glob.glob(os.path.join(pd_dir, "rank*", "*.pkl")))
        if not pkl_files:
            pkl_files = sorted(glob.glob(os.path.join(pd_dir, "*.pkl")))
        if not pkl_files:
            print("No pickle files in %s" % pd_dir, file=sys.stderr)
            sys.exit(1)

        dfs = [pandas.read_pickle(f) for f in pkl_files]
        all_df = pandas.concat(dfs, ignore_index=True)

        # Find input.txt
        input_path = None
        for candidate in [os.path.join(self.cycle_outdir, "input.txt"),
                          os.path.join(os.path.dirname(self.cycle_outdir), "input.txt")]:
            if os.path.isfile(candidate):
                input_path = candidate
                break
        if input_path is None:
            print("Cannot find input.txt for %s" % self.cycle_outdir, file=sys.stderr)
            sys.exit(1)

        lines = [l.strip() for l in open(input_path).readlines() if l.strip()]
        if self.shot_idx >= len(lines):
            print("Shot index %d out of range (max %d)" % (self.shot_idx, len(lines) - 1),
                  file=sys.stderr)
            sys.exit(1)

        from simtbx.diffBragg.hopper_utils import split_line
        exp_path, ref_path, exp_idx, spec = split_line(lines[self.shot_idx])

        # Load experiment + observed reflections
        El = ExperimentListFactory.from_json_file(exp_path, check_format=True)
        self.expt = El[exp_idx]
        self.detector = self.expt.detector
        self.beam = self.expt.beam
        self.observed = flex.reflection_table.from_file(ref_path)

        # Load optimized detector if available
        from simtbx.command_line.hopper_predict import _load_optimized_detector
        opt_det = _load_optimized_detector(self.cycle_outdir)
        if opt_det is not None:
            self.detector = opt_det
            print("Using optimized detector (%d panels)" % len(opt_det))

        # Find matching refined model
        match = all_df[all_df.opt_exp_name.apply(
            lambda x: os.path.splitext(os.path.basename(str(x)))[0]
        ).str.contains(os.path.splitext(os.path.basename(exp_path))[0])]
        if len(match) == 0:
            print("No refined model found for shot %d" % self.shot_idx, file=sys.stderr)
            sys.exit(1)

        row = match.iloc[0]

        # Run forward simulation
        from simtbx.command_line.hopper_predict import simulate_and_predict
        print("Simulating predictions for shot %d..." % self.shot_idx)
        predictions, self.panel_imgs, _sim_expt = simulate_and_predict(
            row, self.expt, self.detector, self.beam, threshold=1e-3)
        print("Found %d predicted spots" % len(predictions))

        # Read actual data images
        try:
            raw_data = self.expt.imageset.get_raw_data(0)
            self.data_imgs = np.array([p.as_numpy_array().astype(float) for p in raw_data])
        except Exception as e:
            print("Cannot read image data: %s" % e, file=sys.stderr)
            sys.exit(1)

        # Build per-ROI metadata
        self._build_roi_data(predictions, row)

    def _build_roi_data(self, predictions, df_row):
        """Extract ROIs and compute per-spot metadata."""
        from scipy.spatial import cKDTree

        sb = self.sb
        half = sb // 2

        # Build KDTree of observed centroids per panel for new-vs-observed tagging
        obs_centroids = {}
        if 'xyzobs.px.value' in self.observed:
            for i_ref in range(len(self.observed)):
                x, y, z = self.observed['xyzobs.px.value'][i_ref]
                pid = int(self.observed['panel'][i_ref]) if 'panel' in self.observed else 0
                if pid not in obs_centroids:
                    obs_centroids[pid] = []
                obs_centroids[pid].append((x, y))
        obs_trees = {pid: cKDTree(np.array(coords))
                     for pid, coords in obs_centroids.items() if coords}

        self.rois = []  # list of dicts with all metadata

        for i_ref in range(len(predictions)):
            try:
                x, y, z = predictions['xyzobs.px.value'][i_ref]
                pid = int(predictions['panel'][i_ref]) if 'panel' in predictions else 0
            except Exception:
                continue

            i_slow = int(round(y))
            i_fast = int(round(x))
            y0 = i_slow - half
            x0 = i_fast - half
            y1 = y0 + sb
            x1 = x0 + sb

            data_img = self.data_imgs[pid]
            model_img = self.panel_imgs[pid]

            if (y0 < 0 or x0 < 0 or
                y1 > data_img.shape[0] or x1 > data_img.shape[1]):
                continue

            dat_roi = data_img[y0:y1, x0:x1].copy()
            mod_roi = model_img[y0:y1, x0:x1].copy()

            if dat_roi.shape != (sb, sb) or mod_roi.shape != (sb, sb):
                continue

            # Resolution
            panel = self.detector[pid]
            s1 = np.array(panel.get_pixel_lab_coord((x, y)))
            s1_norm = s1 / np.linalg.norm(s1) * (1.0 / self.beam.get_wavelength())
            s0 = np.array(self.beam.get_s0())
            q = s1_norm - s0
            d_spacing = 1.0 / max(np.linalg.norm(q), 1e-10)

            # Intensity
            intensity = float(predictions['intensity.sum.value'][i_ref]) \
                if 'intensity.sum.value' in predictions else float(np.sum(mod_roi))

            # Is this near an observed reflection?
            is_new = True
            obs_dist = 999.0
            if pid in obs_trees:
                dist, _ = obs_trees[pid].query([x, y])
                obs_dist = dist
                if dist < 3.0:
                    is_new = False

            # Centroid offset: find peak in data vs peak in model
            data_peak = np.unravel_index(np.argmax(dat_roi), dat_roi.shape)
            model_peak = np.unravel_index(np.argmax(mod_roi), mod_roi.shape)
            offset = np.sqrt((data_peak[0] - model_peak[0])**2 +
                             (data_peak[1] - model_peak[1])**2)

            self.rois.append({
                'idx': i_ref,
                'data_roi': dat_roi,
                'model_roi': mod_roi,
                'd_spacing': d_spacing,
                'intensity': intensity,
                'is_new': is_new,
                'obs_dist': obs_dist,
                'offset': offset,
                'data_peak': data_peak,
                'model_peak': model_peak,
                'panel': pid,
                'centroid': (x, y),
                'score': None,  # populated if score_trainer available
            })

        print("Extracted %d valid ROIs (%d new, %d near observed)"
              % (len(self.rois),
                 sum(1 for r in self.rois if r['is_new']),
                 sum(1 for r in self.rois if not r['is_new'])))

        # Compute resolution bins
        all_d = np.array([r['d_spacing'] for r in self.rois])
        if len(all_d) > 3:
            sorted_d = np.sort(all_d)
            n3 = len(sorted_d) // 3
            self.d_boundaries = (sorted_d[n3], sorted_d[2 * n3])
        else:
            self.d_boundaries = (2.0, 5.0)

        # Print alignment diagnostic
        offsets = np.array([r['offset'] for r in self.rois])
        if len(offsets) > 0:
            print("\nAlignment diagnostic (data peak vs model peak centroid offset):")
            print("  Mean: %.2f px, Median: %.2f px, Max: %.2f px, >2px: %d/%d (%.0f%%)"
                  % (offsets.mean(), np.median(offsets), offsets.max(),
                     np.sum(offsets > 2), len(offsets),
                     100 * np.sum(offsets > 2) / len(offsets)))

    def _recompute_view(self):
        """Recompute filtered + sorted view indices."""
        indices = list(range(len(self.rois)))

        # Apply filters
        if self.new_only:
            indices = [i for i in indices if self.rois[i]['is_new']]

        if self.res_filter != 'all':
            d_lo, d_hi = self.d_boundaries
            if self.res_filter == 'hires':
                indices = [i for i in indices if self.rois[i]['d_spacing'] < d_lo]
            elif self.res_filter == 'midres':
                indices = [i for i in indices
                           if d_lo <= self.rois[i]['d_spacing'] < d_hi]
            elif self.res_filter == 'lores':
                indices = [i for i in indices if self.rois[i]['d_spacing'] >= d_hi]

        # Apply sort
        if self.sort_mode == 'resolution':
            indices.sort(key=lambda i: self.rois[i]['d_spacing'])
        elif self.sort_mode == 'intensity':
            indices.sort(key=lambda i: -self.rois[i]['intensity'])
        elif self.sort_mode == 'score':
            indices.sort(key=lambda i: self.rois[i]['score'] or 0)
        elif self.sort_mode == 'offset':
            indices.sort(key=lambda i: -self.rois[i]['offset'])

        self.view_indices = indices
        self.num_view = len(indices)
        self.num_pages = max(1, int(np.ceil(self.num_view / ROI_PAIRS_PER_PAGE)))
        self.current_page = 0

    def _create_buttons(self):
        """Create filter/sort buttons at bottom of figure."""
        button_specs = [
            (0.02, 'Sort: d', self._btn_sort_d),
            (0.12, 'Sort: I', self._btn_sort_i),
            (0.22, 'Sort: off', self._btn_sort_offset),
            (0.34, 'New only', self._btn_toggle_new),
            (0.46, 'Res: all', self._btn_cycle_res),
            (0.58, 'Score', self._btn_run_score),
        ]
        self.buttons = []
        self.btn_widgets = []
        for x, label, callback in button_specs:
            ax_btn = self.fig.add_axes([x, 0.01, 0.09, 0.04])
            btn = Button(ax_btn, label, hovercolor='lightblue')
            btn.label.set_fontsize(8)
            btn.on_clicked(callback)
            self.buttons.append(ax_btn)
            self.btn_widgets.append(btn)

    def _btn_sort_d(self, event):
        self.sort_mode = 'resolution'
        self._refresh()

    def _btn_sort_i(self, event):
        self.sort_mode = 'intensity'
        self._refresh()

    def _btn_sort_offset(self, event):
        self.sort_mode = 'offset'
        self._refresh()

    def _btn_toggle_new(self, event):
        self.new_only = not self.new_only
        label = "New: ON" if self.new_only else "New only"
        self.btn_widgets[3].label.set_text(label)
        self._refresh()

    def _btn_cycle_res(self, event):
        idx = self.RES_FILTERS.index(self.res_filter)
        self.res_filter = self.RES_FILTERS[(idx + 1) % len(self.RES_FILTERS)]
        self.btn_widgets[4].label.set_text("Res: %s" % self.res_filter)
        self._refresh()

    def _btn_run_score(self, event):
        """Run score_trainer on all ROIs (lazy, one-time)."""
        if self.rois and self.rois[0]['score'] is not None:
            self.sort_mode = 'score'
            self._refresh()
            return

        try:
            from score_trainer import roi_check
            checker = roi_check.roiCheck()
        except ImportError:
            print("score_trainer not available")
            return

        dat_list = [r['data_roi'] for r in self.rois]
        mod_list = [r['model_roi'] for r in self.rois]
        scores = checker.score(dat_list, mod_list)
        if isinstance(scores, (int, float)):
            scores = [scores]
        for r, sc in zip(self.rois, scores):
            r['score'] = float(sc)

        print("Scores computed: min=%.3f median=%.3f max=%.3f"
              % (min(scores), np.median(scores), max(scores)))
        self.sort_mode = 'score'
        self._refresh()

    def _refresh(self):
        self._recompute_view()
        self._plot_page()

    def _plot_page(self):
        # Clear only the plot area, not buttons
        for ax in list(self.fig.axes):
            if ax not in self.buttons:
                self.fig.delaxes(ax)

        if self.num_view == 0:
            self.fig.text(0.5, 0.5, "No ROIs match current filters",
                          ha='center', va='center', fontsize=14)
            self.fig.canvas.draw()
            return

        start = self.current_page * ROI_PAIRS_PER_PAGE
        end = min(start + ROI_PAIRS_PER_PAGE, self.num_view)

        gs = self.fig.add_gridspec(GRID_ROWS, IMAGE_COLS, wspace=0.15, hspace=0.50,
                                   top=0.88, bottom=0.12, left=0.01, right=0.99)

        for i_page, vi in enumerate(range(start, end)):
            roi_idx = self.view_indices[vi]
            roi = self.rois[roi_idx]
            row = i_page // IMAGE_COLS
            col = i_page % IMAGE_COLS

            dat = roi['data_roi']
            mod = roi['model_roi']

            # Composite: Data | sep | Model
            height = dat.shape[0]
            sep = np.full((height, 1), np.nan)
            composite = np.hstack([dat, sep, mod])

            ax = self.fig.add_subplot(gs[row, col])

            m = dat.mean()
            s = max(dat.std(), 0.1)
            vmin = m - s
            vmax = m + 3 * s
            mm = mod.mean()
            ms = max(mod.std(), 0.1)
            vmin = min(mm - ms, vmin)
            vmax = max(mm + 3 * ms, vmax)

            ax.imshow(composite, interpolation='none', cmap='gray_r',
                      vmax=vmax, vmin=vmin)

            # Mark centroids: data peak (cyan) and model peak (red)
            dp = roi['data_peak']
            mp = roi['model_peak']
            ax.plot(dp[1], dp[0], 'c+', markersize=4, markeredgewidth=0.8)
            xw = dat.shape[1]
            ax.plot(mp[1] + xw + 1, mp[0], 'r+', markersize=4, markeredgewidth=0.8)

            # Separator line
            ax.axvline(x=xw + 0.5, color='yellow', lw=0.5, alpha=0.7)

            # Title
            parts = []
            parts.append("%.1fA" % roi['d_spacing'])
            if roi['is_new']:
                parts.append("NEW")
            parts.append("off=%.1f" % roi['offset'])
            if roi['score'] is not None:
                parts.append("sc=%.2f" % roi['score'])
            title = " ".join(parts)
            ax.set_title(title, fontsize=6.5, pad=1)
            ax.set_xticks([])
            ax.set_yticks([])

        # Suptitle with filter/sort state
        filter_str = "sort=%s" % self.sort_mode
        if self.new_only:
            filter_str += " new-only"
        if self.res_filter != 'all':
            filter_str += " res=%s" % self.res_filter
        self.fig.suptitle(
            "Shot %d  |  %d/%d ROIs  |  Page %d/%d  |  %s  |  arrows/keys"
            % (self.shot_idx, self.num_view, len(self.rois),
               self.current_page + 1, self.num_pages, filter_str),
            fontsize=9, fontweight='bold')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _on_key_press(self, event):
        if event.key in ('right', 'up'):
            if self.current_page < self.num_pages - 1:
                self.current_page += 1
                self._plot_page()
        elif event.key in ('left', 'down'):
            if self.current_page > 0:
                self.current_page -= 1
                self._plot_page()
        elif event.key == 'd':
            self.sort_mode = 'resolution'
            self._refresh()
        elif event.key == 'i':
            self.sort_mode = 'intensity'
            self._refresh()
        elif event.key == 's':
            self._btn_run_score(None)
        elif event.key == 'o':
            self.sort_mode = 'offset'
            self._refresh()
        elif event.key == 'n':
            self._btn_toggle_new(None)
        elif event.key == 'r':
            self._btn_cycle_res(None)
        elif event.key == 'q':
            plt.close(self.fig)
            plt.ioff()

    def _print_help(self):
        print("\nPrediction Viewer — Keybindings:")
        print("  LEFT/RIGHT  Navigate pages")
        print("  d           Sort by resolution (high-res first)")
        print("  i           Sort by intensity (brightest first)")
        print("  s           Score all ROIs, sort by score (worst first)")
        print("  o           Sort by centroid offset (largest first)")
        print("  n           Toggle: new predictions only")
        print("  r           Cycle resolution filter: all/hires/midres/lores")
        print("  q           Quit")
        print()

    def show(self):
        plt.show(block=True)


if __name__ == "__main__":
    ap = ArgumentParser(
        description="Interactive viewer for diffBragg DataModeler .npy files "
                    "or prediction results from hopper_predict.")
    ap.add_argument("path", type=str,
                    help="Path to .npy file, hopper outdir, or cycle outdir (with --predict)")
    ap.add_argument("--index", type=int, default=0,
                    help="File/shot index (default: 0)")
    ap.add_argument("--predict", action="store_true",
                    help="Prediction viewer mode: view predictions from a cycle outdir")
    ap.add_argument("--shoebox-size", type=int, default=10,
                    help="Shoebox size for prediction ROIs (default: 10)")
    args = ap.parse_args()

    if args.predict:
        viewer = PredictionViewer(args.path, shot_idx=args.index,
                                  shoebox_size=args.shoebox_size)
        viewer.show()
    else:
        path = args.path
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "*_modeler.npy")))
            if not files:
                files = sorted(glob.glob(os.path.join(path, "imgs", "rank*", "*_modeler.npy")))
            if not files:
                print("No *_modeler.npy files found in %s" % path, file=sys.stderr)
                sys.exit(1)
            idx = min(args.index, len(files) - 1)
            print("Found %d modeler files. Opening index %d: %s" % (len(files), idx, files[idx]))
            path = files[idx]

        viewer = ModelerViewer(path)
        viewer.show()
