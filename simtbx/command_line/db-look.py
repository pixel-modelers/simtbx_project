#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
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


if __name__ == "__main__":
    ap = ArgumentParser(
        description="Interactive viewer for diffBragg DataModeler .npy files "
                    "(from hopper outdir/imgs/).")
    ap.add_argument("npy_path", type=str,
                    help="Path to a modeler .npy file, or a directory containing them.")
    ap.add_argument("--index", type=int, default=0,
                    help="If npy_path is a directory, which .npy file to open (default: 0)")
    args = ap.parse_args()

    path = args.npy_path
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*_modeler.npy")))
        if not files:
            # Try imgs subdirectory
            files = sorted(glob.glob(os.path.join(path, "imgs", "rank*", "*_modeler.npy")))
        if not files:
            print("No *_modeler.npy files found in %s" % path, file=sys.stderr)
            sys.exit(1)
        idx = min(args.index, len(files) - 1)
        print("Found %d modeler files. Opening index %d: %s" % (len(files), idx, files[idx]))
        path = files[idx]

    viewer = ModelerViewer(path)
    viewer.show()
