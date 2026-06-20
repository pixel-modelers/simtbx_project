"""
DiffBragg State: a clean, serializable representation of everything
needed to run, checkpoint, and resume a diffBragg refinement.

Design principles:
  - State = where you are (parameters + data selection)
  - Config = how to get there (sigmas, bounds, restraints, optimizer settings)
  - These are separate concerns - current code conflates them
  - Everything serializable to JSON (params) + npz (arrays)
  - Easy to inspect, diff, checkpoint
"""

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional
import json, os


# =============================================================================
# MODEL STATE: the physics parameters (small, serializable as JSON)
# =============================================================================

@dataclass
class CrystalState:
    """Per-shot crystal model. ~20 numbers."""
    # Orientation: U matrix as 3 rotation angles (radians) or full 3x3
    RotXYZ: list  # [rotx, roty, rotz] in radians

    # Unit cell
    unit_cell: list  # [a, b, c, alpha, beta, gamma]

    # Mosaic domain size (number of unit cells along each axis)
    Nabc: list  # [Na, Nb, Nc]
    Ndef: list = field(default_factory=lambda: [0, 0, 0])  # off-diagonal

    # Mosaic spread (radians)
    eta_abc: list = field(default_factory=lambda: [0, 0, 0])

    # Thermal factor
    Bfactor: float = 0.0

    # Overall scale
    G: float = 1.0


@dataclass
class BeamState:
    """Beam model. Mostly fixed, but spectrum can be adjusted."""
    spectrum_file: str = ""          # path to spectrum file
    spectrum: Optional[list] = None  # [(wavelength, weight), ...] if loaded
    lambda_offset: float = 0.0      # refinable shift
    lambda_scale: float = 1.0       # refinable scale
    direction: list = field(default_factory=lambda: [0, 0, -1])
    polarization_fraction: float = 0.5


@dataclass
class DetectorState:
    """Detector geometry. Can be per-shot (detz_shift) or global (panel geometry)."""
    detz_shift_mm: float = 0.0
    # Per panel-group: list of {rotations: [3], translations: [3]}
    panel_groups: list = field(default_factory=list)
    # Reference detector file (dxtbx format or simple geometry dict)
    detector_file: str = ""


@dataclass
class ShotState:
    """Everything about one shot's model. JSON-serializable."""
    shot_id: str = ""
    crystal: CrystalState = None
    beam: BeamState = None
    detector: DetectorState = None
    # Goniometer
    gonio_angle: float = 0.0  # radians, rotation during exposure


# =============================================================================
# DATA STATE: which pixels are being modeled (large arrays, saved as npz)
# =============================================================================

@dataclass
class ShotData:
    """
    The data side of a shot: which pixels, what they measured,
    what background model we're using, which reflections they belong to.

    Stored separately from ShotState because these are large arrays.
    """
    # Per-ROI info
    rois: np.ndarray = None          # (n_rois, 4) - x1, x2, y1, y2 per shoebox
    hkl: np.ndarray = None           # (n_rois, 3) - Miller index per ROI
    hkl_asu: np.ndarray = None       # (n_rois, 3) - ASU Miller index per ROI

    # Per-pixel info (flattened across all ROIs)
    data: np.ndarray = None          # observed intensities (photons)
    background: np.ndarray = None    # background model (photons)
    trusted: np.ndarray = None       # boolean mask - True = use this pixel
    panel_id: np.ndarray = None      # which detector panel
    fast: np.ndarray = None          # fast-scan coordinate
    slow: np.ndarray = None          # slow-scan coordinate
    roi_id: np.ndarray = None        # which ROI this pixel belongs to

    # Per-ROI scale factors (refined)
    roi_scales: np.ndarray = None    # (n_rois,) scale factor per reflection

    # Per-ROI background plane fits
    tilt_abc: np.ndarray = None      # (n_rois, 3) - fast_slope, slow_slope, offset

    # ROI removal log: lightweight history of what got filtered and why
    # Each entry: {"roi_idx": int, "hkl": (h,k,l), "iteration": int, "reason": str}
    roi_removal_log: list = field(default_factory=list)

    def remove_roi(self, roi_idx, iteration, reason):
        """Mark an ROI as untrusted and log why."""
        # Zero out trusted pixels for this ROI
        if self.trusted is not None and self.roi_id is not None:
            self.trusted[self.roi_id == roi_idx] = False
        hkl = tuple(self.hkl[roi_idx]) if self.hkl is not None else None
        self.roi_removal_log.append({
            "roi_idx": int(roi_idx),
            "hkl": hkl,
            "iteration": iteration,
            "reason": reason,
        })

    def active_roi_mask(self):
        """Boolean array (n_rois,) - True if ROI still has trusted pixels."""
        if self.trusted is None or self.roi_id is None:
            return None
        mask = np.zeros(self.n_rois(), dtype=bool)
        for i in range(self.n_rois()):
            mask[i] = self.trusted[self.roi_id == i].any()
        return mask

    def n_pixels(self):
        return len(self.data) if self.data is not None else 0

    def n_rois(self):
        return len(self.rois) if self.rois is not None else 0

    def n_trusted(self):
        return int(self.trusted.sum()) if self.trusted is not None else 0


# =============================================================================
# REFINEMENT CONFIG: how to refine (separate from state)
# =============================================================================

@dataclass
class ParamConfig:
    """Per-parameter refinement settings."""
    fix: bool = False
    sigma: float = 1.0
    minval: float = -1e20
    maxval: float = 1e20
    # Restraint: pulls parameter toward center with strength 1/beta
    center: Optional[float] = None
    beta: Optional[float] = None  # None = no restraint


@dataclass
class RefinementConfig:
    """How to refine - optimizer settings, per-parameter controls."""
    # Per-parameter-group configs
    RotXYZ: ParamConfig = field(default_factory=ParamConfig)
    G: ParamConfig = field(default_factory=ParamConfig)
    Nabc: ParamConfig = field(default_factory=ParamConfig)
    Ndef: ParamConfig = field(default_factory=ParamConfig)
    eta_abc: ParamConfig = field(default_factory=ParamConfig)
    Bfactor: ParamConfig = field(default_factory=ParamConfig)
    unit_cell: ParamConfig = field(default_factory=ParamConfig)
    detz_shift: ParamConfig = field(default_factory=ParamConfig)
    spectrum: ParamConfig = field(default_factory=ParamConfig)

    # Optimizer
    method: str = "L-BFGS-B"
    max_calls: int = 100

    # Which crystal system (constrains unit cell refinement)
    crystal_system: str = "triclinic"


# =============================================================================
# DIAGNOSTICS: what happened during refinement
# =============================================================================

@dataclass
class RefinementResult:
    """Output from a refinement run."""
    converged: bool = False
    n_iterations: int = 0
    final_target: float = float('inf')
    final_sigZ: float = float('inf')

    # Per-iteration traces (for debugging convergence)
    target_trace: list = field(default_factory=list)
    sigZ_trace: list = field(default_factory=list)

    # Best model prediction (per-pixel, same length as ShotData.data)
    best_model: np.ndarray = None


# =============================================================================
# TOP-LEVEL CONTAINER
# =============================================================================

@dataclass
class DiffBraggState:
    """
    Complete state for one diffBragg refinement shot.
    Can be saved/loaded/inspected/diffed trivially.

    Usage:
        state = DiffBraggState.from_stills_process(expt_file, refl_file)
        state.save("shot_001")
        state = DiffBraggState.load("shot_001")
    """
    model: ShotState = None
    data: ShotData = None
    config: RefinementConfig = None
    result: RefinementResult = None

    # Metadata
    source_expt: str = ""  # where the initial model came from
    source_refl: str = ""  # where the reflections came from
    mtz_file: str = ""     # structure factors file

    def save(self, prefix):
        """
        Save state to disk.
        - {prefix}_model.json  - small, human-readable model params
        - {prefix}_data.npz    - large pixel arrays
        - {prefix}_result.npz  - refinement diagnostics
        """
        os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)

        # Model params -> JSON (small, inspectable)
        model_dict = {
            "model": asdict(self.model) if self.model else None,
            "config": asdict(self.config) if self.config else None,
            "source_expt": self.source_expt,
            "source_refl": self.source_refl,
            "mtz_file": self.mtz_file,
        }
        with open(f"{prefix}_model.json", "w") as f:
            json.dump(model_dict, f, indent=2, default=_json_default)

        # Pixel data -> npz (large, compact)
        if self.data is not None:
            arrays = {}
            for k in ["rois", "hkl", "hkl_asu", "data", "background",
                       "trusted", "panel_id", "fast", "slow", "roi_id",
                       "roi_scales", "tilt_abc"]:
                v = getattr(self.data, k, None)
                if v is not None:
                    arrays[k] = np.asarray(v)
            np.savez_compressed(f"{prefix}_data.npz", **arrays)

            # ROI removal log -> JSON (small, human-readable)
            if self.data.roi_removal_log:
                with open(f"{prefix}_removals.json", "w") as f:
                    json.dump(self.data.roi_removal_log, f, indent=2,
                              default=_json_default)

        # Result -> npz
        if self.result is not None and self.result.best_model is not None:
            result_arrays = {"best_model": self.result.best_model}
            result_meta = {
                "converged": self.result.converged,
                "n_iterations": self.result.n_iterations,
                "final_target": self.result.final_target,
                "final_sigZ": self.result.final_sigZ,
            }
            np.savez_compressed(f"{prefix}_result.npz",
                                **result_arrays,
                                _meta=json.dumps(result_meta))

    @classmethod
    def load(cls, prefix):
        """Load state from disk."""
        state = cls()

        # Model
        model_path = f"{prefix}_model.json"
        if os.path.exists(model_path):
            with open(model_path) as f:
                d = json.load(f)
            if d.get("model"):
                m = d["model"]
                state.model = ShotState(
                    shot_id=m.get("shot_id", ""),
                    crystal=CrystalState(**m["crystal"]) if m.get("crystal") else None,
                    beam=BeamState(**m["beam"]) if m.get("beam") else None,
                    detector=DetectorState(**m["detector"]) if m.get("detector") else None,
                    gonio_angle=m.get("gonio_angle", 0.0),
                )
            if d.get("config"):
                c = d["config"]
                state.config = RefinementConfig(
                    method=c.get("method", "L-BFGS-B"),
                    max_calls=c.get("max_calls", 100),
                    crystal_system=c.get("crystal_system", "triclinic"),
                )
                # Restore per-param configs
                for pname in ["RotXYZ", "G", "Nabc", "Ndef", "eta_abc",
                              "Bfactor", "unit_cell", "detz_shift", "spectrum"]:
                    if pname in c:
                        setattr(state.config, pname, ParamConfig(**c[pname]))

            state.source_expt = d.get("source_expt", "")
            state.source_refl = d.get("source_refl", "")
            state.mtz_file = d.get("mtz_file", "")

        # Data
        data_path = f"{prefix}_data.npz"
        if os.path.exists(data_path):
            npz = np.load(data_path)
            state.data = ShotData(**{k: npz[k] for k in npz.files})

            # Removal log
            removals_path = f"{prefix}_removals.json"
            if os.path.exists(removals_path):
                with open(removals_path) as f:
                    state.data.roi_removal_log = json.load(f)

        # Result
        result_path = f"{prefix}_result.npz"
        if os.path.exists(result_path):
            npz = np.load(result_path, allow_pickle=True)
            state.result = RefinementResult(best_model=npz.get("best_model"))
            if "_meta" in npz:
                meta = json.loads(str(npz["_meta"]))
                state.result.converged = meta.get("converged", False)
                state.result.n_iterations = meta.get("n_iterations", 0)
                state.result.final_target = meta.get("final_target", float('inf'))
                state.result.final_sigZ = meta.get("final_sigZ", float('inf'))

        return state

    def summary(self):
        """Quick human-readable summary."""
        lines = [f"=== DiffBragg State: {self.model.shot_id if self.model else '?'} ==="]
        if self.model and self.model.crystal:
            c = self.model.crystal
            lines.append(f"  Unit cell: {c.unit_cell}")
            lines.append(f"  Nabc: {c.Nabc}, G: {c.G:.4f}, B: {c.Bfactor:.2f}")
        if self.data:
            lines.append(f"  ROIs: {self.data.n_rois()}, "
                         f"pixels: {self.data.n_pixels()} "
                         f"({self.data.n_trusted()} trusted)")
        if self.result:
            lines.append(f"  sigZ: {self.result.final_sigZ:.3f}, "
                         f"iters: {self.result.n_iterations}, "
                         f"converged: {self.result.converged}")
        return "\n".join(lines)

    def diff(self, other):
        """Compare two states - useful for debugging convergence."""
        diffs = {}
        if self.model and other.model and self.model.crystal and other.model.crystal:
            c1, c2 = self.model.crystal, other.model.crystal
            for attr in ["RotXYZ", "unit_cell", "Nabc", "Ndef", "eta_abc",
                         "G", "Bfactor"]:
                v1, v2 = getattr(c1, attr), getattr(c2, attr)
                if v1 != v2:
                    diffs[attr] = {"before": v1, "after": v2}
        if self.data and other.data:
            t1 = self.data.n_trusted()
            t2 = other.data.n_trusted()
            if t1 != t2:
                diffs["n_trusted"] = {"before": t1, "after": t2}
        return diffs


# =============================================================================
# GLOBAL STATE: shared across all shots (detector, structure factors)
# =============================================================================

@dataclass
class GlobalState:
    """
    State shared across all shots in a dataset.
    Used in stage 2 (ensemble refinement).
    """
    detector: DetectorState = None
    beam: BeamState = None

    # Structure factors (what stage 2 refines)
    space_group: str = ""
    Fhkl: dict = field(default_factory=dict)  # {(h,k,l): F} or stored as arrays

    # All shot states
    shots: dict = field(default_factory=dict)  # {shot_id: DiffBraggState}

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        # Save global params
        global_dict = {
            "space_group": self.space_group,
            "detector": asdict(self.detector) if self.detector else None,
            "beam": asdict(self.beam) if self.beam else None,
            "shot_ids": list(self.shots.keys()),
        }
        with open(os.path.join(directory, "global.json"), "w") as f:
            json.dump(global_dict, f, indent=2, default=_json_default)

        # Save Fhkl
        if self.Fhkl:
            hkl_array = np.array(list(self.Fhkl.keys()))
            f_array = np.array(list(self.Fhkl.values()))
            np.savez_compressed(os.path.join(directory, "Fhkl.npz"),
                                hkl=hkl_array, F=f_array)

        # Save each shot
        shots_dir = os.path.join(directory, "shots")
        os.makedirs(shots_dir, exist_ok=True)
        for shot_id, shot_state in self.shots.items():
            shot_state.save(os.path.join(shots_dir, shot_id))


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")
