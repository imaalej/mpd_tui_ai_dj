"""
Mood axes for the vibe cloud — a legible 3-D coordinate frame over the library.

Phase 4's centrepiece is a 3-D point cloud of the library in "mood space".  A
cloud needs three axes that are (a) *legible* — each a named perceptual
dimension, not an abstract principal component — and (b) actually a 3-D shape
rather than a pancake, i.e. the three chosen axes must be decorrelated **over
this library**.

Every candidate axis is a genuine direction in the same centred CLAP space the
app scores in:

    axis = normalise( mean(text-vecs of the HIGH pole) − mean(text-vecs of the
                      LOW pole) )

Projecting a centred track onto it is exactly the descriptor operation the app
already trusts (`DescriptorBank.z_scores`), expressed as a difference of two
poles.  So this is not a new model — it is the descriptor bank, read as a
compass.

Because the axis *directions* come from the fixed text tower while the spread,
the z-score calibration and the *choice of the best triad* are measured against
whatever library is on disk, the frame is inherently per-user and re-fits itself
as the library grows.  So — exactly like the descriptor mean/std (§6) — the
chosen triad and its calibration are computed once at generation time and stored
**beside the descriptors** in `descriptors.npz`, and the runtime just reads them.

Why beside the descriptors, not in `track_embeddings.npz`:
  * the directions are built from the descriptor text vectors, and the
    calibration is coupled to the descriptor mean/std — the same coupling that
    already lives in one file so the two cannot drift apart;
  * the whole thing is derivable from the two existing artifacts with no audio
    re-embedding, so it rebuilds via `--descriptors-only` (seconds) rather than a
    full ~5.5-minute regeneration.

The axis keys are **additive**: `descriptor_bank.SCHEMA_VERSION` is unchanged, an
older `descriptors.npz` without them still loads for the readout, and
`MoodAxes.load` returns `None` (reported, never a crash) when they are absent —
the same "absent is not fatal" discipline the descriptor bank itself follows.

`explore_mood_axes.py` (repo root, a throwaway diagnostic) imports the selection
from here so the offline proof and the stored artifact cannot diverge.
"""

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ── Candidate axes, poles drawn only from the descriptor bank's own words ──────
#
# (low pole → high pole).  A word missing from the bank is dropped from its pole;
# an axis survives as long as each pole keeps at least one word.  Intensity and
# Saturation are the usual collision risk (loud music tends to be dense/gritty),
# which is exactly why the triad is *selected against the library* rather than
# fixed — on this library those two correlate ~0.98 and only one can be used.
CANDIDATE_AXES: Dict[str, Tuple[List[str], List[str]]] = {
    "Intensity":  (["calm", "gentle", "mellow", "serene"],
                   ["aggressive", "intense", "frenetic", "energetic"]),
    "Tone":       (["melancholic", "sombre", "wistful", "menacing"],
                   ["uplifting", "joyful", "triumphant", "sunlit"]),
    "Saturation": (["sparse", "spacious", "polished"],
                   ["dense", "distorted", "gritty"]),
    "Organic":    (["acoustic", "orchestral", "guitar-driven", "piano-led"],
                   ["electronic", "synth-heavy"]),
    "Warmth":     (["cold"],
                   ["warm"]),
    "Groove":     (["free-time", "halftime"],
                   ["danceable", "groovy", "motorik"]),
    "Brightness": (["nocturnal", "cavernous"],
                   ["sunlit", "shimmering"]),
    "Drama":      (["intimate"],
                   ["cinematic", "cavernous"]),
}

# The triad the humans reached for by hand, reported alongside the auto pick.
# Not what gets stored — the selection is against the data — but a useful anchor
# for the diagnostic.
PREFERRED = ("Intensity", "Tone", "Saturation")

# Storage keys in descriptors.npz.  Additive to the descriptor bank's own keys.
AXIS_KEYS = ("axis_labels", "axis_directions", "axis_mean", "axis_std")


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return matrix / norms


def build_axis(name: str, low: Sequence[str], high: Sequence[str],
               bank: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """Unit direction for one axis, or None if a pole has no words in the bank."""
    def pole(words: Sequence[str]) -> Optional[np.ndarray]:
        vecs = [bank[w] for w in words if w in bank]
        return np.mean(vecs, axis=0) if vecs else None

    lo, hi = pole(low), pole(high)
    if lo is None or hi is None:
        return None
    d = hi - lo
    n = float(np.linalg.norm(d))
    return d / n if n > 1e-12 else None


@dataclass
class AxisSelection:
    """The chosen triad and everything needed to store or reproduce it."""
    labels: List[str]                       # 3 axis names, storage order
    directions: np.ndarray                  # (3, D) float32, unit vectors
    mean: np.ndarray                        # (3,) float32, raw-projection mean
    std: np.ndarray                         # (3,) float32, raw-projection std
    diagnostics: Dict[str, object] = field(default_factory=dict)


def select_axes(text_embeddings: np.ndarray,
                labels: Sequence[str],
                library_centred: np.ndarray) -> AxisSelection:
    """
    Choose the most usable legible triad for this library and calibrate it.

    Mirrors `explore_mood_axes.py`'s balanced pick exactly (the script now imports
    this function): score every triad by orthogonality × mean signal, where

        orthogonality = det of the triad's z-score correlation submatrix
                        (1 = mutually orthogonal, → 0 = flat / a pancake)
        signal        = the axes' raw projection spread (contrast)

    Orthogonality alone rewards weak-but-independent axes; multiplying by signal
    keeps the frame from being three directions nothing in the library moves
    along.

    Args:
        text_embeddings: (D_desc, dim) descriptor text-tower vectors (the stored
                         bank's `text_embeddings`; any normalisation).
        labels:          (D_desc,) descriptor words, aligned with the rows above.
        library_centred: (N, dim) library embeddings **after** centring (C5).

    Returns an `AxisSelection`.  The stored `mean`/`std` are of the *raw*
    projection `library @ direction`, so a runtime coordinate is
    `(v @ direction − mean) / std` — the z-score of the vibe on that axis.
    """
    text = _normalise_rows(np.asarray(text_embeddings, dtype=np.float64))
    bank = {str(lab): text[i] for i, lab in enumerate(labels)}
    library = np.asarray(library_centred, dtype=np.float64)

    names: List[str] = []
    directions: Dict[str, np.ndarray] = {}
    raw: Dict[str, np.ndarray] = {}
    for name, (low, high) in CANDIDATE_AXES.items():
        d = build_axis(name, low, high, bank)
        if d is None:
            continue
        names.append(name)
        directions[name] = d
        raw[name] = library @ d

    if len(names) < 3:
        raise ValueError(
            f"only {len(names)} usable mood axes for this library — need 3. "
            "The descriptor bank is missing too many pole words."
        )

    raw_std = {n: float(raw[n].std()) for n in names}
    z = {n: (raw[n] - raw[n].mean()) / (raw[n].std() + 1e-12) for n in names}
    Z = np.column_stack([z[n] for n in names])
    R = np.corrcoef(Z, rowvar=False)
    index = {n: i for i, n in enumerate(names)}

    def balanced(tri: Tuple[str, str, str]) -> float:
        idx = [index[t] for t in tri]
        det = float(np.linalg.det(R[np.ix_(idx, idx)]))
        strength = sum(raw_std[t] for t in tri)
        return det * (strength / 3.0)

    triads = list(itertools.combinations(names, 3))
    ranked = sorted(triads, key=balanced, reverse=True)
    chosen = list(ranked[0])

    directions_arr = np.stack([directions[n] for n in chosen]).astype(np.float32)
    mean = np.array([raw[n].mean() for n in chosen], dtype=np.float32)
    std = np.array([raw[n].std() for n in chosen], dtype=np.float32)

    diagnostics = {
        'candidates': names,
        'raw_std': {n: raw_std[n] for n in names},
        'balanced_top': [(list(t), balanced(t)) for t in ranked[:5]],
    }
    return AxisSelection(labels=chosen, directions=directions_arr,
                         mean=mean, std=std, diagnostics=diagnostics)


def selection_arrays(selection: AxisSelection) -> Dict[str, np.ndarray]:
    """The four npz keys for `descriptors.npz`, ready to splice into the bank."""
    return {
        'axis_labels': np.array(selection.labels, dtype=np.str_),
        'axis_directions': selection.directions.astype(np.float32),
        'axis_mean': selection.mean.astype(np.float32),
        'axis_std': selection.std.astype(np.float32),
    }


class MoodAxes:
    """
    Runtime side: read the stored axes and place a vector in mood space.

    Absent axes are not fatal — an older `descriptors.npz` predates them — so
    `load` reports and returns None, exactly like `DescriptorBank.load`.
    """

    def __init__(self, labels: Sequence[str], directions: np.ndarray,
                 mean: np.ndarray, std: np.ndarray):
        self.labels = [str(x) for x in labels]
        self.directions = np.asarray(directions, dtype=np.float32)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    @property
    def dimension(self) -> int:
        return int(self.directions.shape[1])

    @classmethod
    def load(cls, path: Optional[Path] = None) -> Optional['MoodAxes']:
        import sys

        from config import config

        path = Path(path) if path is not None else config.descriptors_file
        if not path.exists():
            print(f"No descriptor bank at {path} — mood axes unavailable. "
                  f"Build it with: python3 generate_embeddings.py --descriptors-only",
                  file=sys.stderr)
            return None

        try:
            data = np.load(path, allow_pickle=False)
            if any(k not in data for k in AXIS_KEYS):
                print(f"Descriptor bank at {path} carries no mood axes — rebuild it "
                      f"with: python3 generate_embeddings.py --descriptors-only",
                      file=sys.stderr)
                return None
            return cls(
                labels=[str(x) for x in data['axis_labels']],
                directions=data['axis_directions'],
                mean=data['axis_mean'],
                std=data['axis_std'],
            )
        except Exception as exc:                       # noqa: BLE001
            print(f"Failed to load mood axes from {path}: {exc}", file=sys.stderr)
            return None

    def coordinates(self, vector: np.ndarray) -> np.ndarray:
        """
        Place centred-audio-space vector(s) on the three axes, z-scored against
        the library's own spread.

        Accepts a single `(D,)` vector → `(3,)`, or a batch `(N, D)` → `(N, 3)`,
        so Phase 4 can project the whole library for the point cloud in one call
        (`MoodAxes.load().coordinates(centred_library)`) and the session vector
        for the comet with the same method.

        Rows are unit-normalised first — the calibration was measured over the
        normalised library, and `DescriptorBank.z_scores` normalises for the same
        reason.  Because the result is z-scored, the cloud comes out naturally
        centred on the origin with unit spread per axis, so the camera needs no
        separate rescaling.
        """
        vector = np.asarray(vector, dtype=np.float32)
        single = vector.ndim == 1
        rows = vector[None, :] if single else vector
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        rows = rows / norms
        std = np.where(self.std < 1e-12, 1.0, self.std)
        coords = (rows @ self.directions.T - self.mean) / std
        return coords[0] if single else coords
