"""
The CLAP descriptor bank — naming a vibe with the half of the model the project
never used.

Audit D5/H1.  A bank of ~50 descriptor prompts is embedded once with CLAP's text
tower and cached to `data/embeddings/descriptors.npz`.  Any vector living in the
library's (centred) audio space can then be named by scoring it against the bank.

Two things make this honest rather than decorative, and both are enforced here:

1. **Per-descriptor z-scoring.**  CLAP's audio and text towers are contrastively
   aligned but do not share a cone — the modality gap.  Raw `audio · text` dot
   products are *not* comparable across descriptors: one word scores 0.31
   against your entire library while another scores 0.08 against all of it, so a
   naive top-3 returns the same three words forever.  Scoring against each
   descriptor's own distribution over *your* library turns the readout into
   "unusually hypnotic **for this collection**", which is both the correct
   semantics and immune to the gap.

2. **A variance gate at build time.**  A descriptor whose standard deviation
   over the library is near zero describes all of your music equally, so it can
   never be informative — and dividing by that small number turns noise into a
   confident-looking z-score.  Those descriptors are dropped when the artifact is
   built, and both lists are logged.

The floor is a *fraction of the observed median std*, not an absolute number:
the scale of these similarities depends on the model, the template and the
library, so an absolute threshold would be the same mistake this whole rewrite
exists to remove.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SCHEMA_VERSION = 2

# Descriptors grouped by axis.  Complementary axes matter more than raw count:
# three words drawn from three different axes describe a track, three words from
# one axis repeat themselves.
DESCRIPTOR_AXES: Dict[str, Tuple[str, ...]] = {
    'energy': (
        'calm', 'gentle', 'mellow', 'driving', 'energetic', 'aggressive',
        'intense', 'frenetic',
    ),
    'affect': (
        'melancholic', 'sombre', 'wistful', 'uplifting', 'joyful', 'triumphant',
        'tense', 'menacing', 'serene', 'romantic',
    ),
    'texture': (
        'sparse', 'dense', 'lo-fi', 'polished', 'distorted', 'warm', 'cold',
        'shimmering', 'gritty', 'spacious',
    ),
    'rhythm': (
        'hypnotic', 'danceable', 'groovy', 'syncopated', 'free-time', 'motorik',
        'halftime',
    ),
    'setting': (
        'nocturnal', 'sunlit', 'cinematic', 'intimate', 'cavernous', 'dreamlike',
    ),
    'instrumentation': (
        'acoustic', 'electronic', 'orchestral', 'vocal-led', 'instrumental',
        'guitar-driven', 'piano-led', 'synth-heavy',
    ),
}

DESCRIPTORS: Tuple[str, ...] = tuple(
    label for labels in DESCRIPTOR_AXES.values() for label in labels
)

# CLAP was trained on audio captions, not bare adjectives, so the template is
# not cosmetic.  These are the candidates `generate_embeddings.py --compare-templates`
# scores against the real library; DEFAULT_TEMPLATE is the one that won.
PROMPT_TEMPLATES: Dict[str, str] = {
    'recording': 'This is a recording of {} music.',
    'bare': '{}',
    'sounds': 'The music sounds {}.',
    'genre': '{} music',
}
DEFAULT_TEMPLATE = 'recording'

# A descriptor survives if its library std is at least this fraction of the
# median std across the bank.  Relative, because the absolute scale moves with
# the model, the template and the collection.
STD_FLOOR_FRACTION = 0.5


def render_prompts(labels: Sequence[str], template: str) -> List[str]:
    return [template.format(label) for label in labels]


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return matrix / norms


def build_bank(
    text_embeddings: np.ndarray,
    labels: Sequence[str],
    prompts: Sequence[str],
    library_centred: np.ndarray,
    std_floor_fraction: float = STD_FLOOR_FRACTION,
) -> Dict[str, object]:
    """
    Assemble the artifact: per-descriptor mean/std over the centred library, plus
    the variance gate.

    Args:
        text_embeddings: (D, dim) CLAP text-tower output, any normalisation.
        labels:          (D,) descriptor words.
        prompts:         (D,) the rendered prompts, kept so the template is auditable.
        library_centred: (N, dim) library embeddings **after** centring (C5) — the
                         same space the session and taste vectors live in.

    Returns a dict of arrays ready for `np.savez`, plus a `report` entry
    describing what the gate kept and dropped.
    """
    text_embeddings = _normalise_rows(np.asarray(text_embeddings, dtype=np.float32))
    library_centred = np.asarray(library_centred, dtype=np.float32)

    # (N, D) — every descriptor scored against every track, once.
    scores = library_centred @ text_embeddings.T
    means = scores.mean(axis=0)
    stds = scores.std(axis=0)

    median_std = float(np.median(stds))
    floor = median_std * std_floor_fraction

    if median_std < 1e-6:
        # Degenerate library — one track, or every embedding identical.  There is
        # no variance to gate on, and the differences between descriptors are
        # float noise; ranking words by noise and deleting the losers would be
        # the same crime as the thresholds this bank replaces.  Keep the lot and
        # let the caller's log say why.
        keep = np.ones_like(stds, dtype=bool)
    else:
        keep = stds >= floor
        if not keep.any():
            # Floor above every observed std (only reachable with an absurd
            # `std_floor_fraction`).  An empty bank is a worse artifact than an
            # unfiltered one.
            keep = np.ones_like(stds, dtype=bool)

    dropped = [
        (labels[i], float(stds[i])) for i in range(len(labels)) if not keep[i]
    ]
    kept = [(labels[i], float(stds[i])) for i in range(len(labels)) if keep[i]]

    return {
        'schema_version': np.array(SCHEMA_VERSION),
        'labels': np.array([labels[i] for i in range(len(labels)) if keep[i]], dtype=np.str_),
        'prompts': np.array([prompts[i] for i in range(len(prompts)) if keep[i]], dtype=np.str_),
        'text_embeddings': text_embeddings[keep].astype(np.float32),
        'mean': means[keep].astype(np.float32),
        'std': stds[keep].astype(np.float32),
        'report': {
            'kept': kept,
            'dropped': dropped,
            'std_floor': floor,
            'std_floor_fraction': std_floor_fraction,
            'median_std': median_std,
        },
    }


def separation_metrics(scores: np.ndarray) -> Dict[str, float]:
    """
    How well does a bank separate a library?  Used to choose the prompt template
    against evidence instead of taste.

    - `mean_std`: average spread of a descriptor's scores across the library.
      Near zero means the word says the same thing about every track.
    - `effective_rank`: participation ratio of the descriptor correlation matrix,
      i.e. roughly how many *independent* things the bank measures.  A bank whose
      50 words all rank the library identically has an effective rank near 1 and
      is a one-word bank wearing fifty costumes.
    """
    stds = scores.std(axis=0)
    metrics = {'mean_std': float(stds.mean()), 'min_std': float(stds.min())}

    usable = stds > 1e-9
    if usable.sum() < 2:
        metrics['effective_rank'] = 1.0
        return metrics

    corr = np.corrcoef(scores[:, usable], rowvar=False)
    eigenvalues = np.clip(np.linalg.eigvalsh(corr), 0, None)
    total = eigenvalues.sum()
    metrics['effective_rank'] = (
        float(total ** 2 / np.square(eigenvalues).sum()) if total > 0 else 1.0
    )
    return metrics


class DescriptorBank:
    """
    Runtime side: load the artifact and name a vector.

    Consumed by the vibe readout and the `[I]` inspector (Stage 3).  Built in
    Stage 1 so a bad bank is discovered from a script rather than through a TUI.
    """

    def __init__(
        self,
        labels: Sequence[str],
        text_embeddings: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        prompts: Optional[Sequence[str]] = None,
    ):
        self.labels = list(labels)
        self.text_embeddings = np.asarray(text_embeddings, dtype=np.float32)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.prompts = list(prompts) if prompts is not None else []

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def dimension(self) -> int:
        return int(self.text_embeddings.shape[1])

    @classmethod
    def load(cls, path: Optional[Path] = None) -> Optional['DescriptorBank']:
        """
        Load the bank, or return None if it is absent or unusable.

        Absent is not fatal: the descriptor readout is a display feature, and the
        player must still play.  It *is* reported, so a missing bank never looks
        like an empty one.
        """
        from config import config

        path = Path(path) if path is not None else config.descriptors_file
        if not path.exists():
            print(f"No descriptor bank at {path} — vibe descriptors unavailable. "
                  f"Build it with: python3 generate_embeddings.py --descriptors-only",
                  file=sys.stderr)
            return None

        try:
            data = np.load(path, allow_pickle=False)
            version = int(data['schema_version'])
            if version < SCHEMA_VERSION:
                print(f"Descriptor bank at {path} is schema {version}, expected "
                      f"{SCHEMA_VERSION} — rebuild it.", file=sys.stderr)
                return None
            bank = cls(
                labels=[str(x) for x in data['labels']],
                text_embeddings=data['text_embeddings'],
                mean=data['mean'],
                std=data['std'],
                prompts=[str(x) for x in data['prompts']],
            )
        except Exception as exc:
            print(f"Failed to load descriptor bank {path}: {exc}", file=sys.stderr)
            return None

        return bank

    def z_scores(self, vector: np.ndarray) -> np.ndarray:
        """
        z(d, v) = (sim(d, v) - mean_d) / std_d, over the centred library's own
        distribution.  `vector` must be in the centred audio space.
        """
        vector = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm > 1e-12:
            vector = vector / norm
        sims = self.text_embeddings @ vector
        return (sims - self.mean) / np.where(self.std < 1e-12, 1.0, self.std)

    def top(self, vector: np.ndarray, n: int = 3) -> List[Tuple[str, float]]:
        """Top-n descriptors by z-score, most distinctive first."""
        if not self.labels:
            return []
        z = self.z_scores(vector)
        order = np.argsort(z)[::-1][:n]
        return [(self.labels[i], float(z[i])) for i in order]
