"""
Reading, validating and centring the embeddings artifact.

Split out of `embedding_generator` deliberately: this half is pure numpy, and
the player must not import torch to start.  Generation needs the model; loading
needs the schema.

The artifact is the contract between the generation run and everything else
(audit §7), so the rules live in one place:

  schema_version   int    ≥ 2.  A v1 file has no centroid and would be scored on
                          an uncentred space (C5) — refuse it rather than
                          silently do the wrong arithmetic.
  track_files      (N,)   keys exactly as `mpc listall` reports them (M4)
  embeddings       (N,D)  pooled, **uncentred**, L2-normalised
  centroid         (D,)   mean of `embeddings`; applied on load
  window_offsets   (N+1,) CSR index into `windows`
  windows          (ΣW,D) per-window embeddings, L2-normalised
  metadata         JSON   model, versions, window scheme, timings
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np

SCHEMA_VERSION = 2

REQUIRED_KEYS = (
    'schema_version', 'track_files', 'embeddings', 'centroid',
    'window_offsets', 'windows',
)


def centre(embeddings: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """
    C5: `normalise(E − centroid)`.

    CLAP's space is strongly anisotropic — two unrelated tracks of this library
    sat at mean cosine 0.577 before centring, which is why `(sim + 1) / 2`,
    `novelty = (1 − max_sim) / 2` and every weight tuned against them were
    calibrated for a range the data never occupied.  One subtraction restores it.
    """
    centred = np.asarray(embeddings, dtype=np.float32) - np.asarray(centroid, dtype=np.float32)
    norms = np.linalg.norm(centred, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return (centred / norms).astype(np.float32)


def load_metadata(data) -> Dict[str, Any]:
    """Read the metadata block back.  Stored as JSON, so no pickle is involved."""
    if 'metadata' not in data:
        return {}
    raw = data['metadata']
    try:
        value = raw.item()
    except (AttributeError, ValueError):
        value = raw
    if isinstance(value, bytes):
        value = value.decode('utf-8', 'replace')
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def validate_embeddings(embeddings_file: Path) -> Dict[str, Any]:
    """
    Structural validation of an embeddings artifact (M5).

    Checks the schema the rest of the system depends on, rather than guessing at
    the file's provenance from a substring of its model name — which is how a
    random-vector file named `SMOKE-TEST-RANDOM-NOT-CLAP` was once greeted with
    "✓ Loading CLAP embeddings".
    """
    try:
        with np.load(embeddings_file, allow_pickle=False) as data:
            missing = [key for key in REQUIRED_KEYS if key not in data]
            if missing:
                return {'valid': False, 'error': f"missing keys: {', '.join(missing)}"}

            version = int(data['schema_version'])
            if version < SCHEMA_VERSION:
                return {
                    'valid': False,
                    'error': (f"schema version {version}, expected >= {SCHEMA_VERSION} "
                              "— regenerate; a v1 file carries no centroid (C5)"),
                }

            embeddings = data['embeddings']
            track_files = data['track_files']
            if embeddings.ndim != 2:
                return {'valid': False, 'error': f'invalid embedding shape: {embeddings.shape}'}
            if len(track_files) != embeddings.shape[0]:
                return {
                    'valid': False,
                    'error': (f'{len(track_files)} track names for '
                              f'{embeddings.shape[0]} embeddings'),
                }

            centroid = data['centroid']
            if centroid.shape != (embeddings.shape[1],):
                return {
                    'valid': False,
                    'error': f'centroid is {centroid.shape}, expected ({embeddings.shape[1]},)',
                }

            offsets, windows = data['window_offsets'], data['windows']
            if len(offsets) != len(track_files) + 1:
                return {
                    'valid': False,
                    'error': (f'window_offsets has {len(offsets)} entries, expected '
                              f'{len(track_files) + 1}'),
                }
            if int(offsets[-1]) != windows.shape[0]:
                return {
                    'valid': False,
                    'error': (f'window_offsets ends at {int(offsets[-1])} but there are '
                              f'{windows.shape[0]} windows'),
                }
            if windows.shape[0] and windows.shape[1] != embeddings.shape[1]:
                return {
                    'valid': False,
                    'error': (f'windows are {windows.shape[1]}-d, embeddings are '
                              f'{embeddings.shape[1]}-d'),
                }

            norms = np.linalg.norm(embeddings, axis=1)
            if not np.allclose(norms, 1.0, atol=0.01):
                return {
                    'valid': False,
                    'error': (f'embeddings not normalised '
                              f'(norms {norms.min():.3f} – {norms.max():.3f})'),
                }

            return {
                'valid': True,
                'schema_version': version,
                'n_tracks': int(embeddings.shape[0]),
                'dimension': int(embeddings.shape[1]),
                'n_windows': int(windows.shape[0]),
                'metadata': load_metadata(data),
                'norm_min': float(norms.min()),
                'norm_max': float(norms.max()),
                'norm_mean': float(norms.mean()),
            }

    except Exception as e:                             # noqa: BLE001
        return {'valid': False, 'error': f'failed to validate: {e}'}


def similarity_report(embeddings: np.ndarray, sample: int = 400,
                      seed: int = 0) -> Dict[str, float]:
    """
    The random-pair similarity distribution — the measurement C5 is about.

    No quality score.  The old `_assess_embedding_quality()` graded a library
    "Excellent" or "Very Poor" against invented thresholds (mean 0.3–0.5, std >
    0.15) that nothing had ever measured; it is exactly the class of invented
    vocabulary D4 removes.  The distribution is the finding — read it.
    """
    embeddings = np.asarray(embeddings)
    rng = np.random.default_rng(seed)
    n = min(sample, len(embeddings))
    if n < 2:
        return {'n_sampled': n, 'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                'p01': 0.0, 'p05': 0.0, 'p25': 0.0, 'median': 0.0, 'p75': 0.0,
                'p95': 0.0, 'p99': 0.0}

    idx = rng.choice(len(embeddings), n, replace=False) if n < len(embeddings) else np.arange(n)
    subset = embeddings[idx]
    sims = subset @ subset.T
    off_diagonal = sims[~np.eye(n, dtype=bool)]
    percentiles = np.percentile(off_diagonal, [1, 5, 25, 50, 75, 95, 99])
    return {
        'n_sampled': n,
        'mean': float(off_diagonal.mean()),
        'std': float(off_diagonal.std()),
        'min': float(off_diagonal.min()),
        'max': float(off_diagonal.max()),
        'p01': float(percentiles[0]),
        'p05': float(percentiles[1]),
        'p25': float(percentiles[2]),
        'median': float(percentiles[3]),
        'p75': float(percentiles[4]),
        'p95': float(percentiles[5]),
        'p99': float(percentiles[6]),
    }


def get_embedding_stats(embeddings_file: Path) -> Dict[str, Any]:
    """Measured statistics for `--stats`, raw and centred side by side."""
    try:
        with np.load(embeddings_file, allow_pickle=False) as data:
            embeddings = np.asarray(data['embeddings'])
            centroid = np.asarray(data['centroid']) if 'centroid' in data \
                else embeddings.mean(axis=0)
            n_windows = int(data['windows'].shape[0]) if 'windows' in data else 0
            metadata = load_metadata(data)

        return {
            'n_tracks': int(embeddings.shape[0]),
            'dimension': int(embeddings.shape[1]),
            'n_windows': n_windows,
            'raw': similarity_report(embeddings),
            'centred': similarity_report(centre(embeddings, centroid)),
            'centroid_norm': float(np.linalg.norm(centroid)),
            'metadata': metadata,
        }
    except Exception as e:                             # noqa: BLE001
        return {'error': f'Failed to compute stats: {e}'}


def write_failed_tracks(failed: Sequence[Tuple[str, str]], path: Path) -> None:
    """
    Record every file that did not embed, with its exception (C3: the original
    run lost 16 tracks and kept no list, so they were permanently unreachable
    with no diagnostic).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not failed:
        if path.exists():
            path.unlink()
        return
    lines = [f"# {datetime.now().isoformat(timespec='seconds')} — {len(failed)} failures"]
    lines += [f"{track}\t{error}" for track, error in failed]
    path.write_text("\n".join(lines) + "\n")
