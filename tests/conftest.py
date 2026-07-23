"""
Shared pytest fixtures.

Stage 1 note: this is still not the full harness.  The `FakeMPD` that models real
MPD semantics — including consume mode — lands in Stage 2 (audit M1b), together
with the behavioural tests for queue refill, skip handling and mode restore.
What lives here covers what Stages 0 and 1 changed: the deletions cannot come
back, and the embeddings artifact cannot silently change shape.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _restore_stderr():
    """
    Importing `tui` installs a stderr interceptor process-wide.  Put the real
    one back after every test so pytest's own reporting is never captured.
    """
    original = sys.stderr
    yield
    sys.stderr = original


@pytest.fixture
def rng():
    """Seeded generator — these tests assert on numbers, so they must not drift."""
    return np.random.default_rng(20260722)


@pytest.fixture
def library(rng):
    """
    A TrackLibrary populated in memory with normalised random embeddings.

    Random vectors are fine *here* — nothing under test asks what the vectors
    mean, only how the arithmetic around them behaves.
    """
    from track_library import TrackLibrary

    lib = TrackLibrary()
    for i in range(64):
        vec = rng.standard_normal(lib.dimension)
        lib.track_to_embedding[f"artist/album/{i:02d}.flac"] = vec / np.linalg.norm(vec)
    lib._build_matrix()
    return lib


@pytest.fixture
def make_artifact(tmp_path, rng):
    """
    Write an embeddings artifact (audit §7 schema) to a temp file.

    Every keyword is overridable so the loader's validation can be tested one
    broken field at a time — that is the point of having a schema at all.
    """
    from embeddings_io import SCHEMA_VERSION

    def _make(
        name="track_embeddings.npz",
        n_tracks=12,
        dimension=512,
        windows_per_track=3,
        schema_version=SCHEMA_VERSION,
        model='laion/clap-htsat-unfused',
        track_files=None,
        anisotropy=None,
        drop_keys=(),
        **overrides,
    ):
        raw = rng.standard_normal((n_tracks, dimension)).astype(np.float32)
        if anisotropy is not None:
            # Push every embedding toward one direction, the way CLAP's own
            # space is skewed (C5).  Without this, random vectors are already
            # centred and the centring test proves nothing.
            bias = rng.standard_normal(dimension).astype(np.float32)
            bias /= np.linalg.norm(bias)
            raw = raw / np.linalg.norm(raw, axis=1, keepdims=True) + anisotropy * bias
        embeddings = raw / np.linalg.norm(raw, axis=1, keepdims=True)

        n_windows = n_tracks * windows_per_track
        window_matrix = rng.standard_normal((n_windows, dimension)).astype(np.float32)
        window_matrix /= np.linalg.norm(window_matrix, axis=1, keepdims=True)

        names = track_files or [f"artist/album/{i:02d}.flac" for i in range(n_tracks)]
        payload = {
            'schema_version': np.array(schema_version),
            'track_files': np.array(names, dtype=np.str_),
            'embeddings': embeddings,
            'centroid': embeddings.mean(axis=0).astype(np.float32),
            'window_offsets': np.arange(n_tracks + 1, dtype=np.int32) * windows_per_track,
            'windows': window_matrix,
            'metadata': np.array(json.dumps({
                'schema_version': int(schema_version),
                'model': model,
                'dimension': dimension,
            })),
        }
        payload.update(overrides)
        for key in drop_keys:
            payload.pop(key, None)

        path = tmp_path / name
        np.savez_compressed(path, **payload)
        return path

    return _make
