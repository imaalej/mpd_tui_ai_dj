"""
Shared pytest fixtures.

Stage 0 note: this is not the full harness.  The `FakeMPD` that models real MPD
semantics — including consume mode — lands in Stage 2 (audit M1b), together with
the behavioural tests for queue refill, skip handling and mode restore.  What
lives here now covers what Stage 0 actually changed, so the deletions cannot
silently come back.
"""

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
