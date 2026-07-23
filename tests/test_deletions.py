"""
Stage 0 was mostly deletion, and Stage 1 removed a few more things.  These guard
the removals so nothing drifts back in during Stages 2–4, and so a fresh clone
can prove the tree matches the plan.

Reference: PROJECT_AUDIT.md §8, Stages 0 and 1.
"""

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Everything the plan removes outright: the time-context subsystem (D6), the
# stale headless twin and the orphaned setup helper (M2), and the phase tests
# that reported 66 passing checks while ~30 were hardcoded True literals (M1).
DELETED_FILES = [
    "time_context.py",
    "main.py",
    "setup_check.py",
    "test_phase2.py",
    "test_phase3.py",
    "test_phase3_integration.py",
]

# Symbols the deletions orphan.  Grepped rather than imported because some of
# them lived in modules that no longer exist.
DELETED_SYMBOLS = [
    "TimeContext",
    "time_context",
    "enable_time_context",
    "time_context_weight",
    "get_exploration_factor",
    "weekday_exploration_modifier",
    "weekend_exploration_modifier",
    "get_vibe_description",
    "generate_dummy_embeddings",
    # Stage 1.  The quality "score" graded a library Excellent/Poor against
    # invented thresholds nothing had measured — D4's rule applied to a number
    # rather than a word.  `save_embeddings` wrote a file the Stage 1 loader must
    # refuse (no schema version, no centroid, already-centred vectors).
    "_assess_embedding_quality",
    "_quality_interpretation",
    "save_embeddings",
    # Stage 2.  [V] and everything that existed only to serve it (D8/H9): it
    # blended the session vector 50% toward a *random* direction, landing it
    # around 0.56 mean similarity to the nearest real music against 0.73 for a
    # real track, for less pool turnover than two escalated [N] presses.  Its
    # real job was clearing the ten-track queue, which D1 removed.
    "force_shift",
    "process_vibe_skip",
    "set_high_exploration",
    "vibe_shift_magnitude",
    "_skip_vibe",
    "vibe_skips",
    # The queue-depth-10 apparatus (D1/C1).  `check_and_refill` compared
    # `len(mpc playlist)` — which counts already-played tracks with consume off —
    # against a low-water mark, so after the first ten tracks it never fired.
    "recalculate",
    "initialize_queue",
    "check_and_refill",
    "planned_queue",
    "currently_queued_in_mpd",
    "_sync_to_mpd",
    "get_upcoming_tracks",
    "queue_buffer_size",
    "queue_low_threshold",
    # Replaced by the solved repulsion (H9) and the zero seed (L7).
    "penalize_similar",
    "_initialize_session_vector",
    # Stage 3.  `RIGHT_COL_ROWS` was the hand-counted height of the Now Playing
    # right-hand column, and the album art was pinned to it — so every change to
    # that column silently misplaced the image and nothing could notice (H8).
    # `_art_geometry()` reads the widget tree instead.
    "RIGHT_COL_ROWS",
    "NP_BORDER_ROWS",
    # `get_playlist_metadata` built its map from `mpc playlist`, and with
    # `consume on` a played track is gone from it — so the one existing metadata
    # path covered exactly the tracks the session-history panel shows (§8, Stage
    # 3 trap 2).  `fetch_track_tags` is the per-track, cached replacement.
    "get_playlist_metadata",
    "_metadata_cache",
    # The queue panel and its one-line successor, replaced by the Session panel
    # (H1c), and the track counter the vibe line carried while the descriptor
    # bank sat unwired (H1b).
    "_update_queue_display",
    "queue_walker",
    "queue_listbox",
    "_session_line",
]

SEARCHED_SUFFIXES = {".py", ".sh"}


SKIPPED_DIRS = {"__pycache__", ".git", "tests"}


def _source_files():
    """Application sources only — the tests below name the deleted symbols."""
    for path in REPO_ROOT.rglob("*"):
        if path.suffix not in SEARCHED_SUFFIXES:
            continue
        if SKIPPED_DIRS & set(path.parts):
            continue
        yield path


@pytest.mark.parametrize("name", DELETED_FILES)
def test_file_is_deleted(name):
    assert not (REPO_ROOT / name).exists()


@pytest.mark.parametrize("symbol", DELETED_SYMBOLS)
def test_symbol_appears_nowhere_but_in_explanatory_comments(symbol):
    # Whole words only, so a live private helper is not condemned by sharing a
    # suffix with a deleted public one (`_save_embeddings` is the generator's
    # writer; `save_embeddings` was TrackLibrary's).
    pattern = re.compile(rf"\b{re.escape(symbol)}\b")
    quoted = re.compile(rf"`[^`]*\b{re.escape(symbol)}\b[^`]*`")

    offenders = []
    for path in _source_files():
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if not pattern.search(line):
                continue
            stripped = line.strip()
            # Prose explaining why something was removed is the point of the
            # removal being legible; live code referencing it is not.  A comment,
            # or a backtick-quoted mention inside a docstring, is prose.
            if stripped.startswith("#") or quoted.search(line):
                continue
            offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {stripped}")
    assert offenders == []


def test_tests_directory_is_tracked_by_git():
    """
    M1a: `.gitignore` carried `test_*.py`, so a fresh clone had zero tests while
    three untracked files reported a green suite locally.
    """
    listed = subprocess.run(
        ["git", "ls-files", "tests"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    assert any(name.endswith(".py") for name in listed), (
        "tests/ is not tracked — run `git add tests`"
    )


def test_track_keys_are_never_enumerated_from_the_filesystem():
    """
    M4: `mpc listall` is the single source of truth for track keys.  The
    generator used to walk the music directory with `rglob`, producing keys MPD
    might never accept — a mismatch that presents as "the DJ picks nothing"
    rather than as an error.
    """
    offenders = []
    for path in _source_files():
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if 'rglob' in line and not line.strip().startswith('#'):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
    assert offenders == []


def test_no_learned_state_or_embeddings_are_committed():
    """D3: everything under data/ was regenerated or discarded."""
    listed = subprocess.run(
        ["git", "ls-files", "data"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    assert all(Path(name).name == ".gitkeep" for name in listed), listed
