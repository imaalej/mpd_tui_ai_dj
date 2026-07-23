"""
Stage 0 was mostly deletion.  These guard the removals so nothing drifts back in
during Stages 1–4, and so a fresh clone can prove the tree matches the plan.

Reference: PROJECT_AUDIT.md §8, Stage 0.
"""

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
    offenders = []
    for path in _source_files():
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if symbol not in line:
                continue
            stripped = line.strip()
            # A comment explaining why something was removed is the point of
            # the removal being legible; live code referencing it is not.
            if stripped.startswith("#"):
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


def test_no_learned_state_or_embeddings_are_committed():
    """D3: everything under data/ was regenerated or discarded."""
    listed = subprocess.run(
        ["git", "ls-files", "data"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    assert all(Path(name).name == ".gitkeep" for name in listed), listed
