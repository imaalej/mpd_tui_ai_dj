"""
MPDController after the H7 dedupe.

Eight methods were defined twice in the same class body.  Python keeps the
second, so the first was unreachable while still reading as live code — and the
two `add_track` bodies differed: the dead one checked the return code, the live
one discarded it and returned True unconditionally.  Every `if success:` in the
codebase was therefore decorative.
"""

import ast
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from mpd_controller import MPDController

SOURCE = Path(__file__).resolve().parent.parent / "mpd_controller.py"


def test_no_method_is_defined_twice():
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "MPDController")
    names = Counter(n.name for n in cls.body if isinstance(n, ast.FunctionDef))
    duplicates = {name: count for name, count in names.items() if count > 1}
    assert duplicates == {}


class _FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_add_track_reports_mpd_refusal(monkeypatch):
    """
    A track MPD refuses — removed file, stale database, path mismatch — must
    come back as False.  At queue depth 1 (Stage 2) a swallowed failure means
    playback silently runs dry.
    """
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **kw: _FakeCompleted(returncode=1, stderr="error: No such directory\n"),
    )
    assert MPDController().add_track("nope/missing.flac") is False


def test_add_track_reports_success(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _FakeCompleted(returncode=0))
    assert MPDController().add_track("artist/album/01.flac") is True


def test_add_track_survives_a_subprocess_failure(monkeypatch):
    def boom(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="mpc", timeout=2)

    monkeypatch.setattr(subprocess, "run", boom)
    assert MPDController().add_track("artist/album/01.flac") is False


@pytest.mark.parametrize("name", ["toggle", "seek", "get_all_tracks", "get_track_metadata"])
def test_dead_api_removed(name):
    """L8: never called anywhere in the running system."""
    assert not hasattr(MPDController, name)


def test_previous_track_is_retained():
    """L8 exception — kept deliberately, with the consume-mode caveat recorded."""
    assert hasattr(MPDController, "previous_track")
