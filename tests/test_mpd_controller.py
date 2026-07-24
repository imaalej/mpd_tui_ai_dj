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

from mpd_controller import MPDController, parse_mpc_clock

SOURCE = Path(__file__).resolve().parent.parent / "src" / "mpd_controller.py"


# ── time parsing, incl. the H:MM:SS form (audit F2) ──────────────────────────

@pytest.mark.parametrize("clock,seconds", [
    ("0:00", 0),
    ("0:05", 5),
    ("3:45", 225),
    ("1:23", 83),
    ("59", 59),            # a lone seconds field
    ("1:05:30", 3930),     # the hour form the old regex could not read
    ("2:00:00", 7200),
    ("10:00:00", 36000),
])
def test_parse_mpc_clock_folds_minutes_and_hours(clock, seconds):
    assert parse_mpc_clock(clock) == seconds


def test_status_parsing_reads_a_track_over_an_hour():
    """
    F2: a track past an hour prints `H:MM:SS/H:MM:SS`.  The old
    `(\\d+):(\\d+)/(\\d+):(\\d+)` matched a garbage substring, so position and
    duration were wrong and completion never fired honestly for such a track.
    """
    fields = MPDController._parse_status_fields(
        "volume: 40%   repeat: off\n[playing] #1/5   1:05:30/1:23:45 (78%)\n")
    assert fields['state'] == 'playing'
    assert fields['position'] == 3930      # 1:05:30
    assert fields['duration'] == 5025      # 1:23:45
    assert fields['position'] < fields['duration']
    assert fields['volume'] == 40


def test_status_parsing_still_reads_the_minutes_form():
    fields = MPDController._parse_status_fields(
        "volume: 90%\n[paused] #1/10   0:05/3:45 (2%)\n")
    assert fields['state'] == 'paused'
    assert fields['position'] == 5
    assert fields['duration'] == 225
    assert fields['volume'] == 90


def test_status_parsing_reports_stopped_with_no_time():
    fields = MPDController._parse_status_fields("volume: 50%\n")
    assert fields['state'] == 'stopped'
    assert 'position' not in fields and 'duration' not in fields


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
