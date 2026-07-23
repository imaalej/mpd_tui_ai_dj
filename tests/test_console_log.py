"""
The console ring buffer tees to a log file (audit L5).

While the TUI is active stderr is swallowed entirely — the panel shows five
lines and the ring buffer holds 200, so a traceback from the background thread
used to flash past and become unrecoverable.  Everything after Stage 0 is a
refactor that has to be debugged through this.
"""

import sys

import pytest

import tui


@pytest.fixture
def capture(tmp_path):
    """A capture that writes to a temp log and never touches the real stderr."""
    log = tmp_path / "dj.log"

    class _Sink:
        def __init__(self):
            self.text = ""

        def write(self, s):
            self.text += s
            return len(s)

        def flush(self):
            pass

        def fileno(self):
            return sys.__stderr__.fileno()

    sink = _Sink()
    cap = tui._ConsoleCapture(sink, log)
    return cap, sink, log


def test_completed_lines_reach_the_log(capture):
    cap, _, log = capture
    cap.write("Exploration increased to 0.35\n")
    cap.flush()

    contents = log.read_text()
    assert "session started" in contents
    assert "Exploration increased to 0.35" in contents


def test_log_survives_the_tui_taking_the_screen(capture):
    """
    The whole point: once tui_active is set nothing reaches the terminal, so the
    log is the only surviving copy.
    """
    cap, sink, log = capture
    cap.tui_active = True
    cap.write("Background loop error: boom\n")
    cap.flush()

    assert "boom" not in sink.text          # not bled through the urwid layout
    assert "boom" in log.read_text()        # but not lost either


def test_partial_lines_are_buffered_until_newline(capture):
    cap, _, log = capture
    cap.write("half a ")
    assert "half a" not in log.read_text()
    cap.write("line\n")
    assert "half a line" in log.read_text()


def test_ring_buffer_and_log_agree(capture):
    cap, _, log = capture
    for i in range(5):
        cap.write(f"line {i}\n")

    buffered = cap.get_lines()
    assert len(buffered) == 5
    logged = log.read_text()
    for line in buffered:
        assert line in logged


def test_an_unwritable_log_does_not_take_the_app_down(tmp_path):
    """A missing log is a degraded session, not a dead one."""
    class _Sink:
        def write(self, s):
            return len(s)

        def flush(self):
            pass

    # A directory where the log file should be — open() will fail.
    bad = tmp_path / "blocked"
    bad.mkdir()

    cap = tui._ConsoleCapture(_Sink(), bad)
    cap.write("still running\n")
    assert cap.get_lines()[-1].endswith("still running")
