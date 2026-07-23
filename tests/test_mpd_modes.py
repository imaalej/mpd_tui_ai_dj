"""
MPD playback modes and the shutdown paths that restore them (audit C2 + H3).

C2: the application never asserted MPD's modes.  `random on` was the live state
of the development machine, and it silently discards every ordering decision the
DJ makes — MPD picks an arbitrary queue entry on advance.  Nothing in the TUI,
the console or the logs said so.

H3 turned that into a side effect on the user's system rather than merely a bug
in ours: the DJ now forces `consume on`, and `_shutdown()` — the only place that
restored anything — was unreachable on SIGTERM, because the handler set a flag
that urwid's `MainLoop` never read.  So closing the terminal left someone's MPD
in consume mode, which they would meet the next time they used `mpc`.

These drive the real `_force_mpd_modes` / `_restore_mpd_modes` / `_signal_handler`
against a stand-in, rather than a copy of them.
"""

import types

import pytest

from conftest import FakeMPD
from main_tui import AdaptiveDJWithTUI


TRACKS = [f"artist/album/{i:02d}.flac" for i in range(4)]

# The state of the development machine when the audit was written, and the
# reason C2 exists.
USER_MODES = {'repeat': 'off', 'random': 'on', 'single': 'off', 'consume': 'off'}


class _DJ(AdaptiveDJWithTUI):
    """
    The real orchestrator with its constructor replaced.

    A subclass rather than a `SimpleNamespace` so the methods under test are the
    genuine bound ones, calling each other for real — `_signal_handler` calls
    `_restore_mpd_modes`, and that call is part of what H3 is about.  The real
    `__init__` connects to MPD, loads 45 MB of embeddings and builds a TUI, none
    of which these assertions need.
    """

    def __init__(self, modes=None, tui=None):
        import threading
        self.mpd_controller = FakeMPD(
            TRACKS, modes=dict(modes if modes is not None else USER_MODES))
        self.tui = tui
        self.running = True
        self._original_mpd_modes = {}
        self._modes_restored = False
        self._restore_lock = threading.Lock()


def _dj(modes=None, tui=None):
    return _DJ(modes=modes, tui=tui)


def test_forcing_sets_every_mode_the_dj_needs(capsys):
    dj = _dj()
    dj._force_mpd_modes()

    assert dj.mpd_controller.get_modes() == {
        'repeat': 'off', 'random': 'off', 'single': 'off', 'consume': 'on'}


def test_what_changed_is_reported_rather_than_done_silently(capsys):
    """Silently clobbering a user's configuration is worse than telling them."""
    dj = _dj()
    dj._force_mpd_modes()

    out = capsys.readouterr().err
    assert 'random on → off' in out
    assert 'consume off → on' in out
    assert 'restored on exit' in out


def test_modes_already_correct_are_not_touched(capsys):
    dj = _dj(modes={'repeat': 'off', 'random': 'off',
                    'single': 'off', 'consume': 'on'})
    dj._force_mpd_modes()

    assert not [c for c in dj.mpd_controller.calls if c.startswith('mode:')]
    assert 'already as required' in capsys.readouterr().err


def test_restore_puts_the_users_modes_back_exactly(capsys):
    dj = _dj()
    dj._force_mpd_modes()
    dj._restore_mpd_modes()

    assert dj.mpd_controller.get_modes() == USER_MODES


def test_a_three_state_single_setting_survives_the_round_trip():
    """
    `single` is off/on/**oneshot** in modern MPD.  Restoring a user's `oneshot`
    as `off` would be a silent change of their setting, which is why the modes
    are carried as raw strings rather than booleans.
    """
    original = {'repeat': 'on', 'random': 'on', 'single': 'oneshot', 'consume': 'off'}
    dj = _dj(modes=original)

    dj._force_mpd_modes()
    assert dj.mpd_controller.get_modes()['single'] == 'off'

    dj._restore_mpd_modes()
    assert dj.mpd_controller.get_modes() == original


def test_restoring_twice_is_harmless():
    """It is wired into `_shutdown`, the signal handler and an atexit hook."""
    dj = _dj()
    dj._force_mpd_modes()

    dj._restore_mpd_modes()
    calls_after_first = len(dj.mpd_controller.calls)
    dj._restore_mpd_modes()
    dj._restore_mpd_modes()

    assert len(dj.mpd_controller.calls) == calls_after_first
    assert dj.mpd_controller.get_modes() == USER_MODES


def test_unreadable_modes_are_reported_and_nothing_is_forced(capsys):
    """A broken `mpc status` must not leave the DJ half-configuring MPD."""
    dj = _dj()
    dj.mpd_controller.get_modes = lambda: {}

    dj._force_mpd_modes()

    assert not [c for c in dj.mpd_controller.calls if c.startswith('mode:')]
    assert 'Could not read' in capsys.readouterr().err


# ── H3: the signal path ──────────────────────────────────────────────────────

class _FakeTUI:
    def __init__(self):
        self.exit_requested = False

    def request_exit(self):
        self.exit_requested = True


def test_a_signal_restores_the_modes_and_unblocks_the_ui():
    """
    The whole of H3 in one assertion.  Before this, the handler set
    `self.running = False` and stopped there: the background thread died, urwid
    carried on drawing a UI with nothing behind it, `_shutdown()` never ran, and
    the user's MPD stayed in consume mode.
    """
    tui = _FakeTUI()
    dj = _dj(tui=tui)
    dj._force_mpd_modes()

    dj._signal_handler(15, None)

    assert dj.running is False
    assert tui.exit_requested is True
    assert dj.mpd_controller.get_modes() == USER_MODES


def test_the_signal_path_works_before_the_tui_exists():
    """A signal arriving during startup must not raise inside the handler."""
    dj = _dj(tui=None)
    dj._force_mpd_modes()

    dj._signal_handler(15, None)

    assert dj.mpd_controller.get_modes() == USER_MODES


def test_the_tui_exposes_the_unblocking_hook():
    """
    `request_exit` is what the handler calls, and it must be writable from a
    signal context — hence a self-pipe the loop watches rather than raising
    ExitMainLoop, which urwid cannot receive from an arbitrary point.
    """
    from tui import AdaptiveDJTUI

    assert hasattr(AdaptiveDJTUI, 'request_exit')
    assert hasattr(AdaptiveDJTUI, '_on_exit_requested')

    # Safe with no pipe (non-urwid fallback mode) rather than raising.
    stub = types.SimpleNamespace(running=True, _exit_pipe=None)
    AdaptiveDJTUI.request_exit(stub)
    assert stub.running is False


def test_the_desired_modes_are_what_the_findings_call_for():
    assert AdaptiveDJWithTUI.DESIRED_MPD_MODES == {
        'random': 'off', 'repeat': 'off', 'single': 'off', 'consume': 'on'}
