"""
The non-urwid fallback mode, driven for the first time (audit M1c, Stage 4).

Stage 3 reconciled this mode's bindings with the footer and the README by
*reading* them, and said so: it was the only interface claim in the project with
nothing behind it.  Reading is how the three lists came to disagree in the first
place — `↑↓` were volume here, queue navigation in urwid, and something else
again in the README (L9).

Two things make it testable now.  The mode no longer carries its own `if key ==`
ladder: `decode_key()` turns terminal bytes into the names urwid uses and
`_handle_input` does the dispatch, so most of what used to need a terminal is a
pure function.  What is left genuinely needs one — `_run_simple_mode` reads real
stdin through `termios`/`tty`/`select` — so the rest of this file gives it a
pty.  That is the same reason end-to-end verification of the urwid mode runs
under `pty.fork()`.
"""

import os
import pty
import sys
import threading

import pytest

from tui import decode_key, decode_keys


# ── decode_key: the half that needs no terminal ──────────────────────────────


@pytest.mark.parametrize("raw,expected", [
    ("\x1b[A", "up"),
    ("\x1b[B", "down"),
    ("\x1b[C", "right"),
    ("\x1b[D", "left"),
    ("\x1bOA", "up"),          # SS3, sent in application-cursor mode
    ("\x1bOB", "down"),
    ("\x1bOC", "right"),
    ("\x1bOD", "left"),
    ("\r", "enter"),
    ("\n", "enter"),
    (" ", " "),
    ("n", "n"),
    ("L", "L"),
    (",", ","),
    (".", "."),
    ("<", "<"),
    (">", ">"),
])
def test_decode_key_maps_terminal_bytes_to_urwid_names(raw, expected):
    assert decode_key(raw) == expected


@pytest.mark.parametrize("raw", ["", "\x1b", "\x1b[Z", "\x00", "\x1b[1;5A"])
def test_unknown_input_decodes_to_nothing(raw):
    """Unhandled input is ignored, which is what urwid does with it too."""
    assert decode_key(raw) is None


def test_a_burst_of_input_yields_every_key_in_it():
    """
    The defect the pty harness found on its first run.  The mode read one
    character per 0.5 s tick and `select()`ed between reads — but `read()` on a
    buffered stream drains the descriptor into Python's buffer, so `select()`
    then reported nothing pending and the rest of the burst sat unread until
    another keypress dislodged one character of it.
    """
    keys, rest = decode_keys(" nl\x1b[A\x1b[B\r")
    assert keys == [" ", "n", "l", "up", "down", "enter"]
    assert rest == ""


def test_an_escape_sequence_split_across_two_reads_is_held_and_completed():
    keys, rest = decode_keys("n\x1b[")
    assert keys == ["n"]
    assert rest == "\x1b["

    keys, rest = decode_keys(rest + "A")
    assert keys == ["up"]
    assert rest == ""


def test_a_bare_escape_is_dropped_rather_than_eating_the_next_key():
    keys, rest = decode_keys("\x1b\x1b\x1bq")
    assert keys == ["q"]
    assert rest == ""


def test_an_unrecognised_escape_sequence_is_dropped_whole():
    keys, rest = decode_keys("\x1b[Zq")
    assert keys == ["q"]
    assert rest == ""


def test_the_arrow_keys_mean_the_same_thing_in_both_interfaces(tui, library):
    """
    The claim L9 is actually about.  `↑` decoded from a terminal must reach the
    same method as `up` from urwid — not merely be documented as doing so.
    """
    fired = []
    for name in ('_history_scroll', '_seek_forward', '_seek_backward',
                 '_replay_focused', '_volume_up', '_volume_down'):
        setattr(tui, name, (lambda n: lambda *a, **k: fired.append(n))(name))

    for raw in ("\x1b[A", "\x1b[B", "\x1b[C", "\x1b[D", "\r", ",", "."):
        tui._handle_input(decode_key(raw))

    assert fired == ['_history_scroll', '_history_scroll', '_seek_forward',
                     '_seek_backward', '_replay_focused',
                     '_volume_down', '_volume_up']


def test_up_and_down_are_history_not_volume(tui):
    """
    The specific disagreement L9 records: in the fallback mode `↑↓` used to be
    bound to volume, contradicting both the urwid mode and the README.
    """
    before = tui.dj.mpd_controller.volume
    tui.history.note_playing('a.flac')
    tui.history.note_playing('b.flac')

    tui._handle_input(decode_key("\x1b[B"))

    assert tui.dj.mpd_controller.volume == before
    assert tui.history.focus == 1


# ── the loop: this half needs a real terminal ────────────────────────────────


class _Pty:
    """
    A pty whose slave stands in for `sys.stdin` while the loop runs.

    The slave is put in cbreak mode here, before anything is written to it.
    A fresh pty starts in *canonical* mode, where the line discipline holds
    every byte until a newline arrives — so keys sent ahead of the loop's own
    `tty.setcbreak()` are invisible to `select()` and the harness reads nothing
    at all, which is exactly what it did on the first run.
    """

    def __init__(self):
        import tty

        self.master, slave = pty.openpty()
        tty.setcbreak(slave)
        self.stdin = os.fdopen(slave, 'r')

    def send(self, *chunks):
        for chunk in chunks:
            os.write(self.master, chunk.encode())

    def close(self):
        try:
            self.stdin.close()
        except Exception:
            pass
        try:
            os.close(self.master)
        except Exception:
            pass


@pytest.fixture
def simple_tui(tui, monkeypatch):
    """
    The real `AdaptiveDJTUI` in fallback mode, reading a pty.

    `use_urwid` is flipped rather than urwid being hidden at import time: the
    fallback path is a branch inside the same object, and the point is to drive
    the object the application actually builds.
    """
    terminal = _Pty()
    monkeypatch.setattr(sys, 'stdin', terminal.stdin)
    tui.use_urwid = False
    try:
        yield tui, terminal
    finally:
        terminal.close()


def _run(tui, terminal, keys, timeout=5.0, expect_drained=True):
    """
    Run the loop, feed it `keys`, and make sure it cannot outlive the test.

    The keys go in on the loop's **first tick**, not before it.  `tty.setcbreak`
    flushes with `TCSAFLUSH`, so anything typed before the loop calls it is
    discarded by the terminal driver — which is correct for the application
    (keys struck before the UI exists are not for the UI) and is why a harness
    that pre-loaded the pty read nothing at all and quietly passed on a
    watchdog timeout instead.

    Hooking the tick rather than sleeping keeps it deterministic: the first tick
    provably happens after `setcbreak`.

    Keys go in one per tick rather than as one burst, which is both what real
    typing looks like and what keeps `[I]` from deadlocking the harness: a
    burst is consumed by a single read, so a key meant to dismiss the info page
    would already have been eaten before the page opened.  It costs no wall
    clock — a tick whose key is waiting returns from `select` immediately.
    """
    tick = tui._update_display
    remaining = list(keys) + ["q"]

    def paced_tick():
        tick()
        if remaining:
            terminal.send(remaining.pop(0))

    tui._update_display = paced_tick

    watchdog = threading.Timer(timeout, lambda: setattr(tui, 'running', False))
    watchdog.start()
    try:
        tui.run()
    finally:
        watchdog.cancel()
        # Without this, a loop that read nothing at all would end on the
        # watchdog and every assertion downstream would be vacuously true —
        # which is how the first version of this harness "passed".
        if expect_drained:
            assert not remaining, f"the loop stopped with {remaining} unsent"


def test_the_fallback_loop_starts_reads_and_quits(simple_tui, capsys):
    tui, terminal = simple_tui
    _run(tui, terminal, [])

    assert tui.running is False
    assert "simple mode" in capsys.readouterr().out


def test_the_fallback_loop_draws_the_session_panel(simple_tui, capsys, library):
    """
    It has to show the same things the urwid panel does — Stage 3 gave it the
    `↓ next:` line, the history rows and the marks, and nothing had ever checked
    that they render.
    """
    tui, terminal = simple_tui
    tui.dj.mpd_controller.add_track(library.track_list[0])
    tui.dj.mpd_controller.add_track(library.track_list[1])
    tui.dj.mpd_controller.play()

    _run(tui, terminal, [])

    out = capsys.readouterr().out
    assert "↓ next:" in out
    assert "SPACE=Play/Pause" in out
    # The vibe readout refuses before anything has seeded the session.
    assert "♪" in out


def test_keys_pressed_at_the_terminal_reach_the_real_actions(simple_tui):
    tui, terminal = simple_tui
    fired = []
    for name in ('_toggle_play_pause', '_skip_track', '_like_track',
                 '_volume_up', '_volume_down', '_seek_forward',
                 '_seek_backward', '_replay_focused'):
        setattr(tui, name, (lambda n: lambda *a, **k: fired.append(n))(name))

    _run(tui, terminal, [" ", "n", "l", ".", ",", "\x1b[C", "\x1b[D", "\r"])

    assert fired == ['_toggle_play_pause', '_skip_track', '_like_track',
                     '_volume_up', '_volume_down', '_seek_forward',
                     '_seek_backward', '_replay_focused']


def test_the_arrow_keys_scroll_the_history_at_a_real_terminal(simple_tui, library):
    """
    End to end: an escape sequence arriving in pieces off a pty moves the same
    cursor `ENTER` then replays from.
    """
    tui, terminal = simple_tui
    tui.history.note_playing(library.track_list[0])
    tui.history.note_playing(library.track_list[1])
    tui.history.note_playing(library.track_list[2])

    _run(tui, terminal, ["\x1b[B", "\x1b[B", "\x1b[A"])

    assert tui.history.focus == 1
    assert tui.history.focused_track() == library.track_list[1]


def test_the_i_key_shows_the_model_and_waits_for_a_key(simple_tui, capsys):
    """
    `[I]` had no binding here at all, while `_show_model_info()` already
    returned its lines for a non-urwid caller — a loose end rather than an
    intended asymmetry (§8, trap 4).

    The page blocks, so its dismissal cannot come from the tick that opened it;
    a timer supplies the keypress the way a listener would.
    """
    tui, terminal = simple_tui
    dismiss = threading.Timer(0.4, lambda: terminal.send("x"))
    dismiss.start()
    try:
        _run(tui, terminal, ["i"])
    finally:
        dismiss.cancel()

    out = capsys.readouterr().out
    assert "any key to return" in out
    assert "TASTE MODEL" in out.upper()
    assert "I=Info" in out


def test_the_info_page_does_not_survive_a_shutdown_request(simple_tui):
    """
    H3's shape in the one interface H3 was never driven through.  The page holds
    the only thread there is, so a `select` with no timeout would sit there
    through a SIGTERM — `request_exit()` sets `running` false and nothing would
    read it.  It polls instead, so this returns rather than hanging.
    """
    tui, terminal = simple_tui
    tui.use_urwid = False
    tui.running = True

    stopper = threading.Timer(0.3, tui.request_exit)
    stopper.start()
    try:
        finished = threading.Event()
        worker = threading.Thread(
            target=lambda: (tui._wait_for_any_key(), finished.set()))
        worker.start()
        assert finished.wait(timeout=5.0), "the info page ignored the shutdown"
    finally:
        stopper.cancel()
        worker.join(timeout=1)


def test_q_ends_the_session_from_the_terminal(simple_tui):
    tui, terminal = simple_tui
    _run(tui, terminal, [])
    assert tui.running is False


def test_the_terminal_is_left_in_the_mode_it_was_found_in(simple_tui):
    """
    `tty.setcbreak` alters the caller's terminal.  Failing to restore it leaves
    the user's shell with no echo and no line editing after the DJ exits — the
    same class of side effect as leaving MPD in consume mode (C2).
    """
    import termios

    tui, terminal = simple_tui
    before = termios.tcgetattr(sys.stdin)

    _run(tui, terminal, ["n"])

    assert termios.tcgetattr(sys.stdin) == before


def test_the_terminal_is_restored_even_when_an_action_raises(simple_tui):
    """The restore is in a `finally`, and this is what proves it."""
    import termios

    tui, terminal = simple_tui
    before = termios.tcgetattr(sys.stdin)

    def boom():
        raise RuntimeError("action failed")
    tui._skip_track = boom

    with pytest.raises(RuntimeError):
        _run(tui, terminal, ["n"], expect_drained=False)

    assert termios.tcgetattr(sys.stdin) == before
    assert tui.running is False
