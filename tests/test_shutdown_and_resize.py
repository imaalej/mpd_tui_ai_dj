"""
The two Stage 4 findings that are about what happens *around* the session:
the orphaned ueberzugpp child (L9) and the SIGWINCH handler that was installed
and never once invoked (L1).

Both are things the audit says share a root cause with H3 — a shutdown path that
was unreachable, and a handler that something else replaced — so both are
asserted here against the real objects rather than by reading the source.
"""

import signal
import subprocess
import sys

import pytest

import album_art
from album_art import AlbumArtRenderer, UeberzugppProtocol, _terminate


# ── L9 · the ueberzugpp child ────────────────────────────────────────────────


class _FakeChild:
    """A subprocess.Popen stand-in that records how it was ended."""

    def __init__(self, alive=True, ignores_terminate=False):
        self.alive = alive
        self.ignores_terminate = ignores_terminate
        self.terminated = 0
        self.killed = 0
        self.waited = 0

    def poll(self):
        return None if self.alive else 0

    def terminate(self):
        self.terminated += 1
        if not self.ignores_terminate:
            self.alive = False

    def kill(self):
        self.killed += 1
        self.alive = False

    def wait(self, timeout=None):
        self.waited += 1
        if self.alive:
            raise subprocess.TimeoutExpired("ueberzugpp", timeout)
        return 0


def test_terminate_ends_a_running_child():
    child = _FakeChild()
    _terminate(child)
    assert child.terminated == 1
    assert child.alive is False


def test_terminate_kills_a_child_that_ignores_terminate():
    """
    `wait(timeout=1)` used to be the end of it, so a child that did not go on
    SIGTERM was left running with the exception swallowed.
    """
    child = _FakeChild(ignores_terminate=True)
    _terminate(child)
    assert child.terminated == 1
    assert child.killed == 1


def test_terminate_leaves_an_already_dead_child_alone():
    child = _FakeChild(alive=False)
    _terminate(child)
    assert child.terminated == 0


def test_terminate_of_nothing_is_harmless():
    _terminate(None)


def test_the_protocol_ends_its_child_on_shutdown():
    protocol = UeberzugppProtocol()
    child = _FakeChild()
    protocol.process = child

    protocol.shutdown()

    assert child.terminated == 1
    assert protocol.process is None


def test_shutdown_is_idempotent():
    protocol = UeberzugppProtocol()
    protocol.process = _FakeChild()
    protocol.shutdown()
    protocol.shutdown()          # must not raise


class _FakeProtocol:
    def __init__(self):
        self.cleared = 0
        self.shutdowns = 0

    def render(self, *a, **k):
        pass

    def clear(self):
        self.cleared += 1

    def shutdown(self):
        self.shutdowns += 1


@pytest.fixture
def renderer(monkeypatch):
    """An AlbumArtRenderer with detection stubbed, so no child is spawned."""
    monkeypatch.setattr(AlbumArtRenderer, '_detect_protocol', lambda self: None)
    instance = AlbumArtRenderer()
    instance.protocol = _FakeProtocol()
    instance.available = True
    return instance


def test_the_renderer_clears_and_ends_the_child(renderer):
    """
    `_shutdown()` called `clear()` and stopped, which removes the picture and
    leaves the process (L9).  Cleanup was left to `__del__` at interpreter exit,
    which is not guaranteed — and on SIGTERM did not happen at all, because
    `_shutdown()` was itself unreachable there until H3.
    """
    renderer.shutdown()

    assert renderer.protocol.cleared == 1
    assert renderer.protocol.shutdowns == 1
    assert renderer.is_available() is False


def test_a_renderer_shut_down_twice_is_harmless(renderer):
    """
    It is reachable twice by design — the signal path and `_shutdown()` both
    call it — so the second call has to be a no-op rather than an error.  The
    protocol's own `shutdown()` clears its handle, so repeating it terminates
    nothing a second time.
    """
    renderer.shutdown()
    renderer.shutdown()
    assert renderer.is_available() is False


def test_a_protocol_that_raises_on_shutdown_does_not_escape(renderer):
    def boom():
        raise OSError("broken pipe")
    renderer.protocol.shutdown = boom

    renderer.shutdown()                          # must not raise
    assert renderer.is_available() is False


def test_the_dj_ends_the_child_on_a_normal_exit(tui, fake_art):
    """Through the orchestrator, which is where the call has to be."""
    from main_tui import AdaptiveDJWithTUI

    dj = AdaptiveDJWithTUI.__new__(AdaptiveDJWithTUI)
    dj.tui = tui
    AdaptiveDJWithTUI._shutdown_album_art(dj)

    assert fake_art.shutdowns == 1


def test_the_signal_path_ends_the_child_too(tui, fake_art):
    """
    The half that was actually broken.  `[Q]` reached `_shutdown()`; a SIGTERM
    did not, so every stop that was not a keypress orphaned the process.
    """
    import threading
    from main_tui import AdaptiveDJWithTUI

    dj = AdaptiveDJWithTUI.__new__(AdaptiveDJWithTUI)
    dj.tui = tui
    dj.running = True
    dj._original_mpd_modes = {}
    dj._modes_restored = False
    dj._restore_lock = threading.Lock()

    AdaptiveDJWithTUI._signal_handler(dj, signal.SIGTERM, None)

    assert fake_art.shutdowns == 1
    assert dj.running is False


def test_a_failing_art_shutdown_cannot_break_the_signal_path(tui, monkeypatch):
    """
    Nothing in the art cleanup may escape: it runs *before* `request_exit()`,
    the call that unblocks urwid and makes the whole rest of the shutdown
    reachable.  An exception here would trade L9 for H3.
    """
    import threading
    from main_tui import AdaptiveDJWithTUI

    def boom():
        raise RuntimeError("ueberzugpp is gone")
    monkeypatch.setattr(tui.album_art_renderer, 'shutdown', boom)

    dj = AdaptiveDJWithTUI.__new__(AdaptiveDJWithTUI)
    dj.tui = tui
    dj.running = True
    dj._original_mpd_modes = {}
    dj._modes_restored = False
    dj._restore_lock = threading.Lock()

    exits = []
    monkeypatch.setattr(tui, 'request_exit', lambda: exits.append(True))

    AdaptiveDJWithTUI._signal_handler(dj, signal.SIGTERM, None)

    assert exits == [True], "the shutdown never reached request_exit()"


def test_a_tui_without_a_renderer_is_survived():
    """The signal handler runs before the TUI exists on an early failure."""
    from main_tui import AdaptiveDJWithTUI

    dj = AdaptiveDJWithTUI.__new__(AdaptiveDJWithTUI)
    dj.tui = None
    AdaptiveDJWithTUI._shutdown_album_art(dj)


# ── L2 · what a kitty or sixel user is told ──────────────────────────────────


def test_a_kitty_terminal_is_told_why_there_is_no_art(monkeypatch, capsys):
    monkeypatch.setenv("TERM", "xterm-kitty")
    monkeypatch.delenv("KITTY_WINDOW_ID", raising=False)

    album_art._warn_about_unsupported_terminal()

    message = capsys.readouterr().err
    assert "ueberzugpp" in message
    assert "kitty" in message


def test_a_sixel_terminal_is_told_too(monkeypatch, capsys):
    monkeypatch.setenv("TERM", "mlterm")
    monkeypatch.delenv("KITTY_WINDOW_ID", raising=False)

    album_art._warn_about_unsupported_terminal()

    assert "ueberzugpp" in capsys.readouterr().err


def test_an_ordinary_terminal_gets_no_such_message(monkeypatch, capsys):
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("KITTY_WINDOW_ID", raising=False)

    album_art._warn_about_unsupported_terminal()

    assert capsys.readouterr().err == ""


# ── L1 · the SIGWINCH handler ────────────────────────────────────────────────


def test_urwid_already_chains_to_our_handler(tui):
    """
    L1's measurement is right and its conclusion is not, on this urwid.

    `getsignal(SIGWINCH)` after `Screen.start()` does return urwid's handler —
    but `_posix_raw_display.Screen` captures what was there into
    `_prev_sigwinch_handler` (line 129) and calls it (line 98), and
    `Screen.stop()` puts it back (line 142).  urwid wraps ours rather than
    replacing it, which looks identical from `getsignal` and is not the same
    thing at all.  So the handler installed in `_setup_urwid` was being invoked
    the whole time.
    """
    import inspect

    import urwid.display._posix_raw_display as prd

    # The chaining is in urwid's source, not inferred from behaviour.
    handler = inspect.getsource(prd.Screen._sigwinch_handler)
    assert "_prev_sigwinch_handler(signum, frame)" in handler
    started = inspect.getsource(prd.Screen)
    assert "self._prev_sigwinch_handler = self.signal_handler_setter(" in started

    class _Screen:
        _prev_sigwinch_handler = None

    screen = _Screen()
    screen._prev_sigwinch_handler = tui._on_sigwinch
    tui.loop.screen = screen

    assert tui._install_sigwinch_chain() is False, (
        "installing on top of a chain that already reaches us makes a cycle"
    )


def test_installing_on_top_of_urwids_chain_would_recurse(tui, fake_art):
    """
    The bug this replaced, pinned so it cannot come back.  Ours → urwid's →
    ours (as urwid's `_prev`) → urwid's → … which is a `RecursionError` on the
    first real resize, and is exactly what a live run produced.
    """
    calls = []

    def urwid_handler(signum, frame):
        calls.append("urwid")
        # What urwid actually does: call whatever was installed before it.
        if callable(tui._urwid_sigwinch):
            tui._urwid_sigwinch(signum, frame)

    class _Screen:
        _prev_sigwinch_handler = None

    screen = _Screen()
    screen._prev_sigwinch_handler = tui._on_sigwinch
    tui.loop.screen = screen
    signal.signal(signal.SIGWINCH, urwid_handler)

    tui._install_sigwinch_chain()
    tui._on_sigwinch(signal.SIGWINCH, None)      # must terminate

    assert calls == [], "we must not call back into the handler that calls us"


def test_the_handler_is_installed_where_urwid_does_not_chain(tui):
    """
    Older urwid has no `_prev_sigwinch_handler`, and there the finding is
    literally true — so there we do install on top, and delegate onward.
    """
    class _Screen:
        pass

    tui.loop.screen = _Screen()
    seen = []
    signal.signal(signal.SIGWINCH, lambda signum, frame: seen.append(signum))

    assert tui._install_sigwinch_chain() is True
    assert signal.getsignal(signal.SIGWINCH) == tui._on_sigwinch

    tui._on_sigwinch(signal.SIGWINCH, None)
    assert seen == [signal.SIGWINCH], "the displaced handler must still run"


def test_reinstalling_twice_does_not_chain_onto_itself(tui):
    """
    A tick runs every 0.5 s and calls this every time.  Capturing our own
    handler as the one to delegate to would build a recursion one frame deeper
    per tick.
    """
    class _Screen:
        pass

    tui.loop.screen = _Screen()
    signal.signal(signal.SIGWINCH, lambda *a: None)
    tui._install_sigwinch_chain()

    assert tui._install_sigwinch_chain() is False
    assert tui._urwid_sigwinch != tui._on_sigwinch


def test_the_signal_marks_the_art_dirty(tui, fake_art):
    """
    What the handler is *for*.  A resize already self-heals — `_art_geometry()`
    is re-derived every tick, so the renderer's key changes on its own.  What
    does not self-heal is a **window move at unchanged size**: same key, send
    skipped, stale image.  `force_redraw()` is the answer to that one, and this
    is the path that reaches it.
    """
    redraws = []
    fake_art.force_redraw = lambda: redraws.append(True)

    tui._on_sigwinch(signal.SIGWINCH, None)

    assert redraws == [True]


def test_a_default_handler_is_not_chained_into(tui):
    """SIG_DFL and SIG_IGN are integers, not callables."""
    signal.signal(signal.SIGWINCH, signal.SIG_DFL)
    tui._install_sigwinch_chain()

    assert tui._urwid_sigwinch is None
    tui._on_sigwinch(signal.SIGWINCH, None)      # must not raise


def test_the_first_tick_installs_it(tui, monkeypatch):
    """
    The tick is where this has to happen: urwid's handler is installed by
    `loop.run()`, which has not been called when `_setup_urwid` runs.
    """
    monkeypatch.setattr(tui, '_update_display', lambda: None)
    monkeypatch.setattr(tui.loop, 'set_alarm_in', lambda *a, **k: None)
    signal.signal(signal.SIGWINCH, lambda *a: None)

    tui._periodic_update()

    assert signal.getsignal(signal.SIGWINCH) == tui._on_sigwinch


def test_quitting_hands_the_handler_back(tui):
    """
    urwid is still holding the screen when `_quit` runs, and its own handler is
    how it learns about a resize during teardown — so dropping to SIG_DFL there
    would take that away from it.
    """
    def urwid_handler(*a):
        pass
    signal.signal(signal.SIGWINCH, urwid_handler)
    tui._install_sigwinch_chain()

    tui.use_urwid = False        # so _quit does not raise ExitMainLoop
    tui._quit()

    assert signal.getsignal(signal.SIGWINCH) == urwid_handler


# ── the `[I]` overlay's inner size, derived rather than recomputed ───────────


def _overlay_for(tui, lines=40):
    import urwid
    listbox = urwid.ListBox(urwid.SimpleListWalker(
        [urwid.Text(f"line {i}") for i in range(lines)]))
    box = urwid.LineBox(listbox, title="Model Info")
    return urwid.Overlay(box, tui.frame, align="center",
                         width=("relative", 70), valign="middle", height=20), listbox


@pytest.mark.parametrize("size", [(80, 24), (100, 30), (120, 45), (200, 60)])
def test_the_inspector_inner_size_matches_what_urwid_renders(tui, size, monkeypatch):
    """
    The same principle `test_art_geometry.py` applies to the album art: assert
    against what the widget actually produces, not against a repeat of the
    calculation.  The old code recomputed urwid's `("relative", 70)` arithmetic
    by hand as `int(cols * 0.7) - 2`.
    """
    monkeypatch.setattr(tui.loop.screen, 'get_cols_rows', lambda: size)
    overlay, listbox = _overlay_for(tui)

    derived = tui._overlay_inner_size(overlay)

    # What urwid hands the LineBox, straight from the rendered canvas.
    canvas = overlay.render(size, focus=True)
    assert canvas.cols() == size[0] and canvas.rows() == size[1]

    left, right, top, bottom = overlay.calculate_padding_filler(size, True)
    box_cols = size[0] - left - right
    box_rows = size[1] - top - bottom
    assert derived == (box_cols - 2, box_rows - 2)


def test_the_inner_size_follows_a_resize_while_the_page_is_open(tui, monkeypatch):
    """
    `[I]` can be held open indefinitely now that ↑↓ scroll rather than dismiss,
    so a size read once at open time goes stale on a resize.
    """
    overlay, _ = _overlay_for(tui)

    monkeypatch.setattr(tui.loop.screen, 'get_cols_rows', lambda: (80, 24))
    narrow = tui._overlay_inner_size(overlay)
    monkeypatch.setattr(tui.loop.screen, 'get_cols_rows', lambda: (200, 60))
    wide = tui._overlay_inner_size(overlay)

    assert wide[0] > narrow[0]


def test_the_inner_size_is_never_degenerate(tui, monkeypatch):
    """A ListBox handed a zero or negative size raises rather than scrolling."""
    overlay, _ = _overlay_for(tui)
    monkeypatch.setattr(tui.loop.screen, 'get_cols_rows', lambda: (4, 3))

    cols, rows = tui._overlay_inner_size(overlay)
    assert cols >= 1 and rows >= 1


def test_scrolling_the_inspector_actually_moves_it(tui, monkeypatch):
    """The size is only worth deriving if it is the one the ListBox is driven
    with, so this drives it."""
    monkeypatch.setattr(tui.loop.screen, 'get_cols_rows', lambda: (120, 45))
    overlay, listbox = _overlay_for(tui, lines=200)

    before = listbox.get_focus()[1]
    listbox.keypress(tui._overlay_inner_size(overlay), "page down")
    assert listbox.get_focus()[1] > before
