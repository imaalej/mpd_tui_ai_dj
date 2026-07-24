"""
Phase 2 — the album-art lifecycle findings (C1, C2, C4, C5) and the `[I]`
overlay art clear (F3).

Driven against the real `AlbumArtRenderer` / protocol objects, with the child
process and its stdin pipe faked so nothing spawns `ueberzugpp`.  The point that
Stage 3 established for geometry — assert against what the objects actually do,
not against a mirror of the code — applies here too: a pipe break is simulated
at the stdin layer and the respawn is observed through the process handles.
"""

import subprocess

import pytest

import album_art
from album_art import (
    AlbumArtRenderer,
    UeberzugppProtocol,
    _atomic_write_bytes,
    cleanup_cover_cache,
)


# ── fakes ────────────────────────────────────────────────────────────────────


class _FakeStdin:
    def __init__(self, child):
        self.child = child
        self.writes = []

    def write(self, data):
        if self.child.break_pipe:
            raise BrokenPipeError("broken pipe")
        self.writes.append(data)

    def flush(self):
        if self.child.break_pipe:
            raise BrokenPipeError("broken pipe")


class _FakeChild:
    """A Popen stand-in whose pipe can be told to break on write."""

    def __init__(self, alive=True, break_pipe=False):
        self._alive = alive
        self.break_pipe = break_pipe
        self.stdin = _FakeStdin(self)
        self.returncode = 0
        self.terminated = 0
        self.killed = 0

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self.terminated += 1
        self._alive = False

    def kill(self):
        self.killed += 1
        self._alive = False

    def wait(self, timeout=None):
        return 0


@pytest.fixture
def no_sleep(monkeypatch):
    """`_start_layer` sleeps 0.15 s to see if the child survives; skip it."""
    monkeypatch.setattr(album_art.time, "sleep", lambda *_a, **_k: None)


@pytest.fixture
def art_image(tmp_path):
    p = tmp_path / "cover.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n fake image bytes")
    return p


def _spawns(monkeypatch, children):
    """Make `_start_layer` hand out the given children in order, then raise
    FileNotFoundError to model a binary that has gone away."""
    seq = list(children)

    def fake_popen(*_a, **_k):
        if not seq:
            raise FileNotFoundError("ueberzugpp")
        return seq.pop(0)

    monkeypatch.setattr(album_art.subprocess, "Popen", fake_popen)


# ── C1 · a broken pipe must reach the respawn ────────────────────────────────


def test_a_broken_write_respawns_and_still_paints(monkeypatch, no_sleep, art_image):
    """
    The core of C1.  Old behaviour: `BrokenPipeError` set `process = None` and
    returned, and the next render hit `if not self.process: return` before the
    respawn branch — so one broken write disabled art for the session.  Now the
    broken write drops the dead child, respawns, and retries in the same call.
    """
    first = _FakeChild(alive=True)
    second = _FakeChild(alive=True)
    _spawns(monkeypatch, [first, second])

    proto = UeberzugppProtocol()
    assert proto._start_layer() is True
    assert proto.process is first

    first.break_pipe = True                       # its pipe is now broken
    ok = proto.render(art_image, 0, 0, 10, 10)

    assert ok is True, "the render should have respawned and painted"
    assert proto.process is second, "a fresh child should be holding the layer"
    assert proto.is_alive()
    assert len(second.stdin.writes) == 1, "the add command reached the new child"


def test_a_dead_child_is_respawned_before_the_write(monkeypatch, no_sleep, art_image):
    """A child that exited (not a pipe break) is also respawned on next render."""
    dead = _FakeChild(alive=False)
    fresh = _FakeChild(alive=True)
    _spawns(monkeypatch, [fresh])

    proto = UeberzugppProtocol()
    proto.process = dead

    assert proto.render(art_image, 0, 0, 10, 10) is True
    assert proto.process is fresh


def test_render_reports_failure_when_the_binary_is_gone(monkeypatch, no_sleep, art_image):
    """
    If the respawn cannot happen because the binary is missing, render returns
    False and the protocol marks itself unavailable — the honest signal the
    renderer reads.
    """
    broken = _FakeChild(alive=True, break_pipe=True)
    _spawns(monkeypatch, [])                       # every Popen → FileNotFoundError

    proto = UeberzugppProtocol()
    proto.process = broken
    proto.available = True

    assert proto.render(art_image, 0, 0, 10, 10) is False
    assert proto.available is False
    assert proto.process is None


# ── C1/C2 · the renderer's flicker guard must not wedge a dead child ─────────


def _renderer_with(monkeypatch, protocol):
    monkeypatch.setattr(AlbumArtRenderer, "_detect_protocol", lambda self: None)
    r = AlbumArtRenderer()
    r.protocol = protocol
    r.available = True
    return r


def test_the_flicker_skip_holds_only_while_the_child_is_alive(
    monkeypatch, no_sleep, art_image
):
    """
    The renderer skips re-sending an identical frame to avoid ueberzug's
    remove+redraw flicker — but only while a child is alive to hold it.  After
    the child dies the skip would freeze a blank screen (the C2 symptom), so the
    guard must fall through and respawn.
    """
    live = _FakeChild(alive=True)
    fresh = _FakeChild(alive=True)
    _spawns(monkeypatch, [fresh])

    proto = UeberzugppProtocol()
    proto.process = live
    r = _renderer_with(monkeypatch, proto)

    r.render(art_image, 1, 2, 10, 10)
    assert len(live.stdin.writes) == 1

    # Same key, child still alive → skipped, no second write.
    r.render(art_image, 1, 2, 10, 10)
    assert len(live.stdin.writes) == 1

    # The child dies; the identical key must no longer be skipped.
    live._alive = False
    r.render(art_image, 1, 2, 10, 10)
    assert proto.process is fresh
    assert len(fresh.stdin.writes) == 1
    assert r.is_available() is True


def test_a_transient_spawn_failure_keeps_the_renderer_available(
    monkeypatch, no_sleep, art_image
):
    """
    A child that dies at once (busy mid-resize, say) is transient: the renderer
    must keep `is_available()` True and retry next tick, not disable art for the
    session on one hiccup — that would be C2 by another route.
    """
    proto = UeberzugppProtocol()
    proto.process = _FakeChild(alive=True, break_pipe=True)
    proto.available = True                  # a protocol that detected fine earlier
    r = _renderer_with(monkeypatch, proto)

    # Respawn hands back a child that is already dead → _start_layer returns
    # False but leaves `available` True.
    _spawns(monkeypatch, [_FakeChild(alive=False)])

    r.render(art_image, 0, 0, 10, 10)
    assert r.is_available() is True
    assert r._render_key is None, "a failed frame must not be remembered as up"


def test_the_renderer_disables_when_the_binary_is_gone(
    monkeypatch, no_sleep, art_image, capsys
):
    """When the child is gone *and* cannot be restarted, is_available() flips
    honestly to False (C1's 'still reports True' half)."""
    proto = UeberzugppProtocol()
    proto.process = _FakeChild(alive=True, break_pipe=True)
    proto.available = True
    r = _renderer_with(monkeypatch, proto)

    _spawns(monkeypatch, [])                        # respawn → FileNotFoundError

    r.render(art_image, 0, 0, 10, 10)
    assert r.is_available() is False
    assert "disabled" in capsys.readouterr().err


# ── C4 · the cover-cache dir must be cleaned on every exit path ──────────────


def test_cleanup_removes_the_cache_dir_and_is_idempotent():
    d = album_art._get_cache_dir()
    assert d.exists()

    cleanup_cover_cache()
    assert not d.exists()
    assert album_art._COVER_CACHE_DIR is None

    cleanup_cover_cache()          # a second call (atexit after signal) is a no-op


def test_the_renderer_shutdown_cleans_the_cache(monkeypatch):
    """
    `atexit` does not fire on a default SIGTERM, so the dir is torn down from
    `renderer.shutdown()` — which the signal path and `_shutdown()` both call.
    """
    class _Proto:
        def clear(self):
            pass

        def shutdown(self):
            pass

    monkeypatch.setattr(AlbumArtRenderer, "_detect_protocol", lambda self: None)
    r = AlbumArtRenderer()
    r.protocol = _Proto()
    r.available = True

    d = album_art._get_cache_dir()
    assert d.exists()

    r.shutdown()
    assert not d.exists()
    assert r.is_available() is False


def test_the_dj_signal_path_cleans_the_cache(tui, fake_art):
    """End to end: a SIGTERM reaches `renderer.shutdown()`, which reaches the
    cache cleanup.  Uses the real renderer so the cache logic runs."""
    import signal
    import threading

    from album_art import AlbumArtRenderer
    from main_tui import AdaptiveDJWithTUI

    monkeyfree = AlbumArtRenderer.__new__(AlbumArtRenderer)
    monkeyfree.protocol = None
    monkeyfree.available = True
    monkeyfree.current_image = None
    monkeyfree._render_key = None
    tui.album_art_renderer = monkeyfree

    d = album_art._get_cache_dir()
    assert d.exists()

    dj = AdaptiveDJWithTUI.__new__(AdaptiveDJWithTUI)
    dj.tui = tui
    dj.running = True
    dj._original_mpd_modes = {}
    dj._modes_restored = False
    dj._restore_lock = threading.Lock()

    AdaptiveDJWithTUI._signal_handler(dj, signal.SIGTERM, None)

    assert not d.exists()


# ── C5 · cover-cache writes are atomic ───────────────────────────────────────


def test_atomic_write_lands_the_bytes(tmp_path):
    target = tmp_path / "abc.jpg"
    _atomic_write_bytes(target, b"hello cover")
    assert target.read_bytes() == b"hello cover"
    assert not (tmp_path / "abc.jpg.tmp").exists(), "the temp file is renamed away"


def test_a_crash_before_the_rename_leaves_no_servable_file(monkeypatch, tmp_path):
    """
    The whole point: a truncated file must not appear at the path the
    `cached.exists()` fast path serves.  The temp may survive a hard kill, but
    its `.tmp` suffix is not one the fast path checks.
    """
    target = tmp_path / "abc.jpg"

    def boom(*_a, **_k):
        raise OSError("killed mid-rename")

    monkeypatch.setattr(album_art.os, "replace", boom)

    with pytest.raises(OSError):
        _atomic_write_bytes(target, b"x" * 100)

    assert not target.exists(), "no partial at the served path"


# ── F3 · the `[I]` overlay takes the cover down while it is open ─────────────


def test_the_info_overlay_clears_and_restores_the_art(tui, fake_art, monkeypatch):
    """
    The cover is a separate surface urwid's overlay does not paint over, so it
    sits on top of the inspector unless taken down (F3).  Opening `[I]` must
    clear it and closing must mark it dirty so the next tick repaints.
    """
    monkeypatch.setattr(tui.loop, "draw_screen", lambda: None)
    monkeypatch.setattr(tui.loop.screen, "get_input", lambda: ["q"])
    monkeypatch.setattr(tui.loop.screen, "set_input_timeouts", lambda **k: None)
    monkeypatch.setattr(tui.loop.screen, "get_cols_rows", lambda: (80, 24))

    redraws = []
    fake_art.force_redraw = lambda: redraws.append(True)
    clears_before = fake_art.clears

    tui._show_model_info()

    assert fake_art.clears > clears_before, "art was not cleared while [I] was open"
    assert redraws == [True], "art was not marked dirty on close"
