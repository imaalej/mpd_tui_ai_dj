"""
Terminal User Interface - Adaptive Session AI DJ
Full-featured TUI with real-time updates and responsive controls.

Layout (rows, top to bottom):
  [1]  Header bar
  [N]  Now Playing  ── two columns, height packed to its own contents:
         left:  album art area (fixed ART_COLS wide)
         right: status · track info · seek bar · descriptors · drift
  [7]  Console      ── live state updates, fixed height
  [M]  Session      ── `↓ next:` plus session history, takes what is left
  [1]  Footer bar

Two things about this file are the point of Stage 3 rather than incidental to it.

**The album art position is derived, not counted.**  It used to be pinned to
`x=2, y=3, height=RIGHT_COL_ROWS=10`, where `RIGHT_COL_ROWS` was a hand-counted
comment listing the widgets in the right-hand Pile.  Every item in this stage
changes that Pile, and the constant carried no way to notice.  `_art_geometry()`
now asks the widget tree — `Pile.get_item_rows`, `Columns.column_widths`,
`Widget.rows` — so adding a row to the right column moves the art by exactly one
row with no edit here at all (audit H8/L3).  The hand count was also *wrong*:
the art column starts at column 1, not 2, and its true height at 80×40 was 10
rows against the 10 the constant claimed only because the two errors were in
different directions.

**The Now Playing box is packed, which is what makes it render at all.**  It used
to take `('weight', 3, …)` of the body.  Its content is a flow Pile containing a
Columns whose left cell is a box Filler, so urwid resolved it as a box widget and
then handed it a height that did not match the one it rendered — a `WidgetError`
on **every terminal shorter than 33 rows**, including the classic 80×24.  Nothing
caught it because nothing in the suite had ever constructed the widget tree.
`('pack', …)` gives it exactly `rows()`, which is both correct and the reason the
art geometry is now independent of the terminal's height.
"""

import os
import sys
import io
import time
import signal
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional, List

try:
    import urwid

    URWID_AVAILABLE = True
except ImportError:
    URWID_AVAILABLE = False
    print("Warning: urwid not available. TUI features limited.", file=sys.stderr)

from config import config
from album_art import get_album_art_renderer
from session_history import SessionHistory
from vibe_readout import VibeReadout, format_descriptors


# ─── Console log interceptor ────────────────────────────────────────────────


class _ConsoleCapture(io.TextIOBase):
    """
    Drop-in replacement for sys.stderr that:
      • Stores the last N lines in a ring-buffer for the TUI console widget.
      • Tees every line to config.log_file, so nothing is lost when the TUI
        scrolls it off the 5-line console panel or swallows it entirely.
      • Passes everything through to the real stderr ONLY when the TUI is not
        active, so messages never bleed through the urwid layout.

    The log tee is the point of this class existing (audit L5).  Without it a
    traceback from the background thread flashes through a five-line ring buffer
    and is then unrecoverable, which is the single biggest obstacle to
    diagnosing anything that happens during a real session.
    """

    MAX_LINES = 200

    def __init__(self, real_stderr, log_path: Optional[Path] = None):
        super().__init__()
        self._real = real_stderr
        self._buf = deque(maxlen=self.MAX_LINES)
        self._lock = threading.Lock()
        self._partial = ""  # accumulate until newline
        # Set to True by AdaptiveDJTUI.run() while the urwid loop is active.
        # When True, writes are captured only — never forwarded to the raw
        # terminal, which would bleed through the urwid layout.
        self.tui_active: bool = False
        self._log = self._open_log(log_path)

    def _open_log(self, log_path: Optional[Path]):
        """
        Open the session log in append mode.  Failure here must never take the
        application down — a missing log is a degraded session, not a dead one.
        """
        if log_path is None:
            return None
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = open(log_path, "a", encoding="utf-8", buffering=1)
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            handle.write(f"\n===== session started {stamp} =====\n")
            return handle
        except Exception as e:
            self._real.write(f"Warning: could not open log file {log_path}: {e}\n")
            self._real.flush()
            return None

    def write(self, text: str) -> int:
        # Only forward to the real terminal when the TUI is NOT running.
        # While urwid owns the screen, any raw write to the terminal fd
        # will appear on top of the TUI widgets (the bleed-through bug).
        if not self.tui_active:
            self._real.write(text)
            self._real.flush()
        self._partial += text
        while "\n" in self._partial:
            line, self._partial = self._partial.split("\n", 1)
            line = line.rstrip()
            if line:
                ts = datetime.now().strftime("%H:%M:%S")
                stamped = f"[{ts}] {line}"
                with self._lock:
                    self._buf.append(stamped)
                    self._write_log(stamped)
        return len(text)

    def _write_log(self, line: str):
        """Append one already-stamped line to the log. Caller holds the lock."""
        if self._log is None:
            return
        try:
            self._log.write(line + "\n")
        except Exception:
            # A broken log must not turn into an infinite recursion through
            # stderr.  Drop the handle and carry on silently.
            self._log = None

    def flush(self):
        if not self.tui_active:
            self._real.flush()
        with self._lock:
            if self._log is not None:
                try:
                    self._log.flush()
                except Exception:
                    self._log = None

    def get_lines(self) -> List[str]:
        with self._lock:
            return list(self._buf)

    def fileno(self):
        return self._real.fileno()


# Install the interceptor once at module import so every stderr write is
# captured from the very first moment, including startup messages.
_console_capture: Optional[_ConsoleCapture] = None


def _install_console_capture():
    global _console_capture
    if _console_capture is None:
        _console_capture = _ConsoleCapture(sys.__stderr__, config.log_file)
        sys.stderr = _console_capture


_install_console_capture()


# ─── TUI class ──────────────────────────────────────────────────────────────

# Width (in terminal columns) reserved for the album-art column inside the
# Now Playing box.  This one is a *declaration* — the Columns cell is built to
# it — rather than a count of something else, which is why it survives H8.
ART_COLS = 33

# Fixed row count for the console panel's ListBox (its LineBox adds 2 more).
CONSOLE_ROWS = 5

# Below this the art area is too small to be an image rather than a smear, so
# the renderer is cleared instead.  Nothing downstream depends on the numbers;
# they shape behaviour without asserting anything.
MIN_ART_ROWS = 3
MIN_ART_COLS = 8

# How many history rows the Session panel will build widgets for.  The panel is
# a few dozen rows at most; this only stops an all-night session from rendering
# hundreds of Text widgets that are scrolled off screen anyway.
HISTORY_RENDER_LIMIT = 200

# NOTE: `RIGHT_COL_ROWS` is deliberately gone.  It was the hand-counted height of
# the Now Playing right-hand Pile, and the album art was pinned to it — so every
# change to that Pile silently misplaced the image and nothing could notice.
# `_art_geometry()` asks the widget tree instead (audit H8).


# Terminal byte sequences → the key names urwid hands `_handle_input`.
#
# This exists so the fallback text mode can share the urwid mode's one dispatch
# table instead of keeping a second one (audit L9/M1c).  Only the sequences the
# bindings actually use are here; anything else decodes to None and is ignored,
# which is what urwid does with an unhandled key too.
_RAW_KEYS = {
    "\x1b[A": "up",
    "\x1b[B": "down",
    "\x1b[C": "right",
    "\x1b[D": "left",
    "\x1bOA": "up",         # some terminals send SS3 rather than CSI in
    "\x1bOB": "down",       # application-cursor mode
    "\x1bOC": "right",
    "\x1bOD": "left",
    "\r": "enter",
    "\n": "enter",
}


def decode_key(raw: str) -> Optional[str]:
    """
    Turn one terminal keypress into a key name `_handle_input` knows.

    Pure and importable on purpose: the fallback mode needs a pty to run, so
    this is the part of it that can be tested without one — and the part that
    was actually wrong before, since the two interfaces disagreed about what
    `↑` meant (L9).
    """
    if not raw:
        return None
    if raw in _RAW_KEYS:
        return _RAW_KEYS[raw]
    if len(raw) == 1 and (raw.isprintable() or raw == " "):
        return raw
    return None


def decode_keys(buffer: str):
    """
    Split a chunk of terminal input into key names, plus any incomplete tail.

    Returns `(keys, remainder)`.  `remainder` is a partial escape sequence that
    might still be completed by the next read; anything else is consumed.

    The mode used to read one character at a time and `select()` on `sys.stdin`
    between reads.  Those two do not agree about what is pending: `read()` on a
    buffered text stream pulls everything available into Python's buffer, and
    `select()` then reports the file descriptor as empty — so a burst of input
    (a held-down arrow key, a paste, anything faster than one key per tick) left
    keys sitting in the buffer, unread, until the next keypress dislodged one of
    them.  Reading the whole burst and splitting it here is what makes the two
    agree.
    """
    keys, i = [], 0
    while i < len(buffer):
        if buffer[i] == "\x1b":
            sequence = buffer[i:i + 3]
            if sequence in _RAW_KEYS:
                keys.append(_RAW_KEYS[sequence])
                i += 3
                continue
            if len(buffer) - i < 3 and any(k.startswith(buffer[i:])
                                           for k in _RAW_KEYS):
                return keys, buffer[i:]     # truncated — wait for the rest
            if buffer[i:i + 2] in ("\x1b[", "\x1bO"):
                i += 3      # a CSI/SS3 sequence we have no binding for: drop it
                continue    # whole, or its tail arrives as bare keypresses
            i += 1          # a bare ESC
            continue
        key = decode_key(buffer[i])
        if key is not None:
            keys.append(key)
        i += 1
    return keys, ""


class AdaptiveDJTUI:
    """Terminal User Interface for Adaptive Session AI DJ."""

    def __init__(self, dj, art_renderer=None):
        self.dj = dj
        self.running = False

        # UI state
        self.current_status: dict = {}
        # Written to by the signal handler to unblock urwid's MainLoop; see
        # `request_exit`.
        self._exit_pipe: Optional[int] = None
        # urwid's own SIGWINCH handler, if we ever displace it, so it can still
        # be called and still be given back (audit L1).  On urwid 3.0.5 we never
        # do: urwid chains to us rather than replacing us.
        self._urwid_sigwinch = None
        self._sigwinch_installed = False

        # What played and what became of it (audit H1c).  The model is here
        # rather than behind the display because nothing behind the display has
        # this shape — see session_history.py.
        self.history = SessionHistory()
        # Where this session's feedback events begin in the handler's history,
        # which spans every previous session once it is loaded from disk.
        self._feedback_cursor = len(getattr(
            getattr(dj, 'feedback_handler', None), 'feedback_history', []) or [])
        # Hearts survive a restart now: the likes were always on disk, only the
        # set that drew them was in memory (audit L4).
        self.history.rehydrate_likes(getattr(
            getattr(dj, 'feedback_handler', None), 'feedback_history', []) or [])

        # The descriptor readout, finally reading the bank Stage 1 built.
        self.vibe = VibeReadout(getattr(dj, 'descriptor_bank', None))

        # Album art.  Injectable so the widget tests can construct the tree
        # without detecting protocols and spawning a ueberzugpp child.
        self.album_art_renderer = (art_renderer if art_renderer is not None
                                   else get_album_art_renderer())
        self.show_album_art = self.album_art_renderer.is_available()

        if self.show_album_art:
            proto = self.album_art_renderer.protocol.__class__.__name__
            print(f"✓ Album art enabled ({proto})", file=sys.stderr)
        else:
            print("✗ Album art disabled (no supported protocol)", file=sys.stderr)

        if not URWID_AVAILABLE:
            print("\n⚠  urwid not installed. Install with: pip install urwid")
            print("Falling back to simple text interface…\n")
            self.use_urwid = False
        else:
            self.use_urwid = True
            self._setup_urwid()

    # ── urwid setup ──────────────────────────────────────────────────────────

    def _setup_urwid(self):
        palette = [
            ("header", "white,bold", "dark blue"),
            ("footer", "white", "dark blue"),
            ("playing", "light green,bold", "default"),
            ("paused", "yellow,bold", "default"),
            ("track_info", "white", "default"),
            ("vibe", "light cyan", "default"),
            ("liked", "light red,bold", "default"),
            ("seek_bar", "white", "dark gray"),
            ("seek_progress", "black", "light green"),
            ("console_text", "dark cyan", "default"),
            ("console_warn", "yellow", "default"),
            ("console_err", "light red", "default"),
            ("history_item", "light gray", "default"),
            ("history_focus", "black,bold", "dark cyan"),
            ("history_playing", "light green,bold", "default"),
            ("next_up", "light cyan", "default"),
            ("divider", "dark gray", "default"),
        ]

        # ── Header ──
        # A flow widget, not a Filler: `Frame` wants flow widgets for its header
        # and footer, and a real flow widget can report its own `rows()` — which
        # is what `_art_geometry()` reads instead of assuming 1 (H8).
        self.header_text = urwid.Text("🎵 Adaptive Session AI DJ", align="center")
        self.header = urwid.AttrMap(self.header_text, "header")

        # ── Now Playing: right column widgets ──
        self.status_text = urwid.Text(("paused", "⏸ Paused"))
        self.artist_text = urwid.Text("Artist: ---")
        self.album_text = urwid.Text("Album: ---")
        self.track_text = urwid.Text("Track: ---")

        # The vibe readout, in two rows (audit H1b).  The first names the top
        # three descriptors by z-score against this library's own distribution;
        # the second says how many of those three were also on screen a few
        # tracks ago, and how many tracks have played.  Both are gated on the
        # session vector being seeded — `bank.top(zeros)` returns a confident
        # ranking of the bank's own baselines rather than refusing, which is H1's
        # original defect in a new costume.  See vibe_readout.py.
        self.descriptor_text = urwid.Text("")
        self.drift_text = urwid.Text("")

        self.seek_bar_progress = urwid.ProgressBar(
            "seek_bar", "seek_progress", current=0, done=100
        )
        self.seek_time_text = urwid.Text("0:00 / 0:00", align="center")

        # The album art's height is this Pile's height.  Adding a row here moves
        # the art by exactly one row, with no constant to update (H8).
        self.right_col = urwid.Pile(
            [
                urwid.AttrMap(self.status_text, "track_info"),
                urwid.Divider(),
                urwid.AttrMap(self.artist_text, "track_info"),
                urwid.AttrMap(self.album_text, "track_info"),
                urwid.AttrMap(self.track_text, "track_info"),
                urwid.Divider(),
                urwid.AttrMap(self.seek_bar_progress, "seek_bar"),
                self.seek_time_text,
                urwid.Divider(),
                urwid.AttrMap(self.descriptor_text, "vibe"),
                urwid.AttrMap(self.drift_text, "vibe"),
            ]
        )
        right_col = self.right_col

        # ── Now Playing: left column (album art area) ──
        # No LineBox here — a border widget renders visible box characters in
        # the terminal that show through behind the ueberzug overlay.
        # The art area is a plain Filler; ueberzug draws the image on top.
        self.album_art_placeholder = urwid.Text(" ", align="center")
        self.art_inner = urwid.Filler(self.album_art_placeholder, valign="middle")

        # Two-column row: art (fixed width) | track info (rest).  The art cell is
        # index 0, which is what makes its x-offset inside the Columns zero; the
        # geometry derivation reads the index rather than assuming it.
        self.art_column_index = 0
        self.np_columns = urwid.Columns(
            [
                ("fixed", ART_COLS, self.art_inner),
                ("weight", 1, urwid.Padding(right_col, left=1)),
            ]
        )

        self.now_playing_content = urwid.Pile(
            [
                urwid.Divider(),
                self.np_columns,
                urwid.Divider(),
            ]
        )

        self.now_playing_box = urwid.LineBox(self.now_playing_content,
                                             title="♪ Now Playing")

        # ── Console panel ──
        self.console_walker = urwid.SimpleFocusListWalker(
            [urwid.Text(("console_text", "── console ready ──"))]
        )
        console_lb = urwid.ListBox(self.console_walker)
        # BoxAdapter gives the ListBox a fixed height inside a Pile.
        # WidgetDisable makes it non-selectable, so ↑↓ and ENTER reach
        # `unhandled_input` instead of being swallowed by this ListBox's own
        # scrolling — the Pile would otherwise focus it, being the first
        # selectable thing in the body.
        self.console_box = urwid.LineBox(
            urwid.BoxAdapter(console_lb, CONSOLE_ROWS), title="System Console"
        )

        # ── Session panel ──
        #
        # This was the "Upcoming Queue", listing ten tracks drawn from
        # `mpc playlist` — which with consume off is the session's *history*,
        # numbered as if it were the future (audit H2).  Stage 2 cut it to a
        # one-line `↓ next:`; this is what it becomes (H1c): the lookahead, a
        # divider, then what actually played, newest first, with ♥ / ⏭ / ✓.
        #
        # It is truthful because it happened, which is the whole argument for
        # replacing a queue view with a history view at depth 1.
        self.next_up_text = urwid.Text("  ↓ next:  —")
        self.history_walker = urwid.SimpleFocusListWalker([])
        self.history_listbox = urwid.ListBox(self.history_walker)
        self.session_box = urwid.LineBox(
            urwid.Pile([
                ("pack", urwid.AttrMap(self.next_up_text, "next_up")),
                ("pack", urwid.AttrMap(urwid.Divider("─"), "divider")),
                ("weight", 1, self.history_listbox),
            ]),
            title="Session",
        )

        # ── Main layout ──
        #
        # `('pack', …)` for Now Playing, not `('weight', 3, …)`.  Its content is
        # a flow Pile wrapping a Columns whose art cell is a box Filler, so urwid
        # resolves the Columns as a box widget and then renders it at its natural
        # height — which raised `WidgetError` whenever the weighted allocation
        # disagreed, i.e. on every terminal shorter than 33 rows.  Packing gives
        # it exactly `rows()`, so the mismatch cannot arise, the panel is the
        # same height on an 80×24 as on an 80×50, and the album art's position
        # stops depending on the terminal's height at all.
        self.main_pile = urwid.Pile(
            [
                ("pack", self.now_playing_box),
                (CONSOLE_ROWS + 2, urwid.WidgetDisable(self.console_box)),
                ("weight", 1, urwid.WidgetDisable(self.session_box)),
            ]
        )

        # ── Footer ──
        # Reconciled with the README table and the fallback-mode bindings, which
        # disagreed three ways (audit L9): the README listed volume on `,`/`.`,
        # the footer on `<`/`>`, and simple mode bound ↑↓ to volume while the
        # urwid mode had them unbound.  One set of bindings now, everywhere.
        footer_text = urwid.Text(
            "[SPACE] Pause   [N] Skip   [V] Pass   [L] Like   [↑↓] History   "
            "[ENTER] Replay   [,.] Vol   [←→] Seek   [I] Info   [Q] Quit",
            align="center",
        )
        # A flow widget, so a narrow terminal wraps the hints onto a second row
        # rather than silently truncating them — and so `rows()` reports the
        # truth to `_art_geometry()`.
        self.footer = urwid.AttrMap(footer_text, "footer")

        # ── Frame ──
        self.frame = urwid.Frame(
            body=self.main_pile,
            header=self.header,
            footer=self.footer,
        )

        self.loop = urwid.MainLoop(
            self.frame,
            palette=palette,
            unhandled_input=self._handle_input,
        )

        # SIGWINCH fires on both terminal resize AND when the terminal window
        # is moved between monitors.  We use it to mark the art dirty so the
        # next 0.5s tick re-renders without issuing a "remove" first (which
        # would cause a visible blank frame).
        #
        # Installing it here is not enough and never was (audit L1): urwid's
        # `Screen.start()` installs its own handler when `loop.run()` is called,
        # replacing this one.  `_install_sigwinch_chain()` puts it back from
        # inside the running loop; this call covers the fallback text mode,
        # which has no urwid to overwrite it.
        signal.signal(signal.SIGWINCH, self._on_sigwinch)

        # A self-pipe the loop watches, so a signal arriving in the middle of
        # `loop.run()` can end it (audit H3).  `os.write` is async-signal-safe;
        # raising ExitMainLoop from a signal handler is not, because urwid may
        # be anywhere inside its own event processing when it fires.
        self._exit_pipe = self.loop.watch_pipe(self._on_exit_requested)

        self.loop.set_alarm_in(0.5, self._periodic_update)

    def request_exit(self):
        """
        Ask the TUI to shut down.  Safe to call from a signal handler.

        Without this the SIGTERM path set a flag the urwid loop never read, so
        the UI stayed on screen with no MPD polling behind it and `_shutdown()`
        — and therefore every save, and the MPD mode restore — was unreachable.
        """
        self.running = False
        if self._exit_pipe is not None:
            try:
                os.write(self._exit_pipe, b"x")
            except OSError:
                pass

    def _on_exit_requested(self, _data):
        """Runs inside the main loop, so raising ExitMainLoop here is safe."""
        raise urwid.ExitMainLoop()

    # ── Input handling ────────────────────────────────────────────────────────

    def _handle_input(self, key):
        # Don't lowercase arrow keys or special keys
        key_lower = key.lower() if isinstance(key, str) else key

        if key_lower == "q":
            self._quit()
        elif key_lower == " ":
            self._toggle_play_pause()
        elif key_lower == "n":
            self._skip_track()
        elif key_lower == "v":
            self._pass_track()
        elif key_lower == "l":
            self._like_track()
        elif key_lower == "i":
            self._show_model_info()
        elif key == "right":
            self._seek_forward()
        elif key == "left":
            self._seek_backward()
        elif key == "up":
            self._history_scroll(-1)
        elif key == "down":
            self._history_scroll(+1)
        elif key == "enter":
            self._replay_focused()
        elif key_lower in (",", "<"):
            self._volume_down()
        elif key_lower in (".", ">"):
            self._volume_up()

    # ── Actions ──────────────────────────────────────────────────────────────

    def _toggle_play_pause(self):
        s = self.dj.mpd_controller.get_status()
        if s["state"] == "playing":
            self.dj.mpd_controller.pause()
        else:
            self.dj.mpd_controller.play()

    def _skip_track(self):
        """
        [N].  The orchestrator owns the whole skip — feedback, lookahead
        replacement and the single advance — because the *order* of those three
        is what keeps the session alive (audit C4), and that ordering should not
        be something a key handler can get wrong.
        """
        self.dj.skip_current_track()

    def _pass_track(self):
        """
        [V].  Neutral skip: advance to the queued track without moving any vector.

        The counterpart to `[N]`: where a skip is a rejection that repels the
        session vector and escalates, a pass means "not this song, keep the
        vibe" and touches none of the model.  The orchestrator owns the advance —
        add-before-advance and the single `next`, the same C4 ordering the real
        skip rests on — and returns the passed track so the history can mark it
        `»`.  The mark is set here rather than drained from a feedback event
        because a pass, by design, produces no feedback event.
        """
        passed = self.dj.neutral_skip_current_track()
        if passed:
            self.history.mark_passed(passed)
            if self.use_urwid:
                self._update_session_panel()

    def _like_track(self):
        """
        [L].  Like the playing track, or retract the like if it already has one.

        The model half is `FeedbackHandler.process_unlike()`, which deletes the
        like from the feedback history and replays the taste model over what is
        left rather than subtracting a magnitude — see there for why a negative
        update cannot undo a normalised EMA (audit L8).  The `♥` follows the
        model rather than leading it: it is only added or removed once the
        handler reports that the history actually changed, so a track with no
        embedding cannot end up wearing a heart that `[L]` is then unable to
        take off.
        """
        t = self.current_status.get("track_file")
        if not t:
            return
        if self.history.is_liked(t):
            if self.dj.feedback_handler.process_unlike(t) is not None:
                self.history.unlike(t)
        elif self.dj.feedback_handler.process_like(t):
            self.history.like(t)
        if self.use_urwid:
            self._update_session_panel()

    def _history_scroll(self, delta: int):
        """
        [↑] / [↓].  Move the cursor through the session history, newest at top.

        The indices are the history model's own, derived from what this session
        played.  They deliberately share no numbering with MPD: the bindings this
        replaces indexed into `mpc playlist`, which under the old modes held the
        session's history above the current track, so ENTER on the first row
        replayed the first track of the evening (audit H2).
        """
        if self.history.move_focus(delta) and self.use_urwid:
            self._update_session_panel()

    def _replay_focused(self):
        """
        [ENTER].  Put the track under the cursor in the lookahead slot.

        The queue operation itself is `QueueManager.requeue_next()` — the
        ordering of an add against a delete is the thing C4 and M1 are about, and
        a display layer that reimplemented it would be a second place for it to
        drift.
        """
        track = self.history.focused_track()
        if not track:
            return
        if self.dj.queue_manager.requeue_next(track):
            label = self._label_for(track)
            print(f"Replay: {label} queued next", file=sys.stderr)
            if self.use_urwid:
                self._update_session_panel()

    def _volume_up(self):
        self.dj.mpd_controller.volume_up(5)

    def _volume_down(self):
        self.dj.mpd_controller.volume_down(5)

    def _seek_forward(self, delta: int = 10):
        self.dj.mpd_controller.seek_relative(+delta)

    def _seek_backward(self, delta: int = 10):
        self.dj.mpd_controller.seek_relative(-delta)

    # NOTE: _queue_navigate() and _queue_play_selected() are gone with the queue
    # panel (audit D1/H2).  They indexed into `mpc playlist`, which with consume
    # off is the session's *history*, so ENTER on "1." replayed the first track
    # of the evening.  Stage 3 rebinds ↑↓ and ENTER to the session-history panel
    # — deriving the indices from scratch rather than porting these.

    def _quit(self):
        self.running = False
        # Only hand SIGWINCH back if we took it.  This used to drop to SIG_DFL
        # unconditionally, which takes the handler away from urwid while urwid
        # is still holding the screen — and its handler is how it learns about a
        # resize during teardown.  `Screen.stop()` restores what it captured, so
        # the right move is usually to leave it alone (audit L1).
        if self._sigwinch_installed:
            signal.signal(signal.SIGWINCH,
                          self._urwid_sigwinch or signal.SIG_DFL)
            self._sigwinch_installed = False
        if self.use_urwid:
            raise urwid.ExitMainLoop()

    def _model_info_lines(self) -> List[str]:
        """
        What the machine currently believes, in measured quantities only.

        This replaced the time-context overlay, which had no content left after
        that subsystem was removed (audit D6/L9).  Stage 3 (H1d) adds the
        descriptor rows: the top words for the session vector *and* the taste
        vector, and the drift figure the vibe line summarises.  Everything here
        is derived — there is no vocabulary in this overlay that the system
        cannot back with a number it computed.
        """
        session = self.dj.session_state.get_stats()
        taste = self.dj.user_taste.get_stats()
        exploration = self.dj.exploration_controller.get_stats()
        selector = self.dj.track_selector.get_stats()
        weights = self.dj.exploration_controller.get_weights(
            taste_updates=taste['positive_updates'])

        # τ is the effective number of candidates the sampler is choosing among.
        # Reporting it is the point of picking a rank-based rule: it is a true
        # statement about what the machine is doing, derived rather than named.
        tau = self.dj.track_selector.temperature(exploration['exploration'])
        ramp = self.dj.exploration_controller.taste_ramp(taste['positive_updates'])

        schedule = config.skip_turnover_schedule
        run = exploration['consecutive_skips']
        next_target = schedule[min(run + 1, len(schedule)) - 1]

        # Descriptor rows (audit H1d).  Both are gated on their vector carrying
        # evidence: z-scoring a zero vector yields a confident-looking ranking of
        # the bank's own baselines rather than an error, so an ungated readout
        # would state a vibe about nothing (H1, and the module note in
        # vibe_readout.py).
        seeded = session['is_seeded']
        session_vector = self.dj.session_state.get_session_vector()
        taste_vector = self.dj.user_taste.get_taste_vector()
        drift = self.vibe.drift(session_vector, seeded)

        if self.vibe.bank is None:
            descriptor_lines = ["  (no descriptor bank loaded — see the startup log)"]
        else:
            session_words = (format_descriptors(
                self.vibe.descriptors(session_vector, seeded))
                if seeded else "— nothing has played yet")
            taste_words = (format_descriptors(
                self.vibe.bank.top(taste_vector, self.vibe.top_n))
                if taste['is_seeded'] else "— no positive signal yet")
            descriptor_lines = [
                f"  bank             {len(self.vibe.bank)} descriptors, "
                f"z-scored against this library",
                f"  session          {session_words}",
                f"  taste            {taste_words}",
            ]
            if drift is None:
                descriptor_lines.append(
                    "  drift            — (fewer than two tracks recorded)")
            else:
                descriptor_lines.append(
                    f"  drift            {drift['held']} of {drift['of']} words held "
                    f"over {drift['distance']} tracks")
                # The cosine H1 originally specified.  It is kept here rather
                # than on the vibe line because it is a compressed scale: over 40
                # real sessions its median is 0.988 and its p10 is 0.947, so it
                # reads as "0.99" almost always (§10d).  The word count above is
                # what occupies its range; this is the underlying measurement.
                descriptor_lines.append(
                    f"  drift cosine     {drift['cosine']:+.3f}   "
                    f"(ordinary listening: median 0.99, p10 0.95)")

        lines = [
            "MODEL STATE",
            "",
            f"Library             {self.dj.track_library.get_track_count()} tracks",
            f"Session started     {'yes' if session['session_started'] else 'no'}",
            f"Session vector      "
            + ("seeded" if seeded else "unseeded (nothing has played yet)"),
            f"Tracks this session {session['tracks_played']}",
            f"Unique tracks seen  {selector['unique_tracks_played']}",
            "",
            "─" * 50,
            "DESCRIPTORS",
            *descriptor_lines,
            "",
            "─" * 50,
            "SELECTION",
            f"  sampling         rank-Boltzmann, choosing from ~top {tau:.0f}",
            f"  τ                {tau:.1f}   (bounds {config.tau_min:.0f}–{config.tau_max:.0f})",
            f"  last pick        "
            + (f"rank {selector['last_rank']} of {selector['last_pool_size']} scored"
               if selector['last_rank'] is not None else "— (uniform draw: no evidence yet)"),
            "",
            "─" * 50,
            "TASTE MODEL",
            f"  seeded           {'yes' if taste['is_seeded'] else 'no (no positive signal yet)'}",
            f"  updates          {taste['total_updates']} "
            f"({taste['positive_updates']} positive — likes + full listens)",
            f"  β earned         {ramp:.0%}   (full weight after "
            f"{config.taste_ramp_updates} positive signals; skips do not count)",
            f"  likes            {taste['like_count']}",
            f"  full listens     {taste['full_listen_count']}",
            f"  skips            {taste['skip_count']}",
            "",
            "─" * 50,
            "EXPLORATION",
            f"  value            {exploration['exploration']:.2f}"
            f"   (bounds {config.exploration_min:.2f}–{config.exploration_max:.2f})",
            f"  consecutive skips   {run}",
            f"  consecutive listens {exploration['consecutive_listens']}",
            f"  next skip targets   {next_target:.0%} of the candidate pool",
            "",
            "─" * 50,
            "CURRENT SCORING WEIGHTS",
            f"  session          {weights['session_weight']:.3f}",
            f"  taste            {weights['taste_weight']:.3f}",
            f"  novelty          {weights['novelty_weight']:.3f}",
            f"  anti-repetition  {weights['anti_repetition_weight']:.3f}",
            "",
            "[↑↓] scroll   any other key closes",
        ]
        return lines

    # Keys that dismiss the inspector rather than scrolling it.
    _INFO_SCROLL_KEYS = ("up", "down", "page up", "page down", "home", "end")

    # A LineBox draws one row above and below and one column each side.  That is
    # structural to urwid's own widget, not a count of this application's
    # layout, which is the distinction H8 is about.
    _LINEBOX_BORDER = 2

    def _overlay_inner_size(self, overlay):
        """
        The size urwid will hand the inspector's ListBox — asked, not derived.

        This used to be `(int(cols * 0.7) - 2, box_rows - 2)`: our own copy of
        the arithmetic urwid performs on the `("relative", 70)` width the
        overlay declares.  Two places computing the same layout is how H8's
        album-art constants came to be wrong, and being wrong here costs a
        scroll page that jumps by the wrong amount.

        `calculate_padding_filler` and `top_w_size` are the Overlay's own
        methods, so this asks the widget where it is putting its child.  Read
        per keypress rather than once, so the page still scrolls correctly if
        the terminal is resized while `[I]` is held open — which it can be, now
        that ↑↓ no longer dismiss it.
        """
        size = self.loop.screen.get_cols_rows()
        left, right, top, bottom = overlay.calculate_padding_filler(size, True)
        inner = overlay.top_w_size(size, left, right, top, bottom)
        cols = inner[0] if len(inner) > 0 else size[0]
        rows = inner[1] if len(inner) > 1 else size[1]
        return (max(1, cols - self._LINEBOX_BORDER),
                max(1, rows - self._LINEBOX_BORDER))

    def _show_model_info(self):
        """
        [I].  Draw the inspector and block on it until a key dismisses it.

        Stage 2 sized this box to its contents after the sampling and skip rows
        overflowed a fixed 70% height and the last third was silently cut off.
        The descriptor rows push it past a 40-row terminal, so content-sizing is
        no longer enough: it is a ListBox now, and ↑↓ scroll it.  An inspector
        that hides part of what it is inspecting is worse than no inspector.
        """
        lines = self._model_info_lines()
        if not self.use_urwid:
            # The fallback mode had no `[I]` binding at all while this method
            # already returned its lines for a non-urwid caller — a loose end
            # rather than an intended asymmetry (§8, trap 4).  Print the page
            # and hold it until a key, which is the same gesture the overlay
            # makes in a terminal that has no loop to block.
            print("\033[2J\033[H", end="")
            for line in lines:
                print(line)
            print("\n── any key to return ──")
            sys.stdout.flush()
            self._wait_for_any_key()
            return lines

        walker = urwid.SimpleListWalker([urwid.Text(line) for line in lines])
        listbox = urwid.ListBox(walker)
        overlay_box = urwid.LineBox(listbox, title="Model Info")

        _, screen_rows = self.loop.screen.get_cols_rows()
        box_rows = min(len(lines) + self._LINEBOX_BORDER,
                       max(6, screen_rows - 2))

        overlay = urwid.Overlay(
            overlay_box,
            self.frame,
            align="center",
            width=("relative", 70),
            valign="middle",
            height=box_rows,
        )
        orig = self.loop.widget
        self.loop.widget = overlay
        # The album art is a separate X11/Wayland surface, not part of urwid's
        # canvas — so the overlay box draws *over* the TUI but leaves the cover
        # sitting on top of the inspector (inspection F3).  Take it down while `[I]`
        # is open (nothing in the loop below re-sends it), and mark it dirty on
        # the way out so the next tick repaints it in place.
        if self.show_album_art:
            self.album_art_renderer.clear()
        # Wake even with no keypress.  This loop owns the thread that runs the
        # session bookkeeping, and the overlay can stay open indefinitely now
        # that ↑↓ no longer dismiss it — without a timeout, a track that starts
        # *and* finishes while `[I]` is up never reaches the history panel and
        # its ✓ is dropped on the floor.  Stage 2's version closed on any key,
        # so it could not be held open long enough for this to bite.
        try:
            self.loop.screen.set_input_timeouts(max_wait=0.5)
        except Exception:
            pass
        try:
            while True:
                self.loop.draw_screen()
                keys = self.loop.screen.get_input()
                if not keys:
                    self._sync_session_state()
                    continue
                dismissed = False
                for key in keys:
                    if key in self._INFO_SCROLL_KEYS:
                        listbox.keypress(self._overlay_inner_size(overlay), key)
                    elif isinstance(key, tuple):
                        continue        # a mouse event; not a dismissal
                    else:
                        dismissed = True
                if dismissed:
                    break
        finally:
            try:
                self.loop.screen.set_input_timeouts(max_wait=None)
            except Exception:
                pass
            self.loop.widget = orig
            # Force the cover back on the next `_render_art`: `clear()` above
            # already dropped the render key, but a window resize while `[I]`
            # was open could otherwise leave it stale.
            if self.show_album_art:
                self.album_art_renderer.force_redraw()
        return lines

    # ── Periodic update ───────────────────────────────────────────────────────

    def _on_sigwinch(self, signum, frame):
        """
        Called when the terminal is resized or moved to another monitor.
        Marks the album art dirty so _render_art re-sends on the next tick.
        We do NOT call render() here directly — signal handlers must be fast
        and must not write to the ueberzug pipe (not async-signal-safe).

        Then hands the signal on to whatever urwid installed, because urwid's
        own handler is how the screen learns its size changed.  Displacing it
        outright would fix the album art by breaking the layout.
        """
        if self.show_album_art:
            self.album_art_renderer.force_redraw()

        chained = self._urwid_sigwinch
        if callable(chained):
            chained(signum, frame)

    def _install_sigwinch_chain(self):
        """
        Make sure `_on_sigwinch` is actually reached once the loop is running (L1).

        L1 says urwid replaces our handler at startup, and the measurement it
        cites is real:

            after _setup_urwid()    <bound method AdaptiveDJTUI._on_sigwinch …>
            after Screen.start()    <bound method Screen._sigwinch_handler …>

        **But the conclusion drawn from it does not hold on urwid 3.0.5.**
        `Screen.start()` *captures* what was installed and calls it — see
        `_posix_raw_display.py`, where line 129 stores `_prev_sigwinch_handler`
        and line 98 invokes it — and `Screen.stop()` puts it back.  urwid wraps
        our handler rather than displacing it, so it was being invoked all
        along.  `getsignal` shows urwid's handler because urwid's is on the
        outside, which looks identical to being replaced and is not.

        Installing on top of that is not a fix, it is a cycle: ours → urwid's →
        ours (as urwid's `_prev`) → urwid's → … Driving it produced exactly
        that, a `RecursionError` on the first real resize.

        So this asks the screen whether we are already in its chain, and only
        installs when we are not — which is the case on urwid versions without
        `_prev_sigwinch_handler`, and the case the finding describes.  Reading
        the screen object rather than the version number keeps it a property of
        what is actually there.

        **What the handler is for is a window move, not a resize.**  A resize
        already self-heals within 0.5 s: `_art_geometry()` is re-derived from
        `get_cols_rows()` on every tick, so the renderer's key changes and the
        image is re-sent on its own.  A move to another monitor at the same size
        produces the same key, the send is skipped, and the image is left
        behind — which is what `force_redraw()` exists for.

        Idempotent, so the tick can call it freely.
        """
        screen = getattr(self.loop, 'screen', None)
        if getattr(screen, '_prev_sigwinch_handler', None) == self._on_sigwinch:
            self._urwid_sigwinch = None
            return False        # urwid already calls us; do not close the loop

        current = signal.getsignal(signal.SIGWINCH)
        if current == self._on_sigwinch:
            return False

        # Anything that is not a real callable — SIG_DFL, SIG_IGN, None — is not
        # something to chain into.
        self._urwid_sigwinch = current if callable(current) else None
        signal.signal(signal.SIGWINCH, self._on_sigwinch)
        self._sigwinch_installed = True
        return True

    def _periodic_update(self, loop=None, user_data=None):
        # The first tick is the earliest point at which urwid's own handler is
        # in place to be chained: it is installed by `loop.run()`, which has not
        # happened yet when `_setup_urwid` runs (audit L1).
        self._install_sigwinch_chain()
        self._update_display()
        if self.running:
            self.loop.set_alarm_in(0.5, self._periodic_update)

    # ── Display update ────────────────────────────────────────────────────────

    def _sync_session_state(self) -> dict:
        """
        Observe what the player has done since the last look, and return MPD's
        status so the caller does not poll it twice.

        Separate from drawing because it must keep running while the `[I]`
        overlay has the loop blocked.  It is the only place the history learns
        that a track started, so a gap here is a track missing from the panel —
        and the panel's whole claim is that it lists what actually played.

        Order matters: a track must be in the history before an event can mark
        it, and the vibe snapshot must come from the vector as it stands now.
        """
        status = self.dj.mpd_controller.get_status()
        self.current_status = status

        self.history.note_playing(status.get("track_file"))
        self._drain_feedback_events()
        stats = self.dj.session_state.get_stats()
        self.vibe.observe(self.dj.session_state.get_session_vector(),
                          stats['tracks_played'], stats['is_seeded'])
        return status

    def _update_display(self):
        status = self._sync_session_state()
        session_stats = self.dj.session_state.get_stats()

        if not self.use_urwid:
            self._update_simple_display(status)
            return

        # Status / volume
        volume = status.get("volume", 100)
        state = status.get("state", "stopped")
        if state == "playing":
            self.status_text.set_text(("playing", f"▶  Playing   Vol: {volume}%"))
        elif state == "paused":
            self.status_text.set_text(("paused", f"⏸  Paused    Vol: {volume}%"))
        else:
            self.status_text.set_text(f"⏹  Stopped   Vol: {volume}%")

        # Track metadata
        artist = status.get("artist", "Unknown Artist")
        album = status.get("album", "Unknown Album")
        title = status.get("title", "Unknown Title")
        track_file = status.get("track_file")

        self.artist_text.set_text(f"Artist:  {artist}")
        self.album_text.set_text(f"Album:   {album}")

        if self.history.is_liked(track_file):
            self.track_text.set_text(["Track:   ", ("liked", "❤ "), title])
        else:
            self.track_text.set_text(f"Track:   {title}")

        # Seek bar
        position = status.get("position", 0)
        duration = status.get("duration", 0)
        if duration > 0:
            self.seek_bar_progress.set_completion(int((position / duration) * 100))
            self.seek_time_text.set_text(
                f"{self._fmt(position)} / {self._fmt(duration)}"
            )
        else:
            self.seek_bar_progress.set_completion(0)
            self.seek_time_text.set_text("0:00 / 0:00")

        # Vibe readout: the descriptors, then the drift and the counter.
        descriptor_line, drift_line = self.vibe.lines(
            self.dj.session_state.get_session_vector(),
            session_stats['is_seeded'],
            session_stats['tracks_played'],
        )
        self.descriptor_text.set_text(descriptor_line)
        self.drift_text.set_text(drift_line)

        # Console
        self._update_console()

        # Session panel
        self._update_session_panel()

        # Album art
        if self.show_album_art and track_file:
            art_path = self.album_art_renderer.find_album_art(track_file)
            if art_path:
                # Blank the placeholder so no text bleeds through under the image
                self.album_art_placeholder.set_text(" ")
                self._render_art(art_path)
            else:
                self.album_art_placeholder.set_text("🖼  No Cover")
                self.album_art_renderer.clear()
        elif self.show_album_art:
            self.album_art_placeholder.set_text("🖼  Album Art")

    # ── Album art positioning ─────────────────────────────────────────────────

    def _art_geometry(self, cols: int, rows: int):
        """
        Where the album art goes, asked of the widget tree (audit H8/L3).

        Returns `(x, y, width, height)` in 0-indexed terminal cells, or None when
        there is no usable art area.

        This replaces four hand-counted constants — `x=2`, `y=3`, `width=33`,
        `height=RIGHT_COL_ROWS=10` — whose comment block documented the
        arithmetic carefully while nothing enforced it.  Two of them were already
        wrong: the art column starts at column **1** (the LineBox's left border
        is column 0, not column 1), and the art area's true height is whatever
        the right-hand Pile happens to be, which Stage 3 changes from 10 rows to
        11.  Every number below is read from the widgets that produce it, so a
        row added to the Now Playing panel moves the image without an edit here.

        The walk, outside in:

            frame        → header.rows() gives the body's first row
            main_pile    → get_item_rows() gives the Now Playing box's height
                           and, by summing the items above it, its first row
            LineBox      → one border row and one border column
            content Pile → the rows of every widget above np_columns
            np_columns   → column_widths() gives the art cell's width, and the
                           widths of the cells before it give its x-offset
        """
        if cols <= 0 or rows <= 0:
            return None

        header_rows = self.frame.header.rows((cols,)) if self.frame.header else 0
        footer_rows = self.frame.footer.rows((cols,)) if self.frame.footer else 0
        body_rows = rows - header_rows - footer_rows
        if body_rows <= 0:
            return None

        # Row and column of the Now Playing box's top-left corner.
        item_rows = self.main_pile.get_item_rows((cols, body_rows), False)
        y = header_rows
        for (widget, _options), height in zip(self.main_pile.contents, item_rows):
            if widget is self.now_playing_box:
                np_rows = height
                break
            y += height
        else:
            return None
        x = 0

        # Inside the LineBox: one border row, one border column.
        y += 1
        x += 1
        inner_cols = cols - 2
        if inner_cols <= 0:
            return None

        # Inside the content Pile: everything stacked above np_columns.
        for widget, _options in self.now_playing_content.contents:
            if widget is self.np_columns:
                break
            y += widget.rows((inner_cols,))
        else:
            return None

        # Inside the Columns: the cells to the left of the art cell.
        widths = self.np_columns.column_widths((inner_cols,))
        if self.art_column_index >= len(widths):
            return None
        x += sum(widths[:self.art_column_index])
        width = widths[self.art_column_index]

        # The art cell's height is the Columns' height, which is the tallest of
        # its cells — in practice the right-hand Pile.  Clip it to what is
        # actually on screen: the Now Playing box is packed, so it can still be
        # cut off from below by a very short terminal.
        height = self.np_columns.rows((inner_cols,))
        visible = min(height, np_rows - 2, rows - footer_rows - y - 1)

        if width < MIN_ART_COLS or visible < MIN_ART_ROWS:
            return None
        return x, y, width, visible

    def _render_art(self, art_path):
        """
        Position the art from the live screen size and re-render.

        We always re-send the render command (no skip-if-same guard here) so the
        image reappears after window moves and monitor changes.
        `AlbumArtRenderer.render()` is itself the skip guard for the
        same-image/same-position case, to avoid redundant sends.
        """
        cols, rows = self.loop.screen.get_cols_rows()
        geometry = self._art_geometry(cols, rows)
        if geometry is None:
            # Too short or too narrow to draw into — an image squeezed into two
            # rows is a smear over the seek bar, not album art.
            self.album_art_renderer.clear()
            return

        x_art, y_art, art_w, art_h = geometry
        self.album_art_renderer.render(
            art_path, x=x_art, y=y_art, width=art_w, height=art_h
        )

    # ── Console update ────────────────────────────────────────────────────────

    def _update_console(self):
        """Refresh the console widget from the captured stderr lines."""
        if _console_capture is None:
            return
        lines = _console_capture.get_lines()
        if not lines:
            return

        self.console_walker.clear()
        # Show only the last CONSOLE_ROWS lines so the widget fills neatly
        for line in lines[-CONSOLE_ROWS:]:
            # Colour-code by content
            if any(w in line for w in ("error", "Error", "ERROR", "failed", "Failed")):
                attr = "console_err"
            elif any(w in line for w in ("warn", "Warn", "WARN", "shifted", "Vibe")):
                attr = "console_warn"
            else:
                attr = "console_text"
            self.console_walker.append(urwid.AttrMap(urwid.Text(line), attr))

        # Scroll to bottom
        try:
            self.console_walker.set_focus(len(self.console_walker) - 1)
        except Exception:
            pass

    # ── Session panel ─────────────────────────────────────────────────────────

    def _drain_feedback_events(self):
        """
        Pull whatever `FeedbackHandler` has recorded since the last refresh and
        mark the history with it.

        This is how a skip becomes `⏭` and a completed track becomes `✓` without
        the display having to observe either event directly — full listens fire
        on the background thread, and there is no callback into the UI.  The
        cursor is what keeps a file spanning every previous session from
        back-filling tonight's panel.
        """
        events = getattr(self.dj.feedback_handler, 'feedback_history', None)
        if events is None:
            return
        if self._feedback_cursor > len(events):
            # The handler truncated its history (it caps at 1000 events).
            self._feedback_cursor = 0
        self._feedback_cursor = self.history.drain_events(
            events, self._feedback_cursor)

    def _update_session_panel(self):
        """
        The lookahead on top, then what actually played, newest first.

        This is what the queue panel was standing in for.  At depth 1 there is
        exactly one upcoming track and nothing to navigate, so the panel's real
        subject is the past — which has the advantage of being true (audit H1c).
        """
        next_track = self._next_track_label()
        self.next_up_text.set_text(
            f"  ↓ next:  {next_track}" if next_track else "  ↓ next:  —")

        current = self.current_status.get("track_file")
        rows = self.history.rows(current_track=current,
                                 labeller=self._label_for,
                                 limit=HISTORY_RENDER_LIMIT)

        self.history_walker.clear()
        if not rows:
            self.history_walker.append(
                urwid.AttrMap(urwid.Text("  — nothing has played yet —"),
                              "history_item"))
            return

        for row in rows:
            text = urwid.Text(f" {row['marks']} {row['label']}")
            if row['focused']:
                attr = "history_focus"
            elif row['playing']:
                attr = "history_playing"
            else:
                attr = "history_item"
            self.history_walker.append(urwid.AttrMap(text, attr))

        # Keep the cursor on screen as the list grows past the panel's height.
        try:
            self.history_walker.set_focus(
                min(self.history.focus, len(self.history_walker) - 1))
        except Exception:
            pass

    def _label_for(self, track: str) -> str:
        """
        `Artist – Title` for one track key.

        `fetch_track_tags` rather than the deleted `get_playlist_metadata`: with
        `consume on` a played track is gone from `mpc playlist`, so the map that
        method built covered exactly the tracks this panel does not show (audit
        §8, Stage 3 trap 2).
        """
        meta = self.dj.mpd_controller.fetch_track_tags(track)
        artist = meta.get("artist") or "Unknown Artist"
        title = meta.get("title") or Path(track).stem
        return f"{artist} – {title}"

    def _next_track_label(self) -> Optional[str]:
        """`artist – title` for the lookahead, or None if there is none."""
        current = self.current_status.get("track_file")
        track = self.dj.queue_manager.get_next_track(current_track=current)
        if not track:
            return None
        label = self._label_for(track)
        return f"❤ {label}" if self.history.is_liked(track) else label

    # ── Simple (non-urwid) display ────────────────────────────────────────────

    def _update_simple_display(self, status):
        print("\033[2J\033[H", end="")
        print("=" * 60)
        print("Adaptive Session AI DJ  (simple mode)")
        print("=" * 60)

        icon = "▶" if status["state"] == "playing" else "⏸"
        print(f"\n{icon} {status['state'].upper()}")
        print(f"\nArtist: {status.get('artist', 'Unknown')}")
        print(f"Album:  {status.get('album', 'Unknown')}")

        tf = status.get("track_file")
        t = status.get("title", "Unknown")
        if self.history.is_liked(tf):
            t = f"❤ {t}"
        print(f"Track:  {t}")

        pos = status.get("position", 0)
        dur = status.get("duration", 0)
        if dur > 0:
            bar = "█" * int(pos / dur * 40) + "░" * (40 - int(pos / dur * 40))
            print(f"\n[{bar}]")
            print(f"{self._fmt(pos)} / {self._fmt(dur)}")

        stats = self.dj.session_state.get_stats()
        vector = self.dj.session_state.get_session_vector()
        for line in self.vibe.lines(vector, stats['is_seeded'],
                                    stats['tracks_played']):
            print(line)

        print("\n" + "─" * 60)
        next_track = self._next_track_label()
        print(f"↓ next:  {next_track}" if next_track else "↓ next:  —")

        current = status.get("track_file")
        for row in self.history.rows(current_track=current,
                                     labeller=self._label_for, limit=8):
            cursor = "▸" if row['focused'] else " "
            print(f"{cursor} {row['marks']} {row['label']}")

        print("\n" + "=" * 60)
        # The same bindings as the urwid mode — and now literally the same
        # dispatch, via `decode_key()` into `_handle_input` (audit L9/M1c).
        # These used to disagree: ↑↓ were volume here and queue navigation
        # there, while the README said a third thing.
        print("SPACE=Play/Pause  N=Next  V=Pass  L=Like  ↑↓=History  ENTER=Replay")
        print(",.=Vol  ←→=Seek  I=Info  Q=Quit")
        print("=" * 60)
        sys.stdout.flush()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _fmt(self, seconds: int) -> str:
        m = seconds // 60
        s = seconds % 60
        return f"{m}:{s:02d}"

    # NOTE: `_session_line()` is gone.  It reported the track count because that
    # was the only honest thing the session vector could say about itself with
    # the descriptor bank unwired — the mood, momentum and stage words it
    # replaced were all invented against thresholds the data never occupied
    # (audit D4).  The bank is wired now; see `vibe_readout.py`.

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self):
        self.running = True
        # This session's feedback events start here.  Taken at run() rather than
        # at construction because `persistence.load_all()` and `start_session()`
        # both happen in between, and the file it loads spans every previous
        # session.
        events = getattr(self.dj.feedback_handler, 'feedback_history', None)
        if events is not None:
            self._feedback_cursor = len(events)
        if not self.use_urwid:
            self._run_simple_mode()
        else:
            # Suppress raw-terminal passthrough while urwid owns the screen.
            # Without this, every stderr write (background thread, queue
            # manager, etc.) bleeds through the TUI layout as raw text.
            if _console_capture is not None:
                _console_capture.tui_active = True
            try:
                self.loop.run()
            except KeyboardInterrupt:
                pass
            finally:
                self.running = False
                if _console_capture is not None:
                    _console_capture.tui_active = False
                if self.show_album_art:
                    self.album_art_renderer.clear()

    def _run_simple_mode(self):
        """
        The no-urwid fallback.  Reads raw stdin and feeds `_handle_input`.

        It used to carry its own `if key == …` ladder — a second binding table
        beside the urwid one, which is how ↑↓ ended up meaning *volume* here and
        *queue navigation* there while the README claimed a third thing (audit
        L9).  Stage 3 reconciled the three lists by reading them; this makes
        there be one list.  `decode_key()` turns terminal bytes into the names
        urwid uses, and the dispatch after that is the same method the urwid
        mode calls, so a binding cannot exist in one interface and not the
        other.
        """
        import select, termios, tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        pending = ""
        try:
            tty.setcbreak(fd)
            while self.running:
                self._update_display()
                if not select.select([sys.stdin], [], [], 0.5)[0]:
                    # A partial escape sequence that no further input completed
                    # was never a key.  Dropping it here stops it from swallowing
                    # the front of whatever is typed next.
                    pending = ""
                    continue

                # `os.read` on the raw descriptor rather than `sys.stdin.read`,
                # so what `select` reports and what is consumed are the same
                # thing — see `decode_keys`.
                chunk = os.read(fd, 64).decode("utf-8", "ignore")
                if not chunk:
                    break                       # EOF: the terminal went away

                keys, pending = decode_keys(pending + chunk)
                for key in keys:
                    self._handle_input(key)
                    if not self.running:
                        break
        except KeyboardInterrupt:
            pass
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            self.running = False
            print("\n")

    def _wait_for_any_key(self):
        """
        Hold the fallback mode's `[I]` page until a key dismisses it.

        The urwid inspector holds its loop the same way — but it wakes every
        0.5 s without a keypress, and this must too, for the same class of
        reason.  A single unbounded `select` cannot be interrupted by anything:
        `request_exit()` sets `running` false from the signal handler and the
        page would sit there regardless, which is H3's shape in the one
        interface H3 was never driven through.  So it polls, and a shutdown
        gets out of here.

        Reads the raw descriptor rather than `sys.stdin`, so `select` and the
        read agree about what is pending — see `decode_keys`.  Returns at once
        when stdin is not a terminal, so a non-interactive caller cannot be
        wedged here at all.
        """
        import select

        try:
            if not sys.stdin.isatty():
                return
            fd = sys.stdin.fileno()
            while self.running:
                if select.select([sys.stdin], [], [], 0.25)[0]:
                    os.read(fd, 64)
                    return
        except Exception:
            pass
