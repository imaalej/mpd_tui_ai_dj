"""
Terminal User Interface - Adaptive Session AI DJ
Full-featured TUI with real-time updates and responsive controls.

Layout (rows, top to bottom):
  [1]  Header bar
  [N]  Now Playing  ── two columns:
         left:  album art box  (fixed ART_COLS wide, fills available height)
         right: status · track info · seek bar · vibe
  [5]  Console      ── live state updates (exploration, vibe shifts, etc.)
  [M]  Queue        ── upcoming tracks (takes remaining space)
  [1]  Footer bar

Album art is rendered as a terminal overlay (ueberzug/kitty/sixel).
Its position is computed from the live screen size so it never overlaps
the seek bar or vibe line regardless of terminal dimensions.
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
# Now Playing box.  Includes the inner LineBox borders (2 cols each side = 2).
ART_COLS = 33  # terminal columns wide for the album art area (no inner box)

# Header height in rows (urwid Frame header is exactly 1 row tall when the
# header widget is a simple Filler(Text(…))).
HEADER_ROWS = 1
FOOTER_ROWS = 1

# Rows used by the Now Playing LineBox itself (top border + bottom border)
NP_BORDER_ROWS = 2

# Fixed row count for the console panel
CONSOLE_ROWS = 5

# Exact row count of the right column Pile in Now Playing.
# Count: status(1) + Divider(1) + artist(1) + album(1) + track(1) +
#        Divider(1) + seek_bar(1) + seek_time(1) + Divider(1) + vibe(1) = 10
# The art height is pinned to this so a wide cover can never overflow into
# the seek bar or vibe line regardless of the image aspect ratio.
RIGHT_COL_ROWS = 10


class AdaptiveDJTUI:
    """Terminal User Interface for Adaptive Session AI DJ."""

    def __init__(self, dj):
        self.dj = dj
        self.running = False

        # UI state
        self.current_status: dict = {}
        self.liked_tracks: set = set()
        # Written to by the signal handler to unblock urwid's MainLoop; see
        # `request_exit`.
        self._exit_pipe: Optional[int] = None

        # Album art
        self.album_art_renderer = get_album_art_renderer()
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
            ("queue_item", "light gray", "default"),
            ("queue_current", "black,bold", "light green"),
            ("liked", "light red,bold", "default"),
            ("seek_bar", "white", "dark gray"),
            ("seek_progress", "black", "light green"),
            ("console_text", "dark cyan", "default"),
            ("console_warn", "yellow", "default"),
            ("console_err", "light red", "default"),
            ("queue_focused", "black,bold", "dark cyan"),
        ]

        # ── Header ──
        self.header_text = urwid.Text("🎵 Adaptive Session AI DJ", align="center")
        self.header = urwid.AttrMap(urwid.Filler(self.header_text), "header")

        # ── Now Playing: right column widgets ──
        self.status_text = urwid.Text(("paused", "⏸ Paused"))
        self.artist_text = urwid.Text("Artist: ---")
        self.album_text = urwid.Text("Album: ---")
        self.track_text = urwid.Text("Track: ---")
        # The vibe line carries the track count and nothing else until the CLAP
        # descriptor bank lands (audit H1/D5, Stage 1 data + Stage 3 display).
        # The mood/momentum/stage words that used to live here were invented
        # against thresholds the data never occupied — a blank line is honest,
        # a counter is a fact, and the old string was neither.
        self.vibe_text = urwid.Text(("vibe", "Session: —"))

        self.seek_bar_progress = urwid.ProgressBar(
            "seek_bar", "seek_progress", current=0, done=100
        )
        self.seek_time_text = urwid.Text("0:00 / 0:00", align="center")

        right_col = urwid.Pile(
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
                urwid.AttrMap(self.vibe_text, "vibe"),
            ]
        )

        # ── Now Playing: left column (album art area) ──
        # No LineBox here — a border widget renders visible box characters in
        # the terminal that show through behind the ueberzug overlay.
        # The art area is a plain Filler; ueberzug draws the image on top.
        self.album_art_placeholder = urwid.Text(" ", align="center")
        art_inner = urwid.Filler(self.album_art_placeholder, valign="middle")

        # Two-column row: art (fixed width) | track info (rest)
        np_columns = urwid.Columns(
            [
                ("fixed", ART_COLS, art_inner),
                ("weight", 1, urwid.Padding(right_col, left=1)),
            ]
        )

        now_playing_content = urwid.Pile(
            [
                urwid.Divider(),
                np_columns,
                urwid.Divider(),
            ]
        )

        self.now_playing_box = urwid.LineBox(now_playing_content, title="♪ Now Playing")

        # ── Console panel ──
        self.console_walker = urwid.SimpleFocusListWalker(
            [urwid.Text(("console_text", "── console ready ──"))]
        )
        console_lb = urwid.ListBox(self.console_walker)
        # BoxAdapter gives the ListBox a fixed height inside a Pile
        self.console_box = urwid.LineBox(
            urwid.BoxAdapter(console_lb, CONSOLE_ROWS), title="System Console"
        )

        # ── Up Next panel ──
        #
        # This was the "Upcoming Queue", listing ten tracks — which it drew from
        # `mpc playlist`, so with consume off it was actually showing the tracks
        # already *played*, numbered as if they were the future (audit H2).  At
        # depth 1 there is exactly one upcoming track and nothing to navigate, so
        # the list, the ↑↓ bindings and the ENTER-to-play action are all gone.
        #
        # Stage 3 (H1c) replaces this panel with the session history — what
        # actually happened, with ♥ / ⏭ / ✓ marks — which is the visibility the
        # queue panel was really standing in for.  The geometry is deliberately
        # left exactly as it was: the album art is still pinned to hand-counted
        # row constants (H8), and moving the layout before that is fixed would
        # misplace the image.
        self.queue_walker = urwid.SimpleFocusListWalker([])
        self.queue_listbox = urwid.ListBox(self.queue_walker)
        self.queue_box = urwid.LineBox(self.queue_listbox, title="Up Next")

        # ── Main layout ──
        # now_playing_box: weight 0 means it takes only its natural (pack) height.
        # We use ('given', N) via BoxAdapter trick — but the cleanest urwid way
        # is to give now_playing a fixed row count computed at render time.
        # Instead we use weight proportions and let urwid divide the space.
        # Proportions: now_playing=3, queue=2 gives now_playing ~60% of body.
        main_pile = urwid.Pile(
            [
                ("weight", 3, self.now_playing_box),
                (
                    CONSOLE_ROWS + 2,
                    self.console_box,
                ),  # fixed: CONSOLE_ROWS + 2 border rows
                ("weight", 2, self.queue_box),
            ]
        )

        # ── Footer ──
        # [V] is gone (audit D8/H9) and so are the queue-navigation keys, which
        # had nothing left to navigate.  A footer that advertises a key doing
        # nothing is the same dishonesty the rest of this rewrite removes.
        footer_text = urwid.Text(
            [
                "Toggle Pause - [SPACE]  ",
                "Skip Song - [N]  ",
                "Like Song - [L]  ",
                "Vol - [<,>]  ",
                "Seek - [←,→]  ",
                "Model Info - [I]  ",
                "Quit - [Q]",
            ],
            align="center",
        )
        self.footer = urwid.AttrMap(urwid.Filler(footer_text), "footer")

        # ── Frame ──
        self.frame = urwid.Frame(
            body=main_pile,
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
        elif key_lower == "l":
            self._like_track()
        elif key_lower == "i":
            self._show_model_info()
        elif key == "right":
            self._seek_forward()
        elif key == "left":
            self._seek_backward()
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

    def _like_track(self):
        t = self.current_status.get("track_file")
        if t:
            self.dj.feedback_handler.process_like(t)
            self.liked_tracks.add(t)

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
        # Restore default SIGWINCH handler
        signal.signal(signal.SIGWINCH, signal.SIG_DFL)
        if self.use_urwid:
            raise urwid.ExitMainLoop()

    def _show_model_info(self):
        """
        Model inspector: what the machine currently believes, in measured
        quantities only.

        This replaces the time-context overlay, which had no content left after
        that subsystem was removed (audit D6/L9).  Stage 3 (H1d) extends it with
        the top descriptors for the session and taste vectors and the effective
        sampling temperature; everything shown here is already a real number, so
        nothing needs to be invented in the meantime.
        """
        session = self.dj.session_state.get_stats()
        taste = self.dj.user_taste.get_stats()
        exploration = self.dj.exploration_controller.get_stats()
        selector = self.dj.track_selector.get_stats()
        weights = self.dj.exploration_controller.get_weights(
            taste_updates=taste['total_updates'])

        # τ is the effective number of candidates the sampler is choosing among.
        # Reporting it is the point of picking a rank-based rule: it is a true
        # statement about what the machine is doing, derived rather than named.
        tau = self.dj.track_selector.temperature(exploration['exploration'])
        ramp = self.dj.exploration_controller.taste_ramp(taste['total_updates'])

        schedule = config.skip_turnover_schedule
        run = exploration['consecutive_skips']
        next_target = schedule[min(run + 1, len(schedule)) - 1]

        lines = [
            "MODEL STATE",
            "",
            f"Library             {self.dj.track_library.get_track_count()} tracks",
            f"Session started     {'yes' if session['session_started'] else 'no'}",
            f"Session vector      "
            + ("seeded" if session['is_seeded'] else "unseeded (nothing has played yet)"),
            f"Tracks this session {session['tracks_played']}",
            f"Unique tracks seen  {selector['unique_tracks_played']}",
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
            f"  updates          {taste['total_updates']}",
            f"  β earned         {ramp:.0%}   (full weight after "
            f"{config.taste_ramp_updates} updates)",
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
            "Press any key to close…",
        ]

        if self.use_urwid:
            overlay_text = urwid.Text("\n".join(lines))
            overlay_fill = urwid.Filler(overlay_text, valign="top")
            overlay_box = urwid.LineBox(overlay_fill, title="Model Info")

            # Size the box to its contents rather than to a fixed 70% of the
            # screen.  At the old fixed height the last third of this overlay was
            # silently cut off once Stage 2 added the sampling and skip rows —
            # an inspector that hides what it is inspecting is worse than none.
            # Stage 3 adds the descriptor rows (H1d) and will need this to scroll
            # rather than merely fit.
            _, screen_rows = self.loop.screen.get_cols_rows()
            box_rows = min(len(lines) + 2, max(6, screen_rows - 2))

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
            self.loop.draw_screen()
            self.loop.screen.get_input()
            self.loop.widget = orig

    # ── Periodic update ───────────────────────────────────────────────────────

    def _on_sigwinch(self, signum, frame):
        """
        Called when the terminal is resized or moved to another monitor.
        Marks the album art dirty so _render_art re-sends on the next tick.
        We do NOT call render() here directly — signal handlers must be fast
        and must not write to the ueberzug pipe (not async-signal-safe).
        """
        if self.show_album_art:
            self.album_art_renderer.force_redraw()

    def _periodic_update(self, loop=None, user_data=None):
        self._update_display()
        if self.running:
            self.loop.set_alarm_in(0.5, self._periodic_update)

    # ── Display update ────────────────────────────────────────────────────────

    def _update_display(self):
        status = self.dj.mpd_controller.get_status()
        self.current_status = status

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

        if track_file and track_file in self.liked_tracks:
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

        # Session line (see the widget's construction for why this is a counter)
        self.vibe_text.set_text(("vibe", f"Session: {self._session_line()}"))

        # Console
        self._update_console()

        # Queue
        self._update_queue_display()

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

    def _render_art(self, art_path):
        """
        Calculate the art position from live screen dimensions and re-render.

        Position accounting (0-indexed terminal rows/cols):
          col 0      = terminal left edge
          col 1      = now_playing_box LineBox left │ border
          col 2      = art column inner start            ← x_art

          row 0      = header (1 row)
          row 1      = now_playing_box LineBox top ─ border
          row 2      = top Divider() inside now_playing_content
          row 3      = np_columns area starts               ← y_art

        We always re-send the render command (no skip-if-same guard here)
        so the image reappears after window moves/monitor changes.
        The AlbumArtRenderer.render() itself is the skip guard for the
        same-image/same-position case to avoid redundant sends.
        """
        cols, rows = self.loop.screen.get_cols_rows()

        # X: NP LineBox left border (1) = col 1, art inner starts at col 2
        x_art = 2

        # Y: header(1) + NP top border(1) + top Divider(1) = row 3
        y_art = 3

        # Width: ART_COLS columns wide (no inner LineBox to subtract)
        art_w = ART_COLS

        # Height: pin to the exact row count of the right column Pile.
        # This ensures that even a very wide (landscape) album cover — which
        # ueberzug would scale to fill the full width — cannot overflow
        # downward past the seek bar or vibe line, at any terminal size.
        # RIGHT_COL_ROWS = 10 matches the fixed Pile height exactly.
        art_h = RIGHT_COL_ROWS

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

    # ── Queue update ──────────────────────────────────────────────────────────

    def _update_queue_display(self):
        """
        One line: the single track queued ahead of the current one.

        There is nothing else truthful to show here.  The queue is one deep by
        design (audit D1) so that a skip, a like or a full listen changes what
        plays *next* rather than in ten songs' time; the panel's old job — giving
        the listener a picture of where the session is going — passes to the
        descriptor readout and the session history in Stage 3.
        """
        self.queue_walker.clear()

        next_track = self._next_track_label()
        if next_track is None:
            self.queue_walker.append(
                urwid.AttrMap(urwid.Text("  — nothing queued yet —"), "queue_item"))
        else:
            self.queue_walker.append(
                urwid.AttrMap(urwid.Text(f"  ↓ next:  {next_track}"), "queue_item"))

    def _next_track_label(self) -> Optional[str]:
        """`artist – album – title` for the lookahead, or None if there is none."""
        current = self.current_status.get("track_file")
        track = self.dj.queue_manager.get_next_track(current_track=current)
        if not track:
            return None

        meta = self.dj.mpd_controller.get_playlist_metadata().get(track, {})
        artist = meta.get("artist", "Unknown Artist")
        album = meta.get("album", "Unknown Album")
        title = meta.get("title", Path(track).stem)
        label = f"{artist} – {album} – {title}"
        return f"❤ {label}" if track in self.liked_tracks else label

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
        if tf and tf in self.liked_tracks:
            t = f"❤ {t}"
        print(f"Track:  {t}")

        pos = status.get("position", 0)
        dur = status.get("duration", 0)
        if dur > 0:
            bar = "█" * int(pos / dur * 40) + "░" * (40 - int(pos / dur * 40))
            print(f"\n[{bar}]")
            print(f"{self._fmt(pos)} / {self._fmt(dur)}")

        print(f"\nSession: {self._session_line()}")

        print("\n" + "─" * 60)
        next_track = self._next_track_label()
        print(f"↓ next:  {next_track}" if next_track else "↓ next:  —")

        print("\n" + "=" * 60)
        print("SPACE=Play/Pause  N=Next  L=Like  ↑↓=Vol  ←→=Seek  Q=Quit")
        print("=" * 60)
        sys.stdout.flush()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _fmt(self, seconds: int) -> str:
        m = seconds // 60
        s = seconds % 60
        return f"{m}:{s:02d}"

    def _session_line(self) -> str:
        """
        The one honest thing the session vector can say about itself right now:
        how many tracks it has been fed.  Stage 3 replaces this with the top-3
        CLAP descriptors by z-score plus a measured drift word (audit H1).
        """
        n = self.dj.session_state.tracks_played
        if not self.dj.session_state.session_started:
            return "—"
        return f"{n} track{'' if n == 1 else 's'} played"

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self):
        self.running = True
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
        import select, termios, tty

        old = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while self.running:
                self._update_display()
                if select.select([sys.stdin], [], [], 0.5)[0]:
                    key = sys.stdin.read(1)
                    if key == "\x1b":
                        if select.select([sys.stdin], [], [], 0.05)[0]:
                            key += sys.stdin.read(1)
                        if select.select([sys.stdin], [], [], 0.05)[0]:
                            key += sys.stdin.read(1)

                    if key.lower() == "q":
                        self._quit()
                    elif key == " ":
                        self._toggle_play_pause()
                    elif key.lower() == "n":
                        self._skip_track()
                    elif key.lower() == "l":
                        self._like_track()
                    elif key == "\x1b[A":
                        self._volume_up()
                    elif key == "\x1b[B":
                        self._volume_down()
                    elif key == "\x1b[C":
                        self._seek_forward()
                    elif key == "\x1b[D":
                        self._seek_backward()
        except KeyboardInterrupt:
            pass
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
            self.running = False
            print("\n")
