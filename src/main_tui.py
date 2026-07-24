"""
Main Orchestration - Adaptive Session AI DJ
The single entry point.  Launched by start.sh.
"""

import atexit
import sys
import threading
import time
import signal
from pathlib import Path

from config import config
from descriptor_bank import DescriptorBank
from music_directory import validate_music_directory
from mpd_controller import MPDController
from track_library import LibraryError, TrackLibrary
from user_taste import UserTaste
from session_state import SessionState
from exploration_controller import ExplorationController
from track_selector import TrackSelector
from queue_manager import QueueManager
from feedback_handler import FeedbackHandler
from persistence import Persistence, ensure_data_directories

from tui import AdaptiveDJTUI


class AdaptiveDJWithTUI:
    """
    Main orchestrator for the Adaptive Session AI DJ with TUI.
    Integrates Phase 1 intelligence core with Phase 2 user interface.
    """
    
    # What the DJ needs MPD to be doing, and why each one matters (audit C2/D2):
    #
    #   random off   — otherwise MPD picks an arbitrary queue entry on advance,
    #                  discarding the ordering at the very last step.  At depth 1
    #                  it can simply replay the current track, so the DJ looks
    #                  stuck rather than mis-ordered.
    #   repeat off   — same, plus it would resurrect consumed tracks.
    #   single off   — would stop after every track.
    #   consume on   — makes MPD pop each finished track itself, which is what
    #                  turns queue management into "is there one ahead?".
    #
    # These are the *user's* settings, so they are logged when changed and
    # restored on every exit path.
    DESIRED_MPD_MODES = {
        'random': 'off',
        'repeat': 'off',
        'single': 'off',
        'consume': 'on',
    }

    def __init__(self):
        self.running = False
        self.tui = None
        self._original_mpd_modes = {}
        self._modes_restored = False
        self._restore_lock = threading.Lock()
        self._tracks_since_checkpoint = 0
        self._last_skip_time = 0.0

        # Initialize Phase 1 components
        print("="*60, file=sys.stderr)
        print("Adaptive Session AI DJ", file=sys.stderr)
        print("="*60, file=sys.stderr)
        
        # Validate config
        config.validate()
        ensure_data_directories()
        
        # MPD Controller
        print("\n[1/10] Connecting to MPD...", file=sys.stderr)
        self.mpd_controller = MPDController()
        if not self.mpd_controller.connect():
            print("ERROR: Could not connect to MPD", file=sys.stderr)
            print(f"Make sure MPD is running on {config.mpd_host}:{config.mpd_port}", file=sys.stderr)
            sys.exit(1)
        print(f"✓ Connected to MPD at {config.mpd_host}:{config.mpd_port}", file=sys.stderr)

        # `mpc listall` is the single source of truth for track keys (M4): the
        # embeddings are stored under these exact strings, and it is the only
        # enumeration that answers "what will MPD actually play".
        mpd_tracks = self.mpd_controller.list_all_tracks()

        # The music directory is only used for tag reads and album art at
        # runtime, but a wrong one is worth catching here rather than as silently
        # missing artwork (M3).  Non-fatal: playback does not depend on it.
        ok, message = validate_music_directory(Path(config.mpd_music_directory), mpd_tracks)
        print(("✓ " if ok else "⚠️  ") + message.replace("\n", "\n   "), file=sys.stderr)

        # Track Library
        print("\n[2/10] Loading track library...", file=sys.stderr)
        self.track_library = TrackLibrary()

        # Embeddings are a hard requirement.  There is deliberately no
        # "generate random ones for now" fallback (audit M2/M4): random vectors
        # make every downstream number meaningless while the UI keeps reporting
        # them as if they meant something.
        try:
            self.track_library.load_embeddings(mpd_tracks=mpd_tracks)
        except LibraryError as exc:
            print(f"\nERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"✓ Loaded {self.track_library.get_track_count()} playable tracks",
              file=sys.stderr)

        # The descriptor bank is a display feature (Stage 3), so a missing one is
        # reported and survived rather than fatal.
        self.descriptor_bank = DescriptorBank.load()
        if self.descriptor_bank:
            print(f"✓ Descriptor bank: {len(self.descriptor_bank)} descriptors",
                  file=sys.stderr)

        # User Taste
        print("\n[3/10] Initializing user taste model...", file=sys.stderr)
        self.user_taste = UserTaste()
        
        # Session State
        print("\n[4/10] Initializing session state...", file=sys.stderr)
        self.session_state = SessionState()
        
        # Exploration Controller
        print("\n[5/10] Initializing exploration controller...", file=sys.stderr)
        self.exploration_controller = ExplorationController()
        
        # Track Selector
        print("\n[6/10] Initializing track selector...", file=sys.stderr)
        self.track_selector = TrackSelector(self.track_library)
        
        # Queue Manager
        print("\n[7/10] Initializing queue manager...", file=sys.stderr)
        self.queue_manager = QueueManager(
            track_selector=self.track_selector,
            session_state=self.session_state,
            user_taste=self.user_taste,
            exploration_controller=self.exploration_controller,
            mpd_controller=self.mpd_controller
        )
        
        # Feedback Handler
        print("\n[8/10] Initializing feedback handler...", file=sys.stderr)
        self.feedback_handler = FeedbackHandler(
            session_state=self.session_state,
            user_taste=self.user_taste,
            exploration_controller=self.exploration_controller,
            track_library=self.track_library
        )
        
        # Persistence
        print("\n[9/10] Setting up persistence...", file=sys.stderr)
        self.persistence = Persistence(
            user_taste=self.user_taste,
            exploration_controller=self.exploration_controller,
            feedback_handler=self.feedback_handler,
            track_selector=self.track_selector
        )
        
        # Load persistent state
        self.persistence.load_all()
        
        # Initialize TUI
        print("\n[10/10] Initializing Terminal UI...", file=sys.stderr)
        self.tui = AdaptiveDJTUI(self)
        
        print("\n" + "="*60, file=sys.stderr)
        print("✓ Initialization complete!", file=sys.stderr)
        print("="*60, file=sys.stderr)
    
    # ── MPD playback modes (audit C2) ────────────────────────────────────────

    def _force_mpd_modes(self):
        """Snapshot the user's playback modes, force what the DJ needs, say so."""
        self._original_mpd_modes = self.mpd_controller.get_modes()
        if not self._original_mpd_modes:
            print("⚠️  Could not read MPD's playback modes; leaving them alone. "
                  "If `random` is on, track ordering will be discarded.",
                  file=sys.stderr)
            return

        changed = []
        for name, wanted in self.DESIRED_MPD_MODES.items():
            current = self._original_mpd_modes.get(name)
            if current is None or current == wanted:
                continue
            if self.mpd_controller.set_mode(name, wanted):
                changed.append(f"{name} {current} → {wanted}")

        if changed:
            print("MPD playback modes changed for this session (restored on exit): "
                  + ", ".join(changed), file=sys.stderr)
        else:
            print("MPD playback modes already as required", file=sys.stderr)

    def _restore_mpd_modes(self):
        """
        Put the user's playback modes back.

        Wired into `_shutdown()`, the signal path *and* an `atexit` hook, because
        leaving someone's MPD in consume mode is a real side effect on their
        system that they will meet the next time they use `mpc` — and until H3
        was fixed, `_shutdown()` was unreachable on SIGTERM entirely.  Guarded by
        a lock and a flag so running three times is harmless.
        """
        with self._restore_lock:
            if self._modes_restored or not self._original_mpd_modes:
                return
            self._modes_restored = True

            restored = []
            for name, wanted in self.DESIRED_MPD_MODES.items():
                original = self._original_mpd_modes.get(name)
                if original is None or original == wanted:
                    continue
                if self.mpd_controller.set_mode(name, original):
                    restored.append(f"{name} → {original}")

            if restored:
                print("MPD playback modes restored: " + ", ".join(restored),
                      file=sys.stderr)

    def start_session(self):
        """Start a new listening session."""
        print("\n🎵 Starting session...", file=sys.stderr)

        # Before touching the queue: MPD's own modes decide whether any ordering
        # decision below survives at all.
        self._force_mpd_modes()

        # Start session state.  Deliberately unseeded — the session vector stays
        # at zero until a real track plays, so the first pick is a uniform draw
        # rather than the neighbourhood of a random direction (audit L7).
        self.session_state.start_session()

        # Reset session-specific state
        self.feedback_handler.reset_session_stats()

        # Clear MPD queue and put the first track plus its lookahead in it.
        self.mpd_controller.clear_queue()
        self.queue_manager.ensure_one_ahead(mpd_state='stopped')

        # Do NOT auto-play on startup — leave MPD stopped so the user
        # can start playback when ready with [SPACE].
        print("Session ready! Press [SPACE] to begin.", file=sys.stderr)
        time.sleep(1)  # Brief pause before TUI takeover

    # ── The one skip path (audit C4/H9) ──────────────────────────────────────

    def skip_current_track(self):
        """
        Reject the playing track: adjust the vectors, replace the lookahead,
        advance exactly once.

        The order is the whole point.  `mpc next` off the last remaining track
        empties the queue and stops MPD, and a later `mpc add` will not restart
        it — so advancing before the replacement is queued ends the session
        silently, and the only way back would be a `play()` call.  There is no
        `play()` anywhere in this method, which is what makes C4's double-advance
        impossible by construction rather than by care.
        """
        status = self.mpd_controller.get_status()
        state = status.get('state')
        track_file = status.get('track_file')

        # `mpc next` on a stopped player is an error ("Not playing") and changes
        # nothing, so there is no skip to perform.
        if state not in ('playing', 'paused') or not track_file:
            return

        # Stamped before anything else so the track-change detector cannot count
        # the abandoned track as a full listen.
        self._last_skip_time = time.time()

        self.feedback_handler.process_skip(track_file)

        # Add the replacement *before* advancing.
        if self.queue_manager.replace_next() is None:
            print("Skip: nothing to advance into, staying on the current track",
                  file=sys.stderr)
            return

        self.mpd_controller.next_track()

        # Verified against the live MPD: `mpc next` while paused advances *and*
        # resumes playback.  The audit assumed it stayed paused.  Re-pausing
        # honours both the rejection and the user's play state; `mpc pause` is
        # idempotent rather than a toggle, so this cannot un-pause anything.
        if state == 'paused':
            self.mpd_controller.pause()

    def neutral_skip_current_track(self):
        """
        Neutral skip (audit G1): advance to the queued track without moving any
        vector — "not this song right now, but keep the vibe".

        It is *simpler* than `skip_current_track`, not harder.  It deliberately
        does **not** call `feedback_handler.process_skip` (no session repel, no
        taste penalty, no exploration change, no escalation counter) and does
        **not** `replace_next` — it plays the lookahead that already exists at
        depth 1, chosen under the vectors as they stand.  Nothing here is a
        rejection; the vibe is left exactly where it is.

        What it shares with the real skip is the one invariant that keeps the
        session alive: **add the replacement before advancing** (audit C4).  At
        depth 1 the lookahead is normally already queued, but at a track boundary
        the queue can momentarily hold only the current track, and `mpc next` off
        the last remaining track empties the queue and stops MPD — a later `add`
        will not restart it.  So if the queue is 1-deep, top it up first; if it
        still cannot be, the keypress does nothing and says so.  There is no
        `play()` here, for the same reason there is none in `skip_current_track`.

        Returns the track that was passed (for the caller to mark in the session
        history), or None if nothing was skipped.
        """
        status = self.mpd_controller.get_status()
        state = status.get('state')
        track_file = status.get('track_file')

        # `mpc next` on a stopped player is an error that changes nothing.
        if state not in ('playing', 'paused') or not track_file:
            return None

        # Stamped so the completion detector cannot count this partially-heard
        # track as a full listen — the same guard the rejection skip uses.
        self._last_skip_time = time.time()

        # Add-before-advance (audit C4).  ensure_one_ahead is a plain refill — it
        # picks under the current vectors and moves nothing — so it does not turn
        # this into a feedback event.
        if len(self.mpd_controller.get_queue()) < 2:
            self.queue_manager.ensure_one_ahead(mpd_state=state)
        if len(self.mpd_controller.get_queue()) < 2:
            print("Pass: nothing to advance into, staying on the current track",
                  file=sys.stderr)
            return None

        self.mpd_controller.next_track()

        # `mpc next` while paused advances *and* resumes playing (verified live);
        # re-pause to honour the user's play state.  `mpc pause` is idempotent.
        if state == 'paused':
            self.mpd_controller.pause()

        return track_file

    def run(self):
        """Main event loop with TUI."""
        self.running = True

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # A last line of defence for the modes specifically: it survives paths
        # the signal handler misses, including an unhandled exception.
        atexit.register(self._restore_mpd_modes)

        # Start session
        self.start_session()

        # Background event processing
        event_thread = threading.Thread(target=self._background_event_loop, daemon=True)
        event_thread.start()

        # Run TUI (blocks until quit)
        try:
            self.tui.run()
        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()
    
    def _background_event_loop(self):
        """
        Background thread that handles MPD events.
        Runs alongside TUI.
        """
        last_track_file = None
        track_duration = None
        track_completion_processed = False
        last_check_time = time.time()

        while self.running:
            try:
                # Poll MPD status
                status = self.mpd_controller.get_status()

                if status['state'] == 'playing':
                    current_track = status.get('track_file')

                    # Detect track change
                    if current_track and current_track != last_track_file:
                        last_track_file = current_track
                        track_duration = status.get('duration', 0)
                        track_completion_processed = False

                        # The session vector adopts the first track that plays
                        # rather than starting from a random direction (L7).
                        # A no-op once there is a real vibe.
                        if not self.session_state.is_seeded():
                            embedding = self.track_library.get_embedding(current_track)
                            if embedding is not None:
                                self.session_state.seed(embedding)

                    # Check for track completion
                    if last_track_file and current_track and not track_completion_processed:
                        position = status.get('position', 0)

                        if track_duration and position > 0:
                            completion_threshold = self._completion_threshold(track_duration)

                            # Only fire full-listen if track was NOT manually skipped recently
                            recently_skipped = (time.time() - self._last_skip_time) < 2.0
                            if position >= completion_threshold and not recently_skipped:
                                # Track completed - process as full listen
                                self.feedback_handler.process_full_listen(last_track_file)
                                track_completion_processed = True
                                self._maybe_checkpoint()

                # Periodically top the queue back up to one-ahead.  The state is
                # passed through so the refill does not shell out to `mpc status`
                # a second time for something already known.
                current_time = time.time()
                if current_time - last_check_time > 2.0:
                    self.queue_manager.ensure_one_ahead(mpd_state=status.get('state'))
                    last_check_time = current_time

                # Sleep to avoid busy-waiting
                time.sleep(config.mpd_poll_interval)

            except Exception as e:
                print(f"Background loop error: {e}", file=sys.stderr)
                time.sleep(1)

    @staticmethod
    def _completion_threshold(duration: float) -> float:
        """
        Seconds of a track that count as a full listen (audit B1).

        A flat fraction of the duration: three-quarters is a genuine full
        listen, and 90% was stricter than intended.  The old formula took
        `max(0.9·duration, duration − 10)`; the `duration − 10` term made *long*
        tracks stricter, not more lenient (a 4-minute track needed 3:50), which
        is the opposite of the intent, so it is dropped.  A flat fraction also
        widens the window a 0.5 s poll has to land in for the completion to be
        seen at all (audit B2): for a 4-minute track that window grows from ~10 s
        to 60 s, so a poll delayed by thread contention is far less likely to
        miss the track's end.

        Seeking is deliberately not special-cased: the user may seek however they
        like, and a completion reached by a forward seek counting as a listen is
        acceptable by design.
        """
        return config.full_listen_fraction * duration

    def _maybe_checkpoint(self):
        """
        Write learned state every few tracks rather than only at exit (audit H3).

        Saving at exit alone meant a terminal closing, a `systemctl stop` or a
        logout discarded the whole session's exploration state and feedback
        history.  Quiet, because this runs during normal listening.
        """
        self._tracks_since_checkpoint += 1
        if self._tracks_since_checkpoint >= config.checkpoint_every_n_tracks:
            self._tracks_since_checkpoint = 0
            try:
                self.persistence.save_all(quiet=True)
            except Exception as e:
                print(f"Checkpoint failed: {e}", file=sys.stderr)

    def _signal_handler(self, signum, frame):
        """
        Handle shutdown signals.

        This used to set `self.running = False` and nothing else, which stopped
        the background thread while urwid's `MainLoop` carried on unaware —
        leaving a live-looking UI with no MPD polling, no track detection and no
        queue management behind it, and `_shutdown()` (and therefore every save)
        unreachable.  Unblocking the loop is what makes the rest of the shutdown
        path run at all (audit H3).

        Signal handlers must stay minimal, so only the things that alter state
        *outside* this process are done here directly — the mode restore, and
        ending the album-art child.  Both must happen even if urwid never
        yields.  The ueberzugpp leak (L9) is the same finding as this one seen
        from the other side: before H3, `_shutdown()` was unreachable on
        SIGTERM, so the child was orphaned every time the DJ was stopped by
        anything other than `[Q]`.
        """
        self.running = False
        self._restore_mpd_modes()
        self._shutdown_album_art()
        if self.tui is not None:
            self.tui.request_exit()

    def _shutdown_album_art(self):
        """
        End the album-art child process.  Idempotent (audit L9).

        Nothing in here may raise.  It runs on the signal path, ahead of
        `tui.request_exit()` — the call that unblocks urwid and makes the whole
        of the rest of the shutdown reachable (H3) — so an exception escaping
        here would restore H3 in exchange for fixing L9.
        """
        renderer = getattr(self.tui, 'album_art_renderer', None)
        if renderer is None:
            return
        try:
            renderer.shutdown()
        except Exception as e:
            print(f"Album art shutdown failed: {e}", file=sys.stderr)

    def _shutdown(self):
        """Clean shutdown."""
        print("\n\nShutting down...", file=sys.stderr)
        self.running = False

        # Save state
        try:
            self.persistence.save_all()
        except Exception as e:
            print(f"Error saving state: {e}", file=sys.stderr)

        # Hand MPD back the way it was found
        self._restore_mpd_modes()

        # Clear the album art *and* end the child process holding it (L9).
        # `clear()` alone left `ueberzugpp` running with its stdin open.
        self._shutdown_album_art()

        print("Goodbye!", file=sys.stderr)


def main():
    """Main entry point."""
    dj = AdaptiveDJWithTUI()
    dj.run()


if __name__ == '__main__':
    main()
