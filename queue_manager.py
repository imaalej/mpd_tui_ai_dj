"""
Queue Manager — keeps exactly one track in MPD ahead of the current one.

This file used to maintain a ten-track rolling buffer with two parallel books
(`planned_queue` and `currently_queued_in_mpd`), a `_sync_to_mpd` reconciler, a
5% trajectory blend that simulated the session vector forward across the batch,
and a `recalculate()` that cleared and rebuilt everything.  Almost all of it
existed to compensate for the depth (audit D1/D7), and the refill condition at
the centre of it compared `len(mpc playlist)` — which, with consume off, counts
every track already played — against a low-water mark of 3.  That count only
ever grew, so after the first ten tracks it was pinned at 10, `10 < 3` was never
true, and the queue never refilled again: playback stopped dead (C1).

What replaced it rests on two verified facts about MPD under `consume on`:

  • MPD removes a track from the queue when you *leave* it, not when you start
    it.  The currently playing track is still queue position 1, so during normal
    playback `len(playlist)` is exactly 2 — current plus lookahead — and the
    refill condition is simply `< 2`.  No `#N/M` position parsing.
  • Adding to a stopped, empty queue does **not** start playback, and `mpc next`
    off the last remaining track empties the queue and stops.  Together those
    mean a skip must **add the replacement before it advances**; advance-then-add
    kills the session silently and nothing recovers on its own.

Both were re-measured against the live MPD (0.24.0) rather than remembered — see
PROJECT_AUDIT.md §M1 for the full table.
"""

import sys
from typing import List, Optional, Set

from config import config


class QueueManager:
    """Maintains `config.queue_lookahead` tracks in MPD ahead of the current one."""

    def __init__(self, track_selector, session_state, user_taste,
                 exploration_controller, mpd_controller):
        self.track_selector = track_selector
        self.session_state = session_state
        self.user_taste = user_taste
        self.exploration_controller = exploration_controller
        self.mpd_controller = mpd_controller

        # Whether MPD has ever been observed playing this session.  The refill
        # path is allowed to restart a queue that ran dry, but only after
        # playback has actually begun — otherwise it would override the
        # deliberate "wait for [SPACE]" at startup.
        self._playback_started = False

    # ── Selection ────────────────────────────────────────────────────────────

    def _pick(self, exclude: Optional[Set[str]] = None) -> Optional[str]:
        """
        Choose one track under the *current* vectors and weights.

        Every call reads the session vector, the taste vector and the weights
        fresh.  That is the whole benefit of depth 1: a pick made now reflects
        feedback given a moment ago, where the ten-track buffer scored all ten
        under whatever the weights happened to be when the batch was generated.
        """
        return self.track_selector.select_track(
            session_vector=self.session_state.get_session_vector(),
            taste_vector=self.user_taste.get_taste_vector(),
            weights=self.exploration_controller.get_weights(
                taste_updates=self.user_taste.total_updates),
            exploration=self.exploration_controller.exploration,
            exclude_tracks=set(exclude or ()),
        )

    # ── Queue maintenance ────────────────────────────────────────────────────

    def ensure_one_ahead(self, mpd_state: Optional[str] = None) -> List[str]:
        """
        Top the queue up to `1 + queue_lookahead` entries.  Returns what was added.

        Called from the background poller.  During playback the queue holds the
        current track plus the lookahead, so this is a no-op most of the time
        and adds exactly one track just after each track boundary.

        Args:
            mpd_state: MPD's playback state, passed in by the caller that just
                polled it rather than re-shelled here — the poller runs at 2 Hz
                and every `mpc` invocation is a subprocess.
        """
        if mpd_state == 'playing':
            self._playback_started = True

        queue = self.mpd_controller.get_queue()
        target = 1 + config.queue_lookahead
        if len(queue) >= target:
            return []

        started_empty = not queue
        added = []
        while len(queue) < target:
            track = self._pick(exclude=set(queue))
            if track is None:
                print("Queue refill: the selector had no candidate to offer",
                      file=sys.stderr)
                break
            if not self.mpd_controller.add_track(track):
                # add_track logs MPD's own refusal.  Give the selection back so
                # a track MPD would not accept does not sit in the replay gap.
                self.track_selector.forget_selection(track)
                break
            queue.append(track)
            added.append(track)

        # Recovery for the one case that ends a session silently: the queue ran
        # completely dry — a refill failed for a whole track's duration, or the
        # library was exhausted — so MPD hit the end and stopped.  Adding to a
        # stopped queue does not restart it, verified against the live MPD, so
        # nothing recovers without this.
        #
        # It is deliberately *not* the skip path's escape hatch (audit C4): this
        # fires only when the queue was empty on entry, so it can never combine
        # with an advance to produce the double-skip that C4 documents.
        if started_empty and added and self._playback_started and mpd_state == 'stopped':
            print("Queue had run dry and MPD stopped; restarting playback",
                  file=sys.stderr)
            self.mpd_controller.play()

        return added

    def replace_next(self) -> Optional[str]:
        """
        Drop the lookahead and re-pick it under the current vectors.

        This is the first half of a skip, and the order is load-bearing: the
        replacement is added *before* the caller advances.  `mpc next` off the
        last remaining track empties the queue and stops MPD, and a subsequent
        `mpc add` will not restart it — so advance-then-add ends the session, and
        the only way back would be a `play()` call in a skip path, which is what
        C4 forbids.  Add-then-advance never sees an empty queue.

        Returns the newly queued track, or None if nothing could be queued — in
        which case the caller **must not advance**.
        """
        queue = self.mpd_controller.get_queue()

        # Position 2 is the lookahead; position 1 is what is playing.  When the
        # queue holds only the current track there is nothing to drop, and the
        # delete is expected to fail — mpc exits 1 with "song number does not
        # exist" — so the result is checked rather than assumed.
        if len(queue) >= 2:
            if self.mpd_controller.delete_position(2):
                self.track_selector.forget_selection(queue[1])
                queue = queue[:1]

        track = self._pick(exclude=set(queue))
        if track is None:
            print("Skip: no candidate available to replace the lookahead",
                  file=sys.stderr)
            return None

        if not self.mpd_controller.add_track(track):
            self.track_selector.forget_selection(track)
            return None

        return track

    def requeue_next(self, track: str) -> bool:
        """
        Put a *specific* track in the lookahead slot, replacing whatever is there.

        This is what `ENTER` on the session-history panel does (audit H1d).  It
        lives here rather than in the TUI because it is a queue operation, and
        the ordering below is the same thing C4 and M1 are about — a display
        layer that reimplemented it would be one more place for the sequence to
        drift out of agreement with the verified MPD semantics.

        The order is `add` **then** `del`, which is safer than `replace_next()`'s
        `del` then `add`: the new track is appended to the end, so the queue is
        momentarily `[current, old_next, new]` and the delete takes it back to
        `[current, new]`.  If the add fails, nothing was removed and the queue is
        untouched — where deleting first and failing to add would leave the
        session one track deep until the poller noticed.  `replace_next()` cannot
        do it in this order because it has to *choose* the replacement, and the
        choice must exclude the entry it is about to drop.

        The selector is not told about this track: it was chosen by the listener,
        not drawn, so recording it would let a deliberate replay push the
        selector's own history around.  The dropped lookahead *is* given back —
        it never played, and leaving it recorded would sit it inside the 20-track
        replay gap having never been heard.
        """
        queue = self.mpd_controller.get_queue()

        if not self.mpd_controller.add_track(track):
            print(f"Replay: MPD would not accept {track}", file=sys.stderr)
            return False

        # Position 2 is the lookahead; position 1 is what is playing.  When the
        # queue held only the current track there was no lookahead to drop, and
        # the track just added is already in the right place.
        if len(queue) >= 2:
            if self.mpd_controller.delete_position(2):
                self.track_selector.forget_selection(queue[1])

        return True

    def get_next_track(self, current_track: Optional[str] = None) -> Optional[str]:
        """
        The track queued to play next, for display.

        Before playback begins nothing is current, so the first queue entry *is*
        the next track; once something is playing it sits at position 1 and the
        lookahead is position 2.
        """
        queue = self.mpd_controller.get_queue()
        for track in queue:
            if track != current_track:
                return track
        return None

    def get_stats(self) -> dict:
        """Get queue statistics."""
        return {
            'mpd_queue_size': self.mpd_controller.get_queue_length(),
            'lookahead': config.queue_lookahead,
            'playback_started': self._playback_started,
        }
