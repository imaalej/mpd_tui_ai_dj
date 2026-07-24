"""
The session history — what actually played, with what you did about it.

Audit H1c/L4, Stage 3.  This is the panel that replaces the "Upcoming Queue",
and the reason it is truthful where that one was not: it lists the past, which
happened, rather than a future that one keypress erases.

**Why the model lives in the display layer.**  Nothing behind the display stores
this.  `FeedbackHandler.feedback_history` is close — it is an ordered list of
`{type, track}` events — but it is loaded from disk at startup, so it spans every
previous session, and it records outcomes rather than plays: a track that is
still playing has no event yet, and a track played twice appears once per
outcome.  The panel needs the opposite shape (one row per play, in order, marked
with whatever outcome arrived).  So the order comes from watching MPD's current
track change, and the marks come from draining new feedback events from a cursor
taken at session start.  Neither requires the player to grow a new store.

**L4 falls out of this.**  Likes are already on disk with a track path, so the
hearts that used to vanish on restart are rehydrated from the same feedback file
— `♥` means "you have liked this track", across sessions, while `⏭` and `✓` mean
"this is what happened to it tonight".

**Focus is an index into this list, derived from scratch.**  The bindings this
replaces counted from the top of `mpc playlist`, which under `consume off` is the
session's *history* — so `ENTER` on row 1 replayed the first track of the evening
(audit H2).  There is no shared numbering with MPD here at all: the rows are
these entries, newest first, and `ENTER` requeues the entry under the cursor.
"""

from typing import Dict, List, Optional, Sequence

# Outcome marks.  Two slots: a persistent one (do you like this track, ever) and
# a session one (what happened to it tonight).
MARK_LIKED = "♥"
MARK_SKIPPED = "⏭"
MARK_LISTENED = "✓"
MARK_PLAYING = "♪"
# A neutral pass (audit G1): the listener moved on with [V] without rejecting the
# track.  Deliberately neither ⏭ (which reads as rejection and drives the
# session vector away) nor ✓ (a full listen).  "You passed on this one, and the
# vibe is untouched."
MARK_PASSED = "»"
MARK_NONE = " "

# Maps a FeedbackHandler event type onto a session mark.  'like' is deliberately
# absent: it is persistent, so it is carried by `liked` rather than by an entry.
_EVENT_MARKS = {
    'skip': MARK_SKIPPED,
    'full_listen': MARK_LISTENED,
}


class Entry:
    """One play: a track, and what became of it."""

    __slots__ = ('track', 'outcome')

    def __init__(self, track: str):
        self.track = track
        self.outcome: Optional[str] = None

    def __repr__(self):
        return f"Entry({self.track!r}, outcome={self.outcome!r})"


class SessionHistory:
    """
    Ordered record of what played this session, oldest first, plus the scroll
    cursor the panel and `ENTER` share.
    """

    # A long session is a few hundred tracks; the panel shows a screenful.  The
    # cap exists so an all-night session cannot grow without bound, not because
    # any particular number is meaningful.
    MAX_ENTRIES = 500

    def __init__(self, max_entries: int = MAX_ENTRIES):
        self.max_entries = max_entries
        self.entries: List[Entry] = []          # oldest first
        self.liked: set = set()                 # persistent, across sessions
        # Index into the *newest-first* view.  0 follows whatever is newest;
        # anything else is pinned to a specific entry and stays on it as new
        # tracks arrive above it.
        self.focus: int = 0

    # ── Recording ────────────────────────────────────────────────────────────

    def note_playing(self, track: Optional[str]) -> bool:
        """
        Called with MPD's current track on every refresh.  Appends when it
        changes.  Returns True if a new entry was created.

        Repeats are appended rather than deduplicated: playing a track twice is
        two things that happened, and `ENTER` exists specifically to cause that.
        """
        if not track:
            return False
        if self.entries and self.entries[-1].track == track:
            return False

        self.entries.append(Entry(track))

        # Keep the cursor on the entry it was pointing at.  It counts from the
        # newest end, so inserting a newer entry shifts it by one — unless it is
        # already following the newest, which is the state to preserve.
        if self.focus > 0:
            self.focus += 1

        if len(self.entries) > self.max_entries:
            overflow = len(self.entries) - self.max_entries
            del self.entries[:overflow]

        self._clamp_focus()
        return True

    def apply_event(self, event_type: str, track: str) -> bool:
        """
        Apply one `FeedbackHandler` event to the most recent play of that track.

        Most recent, because a track played twice in an evening has two entries
        and the event belongs to the one that just ended.  Returns True if an
        entry was marked.
        """
        if event_type == 'like':
            self.liked.add(track)
            return True

        mark = _EVENT_MARKS.get(event_type)
        if mark is None:
            return False

        for entry in reversed(self.entries):
            if entry.track == track:
                entry.outcome = mark
                return True
        return False

    def mark_passed(self, track: str) -> bool:
        """
        Mark the most recent play of `track` as neutrally passed (audit G1).

        A neutral skip touches none of the model — no session repel, no taste
        penalty, no exploration change — so unlike `⏭` and `✓` it produces *no*
        `FeedbackHandler` event, and there is nothing for `drain_events` to carry.
        The mark is therefore set here directly, from the key handler.

        Only an entry with no outcome yet is touched.  If the track already
        crossed the full-listen threshold and earned `✓` before the listener
        pressed `[V]`, that happened and stands — a pass does not un-count a
        listen.  Returns True if an entry was marked.
        """
        for entry in reversed(self.entries):
            if entry.track == track:
                if entry.outcome is None:
                    entry.outcome = MARK_PASSED
                    return True
                return False
        return False

    def drain_events(self, events: Sequence[dict], cursor: int) -> int:
        """
        Apply every feedback event from `cursor` onward; return the new cursor.

        The cursor is how this stays a *session* history while reading a list
        that spans every session on disk.
        """
        for event in events[cursor:]:
            track = event.get('track')
            if track:
                self.apply_event(event.get('type', ''), track)
        return len(events)

    def rehydrate_likes(self, events: Sequence[dict]):
        """
        Recover `♥` from the feedback file written by previous sessions (L4).

        Every like is already on disk with a track path; the set that drives the
        hearts was in-memory only, so they vanished on restart while the data
        that produced them sat in `feedback_history.json` untouched.
        """
        for event in events:
            if event.get('type') == 'like' and event.get('track'):
                self.liked.add(event['track'])

    # ── Reading ──────────────────────────────────────────────────────────────

    def newest_first(self) -> List[Entry]:
        return list(reversed(self.entries))

    def marks_for(self, entry: Entry, current_track: Optional[str] = None) -> str:
        """
        Two mark slots: `♥` if the track is liked at all, then tonight's outcome.

        A track that is playing right now and has no outcome yet gets `♪` — the
        alternative is a blank row for the one entry the listener is most sure
        about.
        """
        heart = MARK_LIKED if entry.track in self.liked else MARK_NONE
        outcome = entry.outcome
        if outcome is None:
            outcome = MARK_PLAYING if entry.track == current_track else MARK_NONE
        return heart + outcome

    def is_liked(self, track: Optional[str]) -> bool:
        return bool(track) and track in self.liked

    def like(self, track: str):
        self.liked.add(track)

    def unlike(self, track: str):
        """
        Drop the `♥`.  The counterpart to `like()` for `[L]` as a toggle (L8).

        Display-side only, and deliberately so: this set is what the panel
        draws, while what the *model* does with a retraction is
        `FeedbackHandler.process_unlike()`.  The heart is removed here after
        that call succeeds, so a retraction the model could not perform does not
        leave the panel claiming otherwise.
        """
        self.liked.discard(track)

    # ── Focus ────────────────────────────────────────────────────────────────

    def move_focus(self, delta: int) -> bool:
        """
        Move the cursor within the newest-first view.  `−1` is up (newer), `+1`
        is down (older).  Returns True if it moved.
        """
        if not self.entries:
            return False
        before = self.focus
        self.focus = max(0, min(len(self.entries) - 1, self.focus + delta))
        return self.focus != before

    def focused_track(self) -> Optional[str]:
        """The track under the cursor, or None when there is no history."""
        rows = self.newest_first()
        if not rows:
            return None
        index = max(0, min(len(rows) - 1, self.focus))
        return rows[index].track

    def _clamp_focus(self):
        if not self.entries:
            self.focus = 0
        else:
            self.focus = max(0, min(len(self.entries) - 1, self.focus))

    # ── Rendering ────────────────────────────────────────────────────────────

    def rows(self, current_track: Optional[str] = None,
             labeller=None, limit: Optional[int] = None) -> List[Dict]:
        """
        The panel's rows, newest first.

        `labeller(track) -> str` turns a track key into something readable; the
        caller supplies it because resolving tags is MPD's job, not this
        model's.  Each row carries its own focus flag so the widget layer does
        not have to recompute the index.
        """
        entries = self.newest_first()
        if limit is not None:
            entries = entries[:limit]
        out = []
        for i, entry in enumerate(entries):
            out.append({
                'track': entry.track,
                'marks': self.marks_for(entry, current_track),
                'label': labeller(entry.track) if labeller else entry.track,
                'focused': i == self.focus,
                'playing': entry.track == current_track,
            })
        return out
