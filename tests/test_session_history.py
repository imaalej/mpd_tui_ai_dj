"""
The session history model (audit H1c / L4 / H1d, Stage 3).

The panel this drives replaces the "Upcoming Queue", which listed the session's
*history* above the current track numbered as if it were the future, and whose
`ENTER` binding indexed into `mpc playlist` and so replayed the first track of
the evening (H2).  The audit's instruction was to re-derive the indices from
scratch rather than port them; these tests are what "from scratch" is checked
against — in particular `test_enter_targets_the_row_under_the_cursor_not_the_
first_track`.
"""

from session_history import (MARK_LIKED, MARK_LISTENED, MARK_NONE, MARK_PLAYING,
                             MARK_SKIPPED, SessionHistory)


def played(history, *tracks):
    for track in tracks:
        history.note_playing(track)
    return history


# ── Recording what played ────────────────────────────────────────────────────


def test_tracks_are_recorded_as_they_start():
    history = played(SessionHistory(), "a.flac", "b.flac", "c.flac")
    assert [e.track for e in history.entries] == ["a.flac", "b.flac", "c.flac"]


def test_the_panel_lists_newest_first():
    history = played(SessionHistory(), "a.flac", "b.flac", "c.flac")
    assert [e.track for e in history.newest_first()] == \
        ["c.flac", "b.flac", "a.flac"]


def test_the_same_track_polled_repeatedly_is_one_entry():
    """`note_playing` is called twice a second with MPD's current track."""
    history = SessionHistory()
    for _ in range(20):
        history.note_playing("a.flac")
    assert len(history.entries) == 1


def test_a_track_played_twice_is_two_entries():
    """
    Repeats are not deduplicated: playing something twice is two things that
    happened, and `ENTER` exists specifically to cause it.
    """
    history = played(SessionHistory(), "a.flac", "b.flac", "a.flac")
    assert [e.track for e in history.entries] == ["a.flac", "b.flac", "a.flac"]


def test_nothing_is_recorded_for_a_stopped_player():
    history = SessionHistory()
    assert history.note_playing(None) is False
    assert history.note_playing("") is False
    assert history.entries == []


def test_the_list_is_bounded():
    history = SessionHistory(max_entries=5)
    played(history, *[f"{i}.flac" for i in range(20)])
    assert len(history.entries) == 5
    assert history.entries[-1].track == "19.flac"


# ── Marks ────────────────────────────────────────────────────────────────────


def test_a_skip_marks_the_track_it_was_about():
    history = played(SessionHistory(), "a.flac", "b.flac")
    history.apply_event('skip', "a.flac")
    assert history.entries[0].outcome == MARK_SKIPPED
    assert history.entries[1].outcome is None


def test_a_full_listen_marks_the_track():
    history = played(SessionHistory(), "a.flac")
    history.apply_event('full_listen', "a.flac")
    assert history.entries[0].outcome == MARK_LISTENED


def test_an_event_lands_on_the_most_recent_play_of_that_track():
    """
    A track played twice has two entries, and the event belongs to the one that
    just ended — not the one from an hour ago.
    """
    history = played(SessionHistory(), "a.flac", "b.flac", "a.flac")
    history.apply_event('skip', "a.flac")
    assert history.entries[0].outcome is None
    assert history.entries[2].outcome == MARK_SKIPPED


def test_an_event_for_a_track_that_never_played_is_ignored():
    history = played(SessionHistory(), "a.flac")
    assert history.apply_event('skip', "ghost.flac") is False
    assert history.entries[0].outcome is None


def test_a_like_is_persistent_rather_than_an_outcome():
    """
    `♥` means "you like this track", across sessions; `⏭` and `✓` mean "this is
    what happened to it tonight".  They occupy separate slots because a track can
    be both liked and skipped.
    """
    history = played(SessionHistory(), "a.flac")
    history.apply_event('like', "a.flac")
    history.apply_event('skip', "a.flac")

    assert history.is_liked("a.flac")
    assert history.entries[0].outcome == MARK_SKIPPED
    assert history.marks_for(history.entries[0]) == MARK_LIKED + MARK_SKIPPED


def test_the_playing_track_is_marked_rather_than_left_blank():
    history = played(SessionHistory(), "a.flac", "b.flac")
    assert history.marks_for(history.entries[1], current_track="b.flac") == \
        MARK_NONE + MARK_PLAYING
    assert history.marks_for(history.entries[0], current_track="b.flac") == \
        MARK_NONE + MARK_NONE


def test_an_outcome_wins_over_the_playing_mark():
    """A track can be skipped while still being MPD's current track — the skip
    is recorded before the advance (audit C4)."""
    history = played(SessionHistory(), "a.flac")
    history.apply_event('skip', "a.flac")
    assert history.marks_for(history.entries[0], current_track="a.flac") == \
        MARK_NONE + MARK_SKIPPED


# ── Draining the feedback handler's history ──────────────────────────────────


def test_only_events_after_the_cursor_are_applied():
    """
    `FeedbackHandler.feedback_history` is loaded from disk at startup, so it
    spans every previous session.  The cursor is what keeps last week's skips out
    of tonight's panel.
    """
    events = [
        {'type': 'skip', 'track': "a.flac"},     # last week
        {'type': 'like', 'track': "a.flac"},     # last week
    ]
    history = played(SessionHistory(), "a.flac")
    cursor = len(events)

    events.append({'type': 'full_listen', 'track': "a.flac"})
    cursor = history.drain_events(events, cursor)

    assert cursor == 3
    assert history.entries[0].outcome == MARK_LISTENED, "tonight's event applied"
    assert not history.is_liked("a.flac"), "last week's like was not drained"


def test_draining_twice_is_idempotent():
    events = [{'type': 'skip', 'track': "a.flac"}]
    history = played(SessionHistory(), "a.flac")

    cursor = history.drain_events(events, 0)
    assert cursor == 1
    assert history.drain_events(events, cursor) == 1


def test_likes_from_previous_sessions_rehydrate(tmp_path):
    """
    L4: every like was already on disk with a track path; only the set that drew
    the hearts was in memory, so they vanished on restart while the data that
    produced them sat in `feedback_history.json` untouched.
    """
    events = [
        {'type': 'like', 'track': "a.flac"},
        {'type': 'skip', 'track': "b.flac"},
        {'type': 'like', 'track': "c.flac"},
        {'type': 'full_listen', 'track': "d.flac"},
    ]
    history = SessionHistory()
    history.rehydrate_likes(events)

    assert history.liked == {"a.flac", "c.flac"}
    assert not history.is_liked("b.flac")
    assert history.entries == [], "rehydration must not invent plays"


def test_a_rehydrated_like_shows_a_heart_the_first_time_it_plays():
    history = SessionHistory()
    history.rehydrate_likes([{'type': 'like', 'track': "a.flac"}])
    history.note_playing("a.flac")
    assert history.marks_for(history.entries[0]).startswith(MARK_LIKED)


# ── The cursor ───────────────────────────────────────────────────────────────


def test_the_cursor_starts_on_the_newest_entry():
    history = played(SessionHistory(), "a.flac", "b.flac", "c.flac")
    assert history.focus == 0
    assert history.focused_track() == "c.flac"


def test_down_moves_toward_older_tracks():
    history = played(SessionHistory(), "a.flac", "b.flac", "c.flac")
    history.move_focus(+1)
    assert history.focused_track() == "b.flac"
    history.move_focus(+1)
    assert history.focused_track() == "a.flac"


def test_the_cursor_stops_at_both_ends():
    history = played(SessionHistory(), "a.flac", "b.flac")
    assert history.move_focus(-1) is False
    for _ in range(10):
        history.move_focus(+1)
    assert history.focused_track() == "a.flac"
    assert history.move_focus(+1) is False


def test_a_scrolled_cursor_stays_on_its_track_as_new_ones_arrive():
    """
    The list is newest-first, so an append shifts every existing row down by one.
    A cursor that did not compensate would drift to a different track under the
    listener while they were reading it.
    """
    history = played(SessionHistory(), "a.flac", "b.flac", "c.flac")
    history.move_focus(+2)
    assert history.focused_track() == "a.flac"

    history.note_playing("d.flac")
    assert history.focused_track() == "a.flac"


def test_an_unscrolled_cursor_follows_the_newest_track():
    history = played(SessionHistory(), "a.flac")
    assert history.focused_track() == "a.flac"
    history.note_playing("b.flac")
    assert history.focused_track() == "b.flac"


def test_enter_targets_the_row_under_the_cursor_not_the_first_track():
    """
    The defect this binding replaces (audit H2).  The old `ENTER` indexed into
    `mpc playlist`, which with consume off held the session's history above the
    current track — so pressing it on row "1." replayed the first track of the
    evening no matter where the cursor was.  These indices come from this
    session's own plays, newest first, and nothing else.
    """
    history = played(SessionHistory(), "first.flac", "second.flac", "third.flac")

    assert history.focused_track() == "third.flac"
    history.move_focus(+1)
    assert history.focused_track() == "second.flac"
    assert history.focused_track() != "first.flac"


def test_the_cursor_is_safe_on_an_empty_history():
    history = SessionHistory()
    assert history.focused_track() is None
    assert history.move_focus(+1) is False


# ── Rendering ────────────────────────────────────────────────────────────────


def test_rows_are_newest_first_with_marks_labels_and_a_single_focus():
    history = played(SessionHistory(), "a.flac", "b.flac", "c.flac")
    history.apply_event('skip', "b.flac")
    history.like("a.flac")
    history.move_focus(+1)

    rows = history.rows(current_track="c.flac",
                        labeller=lambda t: t.upper())

    assert [row['track'] for row in rows] == ["c.flac", "b.flac", "a.flac"]
    assert [row['label'] for row in rows] == ["C.FLAC", "B.FLAC", "A.FLAC"]
    assert rows[0]['playing'] is True
    assert rows[1]['marks'] == MARK_NONE + MARK_SKIPPED
    assert rows[2]['marks'] == MARK_LIKED + MARK_NONE
    assert sum(row['focused'] for row in rows) == 1
    assert rows[1]['focused'] is True


def test_rows_can_be_limited_without_moving_the_marks():
    history = played(SessionHistory(), *[f"{i}.flac" for i in range(30)])
    rows = history.rows(limit=5)
    assert len(rows) == 5
    assert rows[0]['track'] == "29.flac"


def test_rows_default_to_the_raw_track_key_without_a_labeller():
    history = played(SessionHistory(), "artist/album/01.flac")
    assert history.rows()[0]['label'] == "artist/album/01.flac"
