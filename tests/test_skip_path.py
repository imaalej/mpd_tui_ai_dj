"""
The one skip path (audit C4, constraint retained after D8 dissolved the finding).

C4 was a double advance: `[V]` called `recalculate()`, which cleared the queue,
rebuilt it and called `play()` — landing MPD at queue position 1 — and then the
key handler called `next_track()`, jumping straight past it.  The single
highest-scoring track for the new direction was never heard.

`recalculate()` and `[V]` are both gone, so the code path no longer exists.  What
survives is the constraint it establishes, and these assert it against the new
unified `[N]`:

    exactly one advance per keypress, no `play()` anywhere in the path,
    and the replacement is queued BEFORE the advance.

The ordering is not stylistic.  Verified against the live MPD: `mpc next` off the
last remaining track empties the queue and stops, and a subsequent `mpc add` does
not restart it.  Advance-then-add therefore ends the session in silence, and the
only recovery would be the `play()` call this constraint forbids.
"""

import types

import pytest

from config import config
from main_tui import AdaptiveDJWithTUI


@pytest.fixture
def playing(dj_parts):
    """A session in steady state: something playing, one track queued behind it."""
    dj_parts.queue_manager.ensure_one_ahead(mpd_state='stopped')
    dj_parts.mpd.play()
    dj_parts.queue_manager.ensure_one_ahead(mpd_state='playing')
    dj_parts.mpd.calls.clear()
    return dj_parts


def _skip(parts):
    """
    Drive the **real** `AdaptiveDJWithTUI.skip_current_track`.

    Called unbound against a stand-in holding only the four attributes it
    touches, rather than reimplemented here.  A test that mirrors the ordering
    it is supposed to be checking proves only that the mirror is self-consistent
    — and the ordering is the entire finding.
    """
    dj = types.SimpleNamespace(
        mpd_controller=parts.mpd,
        feedback_handler=parts.feedback,
        queue_manager=parts.queue_manager,
        _last_skip_time=0.0,
    )
    AdaptiveDJWithTUI.skip_current_track(dj)
    return dj._last_skip_time


# ── the constraint ───────────────────────────────────────────────────────────

def test_a_skip_advances_exactly_once(playing):
    before = playing.mpd.get_status()['track_file']

    _skip(playing)

    assert playing.mpd.calls.count('next') == 1
    assert playing.mpd.get_status()['track_file'] != before


def test_a_skip_never_calls_play(playing):
    """
    The structural half of C4.  With no `play()` in the path, a double advance
    cannot recur however the surrounding code is refactored.
    """
    _skip(playing)
    assert 'play' not in playing.mpd.calls


def test_the_replacement_is_queued_before_the_advance(playing):
    """
    Order, asserted directly on the call log rather than inferred from the end
    state — the end state of add-then-advance and advance-then-add differ only
    when the queue is short, which is exactly when it matters.
    """
    _skip(playing)

    adds = [i for i, c in enumerate(playing.mpd.calls) if c.startswith('add:')]
    advance = playing.mpd.calls.index('next')

    assert adds, "the skip queued no replacement at all"
    assert max(adds) < advance, f"advance came first: {playing.mpd.calls}"


def test_the_skipped_track_is_the_one_that_was_playing(playing):
    skipped = playing.mpd.get_status()['track_file']
    _skip(playing)

    events = [e for e in playing.feedback.feedback_history if e['type'] == 'skip']
    assert len(events) == 1
    assert events[0]['track'] == skipped


def test_the_track_that_plays_after_a_skip_is_a_fresh_pick(playing):
    """
    H4's replacement.  At depth 10 the lookahead had been chosen under the
    pre-skip weights, so up to ten more songs of the rejected direction played
    before anything adapted.  At depth 1 the lookahead is dropped and re-picked.
    """
    doomed = playing.mpd.get_queue()[1]

    _skip(playing)

    assert playing.mpd.get_status()['track_file'] != doomed
    assert f'del:2' in playing.mpd.calls


def test_a_dropped_lookahead_does_not_occupy_the_replay_gap(playing):
    """It was queued but never heard, so it must stay eligible."""
    doomed = playing.mpd.get_queue()[1]

    _skip(playing)

    assert doomed not in playing.selector.play_history
    assert doomed not in playing.selector.recent_history


def test_the_session_survives_many_consecutive_skips(playing):
    """The end-to-end version: MPD must still be playing after a long run."""
    for _ in range(15):
        _skip(playing)
        assert playing.mpd.state == 'playing', "a skip killed playback"

    assert playing.mpd.get_queue_length() >= 1


# ── states where a skip must not fire ────────────────────────────────────────

def test_skipping_while_stopped_does_nothing(dj_parts):
    """`mpc next` on a stopped player is an error that changes nothing."""
    dj_parts.queue_manager.ensure_one_ahead(mpd_state='stopped')
    dj_parts.mpd.calls.clear()

    assert _skip(dj_parts) == 0.0, "a stopped player must not even stamp the skip time"
    assert 'next' not in dj_parts.mpd.calls
    assert dj_parts.feedback.session_feedback_count['skips'] == 0


def test_skipping_while_paused_advances_and_stays_paused(playing):
    """
    Measured behaviour, contrary to the audit's assumption: `mpc next` while
    paused consumes *and resumes playing*.  Leaving it there would silently
    start playback for a user who had deliberately paused, and skipping the
    advance instead would replay the very track they just rejected — so the
    path advances and re-pauses.
    """
    playing.mpd.pause()
    rejected = playing.mpd.get_status()['track_file']
    playing.mpd.calls.clear()

    _skip(playing)

    assert playing.mpd.state == 'paused'
    assert playing.mpd.get_status()['track_file'] != rejected
    assert playing.mpd.calls.count('next') == 1


def test_a_skip_that_cannot_queue_a_replacement_does_not_advance(playing):
    """
    The failure that would otherwise end the session: with nothing addable, an
    advance empties the queue, MPD stops, and no later `add` restarts it.  The
    honest outcome is that the keypress does nothing and says so.
    """
    playing.mpd.refuse = set(playing.library.track_list)
    before = playing.mpd.get_status()['track_file']

    _skip(playing)

    assert 'next' not in playing.mpd.calls
    assert playing.mpd.get_status()['track_file'] == before
    assert playing.mpd.state == 'playing'


def test_skipping_the_last_queued_track_still_leaves_something_playing(playing):
    """
    The precise scenario the ordering exists for: only the current track is in
    the queue when the skip arrives.
    """
    playing.mpd.queue = playing.mpd.queue[:1]
    playing.mpd.calls.clear()

    _skip(playing)

    assert playing.mpd.state == 'playing'
    assert playing.mpd.get_queue_length() >= 1
    assert 'play' not in playing.mpd.calls
