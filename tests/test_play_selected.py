"""
Play a specific track — [P] "play the selected cloud point now".

`AdaptiveDJWithTUI.play_track_now(track)` plays a track the *listener* chose by
clicking a point in the vibe cloud.  Like the neutral pass (test_neutral_skip.py)
it moves **no** model state — choosing a song to hear is not feedback about the
one playing — and like every skip variant it obeys the one C4 invariant: the
chosen track is queued *before* the advance, and there is no `play()` in the path.

The one difference from a pass is *what* is queued: `requeue_next(track)` seats
the listener's pick as the lookahead (which bypasses the selector's
replay-exclusion window), then a single `next` steps onto it.

These drive the **real** method against `FakeMPD`, asserting on its call log — the
same discipline test_neutral_skip.py uses.
"""

import copy
import types

import numpy as np
import pytest

from main_tui import AdaptiveDJWithTUI


@pytest.fixture
def playing(dj_parts):
    """A session in steady state: something playing, one track queued behind it."""
    dj_parts.queue_manager.ensure_one_ahead(mpd_state='stopped')
    dj_parts.mpd.play()
    dj_parts.queue_manager.ensure_one_ahead(mpd_state='playing')
    dj_parts.mpd.calls.clear()
    return dj_parts


def _pick_absent_track(parts):
    """A real library track that is not already in the queue — the kind of pick a
    click on a far cloud point makes."""
    in_queue = set(parts.mpd.get_queue())
    for track in parts.library.track_list:
        if track not in in_queue:
            return track
    raise AssertionError("no library track outside the queue to pick")


def _play(parts, track):
    """Drive the real `play_track_now` against a stand-in holding only the three
    attributes it touches — MPD, the queue manager and the skip timestamp.  It
    must **not** touch `feedback_handler`; leaving it off means a regression that
    started recording feedback would raise here rather than pass silently."""
    dj = types.SimpleNamespace(
        mpd_controller=parts.mpd,
        queue_manager=parts.queue_manager,
        _last_skip_time=0.0,
    )
    return AdaptiveDJWithTUI.play_track_now(dj, track)


def _model_snapshot(parts):
    """Everything a play-now must leave untouched."""
    return {
        'session_vector': parts.session_state.get_session_vector().copy(),
        'taste_vector': parts.user_taste.get_taste_vector().copy(),
        'taste_updates': parts.user_taste.total_updates,
        'exploration': parts.exploration.exploration,
        'consecutive_skips': parts.exploration.consecutive_skips,
        'feedback_history': copy.deepcopy(parts.feedback.feedback_history),
    }


# ── it plays the chosen track, add-before-advance ────────────────────────────

def test_play_now_advances_onto_the_chosen_track(playing):
    chosen = _pick_absent_track(playing)
    before = playing.mpd.get_status()['track_file']

    passed = _play(playing, chosen)

    assert playing.mpd.get_status()['track_file'] == chosen
    assert playing.mpd.calls.count('next') == 1
    assert passed == before, "returns the track it passed over"


def test_play_now_queues_the_pick_before_it_advances(playing):
    """C4: the chosen track is added before `next`, so the queue is never empty
    when the advance fires."""
    chosen = _pick_absent_track(playing)

    _play(playing, chosen)

    adds = [i for i, c in enumerate(playing.mpd.calls) if c.startswith('add:')]
    advance = playing.mpd.calls.index('next')
    assert adds, "nothing was queued before advancing"
    assert max(adds) < advance, f"advance came first: {playing.mpd.calls}"


def test_play_now_never_calls_play(playing):
    """The structural half of C4 — play-now is a skip variant and obeys it too."""
    _play(playing, _pick_absent_track(playing))
    assert 'play' not in playing.mpd.calls


def test_play_now_changes_no_model_state(playing):
    """No repel, no taste move, no exploration change, no escalation — hearing a
    chosen track says nothing about the one it interrupted."""
    before = _model_snapshot(playing)

    _play(playing, _pick_absent_track(playing))

    after = _model_snapshot(playing)
    assert np.array_equal(after['session_vector'], before['session_vector'])
    assert np.array_equal(after['taste_vector'], before['taste_vector'])
    assert after['taste_updates'] == before['taste_updates']
    assert after['exploration'] == before['exploration']
    assert after['consecutive_skips'] == before['consecutive_skips']
    assert after['feedback_history'] == before['feedback_history']


# ── states where it must not fire ────────────────────────────────────────────

def test_play_now_while_stopped_does_nothing(dj_parts):
    dj_parts.queue_manager.ensure_one_ahead(mpd_state='stopped')
    chosen = dj_parts.library.track_list[0]
    dj_parts.mpd.calls.clear()

    assert _play(dj_parts, chosen) is None
    assert 'next' not in dj_parts.mpd.calls


def test_play_now_while_paused_advances_and_stays_paused(playing):
    """`mpc next` while paused consumes *and resumes* (verified live), so play-now
    re-pauses to honour the user's play state."""
    chosen = _pick_absent_track(playing)
    playing.mpd.pause()
    passed_from = playing.mpd.get_status()['track_file']
    playing.mpd.calls.clear()

    _play(playing, chosen)

    assert playing.mpd.state == 'paused'
    assert playing.mpd.get_status()['track_file'] == chosen
    assert playing.mpd.calls.count('next') == 1
    assert 'play' not in playing.mpd.calls
