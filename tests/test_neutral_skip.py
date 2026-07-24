"""
The neutral skip — [V] "Pass" (audit G1).

A pass means "not this song right now, but keep the vibe": it advances to the
already-queued lookahead and touches **none** of the model.  Where the rejection
skip ([N], test_skip_path.py) repels the session vector, penalises taste, raises
exploration and escalates, a pass does none of that.

What it *shares* with the rejection skip is the one invariant that keeps the
session alive: the replacement is queued **before** the advance, and there is no
`play()` anywhere in the path (audit C4).  `mpc next` off the last remaining
track empties the queue and stops MPD, and a later `mpc add` will not restart it,
so advance-then-add ends the session in silence.

These drive the **real** `AdaptiveDJWithTUI.neutral_skip_current_track` against
`FakeMPD`, asserting on its call log — the same discipline test_skip_path.py uses,
because a test that mirrors the ordering it checks proves only that the mirror is
self-consistent.
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


def _pass(parts):
    """
    Drive the real `AdaptiveDJWithTUI.neutral_skip_current_track`.

    Called unbound against a stand-in holding only the three attributes it
    touches — MPD, the queue manager and the skip timestamp.  It must **not**
    touch `feedback_handler`; giving the stand-in one it can mutate would let a
    regression that started calling it pass silently, so it is left off.
    """
    dj = types.SimpleNamespace(
        mpd_controller=parts.mpd,
        queue_manager=parts.queue_manager,
        _last_skip_time=0.0,
    )
    return AdaptiveDJWithTUI.neutral_skip_current_track(dj)


def _model_snapshot(parts):
    """Everything a pass must leave untouched."""
    return {
        'session_vector': parts.session_state.get_session_vector().copy(),
        'taste_vector': parts.user_taste.get_taste_vector().copy(),
        'taste_updates': parts.user_taste.total_updates,
        'exploration': parts.exploration.exploration,
        'consecutive_skips': parts.exploration.consecutive_skips,
        'feedback_history': copy.deepcopy(parts.feedback.feedback_history),
        'skip_count': parts.feedback.session_feedback_count['skips'],
    }


# ── the shared invariant ─────────────────────────────────────────────────────

def test_a_pass_advances_exactly_once(playing):
    before = playing.mpd.get_status()['track_file']

    _pass(playing)

    assert playing.mpd.calls.count('next') == 1
    assert playing.mpd.get_status()['track_file'] != before


def test_a_pass_never_calls_play(playing):
    """The structural half of C4 — a pass is a skip variant and obeys it too."""
    _pass(playing)
    assert 'play' not in playing.mpd.calls


def test_a_pass_plays_the_track_that_was_already_queued(playing):
    """
    The defining difference from [N]: a pass does **not** re-pick the lookahead
    (no `replace_next`, so no `del:2`).  It plays the track that was already
    lined up — "keep the vibe".
    """
    already_queued = playing.mpd.get_queue()[1]

    passed = _pass(playing)

    assert playing.mpd.get_status()['track_file'] == already_queued
    assert 'del:2' not in playing.mpd.calls
    assert passed is not None


def test_a_pass_returns_the_track_it_passed(playing):
    rejected = playing.mpd.get_status()['track_file']
    assert _pass(playing) == rejected


# ── the point: nothing in the model moves ────────────────────────────────────

def test_a_pass_changes_no_model_state(playing):
    """
    No session repel, no taste penalty, no exploration change, no escalation
    counter — the whole reason the feature exists.
    """
    before = _model_snapshot(playing)

    _pass(playing)

    after = _model_snapshot(playing)
    assert np.array_equal(after['session_vector'], before['session_vector'])
    assert np.array_equal(after['taste_vector'], before['taste_vector'])
    assert after['taste_updates'] == before['taste_updates']
    assert after['exploration'] == before['exploration']
    assert after['consecutive_skips'] == before['consecutive_skips']


def test_a_pass_records_no_feedback_event(playing):
    before = _model_snapshot(playing)

    _pass(playing)

    assert playing.feedback.feedback_history == before['feedback_history']
    assert playing.feedback.session_feedback_count['skips'] == before['skip_count']


def test_a_pass_does_not_escalate_across_repeated_presses(playing):
    """Ten passes in a row must leave exploration and the skip run exactly where
    they started — a pass is never evidence of anything."""
    exploration0 = playing.exploration.exploration
    for _ in range(10):
        _pass(playing)
    assert playing.exploration.exploration == exploration0
    assert playing.exploration.consecutive_skips == 0
    assert playing.mpd.state == 'playing'


# ── add-before-advance, the C4 ordering ──────────────────────────────────────

def test_a_pass_tops_up_a_one_deep_queue_before_advancing(playing):
    """
    At a track boundary the queue can momentarily hold only the current track.
    Advancing off it empties the queue and stops MPD; a pass must add first.
    """
    playing.mpd.queue = playing.mpd.queue[:1]
    playing.mpd.calls.clear()

    _pass(playing)

    adds = [i for i, c in enumerate(playing.mpd.calls) if c.startswith('add:')]
    advance = playing.mpd.calls.index('next')
    assert adds, "the pass queued no lookahead before advancing"
    assert max(adds) < advance, f"advance came first: {playing.mpd.calls}"
    assert playing.mpd.state == 'playing'
    assert playing.mpd.get_queue_length() >= 1


def test_a_pass_with_a_full_queue_adds_nothing(playing):
    """When the lookahead already exists, a pass just advances into it."""
    assert playing.mpd.get_queue_length() == 2

    _pass(playing)

    assert not any(c.startswith('add:') for c in playing.mpd.calls)


def test_the_session_survives_many_consecutive_passes(playing):
    for _ in range(15):
        _pass(playing)
        assert playing.mpd.state == 'playing', "a pass killed playback"
    assert playing.mpd.get_queue_length() >= 1


# ── states where a pass must not fire ────────────────────────────────────────

def test_passing_while_stopped_does_nothing(dj_parts):
    dj_parts.queue_manager.ensure_one_ahead(mpd_state='stopped')
    dj_parts.mpd.calls.clear()

    assert _pass(dj_parts) is None
    assert 'next' not in dj_parts.mpd.calls


def test_passing_while_paused_advances_and_stays_paused(playing):
    """
    `mpc next` while paused consumes *and resumes playing* (verified live), so a
    pass re-pauses — honouring the user's deliberate pause without replaying the
    track they just moved past.
    """
    playing.mpd.pause()
    passed_from = playing.mpd.get_status()['track_file']
    playing.mpd.calls.clear()

    _pass(playing)

    assert playing.mpd.state == 'paused'
    assert playing.mpd.get_status()['track_file'] != passed_from
    assert playing.mpd.calls.count('next') == 1
    assert 'play' not in playing.mpd.calls


def test_a_pass_that_cannot_queue_a_lookahead_does_not_advance(playing):
    """
    Only the current track is queued and MPD will accept nothing new: advancing
    would empty the queue and stop the session, so the keypress does nothing.
    """
    playing.mpd.queue = playing.mpd.queue[:1]
    playing.mpd.refuse = set(playing.library.track_list)
    before = playing.mpd.get_status()['track_file']
    playing.mpd.calls.clear()

    assert _pass(playing) is None
    assert 'next' not in playing.mpd.calls
    assert playing.mpd.get_status()['track_file'] == before
    assert playing.mpd.state == 'playing'
