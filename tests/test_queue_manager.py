"""
Queue behaviour at depth 1 (audit C1 / D1).

C1 was that `check_and_refill()` compared `len(mpc playlist)` — which counts
every track already played while consume is off — against a low-water mark of 3.
After the initial fill it was pinned at 10, `10 < 3` never fired, and playback
stopped dead.  These are behavioural: they drive the real `QueueManager` against
a `FakeMPD` and assert what the queue *does* over a session, because C1 survived
a green suite full of existence checks.
"""

import pytest

from config import config


def _play_through(parts, n_tracks):
    """Run n tracks to their natural end, refilling the way the poller does."""
    heard = []
    parts.queue_manager.ensure_one_ahead(mpd_state='stopped')
    parts.mpd.play()
    for _ in range(n_tracks):
        status = parts.mpd.get_status()
        assert status['state'] == 'playing', "playback stalled"
        heard.append(status['track_file'])
        parts.mpd.finish_track()
        parts.queue_manager.ensure_one_ahead(mpd_state=parts.mpd.state)
    return heard


def test_the_queue_settles_at_exactly_one_ahead(dj_parts):
    dj_parts.queue_manager.ensure_one_ahead(mpd_state='stopped')
    dj_parts.mpd.play()
    dj_parts.queue_manager.ensure_one_ahead(mpd_state='playing')

    assert dj_parts.mpd.get_queue_length() == 1 + config.queue_lookahead == 2


def test_refill_is_a_no_op_once_the_lookahead_is_there(dj_parts):
    dj_parts.queue_manager.ensure_one_ahead(mpd_state='stopped')
    dj_parts.mpd.play()
    dj_parts.queue_manager.ensure_one_ahead(mpd_state='playing')

    added = dj_parts.queue_manager.ensure_one_ahead(mpd_state='playing')
    assert added == []
    assert dj_parts.mpd.get_queue_length() == 2


def test_playback_does_not_stop_after_ten_tracks(dj_parts):
    """
    C1's direct regression.  The old manager reached `stopped` on track 10; this
    plays past it without a stall.
    """
    heard = _play_through(dj_parts, 30)

    assert len(heard) == 30
    assert dj_parts.mpd.state == 'playing'
    assert dj_parts.mpd.get_queue_length() == 2


def test_no_track_repeats_inside_the_replay_gap(dj_parts):
    """
    The done-criterion for the stage.  The library fixture holds 64 tracks and
    the gap is 20, so a repeat inside the window is a real failure rather than
    an artefact of a small library.
    """
    heard = _play_through(dj_parts, 40)

    gap = config.minimum_replay_gap
    for i, track in enumerate(heard):
        window = heard[max(0, i - gap):i]
        assert track not in window, (
            f"{track} replayed after {i - window.index(track)} tracks, "
            f"inside the {gap}-track gap")


def test_a_track_mpd_refuses_is_not_left_sitting_in_the_replay_gap(dj_parts):
    """
    H7's return code, followed through.  A refused `add` used to be recorded as
    success; now it is reported — and the selection is handed back, so a track
    that never entered the queue does not block itself for the next 20 picks.
    """
    dj_parts.queue_manager.ensure_one_ahead(mpd_state='stopped')
    dj_parts.mpd.play()

    # Refuse everything, then ask for a refill.
    dj_parts.mpd.refuse = set(dj_parts.library.track_list)
    before_index = dj_parts.selector.current_index

    added = dj_parts.queue_manager.ensure_one_ahead(mpd_state='playing')

    assert added == []
    assert dj_parts.selector.current_index == before_index
    assert len(dj_parts.selector.recent_history) == before_index


def test_a_queue_that_ran_dry_mid_session_is_restarted(dj_parts):
    """
    The one path that would otherwise end a session in silence: a refill failed
    for a whole track's duration, MPD reached the end and stopped, and adding to
    a stopped queue does not start it.
    """
    dj_parts.queue_manager.ensure_one_ahead(mpd_state='stopped')
    dj_parts.mpd.play()
    dj_parts.queue_manager.ensure_one_ahead(mpd_state='playing')

    # Drain it the way MPD would if nothing had refilled in time.
    dj_parts.mpd.queue = []
    dj_parts.mpd.state = 'stopped'

    dj_parts.queue_manager.ensure_one_ahead(mpd_state='stopped')

    assert dj_parts.mpd.state == 'playing'
    assert dj_parts.mpd.get_queue_length() >= 1


def test_the_dry_queue_recovery_cannot_fire_before_playback_has_begun(dj_parts):
    """
    Startup deliberately leaves MPD stopped so the user presses [SPACE].  The
    recovery must not turn that into an auto-play.
    """
    dj_parts.queue_manager.ensure_one_ahead(mpd_state='stopped')

    assert dj_parts.mpd.state == 'stopped'
    assert 'play' not in dj_parts.mpd.calls


def test_get_next_track_reports_the_lookahead_not_the_current_track(dj_parts):
    dj_parts.queue_manager.ensure_one_ahead(mpd_state='stopped')
    dj_parts.mpd.play()
    dj_parts.queue_manager.ensure_one_ahead(mpd_state='playing')

    current = dj_parts.mpd.get_status()['track_file']
    nxt = dj_parts.queue_manager.get_next_track(current_track=current)

    assert nxt is not None
    assert nxt != current
    assert nxt == dj_parts.mpd.get_queue()[1]


def test_before_playback_the_first_queued_track_is_the_next_track(dj_parts):
    dj_parts.queue_manager.ensure_one_ahead(mpd_state='stopped')

    assert dj_parts.queue_manager.get_next_track(current_track=None) == \
        dj_parts.mpd.get_queue()[0]


def test_the_deleted_queue_api_stays_deleted():
    """D1/D7: the dual bookkeeping and the batch generator are gone for good."""
    from queue_manager import QueueManager

    for name in ('recalculate', 'initialize_queue', 'check_and_refill',
                 '_sync_to_mpd', '_generate_tracks', 'get_upcoming_tracks',
                 'on_track_started'):
        assert not hasattr(QueueManager, name), f"{name} came back"
