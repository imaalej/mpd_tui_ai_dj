"""
The test double, under test.

C1 — the queue that never refilled — survived a suite reporting 66 passing
checks, because nothing in it modelled how MPD's queue actually behaves.  A
`FakeMPD` written from the same assumptions would have reproduced the bug and
gone green, so the double is only worth having if it is pinned to *measured*
behaviour.  Every assertion below is one row of the table in PROJECT_AUDIT.md
§M1, verified against MPD 0.24.0 / mpc 0.35 on 23 July 2026.

If one of these ever fails after a `FakeMPD` change, the double drifted from the
real thing and everything built on it is suspect.
"""

import pytest

from conftest import FakeMPD


TRACKS = [f"artist/album/{i:02d}.flac" for i in range(6)]


@pytest.fixture
def mpd():
    """Consume on, everything else off — the state the DJ forces (audit D2)."""
    return FakeMPD(TRACKS, modes={'repeat': 'off', 'random': 'off',
                                  'single': 'off', 'consume': 'on'})


def _queue(mpd, n):
    for track in TRACKS[:n]:
        assert mpd.add_track(track)


# ── the seven rows the audit specifies ───────────────────────────────────────

def test_adding_to_a_stopped_queue_does_not_start_it(mpd):
    """The trap that makes advance-then-add fatal: nothing recovers on its own."""
    _queue(mpd, 4)
    assert mpd.state == 'stopped'
    assert mpd.get_status()['track_file'] is None


def test_the_playing_track_stays_in_the_queue_at_position_one(mpd):
    """Consume removes a track when you *leave* it, not when you start it."""
    _queue(mpd, 4)
    mpd.play()
    assert mpd.get_queue_length() == 4
    assert mpd.get_status()['track_file'] == TRACKS[0]


def test_natural_end_consumes(mpd):
    _queue(mpd, 3)
    mpd.play()
    mpd.finish_track()
    assert mpd.get_queue() == TRACKS[1:3]
    assert mpd.get_status()['track_file'] == TRACKS[1]
    assert mpd.state == 'playing'


def test_next_consumes_too_so_a_skip_and_a_completion_look_identical(mpd):
    _queue(mpd, 4)
    mpd.play()
    mpd.next_track()
    assert mpd.get_queue_length() == 3
    assert mpd.get_status()['track_file'] == TRACKS[1]


def test_deleting_position_two_removes_the_lookahead_and_leaves_playback_alone(mpd):
    """Exactly the primitive the skip path is built on."""
    _queue(mpd, 4)
    mpd.play()
    playing = mpd.get_status()['track_file']

    assert mpd.delete_position(2)

    assert mpd.get_status()['track_file'] == playing
    assert mpd.state == 'playing'
    assert mpd.get_queue() == [TRACKS[0], TRACKS[2], TRACKS[3]]


def test_next_on_the_last_remaining_track_empties_the_queue_and_stops(mpd):
    _queue(mpd, 1)
    mpd.play()
    mpd.next_track()
    assert mpd.get_queue() == []
    assert mpd.state == 'stopped'


def test_add_after_that_does_not_restart_playback(mpd):
    """The second half of the trap — the session is dead until someone plays."""
    _queue(mpd, 1)
    mpd.play()
    mpd.next_track()
    assert mpd.add_track(TRACKS[2])
    assert mpd.state == 'stopped'


# ── the three rows this session measured that the audit did not have ─────────

def test_deleting_a_position_that_does_not_exist_fails_without_changing_anything(mpd):
    """mpc exits 1 with "song number does not exist"; the skip path relies on it."""
    _queue(mpd, 1)
    mpd.play()
    assert mpd.delete_position(2) is False
    assert mpd.get_queue() == TRACKS[:1]


def test_next_while_paused_consumes_and_resumes_playing(mpd):
    """
    Measured, and contrary to the audit's assumption that it stays paused.  The
    application re-pauses afterwards so a skip honours both the rejection and
    the user's play state.
    """
    _queue(mpd, 3)
    mpd.play()
    mpd.pause()
    assert mpd.state == 'paused'

    mpd.next_track()

    assert mpd.state == 'playing'
    assert mpd.get_queue_length() == 2


def test_next_on_a_stopped_player_is_an_error_that_changes_nothing(mpd):
    _queue(mpd, 3)
    assert mpd.state == 'stopped'
    assert mpd.next_track() is False
    assert mpd.get_queue_length() == 3


def test_pause_is_idempotent_not_a_toggle(mpd):
    """Otherwise re-pausing after a paused skip would resume playback instead."""
    _queue(mpd, 2)
    mpd.play()
    mpd.pause()
    mpd.pause()
    mpd.pause()
    assert mpd.state == 'paused'


def test_add_of_an_unknown_path_is_refused(mpd):
    """H7's return code, which at depth 1 is the difference between playing and stalling."""
    assert mpd.add_track("no/such/track.flac") is False
    assert mpd.get_queue() == []


# ── consume off, for contrast: this is the world C1 lived in ─────────────────

def test_with_consume_off_the_queue_only_ever_grows():
    """
    The exact condition behind C1.  `len(playlist)` counts everything already
    played, so a refill check of `len < 3` against a ten-track queue can never
    fire — which is why playback stopped dead after ten tracks and the app's
    own dead-code "restart if stopped" branch never ran either.
    """
    mpd = FakeMPD(TRACKS, modes={'consume': 'off', 'random': 'off',
                                 'repeat': 'off', 'single': 'off'})
    _queue(mpd, 4)
    mpd.play()
    for _ in range(3):
        mpd.finish_track()
    assert mpd.get_queue_length() == 4
