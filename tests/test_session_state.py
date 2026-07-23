"""
Session state after the D4 deletions.

`get_vibe_description()` is gone.  Its mood word came from -Σ|v|·log|v| over a
512-d unit vector, which is always ~55, so it returned "eclectic" for every
session that has ever run and two of its three branches were unreachable
(audit H1).
"""

import numpy as np
import pytest

from session_state import SessionState
from config import config


def test_no_vibe_description():
    assert not hasattr(SessionState, 'get_vibe_description')
    assert not hasattr(SessionState, '_update_vibe_trajectory')


def test_stats_report_no_invented_vocabulary():
    state = SessionState()
    stats = state.get_stats()
    assert 'vibe_description' not in stats
    assert set(stats) == {
        'tracks_played', 'session_started', 'is_seeded', 'recent_tracks_count',
        'skip_run_length', 'session_vector_norm',
    }


def test_no_time_context_attached():
    """D6: SessionState used to own a TimeContext instance."""
    assert not hasattr(SessionState(), 'time_context')


def test_update_follows_the_declared_ema(library):
    """
    session' = normalise(decay * session + (1 - decay) * track).
    Asserted directly so a future refactor cannot quietly change the rule.
    """
    state = SessionState()
    seed = library.get_embedding(library.track_list[0])
    state.start_session(seed)

    track = library.get_embedding(library.track_list[1])
    before = state.get_session_vector()
    state.update(track)

    decay = config.session_decay_factor
    expected = decay * before + (1 - decay) * track
    expected = expected / np.linalg.norm(expected)

    assert np.allclose(state.get_session_vector(), expected)
    assert state.tracks_played == 1


def test_a_skip_moves_the_session_away_from_the_rejected_track(library):
    """
    `penalize_similar`'s fixed 0.15 nudge is replaced by a repulsion whose
    magnitude is solved for a turnover target (audit H9) — but the direction is
    the same, and it is worth asserting separately from the magnitude.
    """
    state = SessionState()
    seed = library.get_embedding(library.track_list[0])
    state.start_session(seed)

    skipped = library.get_embedding(library.track_list[1])
    before = float(np.dot(state.get_session_vector(), skipped))
    state.repel_from_skip_run(skipped, run_length=1,
                              embedding_matrix=library.embedding_matrix)
    after = float(np.dot(state.get_session_vector(), skipped))

    assert after < before
    assert float(np.linalg.norm(state.get_session_vector())) == pytest.approx(1.0)


# ── L7: the session vector starts empty, not random ──────────────────────────

def test_a_fresh_session_vector_is_zero_rather_than_a_random_direction():
    """
    It used to be seeded from `randn` at every startup, so the first queue of
    every session was an arbitrary neighbourhood presented as a vibe.  Zero is
    the honest representation of "nothing has played yet", and the candidate
    pool skips the session half entirely while it holds.
    """
    state = SessionState()
    assert not np.any(state.get_session_vector())
    assert not state.is_seeded()

    state.start_session()
    assert not state.is_seeded()


def test_the_first_track_to_play_becomes_the_session_vector(library):
    state = SessionState()
    state.start_session()
    first = library.get_embedding(library.track_list[3])

    assert state.seed(first) is True
    assert state.is_seeded()
    assert np.allclose(state.get_session_vector(), first)


def test_seeding_is_a_no_op_once_there_is_a_real_vibe(library):
    """Otherwise every track change would overwrite the trajectory the EMA built."""
    state = SessionState()
    state.start_session()
    state.seed(library.get_embedding(library.track_list[3]))
    established = state.get_session_vector()

    assert state.seed(library.get_embedding(library.track_list[9])) is False
    assert np.allclose(state.get_session_vector(), established)


def test_the_first_full_listen_becomes_the_session_rather_than_blending_with_zero(library):
    """
    An EMA against a zero vector would leave the first track at 15% strength and
    the rest of the vector at nothing, which is neither the track nor a vibe.
    """
    state = SessionState()
    state.start_session()
    track = library.get_embedding(library.track_list[2])

    state.update(track)

    assert np.allclose(state.get_session_vector(), track)
    assert state.tracks_played == 1


def test_dead_api_removed():
    """L8, plus the D8 deletions."""
    assert not hasattr(SessionState, 'get_recent_average')
    assert not hasattr(SessionState, 'get_similarity')
    assert not hasattr(SessionState, 'reset')
    assert not hasattr(SessionState, 'force_shift')
    assert not hasattr(SessionState, 'penalize_similar')
    assert not hasattr(SessionState, '_initialize_session_vector')
