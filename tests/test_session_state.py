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
        'tracks_played', 'session_started', 'recent_tracks_count',
        'session_vector_norm',
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


def test_penalize_similar_moves_away_from_the_track(library):
    state = SessionState()
    seed = library.get_embedding(library.track_list[0])
    state.start_session(seed)

    skipped = library.get_embedding(library.track_list[1])
    before = float(np.dot(state.get_session_vector(), skipped))
    state.penalize_similar(skipped)
    after = float(np.dot(state.get_session_vector(), skipped))

    assert after < before
    assert float(np.linalg.norm(state.get_session_vector())) == pytest.approx(1.0)


def test_dead_api_removed():
    """L8."""
    assert not hasattr(SessionState, 'get_recent_average')
    assert not hasattr(SessionState, 'get_similarity')
    assert not hasattr(SessionState, 'reset')
