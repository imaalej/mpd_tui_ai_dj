"""
Scoring is exactly the four weighted terms and nothing else (audit D6 / L9).

The time-context bonus was added *after* the weights had been normalised to sum
to 1.0, so the score silently exceeded its nominal range and the fifth term's
influence depended on how the other four happened to land.
"""

import numpy as np
import pytest

from track_selector import TrackSelector


def _weights():
    return {
        'session_weight': 0.4,
        'taste_weight': 0.3,
        'novelty_weight': 0.2,
        'anti_repetition_weight': 0.1,
    }


def test_score_equals_the_declared_weighted_sum(library, rng):
    """Recompute the formula by hand and require an exact match."""
    selector = TrackSelector(library)
    track = library.track_list[0]
    emb = library.get_embedding(track)

    session = rng.standard_normal(library.dimension)
    session /= np.linalg.norm(session)
    taste = rng.standard_normal(library.dimension)
    taste /= np.linalg.norm(taste)

    w = _weights()
    score = selector._calculate_score(
        track_embedding=emb,
        track_file=track,
        session_vector=session,
        taste_vector=taste,
        weights=w,
    )

    # No history yet, so novelty is 1.0 and anti-repetition is 1.0 by definition.
    expected = (w['session_weight'] * (float(np.dot(session, emb)) + 1) / 2
                + w['taste_weight'] * (float(np.dot(taste, emb)) + 1) / 2
                + w['novelty_weight'] * 1.0
                + w['anti_repetition_weight'] * 1.0)

    assert score == pytest.approx(expected)


def test_score_stays_within_unit_range(library, rng):
    """
    With weights summing to 1.0 and every term in [0, 1], the score is in
    [0, 1].  This was not true while the time bonus was bolted on.
    """
    selector = TrackSelector(library)
    session = rng.standard_normal(library.dimension)
    session /= np.linalg.norm(session)
    taste = np.zeros(library.dimension)

    for track in library.track_list:
        score = selector._calculate_score(
            track_embedding=library.get_embedding(track),
            track_file=track,
            session_vector=session,
            taste_vector=taste,
            weights=_weights(),
        )
        assert 0.0 <= score <= 1.0


def test_calculate_score_takes_no_time_context_argument():
    import inspect
    params = inspect.signature(TrackSelector._calculate_score).parameters
    assert 'time_context' not in params
    params = inspect.signature(TrackSelector.select_track).parameters
    assert 'time_context' not in params


def test_zero_taste_vector_cannot_reorder_candidates(library, rng):
    """
    L7: an unseeded taste model must be inert.  A zero vector gives every
    candidate the same taste term, so the ranking is identical to one produced
    with the taste weight moved entirely onto the session term.
    """
    selector = TrackSelector(library)
    session = rng.standard_normal(library.dimension)
    session /= np.linalg.norm(session)
    zero_taste = np.zeros(library.dimension)

    with_taste = _weights()
    without_taste = {**with_taste,
                     'session_weight': with_taste['session_weight'] + with_taste['taste_weight'],
                     'taste_weight': 0.0}

    def ranking(weights):
        scored = [(t, selector._calculate_score(
            track_embedding=library.get_embedding(t),
            track_file=t,
            session_vector=session,
            taste_vector=zero_taste,
            weights=weights)) for t in library.track_list]
        # Sort by score only; ties would make the comparison meaningless, and
        # random embeddings do not produce them.
        return [t for t, _ in sorted(scored, key=lambda x: -x[1])]

    assert ranking(with_taste) == ranking(without_taste)
