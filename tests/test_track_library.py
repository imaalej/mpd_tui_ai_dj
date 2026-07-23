"""
Candidate pool behaviour with an unseeded taste model.

Querying `find_similar` with an all-zero vector returns an arbitrary slice of
the library — every dot product is 0, so the ordering is whatever argpartition
happens to produce.  Half the pool would then be noise presented as preference.
"""

import numpy as np

from track_library import TrackLibrary


def test_zero_taste_pool_comes_from_the_session_neighbourhood(library, rng):
    session = rng.standard_normal(library.dimension)
    session /= np.linalg.norm(session)

    pool = library.get_candidate_pool(
        session_vector=session,
        taste_vector=np.zeros(library.dimension),
        pool_size=16,
    )

    assert len(pool) == 16
    expected = {t for t, _ in library.find_similar(session, k=16)}
    assert set(pool) == expected


def test_seeded_taste_contributes_to_the_pool(library, rng):
    """The guard must only fire for a genuinely zero vector."""
    session = rng.standard_normal(library.dimension)
    session /= np.linalg.norm(session)
    # Pick a taste direction far from the session direction so the two
    # neighbourhoods differ.
    taste = -session

    pool = set(library.get_candidate_pool(
        session_vector=session,
        taste_vector=taste,
        pool_size=16,
    ))
    session_only = {t for t, _ in library.find_similar(session, k=16)}

    assert pool != session_only
    assert pool & {t for t, _ in library.find_similar(taste, k=16)}


def test_exclusions_are_honoured(library, rng):
    session = rng.standard_normal(library.dimension)
    session /= np.linalg.norm(session)
    excluded = set(library.track_list[:10])

    pool = library.get_candidate_pool(
        session_vector=session,
        taste_vector=np.zeros(library.dimension),
        exclude_tracks=excluded,
        pool_size=16,
    )

    assert not (set(pool) & excluded)


def test_dummy_embedding_generator_is_gone():
    """M2/M4: random embeddings made every downstream number meaningless."""
    import track_library
    assert not hasattr(track_library, 'generate_dummy_embeddings')
    assert not hasattr(TrackLibrary, 'has_track')
