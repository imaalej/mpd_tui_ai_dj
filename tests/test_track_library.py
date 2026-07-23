"""
Loading the library, and the candidate pool.

Two Stage 1 changes are guarded here.  **The space is centred on load** (C5) —
every similarity downstream is computed in that space, and doing it anywhere but
here would leave a path that scores on the raw one.  **The keyspace is
reconciled with MPD** (M4) — an embedding MPD cannot play is a track the selector
can pick and `mpc add` will refuse, which at queue depth 1 stalls playback.

The candidate-pool tests below are Stage 0's and cover the unseeded taste
vector: querying `find_similar` with an all-zero vector returns an arbitrary
slice of the library, since every dot product is 0 and the ordering is whatever
argpartition happens to produce.
"""

import numpy as np
import pytest

from embeddings_io import centre, similarity_report
from track_library import LibraryError, TrackLibrary


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


def test_save_embeddings_is_gone():
    """
    M5/L8.  It wrote `track_files` and `embeddings` only — no schema version, no
    centroid, no windows — so a round trip produced a file the loader must now
    refuse, and the vectors it held were already centred and would have been
    centred again on the next load.
    """
    assert not hasattr(TrackLibrary, 'save_embeddings')


# ------------------------------------------------------------------- loading

def test_loading_centres_the_space(make_artifact):
    """
    C5.  The file stores raw, anisotropic vectors; what the library serves must
    be the centred ones, because every scoring formula assumes that range.
    """
    path = make_artifact(n_tracks=200, dimension=64, anisotropy=2.0)
    library = TrackLibrary()
    library.load_embeddings(path)

    with np.load(path, allow_pickle=False) as data:
        raw_mean = similarity_report(data['embeddings'])['mean']

    served_mean = similarity_report(library.embedding_matrix)['mean']
    assert raw_mean > 0.5, "the fixture is supposed to be anisotropic"
    assert abs(served_mean) < 0.05, f"library served an uncentred space ({served_mean})"


def test_the_served_vectors_are_exactly_the_centred_ones(make_artifact):
    path = make_artifact(n_tracks=20, dimension=32, anisotropy=1.5)
    library = TrackLibrary()
    library.load_embeddings(path)

    with np.load(path, allow_pickle=False) as data:
        expected = centre(data['embeddings'], data['centroid'])
        names = [str(t) for t in data['track_files']]

    for name, vector in zip(names, expected):
        assert np.allclose(library.get_embedding(name), vector, atol=1e-6)


def test_the_file_dimension_wins_over_the_config(make_artifact):
    """
    M5: the embeddings are the authority.  Session and taste vectors are sized
    from the config, so it has to follow the file rather than the other way
    round — a mismatch used to surface as a numpy broadcast error deep inside
    the scoring function.
    """
    from config import config

    original = config.embedding_dimension
    try:
        library = TrackLibrary()
        library.load_embeddings(make_artifact(dimension=256))
        assert library.dimension == 256
        assert config.embedding_dimension == 256
    finally:
        config.embedding_dimension = original


def test_a_missing_file_raises_rather_than_loading_an_empty_library(tmp_path):
    """The old loader printed a warning and returned, leaving a library of zero
    tracks that presented as 'the DJ picks nothing'."""
    library = TrackLibrary()
    with pytest.raises(LibraryError, match="No embeddings file"):
        library.load_embeddings(tmp_path / 'absent.npz')


def test_an_old_schema_is_refused(make_artifact):
    library = TrackLibrary()
    with pytest.raises(LibraryError, match="schema version 1"):
        library.load_embeddings(make_artifact(schema_version=1))


def test_a_structurally_broken_file_is_refused(make_artifact):
    library = TrackLibrary()
    with pytest.raises(LibraryError, match="centroid"):
        library.load_embeddings(make_artifact(drop_keys=('centroid',)))


def test_model_identity_is_checked_by_equality_not_substring(make_artifact):
    """
    §0b item 4: a throwaway random-vector file named `SMOKE-TEST-RANDOM-NOT-CLAP`
    was greeted with "✓ Loading CLAP embeddings", because the check was
    `'clap' in model_name.lower()`.
    """
    library = TrackLibrary()
    with pytest.raises(LibraryError, match="SMOKE-TEST-RANDOM-NOT-CLAP"):
        library.load_embeddings(make_artifact(model='SMOKE-TEST-RANDOM-NOT-CLAP'))


def test_a_different_clap_checkpoint_is_refused(make_artifact):
    library = TrackLibrary()
    with pytest.raises(LibraryError, match="clap-htsat-fused"):
        library.load_embeddings(make_artifact(model='laion/clap-htsat-fused'))


def test_the_window_matrix_is_not_read_into_memory(make_artifact):
    """It exists so pooling can be re-decided without regenerating (§9); at
    ~50 MB there is no reason to carry it through every session."""
    library = TrackLibrary()
    library.load_embeddings(make_artifact())
    assert not hasattr(library, 'windows')


# --------------------------------------------------------------- MPD coverage

def _loaded(make_artifact, **kwargs):
    library = TrackLibrary()
    library.load_embeddings(make_artifact(**kwargs))
    return library


def test_tracks_mpd_cannot_play_are_dropped(make_artifact):
    library = _loaded(make_artifact, n_tracks=10)
    mpd_tracks = library.track_list[:8]

    report = library.reconcile_with_mpd(mpd_tracks)

    assert report['matched'] == 8
    assert report['stale'] == 2
    assert library.get_track_count() == 8
    assert set(library.track_list) == set(mpd_tracks)
    assert library.embedding_matrix.shape[0] == 8


def test_mpd_tracks_without_embeddings_are_reported_not_invented(make_artifact):
    library = _loaded(make_artifact, n_tracks=10)
    report = library.reconcile_with_mpd(library.track_list + ['new/track.flac'])

    assert report['unembedded'] == 1
    assert report['stale'] == 0
    assert library.get_track_count() == 10


def test_a_stale_library_refuses_to_start(make_artifact):
    """
    Below the coverage floor the embeddings were generated against a different
    library or music directory, and starting would present a broken keyspace as
    a working DJ.
    """
    library = _loaded(make_artifact, n_tracks=10)
    with pytest.raises(LibraryError, match="Only 20.0%"):
        library.reconcile_with_mpd(library.track_list[:2])


def test_coverage_exactly_at_the_floor_is_accepted(make_artifact):
    library = _loaded(make_artifact, n_tracks=10)
    report = library.reconcile_with_mpd(library.track_list[:5])
    assert report['coverage'] == 0.5


def test_an_empty_mpd_database_does_not_wipe_the_library(make_artifact):
    """`mpc listall` returning nothing means MPD is unhelpful, not that every
    embedding is stale."""
    library = _loaded(make_artifact, n_tracks=10)
    library.reconcile_with_mpd([])
    assert library.get_track_count() == 10


def test_load_reconciles_when_given_the_mpd_track_list(make_artifact):
    path = make_artifact(n_tracks=10)
    with np.load(path, allow_pickle=False) as data:
        names = [str(t) for t in data['track_files']]

    library = TrackLibrary()
    library.load_embeddings(path, mpd_tracks=names[:9])
    assert library.get_track_count() == 9
