"""
The embeddings artifact: schema validation (M5) and centring (C5).

M5 exists because the old loader decided a file was CLAP by asking whether the
string 'clap' appeared somewhere in its model name, and never looked at the
dimension at all.  The schema below is the contract between the generation run
and everything downstream, so each way of breaking it gets its own test.
"""


import numpy as np
import pytest

from embeddings_io import (
    REQUIRED_KEYS,
    SCHEMA_VERSION,
    centre,
    load_metadata,
    similarity_report,
    validate_embeddings,
    write_failed_tracks,
)


# ---------------------------------------------------------------- validation

def test_a_well_formed_artifact_validates(make_artifact):
    result = validate_embeddings(make_artifact())
    assert result['valid'], result.get('error')
    assert result['schema_version'] == SCHEMA_VERSION
    assert result['n_tracks'] == 12
    assert result['dimension'] == 512
    assert result['n_windows'] == 36


@pytest.mark.parametrize("key", REQUIRED_KEYS)
def test_every_required_key_is_actually_required(make_artifact, key):
    result = validate_embeddings(make_artifact(drop_keys=(key,)))
    assert not result['valid']
    assert key in result['error']


def test_schema_one_is_refused_because_it_has_no_centroid(make_artifact):
    """
    The whole point of the version bump.  A v1 file would load and score
    perfectly happily on an uncentred space — silently wrong, which is worse
    than an error (C5).
    """
    result = validate_embeddings(make_artifact(schema_version=1))
    assert not result['valid']
    assert 'schema version 1' in result['error']


def test_track_count_must_match_embedding_count(make_artifact):
    path = make_artifact(track_files=[f"{i}.flac" for i in range(5)], n_tracks=12)
    result = validate_embeddings(path)
    assert not result['valid'] and '5 track names' in result['error']


def test_window_offsets_must_end_where_the_windows_do(make_artifact):
    bad_offsets = np.arange(13, dtype=np.int32) * 2      # claims 24, file has 36
    result = validate_embeddings(make_artifact(window_offsets=bad_offsets))
    assert not result['valid'] and 'window_offsets ends at' in result['error']


def test_window_offsets_length_must_match_the_track_count(make_artifact):
    result = validate_embeddings(make_artifact(window_offsets=np.zeros(4, dtype=np.int32)))
    assert not result['valid'] and 'window_offsets has 4 entries' in result['error']


def test_windows_must_share_the_embedding_dimension(make_artifact):
    result = validate_embeddings(make_artifact(windows=np.zeros((36, 128), np.float32)))
    assert not result['valid'] and '128-d' in result['error']


def test_unnormalised_embeddings_are_refused(make_artifact):
    result = validate_embeddings(make_artifact(embeddings=np.full((12, 512), 0.5, np.float32)))
    assert not result['valid'] and 'not normalised' in result['error']


def test_centroid_shape_is_checked(make_artifact):
    result = validate_embeddings(make_artifact(centroid=np.zeros(8, np.float32)))
    assert not result['valid'] and 'centroid is' in result['error']


def test_a_missing_file_is_an_error_not_an_exception(tmp_path):
    result = validate_embeddings(tmp_path / 'nothing.npz')
    assert not result['valid'] and 'failed to validate' in result['error']


def test_the_artifact_loads_without_pickle(make_artifact):
    """Nothing in the file may require executing a pickle to read it."""
    with np.load(make_artifact(), allow_pickle=False) as data:
        assert set(REQUIRED_KEYS) <= set(data.keys())


def test_metadata_round_trips_as_json(make_artifact):
    with np.load(make_artifact(model='some/model'), allow_pickle=False) as data:
        assert load_metadata(data)['model'] == 'some/model'


def test_missing_metadata_reads_as_empty_rather_than_raising(make_artifact):
    with np.load(make_artifact(drop_keys=('metadata',)), allow_pickle=False) as data:
        assert load_metadata(data) == {}


# ------------------------------------------------------------------ centring

def test_centring_removes_the_anisotropy(rng):
    """
    C5's measured claim: two unrelated tracks of the real library sat at mean
    cosine 0.577 raw.  Centring must move that to ~0 — otherwise `(sim + 1) / 2`
    and `novelty = (1 - max_sim) / 2` are calibrated for a range that does not
    exist.
    """
    bias = rng.standard_normal(64)
    bias /= np.linalg.norm(bias)
    raw = rng.standard_normal((300, 64))
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    raw = raw + 1.5 * bias                            # a narrow cone, like CLAP's
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)

    before = similarity_report(raw)['mean']
    after = similarity_report(centre(raw, raw.mean(axis=0)))['mean']

    assert before > 0.5, "the fixture is supposed to be anisotropic"
    assert abs(after) < 0.05, f"centred pairs should sit near 0, got {after}"


def test_centred_vectors_are_unit_length(rng):
    raw = rng.standard_normal((32, 16))
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    centred = centre(raw, raw.mean(axis=0))
    assert np.allclose(np.linalg.norm(centred, axis=1), 1.0, atol=1e-5)


def test_centring_a_single_vector_matches_the_matrix_form(rng):
    raw = rng.standard_normal((10, 16)).astype(np.float32)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    centroid = raw.mean(axis=0)
    assert np.allclose(centre(raw[3], centroid), centre(raw, centroid)[3], atol=1e-6)


def test_centring_is_stable_when_a_vector_sits_on_the_centroid():
    """A vector identical to the centroid has no direction left; it must not
    become NaN and poison every dot product downstream."""
    embeddings = np.tile(np.eye(1, 8, dtype=np.float32), (4, 1))
    centred = centre(embeddings, embeddings.mean(axis=0))
    assert np.isfinite(centred).all()


# ------------------------------------------------------------------ failures

def test_failed_tracks_are_written_with_their_exception(tmp_path):
    path = tmp_path / 'failed.txt'
    write_failed_tracks([('a/b.flac', 'RuntimeError: bad frame')], path)

    body = path.read_text()
    assert 'a/b.flac' in body and 'RuntimeError: bad frame' in body


def test_a_clean_run_removes_a_stale_failure_list(tmp_path):
    path = tmp_path / 'failed.txt'
    path.write_text('old/failure.flac\tsomething\n')
    write_failed_tracks([], path)
    assert not path.exists()
