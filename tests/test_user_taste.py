"""
The taste model starts from no evidence, not from a random direction (audit
D4 / L7).

The old seed was `normalise(randn(512) * 0.01)` — a unit vector pointing
somewhere arbitrary in CLAP space, carrying β = 0.3 of the score from the very
first track.  That is not a weak opinion; it is a confident wrong one.
"""

import numpy as np
import pytest

from user_taste import UserTaste


def test_fresh_taste_vector_is_zero():
    taste = UserTaste()
    assert np.count_nonzero(taste.get_taste_vector()) == 0
    assert taste.is_seeded() is False


def test_two_fresh_models_are_identical():
    """A random seed made every new user different for no reason."""
    assert np.array_equal(UserTaste().get_taste_vector(),
                          UserTaste().get_taste_vector())


def test_like_seeds_the_model_to_that_track(library):
    taste = UserTaste()
    track = library.track_list[0]
    emb = library.get_embedding(track)

    taste.update_from_like(emb)

    assert taste.is_seeded()
    # From zero, (1-w)*0 + w*emb normalises back to emb exactly.
    assert np.allclose(taste.get_taste_vector(), emb)
    assert float(np.linalg.norm(taste.get_taste_vector())) == pytest.approx(1.0)


def test_skip_on_an_unseeded_model_leaves_it_unseeded(library):
    """
    Subtracting from zero and re-normalising would turn a single skip into
    "your taste is the exact opposite of this track" — a full-strength claim
    from one rejection.  The counter still moves; the vector does not.
    """
    taste = UserTaste()
    taste.update_from_skip(library.get_embedding(library.track_list[0]))

    assert taste.is_seeded() is False
    assert np.count_nonzero(taste.get_taste_vector()) == 0
    assert taste.skip_count == 1
    assert taste.total_updates == 1


def test_skip_on_a_seeded_model_moves_away_from_the_track(library):
    taste = UserTaste()
    liked = library.get_embedding(library.track_list[0])
    taste.update_from_like(liked)

    disliked = library.get_embedding(library.track_list[1])
    before = float(np.dot(taste.get_taste_vector(), disliked))
    taste.update_from_skip(disliked)
    after = float(np.dot(taste.get_taste_vector(), disliked))

    assert after < before


def test_save_load_round_trip_of_an_unseeded_model(tmp_path):
    """An all-zero vector must survive load without being 'fixed' to garbage."""
    path = tmp_path / "user_taste.npz"
    UserTaste().save(path)

    reloaded = UserTaste()
    assert reloaded.load(path) is True
    assert np.count_nonzero(reloaded.get_taste_vector()) == 0
    assert reloaded.is_seeded() is False


def test_dead_api_removed():
    """L8: never called anywhere in the running system."""
    assert not hasattr(UserTaste, 'get_similarity')
    assert not hasattr(UserTaste, 'reset')
