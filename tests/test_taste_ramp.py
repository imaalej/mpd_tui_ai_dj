"""
The taste weight is earned, not assumed (audit L7).

`_initialize_taste_vector()` used to return a normalised `randn` vector, and β
carried 0.3 of every scoring decision from the very first track — so a brand-new
listener's "long-term taste" was a random direction in CLAP space with real
influence over selection until dozens of updates accumulated.  At
`taste_update_full_listen = 0.02`, that is a long time.

Stage 0 zeroed the seed.  This is the other half: β ramps in as evidence
accumulates, and the weight it has not earned goes to the session term, because
"what you are listening to right now" is the only other thing known about a new
listener.
"""

import numpy as np
import pytest

from config import config
from exploration_controller import ExplorationController
from user_taste import UserTaste


def test_a_new_listener_gives_the_taste_term_no_weight_at_all():
    weights = ExplorationController().get_weights(taste_updates=0)
    assert weights['taste_weight'] == pytest.approx(0.0)


def test_the_weight_the_taste_term_has_not_earned_goes_to_the_session_term():
    controller = ExplorationController()
    unseeded = controller.get_weights(taste_updates=0)
    seasoned = controller.get_weights(taste_updates=config.taste_ramp_updates)

    assert unseeded['session_weight'] > seasoned['session_weight']
    # Novelty and anti-repetition are untouched by the ramp: it redistributes
    # between the two "what do you like" terms, not away from exploration.
    assert unseeded['novelty_weight'] == pytest.approx(seasoned['novelty_weight'])
    assert unseeded['anti_repetition_weight'] == pytest.approx(
        seasoned['anti_repetition_weight'])


def test_the_weights_still_sum_to_one_throughout_the_ramp():
    """The invariant `config.validate()` enforces must survive the ramp."""
    controller = ExplorationController()
    for updates in range(0, 3 * config.taste_ramp_updates):
        weights = controller.get_weights(taste_updates=updates)
        assert sum(weights.values()) == pytest.approx(1.0)


def test_the_ramp_is_gradual_rather_than_a_cliff_at_the_first_update():
    controller = ExplorationController()
    taste = [controller.get_weights(taste_updates=n)['taste_weight']
             for n in range(0, config.taste_ramp_updates + 1)]

    assert taste == sorted(taste)
    assert taste[1] > 0.0
    assert taste[1] < taste[-1] / 4, "the first update should not buy full weight"


def test_the_ramp_saturates_and_stays_saturated():
    controller = ExplorationController()
    at_full = controller.get_weights(taste_updates=config.taste_ramp_updates)
    far_past = controller.get_weights(taste_updates=10_000)

    assert at_full == far_past
    assert controller.taste_ramp(config.taste_ramp_updates) == 1.0
    assert controller.taste_ramp(10_000) == 1.0


def test_the_default_is_the_conservative_reading():
    """
    A caller that forgets to pass the count should under-weight taste, not
    over-weight it — the failure mode of the old code was a confident wrong
    opinion, and the default should lean the other way.
    """
    controller = ExplorationController()
    assert controller.get_weights() == controller.get_weights(taste_updates=0)


def test_the_ramp_composes_with_exploration_rather_than_being_overridden():
    """
    The `max(0.1, …)` exploration floor would otherwise stop the taste term
    reaching zero — which is exactly the value it should hold with no evidence.
    The ramp is applied after that floor for this reason.
    """
    controller = ExplorationController()
    controller.exploration = config.exploration_max

    weights = controller.get_weights(taste_updates=0)
    assert weights['taste_weight'] == pytest.approx(0.0)
    assert sum(weights.values()) == pytest.approx(1.0)


# ── the ramp is earned from positive evidence, not skips (audit B5) ──────────

def test_skips_do_not_advance_the_beta_ramp(library):
    """
    A listener who only skips has expressed no positive preference, so the β ramp
    must stay at zero — even though `total_updates` climbs with every skip.
    """
    taste = UserTaste(dimension=library.dimension)
    for i in range(10):
        taste.update_from_skip(library.get_embedding(library.track_list[i]))

    assert taste.total_updates == 10        # skips are still counted overall
    assert taste.positive_updates == 0      # but none of it is positive evidence
    assert ExplorationController().taste_ramp(taste.positive_updates) == 0.0


def test_only_likes_and_full_listens_are_positive_evidence(library):
    taste = UserTaste(dimension=library.dimension)
    taste.update_from_like(library.get_embedding(library.track_list[0]))
    taste.update_from_full_listen(library.get_embedding(library.track_list[1]))
    taste.update_from_skip(library.get_embedding(library.track_list[2]))

    assert taste.positive_updates == 2
    assert taste.total_updates == 3
    assert taste.get_stats()['positive_updates'] == 2


def test_a_skip_run_before_a_like_does_not_pre_earn_beta(library):
    """
    The concrete misbehaviour: skip nine times, then like once.  With the ramp
    reading `total_updates` β would already be near full (10/20); reading
    positive evidence it is 1/20, as a single like should buy.
    """
    taste = UserTaste(dimension=library.dimension)
    for i in range(9):
        taste.update_from_skip(library.get_embedding(library.track_list[i]))
    taste.update_from_like(library.get_embedding(library.track_list[9]))

    controller = ExplorationController()
    assert taste.positive_updates == 1
    assert controller.taste_ramp(taste.positive_updates) == pytest.approx(
        1 / config.taste_ramp_updates)


# ── the guard the ramp does NOT retire ───────────────────────────────────────

def test_a_lone_skip_still_cannot_seed_the_taste_model(library):
    """
    Stage 2's plan says to retire this guard once the ramp lands.  It is kept,
    because retiring it reintroduces in *retrieval* the defect it was added to
    prevent in *scoring*.

    From zero, one negative update normalises to `−track` at unit length: a
    full-strength "your taste is the exact opposite of this song" from a single
    rejection.  β would damp its effect on the *score* — but β never gates which
    candidates are retrieved, and `get_candidate_pool` opens its taste half on
    `np.any(taste_vector)`.  So a lone skip would hand half the pool to "the
    tracks least like the one thing you rejected", at full strength, on press
    one.  The ramp cannot reach that, so the guard stays.
    """
    taste = UserTaste(dimension=library.dimension)
    track = library.get_embedding(library.track_list[0])

    taste.update_from_skip(track)

    assert not taste.is_seeded()
    assert taste.total_updates == 1
    assert taste.skip_count == 1


def test_a_positive_signal_seeds_it_and_then_skips_bite(library):
    taste = UserTaste(dimension=library.dimension)
    liked = library.get_embedding(library.track_list[0])
    disliked = library.get_embedding(library.track_list[1])

    taste.update_from_like(liked)
    assert taste.is_seeded()

    before = float(np.dot(taste.get_taste_vector(), disliked))
    taste.update_from_skip(disliked)
    after = float(np.dot(taste.get_taste_vector(), disliked))

    assert after < before
