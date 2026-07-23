"""
Rank-Boltzmann selection (audit H6).

Selection was a strict argmax: `sorted(candidates)[0]`.  The exploration scalar
only reshuffled the *weights* of that argmax, so raising it produced a different
deterministic answer rather than a more varied one — "exploration" never
explored.

At queue depth 10 this was masked: the within-batch exclusion set forced ten
distinct tracks, so variety came out of the bookkeeping.  At depth 1 the mask is
gone and each pick is an independent argmax over a pool that barely moved, so
D1 could not ship without this.

The rule is `p(i) ∝ exp(−i/τ)` over **rank**, never over score.  A score-softmax
would need τ recalibrated to the width of the score distribution, and that width
moves whenever the weights are reshuffled, whenever β ramps in, and whenever the
embedding space is re-centred.  Rank is scale-invariant, so τ never moves.
"""

import numpy as np
import pytest

from config import config
from track_selector import TrackSelector


# ── the temperature map ──────────────────────────────────────────────────────

def test_tau_spans_its_range_across_the_exploration_bounds(library):
    selector = TrackSelector(library)

    assert selector.temperature(config.exploration_min) == pytest.approx(config.tau_min)
    assert selector.temperature(config.exploration_max) == pytest.approx(config.tau_max)


def test_tau_is_monotone_in_exploration(library):
    selector = TrackSelector(library)
    taus = [selector.temperature(e) for e in np.linspace(0.1, 0.7, 13)]
    assert taus == sorted(taus)


def test_tau_is_clamped_outside_the_bounds(library):
    """The exploration scalar is bounded, but τ must not go negative regardless."""
    selector = TrackSelector(library)
    assert selector.temperature(-1.0) >= config.tau_min
    assert selector.temperature(99.0) == pytest.approx(config.tau_max)


@pytest.mark.parametrize('exploration,expected_p0', [
    (0.1, 0.63),    # floor    — near-argmax, the best track usually wins
    (0.4, 0.13),    # mid      — genuine spread over the top ~15
    (0.7, 0.06),    # ceiling  — broad; the top ~40 are all live
])
def test_the_documented_top_rank_probabilities_are_what_the_map_produces(
        library, exploration, expected_p0):
    """
    The table in H6 is a claim about behaviour that a reader can check.  These
    are the numbers it quotes, recomputed from the shipped map.
    """
    selector = TrackSelector(library)
    tau = selector.temperature(exploration)

    weights = np.exp(-np.arange(100) / tau)
    weights /= weights.sum()

    assert float(weights[0]) == pytest.approx(expected_p0, abs=0.01)


# ── the sampler ──────────────────────────────────────────────────────────────

def test_the_rank_distribution_matches_the_declared_law(library):
    """Sampled empirically and compared against exp(−i/τ), not eyeballed."""
    selector = TrackSelector(library, rng=np.random.default_rng(7))
    tau = 5.0
    n = 50

    draws = np.array([selector._sample_rank(n, tau) for _ in range(40000)])
    observed = np.bincount(draws, minlength=n) / len(draws)

    expected = np.exp(-np.arange(n) / tau)
    expected /= expected.sum()

    assert np.max(np.abs(observed - expected)) < 0.01


def test_low_tau_concentrates_on_the_top_and_high_tau_spreads(library):
    selector = TrackSelector(library, rng=np.random.default_rng(7))

    tight = [selector._sample_rank(100, 1.0) for _ in range(2000)]
    loose = [selector._sample_rank(100, 15.0) for _ in range(2000)]

    assert np.mean(tight) < np.mean(loose)
    assert np.mean(tight) == pytest.approx(0.58, abs=0.15)


def test_a_tiny_candidate_list_is_drawn_uniformly(library):
    """
    With two or three candidates left, τ would decide the outcome and the
    "choice" would be a formality.  Below the floor the sampler stops pretending.
    """
    selector = TrackSelector(library, rng=np.random.default_rng(7))
    n = config.minimum_sampled_pool - 1

    draws = [selector._sample_rank(n, tau=1.0) for _ in range(6000)]
    counts = np.bincount(draws, minlength=n) / len(draws)

    assert np.max(np.abs(counts - 1.0 / n)) < 0.03


def test_a_single_candidate_is_returned_without_ceremony(library):
    selector = TrackSelector(library)
    assert selector._sample_rank(1, tau=10.0) == 0


# ── what it buys ─────────────────────────────────────────────────────────────

def _session(library, seed, n_tracks=30):
    """Play n tracks from one fixed starting state."""
    session_vector = library.get_embedding(library.track_list[0]).copy()
    selector = TrackSelector(library, rng=np.random.default_rng(seed))
    zero_taste = np.zeros(library.dimension)
    weights = {'session_weight': 0.6, 'taste_weight': 0.0,
               'novelty_weight': 0.3, 'anti_repetition_weight': 0.1}

    picked = []
    for _ in range(n_tracks):
        track = selector.select_track(
            session_vector=session_vector, taste_vector=zero_taste,
            weights=weights, exploration=0.4)
        if track is None:
            break
        picked.append(track)
    return picked


def test_two_sessions_from_the_same_state_are_materially_different(library):
    """
    The stage's done-criterion, and the entire gain.  Argmax returns the
    byte-identical run from a given state every time — for a system whose premise
    is an evolving session, reproducing the same evening is the failure mode.
    """
    runs = [_session(library, seed) for seed in (1, 2, 3, 4, 5)]

    assert len({tuple(r) for r in runs}) == len(runs), "sampling produced identical runs"

    baseline = set(runs[0])
    overlaps = [len(baseline & set(r)) / len(baseline) for r in runs[1:]]
    assert max(overlaps) < 0.95


def test_selection_is_reproducible_for_a_given_seed(library):
    """
    Sampling must not mean untestable.  The generator is injected, so the same
    seed gives the same session — which is what lets the test above assert that
    *different* seeds do not.
    """
    assert _session(library, 11) == _session(library, 11)


def test_the_selector_reports_the_rank_it_actually_drew(library):
    """
    For the [I] inspector.  "Choosing from ~top 8" is only worth showing if it
    is derived from what happened rather than from what was configured.
    """
    selector = TrackSelector(library, rng=np.random.default_rng(3))
    session_vector = library.get_embedding(library.track_list[0])

    selector.select_track(
        session_vector=session_vector,
        taste_vector=np.zeros(library.dimension),
        weights={'session_weight': 1.0, 'taste_weight': 0.0,
                 'novelty_weight': 0.0, 'anti_repetition_weight': 0.0},
        exploration=0.4)

    assert selector.last_tau == pytest.approx(selector.temperature(0.4))
    assert 0 <= selector.last_rank < selector.last_pool_size


def test_select_track_does_not_mutate_the_callers_exclusion_set(library):
    """L9: it used to `update()` the set it was handed."""
    selector = TrackSelector(library)
    exclusions = {library.track_list[0]}
    selector.recent_history.append(library.track_list[1])

    selector.select_track(
        session_vector=library.get_embedding(library.track_list[2]),
        taste_vector=np.zeros(library.dimension),
        weights={'session_weight': 1.0, 'taste_weight': 0.0,
                 'novelty_weight': 0.0, 'anti_repetition_weight': 0.0},
        exploration=0.3,
        exclude_tracks=exclusions)

    assert exclusions == {library.track_list[0]}


def test_no_evidence_yields_a_uniform_draw_rather_than_an_arbitrary_ordering(library):
    """
    L7.  With both vectors at zero, querying `find_similar` would give every
    track a dot product of 0 and return whatever `argpartition` happened to
    order — an arbitrary slice of the library, presented as a preference.  The
    honest answer to "no information" is a coin toss.
    """
    zero = np.zeros(library.dimension)
    picks = set()
    for seed in range(30):
        selector = TrackSelector(library, rng=np.random.default_rng(seed))
        picks.add(selector.select_track(
            session_vector=zero, taste_vector=zero,
            weights={'session_weight': 0.4, 'taste_weight': 0.3,
                     'novelty_weight': 0.2, 'anti_repetition_weight': 0.1},
            exploration=0.3))

    assert len(picks) > 10, "the unseeded first pick is not actually uniform"
    assert None not in picks
