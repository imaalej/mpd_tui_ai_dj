"""
Skip escalation against the **real** library (audit H9).

Whether a turnover target is reachable at all is a property of the collection's
structure, not of the solver — over a stand-in library with no clusters, a
repulsion merely rotates a diffuse shell and turnover saturates well below what a
real one reaches.  So the schedule's claims are asserted here, against
`data/embeddings/`, and the file skips if it is absent.  The solver's own
contract is checked on synthetic data in `test_manifold.py`.

This is the evidence for two statements the user reads:

  • "n consecutive skips turn over progressively more of what you would have
    heard, reaching a full reset by the fourth."
  • "however hard you push, the session vector stays inside your music" — the
    thing the deleted `[V]` could not promise, since it landed halfway to noise.
"""

import numpy as np
import pytest

import manifold as M
from config import config
from session_state import SessionState
from track_library import TrackLibrary


pytestmark = pytest.mark.skipif(
    not config.embeddings_file.exists(),
    reason="needs the built embeddings; run generate_embeddings.py")


@pytest.fixture(scope='module')
def real_library():
    lib = TrackLibrary()
    lib.load_embeddings()
    return lib


@pytest.fixture(scope='module')
def matrix(real_library):
    return real_library.embedding_matrix


def _converged_session(matrix, rng, n_tracks=12):
    """
    A session vector that has settled, and the tracks that shaped it.

    Convergence matters: a vector one track old sits loosely in its
    neighbourhood and a nudge moves a lot of the pool, while after a dozen
    tracks it locks on hard.  Measuring on a fresh vector overstates every
    skip's effect, which is part of why the pre-Stage-2 figures were misleading.
    """
    vector = M.normalise(rng.standard_normal(matrix.shape[1]))
    played = []
    for _ in range(n_tracks):
        pick = next(i for i in np.argsort(matrix @ vector)[::-1] if i not in played)
        played.append(pick)
        vector = M.normalise(config.session_decay_factor * vector
                             + (1 - config.session_decay_factor) * matrix[pick])
    return vector, played


# ── the schedule, end to end ─────────────────────────────────────────────────

def test_a_run_of_skips_escalates_turnover_monotonically(matrix):
    """
    The claim the escalation exists to make.  Each press is measured against
    where the vector stood when the run *began*, which is what "how much has
    changed since I started skipping" means to a listener.
    """
    rng = np.random.default_rng(20260723)
    per_press = {1: [], 2: [], 3: [], 4: []}

    for _ in range(12):
        vector, _ = _converged_session(matrix, rng)
        state = SessionState(dimension=matrix.shape[1])
        state.session_vector = vector.copy()

        for press in range(1, 5):
            rejected = matrix[int(np.argmax(matrix @ state.session_vector))]
            outcome = state.repel_from_skip_run(rejected, press, matrix)
            per_press[press].append(outcome['turnover'])

    medians = [float(np.median(per_press[p])) for p in (1, 2, 3, 4)]

    assert medians == sorted(medians), f"escalation is not monotone: {medians}"
    assert medians[0] < 0.15, f"one skip should be a small correction, got {medians[0]:.0%}"
    assert medians[3] > 0.80, (
        f"four consecutive skips must be a full reset, got {medians[3]:.0%}")


def test_no_press_in_a_run_ever_undoes_the_one_before_it(matrix):
    """
    Regression for a defect the medians hid and a live run exposed.

    λ was originally solved against the *un-snapped* vector and the snap applied
    afterwards.  But the snap relocates to a 25-track centroid, and for a modest
    λ those 25 tracks are largely the neighbourhood the vector started in — so a
    second consecutive skip could land back where the run began.  Observed live:
    "Skip #2: 1% of what you would have heard is now different (target 20%)",
    immediately after Skip #1 had achieved 5%.

    The fix is to solve for the turnover of the vector that actually selects, so
    the snap is inside the objective rather than applied to its result.  This
    asserts the property that failure violated: within a run, turnover measured
    against the run's start never decreases.
    """
    rng = np.random.default_rng(20260723)

    for _ in range(15):
        vector, _ = _converged_session(matrix, rng)
        state = SessionState(dimension=matrix.shape[1])
        state.session_vector = vector.copy()

        seen = []
        for press in range(1, 6):
            rejected = matrix[int(np.argmax(matrix @ state.session_vector))]
            seen.append(state.repel_from_skip_run(rejected, press, matrix)['turnover'])

        assert seen == sorted(seen), f"a skip undid the one before it: {seen}"


def test_every_press_meets_the_target_its_run_length_asks_for(matrix):
    """
    The schedule is a floor stated in units the listener can verify, and λ is
    solved until it is met.  Reported turnover is always the measured value, so
    overshoot is honest; falling short would not be.
    """
    rng = np.random.default_rng(20260723)
    schedule = config.skip_turnover_schedule
    shortfalls = []

    for _ in range(15):
        vector, _ = _converged_session(matrix, rng)
        state = SessionState(dimension=matrix.shape[1])
        state.session_vector = vector.copy()

        for press in range(1, len(schedule) + 1):
            rejected = matrix[int(np.argmax(matrix @ state.session_vector))]
            outcome = state.repel_from_skip_run(rejected, press, matrix)
            target = schedule[min(press, len(schedule)) - 1]
            if outcome['turnover'] < target:
                shortfalls.append((press, target, outcome['turnover']))

    assert not shortfalls, f"targets missed: {shortfalls}"


def test_a_single_skip_hits_its_modest_target(matrix):
    """
    The first press targets 5%: "not this song", not "not this direction".  The
    old fixed λ = 0.15 moved about 3% and never escalated, which is why twenty
    presses were needed before anything changed.
    """
    rng = np.random.default_rng(20260723)
    achieved = []

    for _ in range(12):
        vector, _ = _converged_session(matrix, rng)
        state = SessionState(dimension=matrix.shape[1])
        state.session_vector = vector.copy()
        rejected = matrix[int(np.argmax(matrix @ vector))]
        achieved.append(state.repel_from_skip_run(rejected, 1, matrix)['turnover'])

    target = config.skip_turnover_schedule[0]
    assert float(np.median(achieved)) >= target
    assert float(np.median(achieved)) < 4 * target, "one skip overshot its target"


def test_the_session_vector_never_leaves_the_manifold_however_long_the_run(matrix):
    """
    H9's structural guarantee, and the argument that killed `[V]`.

    The bar is the library's **own** distribution rather than an invented
    threshold.  On-manifold quality varies across real music — a track in a
    sparse corner of the collection scores 0.43 while one in a dense cluster
    scores 0.96 — so "still music" cannot mean "above some fixed number".  It
    means: no worse than the least typical real track.

    That is a strong claim.  `[V]` failed it: blending half a random direction in
    landed the vector around 0.56, and the candidate pool was then drawn around a
    point substantially outside the collection.
    """
    rng = np.random.default_rng(20260723)

    # The library's own floor, measured over every track rather than sampled.
    real_quality = np.array([M.on_manifold_quality(matrix, row) for row in matrix])
    floor = float(np.percentile(real_quality, 1))

    worst = 1.0
    for _ in range(8):
        vector, _ = _converged_session(matrix, rng)
        state = SessionState(dimension=matrix.shape[1])
        state.session_vector = vector.copy()

        for press in range(1, 9):
            rejected = matrix[int(np.argmax(matrix @ state.session_vector))]
            outcome = state.repel_from_skip_run(rejected, press, matrix)
            worst = min(worst, outcome['quality'])

    assert worst >= floor, (
        f"a skip run left the manifold: worst quality {worst:.3f}, below the "
        f"1st percentile of real tracks ({floor:.3f})")


def _quality_floor(matrix):
    """The library's own 1st percentile of on-manifold quality."""
    return float(np.percentile(
        [M.on_manifold_quality(matrix, row) for row in matrix], 1))


def test_the_snap_is_what_holds_the_manifold_at_the_escalated_targets(matrix):
    """
    Why the snap is not optional above the first press — the test above shows the
    guarantee holds, not that anything is holding it.

    Solved for 85% turnover and *not* projected back, the vector lands below
    where any real track sits, every single time: precisely the defect `[V]` was
    deleted for.  The snap pulls the same λ back inside the collection.
    """
    rng = np.random.default_rng(20260723)
    floor = _quality_floor(matrix)

    for _ in range(10):
        vector, played = _converged_session(matrix, rng)
        away = M.normalise(matrix[played[-3:]].mean(axis=0))

        _, _, raw = M.solve_repulsion(matrix, vector, away, 0.85,
                                      k=config.candidate_pool_size)

        assert M.on_manifold_quality(matrix, raw) < floor, (
            "an unguarded full-reset repulsion stayed on-manifold — if that is "
            "now generally true, the snap gate deserves re-deriving")
        assert M.on_manifold_quality(matrix, M.snap(matrix, raw)) >= floor


@pytest.mark.parametrize('target', [0.05, 0.20, 0.50, 0.85])
def test_snapping_improves_on_manifold_quality_at_every_target(matrix, target):
    rng = np.random.default_rng(20260723)

    for _ in range(6):
        vector, played = _converged_session(matrix, rng)
        away = M.normalise(matrix[played[-3:]].mean(axis=0))
        _, _, raw = M.solve_repulsion(matrix, vector, away, target,
                                      k=config.candidate_pool_size)

        assert M.on_manifold_quality(matrix, M.snap(matrix, raw)) > \
            M.on_manifold_quality(matrix, raw)


def test_a_single_skip_does_not_need_the_snap(matrix):
    """
    The gate's lower half, measured rather than assumed.  At the 5% target the
    unguarded vector is comfortably inside the collection, so snapping would be
    a displacement bought for nothing — and snap has a turnover floor of its own
    that would overshoot the schedule several times over.
    """
    rng = np.random.default_rng(20260723)
    floor = _quality_floor(matrix)

    for _ in range(10):
        vector, played = _converged_session(matrix, rng)
        away = M.normalise(matrix[played[-1:]].mean(axis=0))
        _, _, raw = M.solve_repulsion(matrix, vector, away,
                                      config.skip_turnover_schedule[0],
                                      k=config.candidate_pool_size)

        assert M.on_manifold_quality(matrix, raw) > floor


def test_snap_is_gated_off_for_the_first_skip(matrix):
    """
    The other side of the gate.  snap() is a move in its own right with a
    turnover floor of its own, so applying it to a single skip would overshoot
    the 5% target several times over.
    """
    rng = np.random.default_rng(20260723)
    vector, _ = _converged_session(matrix, rng)

    state = SessionState(dimension=matrix.shape[1])
    state.session_vector = vector.copy()
    rejected = matrix[int(np.argmax(matrix @ vector))]

    first = state.repel_from_skip_run(rejected, 1, matrix)
    assert first['snapped'] is False

    state.clear_skip_run()
    state.session_vector = vector.copy()
    second = state.repel_from_skip_run(rejected, 2, matrix)
    assert second['snapped'] is True


def test_lambda_is_solved_not_declared(matrix):
    """
    Different run lengths must produce different magnitudes, derived from the
    turnover schedule.  A fixed λ is precisely what H9 replaces.
    """
    rng = np.random.default_rng(20260723)
    vector, _ = _converged_session(matrix, rng)
    rejected = matrix[int(np.argmax(matrix @ vector))]

    lambdas = []
    for press in (1, 2, 3, 4):
        state = SessionState(dimension=matrix.shape[1])
        state.session_vector = vector.copy()
        lambdas.append(state.repel_from_skip_run(rejected, press, matrix)['lambda'])

    assert len(set(lambdas)) > 1, f"λ did not respond to the schedule: {lambdas}"
    assert lambdas == sorted(lambdas), lambdas


# ── run bookkeeping ──────────────────────────────────────────────────────────

def test_a_full_listen_ends_the_skip_run(matrix):
    """Escalation decays the moment you stop skipping — no separate cooldown."""
    rng = np.random.default_rng(20260723)
    vector, _ = _converged_session(matrix, rng)
    state = SessionState(dimension=matrix.shape[1])
    state.session_vector = vector.copy()

    state.repel_from_skip_run(matrix[0], 1, matrix)
    state.repel_from_skip_run(matrix[1], 2, matrix)
    assert len(state.skip_run) == 2

    state.update(matrix[5])

    assert state.skip_run == []
    assert state.skip_run_anchor is None


def test_the_repulsion_uses_the_whole_run_not_just_the_last_track(matrix):
    """
    More evidence, and less sensitivity to one atypical song — which is the
    reason for repelling from a centroid rather than from a single embedding.
    """
    rng = np.random.default_rng(20260723)
    vector, _ = _converged_session(matrix, rng)
    state = SessionState(dimension=matrix.shape[1])
    state.session_vector = vector.copy()

    state.repel_from_skip_run(matrix[10], 1, matrix)
    state.repel_from_skip_run(matrix[200], 2, matrix)

    assert len(state.skip_run) == 2
    assert np.allclose(state.skip_run[0], M.normalise(matrix[10]))
    assert np.allclose(state.skip_run[1], M.normalise(matrix[200]))


def test_an_unseeded_session_vector_is_not_moved_by_a_skip(matrix):
    """
    Nothing has played, so there is no direction to move and the rejection is
    not evidence of what the listener *does* want.  Seeding from the opposite of
    one skipped track would be the full-strength claim from a single rejection
    that the taste model also refuses to make.
    """
    state = SessionState(dimension=matrix.shape[1])
    assert not state.is_seeded()

    outcome = state.repel_from_skip_run(matrix[0], 1, matrix)

    assert not state.is_seeded()
    assert outcome['turnover'] == 0.0
    assert outcome['lambda'] == 0.0
