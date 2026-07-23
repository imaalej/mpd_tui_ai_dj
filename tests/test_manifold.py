"""
The geometry the skip rebuild rests on (audit H9), as pure numpy.

Everything here is expressed in *observable* units — candidate-pool turnover and
mean similarity to real tracks — rather than in vector-space magnitudes.  That is
the point of the module: a λ expressed in cosines needed re-tuning every time the
embedding space moved (a fixed 0.15 nudge went from moving 3.9% of the pool to
2.9% of it between Stage 0 and Stage 1), while "move far enough that half of what
you would have heard is different" re-derives itself.
"""

import numpy as np
import pytest

import manifold as M


@pytest.fixture
def matrix(rng):
    """
    A stand-in library with the two structural properties that matter.

    **Clusters**, because a real collection has them and a repulsion's job is to
    leave one — over uniformly scattered points the top-k is a diffuse shell that
    merely rotates, and turnover saturates well below what a real library
    reaches.  **Mild anisotropy**, matching the *centred* space the code actually
    runs in (mean random-pair similarity ≈ 0, audit C5) rather than the raw CLAP
    cone at +0.67, which nothing downstream of `TrackLibrary.load_embeddings`
    ever sees.
    """
    dimension = 64
    centres = rng.standard_normal((8, dimension))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)

    library = np.repeat(centres, 50, axis=0) + 0.45 * rng.standard_normal((400, dimension))
    bias = rng.standard_normal(dimension)
    bias /= np.linalg.norm(bias)
    library = library + 0.15 * bias
    return library / np.linalg.norm(library, axis=1, keepdims=True)


# Pool size for these tests.  The real system uses 100 of 674 (15%); 40 of 400
# is the same ratio, and the ratio is what bounds how much turnover is even
# achievable — a pool that is most of the library cannot turn over.
POOL_K = 40


# ── turnover ─────────────────────────────────────────────────────────────────

def test_a_vector_has_no_turnover_against_itself(matrix, rng):
    v = M.normalise(rng.standard_normal(64))
    assert M.pool_turnover(matrix, v, v, k=POOL_K) == 0.0


def test_turnover_rises_monotonically_with_displacement(matrix, rng):
    """
    Not a tautology: turnover is a set-overlap count, so it can plateau.  What
    must hold is that pushing harder never turns over *less*, because the skip
    schedule reads it as an escalation.
    """
    v = M.normalise(rng.standard_normal(64))
    away = M.normalise(rng.standard_normal(64))

    turnovers = [M.pool_turnover(matrix, v, M.repel(v, away, lam), k=POOL_K)
                 for lam in (0.1, 0.3, 0.6, 1.0, 1.5)]

    assert turnovers == sorted(turnovers), turnovers
    assert turnovers[-1] > turnovers[0]


def test_turnover_is_a_fraction(matrix, rng):
    for _ in range(10):
        a = M.normalise(rng.standard_normal(64))
        b = M.normalise(rng.standard_normal(64))
        assert 0.0 <= M.pool_turnover(matrix, a, b, k=50) <= 1.0


def test_turnover_against_a_reversed_vector_is_total(matrix, rng):
    v = M.normalise(rng.standard_normal(64))
    assert M.pool_turnover(matrix, v, -v, k=25) == 1.0


# ── the solver ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize('target', [0.05, 0.20, 0.50])
def test_the_solver_reaches_the_target_it_is_given(matrix, rng, target):
    """
    The load-bearing property: λ is *solved* for an observable outcome, not
    declared.  The schedule is the only input; the magnitude falls out of it and
    re-derives itself on any library.

    The full 5/20/50/85 schedule is asserted against the *real* library in
    `test_skip_escalation.py` — whether a given target is reachable at all is a
    property of the collection's structure rather than of the solver, so it
    cannot honestly be checked against a stand-in.
    """
    v = M.normalise(rng.standard_normal(64))
    away = M.normalise(rng.standard_normal(64))

    lam, achieved, moved = M.solve_repulsion(matrix, v, away, target, k=POOL_K)

    assert achieved >= target
    assert 0 < lam <= M.LAMBDA_MAX
    assert M.pool_turnover(matrix, v, moved, k=POOL_K) == pytest.approx(achieved)


def test_the_solver_returns_the_smallest_lambda_that_works(matrix, rng):
    """Overshooting would make the escalation coarser than the schedule says."""
    v = M.normalise(rng.standard_normal(64))
    away = M.normalise(rng.standard_normal(64))

    lam, achieved, _ = M.solve_repulsion(matrix, v, away, 0.50, k=POOL_K)

    smaller = M.pool_turnover(
        matrix, v, M.repel(v, away, lam - M.LAMBDA_STEP), k=POOL_K)
    assert smaller < 0.50


def test_a_bigger_target_never_yields_a_smaller_lambda(matrix, rng):
    v = M.normalise(rng.standard_normal(64))
    away = M.normalise(rng.standard_normal(64))

    lambdas = [M.solve_repulsion(matrix, v, away, t, k=POOL_K)[0]
               for t in (0.05, 0.20, 0.50)]
    assert lambdas == sorted(lambdas), lambdas


def test_an_unreachable_target_returns_the_best_achievable_not_a_promise(matrix, rng):
    """
    Ask for 100% turnover of a pool that is most of the library and no λ can
    deliver it.  The caller must get a real measurement of what happened, and
    the *best* λ available rather than merely the largest one — turnover is a
    set-overlap count, so the largest λ is not always the most effective.
    """
    small = matrix[:60]
    v = M.normalise(rng.standard_normal(64))
    away = M.normalise(rng.standard_normal(64))

    lam, achieved, moved = M.solve_repulsion(small, v, away, 1.0, k=50)

    assert achieved < 1.0, "this target was supposed to be out of reach"
    assert achieved == pytest.approx(M.pool_turnover(small, v, moved, k=50))

    grid_best = max(M.pool_turnover(small, v, M.repel(v, away, l), k=50)
                    for l in np.arange(M.LAMBDA_STEP, M.LAMBDA_MAX + 1e-9, M.LAMBDA_STEP))
    assert achieved == pytest.approx(grid_best)


def test_the_solved_vector_is_a_unit_vector(matrix, rng):
    v = M.normalise(rng.standard_normal(64))
    away = M.normalise(rng.standard_normal(64))
    _, _, moved = M.solve_repulsion(matrix, v, away, 0.5, k=POOL_K)
    assert float(np.linalg.norm(moved)) == pytest.approx(1.0)


# ── snap: the structural guarantee ───────────────────────────────────────────

def test_snap_lands_on_the_centroid_of_the_nearest_real_tracks(matrix, rng):
    v = M.normalise(rng.standard_normal(64))
    snapped = M.snap(matrix, v, k=25)

    expected = M.normalise(matrix[M.top_k_indices(matrix, v, 25)].mean(axis=0))
    assert np.allclose(snapped, expected)
    assert float(np.linalg.norm(snapped)) == pytest.approx(1.0)


def test_snap_keeps_the_vector_inside_the_music_however_hard_it_is_pushed(matrix, rng):
    """
    The structural half of H9, and the reason the old `[V]` had to go: it blended
    half a random direction in and landed *off* the manifold, so the candidate
    pool was drawn around a point no music occupies.  Here, no λ — however
    extreme — can produce a snapped vector worse than an unguarded small one.
    """
    v = M.normalise(rng.standard_normal(64))
    away = M.normalise(rng.standard_normal(64))
    baseline = M.on_manifold_quality(matrix, v)

    for lam in (0.5, 1.0, 1.5, 2.0, 5.0):
        raw = M.repel(v, away, lam)
        snapped = M.snap(matrix, raw)
        assert M.on_manifold_quality(matrix, snapped) >= baseline * 0.9, (
            f"snap failed to hold the manifold at λ={lam}")


def test_an_unguarded_large_repulsion_does_leave_the_manifold(matrix, rng):
    """
    The other side of the same coin — this is *why* the snap is not optional at
    large λ.  Without it the vector degrades exactly as the deleted `[V]` did.
    """
    v = M.normalise(rng.standard_normal(64))
    away = M.normalise(rng.standard_normal(64))

    unguarded = M.on_manifold_quality(matrix, M.repel(v, away, 1.0))
    snapped = M.on_manifold_quality(matrix, M.snap(matrix, M.repel(v, away, 1.0)))

    assert unguarded < snapped


def test_repeated_snapping_converges_to_a_fixed_point(matrix, rng):
    """
    snap() settles rather than wandering: applied repeatedly it reaches a vector
    whose own 25 nearest tracks are the ones it was built from, and stops
    exactly.  That is what makes it a projection onto the library rather than a
    random walk across it — and it is why one application is a bounded move,
    which the escalation schedule depends on.

    Asserted as convergence rather than as "moves less than 0.05", because that
    threshold would be invented; the fixed point is a fact about the operator.
    """
    for _ in range(5):
        vector = M.normalise(rng.standard_normal(64))

        sequence = [M.snap(matrix, vector)]
        for _ in range(8):
            sequence.append(M.snap(matrix, sequence[-1]))

        steps = [1 - float(np.dot(sequence[i], sequence[i + 1]))
                 for i in range(len(sequence) - 1)]
        assert steps[-1] == pytest.approx(0.0, abs=1e-12), (
            f"snap did not settle: {steps}")
        assert steps[0] >= steps[-1]


def test_a_random_direction_scores_far_below_a_real_track(matrix, rng):
    """The measurement that makes 'is this vector still music?' meaningful."""
    real = M.on_manifold_quality(matrix, matrix[0])
    random_direction = M.on_manifold_quality(
        matrix, M.normalise(rng.standard_normal(64)))
    assert real > random_direction


# ── normalise ────────────────────────────────────────────────────────────────

def test_normalise_leaves_a_zero_vector_alone_rather_than_exploding():
    """A zero vector is the legitimate 'no evidence yet' state (audit L7)."""
    zero = np.zeros(64)
    assert np.all(M.normalise(zero) == 0)
    assert np.isfinite(M.normalise(zero)).all()
