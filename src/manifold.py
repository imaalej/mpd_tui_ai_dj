"""
Manifold geometry — the library's own embedding matrix as the reference frame.

Everything here answers a question in *observable* units rather than in
vector-space magnitudes, which is the whole reason the module exists (audit H9,
and the governing principle in CLAUDE.md).  "Move the session vector by 0.15" is
a number nobody can verify.  "Move it far enough that half of what you would
have heard next is now different" is a number the listener experiences directly,
and it re-derives itself when the embedding space changes underneath it — as it
did between Stage 0 and Stage 1, where a fixed 0.15 nudge went from moving 3.9%
of the candidate pool to moving 0.3% of it.

Three operations, all pure numpy over an `(N, D)` matrix of L2-normalised,
already-centred library embeddings:

  `pool_turnover`   — how much of the top-k candidate pool two vectors disagree
                      about.  The unit everything else is expressed in.
  `solve_repulsion` — the smallest λ for which `normalise(v − λ·away)` reaches a
                      requested turnover.  Solved, not declared.
  `snap`            — replace a vector with the centroid of its 25 nearest real
                      tracks.  A structural guarantee that the session vector
                      cannot leave the region the music occupies, however large
                      λ grows.  It is a *move* in its own right, not a harmless
                      projection — see `snap`'s own docstring.
"""

import numpy as np

# How many library tracks the "is this vector still music?" measurement averages
# over.  Also the neighbourhood `snap()` collapses to.  25 is large enough that
# one atypical near-neighbour cannot define the answer and small enough that the
# result is still a *place* rather than the library mean.
MANIFOLD_K = 25

# The λ search grid.  0.05 resolution over [0.05, 2.0]: 40 candidate vectors,
# evaluated in one (N, D) @ (D, 40) product.  Finer resolution buys nothing —
# turnover moves by a few percent per 0.05 step, well inside the run-to-run
# spread of the quantity being targeted.
LAMBDA_STEP = 0.05
LAMBDA_MAX = 2.0


def normalise(vector: np.ndarray) -> np.ndarray:
    """L2-normalise, leaving a (near-)zero vector alone rather than exploding."""
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return np.asarray(vector, dtype=np.float64)
    return np.asarray(vector, dtype=np.float64) / norm


def top_k_indices(matrix: np.ndarray, vector: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k library rows most similar to `vector`, unordered."""
    k = min(k, matrix.shape[0])
    similarities = matrix @ vector
    return np.argpartition(similarities, -k)[-k:]


def pool_turnover(matrix: np.ndarray,
                  old_vector: np.ndarray,
                  new_vector: np.ndarray,
                  k: int = 100) -> float:
    """
    Fraction of the top-k candidate pool that `new_vector` disagrees with
    `old_vector` about.

    This is the unit the skip schedule is written in.  It is deliberately
    computed over the whole library with no exclusions applied: the question is
    "how much did the *neighbourhood* move", not "which specific tracks are
    eligible right now", and mixing the anti-repetition state into it would make
    the answer depend on how long the session had been running.
    """
    k = min(k, matrix.shape[0])
    if k == 0:
        return 0.0

    old_index = top_k_indices(matrix, old_vector, k)
    new_index = top_k_indices(matrix, new_vector, k)

    membership = np.zeros(matrix.shape[0], dtype=bool)
    membership[old_index] = True
    retained = int(membership[new_index].sum())
    return 1.0 - retained / k


def repel(vector: np.ndarray, away_from: np.ndarray, lam: float) -> np.ndarray:
    """`normalise(v − λ·away)` — the one displacement the skip path applies."""
    return normalise(np.asarray(vector, dtype=np.float64)
                     - lam * normalise(away_from))


def solve_repulsion(matrix: np.ndarray,
                    vector: np.ndarray,
                    away_from: np.ndarray,
                    target_turnover: float,
                    k: int = 100,
                    snap_result: bool = False,
                    snap_k: int = MANIFOLD_K,
                    lambda_max: float = LAMBDA_MAX,
                    lambda_step: float = LAMBDA_STEP) -> tuple:
    """
    Find the smallest λ that reaches `target_turnover`, and return what it did.

    Returns `(lam, achieved_turnover, new_vector)`.

    **`snap_result` is not a post-processing flag — it changes what is solved
    for.**  With it set, each candidate is snapped back onto the manifold
    *before* its turnover is measured, so the target is met by the vector that
    will actually select the next track.  Solving against the un-snapped vector
    and snapping afterwards is a real trap: the snap relocates to a 25-track
    centroid, and for a modest λ those 25 tracks are largely the *original*
    neighbourhood, so the result can land back where it started.  Measured live,
    that produced a second consecutive skip reporting 1% turnover against a 20%
    target — undoing the first skip rather than escalating past it.

    If even `lambda_max` falls short — possible on a small library, where the
    pool is most of the collection and cannot turn over — the best λ the grid
    can offer is returned along with whatever it actually achieved, so the
    caller always gets a measurement rather than a promise.

    The whole grid is evaluated in two matrix products regardless of `snap_k`:
    40 candidate vectors against 674 embeddings is 13.8 M multiply-adds,
    microseconds.  A grid rather than a bisection because turnover is only
    *nearly* monotone in λ — it is a set-overlap count, so it plateaus and can
    wobble by a track or two — and a bisection on a non-monotone function can
    settle in the wrong plateau.
    """
    vector = normalise(vector)
    away = normalise(away_from)
    k = min(k, matrix.shape[0])
    snap_k = min(snap_k, matrix.shape[0])

    lambdas = np.arange(lambda_step, lambda_max + lambda_step / 2, lambda_step)

    # (L, D) candidate vectors, one row per λ.
    candidates = vector[None, :] - lambdas[:, None] * away[None, :]
    candidates = candidates / np.maximum(
        np.linalg.norm(candidates, axis=1, keepdims=True), 1e-8)

    similarities = matrix @ candidates.T          # (N, L)

    if snap_result:
        # Snap every candidate at once, reusing the similarities just computed.
        neighbours = np.argpartition(similarities, -snap_k, axis=0)[-snap_k:]
        candidates = matrix[neighbours].mean(axis=0)               # (L, D)
        candidates = candidates / np.maximum(
            np.linalg.norm(candidates, axis=1, keepdims=True), 1e-8)
        similarities = matrix @ candidates.T                       # (N, L)

    baseline = np.zeros(matrix.shape[0], dtype=bool)
    baseline[top_k_indices(matrix, vector, k)] = True

    top_rows = np.argpartition(similarities, -k, axis=0)[-k:]   # (k, L)
    retained = baseline[top_rows].sum(axis=0)                   # (L,)
    turnovers = 1.0 - retained / k

    reached = np.flatnonzero(turnovers >= target_turnover)
    if reached.size:
        index = int(reached[0])          # the smallest λ that gets there
    else:
        # Unreachable — return the best the grid can actually do rather than
        # whatever the largest λ happened to give.  `argmax` takes the first
        # occurrence, so on a plateau this is still the smallest λ achieving it.
        index = int(np.argmax(turnovers))

    return float(lambdas[index]), float(turnovers[index]), candidates[index].copy()


def snap(matrix: np.ndarray, vector: np.ndarray, k: int = MANIFOLD_K) -> np.ndarray:
    """
    Replace `vector` with the normalised centroid of its k nearest real tracks.

    This is what makes "however hard you push, the session vector stays inside
    the music" a structural property rather than a hope.  A repulsion large
    enough to be worth calling a change of direction can otherwise land in a
    region of the space no track occupies, which is exactly the defect that
    killed the old `[V]` key: it blended half a random direction in and landed
    at 0.450 on-manifold quality against 0.641 for an ordinary session vector.

    **`snap()` is a move, not a projection.**  It does not leave small
    displacements alone — relocating to a 25-track centroid is itself a
    displacement, and applied after a small λ it overshoots badly (measured:
    turnover 5.7% → 17.8% at the 5% target, three times the schedule).  Only
    apply it where λ is already large; the skip path gates it at run length ≥ 2.
    """
    index = top_k_indices(matrix, normalise(vector), k)
    return normalise(matrix[index].mean(axis=0))


def on_manifold_quality(matrix: np.ndarray,
                        vector: np.ndarray,
                        k: int = MANIFOLD_K) -> float:
    """
    Mean similarity to the k nearest real tracks — "is this vector still music?"

    Reference values on the built 674-track library (audit §10b): a real track
    0.729, an ordinary session vector 0.641, a random direction 0.085.
    """
    k = min(k, matrix.shape[0])
    similarities = matrix @ normalise(vector)
    index = np.argpartition(similarities, -k)[-k:]
    return float(np.mean(similarities[index]))
