"""
Mood axes for the vibe cloud (Phase 3, inspection G3 groundwork).

Two halves, mirroring the descriptor-bank tests:

  * pure-numpy mechanics — the selection avoids a pancake, the frame round-trips
    through the artifact, and an axis-less bank degrades gracefully.  These run
    everywhere, no model, no real data.
  * the real library, if built — the axes stored in `descriptors.npz` reproduce
    `mood_axes.select_axes` (which is what `explore_mood_axes.py` now calls), so
    the stored frame is provably the one the selection logic picks.
"""

import numpy as np
import pytest

import mood_axes
from config import config
from mood_axes import MoodAxes, select_axes, selection_arrays


# ------------------------------------------------------------- selection logic

def _controlled_bank(rng, dim=16):
    """
    A four-axis candidate set with single-word poles I fully control, so the
    library projections are known: X, Y, Z decorrelated, and Xdup collinear with
    X.  A round frame must take one of {X, Xdup} plus Y and Z — never both of the
    collinear pair, which would be the pancake the whole exercise avoids.
    """
    e = np.eye(dim, dtype=np.float64)
    vecs = {
        'xlo': -e[0], 'xhi': e[0],
        'ylo': -e[1], 'yhi': e[1],
        'zlo': -e[2], 'zhi': e[2],
        # Xdup points essentially along e0 too (a whisker of e3 keeps it a
        # distinct row), so its projection tracks X's.
        'xlo2': -(0.98 * e[0] + 0.02 * e[3]), 'xhi2': (0.98 * e[0] + 0.02 * e[3]),
    }
    labels = list(vecs)
    text = np.stack([vecs[l] for l in labels]).astype(np.float32)

    axes = {
        'X': (['xlo'], ['xhi']),
        'Y': (['ylo'], ['yhi']),
        'Z': (['zlo'], ['zhi']),
        'Xdup': (['xlo2'], ['xhi2']),
    }

    # Library: independent spread along e0/e1/e2 so the three real axes are
    # decorrelated; nothing along e3 so Xdup ≈ X.
    coords = rng.standard_normal((400, dim)).astype(np.float64)
    coords[:, 3:] *= 0.001
    library = coords / np.linalg.norm(coords, axis=1, keepdims=True)
    return text, labels, library, axes


def test_selection_avoids_the_collinear_pancake(rng, monkeypatch):
    text, labels, library, axes = _controlled_bank(rng)
    monkeypatch.setattr(mood_axes, 'CANDIDATE_AXES', axes)

    sel = select_axes(text, labels, library)

    assert len(sel.labels) == 3
    assert set(sel.labels) <= set(axes)
    # The two independent axes are always in; exactly one of the collinear pair.
    assert {'Y', 'Z'} <= set(sel.labels)
    assert not ({'X', 'Xdup'} <= set(sel.labels)), "picked both collinear axes"


def test_selection_is_deterministic(rng, monkeypatch):
    text, labels, library, axes = _controlled_bank(rng)
    monkeypatch.setattr(mood_axes, 'CANDIDATE_AXES', axes)
    a = select_axes(text, labels, library)
    b = select_axes(text, labels, library)
    assert a.labels == b.labels
    assert np.array_equal(a.directions, b.directions)


def test_selected_directions_are_unit_and_calibration_is_shaped(rng, monkeypatch):
    text, labels, library, axes = _controlled_bank(rng)
    monkeypatch.setattr(mood_axes, 'CANDIDATE_AXES', axes)
    sel = select_axes(text, labels, library)

    assert sel.directions.shape == (3, library.shape[1])
    assert np.allclose(np.linalg.norm(sel.directions, axis=1), 1.0, atol=1e-5)
    assert sel.mean.shape == (3,) and sel.std.shape == (3,)


def test_too_few_usable_axes_raises(rng, monkeypatch):
    text, labels, library, _ = _controlled_bank(rng)
    monkeypatch.setattr(mood_axes, 'CANDIDATE_AXES', {
        'X': (['xlo'], ['xhi']),
        'Y': (['ylo'], ['yhi']),
        # Third axis references words the bank does not carry → dropped.
        'Gone': (['missing_lo'], ['missing_hi']),
    })
    with pytest.raises(ValueError, match="usable mood axes"):
        select_axes(text, labels, library)


# ----------------------------------------------------------- artifact round-trip

def test_axes_round_trip_through_an_npz(rng, monkeypatch, tmp_path):
    text, labels, library, axes = _controlled_bank(rng)
    monkeypatch.setattr(mood_axes, 'CANDIDATE_AXES', axes)
    sel = select_axes(text, labels, library)

    path = tmp_path / 'descriptors.npz'
    np.savez_compressed(path, **selection_arrays(sel))

    loaded = MoodAxes.load(path)
    assert loaded is not None
    assert loaded.labels == sel.labels
    assert loaded.dimension == library.shape[1]
    assert np.allclose(loaded.directions, sel.directions)

    # A vector on one axis's high pole reads positive on that axis.
    coords = loaded.coordinates(sel.directions[0])
    assert coords.shape == (3,)
    assert coords[0] > 0


def test_coordinates_batch_matches_per_row(rng, monkeypatch, tmp_path):
    """
    Phase 4 projects the whole library in one call for the point cloud; a batch
    (N, D) → (N, 3) must equal projecting each row on its own.
    """
    text, labels, library, axes = _controlled_bank(rng)
    monkeypatch.setattr(mood_axes, 'CANDIDATE_AXES', axes)
    sel = select_axes(text, labels, library)
    m = MoodAxes(sel.labels, sel.directions, sel.mean, sel.std)

    lib32 = library.astype(np.float32)
    batch = m.coordinates(lib32)
    assert batch.shape == (library.shape[0], 3)
    per_row = np.stack([m.coordinates(v) for v in lib32])
    assert np.allclose(batch, per_row, atol=1e-5)


def test_a_bank_without_axes_loads_as_none(tmp_path):
    """An older descriptors.npz predates the axes — reported, never a crash."""
    path = tmp_path / 'descriptors.npz'
    np.savez_compressed(path, labels=np.array(['calm'], dtype=np.str_),
                        text_embeddings=np.zeros((1, 8), np.float32))
    assert MoodAxes.load(path) is None


def test_a_missing_bank_loads_as_none(tmp_path):
    assert MoodAxes.load(tmp_path / 'nothing.npz') is None


# ------------------------------------------------- the real library, if built

@pytest.fixture
def real_artifacts():
    if not config.embeddings_file.exists() or not config.descriptors_file.exists():
        pytest.skip("no generated library — run generate_embeddings.py")
    with np.load(config.descriptors_file, allow_pickle=False) as bank:
        if 'axis_labels' not in bank:
            pytest.skip("descriptors.npz predates mood axes — rebuild with "
                        "--descriptors-only")
        stored = {
            'labels': [str(x) for x in bank['axis_labels']],
            'directions': bank['axis_directions'],
            'mean': bank['axis_mean'],
            'std': bank['axis_std'],
            'text_embeddings': bank['text_embeddings'],
            'desc_labels': [str(x) for x in bank['labels']],
        }
    from embeddings_io import centre
    with np.load(config.embeddings_file, allow_pickle=False) as emb:
        library = centre(emb['embeddings'], emb['centroid'])
    return stored, library


def test_stored_axes_reproduce_the_selection_on_the_real_library(real_artifacts):
    """
    The Phase 3 acceptance check: the frame stored at generation time is exactly
    what `select_axes` picks from the same inputs — the logic `explore_mood_axes.py`
    now shares.  So the stored axes cannot silently drift from the offline proof.
    """
    stored, library = real_artifacts
    sel = select_axes(stored['text_embeddings'], stored['desc_labels'], library)

    assert stored['labels'] == sel.labels
    assert np.allclose(stored['directions'], sel.directions, atol=1e-5)
    assert np.allclose(stored['mean'], sel.mean, atol=1e-5)
    assert np.allclose(stored['std'], sel.std, atol=1e-5)


def test_the_real_frame_is_a_cloud_not_a_pancake(real_artifacts):
    """
    Why the triad is selected rather than fixed: the stored axes must actually
    span a 3-D shape over this library (participation ratio well above 1), which
    the hand-picked Intensity·Tone·Saturation triad does not (its top two axes
    correlate ~0.98 here).
    """
    stored, library = real_artifacts
    axes = MoodAxes(stored['labels'], stored['directions'], stored['mean'], stored['std'])
    coords = axes.coordinates(library)          # whole library in one call
    assert coords.shape == (library.shape[0], 3)
    cov = np.cov(coords, rowvar=False)
    ev = np.clip(np.linalg.eigvalsh(cov), 0, None)
    pr = float(ev.sum() ** 2 / np.square(ev).sum())
    assert pr > 2.0, f"the stored frame is nearly flat (participation ratio {pr:.2f})"
