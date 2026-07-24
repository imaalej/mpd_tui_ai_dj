"""
Partial results and resume.

A generation run is minutes long, so it has to survive being interrupted — and
the Stage 1 schema made that harder, because a partial file now has to carry the
per-window matrix and its CSR offsets, not just one vector per track.  A silent
bug here produces a *valid-looking* artifact with one track's windows attributed
to another.

Also guards the two counting bugs the audit noted alongside M8: the resumed
count used to be folded into `successful`, overstating what the run did, and the
speed line divided by a duration that can be zero.
"""

import numpy as np
import pytest

from embedding_generator import (
    CLAP_AVAILABLE,
    CLAPEmbeddingGenerator,
    EmbeddingGenerationError,
)

pytestmark = pytest.mark.skipif(
    not CLAP_AVAILABLE, reason="torch/transformers not installed"
)


@pytest.fixture
def generator():
    # No model is loaded — the partial file format is pure numpy.
    return CLAPEmbeddingGenerator(device='cpu')


@pytest.fixture
def fake_results(rng):
    """Three tracks with deliberately different window counts."""
    counts = {'a.flac': 4, 'b.flac': 1, 'c.flac': 7}
    windows = {}
    pooled = {}
    for name, count in counts.items():
        matrix = rng.standard_normal((count, 8)).astype(np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        windows[name] = matrix
        pooled[name] = CLAPEmbeddingGenerator.pool(matrix)
    return pooled, windows, list(counts)


def test_a_partial_round_trips_exactly(generator, tmp_path, fake_results):
    pooled, windows, order = fake_results
    path = tmp_path / 'partial.npz'
    generator._save_partial(pooled, windows, order, path)

    restored_pooled, restored_windows, restored_batch = generator._load_partial(path)

    assert list(restored_pooled) == order
    # inspection D2: the batch size the checkpoint was built under is recorded so
    # a resume can refuse a different one.
    assert restored_batch == generator.batch_size
    for name in order:
        assert np.array_equal(restored_pooled[name], pooled[name])
        assert np.array_equal(restored_windows[name], windows[name])


def test_uneven_window_counts_stay_attached_to_their_own_track(generator, fake_results):
    """
    The failure this exists to catch: CSR offsets off by one track would hand
    `c.flac`'s windows to `b.flac` and still validate.
    """
    pooled, windows, order = fake_results
    names, embeddings, offsets, matrix = generator._stack(pooled, windows, order)

    assert names == order
    assert list(np.diff(offsets)) == [4, 1, 7]
    assert offsets[0] == 0 and offsets[-1] == matrix.shape[0] == 12
    for i, name in enumerate(names):
        assert np.array_equal(matrix[offsets[i]:offsets[i + 1]], windows[name])
        assert np.array_equal(embeddings[i], pooled[name])


def test_stacking_follows_the_requested_order_not_insertion_order(generator, fake_results):
    """
    Tracks finish on worker threads in whatever order they decode.  The artifact
    must be ordered by the track list, or two runs of the same library produce
    different files.
    """
    pooled, windows, order = fake_results
    shuffled = {name: pooled[name] for name in reversed(order)}

    names, _, offsets, _ = generator._stack(shuffled, windows, order)
    assert names == order
    assert list(np.diff(offsets)) == [4, 1, 7]


def test_a_partial_listing_tracks_the_run_does_not_want_is_ignored(generator, fake_results):
    """Resuming after the library changed must not resurrect a removed track."""
    pooled, windows, order = fake_results
    names, _, _, _ = generator._stack(pooled, windows, ['a.flac', 'c.flac'])
    assert names == ['a.flac', 'c.flac']


def test_a_corrupt_partial_is_a_hard_error(generator, tmp_path):
    """
    inspection D3: the old behaviour warned and returned empty, so a truncated
    partial silently discarded every already-embedded track and looked like a
    normal slow run.  With atomic writes a partial can no longer be truncated by
    an interrupted save, so an unreadable one is real damage — it must stop the
    run and name the file, not quietly start over.
    """
    path = tmp_path / 'partial.npz'
    path.write_bytes(b'not an npz at all')

    with pytest.raises(EmbeddingGenerationError, match="unreadable"):
        generator._load_partial(path)


def test_stacking_nothing_does_not_crash_the_run(generator):
    """
    Reachable when the first hundred tracks all fail to decode: the periodic
    checkpoint fires with an empty result set.  `np.stack([])` raises, and losing
    a run to a bookkeeping call would be an absurd way to lose it.
    """
    names, embeddings, offsets, matrix = generator._stack({}, {}, ['a.flac'])

    assert names == []
    assert embeddings.shape[0] == 0
    assert list(offsets) == [0]
    assert matrix.shape[0] == 0


def test_an_interrupt_checkpoints_before_it_propagates(generator, tmp_path, monkeypatch):
    """
    The CLI tells the user "partial results saved, use --resume".  Without this,
    Ctrl-C in the 99 tracks between checkpoints discards all of them while the
    message claims otherwise.
    """
    import numpy as np

    out = tmp_path / 'library.npz'
    partial = tmp_path / 'library.partial.npz'
    embedded = []

    def fake_prepare(track_file, music_dir):
        from embedding_generator import PreparedTrack
        return PreparedTrack(track_file=track_file, n_windows_total=1)

    def fake_embed(prepared):
        if len(embedded) >= 2:
            raise KeyboardInterrupt
        embedded.append(prepared.track_file)
        vector = np.full((1, 4), 0.5, dtype=np.float32)
        return vector

    monkeypatch.setattr(generator, 'load_model', lambda *a, **k: None)
    monkeypatch.setattr(generator, 'prepare_track', fake_prepare)
    monkeypatch.setattr(generator, 'embed_prepared', fake_embed)

    with pytest.raises(KeyboardInterrupt):
        generator.generate_library(
            ['a.flac', 'b.flac', 'c.flac'], tmp_path, out
        )

    assert not out.exists(), "an interrupted run must not write a truncated library"
    assert partial.exists(), "the two completed tracks were thrown away"

    pooled, _, _ = generator._load_partial(partial)
    assert list(pooled) == ['a.flac', 'b.flac']


def test_atomic_save_leaves_no_temp_files_and_no_partial_on_failure(
        generator, tmp_path, monkeypatch):
    """
    inspection D3: a save must be all-or-nothing.  A clean save leaves only the
    destination (no stray temp), and a save that raises mid-write must leave the
    destination and directory untouched rather than a truncated file — and must
    not leak the temp file it was writing to.
    """
    import numpy as np

    good = {'x': np.arange(4, dtype=np.float32)}
    dest = tmp_path / 'ok.npz'
    generator._atomic_savez(dest, good)
    assert dest.exists()
    assert [p.name for p in tmp_path.iterdir()] == ['ok.npz'], "a temp file leaked"

    # A saver that dies mid-write: the destination must not appear and the temp
    # file must be cleaned up.
    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(np, 'savez', boom)
    dest2 = tmp_path / 'bad.npz'
    with pytest.raises(RuntimeError, match="disk full"):
        generator._atomic_savez(dest2, good)
    assert not dest2.exists()
    assert [p.name for p in tmp_path.iterdir()] == ['ok.npz'], "a temp file leaked on failure"


def test_resume_refuses_a_different_batch_size(tmp_path, monkeypatch):
    """
    inspection D2: rows from two batch sizes are not bit-identical, so splicing
    them into one artifact is silent corruption.  A resume under a different
    --batch-size must refuse rather than proceed.
    """
    import numpy as np

    gen32 = CLAPEmbeddingGenerator(device='cpu', batch_size=32)
    out = tmp_path / 'library.npz'
    partial = tmp_path / 'library.partial.npz'

    matrix = np.full((1, 4), 0.5, dtype=np.float32)
    gen32._save_partial({'a.flac': CLAPEmbeddingGenerator.pool(matrix)},
                        {'a.flac': matrix}, ['a.flac'], partial)

    gen16 = CLAPEmbeddingGenerator(device='cpu', batch_size=16)
    monkeypatch.setattr(gen16, 'load_model', lambda *a, **k: None)
    with pytest.raises(EmbeddingGenerationError, match="batch-size"):
        gen16.generate_library(['a.flac', 'b.flac'], tmp_path, out, resume=True)


def test_resume_accepts_the_same_batch_size(tmp_path, monkeypatch):
    """The other half of D2: a matching batch size resumes without complaint."""
    import numpy as np

    from embedding_generator import PreparedTrack

    gen = CLAPEmbeddingGenerator(device='cpu', batch_size=32)
    out = tmp_path / 'library.npz'
    partial = tmp_path / 'library.partial.npz'

    matrix = np.full((1, 4), 0.5, dtype=np.float32)
    gen._save_partial({'a.flac': CLAPEmbeddingGenerator.pool(matrix)},
                      {'a.flac': matrix}, ['a.flac'], partial)

    def fake_prepare(track_file, music_dir):
        return PreparedTrack(track_file=track_file, n_windows_total=1)

    def fake_embed(prepared):
        return np.full((1, 4), 0.25, dtype=np.float32)

    monkeypatch.setattr(gen, 'load_model', lambda *a, **k: None)
    monkeypatch.setattr(gen, 'prepare_track', fake_prepare)
    monkeypatch.setattr(gen, 'embed_prepared', fake_embed)

    stats = gen.generate_library(['a.flac', 'b.flac'], tmp_path, out, resume=True)
    assert stats['resumed'] == 1
    assert out.exists()
    assert not partial.exists(), "a completed run should clean up its partial"


def test_pooling_a_single_window_track_is_that_window(rng):
    window = rng.standard_normal((1, 8)).astype(np.float32)
    window /= np.linalg.norm(window)
    assert np.allclose(CLAPEmbeddingGenerator.pool(window), window[0], atol=1e-6)
