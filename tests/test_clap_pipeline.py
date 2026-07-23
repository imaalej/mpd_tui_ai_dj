"""
Stage 1's acceptance criteria, run against the real model (audit §8, Stage 1).

    "embed the same file twice, assert **bit-identical** vectors.  Re-run the
     audit's measurement: same-track self-similarity must be exactly 1.0, and
     post-centring the random-pair distribution must sit near 0 rather than
     0.577."

C3 made those *testable properties* for the first time — under the old
random-crop generator none of them could hold.  So they are tests, not notes.

These load the CLAP checkpoint and skip if it is not already cached: a test that
silently downloads 700 MB is not a test anyone can run.  The pure-arithmetic
half of the same work is in `test_windowing.py` and runs everywhere.
"""

import numpy as np
import pytest

from config import config
from embedding_generator import CLAP_AVAILABLE, CLAPEmbeddingGenerator
from embeddings_io import centre, similarity_report, validate_embeddings

pytestmark = pytest.mark.skipif(
    not CLAP_AVAILABLE, reason="torch/transformers not installed"
)

SAMPLE_RATE = 48000


@pytest.fixture(scope="module")
def generator():
    """One model load for the whole module — it costs seconds, not milliseconds."""
    from transformers import ClapProcessor

    try:
        ClapProcessor.from_pretrained(CLAPEmbeddingGenerator.MODEL_NAME, local_files_only=True)
    except Exception:                                  # noqa: BLE001
        pytest.skip(f"{CLAPEmbeddingGenerator.MODEL_NAME} is not in the local cache")

    gen = CLAPEmbeddingGenerator(batch_size=16, workers=2)
    gen.load_model()
    return gen


@pytest.fixture(scope="module")
def three_part_track(tmp_path_factory):
    """
    Twenty-five seconds in three musically distinct sections — the same shape of
    signal the audit used to measure C3.  Not a multiple of the window, so the
    end-aligned tail is exercised too.
    """
    import torchaudio
    import torch

    rng = np.random.default_rng(1974)
    t = np.arange(25 * SAMPLE_RATE) / SAMPLE_RATE
    signal = np.zeros_like(t, dtype=np.float32)
    signal[:8 * SAMPLE_RATE] = 0.6 * np.sin(2 * np.pi * 110 * t[:8 * SAMPLE_RATE])
    middle = slice(8 * SAMPLE_RATE, 17 * SAMPLE_RATE)
    signal[middle] = 0.5 * rng.standard_normal(17 * SAMPLE_RATE - 8 * SAMPLE_RATE)
    signal[17 * SAMPLE_RATE:] = 0.6 * np.sin(2 * np.pi * 880 * t[17 * SAMPLE_RATE:])

    path = tmp_path_factory.mktemp("audio") / "three_part.wav"
    torchaudio.save(str(path), torch.from_numpy(signal)[None, :], SAMPLE_RATE)
    return path


# --------------------------------------------------------------- determinism

def test_embedding_the_same_file_twice_is_bit_identical(generator, three_part_track):
    """
    C3's headline defect, inverted into an assertion.  The old generator handed
    the whole track to the processor, which cropped ten seconds at random: the
    same file embedded ten times had a *median* self-similarity of 0.884 and a
    minimum of 0.354.
    """
    first_pooled, first_windows = generator.embed_track(three_part_track.name,
                                                        three_part_track.parent)
    second_pooled, second_windows = generator.embed_track(three_part_track.name,
                                                          three_part_track.parent)

    assert np.array_equal(first_pooled, second_pooled)
    assert np.array_equal(first_windows, second_windows)


def test_self_similarity_is_exactly_one(generator, three_part_track):
    first, _ = generator.embed_track(three_part_track.name, three_part_track.parent)
    second, _ = generator.embed_track(three_part_track.name, three_part_track.parent)
    assert float(first @ second) == pytest.approx(1.0, abs=1e-6)


def test_the_whole_track_is_represented_not_ten_seconds_of_it(generator, three_part_track):
    """
    The point of full coverage.  The pooled vector must sit between the
    sections; if it were one random ten-second crop it would be nearly identical
    to one window and remote from the others.
    """
    pooled, windows = generator.embed_track(three_part_track.name, three_part_track.parent)

    assert windows.shape[0] == 3, "25 s should tile into three windows"
    similarities = windows @ pooled
    assert similarities.min() > 0.3, "a section is not represented at all"
    assert similarities.max() < 0.999, "the pool collapsed onto a single window"
    assert windows[0] @ windows[2] < similarities.min(), (
        "the fixture's sections are supposed to be distinct"
    )


def test_pooling_is_mean_then_renormalise(generator, three_part_track):
    pooled, windows = generator.embed_track(three_part_track.name, three_part_track.parent)
    expected = windows.mean(axis=0)
    expected /= np.linalg.norm(expected)
    assert np.allclose(pooled, expected, atol=1e-6)
    assert float(np.linalg.norm(pooled)) == pytest.approx(1.0, abs=1e-5)


def test_window_embeddings_are_unit_length(generator, three_part_track):
    _, windows = generator.embed_track(three_part_track.name, three_part_track.parent)
    assert np.allclose(np.linalg.norm(windows, axis=1), 1.0, atol=1e-5)


def test_text_and_audio_towers_produce_the_same_dimension(generator):
    text = generator.embed_texts(["This is a recording of hypnotic music."])
    assert text.shape == (1, generator.EMBEDDING_DIM)
    assert np.allclose(np.linalg.norm(text, axis=1), 1.0, atol=1e-5)


def test_the_fusion_truncation_mode_is_not_used(generator):
    """
    The audit expected the fusion/rand_trunc question to become irrelevant at
    exactly `max_length_s`.  It does not: at that length `fusion` still stacks
    four mel crops, and the *unfused* checkpoint's patch embedding takes one
    channel, so it raises rather than degrading.  Pinning the mode is therefore
    load-bearing, not tidiness.
    """
    assert generator.TRUNCATION == 'rand_trunc'
    assert generator.processor.feature_extractor.truncation == 'rand_trunc'


# ------------------------------------------------- the real library, if built

@pytest.fixture(scope="module")
def library_file():
    if not config.embeddings_file.exists():
        pytest.skip("no generated library — run generate_embeddings.py")
    return config.embeddings_file


def test_the_generated_library_matches_the_schema(library_file):
    result = validate_embeddings(library_file)
    assert result['valid'], result.get('error')
    assert result['dimension'] == CLAPEmbeddingGenerator.EMBEDDING_DIM
    assert result['n_windows'] > result['n_tracks'], (
        "full coverage means many windows per track, not one"
    )


def test_centring_moves_the_real_library_to_near_zero(library_file):
    """
    C5's acceptance criterion, on the actual data: raw random pairs sat at 0.577
    in the audit (0.670 with pooled vectors, which average toward the library
    mean and so are *more* anisotropic).  Centred, the distribution must sit
    near 0 — that is what makes `(sim + 1) / 2` and the novelty term mean what
    they claim.
    """
    with np.load(library_file, allow_pickle=False) as data:
        raw = similarity_report(data['embeddings'])
        centred = similarity_report(centre(data['embeddings'], data['centroid']))

    assert raw['mean'] > 0.4, "the raw space is supposed to be a narrow cone"
    assert abs(centred['mean']) < 0.05, f"centred pairs sit at {centred['mean']}"
    assert centred['p05'] < -0.1, "centring should open up the negative half"


def test_the_generated_library_carries_its_provenance(library_file):
    metadata = validate_embeddings(library_file)['metadata']
    assert metadata['model'] == config.clap_model_name
    scheme = metadata['window_scheme']
    assert scheme['length_s'] == CLAPEmbeddingGenerator.WINDOW_SECONDS
    assert scheme['tail'] == 'end-aligned'
    assert scheme['truncation'] == 'rand_trunc'
    assert scheme['rms_gate'] == CLAPEmbeddingGenerator.RMS_GATE


def _real_track(library_file, index=0):
    from pathlib import Path

    with np.load(library_file, allow_pickle=False) as data:
        name = str(data['track_files'][index])
        stored = data['embeddings'][index]
        batch_size = validate_embeddings(library_file)['metadata']['window_scheme']['batch_size']
    return name, stored, batch_size, Path(config.mpd_music_directory)


def test_a_real_track_reproduces_its_stored_embedding(library_file):
    """
    Determinism across *runs*, not just within one: re-embedding a track
    reproduces the vector already on disk, bit for bit.  That is what makes the
    library incrementally extendable — a new album can be added without every
    existing vector shifting underneath the taste model.
    """
    name, stored, batch_size, music_dir = _real_track(library_file)
    generator = CLAPEmbeddingGenerator(batch_size=batch_size, workers=1)
    generator.load_model()

    pooled, _ = generator.embed_track(name, music_dir)
    assert np.array_equal(pooled, stored)


def test_bit_identity_is_a_property_of_the_file_and_the_batch_size(library_file):
    """
    The limit of the guarantee, asserted rather than assumed.

    GPU reductions depend on how many rows are in the batch, so a different
    `--batch-size` reproduces a track to ~1e-8 but not bit for bit.  This is why
    `batch_size` is recorded in the artifact's metadata, and why batches are
    filled from one track at a time: within a run, a track's vector cannot
    depend on which tracks happened to sit next to it.

    1e-8 is nowhere near musically meaningful — the point is that the claim in
    the docs is the claim the code makes.
    """
    name, stored, batch_size, music_dir = _real_track(library_file)
    other = 1 if batch_size != 1 else 2
    generator = CLAPEmbeddingGenerator(batch_size=other, workers=1)
    generator.load_model()

    pooled, _ = generator.embed_track(name, music_dir)
    assert float(pooled @ stored) == pytest.approx(1.0, abs=1e-6)
    assert np.abs(pooled - stored).max() < 1e-6
