"""
The window scheme (audit C3).

The defect this replaces: each track's embedding described roughly ten seconds
sampled at *random* from it, so a track was less recognisable as itself than a
stranger was, 36% of the time.  The replacement covers the whole track with
non-overlapping windows and an end-aligned tail.

These tests are arithmetic and gating only — they need no model and no audio.
The acceptance tests that do need the model live in `test_clap_pipeline.py`.
"""

import numpy as np
import pytest

from embedding_generator import CLAP_AVAILABLE, CLAPEmbeddingGenerator, window_bounds

W = 10  # a "window" of ten units — the arithmetic does not care about the unit


# ------------------------------------------------------------------- tiling

def test_an_exact_multiple_tiles_without_a_tail():
    assert window_bounds(30, W) == [(0, 10), (10, 20), (20, 30)]


def test_a_remainder_produces_an_end_aligned_tail():
    """
    The last window covers the *last* ten seconds, overlapping its predecessor.
    Dropping the tail loses the ending; zero-padding it is worse — CLAP maps
    silence to a consistent direction that would then contaminate the pool.
    """
    bounds = window_bounds(25, W)
    assert bounds == [(0, 10), (10, 20), (15, 25)]
    assert bounds[-1][1] == 25


def test_every_sample_is_covered_by_some_window():
    for length in range(1, 200):
        covered = np.zeros(length, dtype=bool)
        for start, end in window_bounds(length, W):
            covered[start:end] = True
        assert covered.all(), f"length {length} left a gap"


def test_only_the_tail_ever_overlaps():
    for length in range(W, 200):
        bounds = window_bounds(length, W)
        starts = [s for s, _ in bounds]
        # Every window but the last begins exactly one window after its
        # predecessor; only the end-aligned tail may sit closer.
        for previous, current in zip(starts, starts[1:-1]):
            assert current - previous == W


def test_a_track_shorter_than_one_window_yields_one_window():
    assert window_bounds(4, W) == [(0, 4)]
    assert window_bounds(W, W) == [(0, W)]


def test_an_empty_waveform_yields_nothing():
    assert window_bounds(0, W) == []


def test_window_count_is_what_the_cost_estimate_assumed():
    """~24 windows for a four-minute track at 10 s — the number M8 sized the
    batching prerequisite against."""
    four_minutes = 240 * 48000
    assert len(window_bounds(four_minutes, 10 * 48000)) == 24


# --------------------------------------------------------------- silence gate

@pytest.fixture
def generator():
    if not CLAP_AVAILABLE:
        pytest.skip("torch/transformers not installed")
    # No model is loaded — slice_windows is pure signal processing.
    return CLAPEmbeddingGenerator(device='cpu')


def _signal(generator, blocks):
    """Concatenate one window per amplitude in `blocks`."""
    import torch

    rng = np.random.default_rng(4)
    parts = [
        (rng.standard_normal(generator.WINDOW_SAMPLES) * amplitude).astype(np.float32)
        for amplitude in blocks
    ]
    return torch.from_numpy(np.concatenate(parts))


def test_silent_windows_are_dropped(generator):
    waveform = _signal(generator, [0.3, 0.0, 0.3])
    windows, rms, total, gated = generator.slice_windows(waveform)

    assert total == 3
    assert gated == 1
    assert len(windows) == 2
    assert min(rms) >= generator.RMS_GATE


def test_a_wholly_quiet_track_keeps_every_window(generator):
    """
    A genuinely quiet ambient piece is still a track.  Returning nothing would
    turn it into a silent data loss — the exact failure mode C3 documents for
    the 16 tracks the original run dropped without a list.
    """
    waveform = _signal(generator, [1e-6, 1e-6, 1e-6])
    windows, _, total, gated = generator.slice_windows(waveform)

    assert total == 3
    assert gated == 0
    assert len(windows) == 3


def test_loud_content_is_never_gated(generator):
    waveform = _signal(generator, [0.5, 0.4, 0.6])
    _, _, total, gated = generator.slice_windows(waveform)
    assert (total, gated) == (3, 0)


def test_the_gate_sits_in_the_measured_gap(generator):
    """
    The threshold is not a taste decision: over 1,475 windows of this library,
    1.8% sat at RMS ~4e-5 (digital silence) and real content began around 0.04.
    Anything in between separates the two populations; 0.01 is in the middle of
    that empty band.
    """
    assert 4e-5 < generator.RMS_GATE < 0.04
