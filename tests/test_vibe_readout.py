"""
The vibe readout (audit H1b, Stage 3).

Two of these are the point of the file.

`test_a_zero_session_vector_is_refused_even_though_the_bank_answers` is H1's
original defect caught in its new costume: `bank.top(zeros)` does not fail, it
returns a confident ranking of the bank's own baselines.  The test asserts both
halves — that the bank still answers, so nobody "fixes" it there, and that the
readout refuses anyway.

`test_the_drift_headline_is_a_count_of_words_the_listener_saw` pins the decision
that the drift figure is a count rather than the cosine H1 specified.  That was a
measurement, not a preference: over 40 real sessions the cosine's median is 0.988
and its p10 is 0.947 (§10d), so it reads as "0.99" almost always — a compressed
scale, which is precisely the shape of C5 and of the entropy heuristic H1
replaced.  The cosine is still computed; it is shown in `[I]`, labelled.
"""

import numpy as np
import pytest

from vibe_readout import (NO_BANK_LINE, UNSEEDED_LINE, VibeReadout,
                          format_descriptors)


@pytest.fixture
def readout(stub_bank):
    return VibeReadout(stub_bank, window=5)


def vector_for(bank, label, dimension=16):
    """A vector that points straight at one descriptor, so `top` is predictable."""
    return np.array(bank.text_embeddings[bank.labels.index(label)], dtype=np.float64)


# ── The unseeded gate ────────────────────────────────────────────────────────


def test_a_zero_session_vector_is_refused_even_though_the_bank_answers(readout,
                                                                       stub_bank):
    """
    Stage 2 made "nothing has played yet" a real state (L7), and z-scoring a zero
    vector does not fail loudly: every similarity is exactly 0, so every z-score
    is `−mean_d / std_d` — finite, deterministic and entirely plausible.  On the
    real 49-word bank it reads `shimmering · orchestral · serene`, about no music
    at all.

    The bank cannot defend against this; the arithmetic is valid.  The gate has
    to be here.
    """
    zeros = np.zeros(stub_bank.dimension)

    # The hazard is real: the bank does not refuse.
    assert len(stub_bank.top(zeros, 3)) == 3

    # The readout does.
    assert readout.descriptors(zeros, seeded=False) == []
    assert readout.descriptor_line(zeros, seeded=False) == UNSEEDED_LINE
    assert "nothing has played yet" in readout.descriptor_line(zeros, seeded=False)


def test_a_nonzero_vector_is_still_refused_while_unseeded(readout, stub_bank):
    """
    The gate is `seeded`, not "is the vector zero".  `SessionState.is_seeded()`
    is the authority on whether anything has fed the vector, and the readout
    defers to it rather than re-deriving the answer from the numbers.
    """
    vector = vector_for(stub_bank, 'driving')
    assert readout.descriptors(vector, seeded=False) == []
    assert readout.descriptors(vector, seeded=True) != []


def test_no_bank_says_so_rather_than_showing_nothing(stub_bank):
    """
    A missing bank is survivable — the player must still play — but it must not
    look like a session with no character.
    """
    readout = VibeReadout(None)
    vector = np.ones(4)
    assert readout.descriptor_line(vector, seeded=True) == NO_BANK_LINE
    assert readout.descriptors(vector, seeded=True) == []
    assert readout.drift(vector, seeded=True) is None


# ── The descriptors ──────────────────────────────────────────────────────────


def test_the_line_names_the_banks_top_descriptors(readout, stub_bank):
    vector = vector_for(stub_bank, 'nocturnal')
    line = readout.descriptor_line(vector, seeded=True)

    assert line.startswith("♪ ")
    assert "nocturnal" in line
    words = line[2:].split(" · ")
    assert len(words) == 3
    assert words[0] == 'nocturnal', "the aimed-at descriptor should rank first"
    assert all(word in stub_bank.labels for word in words)


def test_the_line_does_not_repeat_a_descriptor(readout, stub_bank):
    for label in stub_bank.labels:
        words = readout.descriptor_line(vector_for(stub_bank, label),
                                        seeded=True)[2:].split(" · ")
        assert len(set(words)) == len(words)


# ── The drift store ──────────────────────────────────────────────────────────


def test_nothing_is_recorded_while_unseeded(readout, stub_bank):
    zeros = np.zeros(stub_bank.dimension)
    for played in range(5):
        readout.observe(zeros, played, seeded=False)
    assert readout.drift(zeros, seeded=False) is None
    assert len(readout._snapshots) == 0


def test_one_snapshot_per_track_not_per_refresh(readout, stub_bank):
    """
    `observe` is called twice a second by the display loop, and must record only
    at track boundaries — otherwise "five tracks ago" would mean 2.5 seconds ago.
    """
    vector = vector_for(stub_bank, 'calm')
    for _ in range(20):
        readout.observe(vector, tracks_played=3, seeded=True)
    assert len(readout._snapshots) == 1

    readout.observe(vector, tracks_played=4, seeded=True)
    assert len(readout._snapshots) == 2


def test_drift_needs_something_to_compare_against(readout, stub_bank):
    vector = vector_for(stub_bank, 'calm')
    assert readout.drift(vector, seeded=True) is None, "no snapshots yet"

    readout.observe(vector, tracks_played=1, seeded=True)
    assert readout.drift(vector, seeded=True) is None, "one snapshot is not a span"

    readout.observe(vector, tracks_played=2, seeded=True)
    assert readout.drift(vector, seeded=True) is not None


def test_the_drift_headline_is_a_count_of_words_the_listener_saw(readout,
                                                                stub_bank):
    """
    `held` counts how many of the three words on screen now were also on screen
    `distance` tracks ago — a statement about something the listener actually
    read, needing no threshold and no vocabulary.
    """
    calm = vector_for(stub_bank, 'calm')
    for played in range(1, 7):
        readout.observe(calm, tracks_played=played, seeded=True)

    unchanged = readout.drift(calm, seeded=True)
    assert unchanged['held'] == 3
    assert unchanged['of'] == 3
    assert unchanged['distance'] == 5

    # Move somewhere else entirely; the words must stop being held.
    gritty = vector_for(stub_bank, 'gritty')
    moved = readout.drift(gritty, seeded=True)
    assert moved['held'] < 3
    assert moved['distance'] == 5


def test_the_window_reaches_exactly_as_far_back_as_it_says(readout, stub_bank):
    """
    Early in a session there are fewer than five tracks to compare against, so
    the line reports the distance it actually reached rather than implying a full
    window.
    """
    vector = vector_for(stub_bank, 'calm')
    for played in range(1, 4):
        readout.observe(vector, tracks_played=played, seeded=True)
    assert readout.drift(vector, seeded=True)['distance'] == 2

    for played in range(4, 12):
        readout.observe(vector, tracks_played=played, seeded=True)
    assert readout.drift(vector, seeded=True)['distance'] == 5, (
        "the window must stop growing at `window` tracks")


def test_the_cosine_is_still_measured_alongside_the_count(readout, stub_bank):
    """
    H1's original statistic is kept — it is a real measurement, just a compressed
    one — and reported in `[I]` where there is room to label its scale.
    """
    calm = vector_for(stub_bank, 'calm')
    for played in range(1, 7):
        readout.observe(calm, tracks_played=played, seeded=True)

    assert readout.drift(calm, seeded=True)['cosine'] == pytest.approx(1.0, abs=1e-5)
    assert readout.drift(vector_for(stub_bank, 'gritty'),
                         seeded=True)['cosine'] < 1.0


def test_a_new_session_is_not_a_continuation_of_the_old_one(readout, stub_bank):
    """
    `tracks_played` going backwards means `start_session()` ran.  The old
    snapshots describe a different evening, and a drift measured across the
    boundary would be a comparison between two unrelated things.
    """
    vector = vector_for(stub_bank, 'calm')
    for played in range(1, 7):
        readout.observe(vector, tracks_played=played, seeded=True)
    assert len(readout._snapshots) > 1

    readout.observe(vector, tracks_played=1, seeded=True)
    assert len(readout._snapshots) == 1
    assert readout.drift(vector, seeded=True) is None


def test_reset_forgets_everything(readout, stub_bank):
    vector = vector_for(stub_bank, 'calm')
    for played in range(1, 7):
        readout.observe(vector, tracks_played=played, seeded=True)
    readout.reset()
    assert readout.drift(vector, seeded=True) is None


# ── Formatting ───────────────────────────────────────────────────────────────


def test_the_two_lines_carry_the_words_the_drift_and_the_count(readout,
                                                               stub_bank):
    vector = vector_for(stub_bank, 'calm')
    for played in range(1, 7):
        readout.observe(vector, tracks_played=played, seeded=True)

    descriptors, drift = readout.lines(vector, seeded=True, tracks_played=6)
    assert descriptors.startswith("♪ ") and "calm" in descriptors
    assert drift.startswith("⟳ ")
    assert "3 of 3 held over 5 tracks" in drift
    assert "6 played" in drift


def test_the_drift_line_degrades_to_the_counter_alone(readout, stub_bank):
    """With nothing to compare against, the line says only what it knows."""
    zeros = np.zeros(stub_bank.dimension)
    assert readout.drift_line(zeros, seeded=False, tracks_played=0) == "⟳ 0 played"


def test_a_single_track_span_is_singular(readout, stub_bank):
    vector = vector_for(stub_bank, 'calm')
    readout.observe(vector, tracks_played=1, seeded=True)
    readout.observe(vector, tracks_played=2, seeded=True)
    assert "over 1 track ·" in readout.drift_line(vector, True, 2)


def test_the_inspector_form_shows_the_z_scores(readout, stub_bank):
    """`[I]`'s job is to show the numbers, so its descriptor rows carry them."""
    pairs = readout.descriptors(vector_for(stub_bank, 'driving'), seeded=True)
    rendered = format_descriptors(pairs)
    assert "driving" in rendered
    assert "+" in rendered or "-" in rendered
    assert format_descriptors([]) == "—"
