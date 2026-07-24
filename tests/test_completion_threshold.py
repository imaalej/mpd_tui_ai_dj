"""
The full-listen threshold — how much of a track counts as "listened through"
(audit B1).

The bar was lowered from 90% to 75%: a track heard for three-quarters of its
length is a genuine full listen.  The old computation took
`max(0.9·duration, duration − 10)`, and the `duration − 10` term made *long*
tracks stricter, not more lenient (a 4-minute track needed 3:50) — the opposite
of the intent — so it is gone.  A flat fraction also widens the window a 0.5 s
poll has to land in for the completion to be seen at all (audit B2).
"""

import pytest

from config import config
from main_tui import AdaptiveDJWithTUI

threshold = AdaptiveDJWithTUI._completion_threshold


def test_the_threshold_is_a_flat_fraction_of_the_duration():
    assert config.full_listen_fraction == 0.75
    assert threshold(240) == pytest.approx(180.0)   # a 4:00 track counts at 3:00
    assert threshold(200) == pytest.approx(150.0)


def test_there_is_no_duration_minus_ten_floor_making_long_tracks_stricter():
    """
    The regression the finding names: under the old formula a 4-minute track
    needed 3:50.  A flat fraction must fire well before the track's final
    seconds, on tracks of any length.
    """
    for duration in (120, 240, 360, 600):
        old_formula = max(0.9 * duration, duration - 10)
        assert threshold(duration) < old_formula
        assert threshold(duration) < duration - 10


def test_lowering_the_bar_widens_the_completion_catch_window():
    """
    B2 is mitigated as a side effect: the window [threshold, end] a poll must
    land in is now a quarter of the track, not the old ~10 s tail.
    """
    duration = 240
    window = duration - threshold(duration)
    assert window == pytest.approx(60.0)


def test_the_threshold_scales_monotonically():
    assert threshold(100) < threshold(200) < threshold(300)


def test_the_fraction_is_a_validated_config_knob():
    """It shapes behaviour without asserting a fact, so it is a knob — but a
    nonsensical value must still be refused by config.validate()."""
    original = config.full_listen_fraction
    try:
        config.full_listen_fraction = 1.5
        with pytest.raises(ValueError):
            config.validate()
        config.full_listen_fraction = 0.0
        with pytest.raises(ValueError):
            config.validate()
        config.full_listen_fraction = 0.75
        assert config.validate() is True
    finally:
        config.full_listen_fraction = original
