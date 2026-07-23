"""
Config invariants (audit L9, D6).

The weight-sum invariant is the one that matters: it was enforced with a bare
`assert`, so `python -O` removed it, and the time-context bonus broke it anyway
by adding 0.15 * time_sim on top of weights that had already been normalised.
"""

import copy

import pytest

from config import Config, config


def test_default_weights_sum_to_one():
    assert config.validate() is True
    total = (config.weight_session_similarity
             + config.weight_taste_similarity
             + config.weight_novelty
             + config.weight_anti_repetition)
    assert total == pytest.approx(1.0)


def test_validate_raises_not_asserts_on_bad_weights():
    """
    Must raise a real exception, not an AssertionError — an AssertionError would
    vanish under `python -O` and take the invariant with it.
    """
    broken = copy.copy(config)
    broken.weight_novelty = config.weight_novelty + 0.2

    with pytest.raises(ValueError) as exc:
        broken.validate()
    assert "sum to 1.0" in str(exc.value)


def test_validate_rejects_out_of_range_weight():
    broken = copy.copy(config)
    broken.weight_novelty = -0.1
    with pytest.raises(ValueError):
        broken.validate()


def test_validate_rejects_inverted_exploration_bounds():
    broken = copy.copy(config)
    broken.exploration_min = 0.9
    broken.exploration_max = 0.2
    with pytest.raises(ValueError):
        broken.validate()


@pytest.mark.parametrize("key", [
    "enable_time_context",
    "enable_day_context",
    "time_periods",
    "weekdays",
    "time_update_rate_like",
    "time_update_rate_listen",
    "time_context_weight",
    "weekday_exploration_modifier",
    "weekend_exploration_modifier",
    "context_file",
])
def test_time_context_config_keys_are_gone(key):
    """D6: the subsystem is removed, so its config surface must go with it."""
    assert not hasattr(config, key)


def test_log_file_is_configured():
    """L5: there must be a log path for the console tee to write to."""
    fresh = Config()
    assert fresh.log_file.name == "dj.log"
    assert fresh.log_file.parent == fresh.data_dir
