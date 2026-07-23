"""
Anti-repetition survives a restart (audit M6b).

`clear_history()` carried the comment *"Don't clear play_history to maintain
long-term anti-repetition"* — but nothing ever saved or loaded `play_history`,
and `clear_history()` itself was never called.  Both histories were rebuilt empty
on every launch, so the README's "recently played tracks are excluded for at
least 20 songs" quietly meant "until you restart the program".

This becomes visible rather than merely wrong once the session-history panel
lands in Stage 3: the panel will say a track played twenty minutes ago while the
DJ queues it again.
"""

import json

import numpy as np
import pytest

from config import config
from track_selector import TrackSelector


def _play(selector, tracks):
    for track in tracks:
        selector._record_selection(track)


def test_play_history_round_trips(library, tmp_path):
    path = tmp_path / 'play_history.json'
    original = TrackSelector(library)
    _play(original, library.track_list[:25])
    original.save(path)

    restored = TrackSelector(library)
    assert restored.load(path) is True

    assert restored.current_index == original.current_index
    assert restored.play_history == original.play_history
    assert list(restored.recent_history) == list(original.recent_history)


def test_the_replay_gap_is_still_closed_after_a_restart(library, tmp_path):
    """
    The behavioural version, which is the one that matters: a track played just
    before the restart must still be penalised afterwards.
    """
    path = tmp_path / 'play_history.json'
    recent = library.track_list[0]

    before = TrackSelector(library)
    _play(before, library.track_list[:5])
    before.save(path)

    after = TrackSelector(library)
    after.load(path)

    assert after._calculate_anti_repetition(recent) == pytest.approx(0.1)
    assert after._calculate_anti_repetition(library.track_list[60]) == 1.0


def test_a_restored_track_inside_the_gap_is_excluded_from_selection(library, tmp_path):
    path = tmp_path / 'play_history.json'
    before = TrackSelector(library)
    _play(before, library.track_list[:10])
    before.save(path)

    after = TrackSelector(library, rng=np.random.default_rng(1))
    after.load(path)

    picks = set()
    for _ in range(20):
        picks.add(after.select_track(
            session_vector=library.get_embedding(library.track_list[0]),
            taste_vector=np.zeros(library.dimension),
            weights={'session_weight': 1.0, 'taste_weight': 0.0,
                     'novelty_weight': 0.0, 'anti_repetition_weight': 0.0},
            exploration=0.4))

    assert not (picks & set(library.track_list[:10]))


def test_a_missing_file_is_a_fresh_start_not_an_error(library, tmp_path):
    selector = TrackSelector(library)
    assert selector.load(tmp_path / 'nothing.json') is False
    assert selector.current_index == 0


def test_a_corrupt_file_is_survived(library, tmp_path):
    path = tmp_path / 'play_history.json'
    path.write_text("{ this is not json")

    selector = TrackSelector(library)
    assert selector.load(path) is False


def test_the_file_is_plain_json_a_human_can_read(library, tmp_path):
    """
    It is state a user might reasonably want to inspect or clear by hand, and an
    npz of two dicts would buy nothing.
    """
    path = tmp_path / 'play_history.json'
    selector = TrackSelector(library)
    _play(selector, library.track_list[:3])
    selector.save(path)

    payload = json.loads(path.read_text())
    assert set(payload) == {'current_index', 'play_history', 'recent_history'}
    assert payload['current_index'] == 3


def test_recent_history_is_capped_on_load(library, tmp_path):
    """A file grown past the configured window must not widen it on load."""
    path = tmp_path / 'play_history.json'
    path.write_text(json.dumps({
        'current_index': 500,
        'play_history': {},
        'recent_history': [f"t{i}.flac" for i in range(500)],
    }))

    selector = TrackSelector(library)
    selector.load(path)

    assert len(selector.recent_history) == config.recent_history_size


def test_persistence_saves_and_loads_the_selector(library, tmp_path, monkeypatch):
    """The wiring, not just the method — M6b is 'checkpointed with the rest'."""
    from exploration_controller import ExplorationController
    from feedback_handler import FeedbackHandler
    from persistence import Persistence
    from user_taste import UserTaste

    monkeypatch.setattr(config, 'play_history_file', tmp_path / 'play_history.json')
    monkeypatch.setattr(config, 'taste_file', tmp_path / 'taste.npz')
    monkeypatch.setattr(config, 'exploration_file', tmp_path / 'exploration.json')
    monkeypatch.setattr(config, 'feedback_history_file', tmp_path / 'feedback.json')

    selector = TrackSelector(library)
    _play(selector, library.track_list[:7])

    taste = UserTaste(dimension=library.dimension)
    feedback = FeedbackHandler(None, taste, ExplorationController(), library)
    Persistence(taste, ExplorationController(), feedback, selector).save_all()

    assert (tmp_path / 'play_history.json').exists()

    restored = TrackSelector(library)
    Persistence(UserTaste(dimension=library.dimension), ExplorationController(),
                FeedbackHandler(None, taste, ExplorationController(), library),
                restored).load_all()

    assert restored.current_index == 7
