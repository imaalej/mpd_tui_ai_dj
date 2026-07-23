"""
The three state files that had no round-trip test (audit M1c, Stage 4).

`play_history.json` already had eight tests, including corruption and a missing
file — `test_play_history_persistence.py`.  The other three did not:

  • `feedback_history.json` — `save_feedback_history()` and
    `load_feedback_history()` were never called directly by the suite at all.
    This is the one that matters most now: Stage 3 made the `♥` marks depend on
    this file surviving a restart (L4), so the file is load-bearing for
    something the user can see, and un-liking (L8) reads it back to rebuild the
    taste model.
  • `exploration_state.json` — no round-trip test.  It carries
    `consecutive_skips`, which stopped being a spare counter in Stage 2 and
    became the input the `[N]` escalation's turnover target is chosen from.
  • `user_taste.npz` — one test, and only of an *unseeded* model, so nothing
    covered the case where there is actually a vector to lose.

The shape follows the play-history suite deliberately: round-trip, then the
*behavioural* consequence of the round-trip, then a missing file, then a corrupt
one.  A round-trip test that only compares the fields it just wrote proves the
serialiser agrees with itself; what these files are for is that something
downstream still works after a restart.
"""

import json

import numpy as np
import pytest

from config import config
from exploration_controller import ExplorationController
from feedback_handler import FeedbackHandler
from persistence import Persistence
from session_history import SessionHistory
from track_selector import TrackSelector
from user_taste import UserTaste


# ── feedback_history.json ────────────────────────────────────────────────────


def _handler(library, taste=None):
    return FeedbackHandler(
        session_state=None,
        user_taste=taste or UserTaste(dimension=library.dimension),
        exploration_controller=ExplorationController(),
        track_library=library,
    )


def test_feedback_history_round_trips(library, tmp_path):
    path = tmp_path / 'feedback_history.json'
    original = _handler(library)
    original.process_like(library.track_list[0])
    original._record_feedback('skip', library.track_list[1])
    original._record_feedback('full_listen', library.track_list[2])
    original.save_feedback_history(path)

    restored = _handler(library)
    assert restored.load_feedback_history(path) is True

    assert restored.feedback_history == original.feedback_history
    assert [e['type'] for e in restored.feedback_history] == [
        'like', 'skip', 'full_listen']


def test_the_hearts_come_back_after_a_restart(library, tmp_path):
    """
    L4's behavioural claim, asserted across the file rather than in memory.

    `SessionHistory.rehydrate_likes()` is the only thing that puts a `♥` on a
    track from a previous session, and it reads exactly this file.  Before Stage
    3 the likes were on disk while the set that drew them was in memory only, so
    they vanished on restart with the data that produced them sitting untouched.
    """
    path = tmp_path / 'feedback_history.json'
    liked, skipped = library.track_list[0], library.track_list[1]

    before = _handler(library)
    before.process_like(liked)
    before._record_feedback('skip', skipped)
    before.save_feedback_history(path)

    after = _handler(library)
    after.load_feedback_history(path)

    history = SessionHistory()
    history.rehydrate_likes(after.feedback_history)

    assert history.is_liked(liked)
    # A skip is a fact about one evening, not about the track.  Only likes are
    # persistent, and only likes rehydrate.
    assert not history.is_liked(skipped)


def test_a_missing_feedback_file_is_a_fresh_start_not_an_error(library, tmp_path):
    handler = _handler(library)
    assert handler.load_feedback_history(tmp_path / 'nothing.json') is False
    assert handler.feedback_history == []


def test_a_corrupt_feedback_file_is_survived(library, tmp_path):
    """
    It is read at startup, before the TUI exists, so an exception here is a DJ
    that will not launch because of a file nothing plays from.
    """
    path = tmp_path / 'feedback_history.json'
    path.write_text("[ {'not': 'json'")

    handler = _handler(library)
    assert handler.load_feedback_history(path) is False
    assert handler.feedback_history == []


def test_the_feedback_file_is_plain_json_a_human_can_read(library, tmp_path):
    path = tmp_path / 'feedback_history.json'
    handler = _handler(library)
    handler.process_like(library.track_list[0])
    handler.save_feedback_history(path)

    payload = json.loads(path.read_text())
    assert set(payload[0]) == {'timestamp', 'type', 'track'}
    assert payload[0]['track'] == library.track_list[0]


# ── exploration_state.json ───────────────────────────────────────────────────


def test_exploration_state_round_trips(tmp_path):
    path = tmp_path / 'exploration_state.json'
    original = ExplorationController()
    for _ in range(3):
        original.increase_exploration()
    for _ in range(2):
        original.decrease_exploration()
    original.increase_exploration()
    original.save(path)

    restored = ExplorationController()
    assert restored.load(path) is True

    assert restored.exploration == pytest.approx(original.exploration)
    assert restored.consecutive_skips == original.consecutive_skips
    assert restored.consecutive_listens == original.consecutive_listens
    assert restored.total_skips == original.total_skips
    assert restored.total_listens == original.total_listens


def test_the_restored_weights_are_the_weights_that_were_saved(tmp_path):
    """
    The consequence, not the fields.  `exploration` is what shifts weight from
    session/taste onto novelty, so a round-trip that loses it changes what gets
    selected on the next launch.
    """
    path = tmp_path / 'exploration_state.json'
    original = ExplorationController()
    for _ in range(6):
        original.increase_exploration()
    original.save(path)

    restored = ExplorationController()
    restored.load(path)

    assert restored.get_weights(taste_updates=50) == pytest.approx(
        original.get_weights(taste_updates=50))
    assert restored.get_weights() != pytest.approx(
        ExplorationController().get_weights())


def test_the_skip_run_length_survives_a_restart(tmp_path):
    """
    `consecutive_skips` stopped being spare in Stage 2: it is the run length the
    `[N]` escalation picks its turnover target from, so losing it on restart
    silently resets an escalation the listener is in the middle of.
    """
    path = tmp_path / 'exploration_state.json'
    original = ExplorationController()
    for _ in range(3):
        original.increase_exploration()
    original.save(path)

    restored = ExplorationController()
    restored.load(path)
    assert restored.consecutive_skips == 3


def test_a_missing_exploration_file_is_a_fresh_start_not_an_error(tmp_path):
    controller = ExplorationController()
    assert controller.load(tmp_path / 'nothing.json') is False
    assert controller.exploration == config.exploration_initial


def test_a_corrupt_exploration_file_is_survived(tmp_path):
    path = tmp_path / 'exploration_state.json'
    path.write_text("not json at all")

    controller = ExplorationController()
    assert controller.load(path) is False
    assert controller.exploration == config.exploration_initial


def test_an_exploration_file_missing_a_key_leaves_the_state_untouched(tmp_path):
    """
    A file written by an older build, or truncated by a full disk.

    Found by writing this test: `load()` assigned field by field, so a file
    carrying `exploration` and nothing else left the controller with that value
    from disk and its counters at their defaults — while returning False, so the
    caller believed nothing had been read.  A load now either replaces the state
    or leaves it alone.
    """
    path = tmp_path / 'exploration_state.json'
    path.write_text(json.dumps({'exploration': 0.5}))

    controller = ExplorationController()
    assert controller.load(path) is False
    assert controller.exploration == config.exploration_initial
    assert controller.consecutive_skips == 0


def test_a_taste_file_missing_a_key_leaves_the_model_untouched(library, tmp_path):
    """
    The same defect on the other file, and worse there: the β ramp reads
    `total_updates`, so a vector loaded beside default counters would arrive
    with none of its weight earned.
    """
    path = tmp_path / 'user_taste.npz'
    seeded = UserTaste(dimension=library.dimension)
    seeded.update_from_like(library.get_embedding(library.track_list[0]))
    np.savez_compressed(path, taste_vector=seeded.get_taste_vector())

    model = UserTaste(dimension=library.dimension)
    assert model.load(path) is False
    assert model.is_seeded() is False
    assert model.total_updates == 0


# ── user_taste.npz ───────────────────────────────────────────────────────────


def test_a_seeded_taste_model_round_trips(library, tmp_path):
    """
    The existing test covers an unseeded model, where there is no vector to
    lose.  This is the case with something in it.
    """
    path = tmp_path / 'user_taste.npz'
    original = UserTaste(dimension=library.dimension)
    original.update_from_like(library.get_embedding(library.track_list[0]))
    original.update_from_full_listen(library.get_embedding(library.track_list[1]))
    original.update_from_skip(library.get_embedding(library.track_list[2]))
    original.save(path)

    restored = UserTaste(dimension=library.dimension)
    assert restored.load(path) is True

    assert np.allclose(restored.get_taste_vector(), original.get_taste_vector())
    assert restored.is_seeded()
    assert restored.total_updates == 3
    assert restored.like_count == 1
    assert restored.full_listen_count == 1
    assert restored.skip_count == 1


def test_the_restored_taste_ranks_tracks_the_same_way(library, tmp_path):
    """
    What the vector is *for*.  A round-trip that rounded it would still pass a
    field comparison at the wrong tolerance; this fails if the ordering moves.
    """
    path = tmp_path / 'user_taste.npz'
    original = UserTaste(dimension=library.dimension)
    for track in library.track_list[:5]:
        original.update_from_like(library.get_embedding(track))
    original.save(path)

    restored = UserTaste(dimension=library.dimension)
    restored.load(path)

    def ranking(model):
        v = model.get_taste_vector()
        return sorted(library.track_list,
                      key=lambda t: -float(np.dot(library.get_embedding(t), v)))

    assert ranking(restored) == ranking(original)


def test_a_taste_file_saved_unnormalised_is_repaired_on_load(library, tmp_path):
    """
    The loader claims to fix a non-unit vector.  A vector at 3× length would
    otherwise carry 3× its share of every score.
    """
    path = tmp_path / 'user_taste.npz'
    model = UserTaste(dimension=library.dimension)
    model.update_from_like(library.get_embedding(library.track_list[0]))
    model.taste_vector = model.taste_vector * 3.0
    model.save(path)

    restored = UserTaste(dimension=library.dimension)
    restored.load(path)
    assert float(np.linalg.norm(restored.get_taste_vector())) == pytest.approx(1.0)


def test_a_missing_taste_file_is_a_fresh_start_not_an_error(tmp_path):
    model = UserTaste()
    assert model.load(tmp_path / 'nothing.npz') is False
    assert model.is_seeded() is False


def test_a_corrupt_taste_file_is_survived(tmp_path):
    path = tmp_path / 'user_taste.npz'
    path.write_bytes(b"PK\x03\x04 this is not an npz")

    model = UserTaste()
    assert model.load(path) is False
    assert model.is_seeded() is False


# ── all four together ────────────────────────────────────────────────────────


def _wire(library, selector=None, taste=None, exploration=None, feedback=None):
    taste = taste or UserTaste(dimension=library.dimension)
    exploration = exploration or ExplorationController()
    return Persistence(
        user_taste=taste,
        exploration_controller=exploration,
        feedback_handler=feedback or FeedbackHandler(
            session_state=None, user_taste=taste,
            exploration_controller=exploration, track_library=library),
        track_selector=selector or TrackSelector(library),
    )


def test_save_all_then_load_all_restores_every_file(library, tmp_path, monkeypatch):
    """
    The wiring, end to end.  `save_all()` writes four files and `load_all()`
    reads four; a component dropped from either list fails silently in exactly
    the way M6b did, where the comment claimed a persistence nothing performed.
    """
    for key, name in (('taste_file', 'user_taste.npz'),
                      ('exploration_file', 'exploration_state.json'),
                      ('feedback_history_file', 'feedback_history.json'),
                      ('play_history_file', 'play_history.json')):
        monkeypatch.setattr(config, key, tmp_path / name)

    from session_state import SessionState

    session = SessionState(dimension=library.dimension)
    session.start_session()
    taste = UserTaste(dimension=library.dimension)
    exploration = ExplorationController()
    feedback = FeedbackHandler(session, taste, exploration, library)
    selector = TrackSelector(library)

    feedback.process_like(library.track_list[0])
    feedback.process_skip(library.track_list[1])
    for track in library.track_list[:4]:
        selector._record_selection(track)

    _wire(library, selector, taste, exploration, feedback).save_all(quiet=True)

    for name in ('user_taste.npz', 'exploration_state.json',
                 'feedback_history.json', 'play_history.json'):
        assert (tmp_path / name).exists(), f"{name} was not written"

    taste2 = UserTaste(dimension=library.dimension)
    exploration2 = ExplorationController()
    feedback2 = FeedbackHandler(None, taste2, exploration2, library)
    selector2 = TrackSelector(library)

    assert _wire(library, selector2, taste2, exploration2,
                 feedback2).load_all() is True

    assert np.allclose(taste2.get_taste_vector(), taste.get_taste_vector())
    assert exploration2.consecutive_skips == exploration.consecutive_skips
    assert feedback2.feedback_history == feedback.feedback_history
    assert selector2.current_index == selector.current_index


def test_load_all_on_an_empty_directory_reports_no_previous_state(library, tmp_path,
                                                                  monkeypatch):
    for key, name in (('taste_file', 'user_taste.npz'),
                      ('exploration_file', 'exploration_state.json'),
                      ('feedback_history_file', 'feedback_history.json'),
                      ('play_history_file', 'play_history.json')):
        monkeypatch.setattr(config, key, tmp_path / name)

    assert _wire(library).load_all() is False


def test_a_checkpoint_is_readable_by_the_next_launch(library, tmp_path, monkeypatch):
    """
    H3's periodic save is the copy that survives a kill, so it has to produce
    files a cold start can actually read — `save_all(quiet=True)` differs from
    the exit save only in what it prints, and this is what pins that.
    """
    for key, name in (('taste_file', 'user_taste.npz'),
                      ('exploration_file', 'exploration_state.json'),
                      ('feedback_history_file', 'feedback_history.json'),
                      ('play_history_file', 'play_history.json')):
        monkeypatch.setattr(config, key, tmp_path / name)

    taste = UserTaste(dimension=library.dimension)
    exploration = ExplorationController()
    feedback = FeedbackHandler(None, taste, exploration, library)
    feedback.process_like(library.track_list[3])
    _wire(library, None, taste, exploration, feedback).save_all(quiet=True)

    taste2 = UserTaste(dimension=library.dimension)
    feedback2 = FeedbackHandler(None, taste2, ExplorationController(), library)
    assert _wire(library, None, taste2, None, feedback2).load_all() is True
    assert taste2.is_seeded()
    assert len(feedback2.feedback_history) == 1
