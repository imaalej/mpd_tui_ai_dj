"""
`[L]` on an already-liked track retracts the like (audit L8, Stage 4).

The display half arrived in Stage 3 — `SessionHistory.liked` is one set,
rehydrated from `feedback_history.json` (L4).  What was left was the modelling
question: what does `UserTaste` do with a retraction?

**Why it is a replay and not a negative update.**  `taste_update_like` is +0.1,
so subtracting 0.1 looks like symmetry.  It is not: `_update` is a *normalised*
EMA, `v ← normalise((1 − w)·v + w·e)`, and the inverse of that is not another
step of the same size.  Measured on the real 674-track library:

    like is the only event      cos(subtract-0.1, truth) = 0.000
    like is the 2nd event                                  0.9999
    like early, 3 events after                             0.9999
    settled model, 20 after                                0.9999

The middling cases are nearly right — and the first one is a total failure, for
a reason worth stating because it is the case a new listener meets first.  From
zero, one like normalises to the track itself.  Subtracting 0.1·e from e gives
0.9·e, which normalises straight back to e.  **Retract your only like and the
naive method leaves the taste model permanently and fully committed to the track
you just said you did not like** — the truth is an unseeded model, and cos to a
zero vector is 0.  No choice of magnitude fixes that, because the problem is
that a subtraction cannot un-seed.

Replaying the feedback history without the like has no such case.  It asserts
nothing and reproduces the incremental model bit for bit — 0 events of error at
10, 50, 200 and 1000 events (§10e).

**Why the replay is checked before it is used.**  It is only exact if the
history is a complete account of the model, and `_record_feedback` caps the
history at 1000 events.  Measured: while the history is complete the replay
reproduces the stored vector exactly; one event past the cap it is at cos 0.994
and at 1400 lifetime events, 0.923.  So a blind replay would move a long-time
listener's taste by more than the retraction ever could, silently.
`UserTaste.explains()` is that check, and when it says no the retraction is
display-only and says so.
"""

import numpy as np
import pytest

from exploration_controller import ExplorationController
from feedback_handler import FeedbackHandler
from session_history import SessionHistory
from session_state import SessionState
from user_taste import UserTaste


def _handler(library, tmp_path, monkeypatch, taste=None):
    from config import config
    monkeypatch.setattr(config, 'taste_file', tmp_path / 'user_taste.npz')
    monkeypatch.setattr(config, 'feedback_history_file', tmp_path / 'feedback.json')

    session = SessionState(dimension=library.dimension)
    session.start_session()
    taste = taste or UserTaste(dimension=library.dimension)
    return FeedbackHandler(
        session_state=session,
        user_taste=taste,
        exploration_controller=ExplorationController(),
        track_library=library,
    )


# ── the replay itself ────────────────────────────────────────────────────────


def test_replay_reproduces_the_incrementally_built_model(library):
    """
    The property the whole design rests on: recomputing from the history gives
    the same model as applying the events one at a time, exactly.
    """
    rng = np.random.default_rng(11)
    incremental = UserTaste(dimension=library.dimension)
    events = []
    for _ in range(200):
        track = library.track_list[rng.integers(len(library.track_list))]
        kind = str(rng.choice(['like', 'full_listen', 'skip'], p=[.15, .7, .15]))
        getattr(incremental, f'update_from_{kind}')(library.get_embedding(track))
        events.append({'type': kind, 'track': track})

    replayed = UserTaste(dimension=library.dimension)
    replayed.replay(events, library.get_embedding)

    assert np.array_equal(replayed.get_taste_vector(),
                          incremental.get_taste_vector())
    assert replayed.total_updates == incremental.total_updates
    assert replayed.like_count == incremental.like_count
    assert replayed.skip_count == incremental.skip_count
    assert replayed.full_listen_count == incremental.full_listen_count


def test_replay_of_an_empty_history_is_an_unseeded_model(library):
    model = UserTaste(dimension=library.dimension)
    model.update_from_like(library.get_embedding(library.track_list[0]))
    assert model.is_seeded()

    model.replay([], library.get_embedding)

    assert model.is_seeded() is False
    assert model.total_updates == 0


def test_replay_reports_events_it_could_not_apply(library):
    """
    A track that has left the library cannot be re-applied, and that is the one
    way a replay diverges from the vector it replaced.  It is counted rather
    than swallowed, so the caller can say so.
    """
    model = UserTaste(dimension=library.dimension)
    report = model.replay([
        {'type': 'like', 'track': library.track_list[0]},
        {'type': 'like', 'track': 'gone/from/the/library.flac'},
        {'type': 'nonsense', 'track': library.track_list[1]},
    ], library.get_embedding)

    assert report == {'applied': 1, 'missing_embedding': 1, 'unrecognised': 1}


def test_explains_is_true_while_the_history_is_complete(library):
    model = UserTaste(dimension=library.dimension)
    events = []
    for track in library.track_list[:20]:
        model.update_from_full_listen(library.get_embedding(track))
        events.append({'type': 'full_listen', 'track': track})

    assert model.explains(events, library.get_embedding) is True


def test_explains_is_false_when_the_history_has_been_truncated(library):
    """
    The case the cap creates.  Replaying here would be a silent rewrite of the
    taste vector, not a retraction.
    """
    model = UserTaste(dimension=library.dimension)
    events = []
    for track in library.track_list[:40]:
        model.update_from_full_listen(library.get_embedding(track))
        events.append({'type': 'full_listen', 'track': track})

    assert model.explains(events[10:], library.get_embedding) is False


def test_explains_does_not_mutate_the_model(library):
    model = UserTaste(dimension=library.dimension)
    model.update_from_like(library.get_embedding(library.track_list[0]))
    before = model.get_taste_vector()

    model.explains([{'type': 'skip', 'track': library.track_list[5]}],
                   library.get_embedding)

    assert np.array_equal(model.get_taste_vector(), before)
    assert model.total_updates == 1


# ── the retraction ───────────────────────────────────────────────────────────


def test_unliking_returns_the_model_to_where_it_was_before_the_like(library, tmp_path,
                                                                   monkeypatch):
    """
    The claim in one assertion: after retracting, the taste vector is what it
    would have been had the key never been pressed — not approximately.
    """
    handler = _handler(library, tmp_path, monkeypatch)
    for track in library.track_list[:6]:
        handler.process_full_listen(track)
    before = handler.user_taste.get_taste_vector()

    liked = library.track_list[9]
    handler.process_like(liked)
    assert not np.array_equal(handler.user_taste.get_taste_vector(), before)

    report = handler.process_unlike(liked)

    assert report['exact'] is True
    assert np.array_equal(handler.user_taste.get_taste_vector(), before)


def test_unliking_the_only_like_leaves_the_model_unseeded(library, tmp_path,
                                                          monkeypatch):
    """
    The case a subtraction cannot express.  From zero, one like normalises to
    the track; subtracting 0.1·e from e gives 0.9·e, which normalises back to e
    — so `[L]` twice on a fresh model would leave the listener's long-term taste
    pinned, at unit strength, to a track they had just un-liked.
    """
    handler = _handler(library, tmp_path, monkeypatch)
    liked = library.track_list[0]

    handler.process_like(liked)
    assert handler.user_taste.is_seeded()

    handler.process_unlike(liked)

    assert handler.user_taste.is_seeded() is False
    assert np.count_nonzero(handler.user_taste.get_taste_vector()) == 0


def test_the_naive_subtraction_would_not_have_done_that(library):
    """
    Pins the measurement the design rests on, so that "just subtract 0.1" cannot
    be reintroduced as an obvious simplification.
    """
    liked = library.get_embedding(library.track_list[0])

    naive = UserTaste(dimension=library.dimension)
    naive.update_from_like(liked)
    naive._update(liked, -0.1)

    assert naive.is_seeded() is True
    assert float(np.dot(naive.get_taste_vector(), liked)) == pytest.approx(1.0)


def test_unliking_removes_the_like_from_the_history(library, tmp_path, monkeypatch):
    handler = _handler(library, tmp_path, monkeypatch)
    liked = library.track_list[0]
    handler.process_like(liked)
    handler.process_full_listen(library.track_list[1])

    handler.process_unlike(liked)

    assert [e['type'] for e in handler.feedback_history] == ['full_listen']


def test_unliking_removes_every_like_for_that_track(library, tmp_path, monkeypatch):
    """One `♥` is shown however many times `[L]` was pressed, so one press takes
    it off."""
    handler = _handler(library, tmp_path, monkeypatch)
    liked = library.track_list[0]
    handler.process_like(liked)
    handler.process_like(liked)

    report = handler.process_unlike(liked)

    assert report['removed'] == 2
    assert handler.feedback_history == []
    assert handler.user_taste.is_seeded() is False


def test_unliking_leaves_other_tracks_likes_alone(library, tmp_path, monkeypatch):
    handler = _handler(library, tmp_path, monkeypatch)
    kept, dropped = library.track_list[0], library.track_list[1]
    handler.process_like(kept)
    handler.process_like(dropped)

    handler.process_unlike(dropped)

    assert [e['track'] for e in handler.feedback_history] == [kept]
    assert handler.user_taste.like_count == 1


def test_unliking_a_track_that_was_never_liked_does_nothing(library, tmp_path,
                                                            monkeypatch):
    handler = _handler(library, tmp_path, monkeypatch)
    handler.process_like(library.track_list[0])
    before = handler.user_taste.get_taste_vector()

    assert handler.process_unlike(library.track_list[5]) is None
    assert np.array_equal(handler.user_taste.get_taste_vector(), before)
    assert len(handler.feedback_history) == 1


def test_both_files_are_written_so_a_restart_is_consistent(library, tmp_path,
                                                           monkeypatch):
    """
    The taste model is authoritative at the next launch while the hearts are
    rehydrated from the history.  Saving one without the other gives a restart
    that shows `♥` for a like the model no longer holds.
    """
    handler = _handler(library, tmp_path, monkeypatch)
    liked = library.track_list[0]
    handler.process_like(liked)
    handler.save_feedback_history()

    handler.process_unlike(liked)

    reloaded = UserTaste(dimension=library.dimension)
    assert reloaded.load(tmp_path / 'user_taste.npz') is True
    assert reloaded.is_seeded() is False

    fresh = _handler(library, tmp_path, monkeypatch)
    fresh.load_feedback_history()
    history = SessionHistory()
    history.rehydrate_likes(fresh.feedback_history)
    assert history.is_liked(liked) is False


def test_a_history_that_cannot_account_for_the_model_falls_back_to_the_heart(
        library, tmp_path, monkeypatch):
    """
    The truncation case.  The like leaves the history and the `♥` goes, but the
    taste vector is not rebuilt from a partial account — measured at cos 0.923
    against the model it would replace, which is a change the listener did not
    ask for and could not see.
    """
    handler = _handler(library, tmp_path, monkeypatch)
    for track in library.track_list[:10]:
        handler.process_full_listen(track)
    liked = library.track_list[0]
    handler.process_like(liked)

    # Simulate the 1000-event cap having discarded the early history.
    handler.feedback_history = handler.feedback_history[6:]
    before = handler.user_taste.get_taste_vector()

    report = handler.process_unlike(liked)

    assert report['exact'] is False
    assert np.array_equal(handler.user_taste.get_taste_vector(), before)
    assert not any(e['type'] == 'like' for e in handler.feedback_history)


def test_the_session_like_count_comes_back_down(library, tmp_path, monkeypatch):
    handler = _handler(library, tmp_path, monkeypatch)
    handler.process_like(library.track_list[0])
    assert handler.get_session_stats()['likes'] == 1

    handler.process_unlike(library.track_list[0])
    assert handler.get_session_stats()['likes'] == 0


def test_a_like_with_no_embedding_is_not_recorded(library, tmp_path, monkeypatch):
    """
    `process_like` reports whether it recorded anything, so the TUI cannot draw
    a `♥` that `[L]` is then unable to take off.
    """
    handler = _handler(library, tmp_path, monkeypatch)
    assert handler.process_like('not/in/the/library.flac') is False
    assert handler.feedback_history == []
    assert handler.process_like(library.track_list[0]) is True


# ── the display half ─────────────────────────────────────────────────────────


def test_session_history_unlike_removes_the_heart():
    history = SessionHistory()
    history.note_playing('a.flac')
    history.like('a.flac')
    assert history.marks_for(history.entries[0]).startswith('♥')

    history.unlike('a.flac')
    assert not history.marks_for(history.entries[0]).startswith('♥')


def test_unliking_a_track_with_no_heart_is_harmless():
    history = SessionHistory()
    history.unlike('never/liked.flac')
    assert history.liked == set()


def test_the_l_key_toggles(tui, library):
    """
    Through the real `_handle_input`, against the real handler — the binding,
    the model and the panel in one path.
    """
    track = library.track_list[0]
    tui.current_status = {'track_file': track}

    tui._handle_input('l')
    assert tui.history.is_liked(track)
    assert tui.dj.user_taste.is_seeded()

    tui._handle_input('l')
    assert tui.history.is_liked(track) is False
    assert tui.dj.user_taste.is_seeded() is False


def test_the_heart_is_not_drawn_for_a_like_that_was_not_recorded(tui):
    """A track with no embedding produces no like event, so it must not wear a
    heart that nothing can retract."""
    tui.current_status = {'track_file': 'not/in/the/library.flac'}

    tui._handle_input('l')

    assert tui.history.is_liked('not/in/the/library.flac') is False


def test_the_heart_survives_a_toggle_the_model_could_not_perform(tui, library,
                                                                 monkeypatch):
    """
    If the retraction fails outright, the panel must keep claiming what the
    model still holds rather than quietly disagreeing with it.
    """
    track = library.track_list[0]
    tui.current_status = {'track_file': track}
    tui._handle_input('l')
    assert tui.history.is_liked(track)

    monkeypatch.setattr(tui.dj.feedback_handler, 'process_unlike',
                        lambda _t: None)
    tui._handle_input('l')

    assert tui.history.is_liked(track) is True
