"""
Shared pytest fixtures.

`FakeMPD` is the centre of the Stage 2 harness (audit M1b).  It is built to the
semantics table that was *verified against a live MPD*, not to what the protocol
seems like it ought to do — because C1 (the queue that never refilled) existed
precisely because nobody checked, and a double built on the same assumptions
would reproduce the bug and pass.  `tests/test_fake_mpd.py` asserts the double
against that table row by row, so the harness itself is under test.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import config  # noqa: E402  (needs the path insert above)


@pytest.fixture(autouse=True)
def _isolate_state_files(tmp_path_factory, monkeypatch):
    """
    Point every persistent-state path at a temp directory, for every test.

    Found in Stage 4 while writing the persistence round-trips (M1c): the suite
    was writing to the developer's **live** `data/state/`.  `process_like()`
    calls `user_taste.save()` with no argument, which resolves to
    `config.taste_file`, so a single green run replaced a real listener's taste
    model with a fixture's and emptied `feedback_history.json` — the file Stage 3
    made the `♥` marks depend on (L4).

    A suite that destroys the state it is meant to protect is not green in any
    sense worth having.  Autouse rather than opt-in, because the leak was in a
    method three layers below the test that called it, and the next one will be
    too.  Tests that want to assert on a specific path still monkeypatch it
    themselves; this only changes where the default lands.
    """
    state = tmp_path_factory.mktemp("state")
    for key, name in (('taste_file', 'user_taste.npz'),
                      ('exploration_file', 'exploration_state.json'),
                      ('feedback_history_file', 'feedback_history.json'),
                      ('play_history_file', 'play_history.json')):
        monkeypatch.setattr(config, key, state / name, raising=True)
    return state


@pytest.fixture(autouse=True)
def _restore_stderr():
    """
    Importing `tui` installs a stderr interceptor process-wide.  Put the real
    one back after every test so pytest's own reporting is never captured.
    """
    original = sys.stderr
    yield
    sys.stderr = original


@pytest.fixture
def rng():
    """Seeded generator — these tests assert on numbers, so they must not drift."""
    return np.random.default_rng(20260722)


@pytest.fixture
def library(rng):
    """
    A TrackLibrary populated in memory with normalised random embeddings.

    Random vectors are fine *here* — nothing under test asks what the vectors
    mean, only how the arithmetic around them behaves.
    """
    from track_library import TrackLibrary

    lib = TrackLibrary()
    for i in range(64):
        vec = rng.standard_normal(lib.dimension)
        lib.track_to_embedding[f"artist/album/{i:02d}.flac"] = vec / np.linalg.norm(vec)
    lib._build_matrix()
    return lib


class FakeMPD:
    """
    An MPD that behaves the way the real one was measured to behave.

    Every rule below was run against MPD 0.24.0 / mpc 0.35 with `consume on` and
    `random`/`repeat`/`single` off, and the transcript is in PROJECT_AUDIT.md
    §M1.  The three that are easiest to get wrong, and that the application's
    correctness rests on:

      • Consume removes a track when you *leave* it, not when you start it, so
        the currently playing track is still queue position 1 and `len(queue)`
        during normal playback is exactly 2.  This is what makes the refill
        condition `< 2` with no `#N/M` parsing.
      • `next` off the last remaining track empties the queue and stops, and
        `add` to a stopped empty queue does **not** start it.  So a skip must add
        before it advances; advance-then-add ends the session silently.
      • `next` while *paused* consumes and **resumes playing** — it does not stay
        paused.  The audit assumed otherwise; the application re-pauses.

    `mpc pause` is idempotent rather than a toggle, `mpc next` on a stopped
    player is an error that changes nothing, and `mpc del N` past the end of the
    queue exits non-zero.  All three are modelled.
    """

    DEFAULT_DURATION = 200

    def __init__(self, tracks, modes=None, durations=None):
        self.known_tracks = list(tracks)
        self.queue = []
        self.state = 'stopped'
        self.position = 0
        self.volume = 100
        self.modes = dict(modes or {'repeat': 'off', 'random': 'off',
                                    'single': 'off', 'consume': 'off'})
        self.durations = durations or {}
        # Ordered log of every mutating operation, so a test can assert things
        # like "exactly one advance, and no play() anywhere in the skip path"
        # (audit C4) rather than inferring them from the end state.
        self.calls = []
        self.refuse = set()   # tracks this MPD will not accept, for the H7 path

    # ── helpers for tests ────────────────────────────────────────────────────

    @property
    def consuming(self):
        return self.modes.get('consume') == 'on'

    def duration_of(self, track):
        return self.durations.get(track, self.DEFAULT_DURATION)

    def finish_track(self):
        """Simulate a track reaching its natural end (MPD auto-advances)."""
        self.calls.append('finish')
        if self.state == 'stopped' or not self.queue:
            return
        if self.consuming:
            self.queue.pop(0)
        else:
            self.queue.append(self.queue.pop(0))
        self.position = 0
        self.state = 'playing' if self.queue else 'stopped'

    # ── the MPDController surface the application uses ───────────────────────

    def connect(self):
        return True

    def get_status(self):
        track = self.queue[0] if (self.queue and self.state != 'stopped') else None
        return {
            'state': self.state,
            'track_file': track,
            'artist': 'Test Artist',
            'album': 'Test Album',
            'title': Path(track).stem if track else 'Unknown Title',
            'position': self.position,
            'duration': self.duration_of(track) if track else 0,
            'volume': self.volume,
        }

    def get_modes(self):
        return dict(self.modes)

    def set_mode(self, name, value):
        self.calls.append(f'mode:{name}={value}')
        self.modes[name] = value
        return True

    def get_queue(self):
        return list(self.queue)

    def get_queue_length(self):
        return len(self.queue)

    def add_track(self, track_file):
        self.calls.append(f'add:{track_file}')
        if track_file not in self.known_tracks or track_file in self.refuse:
            return False
        # Verified: adding to a stopped queue leaves it stopped.
        self.queue.append(track_file)
        return True

    def delete_position(self, position):
        self.calls.append(f'del:{position}')
        if position < 1 or position > len(self.queue):
            return False        # mpc exits 1: "song number does not exist"
        self.queue.pop(position - 1)
        return True

    def clear_queue(self):
        self.calls.append('clear')
        self.queue = []
        self.state = 'stopped'
        self.position = 0
        return True

    def play(self):
        self.calls.append('play')
        self.state = 'playing' if self.queue else 'stopped'
        return True

    def pause(self):
        self.calls.append('pause')
        if self.state == 'playing':
            self.state = 'paused'
        return True

    def next_track(self):
        self.calls.append('next')
        if self.state == 'stopped':
            return False        # "MPD error: Not playing" — nothing changes
        if self.consuming:
            self.queue.pop(0)
        else:
            self.queue.append(self.queue.pop(0))
        self.position = 0
        # Verified: this resumes playback even from paused.
        self.state = 'playing' if self.queue else 'stopped'
        return True

    def list_all_tracks(self):
        return list(self.known_tracks)

    def fetch_track_tags(self, track_file):
        # Per-track and cached in the real controller.  Note there is no
        # `get_playlist_metadata` here: it was built from `mpc playlist`, and
        # with consume on a played track is gone from it — so it could never
        # answer for the tracks the session-history panel shows.
        self.calls.append(f'tags:{track_file}')
        return {'artist': 'Test Artist', 'album': 'Test Album',
                'title': Path(track_file).stem, 'file': track_file}

    def set_volume(self, volume):
        self.volume = max(0, min(100, volume))

    def volume_up(self, delta=5):
        self.set_volume(self.volume + delta)

    def volume_down(self, delta=5):
        self.set_volume(self.volume - delta)

    def seek_relative(self, delta):
        self.position = max(0, self.position + delta)

    def update_database(self):
        return True


@pytest.fixture
def fake_mpd(library):
    """
    A FakeMPD that knows exactly the tracks the `library` fixture holds, in the
    mode the DJ forces at session start (audit C2/D2).

    `consume on` is not a detail: with it off, `finish_track` rotates the queue
    instead of popping it, `len(playlist)` only ever grows, and every component
    below is running in the world C1 lived in.  Forcing it is `start_session`'s
    job, which these fixtures sit underneath — so the fixture has to establish
    the same ground, or the tests quietly measure the wrong system.
    """
    return FakeMPD(list(library.track_list),
                   modes={'repeat': 'off', 'random': 'off',
                          'single': 'off', 'consume': 'on'})


@pytest.fixture
def dj_parts(library, fake_mpd, rng):
    """
    The selection stack wired together against FakeMPD.

    Deliberately the real components — only MPD is a double.  The defects this
    stage exists to fix (C1, C4) all lived in how the real components sequence
    calls to MPD, so stubbing any of them out would test the wrong thing.
    """
    from exploration_controller import ExplorationController
    from feedback_handler import FeedbackHandler
    from queue_manager import QueueManager
    from session_state import SessionState
    from track_selector import TrackSelector
    from user_taste import UserTaste

    import types

    session_state = SessionState(dimension=library.dimension)
    session_state.start_session()
    user_taste = UserTaste(dimension=library.dimension)
    exploration = ExplorationController()
    selector = TrackSelector(library, rng=np.random.default_rng(4242))
    queue_manager = QueueManager(
        track_selector=selector,
        session_state=session_state,
        user_taste=user_taste,
        exploration_controller=exploration,
        mpd_controller=fake_mpd,
    )
    feedback = FeedbackHandler(
        session_state=session_state,
        user_taste=user_taste,
        exploration_controller=exploration,
        track_library=library,
    )
    return types.SimpleNamespace(
        library=library, mpd=fake_mpd, session_state=session_state,
        user_taste=user_taste, exploration=exploration, selector=selector,
        queue_manager=queue_manager, feedback=feedback,
    )


class FakeArtRenderer:
    """
    An album-art renderer that records what it was asked to draw.

    Injected so the widget tests can build the real tree without detecting image
    protocols and leaving a `ueberzugpp` child process behind — and so H8's
    geometry can be asserted against the calls it produces rather than inferred
    from the source.
    """

    def __init__(self, available=True):
        self.available = available
        self.protocol = None
        self.renders = []
        self.clears = 0
        self.shutdowns = 0
        self.art_for = {}

    def is_available(self):
        return self.available

    def find_album_art(self, track_file):
        return self.art_for.get(track_file)

    def render(self, path, x, y, width, height):
        self.renders.append({'path': path, 'x': x, 'y': y,
                             'width': width, 'height': height})

    def clear(self):
        self.clears += 1

    def shutdown(self):
        """The real renderer ends its child process here (audit L9)."""
        self.shutdowns += 1
        self.clears += 1
        self.available = False

    def force_redraw(self):
        pass


@pytest.fixture
def fake_art():
    return FakeArtRenderer()


@pytest.fixture
def stub_bank(rng):
    """A small DescriptorBank over an arbitrary space, for readout tests."""
    from descriptor_bank import DescriptorBank

    labels = ['calm', 'driving', 'nocturnal', 'orchestral', 'gritty', 'sparse']
    text = rng.standard_normal((len(labels), 16)).astype(np.float32)
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    return DescriptorBank(labels=labels, text_embeddings=text,
                          mean=np.zeros(len(labels), dtype=np.float32),
                          std=np.ones(len(labels), dtype=np.float32))


@pytest.fixture
def dj_stub(dj_parts, library):
    """
    An `AdaptiveDJWithTUI` stand-in carrying the real selection stack.

    The TUI reaches into `dj.session_state`, `dj.queue_manager`,
    `dj.feedback_handler`, `dj.mpd_controller`, `dj.track_selector`,
    `dj.user_taste`, `dj.exploration_controller`, `dj.track_library` and
    `dj.descriptor_bank`, plus `dj.skip_current_track()`.  All of them are the
    real components except MPD, which is `FakeMPD` — the same principle
    `dj_parts` is built on.
    """
    import types

    from descriptor_bank import DescriptorBank

    rng = np.random.default_rng(99)
    labels = ['calm', 'driving', 'nocturnal', 'orchestral', 'gritty', 'sparse']
    text = rng.standard_normal((len(labels), library.dimension)).astype(np.float32)
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    bank = DescriptorBank(labels=labels, text_embeddings=text,
                          mean=np.zeros(len(labels), dtype=np.float32),
                          std=np.ones(len(labels), dtype=np.float32))

    dj = types.SimpleNamespace(
        mpd_controller=dj_parts.mpd,
        session_state=dj_parts.session_state,
        user_taste=dj_parts.user_taste,
        exploration_controller=dj_parts.exploration,
        track_selector=dj_parts.selector,
        track_library=library,
        queue_manager=dj_parts.queue_manager,
        feedback_handler=dj_parts.feedback,
        descriptor_bank=bank,
    )
    dj.skips = []
    dj.skip_current_track = lambda: dj.skips.append(True)
    return dj


@pytest.fixture
def tui(dj_stub, fake_art):
    """
    The real widget tree, built against the stand-in DJ.

    Nothing in the suite constructed this before Stage 3 — which is how H1, C1
    and C4 all shipped under a green suite, and how a `WidgetError` on every
    terminal shorter than 33 rows shipped with them.
    """
    import signal as _signal

    from tui import AdaptiveDJTUI

    previous = _signal.getsignal(_signal.SIGWINCH)
    instance = AdaptiveDJTUI(dj_stub, art_renderer=fake_art)
    try:
        yield instance
    finally:
        # `_setup_urwid` installs a SIGWINCH handler process-wide.
        _signal.signal(_signal.SIGWINCH, previous)
        if instance._exit_pipe is not None:
            try:
                instance.loop.remove_watch_pipe(instance._exit_pipe)
            except Exception:
                pass


@pytest.fixture
def make_artifact(tmp_path, rng):
    """
    Write an embeddings artifact (audit §7 schema) to a temp file.

    Every keyword is overridable so the loader's validation can be tested one
    broken field at a time — that is the point of having a schema at all.
    """
    from embeddings_io import SCHEMA_VERSION

    def _make(
        name="track_embeddings.npz",
        n_tracks=12,
        dimension=512,
        windows_per_track=3,
        schema_version=SCHEMA_VERSION,
        model='laion/clap-htsat-unfused',
        track_files=None,
        anisotropy=None,
        drop_keys=(),
        **overrides,
    ):
        raw = rng.standard_normal((n_tracks, dimension)).astype(np.float32)
        if anisotropy is not None:
            # Push every embedding toward one direction, the way CLAP's own
            # space is skewed (C5).  Without this, random vectors are already
            # centred and the centring test proves nothing.
            bias = rng.standard_normal(dimension).astype(np.float32)
            bias /= np.linalg.norm(bias)
            raw = raw / np.linalg.norm(raw, axis=1, keepdims=True) + anisotropy * bias
        embeddings = raw / np.linalg.norm(raw, axis=1, keepdims=True)

        n_windows = n_tracks * windows_per_track
        window_matrix = rng.standard_normal((n_windows, dimension)).astype(np.float32)
        window_matrix /= np.linalg.norm(window_matrix, axis=1, keepdims=True)

        names = track_files or [f"artist/album/{i:02d}.flac" for i in range(n_tracks)]
        payload = {
            'schema_version': np.array(schema_version),
            'track_files': np.array(names, dtype=np.str_),
            'embeddings': embeddings,
            'centroid': embeddings.mean(axis=0).astype(np.float32),
            'window_offsets': np.arange(n_tracks + 1, dtype=np.int32) * windows_per_track,
            'windows': window_matrix,
            'metadata': np.array(json.dumps({
                'schema_version': int(schema_version),
                'model': model,
                'dimension': dimension,
            })),
        }
        payload.update(overrides)
        for key in drop_keys:
            payload.pop(key, None)

        path = tmp_path / name
        np.savez_compressed(path, **payload)
        return path

    return _make
