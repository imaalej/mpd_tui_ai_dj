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

    def get_playlist_metadata(self):
        return {t: {'artist': 'Test Artist', 'album': 'Test Album',
                    'title': Path(t).stem, 'file': t} for t in self.queue}

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
