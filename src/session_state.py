"""
Session State Model - Dynamic short-term vibe representation
Tracks evolving session direction separate from long-term taste.

Two Stage 2 changes worth knowing before reading:

**The session vector starts at zero, not at `randn`** (audit L7).  It used to be
seeded with a random direction at every startup, which meant the first queue of
every session was an arbitrary neighbourhood presented as a vibe.  Zero is the
honest representation of "nothing has played yet": the candidate pool skips the
session half entirely while it holds, so the first track is drawn uniformly at
random from the library — which is what "no information" actually implies — and
the vector is seeded from that track's own embedding the moment it starts.

**Skips repel from the skip-run centroid by a solved magnitude** (audit H9).
The old `penalize_similar()` subtracted a fixed 0.15 of the skipped track, which
moves 2.9% of the candidate pool: twenty presses before anything you would hear
changes.  `repel_from_skip_run()` instead solves for how far to move, against a
turnover target that escalates with the length of the consecutive-skip run, and
projects back onto the manifold once the displacement is large enough to need it.
`force_shift()` — the old `[V]` — is gone; see manifold.py and audit D8/H9.
"""

import numpy as np
from typing import Optional
from collections import deque
from config import config

import manifold


class SessionState:
    """
    Maintains dynamic session vector representing current listening vibe.
    Uses exponential decay and recent track averaging.
    Separate from long-term user taste.
    """

    def __init__(self, dimension: int = None):
        self.dimension = dimension or config.embedding_dimension
        self.session_vector = np.zeros(self.dimension, dtype=np.float64)
        self.recent_tracks = deque(maxlen=config.session_influence_window)
        self.tracks_played = 0
        self.session_started = False

        # Embeddings of the tracks rejected in the current consecutive-skip run,
        # and the session vector as it stood when that run began.  The centroid
        # of the run is what a skip repels *from* — more evidence than the single
        # last track, and less sensitive to one atypical song.  The anchor is
        # what the reported turnover is measured against, so the number the user
        # reads answers "how much has changed since I started skipping".
        self.skip_run: list = []
        self.skip_run_anchor: Optional[np.ndarray] = None

    def is_seeded(self) -> bool:
        """True once any real track has fed the session vector."""
        return bool(np.any(self.session_vector))

    def start_session(self, initial_track_embedding: Optional[np.ndarray] = None):
        """
        Start a new session.

        With no initial track the vector stays at zero — "nothing is playing
        yet" — rather than being filled with a random direction.  `seed()` picks
        it up as soon as MPD reports a track.
        """
        if initial_track_embedding is not None:
            self.session_vector = manifold.normalise(initial_track_embedding)
        else:
            self.session_vector = np.zeros(self.dimension, dtype=np.float64)

        self.recent_tracks.clear()
        self.tracks_played = 0
        self.session_started = True
        self.clear_skip_run()

        print("Session started", file=__import__("sys").stderr)

    def seed(self, track_embedding: np.ndarray) -> bool:
        """
        Adopt a track's embedding as the session vector, but only while unseeded.

        Called when a track starts playing.  Once there is a real vibe this is a
        no-op, so a track change cannot overwrite the trajectory the EMA built.
        """
        if self.is_seeded():
            return False
        self.session_vector = manifold.normalise(track_embedding)
        return True

    def update(self, track_embedding: np.ndarray):
        """
        Update session state with a fully-listened track.

        `session' = normalise(decay·session + (1 − decay)·track)`, except while
        unseeded, where the first track *is* the session.  A full listen also
        ends any consecutive-skip run: the escalation decays the moment you stop
        skipping, so no separate cooldown is needed.
        """
        track_embedding = manifold.normalise(track_embedding)

        self.recent_tracks.append(track_embedding.copy())
        self.tracks_played += 1
        self.clear_skip_run()

        if not self.is_seeded():
            self.session_vector = track_embedding.copy()
            return

        decay = config.session_decay_factor
        blended = decay * self.session_vector + (1 - decay) * track_embedding
        norm = float(np.linalg.norm(blended))
        if norm > 1e-8:
            self.session_vector = blended / norm

    # ── Skips ────────────────────────────────────────────────────────────────

    def clear_skip_run(self):
        """End the current consecutive-skip run."""
        self.skip_run = []
        self.skip_run_anchor = None

    def repel_from_skip_run(self,
                            track_embedding: np.ndarray,
                            run_length: int,
                            embedding_matrix: np.ndarray) -> dict:
        """
        Move the session vector away from what is being rejected, by a magnitude
        solved for this run length's turnover target (audit H9).

        Args:
            track_embedding:  the track just skipped.
            run_length:       how many consecutive skips this is, counting this
                              one.  Drives the turnover target and the snap gate.
            embedding_matrix: the library, `(N, D)`, centred and normalised.  The
                              turnover target is meaningless without it — "5% of
                              the pool" is a statement about *this* collection.

        Returns a dict of what actually happened, all measured rather than
        declared, for the console line and the `[I]` inspector.
        """
        track_embedding = manifold.normalise(track_embedding)
        self.skip_run.append(track_embedding.copy())

        if self.skip_run_anchor is None:
            self.skip_run_anchor = self.session_vector.copy()

        schedule = config.skip_turnover_schedule
        target = schedule[min(run_length, len(schedule)) - 1]

        # An unseeded session has no direction to move, and no track has been
        # heard for the skip to be evidence about.  Seed from the rejection's
        # opposite?  No — that is exactly the full-strength claim from a single
        # rejection that the taste model refuses to make.  Do nothing to the
        # vector; the counters still move.
        if not self.is_seeded() or embedding_matrix is None:
            return {'run_length': run_length, 'target': target, 'lambda': 0.0,
                    'turnover': 0.0, 'snapped': False, 'quality': None}

        away = manifold.normalise(np.mean(self.skip_run, axis=0))
        before = self.session_vector.copy()

        # The snap is what makes "however hard you push, the session vector stays
        # inside the music" structural rather than hopeful — but it is a move in
        # its own right, so it is gated on the run being long enough that λ is
        # large.  Both directions of that gate are measured; see config.
        #
        # It is passed *into* the solve rather than applied after it, so λ is
        # chosen against the turnover of the vector that will actually select.
        # Snapping afterwards let a second skip land back where the run started.
        snapped = run_length >= config.skip_snap_from_run_length

        lam, _turnover_vs_previous, moved = manifold.solve_repulsion(
            embedding_matrix, before, away, target,
            k=config.candidate_pool_size,
            snap_result=snapped,
        )

        self.session_vector = moved

        return {
            'run_length': run_length,
            'target': target,
            'lambda': lam,
            # Measured on the vector that will actually select the next track,
            # against where the vector stood when this skip run began.
            'turnover': manifold.pool_turnover(
                embedding_matrix, self.skip_run_anchor, moved,
                config.candidate_pool_size),
            'snapped': snapped,
            'quality': manifold.on_manifold_quality(embedding_matrix, moved),
        }

    def get_session_vector(self) -> np.ndarray:
        """Get current session vector (zero while nothing has played)."""
        return self.session_vector.copy()

    # NOTE: there is deliberately no get_vibe_description() here.
    #
    # The old one derived a mood word from -Σ|v|·log|v| over the session vector
    # and branched on thresholds (>5.0 "eclectic", >4.0 "diverse", else
    # "cohesive").  For a 512-d unit vector that quantity is always ~55, so the
    # function returned "eclectic" 100% of the time and two of its three
    # branches were unreachable.  The momentum words (focused/flowing/drifting/
    # exploring) and the stage words (warming up/building/deep in the zone) were
    # equally invented — the stage word was a tracks_played counter in a costume.
    #
    # The replacement is a CLAP text-encoder descriptor bank scored by z-score
    # against this library's own distribution (audit H1/D5), built in Stage 1 and
    # wired to the display in Stage 3.  Until then the UI shows the track count,
    # which is a fact.
    #
    # NOTE: force_shift() is gone too (audit D8/H9).  It blended the session
    # vector 50% toward a *random* 512-d direction, which is near-orthogonal to
    # every real embedding — landing it at 0.56 mean similarity to the nearest
    # real music against 0.73 for a real track.  So [V] did not mean "different
    # vibe", it meant "weaker vibe, plus noise", and it turned over less of the
    # candidate pool than two escalated skips do.  Its real job was clearing the
    # ten-track queue, and there is no longer a queue to clear.

    def get_stats(self) -> dict:
        """Get session statistics."""
        return {
            'tracks_played': self.tracks_played,
            'session_started': self.session_started,
            'is_seeded': self.is_seeded(),
            'recent_tracks_count': len(self.recent_tracks),
            'skip_run_length': len(self.skip_run),
            'session_vector_norm': float(np.linalg.norm(self.session_vector)),
        }
