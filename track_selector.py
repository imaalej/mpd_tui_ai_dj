"""
Track Selector - Weighted multi-factor scoring and selection
Combines session similarity, taste similarity, novelty, and anti-repetition.

Selection draws one track by **Boltzmann sampling over rank** rather than taking
the argmax (audit H6):

    p(i) ∝ exp(−i / τ)          i = 0, 1, 2, … over the scored candidate list

Sampling over rank rather than over score is the whole point.  A score-softmax
needs a temperature calibrated to the *width* of the score distribution, and that
width moves whenever the exploration controller reshuffles α/β/γ/δ, whenever β
ramps in, whenever the embedding space is re-centred, and on any other library —
so it would need re-tuning every time, which is the failure pattern behind half
the findings in the audit.  Rank does not care what the scores look like, only
about their ordering, so τ never needs recalibration.

Why this had to land with the queue-depth change and not after it: at depth 10
the within-batch exclusion set forced ten *distinct* tracks, so variety came out
of the bookkeeping.  At depth 1 that mask is gone, and each pick is an
independent argmax over a pool that barely moved — the session vector sits at
cosine ~0.97 to the track it is about to play.  Argmax there is effectively
deterministic: it reproduces the same session from the same state, every time.
"""

import json
import sys
import numpy as np
from pathlib import Path
from typing import List, Optional, Set
from collections import deque
from config import config


class TrackSelector:
    """
    Selects next track using weighted multi-factor scoring.
    Operates on bounded candidate pool to ensure performance.
    """

    def __init__(self, track_library, rng: Optional[np.random.Generator] = None):
        self.track_library = track_library
        self.recent_history = deque(maxlen=config.recent_history_size)
        self.play_history = {}  # track -> last_play_index
        self.current_index = 0
        # Injectable so the sampling can be asserted on rather than merely
        # observed to "look random".
        self.rng = rng if rng is not None else np.random.default_rng()
        # Set by the last select_track() call, for the [I] inspector.
        self.last_tau: Optional[float] = None
        self.last_rank: Optional[int] = None
        self.last_pool_size: Optional[int] = None

    def temperature(self, exploration: float) -> float:
        """
        Map the exploration scalar to τ, the effective number of candidates in
        play.  Linear over the configured exploration bounds, floored at τ_min.

            exploration 0.1 (floor)  → τ 1.0   → p(rank 0) 63%
            exploration 0.4 (mid)    → τ 7.5   → p(rank 0) 12%
            exploration 0.7 (ceiling)→ τ 15.0  → p(rank 0)  6%

        Reportable as a true statement about what the machine is doing —
        "choosing from ~top 8" — rather than as an invented mood word.
        """
        low, high = config.exploration_min, config.exploration_max
        if high <= low:
            return config.tau_max
        fraction = (float(exploration) - low) / (high - low)
        fraction = min(1.0, max(0.0, fraction))
        return max(config.tau_min, config.tau_max * fraction)

    def _sample_rank(self, n_candidates: int, tau: float) -> int:
        """
        Draw an index from p(i) ∝ exp(−i/τ) over `n_candidates` ranks.

        Below `minimum_sampled_pool` candidates the distribution is replaced by a
        uniform draw: with two or three entries left, τ would decide the outcome
        and the "choice" would be a formality.
        """
        if n_candidates <= 1:
            return 0
        if n_candidates < config.minimum_sampled_pool:
            return int(self.rng.integers(n_candidates))

        weights = np.exp(-np.arange(n_candidates) / tau)
        weights /= weights.sum()
        return int(self.rng.choice(n_candidates, p=weights))

    def select_track(self,
                     session_vector: np.ndarray,
                     taste_vector: np.ndarray,
                     weights: dict,
                     exploration: float,
                     exclude_tracks: Set[str] = None) -> Optional[str]:
        """
        Select the next track by scoring a bounded candidate pool and sampling
        over rank.

        Args:
            session_vector: Current session state vector (zero while unseeded)
            taste_vector: User taste vector (zero until a positive signal)
            weights: session_weight, taste_weight, novelty_weight, anti_repetition_weight
            exploration: the exploration scalar, which sets τ
            exclude_tracks: Tracks to exclude from selection

        Returns:
            Selected track file path, or None if no suitable track found
        """
        # A copy, not the caller's set (audit L9).  This used to mutate the
        # exclusion set it was handed, which was harmless only because there
        # happened to be one caller.
        exclude_tracks = set(exclude_tracks) if exclude_tracks else set()
        exclude_tracks.update(self.recent_history)

        candidates = self.track_library.get_candidate_pool(
            session_vector=session_vector,
            taste_vector=taste_vector,
            exclude_tracks=exclude_tracks
        )

        if not candidates:
            # Reached when neither vector carries any evidence — the first track
            # of a fresh session.  A uniform draw is the honest answer to "no
            # information"; the alternative is querying with a zero vector,
            # whose dot products are all 0, and presenting whatever order
            # argpartition happened to produce as a preference.
            selected = self.track_library.get_random_track(exclude_tracks, rng=self.rng)
            if selected is not None:
                self.last_tau = None
                self.last_rank = None
                self.last_pool_size = 0
                self._record_selection(selected)
            return selected

        scored_candidates = []
        for track_file in candidates:
            track_embedding = self.track_library.get_embedding(track_file)
            if track_embedding is None:
                continue

            score = self._calculate_score(
                track_embedding=track_embedding,
                track_file=track_file,
                session_vector=session_vector,
                taste_vector=taste_vector,
                weights=weights
            )
            scored_candidates.append((track_file, score))

        if not scored_candidates:
            return None

        # Sort descending, then draw by rank.  Ties are broken by the track key
        # so the ordering handed to the sampler is deterministic given the
        # scores — otherwise "identical run from identical state" would be
        # untestable for reasons that have nothing to do with the sampling.
        scored_candidates.sort(key=lambda pair: (-pair[1], pair[0]))

        tau = self.temperature(exploration)
        rank = self._sample_rank(len(scored_candidates), tau)
        selected_track = scored_candidates[rank][0]

        self.last_tau = tau
        self.last_rank = rank
        self.last_pool_size = len(scored_candidates)

        self._record_selection(selected_track)
        return selected_track

    def _calculate_score(self,
                         track_embedding: np.ndarray,
                         track_file: str,
                         session_vector: np.ndarray,
                         taste_vector: np.ndarray,
                         weights: dict) -> float:
        """
        Calculate weighted score for a candidate track.

        Score = α * session_sim + β * taste_sim + γ * novelty + δ * anti_repetition

        Those four weights sum to 1.0 and nothing is added on top of them.  The
        `0.15 * time_sim` bonus that used to break that invariant went with the
        time-context subsystem (audit D6 / L9).
        """
        # Normalize track embedding
        norm = np.linalg.norm(track_embedding)
        if norm > 0:
            track_embedding = track_embedding / norm

        # 1. Session similarity (cosine similarity)
        session_sim = float(np.dot(session_vector, track_embedding))
        session_sim = (session_sim + 1) / 2  # Normalize to [0, 1]

        # 2. Taste similarity
        taste_sim = float(np.dot(taste_vector, track_embedding))
        taste_sim = (taste_sim + 1) / 2  # Normalize to [0, 1]

        # 3. Novelty (distance from recent tracks)
        novelty = self._calculate_novelty(track_embedding)

        # 4. Anti-repetition (time since last played)
        anti_rep = self._calculate_anti_repetition(track_file)

        # Weighted combination
        score = (weights['session_weight'] * session_sim +
                 weights['taste_weight'] * taste_sim +
                 weights['novelty_weight'] * novelty +
                 weights['anti_repetition_weight'] * anti_rep)

        return score

    def _calculate_novelty(self, track_embedding: np.ndarray) -> float:
        """
        Calculate novelty score based on minimum distance to recent tracks.
        Returns value in [0, 1] where higher is more novel.
        """
        if not self.recent_history:
            return 1.0  # Maximum novelty if no history

        # Get embeddings of recent tracks
        recent_embeddings = []
        for track_file in list(self.recent_history)[-10:]:  # Check last 10
            emb = self.track_library.get_embedding(track_file)
            if emb is not None:
                recent_embeddings.append(emb)

        if not recent_embeddings:
            return 1.0

        # Calculate minimum cosine similarity to recent tracks
        similarities = [float(np.dot(track_embedding, recent_emb))
                        for recent_emb in recent_embeddings]

        max_similarity = max(similarities)

        # Convert similarity to novelty
        # High similarity = low novelty
        # Cosine similarity range [-1, 1] maps to novelty [1, 0]
        novelty = (1 - max_similarity) / 2  # Map [-1, 1] -> [0, 1]

        return novelty

    def _calculate_anti_repetition(self, track_file: str) -> float:
        """
        Calculate anti-repetition score based on time since last played.
        Returns value in [0, 1] where higher means played longer ago.
        """
        if track_file not in self.play_history:
            return 1.0  # Never played = maximum score

        last_play_index = self.play_history[track_file]
        tracks_since = self.current_index - last_play_index

        if tracks_since < config.minimum_replay_gap:
            # Strong penalty for recent replays
            return 0.1

        # Logarithmic decay
        # After minimum gap, score increases slowly
        score = min(1.0, 0.5 + 0.5 * np.log(tracks_since - config.minimum_replay_gap + 1) / 5)

        return score

    def _record_selection(self, track_file: str):
        """Record that a track was selected."""
        self.recent_history.append(track_file)
        self.play_history[track_file] = self.current_index
        self.current_index += 1

    def forget_selection(self, track_file: str) -> bool:
        """
        Undo the bookkeeping for a track that was selected but never played.

        At queue depth 1 a selection is normally a play, but two paths break
        that: a skip drops the lookahead and re-picks it, and MPD can refuse an
        `add`.  Left recorded, those tracks would sit inside the 20-track replay
        gap having never been heard, and `total_selections` would claim more
        music than the session actually played — a number the user reads in the
        `[I]` inspector that the system cannot back up.

        Only undoes the *most recent* selection, which is the only one a caller
        can legitimately retract; returns False otherwise rather than corrupting
        an index it cannot reconstruct.
        """
        if not self.recent_history or self.recent_history[-1] != track_file:
            return False
        if self.play_history.get(track_file) != self.current_index - 1:
            return False

        self.recent_history.pop()
        del self.play_history[track_file]
        self.current_index -= 1
        return True

    def get_recent_history(self) -> List[str]:
        """Get list of recently played tracks."""
        return list(self.recent_history)

    # ── Persistence (audit M6b) ──────────────────────────────────────────────

    def save(self, filepath: Optional[Path] = None):
        """
        Persist the anti-repetition state.

        Without this, `recent_history` and `play_history` were rebuilt empty on
        every launch, so the README's "recently played tracks are excluded for at
        least 20 songs" quietly meant "until you restart".  `recent_history` is
        saved alongside the two the audit names because it is the half that does
        the actual excluding — `play_history` only shapes the score.
        """
        if filepath is None:
            filepath = config.play_history_file

        filepath.parent.mkdir(parents=True, exist_ok=True)
        state = {
            'current_index': int(self.current_index),
            'play_history': {k: int(v) for k, v in self.play_history.items()},
            'recent_history': list(self.recent_history),
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load(self, filepath: Optional[Path] = None) -> bool:
        """Restore anti-repetition state.  Returns True if anything was read."""
        if filepath is None:
            filepath = config.play_history_file

        if not Path(filepath).exists():
            print(f"No saved play history found at {filepath}", file=sys.stderr)
            return False

        try:
            with open(filepath, 'r') as f:
                state = json.load(f)

            self.current_index = int(state.get('current_index', 0))
            self.play_history = {str(k): int(v)
                                 for k, v in state.get('play_history', {}).items()}
            self.recent_history = deque(state.get('recent_history', []),
                                        maxlen=config.recent_history_size)

            print(f"Loaded play history: {len(self.play_history)} tracks seen, "
                  f"{len(self.recent_history)} still inside the replay gap",
                  file=sys.stderr)
            return True

        except Exception as e:
            print(f"Error loading play history: {e}", file=sys.stderr)
            return False

    def get_stats(self) -> dict:
        """Get selector statistics."""
        return {
            'recent_history_size': len(self.recent_history),
            'total_tracks_played': self.current_index,
            'unique_tracks_played': len(self.play_history),
            'last_tau': self.last_tau,
            'last_rank': self.last_rank,
            'last_pool_size': self.last_pool_size,
        }
