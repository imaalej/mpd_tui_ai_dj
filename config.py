"""
Configuration Management for Adaptive Session AI DJ
Provides all configurable parameters for the system.
"""

import os
from pathlib import Path

from music_directory import LEGACY_DEFAULT, detect_music_directory


class Config:
    """Central configuration for the adaptive DJ system."""

    def __init__(self):
        # MPD Connection Settings
        self.mpd_host = os.getenv('MPD_HOST', 'localhost')
        self.mpd_port = int(os.getenv('MPD_PORT', '6600'))

        # Where MPD's music lives.  Read from MPD's own config rather than
        # assumed (audit M3) — the old hardcoded /var/lib/mpd/music default was
        # wrong on this machine and only worked because of a symlink.  The
        # legacy path survives as a last resort so there is always something to
        # show the user; `mpd_music_directory_source` records which it was, and
        # startup refuses to launch if real track paths do not resolve under it.
        detected, source = detect_music_directory()
        if detected is None:
            detected, source = Path(LEGACY_DEFAULT), 'legacy default (unverified)'
        self.mpd_music_directory = str(detected)
        self.mpd_music_directory_source = source

        # Track Selection Scoring Weights (α, β, γ, δ)
        # These control the multi-factor scoring function
        self.weight_session_similarity = 0.4  # α - How much session state influences selection
        self.weight_taste_similarity = 0.3     # β - How much long-term taste influences selection
        self.weight_novelty = 0.2              # γ - How much we prefer novel tracks
        self.weight_anti_repetition = 0.1      # δ - How much we penalize recent plays
        
        # Exploration vs Exploitation Parameters
        self.exploration_initial = 0.3         # Starting exploration tendency (0-1)
        self.exploration_min = 0.1             # Minimum exploration (always some novelty)
        self.exploration_max = 0.7             # Maximum exploration (never completely random)
        self.exploration_increase_per_skip = 0.05  # How much to increase after skip
        self.exploration_decrease_per_listen = 0.02  # How much to decrease after full listen
        
        # Session State Parameters
        self.session_decay_factor = 0.85       # Exponential decay for session vector updates
        self.session_influence_window = 5      # Number of recent tracks influencing session
        self.vibe_shift_magnitude = 0.5        # How strongly to shift on vibe skip (0-1)
        
        # User Taste Update Parameters
        self.taste_update_like = 0.1           # Weight for explicit likes
        self.taste_update_full_listen = 0.02   # Weight for passive full listens
        self.taste_update_skip_penalty = -0.05  # Penalty for skips
        
        # Queue Management
        self.queue_buffer_size = 10            # Number of tracks to maintain in queue
        self.queue_low_threshold = 3           # Generate more tracks when below this
        
        # Candidate Pool Parameters
        self.candidate_pool_size = 100         # Number of candidates to retrieve for scoring
        self.similarity_search_k = 200         # Initial similarity search results
        
        # Repetition Avoidance
        self.recent_history_size = 50          # Tracks to remember for anti-repetition
        self.minimum_replay_gap = 20           # Minimum tracks before replaying
        
        # Persistence Paths
        self.data_dir = Path(__file__).parent / 'data'
        self.embeddings_file = self.data_dir / 'embeddings' / 'track_embeddings.npz'
        self.descriptors_file = self.data_dir / 'embeddings' / 'descriptors.npz'
        self.failed_tracks_file = self.data_dir / 'embeddings' / 'failed.txt'
        self.taste_file = self.data_dir / 'state' / 'user_taste.npz'
        self.exploration_file = self.data_dir / 'state' / 'exploration_state.json'
        self.feedback_history_file = self.data_dir / 'state' / 'feedback_history.json'
        self.log_file = self.data_dir / 'dj.log'

        # System Parameters
        self.mpd_poll_interval = 0.5           # Seconds between MPD status polls

        # Expected embedding vector size (CLAP).  This is the size the session
        # and taste vectors are built at; the *embeddings file* is the authority
        # (M5), and TrackLibrary overwrites this on load if they disagree.
        self.embedding_dimension = 512

        # The embeddings artifact.  Schema 2 adds the library centroid (C5) and
        # the per-window matrix (C3); a schema-1 file has no centroid and would
        # be scored on an uncentred space, so loading it is refused rather than
        # silently downgraded.
        self.embedding_schema_version = 2
        self.clap_model_name = 'laion/clap-htsat-unfused'

        # Embedding keys come from `mpc listall` (M4), but a library can drift
        # between generation and use — files deleted, MPD re-scanned.  Below
        # this fraction of the embeddings still being present in MPD, the file
        # is stale enough that starting would be misleading rather than useful.
        self.minimum_mpd_coverage = 0.5

        # Ensure data directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / 'embeddings').mkdir(exist_ok=True)
        (self.data_dir / 'state').mkdir(exist_ok=True)
    
    def validate(self):
        """
        Validate configuration parameters.

        Raises ValueError on the first violation.  Deliberately NOT written with
        bare `assert`: under `python -O` every assert is stripped, which would
        silently disable the weight-sum invariant below (L9).
        """
        def _require(condition, message):
            if not condition:
                raise ValueError(f"Invalid configuration: {message}")

        weights = {
            'weight_session_similarity': self.weight_session_similarity,
            'weight_taste_similarity': self.weight_taste_similarity,
            'weight_novelty': self.weight_novelty,
            'weight_anti_repetition': self.weight_anti_repetition,
        }
        for name, value in weights.items():
            _require(0 <= value <= 1, f"{name} must be in [0, 1], got {value}")

        # The four scoring weights are the whole of the score.  Nothing may be
        # added on top of them (the time-context bonus that used to break this
        # invariant is gone — D6).
        total_weight = sum(weights.values())
        _require(abs(total_weight - 1.0) < 0.01,
                 f"scoring weights must sum to 1.0, got {total_weight}")

        _require(0 <= self.exploration_min <= self.exploration_max <= 1,
                 "require 0 <= exploration_min <= exploration_max <= 1, got "
                 f"{self.exploration_min} / {self.exploration_max}")
        _require(self.queue_buffer_size >= self.queue_low_threshold,
                 f"queue_buffer_size ({self.queue_buffer_size}) must be >= "
                 f"queue_low_threshold ({self.queue_low_threshold})")
        _require(0 <= self.minimum_mpd_coverage <= 1,
                 f"minimum_mpd_coverage must be in [0, 1], got {self.minimum_mpd_coverage}")

        return True


# Global config instance
config = Config()
