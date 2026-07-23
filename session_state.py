"""
Session State Model - Dynamic short-term vibe representation
Tracks evolving session direction separate from long-term taste.
"""

import numpy as np
from typing import Optional
from collections import deque
from config import config


class SessionState:
    """
    Maintains dynamic session vector representing current listening vibe.
    Uses exponential decay and recent track averaging.
    Separate from long-term user taste.
    """
    
    def __init__(self, dimension: int = None):
        self.dimension = dimension or config.embedding_dimension
        self.session_vector = self._initialize_session_vector()
        self.recent_tracks = deque(maxlen=config.session_influence_window)
        self.tracks_played = 0
        self.session_started = False
        
    def _initialize_session_vector(self) -> np.ndarray:
        """Initialize session vector as random normalized vector."""
        vector = np.random.randn(self.dimension) * 0.1
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    def start_session(self, initial_track_embedding: Optional[np.ndarray] = None):
        """
        Start a new session.
        Can optionally seed with an initial track.
        """
        if initial_track_embedding is not None:
            norm = np.linalg.norm(initial_track_embedding)
            if norm > 0:
                self.session_vector = initial_track_embedding / norm
        else:
            self.session_vector = self._initialize_session_vector()
        
        self.recent_tracks.clear()
        self.tracks_played = 0
        self.session_started = True

        print("Session started", file=__import__("sys").stderr)
    
    def update(self, track_embedding: np.ndarray):
        """
        Update session state with newly played track.
        Uses exponential decay for smooth evolution.
        """
        # Normalize input
        norm = np.linalg.norm(track_embedding)
        if norm > 0:
            track_embedding = track_embedding / norm
        
        # Add to recent history
        self.recent_tracks.append(track_embedding.copy())
        self.tracks_played += 1
        
        # Update session vector with decay
        decay = config.session_decay_factor
        self.session_vector = decay * self.session_vector + (1 - decay) * track_embedding
        
        # Re-normalize
        norm = np.linalg.norm(self.session_vector)
        if norm > 1e-8:
            self.session_vector = self.session_vector / norm
        else:
            self.session_vector = self._initialize_session_vector()

    def penalize_similar(self, track_embedding: np.ndarray):
        """
        Penalize tracks similar to this embedding (after skip).
        Nudges session away from skipped track direction.
        """
        norm = np.linalg.norm(track_embedding)
        if norm > 0:
            track_embedding = track_embedding / norm
        
        # Move away from skipped track
        penalty_weight = 0.15
        self.session_vector = self.session_vector - penalty_weight * track_embedding
        
        # Re-normalize
        norm = np.linalg.norm(self.session_vector)
        if norm > 1e-8:
            self.session_vector = self.session_vector / norm
    
    def force_shift(self):
        """
        Force a significant trajectory shift (skip entire vibe).
        Rotates session vector in a random direction.
        """
        shift_magnitude = config.vibe_shift_magnitude
        
        # Generate random orthogonal direction
        random_direction = np.random.randn(self.dimension)
        random_direction = random_direction / np.linalg.norm(random_direction)
        
        # Blend current direction with random direction
        self.session_vector = (1 - shift_magnitude) * self.session_vector + shift_magnitude * random_direction
        
        # Normalize
        norm = np.linalg.norm(self.session_vector)
        if norm > 1e-8:
            self.session_vector = self.session_vector / norm
        
        print("Vibe shifted!", file=__import__("sys").stderr)

    def get_session_vector(self) -> np.ndarray:
        """Get current session vector (normalized)."""
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

    def get_stats(self) -> dict:
        """Get session statistics."""
        return {
            'tracks_played': self.tracks_played,
            'session_started': self.session_started,
            'recent_tracks_count': len(self.recent_tracks),
            'session_vector_norm': float(np.linalg.norm(self.session_vector))
        }
