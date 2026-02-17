"""
Session State Model - Dynamic short-term vibe representation
Tracks evolving session direction separate from long-term taste.
"""

import numpy as np
from typing import List, Optional
from collections import deque
from config import config
from time_context import TimeContext


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
        
        # Vibe state tracking
        self.current_vibe_keywords = []
        self.vibe_trajectory = []
        
        # Time context awareness (Phase 3)
        self.time_context = TimeContext(dimension=self.dimension)
        
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
        self.vibe_trajectory = []
        
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
        
        # Track vibe trajectory
        self._update_vibe_trajectory()
    
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
        self._update_vibe_trajectory()
    
    def get_session_vector(self) -> np.ndarray:
        """Get current session vector (normalized)."""
        return self.session_vector.copy()
    
    def get_recent_average(self) -> np.ndarray:
        """
        Get average of recent tracks as alternative session representation.
        """
        if not self.recent_tracks:
            return self.session_vector.copy()
        
        avg = np.mean(list(self.recent_tracks), axis=0)
        norm = np.linalg.norm(avg)
        if norm > 1e-8:
            return avg / norm
        return self.session_vector.copy()
    
    def get_similarity(self, track_embedding: np.ndarray) -> float:
        """
        Calculate cosine similarity between session state and track.
        """
        norm = np.linalg.norm(track_embedding)
        if norm > 0:
            track_embedding = track_embedding / norm
        
        return float(np.dot(self.session_vector, track_embedding))
    
    def _update_vibe_trajectory(self):
        """Track vibe evolution for description generation."""
        if len(self.recent_tracks) >= 2:
            # Calculate momentum (similarity between consecutive recent tracks)
            recent = list(self.recent_tracks)
            momentum = np.dot(recent[-1], recent[-2])
            self.vibe_trajectory.append(float(momentum))
            
            # Keep only recent trajectory
            if len(self.vibe_trajectory) > 10:
                self.vibe_trajectory = self.vibe_trajectory[-10:]
    
    def get_vibe_description(self) -> str:
        """
        Generate human-readable description of current vibe.
        Maps embedding-space characteristics to mood descriptors.
        """
        if not self.session_started:
            return "Initializing..."
        
        if self.tracks_played == 0:
            return "Building your vibe..."
        
        if self.tracks_played == 1:
            return "Establishing mood..."
        
        # Analyze trajectory
        if len(self.vibe_trajectory) >= 3:
            recent_momentum = np.mean(self.vibe_trajectory[-3:])
            
            if recent_momentum > 0.85:
                consistency = "focused"
            elif recent_momentum > 0.7:
                consistency = "flowing"
            elif recent_momentum > 0.5:
                consistency = "drifting"
            else:
                consistency = "exploring"
        else:
            consistency = "developing"
        
        # Analyze session vector characteristics
        # Use simple heuristics based on vector properties
        vector_magnitude = np.linalg.norm(self.session_vector)
        vector_entropy = -np.sum(np.abs(self.session_vector) * np.log(np.abs(self.session_vector) + 1e-10))
        
        if vector_entropy > 5.0:
            mood = "eclectic"
        elif vector_entropy > 4.0:
            mood = "diverse"
        else:
            mood = "cohesive"
        
        # Count played tracks
        if self.tracks_played < 3:
            stage = "warming up"
        elif self.tracks_played < 8:
            stage = "building"
        else:
            stage = "deep in the zone"
        
        return f"{consistency} {mood} vibe, {stage}"
    
    def get_stats(self) -> dict:
        """Get session statistics."""
        return {
            'tracks_played': self.tracks_played,
            'session_started': self.session_started,
            'recent_tracks_count': len(self.recent_tracks),
            'vibe_description': self.get_vibe_description(),
            'session_vector_norm': float(np.linalg.norm(self.session_vector))
        }
    
    def reset(self):
        """Reset session state."""
        self.session_vector = self._initialize_session_vector()
        self.recent_tracks.clear()
        self.tracks_played = 0
        self.session_started = False
        self.vibe_trajectory = []
        print("Session reset", file=__import__("sys").stderr)
