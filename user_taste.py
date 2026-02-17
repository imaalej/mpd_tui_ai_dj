"""
User Taste Model - Persistent long-term preference representation
Maintains and updates user taste vector based on feedback.
"""

import sys
import numpy as np
from typing import Optional
from pathlib import Path
from config import config


class UserTaste:
    """
    Maintains persistent user taste as a normalized embedding vector.
    Updated incrementally based on likes, listens, and skip penalties.
    """
    
    def __init__(self, dimension: int = None):
        self.dimension = dimension or config.embedding_dimension
        self.taste_vector = self._initialize_taste_vector()
        self.total_updates = 0
        self.like_count = 0
        self.skip_count = 0
        self.full_listen_count = 0
        
    def _initialize_taste_vector(self) -> np.ndarray:
        """Initialize taste vector as small random normalized vector."""
        # Start with small random values so initial sessions are exploratory
        vector = np.random.randn(self.dimension) * 0.01
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    def get_taste_vector(self) -> np.ndarray:
        """Get current taste vector (normalized)."""
        return self.taste_vector.copy()
    
    def update_from_like(self, track_embedding: np.ndarray):
        """
        Update taste based on explicit like.
        Strongest positive signal.
        """
        weight = config.taste_update_like
        self._update(track_embedding, weight)
        self.like_count += 1
        
    def update_from_full_listen(self, track_embedding: np.ndarray):
        """
        Update taste based on full listen (passive approval).
        Weaker than explicit like but still positive.
        """
        weight = config.taste_update_full_listen
        self._update(track_embedding, weight)
        self.full_listen_count += 1
    
    def update_from_skip(self, track_embedding: np.ndarray):
        """
        Update taste based on skip (negative signal).
        Pushes taste away from skipped track.
        """
        weight = config.taste_update_skip_penalty
        self._update(track_embedding, weight)
        self.skip_count += 1
    
    def _update(self, track_embedding: np.ndarray, weight: float):
        """
        Core update logic using exponential moving average.
        
        Args:
            track_embedding: Normalized embedding of track
            weight: Update weight (positive for attraction, negative for repulsion)
        """
        # Normalize input
        norm = np.linalg.norm(track_embedding)
        if norm > 0:
            track_embedding = track_embedding / norm
        
        # Update with weighted combination
        if weight > 0:
            # Positive update: move toward track
            self.taste_vector = (1 - weight) * self.taste_vector + weight * track_embedding
        else:
            # Negative update: move away from track
            # Subtract weighted track vector
            self.taste_vector = self.taste_vector - abs(weight) * track_embedding
        
        # Re-normalize to maintain unit length
        norm = np.linalg.norm(self.taste_vector)
        if norm > 1e-8:
            self.taste_vector = self.taste_vector / norm
        else:
            # If vector collapsed, reinitialize
            self.taste_vector = self._initialize_taste_vector()
        
        self.total_updates += 1
    
    def get_similarity(self, track_embedding: np.ndarray) -> float:
        """
        Calculate cosine similarity between taste and track.
        Returns value in [-1, 1].
        """
        norm = np.linalg.norm(track_embedding)
        if norm > 0:
            track_embedding = track_embedding / norm
        
        return float(np.dot(self.taste_vector, track_embedding))
    
    def save(self, filepath: Optional[Path] = None):
        """Save taste model to disk."""
        if filepath is None:
            filepath = config.taste_file
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(
            filepath,
            taste_vector=self.taste_vector,
            total_updates=self.total_updates,
            like_count=self.like_count,
            skip_count=self.skip_count,
            full_listen_count=self.full_listen_count
        )
    
    def load(self, filepath: Optional[Path] = None) -> bool:
        """
        Load taste model from disk.
        Returns True if successful, False otherwise.
        """
        if filepath is None:
            filepath = config.taste_file
        
        if not filepath.exists():
            print(f"No saved taste model found at {filepath}", file=sys.stderr)
            return False
        
        try:
            data = np.load(filepath)
            self.taste_vector = data['taste_vector']
            self.total_updates = int(data['total_updates'])
            self.like_count = int(data['like_count'])
            self.skip_count = int(data['skip_count'])
            self.full_listen_count = int(data['full_listen_count'])
            
            print(f"Loaded taste model: {self.total_updates} updates, "
                  f"{self.like_count} likes, {self.full_listen_count} full listens",
                  file=sys.stderr)
            
            # Verify normalization
            norm = np.linalg.norm(self.taste_vector)
            if not np.isclose(norm, 1.0, atol=1e-5):
                print(f"Warning: Taste vector not normalized (norm={norm}), fixing...", file=sys.stderr)
                if norm > 0:
                    self.taste_vector = self.taste_vector / norm
            
            return True
            
        except Exception as e:
            print(f"Error loading taste model: {e}", file=sys.stderr)
            return False
    
    def reset(self):
        """Reset taste to initial state."""
        self.taste_vector = self._initialize_taste_vector()
        self.total_updates = 0
        self.like_count = 0
        self.skip_count = 0
        self.full_listen_count = 0
        print("Taste model reset", file=sys.stderr)
    
    def get_stats(self) -> dict:
        """Get statistics about taste model."""
        return {
            'total_updates': self.total_updates,
            'like_count': self.like_count,
            'skip_count': self.skip_count,
            'full_listen_count': self.full_listen_count,
            'taste_vector_norm': float(np.linalg.norm(self.taste_vector))
        }
