"""
Track Selector - Weighted multi-factor scoring and selection
Combines session similarity, taste similarity, novelty, and anti-repetition.
"""

import sys
import numpy as np
from typing import List, Set, Optional, Tuple
from collections import deque
from config import config


class TrackSelector:
    """
    Selects next track using weighted multi-factor scoring.
    Operates on bounded candidate pool to ensure performance.
    """
    
    def __init__(self, track_library):
        self.track_library = track_library
        self.recent_history = deque(maxlen=config.recent_history_size)
        self.play_history = {}  # track -> last_play_index
        self.current_index = 0
    
    def select_track(self,
                    session_vector: np.ndarray,
                    taste_vector: np.ndarray,
                    weights: dict,
                    exclude_tracks: Set[str] = None,
                    time_context=None) -> Optional[str]:
        """
        Select next track using multi-factor scoring.
        
        Args:
            session_vector: Current session state vector
            taste_vector: User taste vector
            weights: Dict with session_weight, taste_weight, novelty_weight, anti_repetition_weight
            exclude_tracks: Tracks to exclude from selection
            time_context: Optional TimeContext for time-aware scoring
        
        Returns:
            Selected track file path, or None if no suitable track found
        """
        exclude_tracks = exclude_tracks or set()
        
        # Add recent history to exclusions
        exclude_tracks.update(self.recent_history)
        
        # Get candidate pool
        candidates = self.track_library.get_candidate_pool(
            session_vector=session_vector,
            taste_vector=taste_vector,
            exclude_tracks=exclude_tracks
        )
        
        if not candidates:
            print("Warning: No candidates found, selecting random track", file=sys.stderr)
            return self.track_library.get_random_track(exclude_tracks)
        
        # Score all candidates
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
                weights=weights,
                time_context=time_context
            )
            
            scored_candidates.append((track_file, score))
        
        if not scored_candidates:
            return None
        
        # Select track with highest score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        selected_track = scored_candidates[0][0]
        
        # Update history
        self._record_selection(selected_track)
        
        return selected_track
    
    def _calculate_score(self,
                        track_embedding: np.ndarray,
                        track_file: str,
                        session_vector: np.ndarray,
                        taste_vector: np.ndarray,
                        weights: dict,
                        time_context=None) -> float:
        """
        Calculate weighted score for a candidate track.
        
        Score = α * session_sim + β * taste_sim + γ * novelty + δ * anti_repetition + ε * time_context
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
        
        # Base weighted combination
        score = (weights['session_weight'] * session_sim +
                weights['taste_weight'] * taste_sim +
                weights['novelty_weight'] * novelty +
                weights['anti_repetition_weight'] * anti_rep)
        
        # 5. Time context (Phase 3) - add as bonus if enabled
        if time_context and config.enable_time_context:
            time_sim = time_context.get_similarity(track_embedding)
            # Add time context as bonus (not part of main weight sum)
            score = score + config.time_context_weight * time_sim
        
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
    
    def get_recent_history(self) -> List[str]:
        """Get list of recently played tracks."""
        return list(self.recent_history)
    
    def clear_history(self):
        """Clear play history (for new session)."""
        self.recent_history.clear()
        # Don't clear play_history to maintain long-term anti-repetition
        self.current_index = 0
    
    def get_stats(self) -> dict:
        """Get selector statistics."""
        return {
            'recent_history_size': len(self.recent_history),
            'total_tracks_played': self.current_index,
            'unique_tracks_played': len(self.play_history)
        }
