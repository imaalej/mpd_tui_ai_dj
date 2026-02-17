"""
Feedback Handler - Processes user feedback and coordinates updates
Routes feedback to appropriate components.
"""

import sys
from typing import Optional
import json
from pathlib import Path
from datetime import datetime
from config import config


class FeedbackHandler:
    """
    Processes user feedback and coordinates updates across components.
    Maintains feedback history for persistence.
    """
    
    def __init__(self, 
                 session_state, 
                 user_taste, 
                 exploration_controller,
                 queue_manager,
                 track_library):
        self.session_state = session_state
        self.user_taste = user_taste
        self.exploration_controller = exploration_controller
        self.queue_manager = queue_manager
        self.track_library = track_library
        
        self.feedback_history = []
        self.session_feedback_count = {
            'skips': 0,
            'vibe_skips': 0,
            'likes': 0,
            'full_listens': 0
        }
    
    def process_skip(self, track_file: str):
        """
        Process skip feedback: user skipped current song.
        - Penalizes similar tracks in session
        - Increases exploration
        - Keeps session direction but nudges away
        """
        # Silently process skip (no print to avoid TUI interference)
        
        track_embedding = self.track_library.get_embedding(track_file)
        if track_embedding is None:
            print(f"Warning: No embedding found for skipped track", file=sys.stderr)
            return
        
        # Update session state: penalize similar tracks
        self.session_state.penalize_similar(track_embedding)
        
        # Update user taste: small negative signal
        self.user_taste.update_from_skip(track_embedding)
        
        # Increase exploration
        self.exploration_controller.increase_exploration()
        
        # Record feedback
        self._record_feedback('skip', track_file)
        self.session_feedback_count['skips'] += 1
        
        # NOTE: recalculate() is intentionally NOT called here.
        # [N] = skip one track, keeps the current queue direction.
        # Only [V] (vibe skip) should recalculate the entire queue.
    
    def process_vibe_skip(self, track_file: str):
        """
        Process vibe skip: user wants completely different direction.
        - Forces significant trajectory shift
        - Sets high exploration mode
        - Recalculates entire queue
        """
        # Silently process vibe skip (no print to avoid TUI interference)
        
        # Force session shift
        self.session_state.force_shift()
        
        # Set high exploration
        self.exploration_controller.set_high_exploration()
        
        # Recalculate entire queue
        self.queue_manager.recalculate()
        
        # Record feedback
        self._record_feedback('vibe_skip', track_file)
        self.session_feedback_count['vibe_skips'] += 1
    
    def process_like(self, track_file: str):
        """
        Process like: user explicitly liked current track.
        - Strong positive signal for user taste
        - Confirms session direction
        - No exploration change (likes don't affect exploration directly)
        - Updates time context (Phase 3)
        """
        # Silently process like (no print to avoid TUI interference)
        
        track_embedding = self.track_library.get_embedding(track_file)
        if track_embedding is None:
            print(f"Warning: No embedding found for liked track", file=sys.stderr)
            return
        
        # Strong update to user taste
        self.user_taste.update_from_like(track_embedding)
        
        # Update time context (Phase 3)
        if hasattr(self.session_state, 'time_context') and config.enable_time_context:
            self.session_state.time_context.update(
                track_embedding,
                config.time_update_rate_like
            )
        
        # Save taste after explicit like
        self.user_taste.save()
        
        # Record feedback
        self._record_feedback('like', track_file)
        self.session_feedback_count['likes'] += 1
    
    def process_full_listen(self, track_file: str):
        """
        Process full listen: user listened to entire track.
        - Updates session state (confirms direction)
        - Weak positive signal for user taste
        - Decreases exploration (exploitation)
        - Updates time context (Phase 3)
        """
        # Don't print for every full listen (too noisy)
        
        track_embedding = self.track_library.get_embedding(track_file)
        if track_embedding is None:
            print(f"Warning: No embedding found for completed track", file=sys.stderr)
            return
        
        # Update session state: this is the primary driver of session evolution
        self.session_state.update(track_embedding)
        
        # Weak update to user taste
        self.user_taste.update_from_full_listen(track_embedding)
        
        # Update time context (Phase 3)
        if hasattr(self.session_state, 'time_context') and config.enable_time_context:
            self.session_state.time_context.update(
                track_embedding,
                config.time_update_rate_listen
            )
        
        # Decrease exploration (things are working)
        self.exploration_controller.decrease_exploration()
        
        # Record feedback
        self._record_feedback('full_listen', track_file)
        self.session_feedback_count['full_listens'] += 1
    
    def _record_feedback(self, feedback_type: str, track_file: str):
        """Record feedback event to history."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': feedback_type,
            'track': track_file
        }
        self.feedback_history.append(event)
        
        # Keep history bounded
        if len(self.feedback_history) > 1000:
            self.feedback_history = self.feedback_history[-1000:]
    
    def save_feedback_history(self, filepath: Optional[Path] = None):
        """Save feedback history to disk."""
        if filepath is None:
            filepath = config.feedback_history_file
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.feedback_history, f, indent=2)
    
    def load_feedback_history(self, filepath: Optional[Path] = None) -> bool:
        """Load feedback history from disk."""
        if filepath is None:
            filepath = config.feedback_history_file
        
        if not filepath.exists():
            print(f"No feedback history found at {filepath}", file=sys.stderr)
            return False
        
        try:
            with open(filepath, 'r') as f:
                self.feedback_history = json.load(f)
            
            print(f"Loaded {len(self.feedback_history)} feedback events", file=sys.stderr)
            return True
            
        except Exception as e:
            print(f"Error loading feedback history: {e}", file=sys.stderr)
            return False
    
    def get_session_stats(self) -> dict:
        """Get statistics for current session."""
        return self.session_feedback_count.copy()
    
    def reset_session_stats(self):
        """Reset session statistics."""
        self.session_feedback_count = {
            'skips': 0,
            'vibe_skips': 0,
            'likes': 0,
            'full_listens': 0
        }
