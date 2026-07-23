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
    
    # No `queue_manager` here, deliberately.  This class updates the models and
    # records history; it does not touch MPD or the queue.  The skip path's
    # ordering — feedback, then replace the lookahead, then advance exactly once
    # — is the thing C4 is about, so it lives in one place on the orchestrator
    # rather than being split across a component that cannot see the advance.
    def __init__(self,
                 session_state,
                 user_taste,
                 exploration_controller,
                 track_library):
        self.session_state = session_state
        self.user_taste = user_taste
        self.exploration_controller = exploration_controller
        self.track_library = track_library
        
        self.feedback_history = []
        # No 'vibe_skips' counter: [V] is deleted (audit D8/H9) and a key that
        # can never fire should not have a tally the [I] overlay reports as 0.
        self.session_feedback_count = {
            'skips': 0,
            'likes': 0,
            'full_listens': 0
        }
    
    def process_skip(self, track_file: str) -> Optional[dict]:
        """
        Process a skip, and re-pick the lookahead under the result.

        There is exactly **one** skip path (audit C4/H9).  `[V]` and
        `process_vibe_skip()` are gone: `[V]` blended half a random direction
        into the session vector, which is not a direction at all — it landed the
        vector at 0.56 mean similarity to the nearest real music against 0.73 for
        a real track — while turning over less of the candidate pool than two
        escalated presses of this key do.  Its actual job was clearing the
        ten-track queue, and there is no longer a queue to clear.

        The escalation replaces it, driven by evidence rather than by a second
        keypress: *n* consecutive rejections is the system observing that the
        neighbourhood is wrong, which is strictly better information than the
        user asserting the same thing, because it arrives without them having to
        diagnose their own dissatisfaction first.

        Returns the measured outcome of the vector move, or None if the track had
        no embedding.  **The caller advances MPD; this method must not** — see
        `QueueManager.replace_next()` for why the ordering matters.
        """
        track_embedding = self.track_library.get_embedding(track_file)
        if track_embedding is None:
            print("Warning: No embedding found for skipped track", file=sys.stderr)
            return None

        # Exploration first: it owns `consecutive_skips`, and the run length it
        # produces is what the repulsion's turnover target is chosen from.
        self.exploration_controller.increase_exploration()
        run_length = self.exploration_controller.consecutive_skips

        outcome = self.session_state.repel_from_skip_run(
            track_embedding,
            run_length=run_length,
            embedding_matrix=self.track_library.embedding_matrix,
        )

        # Small negative signal to long-term taste.  Unlike the session move,
        # this is deliberately not escalated: a run of skips says a great deal
        # about tonight and very little about what the listener likes in general.
        self.user_taste.update_from_skip(track_embedding)

        self._record_feedback('skip', track_file)
        self.session_feedback_count['skips'] += 1

        if outcome and outcome['turnover'] > 0:
            print(f"Skip #{run_length}: λ={outcome['lambda']:.2f}"
                  f"{' + snap' if outcome['snapped'] else ''}, "
                  f"{outcome['turnover']:.0%} of what you would have heard is now "
                  f"different (target {outcome['target']:.0%})", file=sys.stderr)

        return outcome

    def process_like(self, track_file: str):
        """
        Process like: user explicitly liked current track.
        - Strong positive signal for user taste
        - Confirms session direction
        - No exploration change (likes don't affect exploration directly)
        """
        # Silently process like (no print to avoid TUI interference)
        
        track_embedding = self.track_library.get_embedding(track_file)
        if track_embedding is None:
            print(f"Warning: No embedding found for liked track", file=sys.stderr)
            return
        
        # Strong update to user taste
        self.user_taste.update_from_like(track_embedding)

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
            'likes': 0,
            'full_listens': 0
        }
