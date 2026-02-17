"""
Persistence - Manages saving and loading of persistent state
Handles user taste, exploration state, and feedback history.
"""

import sys
from pathlib import Path
from typing import Optional
from config import config


class Persistence:
    """
    Manages all persistent state save/load operations.
    Coordinates between user_taste, exploration_controller, and feedback_handler.
    """
    
    def __init__(self, user_taste, exploration_controller, feedback_handler, session_state=None):
        self.user_taste = user_taste
        self.exploration_controller = exploration_controller
        self.feedback_handler = feedback_handler
        self.session_state = session_state  # For time_context (Phase 3)
    
    def save_all(self):
        """Save all persistent state."""
        print("\nSaving state...", file=sys.stderr)
        
        self.user_taste.save()
        self.exploration_controller.save()
        self.feedback_handler.save_feedback_history()
        
        # Save time context (Phase 3)
        if self.session_state and hasattr(self.session_state, 'time_context'):
            try:
                self.session_state.time_context.save(config.context_file)
            except Exception as e:
                print(f"Warning: Failed to save time context: {e}", file=sys.stderr)
        
        print("State saved successfully", file=sys.stderr)
    
    def load_all(self) -> bool:
        """
        Load all persistent state.
        Returns True if at least some state was loaded.
        """
        print("\nLoading persistent state...", file=sys.stderr)
        
        taste_loaded = self.user_taste.load()
        exploration_loaded = self.exploration_controller.load()
        feedback_loaded = self.feedback_handler.load_feedback_history()
        
        # Load time context (Phase 3)
        context_loaded = False
        if self.session_state and hasattr(self.session_state, 'time_context'):
            try:
                from time_context import TimeContext
                loaded_context = TimeContext.load(config.context_file)
                if loaded_context:
                    self.session_state.time_context = loaded_context
                    context_loaded = True
                    print("Time context loaded", file=sys.stderr)
            except Exception as e:
                print(f"Note: Time context not loaded: {e}", file=sys.stderr)
        
        if taste_loaded or exploration_loaded or feedback_loaded or context_loaded:
            print("State loaded successfully", file=sys.stderr)
            return True
        else:
            print("No previous state found, starting fresh", file=sys.stderr)
            return False
    
    def reset_all(self):
        """Reset all persistent state (for fresh start)."""
        print("\nResetting all persistent state...", file=sys.stderr)
        
        self.user_taste.reset()
        self.exploration_controller.reset()
        self.feedback_handler.reset_session_stats()
        
        # Reset time context (Phase 3)
        if self.session_state and hasattr(self.session_state, 'time_context'):
            self.session_state.time_context.reset_all()
        
        # Save the reset state
        self.save_all()
        
        print("All state reset", file=sys.stderr)


def ensure_data_directories():
    """Ensure all data directories exist."""
    config.data_dir.mkdir(parents=True, exist_ok=True)
    (config.data_dir / 'embeddings').mkdir(exist_ok=True)
    (config.data_dir / 'state').mkdir(exist_ok=True)
