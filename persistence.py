"""
Persistence - Manages saving and loading of persistent state
Handles user taste, exploration state, and feedback history.
"""

import sys
from config import config


class Persistence:
    """
    Manages all persistent state save/load operations.
    Coordinates between user_taste, exploration_controller, and feedback_handler.
    """
    
    def __init__(self, user_taste, exploration_controller, feedback_handler):
        self.user_taste = user_taste
        self.exploration_controller = exploration_controller
        self.feedback_handler = feedback_handler

    def save_all(self):
        """Save all persistent state."""
        print("\nSaving state...", file=sys.stderr)

        self.user_taste.save()
        self.exploration_controller.save()
        self.feedback_handler.save_feedback_history()

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

        if taste_loaded or exploration_loaded or feedback_loaded:
            print("State loaded successfully", file=sys.stderr)
            return True
        else:
            print("No previous state found, starting fresh", file=sys.stderr)
            return False


def ensure_data_directories():
    """Ensure all data directories exist."""
    config.data_dir.mkdir(parents=True, exist_ok=True)
    (config.data_dir / 'embeddings').mkdir(exist_ok=True)
    (config.data_dir / 'state').mkdir(exist_ok=True)
