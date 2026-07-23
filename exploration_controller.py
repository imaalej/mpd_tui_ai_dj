"""
Exploration Controller - Adaptive exploration vs exploitation
Dynamically adjusts based on skip/listen patterns.
"""

import json
import sys
from typing import Optional
from pathlib import Path
from config import config


class ExplorationController:
    """
    Manages exploration tendency that adapts based on user feedback.
    Increases exploration after skips, decreases after full listens.
    """
    
    def __init__(self):
        self.exploration = config.exploration_initial
        self.min_exploration = config.exploration_min
        self.max_exploration = config.exploration_max
        
        # Track recent behavior
        self.consecutive_skips = 0
        self.consecutive_listens = 0
        self.total_skips = 0
        self.total_listens = 0
        
        # Weight adjustments
        self.increase_per_skip = config.exploration_increase_per_skip
        self.decrease_per_listen = config.exploration_decrease_per_listen
    
    def increase_exploration(self):
        """
        Increase exploration tendency (after skip).
        System isn't finding tracks the user likes, so explore more.
        """
        self.exploration = min(self.max_exploration, 
                              self.exploration + self.increase_per_skip)
        self.consecutive_skips += 1
        self.consecutive_listens = 0
        self.total_skips += 1
        
        print(f"Exploration increased to {self.exploration:.2f} (skip #{self.consecutive_skips})", file=__import__("sys").stderr)
    
    def decrease_exploration(self):
        """
        Decrease exploration tendency (after full listen).
        System is finding tracks the user likes, so exploit current direction.
        """
        self.exploration = max(self.min_exploration,
                              self.exploration - self.decrease_per_listen)
        self.consecutive_listens += 1
        self.consecutive_skips = 0
        self.total_listens += 1
        
        if self.consecutive_listens % 3 == 0:
            print(f"Exploration decreased to {self.exploration:.2f} ({self.consecutive_listens} consecutive listens)", file=__import__("sys").stderr)
    
    # NOTE: set_high_exploration() is gone (audit D8/H9).  It existed only for
    # the deleted [V] key, and it did something the rest of the controller is
    # built to avoid: it discarded the accumulated evidence, resetting both
    # consecutive counters and jumping the scalar to 90% of its ceiling because
    # a key had been pressed.  [N]'s escalation reads the same intent off
    # `consecutive_skips`, which is evidence rather than an assertion.

    def taste_ramp(self, taste_updates: int) -> float:
        """
        How much of the configured taste weight β has been earned, in [0, 1].

        β used to apply at full strength from the very first track, so a
        brand-new user's "long-term taste" carried 0.3 of every scoring decision
        while being a random direction (audit L7).  The seed is now zero, and
        this ramps the weight in as updates accumulate so the handover is
        gradual rather than a cliff at the first like.
        """
        return min(1.0, max(0, taste_updates) / config.taste_ramp_updates)

    def get_weights(self, taste_updates: int = 0) -> dict:
        """
        Get current scoring weights adjusted for exploration and taste evidence.

        Args:
            taste_updates: how many updates the taste model has accumulated.
                Defaults to 0, i.e. "no evidence" — the conservative reading, so
                a caller that forgets to pass it under-weights taste rather than
                over-weighting it.

        Returns dict with:
            - session_weight: How much to weight session similarity
            - taste_weight: How much to weight taste similarity
            - novelty_weight: How much to weight novelty
            - anti_repetition_weight: How much to weight anti-repetition
        """
        # Base weights from config
        base_session = config.weight_session_similarity
        base_taste = config.weight_taste_similarity
        base_novelty = config.weight_novelty
        base_repetition = config.weight_anti_repetition

        # Adjust based on exploration tendency
        # High exploration: increase novelty, decrease session/taste similarity
        exploration_factor = self.exploration

        # Shift weight from session/taste to novelty
        novelty_boost = (exploration_factor - config.exploration_initial) * 0.5

        session_weight = max(0.1, base_session - novelty_boost * 0.5)
        taste_weight = max(0.1, base_taste - novelty_boost * 0.5)
        novelty_weight = min(0.6, base_novelty + novelty_boost)
        repetition_weight = base_repetition

        # L7's ramp, applied after the exploration shift and its floors — the
        # `max(0.1, …)` above would otherwise stop the taste term from reaching
        # zero, which is precisely the value it should hold when there is no
        # taste model.  The weight the taste term has not earned goes to the
        # session term, because "what you are listening to right now" is the only
        # other thing known about a new listener.
        ramp = self.taste_ramp(taste_updates)
        unearned = taste_weight * (1.0 - ramp)
        taste_weight -= unearned
        session_weight += unearned

        # Normalize to sum to 1.0
        total = session_weight + taste_weight + novelty_weight + repetition_weight

        return {
            'session_weight': session_weight / total,
            'taste_weight': taste_weight / total,
            'novelty_weight': novelty_weight / total,
            'anti_repetition_weight': repetition_weight / total
        }
    
    def get_stats(self) -> dict:
        """Get exploration statistics."""
        return {
            'exploration': self.exploration,
            'consecutive_skips': self.consecutive_skips,
            'consecutive_listens': self.consecutive_listens,
            'total_skips': self.total_skips,
            'total_listens': self.total_listens
        }
    
    def save(self, filepath: Optional[Path] = None):
        """Save exploration state to disk."""
        if filepath is None:
            filepath = config.exploration_file
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'exploration': float(self.exploration),
            'consecutive_skips': self.consecutive_skips,
            'consecutive_listens': self.consecutive_listens,
            'total_skips': self.total_skips,
            'total_listens': self.total_listens
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load(self, filepath: Optional[Path] = None) -> bool:
        """
        Load exploration state from disk.
        Returns True if successful, False otherwise.
        """
        if filepath is None:
            filepath = config.exploration_file
        
        if not filepath.exists():
            print(f"No saved exploration state found at {filepath}", file=__import__("sys").stderr)
            return False
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.exploration = state['exploration']
            self.consecutive_skips = state['consecutive_skips']
            self.consecutive_listens = state['consecutive_listens']
            self.total_skips = state['total_skips']
            self.total_listens = state['total_listens']
            
            print(f"Loaded exploration state: {self.exploration:.2f}, "
                  f"{self.total_skips} total skips, {self.total_listens} total listens", file=__import__("sys").stderr)
            
            return True
            
        except Exception as e:
            print(f"Error loading exploration state: {e}", file=sys.stderr)
            return False
