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
    
    def set_high_exploration(self):
        """
        Force high exploration mode (after vibe skip).
        User wants something completely different.
        """
        self.exploration = self.max_exploration * 0.9
        self.consecutive_skips = 0
        self.consecutive_listens = 0
        print(f"High exploration mode activated: {self.exploration:.2f}", file=__import__("sys").stderr)
    
    def get_exploration_factor(self, time_context=None) -> float:
        """
        Get current exploration factor [0, 1].
        Higher values mean more exploration (novelty seeking).
        Applies day-of-week modifier if time context provided (Phase 3).
        
        Args:
            time_context: Optional TimeContext for day-of-week awareness
        
        Returns:
            Exploration factor adjusted for current context
        """
        base_exploration = self.exploration
        
        # Apply day-of-week modifier (Phase 3)
        if time_context and config.enable_day_context:
            modifier = time_context.get_day_modifier()
            return base_exploration * modifier
        
        return base_exploration
    
    def get_weights(self) -> dict:
        """
        Get current scoring weights adjusted for exploration.
        
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
    
    def reset(self):
        """Reset exploration state."""
        self.exploration = config.exploration_initial
        self.consecutive_skips = 0
        self.consecutive_listens = 0
        self.total_skips = 0
        self.total_listens = 0
        print("Exploration state reset", file=__import__("sys").stderr)
