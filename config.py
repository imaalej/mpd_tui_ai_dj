"""
Configuration Management for Adaptive Session AI DJ
Provides all configurable parameters for the system.
"""

import os
from pathlib import Path


class Config:
    """Central configuration for the adaptive DJ system."""
    
    def __init__(self):
        # MPD Connection Settings
        self.mpd_host = os.getenv('MPD_HOST', 'localhost')
        self.mpd_port = int(os.getenv('MPD_PORT', '6600'))
        self.mpd_music_directory = os.getenv('MPD_MUSIC_DIR', '/var/lib/mpd/music')
        
        # Track Selection Scoring Weights (α, β, γ, δ)
        # These control the multi-factor scoring function
        self.weight_session_similarity = 0.4  # α - How much session state influences selection
        self.weight_taste_similarity = 0.3     # β - How much long-term taste influences selection
        self.weight_novelty = 0.2              # γ - How much we prefer novel tracks
        self.weight_anti_repetition = 0.1      # δ - How much we penalize recent plays
        
        # Exploration vs Exploitation Parameters
        self.exploration_initial = 0.3         # Starting exploration tendency (0-1)
        self.exploration_min = 0.1             # Minimum exploration (always some novelty)
        self.exploration_max = 0.7             # Maximum exploration (never completely random)
        self.exploration_increase_per_skip = 0.05  # How much to increase after skip
        self.exploration_decrease_per_listen = 0.02  # How much to decrease after full listen
        
        # Session State Parameters
        self.session_decay_factor = 0.85       # Exponential decay for session vector updates
        self.session_influence_window = 5      # Number of recent tracks influencing session
        self.vibe_shift_magnitude = 0.5        # How strongly to shift on vibe skip (0-1)
        
        # User Taste Update Parameters
        self.taste_update_like = 0.1           # Weight for explicit likes
        self.taste_update_full_listen = 0.02   # Weight for passive full listens
        self.taste_update_skip_penalty = -0.05  # Penalty for skips
        
        # Queue Management
        self.queue_buffer_size = 10            # Number of tracks to maintain in queue
        self.queue_low_threshold = 3           # Generate more tracks when below this
        
        # Candidate Pool Parameters
        self.candidate_pool_size = 100         # Number of candidates to retrieve for scoring
        self.similarity_search_k = 200         # Initial similarity search results
        
        # Repetition Avoidance
        self.recent_history_size = 50          # Tracks to remember for anti-repetition
        self.minimum_replay_gap = 20           # Minimum tracks before replaying
        
        # Persistence Paths
        self.data_dir = Path(__file__).parent / 'data'
        self.embeddings_file = self.data_dir / 'embeddings' / 'track_embeddings.npz'
        self.taste_file = self.data_dir / 'state' / 'user_taste.npz'
        self.exploration_file = self.data_dir / 'state' / 'exploration_state.json'
        self.feedback_history_file = self.data_dir / 'state' / 'feedback_history.json'
        
        # System Parameters
        self.mpd_poll_interval = 0.5           # Seconds between MPD status polls
        self.embedding_dimension = 512         # Expected embedding vector size (CLAP Phase 3)
        
        # Context Awareness Parameters (Phase 3)
        self.enable_time_context = True        # Enable time-of-day awareness
        self.enable_day_context = True         # Enable day-of-week awareness
        
        # Time period boundaries (hour ranges)
        self.time_periods = {
            'morning': (6, 10),       # 6am - 10am
            'midday': (10, 14),       # 10am - 2pm
            'afternoon': (14, 18),    # 2pm - 6pm
            'evening': (18, 22),      # 6pm - 10pm
            'night': (22, 6),         # 10pm - 6am (wraps around)
        }
        
        # Weekday definition (Monday=0, Sunday=6)
        self.weekdays = [0, 1, 2, 3, 4]        # Monday-Friday
        
        # Context update rates
        self.time_update_rate_like = 0.05      # Update rate for explicit likes
        self.time_update_rate_listen = 0.02    # Update rate for full listens
        
        # Context scoring weight
        self.time_context_weight = 0.15        # Weight for time similarity in scoring
        
        # Day-of-week exploration modifiers
        self.weekday_exploration_modifier = 0.8   # Reduce exploration on weekdays
        self.weekend_exploration_modifier = 1.2   # Increase exploration on weekends
        
        # Context persistence file
        self.context_file = self.data_dir / 'state' / 'time_context.npz'
        
        # Ensure data directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / 'embeddings').mkdir(exist_ok=True)
        (self.data_dir / 'state').mkdir(exist_ok=True)
    
    def validate(self):
        """Validate configuration parameters."""
        assert 0 <= self.weight_session_similarity <= 1
        assert 0 <= self.weight_taste_similarity <= 1
        assert 0 <= self.weight_novelty <= 1
        assert 0 <= self.weight_anti_repetition <= 1
        
        total_weight = (self.weight_session_similarity + 
                       self.weight_taste_similarity + 
                       self.weight_novelty + 
                       self.weight_anti_repetition)
        assert abs(total_weight - 1.0) < 0.01, f"Weights must sum to 1.0, got {total_weight}"
        
        assert 0 <= self.exploration_min <= self.exploration_max <= 1
        assert self.queue_buffer_size >= self.queue_low_threshold
        
        return True


# Global config instance
config = Config()
