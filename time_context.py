"""
Time Context Awareness
Learns and adapts to time-of-day and day-of-week patterns.
"""

import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Optional
from config import config


class TimeContext:
    """
    Maintains time-aware context vectors that learn listening patterns
    for different times of day and days of the week.
    """
    
    def __init__(self, dimension: int = None):
        """
        Initialize time context.
        
        Args:
            dimension: Embedding dimension
        """
        self.dimension = dimension or config.embedding_dimension
        
        # Time period context vectors
        self.time_vectors = {
            period: np.zeros(self.dimension)
            for period in config.time_periods.keys()
        }
        
        # Update counts for each period
        self.update_counts = {
            period: 0
            for period in config.time_periods.keys()
        }
        
        # Last update time for each period
        self.last_updates = {
            period: None
            for period in config.time_periods.keys()
        }
        
        # Overall statistics
        self.total_updates = 0
        
    def get_current_period(self) -> str:
        """
        Get current time period based on hour of day.
        
        Returns:
            Time period name (e.g., 'morning', 'evening')
        """
        now = datetime.now()
        hour = now.hour
        
        for period, (start, end) in config.time_periods.items():
            if start < end:
                # Normal range (e.g., morning 6-10)
                if start <= hour < end:
                    return period
            else:
                # Wraps midnight (e.g., night 22-6)
                if hour >= start or hour < end:
                    return period
        
        # Default to afternoon if not matched
        return 'afternoon'
    
    def get_current_day_type(self) -> str:
        """
        Get current day type (weekday or weekend).
        
        Returns:
            'weekday' or 'weekend'
        """
        now = datetime.now()
        weekday = now.weekday()  # Monday=0, Sunday=6
        
        if weekday in config.weekdays:
            return 'weekday'
        else:
            return 'weekend'
    
    def update(
        self,
        track_embedding: np.ndarray,
        update_rate: float,
        period: Optional[str] = None
    ):
        """
        Update time context vector for current (or specified) time period.
        
        Args:
            track_embedding: Track embedding vector
            update_rate: Learning rate for update
            period: Optional specific period (uses current if None)
        """
        if period is None:
            period = self.get_current_period()
        
        if period not in self.time_vectors:
            return
        
        # Normalize input
        norm = np.linalg.norm(track_embedding)
        if norm > 1e-8:
            track_embedding = track_embedding / norm
        
        # Update vector using exponential moving average
        alpha = update_rate
        self.time_vectors[period] = (
            (1 - alpha) * self.time_vectors[period] +
            alpha * track_embedding
        )
        
        # Re-normalize (important for similarity calculations)
        vector_norm = np.linalg.norm(self.time_vectors[period])
        if vector_norm > 1e-8:
            self.time_vectors[period] = self.time_vectors[period] / vector_norm
        
        # Update counts
        self.update_counts[period] += 1
        self.last_updates[period] = datetime.now()
        self.total_updates += 1
    
    def get_similarity(
        self,
        track_embedding: np.ndarray,
        period: Optional[str] = None
    ) -> float:
        """
        Calculate similarity between track and time context.
        
        Args:
            track_embedding: Track embedding vector
            period: Optional specific period (uses current if None)
            
        Returns:
            Cosine similarity score (0-1, or 0 if not enough data)
        """
        if period is None:
            period = self.get_current_period()
        
        if period not in self.time_vectors:
            return 0.0
        
        # Check if we have enough data for this period
        if self.update_counts[period] < 5:
            # Not enough data yet - return neutral
            return 0.0
        
        # Normalize input
        norm = np.linalg.norm(track_embedding)
        if norm > 1e-8:
            track_embedding = track_embedding / norm
        
        # Calculate cosine similarity
        similarity = float(np.dot(self.time_vectors[period], track_embedding))
        
        # Clamp to [0, 1] range
        return max(0.0, min(1.0, similarity))
    
    def get_day_modifier(self) -> float:
        """
        Get exploration modifier based on day of week.
        
        Returns:
            Modifier value (< 1 reduces exploration, > 1 increases it)
        """
        if not config.enable_day_context:
            return 1.0
        
        day_type = self.get_current_day_type()
        
        if day_type == 'weekday':
            return config.weekday_exploration_modifier
        else:
            return config.weekend_exploration_modifier
    
    def get_stats(self) -> Dict:
        """
        Get statistics about time context.
        
        Returns:
            Dictionary with statistics
        """
        current_period = self.get_current_period()
        current_day = self.get_current_day_type()
        
        period_stats = {}
        for period in config.time_periods.keys():
            last_update = self.last_updates[period]
            period_stats[period] = {
                'updates': self.update_counts[period],
                'last_update': last_update.isoformat() if last_update else None,
                'has_data': self.update_counts[period] >= 5
            }
        
        return {
            'current_period': current_period,
            'current_day_type': current_day,
            'total_updates': self.total_updates,
            'periods': period_stats,
            'day_modifier': self.get_day_modifier()
        }
    
    def reset_period(self, period: str):
        """Reset a specific time period."""
        if period in self.time_vectors:
            self.time_vectors[period] = np.zeros(self.dimension)
            self.update_counts[period] = 0
            self.last_updates[period] = None
    
    def reset_all(self):
        """Reset all time context data."""
        for period in self.time_vectors.keys():
            self.time_vectors[period] = np.zeros(self.dimension)
            self.update_counts[period] = 0
            self.last_updates[period] = None
        self.total_updates = 0
    
    def to_dict(self) -> Dict:
        """
        Serialize to dictionary for persistence.
        
        Returns:
            Dictionary representation
        """
        return {
            'dimension': self.dimension,
            'time_vectors': {k: v.tolist() for k, v in self.time_vectors.items()},
            'update_counts': self.update_counts,
            'last_updates': {
                k: v.isoformat() if v else None
                for k, v in self.last_updates.items()
            },
            'total_updates': self.total_updates
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TimeContext':
        """
        Deserialize from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            TimeContext instance
        """
        dimension = data['dimension']
        context = cls(dimension=dimension)
        
        # Restore time vectors
        for period, vector_list in data['time_vectors'].items():
            context.time_vectors[period] = np.array(vector_list)
        
        # Restore counts
        context.update_counts = data['update_counts']
        
        # Restore last updates
        for period, iso_str in data['last_updates'].items():
            if iso_str:
                context.last_updates[period] = datetime.fromisoformat(iso_str)
        
        context.total_updates = data['total_updates']
        
        return context
    
    def save(self, filepath):
        """
        Save time context to file.
        
        Args:
            filepath: Path to save file
        """
        import json
        data = self.to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath) -> Optional['TimeContext']:
        """
        Load time context from file.
        
        Args:
            filepath: Path to load file
            
        Returns:
            TimeContext instance or None if file doesn't exist
        """
        import json
        from pathlib import Path
        
        path = Path(filepath)
        if not path.exists():
            return None
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            print(f"Warning: Failed to load time context: {e}")
            return None
