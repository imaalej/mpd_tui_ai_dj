"""
Queue Manager - Dynamic queue generation and management
Maintains rolling buffer of upcoming tracks.
"""

import sys
from typing import List, Set, Optional
from config import config


class QueueManager:
    """
    Manages dynamic queue generation.
    Generates tracks on-demand rather than precomputing long playlists.
    """
    
    def __init__(self, track_selector, session_state, user_taste, exploration_controller, mpd_controller):
        self.track_selector = track_selector
        self.session_state = session_state
        self.user_taste = user_taste
        self.exploration_controller = exploration_controller
        self.mpd_controller = mpd_controller
        
        self.planned_queue = []  # Tracks we plan to play
        self.currently_queued_in_mpd = []  # Tracks actually in MPD queue
        
    def initialize_queue(self):
        """Generate initial queue for session start."""
        print("Generating initial queue...", file=__import__("sys").stderr)
        self._generate_tracks(config.queue_buffer_size)
        self._sync_to_mpd()
    
    def check_and_refill(self):
        """
        Check queue level and refill if needed.
        Called regularly from main loop.
        """
        # Get current MPD queue state
        mpd_queue = self.mpd_controller.get_queue()
        queue_length = len(mpd_queue)

        # Also check MPD playback state so we can restart if it stopped
        # because the queue ran out.
        status = self.mpd_controller.get_status()
        mpd_state = status.get("state", "stopped")

        # Refill whenever we drop below the low threshold OR the queue is empty
        if queue_length < config.queue_low_threshold or queue_length == 0:
            tracks_needed = config.queue_buffer_size - queue_length
            if tracks_needed > 0:
                self._generate_tracks(tracks_needed)
                self._sync_to_mpd()

                # If MPD stopped because the queue was exhausted, restart it
                # automatically so playback is seamless.
                # (paused means the user paused deliberately — don't touch it.)
                if mpd_state == "stopped":
                    self.mpd_controller.play()
    
    def recalculate(self):
        """
        Recalculate entire queue (after significant feedback).
        Called after vibe skip or multiple consecutive skips.
        """
        # Check if MPD is currently playing so we can restore that state
        # after rebuilding the queue (clear_queue() stops MPD).
        status = self.mpd_controller.get_status()
        was_playing = status.get("state") == "playing"

        # Clear planned queue
        self.planned_queue.clear()

        # Clear MPD queue (this stops MPD playback)
        self.mpd_controller.clear_queue()
        self.currently_queued_in_mpd.clear()

        # Generate fresh queue and push it to MPD
        self._generate_tracks(config.queue_buffer_size)
        self._sync_to_mpd()

        # Resume playback if it was active before the recalculate.
        # This makes [V] seamless: new direction starts immediately.
        if was_playing:
            self.mpd_controller.play()
    
    def _generate_tracks(self, count: int):
        """
        Generate specified number of tracks to add to queue.
        """
        session_vector = self.session_state.get_session_vector()
        taste_vector = self.user_taste.get_taste_vector()
        weights = self.exploration_controller.get_weights()
        
        # Get time context if available (Phase 3)
        time_context = getattr(self.session_state, 'time_context', None)
        
        # Build exclusion set
        exclude_tracks = set(self.planned_queue)
        exclude_tracks.update(self.currently_queued_in_mpd)
        exclude_tracks.update(self.track_selector.get_recent_history())
        
        # Generate tracks
        for _ in range(count):
            track = self.track_selector.select_track(
                session_vector=session_vector,
                taste_vector=taste_vector,
                weights=weights,
                exclude_tracks=exclude_tracks,
                time_context=time_context
            )
            
            if track is None:
                print("Warning: Could not generate track", file=sys.stderr)
                break
            
            self.planned_queue.append(track)
            exclude_tracks.add(track)
            
            # Update session vector slightly to ensure smooth progression
            # This creates a trajectory rather than similar tracks
            track_embedding = self.track_selector.track_library.get_embedding(track)
            if track_embedding is not None:
                # Blend new track into session vector (small weight for lookahead)
                import numpy as np
                blend_weight = 0.05
                session_vector = (1 - blend_weight) * session_vector + blend_weight * track_embedding
                norm = np.linalg.norm(session_vector)
                if norm > 0:
                    session_vector = session_vector / norm
    
    def _sync_to_mpd(self):
        """
        Sync planned queue to MPD.
        Only adds new tracks not already in MPD.
        """
        # Get current MPD queue
        mpd_queue = self.mpd_controller.get_queue()
        
        # Add planned tracks that aren't in MPD yet
        for track in self.planned_queue:
            if track not in mpd_queue:
                success = self.mpd_controller.add_track(track)
                if success:
                    self.currently_queued_in_mpd.append(track)
                else:
                    print(f"Failed to add track to MPD: {track}", file=__import__("sys").stderr)
    
    def on_track_started(self, track_file: str):
        """
        Called when a track starts playing.
        Removes it from planned queue.
        """
        if track_file in self.planned_queue:
            self.planned_queue.remove(track_file)
        
        if track_file in self.currently_queued_in_mpd:
            self.currently_queued_in_mpd.remove(track_file)
    
    def get_upcoming_tracks(self) -> List[str]:
        """Get list of upcoming tracks in queue."""
        # Combine MPD queue with planned queue
        mpd_queue = self.mpd_controller.get_queue()
        
        # Return MPD queue (which should match our planned queue)
        return mpd_queue
    
    def get_stats(self) -> dict:
        """Get queue statistics."""
        return {
            'planned_queue_size': len(self.planned_queue),
            'mpd_queue_size': self.mpd_controller.get_queue_length(),
            'currently_queued_count': len(self.currently_queued_in_mpd)
        }
