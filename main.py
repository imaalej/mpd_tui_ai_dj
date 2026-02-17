"""
Main Orchestration - Adaptive Session AI DJ (Phase 1)
Headless adaptive intelligence core with MPD integration.
"""

import sys
import time
import signal
from pathlib import Path

# Import all components
from config import config
from mpd_controller import MPDController
from track_library import TrackLibrary, generate_dummy_embeddings
from user_taste import UserTaste
from session_state import SessionState
from exploration_controller import ExplorationController
from track_selector import TrackSelector
from queue_manager import QueueManager
from feedback_handler import FeedbackHandler
from persistence import Persistence, ensure_data_directories


class AdaptiveDJ:
    """Main orchestrator for the Adaptive Session AI DJ."""
    
    def __init__(self):
        self.running = False
        self.last_track_file = None
        self.track_start_time = None
        self.track_duration = None
        self.track_completion_processed = False  # Flag to prevent double-counting
        
        # Initialize components
        print("="*60)
        print("Adaptive Session AI DJ - Phase 1 (Headless)")
        print("="*60)
        
        # Validate config
        config.validate()
        ensure_data_directories()
        
        # MPD Controller
        print("\n[1/9] Connecting to MPD...")
        self.mpd_controller = MPDController()
        if not self.mpd_controller.connect():
            print("ERROR: Could not connect to MPD")
            print(f"Make sure MPD is running on {config.mpd_host}:{config.mpd_port}")
            sys.exit(1)
        print(f"✓ Connected to MPD at {config.mpd_host}:{config.mpd_port}")
        
        # Track Library
        print("\n[2/9] Loading track library...")
        self.track_library = TrackLibrary()
        
        # Check if embeddings exist, if not, generate dummy ones for testing
        if not config.embeddings_file.exists():
            print("\n⚠️  No embeddings file found!")
            print("For testing, I can generate random embeddings.")
            print("In production, you'd generate real embeddings using an audio model.")
            response = input("Generate dummy embeddings now? (y/n): ")
            
            if response.lower() == 'y':
                print("Fetching tracks from MPD...")
                mpd_tracks = self.mpd_controller.list_all_tracks()
                if not mpd_tracks:
                    print("ERROR: No tracks found in MPD database")
                    print("Make sure your MPD music directory contains music files")
                    sys.exit(1)
                
                generate_dummy_embeddings(mpd_tracks, config.embeddings_file)
            else:
                print("Cannot proceed without embeddings")
                sys.exit(1)
        
        self.track_library.load_embeddings()
        print(f"✓ Loaded {self.track_library.get_track_count()} tracks")
        
        # User Taste
        print("\n[3/9] Initializing user taste model...")
        self.user_taste = UserTaste()
        
        # Session State
        print("\n[4/9] Initializing session state...")
        self.session_state = SessionState()
        
        # Exploration Controller
        print("\n[5/9] Initializing exploration controller...")
        self.exploration_controller = ExplorationController()
        
        # Track Selector
        print("\n[6/9] Initializing track selector...")
        self.track_selector = TrackSelector(self.track_library)
        
        # Queue Manager
        print("\n[7/9] Initializing queue manager...")
        self.queue_manager = QueueManager(
            track_selector=self.track_selector,
            session_state=self.session_state,
            user_taste=self.user_taste,
            exploration_controller=self.exploration_controller,
            mpd_controller=self.mpd_controller
        )
        
        # Feedback Handler
        print("\n[8/9] Initializing feedback handler...")
        self.feedback_handler = FeedbackHandler(
            session_state=self.session_state,
            user_taste=self.user_taste,
            exploration_controller=self.exploration_controller,
            queue_manager=self.queue_manager,
            track_library=self.track_library
        )
        
        # Persistence
        print("\n[9/9] Setting up persistence...")
        self.persistence = Persistence(
            user_taste=self.user_taste,
            exploration_controller=self.exploration_controller,
            feedback_handler=self.feedback_handler,
            session_state=self.session_state   # needed to persist time context
        )
        
        # Load persistent state
        self.persistence.load_all()
        
        print("\n" + "="*60)
        print("✓ Initialization complete!")
        print("="*60)
    
    def start_session(self):
        """Start a new listening session."""
        print("\n🎵 Starting new session...")
        
        # Start session state
        self.session_state.start_session()
        
        # Reset session-specific state
        self.feedback_handler.reset_session_stats()
        
        # Clear MPD queue
        self.mpd_controller.clear_queue()
        
        # Generate initial queue
        self.queue_manager.initialize_queue()
        
        # Start playback
        self.mpd_controller.play()
        
        print(f"Session started! Current vibe: {self.session_state.get_vibe_description()}")
        self._print_queue()
    
    def run(self):
        """Main event loop."""
        self.running = True
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print("\n" + "="*60)
        print("Controls:")
        print("  s - Skip current song")
        print("  v - Skip vibe (change direction)")
        print("  l - Like current track")
        print("  i - Show info")
        print("  q - Quit")
        print("="*60 + "\n")
        
        # Start session
        self.start_session()
        
        last_check_time = time.time()
        
        try:
            while self.running:
                # Poll MPD status
                status = self.mpd_controller.get_status()
                
                if status['state'] == 'playing':
                    current_track = status.get('track_file')
                    
                    # Detect track change
                    if current_track and current_track != self.last_track_file:
                        self._on_track_changed(current_track, status)
                    
                    # Check for track completion
                    if self.last_track_file and current_track:
                        self._check_track_completion(status)
                
                # Periodically check queue and refill
                current_time = time.time()
                if current_time - last_check_time > 2.0:
                    self.queue_manager.check_and_refill()
                    last_check_time = current_time
                
                # Check for user input (non-blocking)
                self._check_input()
                
                # Sleep to avoid busy-waiting
                time.sleep(config.mpd_poll_interval)
                
        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()
    
    def _on_track_changed(self, track_file: str, status: dict):
        """Handle track change event."""
        self.last_track_file = track_file
        self.track_start_time = time.time()
        self.track_duration = status.get('duration', 0)
        self.track_completion_processed = False  # Reset flag for new track
        
        # Notify queue manager
        self.queue_manager.on_track_started(track_file)
        
        # Display now playing
        artist = status.get('artist', 'Unknown')
        title = status.get('title', 'Unknown')
        album = status.get('album', 'Unknown')
        
        print(f"\n♪ Now Playing: {artist} - {title}")
        print(f"   Album: {album}")
        print(f"   Vibe: {self.session_state.get_vibe_description()}")
    
    def _check_track_completion(self, status: dict):
        """Check if current track completed (full listen)."""
        if not self.track_start_time or not self.track_duration:
            return
        
        # Already processed this track's completion
        if self.track_completion_processed:
            return
        
        position = status.get('position', 0)
        
        # Consider track completed if played >90% or within 10 seconds of end
        completion_threshold = max(0.9 * self.track_duration, self.track_duration - 10)
        
        if position >= completion_threshold and self.last_track_file:
            # Track completed - process as full listen
            self.feedback_handler.process_full_listen(self.last_track_file)
            self.track_completion_processed = True  # Mark as processed
    
    def _check_input(self):
        """Check for user input (non-blocking on Unix)."""
        import select
        
        # Check if input is available
        if select.select([sys.stdin], [], [], 0.0)[0]:
            command = sys.stdin.readline().strip().lower()
            self._process_command(command)
    
    def _process_command(self, command: str):
        """Process user command."""
        status = self.mpd_controller.get_status()
        current_track = status.get('track_file')
        
        if command == 's':
            # Skip current song
            if current_track:
                self.feedback_handler.process_skip(current_track)
            self.mpd_controller.next_track()
            
        elif command == 'v':
            # Skip vibe
            if current_track:
                self.feedback_handler.process_vibe_skip(current_track)
            self.mpd_controller.next_track()
            
        elif command == 'l':
            # Like current track
            if current_track:
                self.feedback_handler.process_like(current_track)
            
        elif command == 'i':
            # Show info
            self._show_info()
            
        elif command == 'q':
            # Quit
            print("\nShutting down...")
            self.running = False
        
        elif command:
            print(f"Unknown command: {command}")
    
    def _show_info(self):
        """Display current system information."""
        print("\n" + "="*60)
        print("SYSTEM INFO")
        print("="*60)
        
        # Session stats
        print("\nSession:")
        session_stats = self.session_state.get_stats()
        print(f"  Tracks played: {session_stats['tracks_played']}")
        print(f"  Vibe: {session_stats['vibe_description']}")
        
        # Taste stats
        taste_stats = self.user_taste.get_stats()
        print(f"\nUser Taste:")
        print(f"  Total updates: {taste_stats['total_updates']}")
        print(f"  Likes: {taste_stats['like_count']}")
        print(f"  Full listens: {taste_stats['full_listen_count']}")
        print(f"  Skips: {taste_stats['skip_count']}")
        
        # Exploration stats
        exploration_stats = self.exploration_controller.get_stats()
        print(f"\nExploration:")
        print(f"  Current level: {exploration_stats['exploration']:.2f}")
        print(f"  Consecutive skips: {exploration_stats['consecutive_skips']}")
        print(f"  Consecutive listens: {exploration_stats['consecutive_listens']}")
        
        # Queue info
        self._print_queue()
        
        print("="*60 + "\n")
    
    def _print_queue(self):
        """Print current queue."""
        queue = self.queue_manager.get_upcoming_tracks()
        print(f"\nUpcoming Queue ({len(queue)} tracks):")
        for i, track in enumerate(queue[:5], 1):
            # Shorten track path for display
            display_name = Path(track).stem
            print(f"  {i}. {display_name}")
        if len(queue) > 5:
            print(f"  ... and {len(queue) - 5} more")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print("\nReceived shutdown signal...")
        self.running = False
    
    def _shutdown(self):
        """Clean shutdown."""
        print("\nShutting down...")
        
        # Save state
        self.persistence.save_all()
        
        print("Goodbye!")


def main():
    """Main entry point."""
    dj = AdaptiveDJ()
    dj.run()


if __name__ == '__main__':
    main()
