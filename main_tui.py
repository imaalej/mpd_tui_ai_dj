"""
Main Orchestration - Adaptive Session AI DJ
The single entry point.  Launched by start.sh.
"""

import sys
import time
import signal

from config import config
from mpd_controller import MPDController
from track_library import TrackLibrary
from user_taste import UserTaste
from session_state import SessionState
from exploration_controller import ExplorationController
from track_selector import TrackSelector
from queue_manager import QueueManager
from feedback_handler import FeedbackHandler
from persistence import Persistence, ensure_data_directories

from tui import AdaptiveDJTUI


class AdaptiveDJWithTUI:
    """
    Main orchestrator for the Adaptive Session AI DJ with TUI.
    Integrates Phase 1 intelligence core with Phase 2 user interface.
    """
    
    def __init__(self):
        self.running = False
        self.tui = None
        
        # Initialize Phase 1 components
        print("="*60, file=sys.stderr)
        print("Adaptive Session AI DJ", file=sys.stderr)
        print("="*60, file=sys.stderr)
        
        # Validate config
        config.validate()
        ensure_data_directories()
        
        # MPD Controller
        print("\n[1/10] Connecting to MPD...", file=sys.stderr)
        self.mpd_controller = MPDController()
        if not self.mpd_controller.connect():
            print("ERROR: Could not connect to MPD", file=sys.stderr)
            print(f"Make sure MPD is running on {config.mpd_host}:{config.mpd_port}", file=sys.stderr)
            sys.exit(1)
        print(f"✓ Connected to MPD at {config.mpd_host}:{config.mpd_port}", file=sys.stderr)
        
        # Track Library
        print("\n[2/10] Loading track library...", file=sys.stderr)
        self.track_library = TrackLibrary()
        
        # Embeddings are a hard requirement.  There is deliberately no
        # "generate random ones for now" fallback (audit M2/M4): random vectors
        # make every downstream number meaningless while the UI keeps reporting
        # them as if they meant something.
        if not config.embeddings_file.exists():
            print(f"\nERROR: No embeddings file at {config.embeddings_file}", file=sys.stderr)
            print("Generate them first:  python3 generate_embeddings.py", file=sys.stderr)
            sys.exit(1)

        self.track_library.load_embeddings()
        print(f"✓ Loaded {self.track_library.get_track_count()} tracks", file=sys.stderr)
        
        # User Taste
        print("\n[3/10] Initializing user taste model...", file=sys.stderr)
        self.user_taste = UserTaste()
        
        # Session State
        print("\n[4/10] Initializing session state...", file=sys.stderr)
        self.session_state = SessionState()
        
        # Exploration Controller
        print("\n[5/10] Initializing exploration controller...", file=sys.stderr)
        self.exploration_controller = ExplorationController()
        
        # Track Selector
        print("\n[6/10] Initializing track selector...", file=sys.stderr)
        self.track_selector = TrackSelector(self.track_library)
        
        # Queue Manager
        print("\n[7/10] Initializing queue manager...", file=sys.stderr)
        self.queue_manager = QueueManager(
            track_selector=self.track_selector,
            session_state=self.session_state,
            user_taste=self.user_taste,
            exploration_controller=self.exploration_controller,
            mpd_controller=self.mpd_controller
        )
        
        # Feedback Handler
        print("\n[8/10] Initializing feedback handler...", file=sys.stderr)
        self.feedback_handler = FeedbackHandler(
            session_state=self.session_state,
            user_taste=self.user_taste,
            exploration_controller=self.exploration_controller,
            queue_manager=self.queue_manager,
            track_library=self.track_library
        )
        
        # Persistence
        print("\n[9/10] Setting up persistence...", file=sys.stderr)
        self.persistence = Persistence(
            user_taste=self.user_taste,
            exploration_controller=self.exploration_controller,
            feedback_handler=self.feedback_handler
        )
        
        # Load persistent state
        self.persistence.load_all()
        
        # Initialize TUI
        print("\n[10/10] Initializing Terminal UI...", file=sys.stderr)
        self.tui = AdaptiveDJTUI(self)
        
        print("\n" + "="*60, file=sys.stderr)
        print("✓ Initialization complete!", file=sys.stderr)
        print("="*60, file=sys.stderr)
    
    def start_session(self):
        """Start a new listening session."""
        print("\n🎵 Starting session...", file=sys.stderr)
        
        # Start session state
        self.session_state.start_session()
        
        # Reset session-specific state
        self.feedback_handler.reset_session_stats()
        
        # Clear MPD queue
        self.mpd_controller.clear_queue()
        
        # Generate initial queue
        self.queue_manager.initialize_queue()

        # Do NOT auto-play on startup — leave MPD paused so the user
        # can start playback when ready with [SPACE].
        # (Queue changes mid-session still auto-play via recalculate().)
        print(f"Session ready! Press [SPACE] to begin.", file=sys.stderr)
        time.sleep(1)  # Brief pause before TUI takeover
    
    def run(self):
        """Main event loop with TUI."""
        self.running = True
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Start session
        self.start_session()
        
        # Background event processing
        import threading
        event_thread = threading.Thread(target=self._background_event_loop, daemon=True)
        event_thread.start()
        
        # Run TUI (blocks until quit)
        try:
            self.tui.run()
        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()
    
    def _background_event_loop(self):
        """
        Background thread that handles MPD events.
        Runs alongside TUI.
        """
        last_track_file = None
        track_start_time = None
        track_duration = None
        track_completion_processed = False
        last_check_time = time.time()
        # Track when the last skip action happened so we don't also
        # process a full-listen for a track that was skipped near its end.
        self._last_skip_time = 0  # set by TUI when user skips
        
        while self.running:
            try:
                # Poll MPD status
                status = self.mpd_controller.get_status()
                
                if status['state'] == 'playing':
                    current_track = status.get('track_file')
                    
                    # Detect track change
                    if current_track and current_track != last_track_file:
                        last_track_file = current_track
                        track_start_time = time.time()
                        track_duration = status.get('duration', 0)
                        track_completion_processed = False
                        
                        # Notify queue manager
                        self.queue_manager.on_track_started(current_track)
                    
                    # Check for track completion
                    if last_track_file and current_track and not track_completion_processed:
                        position = status.get('position', 0)
                        
                        if track_duration and position > 0:
                            completion_threshold = max(0.9 * track_duration, 
                                                      track_duration - 10)
                            
                            # Only fire full-listen if track was NOT manually skipped recently
                            recently_skipped = (time.time() - self._last_skip_time) < 2.0
                            if position >= completion_threshold and not recently_skipped:
                                # Track completed - process as full listen
                                self.feedback_handler.process_full_listen(last_track_file)
                                track_completion_processed = True
                
                # Periodically check queue and refill
                current_time = time.time()
                if current_time - last_check_time > 2.0:
                    self.queue_manager.check_and_refill()
                    last_check_time = current_time
                
                # Sleep to avoid busy-waiting
                time.sleep(config.mpd_poll_interval)
                
            except Exception as e:
                print(f"Background loop error: {e}", file=sys.stderr)
                time.sleep(1)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.running = False
    
    def _shutdown(self):
        """Clean shutdown."""
        print("\n\nShutting down...", file=sys.stderr)
        self.running = False
        
        # Save state
        self.persistence.save_all()
        
        # Clear album art if displayed
        if self.tui and self.tui.show_album_art:
            self.tui.album_art_renderer.clear()
        
        print("Goodbye!", file=sys.stderr)


def main():
    """Main entry point."""
    dj = AdaptiveDJWithTUI()
    dj.run()


if __name__ == '__main__':
    main()
