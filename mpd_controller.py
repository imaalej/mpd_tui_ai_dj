"""
MPD Controller - Integration layer for Music Player Daemon
All playback control via MPC commands with robust metadata parsing.
"""

import sys
import subprocess
import re
from typing import Optional, Dict, List
from pathlib import Path
from config import config


class MPDController:
    """Manages all interaction with MPD via MPC commands."""
    
    def __init__(self):
        self.host = config.mpd_host
        self.port = config.mpd_port
        self.music_directory = config.mpd_music_directory
        self.connected = False
        self._current_track = None
        self._last_status = {}
        
    def connect(self) -> bool:
        """Verify connection to MPD."""
        try:
            result = subprocess.run(
                ['mpc', '-h', self.host, '-p', str(self.port), 'status'],
                capture_output=True,
                text=True,
                timeout=5
            )
            self.connected = result.returncode == 0
            return self.connected
        except Exception as e:
            print(f"MPD connection failed: {e}", file=sys.stderr)
            self.connected = False
            return False
    
    def get_status(self) -> Dict:
        """
        Poll MPD for current playback status.
        Returns dict with: state, track, position, duration, volume
        """
        if not self.connected and not self.connect():
            return {'state': 'disconnected'}
        
        try:
            # Use mpc current to get current track info
            current_result = subprocess.run(
                ['mpc', '-h', self.host, '-p', str(self.port), 'current', '-f',
                 '%file%\n%artist%\n%album%\n%title%'],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            # Get status line with position/duration
            status_result = subprocess.run(
                ['mpc', '-h', self.host, '-p', str(self.port), 'status'],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            status = {
                'state': 'stopped',
                'track_file': None,
                'artist': 'Unknown Artist',
                'album': 'Unknown Album',
                'title': 'Unknown Title',
                'position': 0,
                'duration': 0,
                'volume': 100
            }
            
            # Parse current track info
            if current_result.returncode == 0 and current_result.stdout.strip():
                lines = current_result.stdout.strip().split('\n')
                if len(lines) >= 1:
                    status['track_file'] = lines[0].strip()
                if len(lines) >= 2 and lines[1].strip():
                    status['artist'] = lines[1].strip()
                if len(lines) >= 3 and lines[2].strip():
                    status['album'] = lines[2].strip()
                if len(lines) >= 4 and lines[3].strip():
                    status['title'] = lines[3].strip()
                
                # If title is still unknown, use filename
                if status['title'] == 'Unknown Title' and status['track_file']:
                    status['title'] = Path(status['track_file']).stem
            
            # Parse status output
            if status_result.returncode == 0:
                status_output = status_result.stdout
                
                # Parse state
                if '[playing]' in status_output:
                    status['state'] = 'playing'
                elif '[paused]' in status_output:
                    status['state'] = 'paused'
                else:
                    status['state'] = 'stopped'
                
                # Parse position and duration (format: #1/10 0:05/3:45)
                # Look for pattern like "0:05/3:45" or "1:23/4:56"
                time_match = re.search(r'(\d+):(\d+)/(\d+):(\d+)', status_output)
                if time_match:
                    pos_min, pos_sec, dur_min, dur_sec = map(int, time_match.groups())
                    status['position'] = pos_min * 60 + pos_sec
                    status['duration'] = dur_min * 60 + dur_sec
                
                # Parse volume
                volume_match = re.search(r'volume:\s*(\d+)%', status_output)
                if volume_match:
                    status['volume'] = int(volume_match.group(1))
            
            self._last_status = status
            return status
            
        except subprocess.TimeoutExpired:
            return {'state': 'timeout'}
        except Exception as e:
            print(f"Error getting MPD status: {e}", file=sys.stderr)
            return self._last_status if self._last_status else {'state': 'error'}
    
    def play(self):
        """Start playback."""
        try:
            subprocess.run(
                ['mpc', '-h', self.host, '-p', str(self.port), 'play'],
                capture_output=True,
                timeout=2
            )
        except Exception as e:
            print(f"Play command failed: {e}", file=sys.stderr)
    
    def pause(self):
        """Pause playback."""
        try:
            subprocess.run(
                ['mpc', '-h', self.host, '-p', str(self.port), 'pause'],
                capture_output=True,
                timeout=2
            )
        except Exception as e:
            print(f"Pause command failed: {e}", file=sys.stderr)
    
    def toggle(self):
        """Toggle play/pause."""
        try:
            subprocess.run(
                ['mpc', '-h', self.host, '-p', str(self.port), 'toggle'],
                capture_output=True,
                timeout=2
            )
        except Exception as e:
            print(f"Toggle command failed: {e}", file=sys.stderr)
    
    def next_track(self):
        """Skip to next track."""
        try:
            subprocess.run(
                ['mpc', '-h', self.host, '-p', str(self.port), 'next'],
                capture_output=True,
                timeout=2
            )
        except Exception as e:
            print(f"Next command failed: {e}", file=sys.stderr)
    
    def previous_track(self):
        """Go to previous track."""
        try:
            subprocess.run(
                ['mpc', '-h', self.host, '-p', str(self.port), 'prev'],
                capture_output=True,
                timeout=2
            )
        except Exception as e:
            print(f"Previous command failed: {e}", file=sys.stderr)
    
    def seek(self, seconds: int):
        """Seek to absolute position in current track (seconds)."""
        try:
            subprocess.run(
                ['mpc', '-h', self.host, '-p', str(self.port), 'seek', str(seconds)],
                capture_output=True,
                timeout=2
            )
        except Exception as e:
            print(f"Seek command failed: {e}", file=sys.stderr)

    def seek_relative(self, delta: int):
        """
        Seek forward (positive) or backward (negative) by delta seconds.
        Uses mpc seek +N / -N syntax which is relative to current position.
        Clamps to track bounds so we never seek past the end.
        """
        status = self._last_status if self._last_status else {}
        position = status.get('position', 0)
        duration = status.get('duration', 0)

        if delta < 0:
            # Seeking backward — clamp to 0
            new_pos = max(0, position + delta)
            mpc_arg = str(new_pos)          # absolute is safer for backward
        else:
            # Seeking forward — clamp to duration - 1 so we don't skip track
            if duration > 0:
                new_pos = min(duration - 1, position + delta)
                mpc_arg = str(new_pos)
            else:
                mpc_arg = f'+{delta}'        # no duration info; let mpc handle it

        try:
            subprocess.run(
                ['mpc', '-h', self.host, '-p', str(self.port), 'seek', mpc_arg],
                capture_output=True,
                timeout=2
            )
        except Exception as e:
            print(f"Seek relative command failed: {e}", file=__import__('sys').stderr)
    
    def set_volume(self, volume: int):
        """Set volume (0-100)."""
        volume = max(0, min(100, volume))
        try:
            subprocess.run(
                ['mpc', '-h', self.host, '-p', str(self.port), 'volume', str(volume)],
                capture_output=True,
                timeout=2
            )
        except Exception as e:
            print(f"Volume command failed: {e}", file=sys.stderr)
    
    def volume_up(self, delta: int = 5):
        """Increase volume."""
        try:
            subprocess.run(
                ['mpc', '-h', self.host, '-p', str(self.port), 'volume', f'+{delta}'],
                capture_output=True,
                timeout=2
            )
        except Exception as e:
            print(f"Volume up command failed: {e}", file=sys.stderr)
    
    def volume_down(self, delta: int = 5):
        """Decrease volume."""
        try:
            subprocess.run(
                ['mpc', '-h', self.host, '-p', str(self.port), 'volume', f'-{delta}'],
                capture_output=True,
                timeout=2
            )
        except Exception as e:
            print(f"Volume down command failed: {e}", file=sys.stderr)
    
    def add_track(self, track_file: str) -> bool:
        """Add track to queue."""
        try:
            result = subprocess.run(
                ['mpc', '-h', self.host, '-p', str(self.port), 'add', track_file],
                capture_output=True,
                timeout=2
            )
            return result.returncode == 0
        except Exception as e:
            print(f"Add track failed: {e}", file=sys.stderr)
            return False
    
    def clear_queue(self):
        """Clear the playback queue."""
        try:
            subprocess.run(
                ['mpc', '-h', self.host, '-p', str(self.port), 'clear'],
                capture_output=True,
                timeout=2
            )
        except Exception as e:
            print(f"Clear queue failed: {e}", file=sys.stderr)
    
    def get_queue(self) -> List[str]:
        """Get current queue as list of track files."""
        try:
            result = subprocess.run(
                ['mpc', '-h', self.host, '-p', str(self.port), 'playlist', '-f', '%file%'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                return [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            return []
        except Exception as e:
            print(f"Get queue failed: {e}", file=sys.stderr)
            return []
    
    def get_queue_length(self) -> int:
        """Get number of tracks in queue."""
        return len(self.get_queue())
    
    def get_all_tracks(self) -> List[str]:
        """Get all tracks in MPD database."""
        try:
            result = subprocess.run(
                ['mpc', '-h', self.host, '-p', str(self.port), 'listall'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            return []
        except Exception as e:
            print(f"Get all tracks failed: {e}", file=sys.stderr)
            return []
    
    def update_database(self):
        """Trigger MPD database update."""
        try:
            subprocess.run(
                ['mpc', '-h', self.host, '-p', str(self.port), 'update'],
                capture_output=True,
                timeout=2
            )
            print("MPD database update triggered", file=sys.stderr)
        except Exception as e:
            print(f"Database update failed: {e}", file=sys.stderr)
    
    def get_playlist_metadata(self) -> dict:
        """
        Get metadata for all tracks in current playlist.
        Returns dict keyed by file path.

        NOTE: `mpc playlist -f FORMAT` does NOT support the -f flag — it is
        silently ignored, producing no tag data.  Instead we fetch the plain
        file list and then resolve each track's tags via `mpc search file`.
        Results are cached so repeated calls are fast.
        """
        # --- Step 1: get file list from playlist ---
        try:
            list_result = subprocess.run(
                ['mpc', '-h', self.host, '-p', str(self.port), 'playlist', '-f', '%file%'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if list_result.returncode != 0 or not list_result.stdout.strip():
                return {}
            track_files = [
                line.strip()
                for line in list_result.stdout.strip().split('\n')
                if line.strip()
            ]
        except Exception:
            return {}

        # --- Step 2: for each file resolve tags, using cache ---
        if not hasattr(self, '_metadata_cache'):
            self._metadata_cache: Dict[str, dict] = {}

        metadata_map: Dict[str, dict] = {}
        for track_file in track_files:
            if track_file in self._metadata_cache:
                metadata_map[track_file] = self._metadata_cache[track_file]
                continue

            meta = self._fetch_track_tags(track_file)
            self._metadata_cache[track_file] = meta
            metadata_map[track_file] = meta

        return metadata_map

    def _fetch_track_tags(self, track_file: str) -> dict:
        """
        Fetch artist/album/title for a single track file.

        Strategy (in order of preference):
          1. mutagen  – reads ID3/FLAC/etc. tags directly from the file.
          2. mpc search file – asks MPD for the track info by file path.
          3. Filename stem fallback.
        """
        # --- Try mutagen first (most reliable, zero extra processes) ---
        try:
            import mutagen
            full_path = Path(self.music_directory) / track_file
            audio = mutagen.File(str(full_path), easy=True)
            if audio is not None:
                artist = (audio.get('artist') or audio.get('albumartist') or [''])[0]
                album  = (audio.get('album')  or [''])[0]
                title  = (audio.get('title')  or [''])[0]
                return {
                    'artist': artist.strip() or 'Unknown Artist',
                    'album':  album.strip()  or 'Unknown Album',
                    'title':  title.strip()  or Path(track_file).stem,
                    'file':   track_file,
                }
        except Exception:
            pass

        # --- Fallback: mpc search file "FILENAME" ---
        # `mpc search file FILENAME` queries the MPD database and returns
        # full tag lines in the format:  Artist Album Title  (one track per hit)
        # We use the format flag here because `mpc search` DOES support -f.
        try:
            search_result = subprocess.run(
                ['mpc', '-h', self.host, '-p', str(self.port),
                 'search', 'file', track_file,
                 '-f', '%artist%|||%album%|||%title%'],
                capture_output=True,
                text=True,
                timeout=3
            )
            if search_result.returncode == 0 and search_result.stdout.strip():
                # Take the first matching line
                for line in search_result.stdout.strip().split('\n'):
                    if '|||' in line:
                        parts = line.split('|||')
                        if len(parts) >= 3:
                            return {
                                'artist': parts[0].strip() or 'Unknown Artist',
                                'album':  parts[1].strip() or 'Unknown Album',
                                'title':  parts[2].strip() or Path(track_file).stem,
                                'file':   track_file,
                            }
        except Exception:
            pass

        # --- Last resort: filename stem only ---
        return {
            'artist': 'Unknown Artist',
            'album':  'Unknown Album',
            'title':  Path(track_file).stem,
            'file':   track_file,
        }
    
    def get_track_metadata(self, track_file: str) -> dict:
        """Get metadata for a specific track file."""
        # Re-use the cache-aware helper so both call sites stay consistent.
        if not hasattr(self, '_metadata_cache'):
            self._metadata_cache: Dict[str, dict] = {}
        if track_file not in self._metadata_cache:
            self._metadata_cache[track_file] = self._fetch_track_tags(track_file)
        return self._metadata_cache[track_file]
    
    def get_current_track_file(self) -> Optional[str]:
        """Get the file path of currently playing track."""
        status = self.get_status()
        return status.get('track_file')
    
    def play(self) -> bool:
        """Start/resume playback."""
        try:
            subprocess.run(
                ['mpc', '-h', self.host, '-p', str(self.port), 'play'],
                capture_output=True,
                timeout=2
            )
            return True
        except Exception as e:
            print(f"Error playing: {e}", file=sys.stderr)
            return False
    
    def pause(self) -> bool:
        """Pause playback."""
        try:
            subprocess.run(
                ['mpc', '-h', self.host, '-p', str(self.port), 'pause'],
                capture_output=True,
                timeout=2
            )
            return True
        except Exception as e:
            print(f"Error pausing: {e}", file=sys.stderr)
            return False
    
    def next_track(self) -> bool:
        """Skip to next track in queue."""
        try:
            subprocess.run(
                ['mpc', '-h', self.host, '-p', str(self.port), 'next'],
                capture_output=True,
                timeout=2
            )
            return True
        except Exception as e:
            print(f"Error skipping: {e}", file=sys.stderr)
            return False
    
    def clear_queue(self) -> bool:
        """Clear the entire MPD queue."""
        try:
            subprocess.run(
                ['mpc', '-h', self.host, '-p', str(self.port), 'clear'],
                capture_output=True,
                timeout=2
            )
            return True
        except Exception as e:
            print(f"Error clearing queue: {e}", file=sys.stderr)
            return False
    
    def add_track(self, track_file: str) -> bool:
        """
        Add a track to the MPD queue.
        track_file must be relative to MPD's music directory.
        """
        try:
            subprocess.run(
                ['mpc', '-h', self.host, '-p', str(self.port), 'add', track_file],
                capture_output=True,
                timeout=2
            )
            return True
        except Exception as e:
            print(f"Error adding track {track_file}: {e}", file=sys.stderr)
            return False
    
    def get_queue(self) -> List[str]:
        """Get list of tracks currently in MPD queue."""
        try:
            result = subprocess.run(
                ['mpc', '-h', self.host, '-p', str(self.port), 'playlist', '-f', '%file%'],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if result.returncode == 0:
                return [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            return []
        except Exception as e:
            print(f"Error getting queue: {e}", file=sys.stderr)
            return []
    
    def get_queue_length(self) -> int:
        """Get number of tracks in queue."""
        return len(self.get_queue())
    
    def play_track(self, track_file: str) -> bool:
        """Jump to a specific track in the current MPD queue by file path."""
        try:
            # Get playlist with positions
            result = subprocess.run(
                ['mpc', '-h', self.host, '-p', str(self.port), 'playlist', '-f', '%position% %file%'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode != 0:
                return False
            for line in result.stdout.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    pos, fpath = parts
                    if fpath.strip() == track_file:
                        subprocess.run(
                            ['mpc', '-h', self.host, '-p', str(self.port), 'play', pos],
                            capture_output=True,
                            timeout=2
                        )
                        return True
            return False
        except Exception as e:
            print(f"Error playing track {track_file}: {e}", file=sys.stderr)
            return False

    def update_database(self) -> bool:
        """Trigger MPD database update."""
        try:
            subprocess.run(
                ['mpc', '-h', self.host, '-p', str(self.port), 'update'],
                capture_output=True,
                timeout=2
            )
            return True
        except Exception as e:
            print(f"Error updating database: {e}", file=sys.stderr)
            return False
    
    def list_all_tracks(self) -> List[str]:
        """
        Get all tracks in MPD database.
        Returns list of file paths relative to music directory.
        """
        try:
            result = subprocess.run(
                ['mpc', '-h', self.host, '-p', str(self.port), 'listall'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                tracks = []
                for line in result.stdout.strip().split('\n'):
                    line = line.strip()
                    # Filter for music files only
                    if line and any(line.lower().endswith(ext) for ext in 
                                   ['.mp3', '.flac', '.ogg', '.m4a', '.wav', '.opus']):
                        tracks.append(line)
                return tracks
            return []
        except Exception as e:
            print(f"Error listing tracks: {e}", file=sys.stderr)
            return []
