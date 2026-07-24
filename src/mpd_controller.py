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
        # Tags resolved per track, keyed by the `mpc listall` path.  The session
        # history panel asks for the same handful of tracks on every 0.5 s
        # refresh, and the first resolution reads the file with mutagen.
        self._tag_cache: Dict[str, dict] = {}

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
    
    def previous_track(self):
        """
        Go to previous track.

        Currently unbound.  Retained per audit L8, but note that Stage 2 forces
        MPD's `consume` mode on (D2), after which played tracks are gone from
        MPD's queue and this cannot work — a working "previous" will have to
        re-add from the application's own history instead of delegating here.
        """
        try:
            subprocess.run(
                ['mpc', '-h', self.host, '-p', str(self.port), 'prev'],
                capture_output=True,
                timeout=2
            )
        except Exception as e:
            print(f"Previous command failed: {e}", file=sys.stderr)

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
    
    # NOTE: `get_playlist_metadata()` is gone (audit §8, Stage 3 trap 2).  It
    # built its map from `mpc playlist`, and with `consume on` a played track is
    # *removed* from the playlist — so the one existing metadata path covered
    # exactly the tracks the session-history panel does not show, and shelled out
    # an extra `mpc playlist` on every 0.5 s refresh to do it.  Its only caller
    # wanted tags for one track at a time, which is what `fetch_track_tags` is.

    def fetch_track_tags(self, track_file: str) -> dict:
        """
        Fetch artist/album/title for a single track file, cached.

        Strategy (in order of preference):
          1. mutagen  – reads ID3/FLAC/etc. tags directly from the file.
          2. mpc search file – asks MPD for the track info by file path.
          3. Filename stem fallback.

        The cache is what makes this usable from the display loop: the history
        panel asks about the same tracks twice a second, and step 1 is file I/O.
        """
        cached = self._tag_cache.get(track_file)
        if cached is not None:
            return cached
        tags = self._resolve_track_tags(track_file)
        self._tag_cache[track_file] = tags
        return tags

    def _resolve_track_tags(self, track_file: str) -> dict:
        """Uncached resolution — see `fetch_track_tags`."""
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

        Returns True only if MPD actually accepted the track.  This used to
        return True unconditionally (audit H7), which made every caller's
        `if success:` check decorative: a track MPD refused — removed file,
        stale database, path mismatch — was recorded as queued.  Once the queue
        is one track deep (D1) a swallowed failure means playback silently runs
        dry, so the return code is load-bearing, not hygiene.
        """
        try:
            result = subprocess.run(
                ['mpc', '-h', self.host, '-p', str(self.port), 'add', track_file],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode != 0:
                detail = (result.stderr or '').strip() or f"mpc exited {result.returncode}"
                print(f"MPD refused track {track_file}: {detail}", file=sys.stderr)
                return False
            return True
        except Exception as e:
            print(f"Error adding track {track_file}: {e}", file=sys.stderr)
            return False
    
    # ── Playback modes (audit C2/D2) ─────────────────────────────────────────
    #
    # MPD's playback modes are the *user's* state, and the application only gets
    # to borrow them.  `random on` was the live setting of the development
    # machine, and it silently discards every ordering decision the DJ makes:
    # MPD picks an arbitrary queue entry on auto-advance and on `mpc next`.  At
    # queue depth 1 it is worse than mis-ordering — with two entries in the
    # queue, MPD may replay the current track instead of advancing, so the DJ
    # looks stuck rather than merely wrong.
    #
    # So the modes are read, forced, *logged*, and restored on every exit path.
    # Silently clobbering a user's configuration is worse than telling them.

    MODE_NAMES = ('repeat', 'random', 'single', 'consume')

    def get_modes(self) -> Dict[str, str]:
        """
        Read MPD's playback modes as raw strings.

        Strings rather than bools because `single` has three states in modern
        MPD — `off`, `on`, `oneshot` — and restoring a user's `oneshot` as `off`
        would be a silent change of their setting.
        """
        try:
            result = subprocess.run(
                ['mpc', '-h', self.host, '-p', str(self.port), 'status'],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode != 0:
                return {}
            modes = {}
            for name in self.MODE_NAMES:
                match = re.search(rf'{name}:\s*(\S+)', result.stdout)
                if match:
                    modes[name] = match.group(1)
            return modes
        except Exception as e:
            print(f"Error reading MPD modes: {e}", file=sys.stderr)
            return {}

    def set_mode(self, name: str, value: str) -> bool:
        """Set one playback mode.  `value` is passed through to mpc verbatim."""
        if name not in self.MODE_NAMES:
            raise ValueError(f"unknown MPD mode {name!r}")
        try:
            result = subprocess.run(
                ['mpc', '-h', self.host, '-p', str(self.port), name, value],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode != 0:
                detail = (result.stderr or '').strip() or f"mpc exited {result.returncode}"
                print(f"MPD refused {name} {value}: {detail}", file=sys.stderr)
                return False
            return True
        except Exception as e:
            print(f"Error setting {name}={value}: {e}", file=sys.stderr)
            return False

    def delete_position(self, position: int) -> bool:
        """
        Remove one entry from the queue by 1-based position.

        Position 2 is the lookahead while something is playing: verified against
        the live MPD, `mpc del 2` removes it and the current track keeps playing,
        uninterrupted, still at #1.  Deleting a position that does not exist is
        an ordinary miss — mpc exits 1 with "song number does not exist" — so it
        returns False rather than raising; the skip path relies on that when the
        queue happens to hold only the current track.
        """
        try:
            result = subprocess.run(
                ['mpc', '-h', self.host, '-p', str(self.port), 'del', str(position)],
                capture_output=True, text=True, timeout=2
            )
            return result.returncode == 0
        except Exception as e:
            print(f"Error deleting queue position {position}: {e}", file=sys.stderr)
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
