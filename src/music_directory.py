"""
Where MPD's music actually lives.

Audit M3: `config.mpd_music_directory` used to default to `/var/lib/mpd/music`,
which on this machine is a symlink to the real `/mnt/storage/music`.  It worked
by accident, was never validated, and a wrong value silently breaks album art,
tag reads and — worst — the whole embedding-generation run, which used to walk
this directory to enumerate tracks.

So: ask MPD's own configuration rather than guessing, and prove the answer by
checking that real track paths exist beneath it before anything expensive
happens.

The proof is an *existence* check on the joined path, not a strict containment
proof (audit E3): it relies on `mpc listall` reporting paths **relative** to the
music directory, which it does.  An absolute key, or one containing `..`, would
sidestep the join and is deliberately not defended against here — MPD does not
produce such keys, and the caller passes exactly what `mpc listall` returned.

Nothing here shells out to `mpc`.  Detection reads config files only, so it is
cheap enough to run at import time; validation takes the track list from the
caller (which owns the MPD connection).
"""

import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

# Searched in order.  The user's own config wins over the system one, which is
# how MPD itself resolves it.
CONFIG_CANDIDATES = (
    '$XDG_CONFIG_HOME/mpd/mpd.conf',
    '~/.config/mpd/mpd.conf',
    '~/.mpdconf',
    '~/.mpd/mpd.conf',
    '/etc/mpd.conf',
    '/etc/mpd/mpd.conf',
)

# `music_directory "/path"` — quotes optional, leading whitespace allowed, a
# trailing `# comment` ignored.  Two value forms (audit E3): a *quoted* value may
# contain anything but a quote — including a `#`, which MPD treats as literal
# inside quotes, so `"/music/#rare"` must not be truncated — while an *unquoted*
# value runs to the first whitespace or `#`.
_DIRECTIVE = re.compile(
    r'^\s*music_directory\s+(?:"([^"]+)"|([^\s#]+))\s*(?:#.*)?$')

# The legacy default.  Kept only as a last resort so the app has *something* to
# put in front of the user, never as a silent assumption — every code path that
# uses it reports where it came from.
LEGACY_DEFAULT = '/var/lib/mpd/music'


def parse_music_directory(config_text: str) -> Optional[str]:
    """Return the `music_directory` value from an MPD config, or None."""
    for line in config_text.splitlines():
        if line.lstrip().startswith('#'):
            continue
        match = _DIRECTIVE.match(line)
        if match:
            value = (match.group(1) or match.group(2)).strip()
            if value:
                return value
    return None


def _expand(raw: str) -> Path:
    """Expand our own `MPD_MUSIC_DIR` env var: both `~` and `$VAR`, as a shell
    would — this is a convenience knob we define, so shell-style expansion is
    what a user setting it expects."""
    return Path(os.path.expandvars(os.path.expanduser(raw)))


def _expand_config_value(raw: str) -> Path:
    """
    Expand a value read from MPD's own config (audit E3).

    MPD expands a leading `~` but takes `$VAR` **literally**, so this expands the
    user home and stops there — `os.path.expandvars` would rewrite a `$` MPD
    would have kept, turning a valid (if unusual) path into a wrong one.
    """
    return Path(os.path.expanduser(raw))


def config_file_candidates() -> List[Path]:
    """The MPD config paths that will be searched, in order, expanded."""
    candidates = []
    for raw in CONFIG_CANDIDATES:
        if raw.startswith('$XDG_CONFIG_HOME') and not os.getenv('XDG_CONFIG_HOME'):
            continue
        candidates.append(_expand(raw))
    return candidates


def detect_music_directory() -> Tuple[Optional[Path], str]:
    """
    Find MPD's music directory.

    Returns (path, source) where `source` is a short human-readable account of
    where the value came from — printed at startup so a wrong directory is
    diagnosable rather than mysterious.  `path` is None when nothing could be
    determined; the caller decides what to do about it.
    """
    env = os.getenv('MPD_MUSIC_DIR')
    if env:
        return _expand(env), 'MPD_MUSIC_DIR'

    for candidate in config_file_candidates():
        try:
            if not candidate.is_file():
                continue
            value = parse_music_directory(candidate.read_text(errors='replace'))
        except OSError:
            continue
        if value:
            return _expand_config_value(value), str(candidate)

    return None, 'not found'


def sample_tracks(tracks: Sequence[str], count: int = 5) -> List[str]:
    """
    Spread `count` probes across the track list.

    One probe is not enough: any single file can be missing from disk while the
    directory is perfectly correct.  Evenly spaced samples also catch the case
    where only part of the library is mounted.
    """
    if not tracks:
        return []
    if len(tracks) <= count:
        return list(tracks)
    step = len(tracks) / count
    return [tracks[min(len(tracks) - 1, int(i * step))] for i in range(count)]


def validate_music_directory(
    music_dir: Optional[Path],
    tracks: Iterable[str],
    count: int = 5,
) -> Tuple[bool, str]:
    """
    Prove the directory by checking that real MPD track paths exist beneath it.

    `tracks` are paths as MPD reports them (`mpc listall`), i.e. relative to the
    music directory, so `music_dir / t` is the file on disk and its existence is
    the proof.  This is an existence check, not a strict containment proof — an
    absolute or `..`-escaping key would join to somewhere else, but MPD does not
    emit those (see the module docstring).  Returns (ok, message); the message is
    written for a user, not a developer.
    """
    if music_dir is None:
        return False, (
            "MPD's music directory could not be determined.\n"
            "  Set it explicitly:  MPD_MUSIC_DIR=/path/to/music\n"
            f"  Config files searched: {', '.join(str(p) for p in config_file_candidates())}"
        )

    if not music_dir.is_dir():
        return False, f"Music directory does not exist: {music_dir}"

    probes = sample_tracks(list(tracks), count)
    if not probes:
        return False, (
            "MPD's database is empty, so the music directory cannot be verified.\n"
            "  Run `mpc update` once MPD points at your music."
        )

    missing = [t for t in probes if not (music_dir / t).exists()]
    if missing:
        return False, (
            f"Music directory {music_dir} does not contain MPD's tracks — "
            f"{len(missing)} of {len(probes)} probes missing, e.g.\n"
            f"  {music_dir / missing[0]}\n"
            "  This is the wrong directory, or the library is not mounted.\n"
            "  Set it explicitly:  MPD_MUSIC_DIR=/path/to/music"
        )

    return True, f"Music directory verified: {music_dir} ({len(probes)} probes resolved)"


def _main() -> int:
    """
    `python3 music_directory.py` — print the detected directory, or fail.

    start.sh calls this so it can prompt for a path when detection fails, and so
    the same detection logic serves the shell and the application (M3/M4: one
    source of truth, not two).
    """
    path, source = detect_music_directory()
    if path is None:
        print("not found", file=sys.stderr)
        return 1
    print(path)
    print(f"source: {source}", file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(_main())
