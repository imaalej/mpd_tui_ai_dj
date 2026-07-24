"""
Album Art Renderer - Detects and uses available image protocols
Supports: ueberzugpp, ueberzug (classic), kitty graphics, sixel
"""

import subprocess
import os
import sys
import json
import time
import tempfile
import hashlib
from pathlib import Path
from typing import Optional
from config import config


# ---------------------------------------------------------------------------
# Protocol base class
# ---------------------------------------------------------------------------


class ImageProtocol:
    """
    A layer-drawing overlay protocol (ueberzugpp or classic ueberzug).

    Both work by keeping a long-lived child process open on a stdin pipe and
    feeding it JSON `add`/`remove` commands.  The two used to be near-identical
    copies of the same fifty lines — including, twice, the inspection-C1 pipe-break bug — so
    the shared machinery lives here and each subclass sets only `binary` and
    `launch`.  One copy is one place to fix.
    """

    #: The executable name, checked with `which` and named in messages.
    binary: str = ""
    #: The full argv that starts a silent JSON layer.
    launch: list = []

    def __init__(self):
        self.available = False
        self.process = None
        self.identifier = "adaptive_dj_cover"
        # Throttle the "exited immediately" message so a persistently broken
        # binary does not print twice a second forever — once per failure run.
        self._start_failed_notified = False

    def detect(self) -> bool:
        if not self.binary:
            return False
        try:
            r = subprocess.run(["which", self.binary], capture_output=True, timeout=2)
            if r.returncode != 0:
                return False
            self.available = True
            return self._start_layer()
        except Exception as e:
            print(f"{self.binary} detection error: {e}", file=sys.stderr)
            return False

    def _start_layer(self) -> bool:
        try:
            self.process = subprocess.Popen(
                self.launch,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(0.15)
            if self.process.poll() is not None:
                # Started but died at once.  This may be transient (the binary
                # was momentarily busy mid-resize), so `available` stays True and
                # the next tick retries — we only report it once per run.
                if not self._start_failed_notified:
                    print(
                        f"{self.binary} exited immediately "
                        f"(code {self.process.returncode})",
                        file=sys.stderr,
                    )
                    self._start_failed_notified = True
                self.process = None
                return False
            self._start_failed_notified = False
            self.available = True
            return True
        except FileNotFoundError:
            # The binary is genuinely gone — permanent, so stop claiming art
            # works.  `available` is the signal the renderer reads to give up.
            self.available = False
            self.process = None
            return False
        except Exception as e:
            print(f"Failed to start {self.binary}: {e}", file=sys.stderr)
            self.process = None
            return False

    def is_alive(self) -> bool:
        """True only while a live child is holding the layer open."""
        return self.process is not None and self.process.poll() is None

    def _ensure_process(self) -> bool:
        """(Re)start the layer unless a live child already exists."""
        if self.is_alive():
            return True
        return self._start_layer()

    def render(self, image_path: Path, x: int, y: int, width: int, height: int) -> bool:
        """
        Draw `image_path` into the layer.  Returns True if it was written to a
        live child, False if there is nothing on screen after this call.

        The one non-obvious part is the pipe-break recovery (inspection C1).  The old
        code caught `BrokenPipeError`, set `self.process = None`, and returned —
        and the *next* render hit `if not self.process: return` before it could
        reach the respawn branch, so a single broken write disabled album art for
        the rest of the session while `is_available()` still reported True.  A
        resize that made ueberzugpp choke on one command was enough to trigger it,
        which is the mechanism behind inspection C2's "finagle until it comes back."

        So a broken write now drops the dead child and retries once against a
        fresh layer, and this same call paints.
        """
        if not image_path.exists():
            return False
        if not self._ensure_process():
            return False
        if self._send_add(image_path, x, y, width, height):
            return True
        # The write broke the pipe.  Drop the dead child, respawn, retry once.
        self.process = None
        if not self._ensure_process():
            return False
        return self._send_add(image_path, x, y, width, height)

    def _send_add(self, image_path: Path, x: int, y: int, width: int, height: int) -> bool:
        try:
            cmd = {
                "action": "add",
                "identifier": self.identifier,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "scaler": "fit_contain",
                "path": str(image_path.absolute()),
            }
            self.process.stdin.write((json.dumps(cmd) + "\n").encode())
            self.process.stdin.flush()
            return True
        except BrokenPipeError:
            return False
        except Exception as e:
            print(f"{self.binary} render error: {e}", file=sys.stderr)
            return False

    def clear(self):
        if not self.is_alive():
            return
        try:
            cmd = {"action": "remove", "identifier": self.identifier}
            self.process.stdin.write((json.dumps(cmd) + "\n").encode())
            self.process.stdin.flush()
        except Exception:
            pass

    def shutdown(self):
        """
        End the child process and reap it.

        Separate from `clear()`, which only removes the image: the overlay
        protocols run a child process, and cleanup used to rely on `__del__`
        firing at interpreter exit.  That is not guaranteed, and on SIGTERM it
        did not happen at all, so the child outlived the DJ (audit L9).
        """
        _terminate(self.process)
        self.process = None

    def __del__(self):
        self.shutdown()


# ---------------------------------------------------------------------------
# ueberzugpp  (modern C++ rewrite — JSON stdin)
# ---------------------------------------------------------------------------


class UeberzugppProtocol(ImageProtocol):
    """ueberzugpp — the modern C++ rewrite, JSON on stdin."""

    binary = "ueberzugpp"
    launch = ["ueberzugpp", "layer", "--silent"]


# ---------------------------------------------------------------------------
# Classic ueberzug  (Python package, X11/Wayland)
# Must be launched with --parser json so it accepts the same JSON format.
# The original code omitted --parser json and also forgot 'scaler', causing
# silent failures on every render call.
# ---------------------------------------------------------------------------


class UeberzugProtocol(ImageProtocol):
    """Classic ueberzug — the Python package; needs `--parser json`."""

    binary = "ueberzug"
    launch = ["ueberzug", "layer", "--silent", "--parser", "json"]


# ---------------------------------------------------------------------------
# Kitty and sixel are gone (audit L2)
# ---------------------------------------------------------------------------
#
# `KittyProtocol` and `SixelProtocol` used to live here.  Both wrote escape
# sequences straight to `sys.__stdout__` while urwid owned the terminal, so
# urwid's next full redraw — every 0.5 s — painted over them.  They were not
# "mostly working"; they were a picture that appeared and was erased twice a
# second, and only the detection order (ueberzug first) kept that out of sight.
#
# They are deleted rather than marked, per D7: a branch that cannot work is not
# a feature with a caveat.  `_warn_about_unsupported_terminal()` below tells a
# kitty or sixel user why there is no art and what would give them some, which
# is the part of those 100 lines that was ever worth anything.
#
# Restoring them would need urwid to stop owning the screen for the region the
# image occupies, which is a different design, not a repair.


def _terminate(process):
    """
    End a child process and reap it.

    Cleanup used to rely on `__del__` firing at interpreter exit, which is not
    guaranteed — and on SIGTERM, before H3, `_shutdown()` was unreachable
    entirely, so the ueberzugpp child outlived the DJ (audit L9).  The two share
    that root cause, which is why the shutdown path now calls this explicitly.
    """
    if process is None:
        return
    try:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=1)
        except Exception:
            process.kill()
            process.wait(timeout=1)
    except Exception:
        pass


def _warn_about_unsupported_terminal():
    """Say why there is no album art here, when the terminal looked promising."""
    term = os.environ.get("TERM", "")
    kitty = term == "xterm-kitty" or bool(os.environ.get("KITTY_WINDOW_ID", ""))
    sixel = any(x in term for x in ("mlterm", "yaft", "sixel"))
    if not (kitty or sixel):
        return
    which = "kitty's graphics protocol" if kitty else "sixel"
    print(
        f"Album art: this terminal supports {which}, but that path is not "
        "implemented — the images fought urwid for the screen and lost every "
        "0.5 s. Install `ueberzugpp` for working album art (audit L2).",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Embedded art extraction
# ---------------------------------------------------------------------------

_COVER_CACHE_DIR: Optional[Path] = None


def _get_cache_dir() -> Path:
    global _COVER_CACHE_DIR
    if _COVER_CACHE_DIR is None:
        import atexit

        _COVER_CACHE_DIR = Path(tempfile.mkdtemp(prefix="adaptive_dj_covers_"))
        # `atexit` is the backstop, not the whole story: it does not run on a
        # default SIGTERM, which is the same failure class the shutdown refactor
        # fixed for the ueberzugpp child (inspection C4 / audit L9).  So the signal and
        # `_shutdown()` paths call `cleanup_cover_cache()` too, via the renderer.
        atexit.register(cleanup_cover_cache)
    return _COVER_CACHE_DIR


def cleanup_cover_cache():
    """
    Remove the extracted-cover cache dir.  Idempotent, and safe on the signal
    path — it swallows everything and clears the handle so a second call (atexit
    after the signal path, say) is a no-op (inspection C4).
    """
    global _COVER_CACHE_DIR
    if _COVER_CACHE_DIR is None:
        return
    import shutil

    shutil.rmtree(str(_COVER_CACHE_DIR), ignore_errors=True)
    _COVER_CACHE_DIR = None


def _atomic_write_bytes(path: Path, data: bytes):
    """
    Write `data` to `path` via a temp file and an atomic rename (inspection C5).

    A plain `write_bytes` truncated by a crash or a full disk leaves a partial
    file at `path`, which the `cached.exists()` fast path then serves forever.
    `os.replace` is atomic within the one directory, and the `.tmp` name never
    matches the extensions the fast path checks — so a leftover temp from a hard
    kill is inert rather than served.
    """
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _extract_embedded_art(audio_file: Path) -> Optional[Path]:
    """
    Extract embedded cover art from an audio file using mutagen.
    Supports: MP3/ID3 (APIC), FLAC (PICTURE), MP4/M4A (covr), OGG/OPUS.
    Caches the extracted image to a temp file keyed by file path hash.
    """
    try:
        import mutagen
    except ImportError:
        return None  # mutagen not installed; skip silently

    if not audio_file.exists():
        return None

    cache_key = hashlib.md5(str(audio_file).encode()).hexdigest()
    cache_dir = _get_cache_dir()

    # Return cached file if it already exists
    for ext in (".jpg", ".png", ".webp"):
        cached = cache_dir / f"{cache_key}{ext}"
        if cached.exists():
            return cached

    try:
        audio = mutagen.File(str(audio_file))
        if audio is None:
            return None

        img_data: Optional[bytes] = None
        mime = "image/jpeg"

        # --- ID3 tags (MP3, AIFF, WAV with ID3) ---
        if hasattr(audio, "tags") and audio.tags is not None:
            tags = audio.tags
            for key in list(tags.keys()):
                if key.startswith("APIC"):
                    apic = tags[key]
                    img_data = apic.data
                    mime = getattr(apic, "mime", "image/jpeg")
                    break

        # --- FLAC picture blocks ---
        if img_data is None and hasattr(audio, "pictures") and audio.pictures:
            pics = audio.pictures
            front = next((p for p in pics if p.type == 3), None) or pics[0]
            img_data = front.data
            mime = getattr(front, "mime", "image/jpeg")

        # --- MP4/M4A covr atom ---
        if img_data is None:
            try:
                covr = audio.get("covr") or audio.get("\xa9cov")
                if covr:
                    img_data = bytes(covr[0])
                    mime = "image/jpeg"
            except Exception:
                pass

        # --- OGG/OPUS: base64-encoded METADATA_BLOCK_PICTURE in comment ---
        if img_data is None:
            try:
                import base64
                from mutagen.flac import Picture

                for val in audio.get("metadata_block_picture") or []:
                    pic = Picture(base64.b64decode(val))
                    img_data = pic.data
                    mime = getattr(pic, "mime", "image/jpeg")
                    break
            except Exception:
                pass

        if img_data is None:
            return None

        ext = ".png" if "png" in mime else (".webp" if "webp" in mime else ".jpg")
        out_path = cache_dir / f"{cache_key}{ext}"
        _atomic_write_bytes(out_path, img_data)  # temp-then-rename (inspection C5)
        return out_path

    except Exception as e:
        print(
            f"Embedded art extraction failed ({audio_file.name}): {e}", file=sys.stderr
        )
        return None


# ---------------------------------------------------------------------------
# AlbumArtRenderer
# ---------------------------------------------------------------------------


class AlbumArtRenderer:
    """
    Manages album art rendering across different protocols.
    Auto-detects available protocol and gracefully degrades.
    """

    def __init__(self):
        self.protocol: Optional[ImageProtocol] = None
        self.available = False
        self.current_image: Optional[Path] = None
        self._last_track_file: Optional[str] = None
        self._last_art_path: Optional[Path] = None
        # Render key: (path_str, x, y, w, h).  None = never rendered / dirty.
        # We skip re-sending to the protocol only when this matches exactly.
        # Call force_redraw() to mark dirty without sending a "remove" command
        # (which would cause a visible blank frame = flicker).
        self._render_key: Optional[tuple] = None
        self._detect_protocol()

    def _detect_protocol(self):
        """Detect and initialise best available protocol.

        Order: ueberzugpp → ueberzug.  Those are the two that work: they draw
        into a separate X11/Wayland surface, so urwid's redraw does not erase
        them.  The kitty and sixel branches were deleted (audit L2) — see the
        note where they used to be.
        """
        protocols = [
            ("ueberzugpp", UeberzugppProtocol()),
            ("ueberzug", UeberzugProtocol()),
        ]
        for name, protocol in protocols:
            if protocol.detect():
                self.protocol = protocol
                self.available = True
                print(f"Album art: {name} protocol active", file=sys.stderr)
                return
            protocol.shutdown()
        print("Album art: no supported protocol found (disabled)", file=sys.stderr)
        _warn_about_unsupported_terminal()

    def is_available(self) -> bool:
        return self.available

    def render(
        self,
        image_path: Optional[Path],
        x: int = 0,
        y: int = 0,
        width: int = 20,
        height: int = 20,
    ):
        if not self.available or not self.protocol:
            return
        if image_path is None or not image_path.exists():
            self.clear()
            return
        # Build a key from every parameter that determines what ueberzug shows.
        # Skip the protocol call entirely when the key matches — this prevents
        # ueberzug from doing its internal remove+redraw cycle, eliminating flicker.
        # When something genuinely changes (new track, resize, window move) the
        # key will differ and we re-render exactly once.
        #
        # The liveness check is the inspection-C2 half of inspection-C1: the skip is only safe while a
        # child is actually holding the image up.  If it died — a pipe break, a
        # crash, a resize command ueberzugpp choked on — the on-screen image is
        # gone even though the key is unchanged, so we must fall through and let
        # the protocol respawn rather than believe a dead frame is still there.
        key = (str(image_path), x, y, width, height)
        if key == self._render_key and self.protocol.is_alive():
            return  # nothing changed — stable image, no flicker
        try:
            drawn = self.protocol.render(image_path, x, y, width, height)
        except Exception:
            drawn = False
        if drawn:
            self.current_image = image_path
            self._render_key = key
        else:
            # Nothing is on screen after this call.  Forget the key so the next
            # tick re-attempts rather than assuming the frame is up (that stale
            # assumption is exactly what wedged inspection-C1).
            self.current_image = None
            self._render_key = None
            # `protocol.available` goes False only when the binary is confirmed
            # gone (FileNotFoundError), not on a transient spawn failure — so a
            # brief mid-resize hiccup keeps retrying, and only a real absence
            # makes `is_available()` honestly report there is no art.
            if not self.protocol.available:
                print(
                    "Album art: renderer disabled — the overlay process is gone "
                    "and could not be restarted",
                    file=sys.stderr,
                )
                self.available = False

    def clear(self):
        if self.available and self.protocol:
            try:
                self.protocol.clear()
            except Exception:
                pass
        self.current_image = None
        self._render_key = None

    def shutdown(self):
        """
        Clear the image and end the protocol's child process (audit L9).

        `_shutdown()` called `clear()` and stopped there, which removes the
        picture but leaves `ueberzugpp` running with its stdin held open —
        cleanup was left to `__del__` at interpreter exit, which is not
        guaranteed to fire and did not fire at all on SIGTERM, because
        `_shutdown()` was itself unreachable there until H3.  The two findings
        share that root cause, so the signal path calls this as well.

        Safe to call more than once, and after it the renderer reports itself
        unavailable rather than writing to a pipe that is gone.
        """
        self.clear()
        if self.protocol is not None:
            try:
                self.protocol.shutdown()
            except Exception:
                pass
        # The extracted-cover cache is torn down here too, so a SIGTERM stop —
        # which never reaches `atexit` — does not leak the temp dir (inspection C4).  This
        # runs on the signal path, so it must not raise.
        try:
            cleanup_cover_cache()
        except Exception:
            pass
        self.available = False

    def force_redraw(self):
        """
        Mark the renderer dirty so the next render() call re-sends to the
        protocol, without issuing a "remove" command first.

        Use this when the terminal redraws itself (SIGWINCH — resize or
        window move) so the image reappears without any visible blank frame.
        """
        self._render_key = None

    def find_album_art(self, track_file: str) -> Optional[Path]:
        """
        Find album art for a track.

        Strategy:
          1. Scan the track's directory for image files (case-insensitive,
             preferred names first).
          2. Extract embedded art from the audio file's tags via mutagen.

        Results cached per track_file so repeated calls during the 0.5s
        display refresh loop cost nothing after the first lookup.
        """
        if track_file == self._last_track_file:
            return self._last_art_path

        self._last_track_file = track_file
        result = self._find_art(track_file)
        self._last_art_path = result
        return result

    def _find_art(self, track_file: str) -> Optional[Path]:
        try:
            track_path = Path(config.mpd_music_directory) / track_file
            track_dir = track_path.parent

            # --- 1. Image file in the same directory ---
            if track_dir.is_dir():
                preferred_stems = [
                    "cover",
                    "folder",
                    "front",
                    "album",
                    "albumart",
                    "albumartsmall",
                    "artwork",
                    "art",
                    "thumb",
                    "thumbnail",
                    "jacket",
                ]
                image_exts = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}

                # Case-insensitive map of everything in the directory
                dir_files: dict = {}
                try:
                    for p in track_dir.iterdir():
                        if p.suffix.lower() in image_exts:
                            dir_files[p.name.lower()] = p
                except PermissionError:
                    pass

                for stem in preferred_stems:
                    for ext in (".jpg", ".jpeg", ".png", ".webp"):
                        hit = dir_files.get(stem + ext)
                        if hit:
                            return hit

                # Any image will do
                if dir_files:
                    return next(iter(dir_files.values()))

            # --- 2. Embedded art from audio tags ---
            return _extract_embedded_art(track_path)

        except Exception as e:
            print(f"find_album_art error: {e}", file=sys.stderr)
            return None


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_renderer: Optional[AlbumArtRenderer] = None


def get_album_art_renderer() -> AlbumArtRenderer:
    global _renderer
    if _renderer is None:
        _renderer = AlbumArtRenderer()
    return _renderer
