"""
The vibe cloud — Phase 4's centrepiece, a rotating 3-D point cloud of the library
in mood space, drawn in the terminal with Braille + 256 colour.

Two halves live here, split so the arithmetic is testable without a tty:

  * **Pure geometry** (`rotation_matrix`, `point_rgb`, `compute_frame`, `hit_test`)
    — turns `(N, 3)` mood coordinates into a grid of Braille cells with a colour
    and a source-point index per cell.  No urwid, no I/O.  This is the part the
    unit tests drive, the way `explore_mood_axes.py` drives `mood_axes.select`.
  * **The urwid box widget** (`VibeCloudWidget`) — wraps the geometry, owns the
    camera and comet state, renders to a canvas and answers mouse hits.  Guarded
    on `URWID_AVAILABLE` exactly like `tui.py`, so importing this module in the
    fallback text mode (which has no cloud) does not require urwid.

Where the numbers come from (all derived, none invented — §1):

  * **Positions** are `MoodAxes.coordinates(centred_library)` — the same centred
    CLAP space the app scores in, projected onto three descriptor-pole axes and
    z-scored against this library.  Because they are z-scored the cloud is
    already centred on the origin with ~unit spread per axis, so the camera needs
    no separate rescale (Phase 3 handoff).
  * **Colour** is each point's own three coordinates as HSV, so a point's hue and
    its position cannot disagree (G3).  The axis→channel mapping is fixed here,
    not per-point-arbitrary.
  * **The camera orbits ambiently**; **the comet moves only on real session
    events** (a full listen or a skip moves the session vector — never the
    ambient rotation).  Keeping those two motions separate is the §4 invariant
    "never blur ambient and data motion".

Rendering is Braille (2×4 dots per character cell) so one 90×30 panel is a
360×120 dot canvas — coarse next to the browser preview in `archive/`, but the
honest terminal target (see `archive/NOTES.md`).
"""

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import urwid

    URWID_AVAILABLE = True
except ImportError:                                    # pragma: no cover
    URWID_AVAILABLE = False


# ── Braille: 2×4 dots per glyph, base U+2800.  Bit per (col, row). ────────────
BRAILLE_BITS: Dict[Tuple[int, int], int] = {
    (0, 0): 0x01, (0, 1): 0x02, (0, 2): 0x04, (0, 3): 0x40,
    (1, 0): 0x08, (1, 1): 0x10, (1, 2): 0x20, (1, 3): 0x80,
}
BRAILLE_BASE = 0x2800
# The same eight bits as a flat lookup indexed by `(dx%2)*4 + (dy%4)`, so the
# vectorised rasteriser can turn a whole array of dot coordinates into glyph bits
# in one gather.  Derived from BRAILLE_BITS so the two cannot drift.
BRAILLE_LUT = np.array([BRAILLE_BITS[(i // 4, i % 4)] for i in range(8)],
                       dtype=np.uint8)

# Every point and every marker is a single Braille dot.  Fatter markers
# (squares/discs/rings) are billboarded — always facing the camera — which
# flattens the cloud into a swimming plane; a 1-dot mark has no facing.  So depth
# is carried by **shading** (SHADE_MIN) and markers are distinguished purely by a
# bright contrasting colour on top, never by size or an outline.  Clicking is
# handled by the nearest-within-radius hit search, not by making the dot bigger.
SHADE_MIN = 0.4

# How far (in character cells) a click searches for the nearest point.  Bigger
# than the old default because a single-dot point is a small target.
CLICK_RADIUS = 3

# The z-scored coordinates sit within roughly ±3.5σ, so a half-extent of 3.0
# fills the panel with a little headroom before the outliers clip.  A shaping
# constant (it does not assert a fact), tunable by eye.
BASE_HALF_EXTENT = 3.0

# Camera.  Everything here is in **real time** (radians per *second*), not per
# frame, so the animation framerate can change without changing the feel: the
# widget's `advance(dt)` multiplies by the frame's own `dt`.  Orbit speed is set
# live by the on-panel slider from 0 (static) to ORBIT_SPEED_MAX; the default is
# deliberately slow — a barely-there drift — so the cloud is calm until dialled up.
DEFAULT_TILT = 0.5
ORBIT_SPEED_MAX = 1.5                 # rad/s at the top of the slider
ORBIT_SPEED_DEFAULT = 0.15            # rad/s: ~40 s per revolution
TILT_MIN, TILT_MAX = -1.4, 1.4
ZOOM_MIN, ZOOM_MAX = 0.5, 3.5

# The camera **eases** toward where the mouse dragged it rather than snapping, so
# motion is smooth at the framerate regardless of how coarsely the mouse reports
# (a vertical drag reports in the terminal's taller rows, so without this a tilt
# lurches while a left/right spin glides).  `CAMERA_TAU` is the ease time constant
# in seconds — smaller is snappier.  `POSE_EPS*` are how far the camera must move
# before a repaint is worth it, so a slow ambient spin repaints a few times a
# second while a drag repaints every frame (cheap when idle, buttery when moving).
CAMERA_TAU = 0.07
POSE_EPS_ANGLE = 0.004
POSE_EPS_ZOOM = 0.003

# Comet trail: how many past session positions to keep, and how far the session
# vector must move (in z-score space) before a new bead is laid down — so the
# trail records genuine vibe motion, not floating-point jitter.
COMET_MAX = 48
COMET_MOVE_EPSILON = 0.05

# Bright, full-brightness contrasting colours for the markers — each drawn as a
# single dot on top of the (depth-shaded) library so it stands out.
COL_COMET_HEAD = 231          # white — the session's current position
COL_COMET_TRAIL = (223, 216, 209, 202)   # warm amber → orange, tail → head
COL_CURRENT = 51              # bright cyan — the playing track
COL_SELECTED = 201            # bright magenta — a clicked point


# ── Colour ────────────────────────────────────────────────────────────────────


def _hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorised HSV→RGB for arrays of hue/sat/val in [0, 1] → `(N, 3)` uint8."""
    h = np.asarray(h, dtype=np.float64)
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.clip(np.stack([r, g, b], axis=1) * 255, 0, 255).astype(np.uint8)


def point_rgb(coords: np.ndarray) -> np.ndarray:
    """
    `(N, 3)` mood coordinates → `(N, 3)` uint8 RGB (fully vectorised).

    Each of the three axes drives one HSV channel, so a point's colour is a pure
    function of where it sits (G3: "position and hue cannot disagree").  The
    stored triad on this library is Tone · Saturation · Organic, and the mapping
    is chosen to read naturally:

        Hue        ← axis 0  (Tone: melancholic→blue, uplifting→warm)
        Saturation ← axis 1  (Saturation: sparse→washed out, dense→vivid)
        Value      ← axis 2  (Organic: acoustic→dim, electronic→bright)

    A library whose triad differs still gets a deterministic, distinct colour per
    point; only the perceptual story changes, which is fine — the invariant is
    determinism, not that axis 0 is always "Tone".
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim == 1:
        coords = coords[None, :]

    def unit(a: np.ndarray, lo: float = -2.5, hi: float = 2.5) -> np.ndarray:
        return np.clip((a - lo) / (hi - lo), 0.0, 1.0)

    hue = np.mod(0.62 - 0.55 * unit(coords[:, 0]), 1.0)       # blue → red
    sat = 0.35 + 0.60 * unit(coords[:, 1])
    val = 0.45 + 0.55 * unit(coords[:, 2])
    return _hsv_to_rgb(hue, sat, val)


def _rgb_to_256_array(rgb: np.ndarray) -> np.ndarray:
    """`(N, 3)` RGB → `(N,)` xterm-256 codes (the 6×6×6 cube), vectorised."""
    q = np.round(np.asarray(rgb, dtype=np.float64) / 255 * 5).astype(np.int32)
    return (16 + 36 * q[:, 0] + 6 * q[:, 1] + q[:, 2]).astype(np.int32)


# ── Geometry ──────────────────────────────────────────────────────────────────


def rotation_matrix(azimuth: float, tilt: float) -> np.ndarray:
    """Y-rotation (orbit) then X-rotation (tilt), as a single 3×3."""
    cy, sy = math.cos(azimuth), math.sin(azimuth)
    cx, sx = math.cos(tilt), math.sin(tilt)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return rx @ ry


class Frame:
    """
    The result of projecting the cloud for one camera pose, as dense grids.

    All three are `(rows, cols)` arrays: `bits` is the OR of the Braille dot bits
    lit in each character cell; `color` is the 256-colour code of the front-most
    thing there (−1 = empty); `hit` is the index of the front-most *library* point
    whose dot falls there (−1 = none / a mark), so a click resolves to a track,
    never to the comet.  Dense rather than dict because the rasteriser fills them
    with vectorised numpy, and the markup builder and hit-test read them directly.
    """

    __slots__ = ("bits", "color", "hit", "cols", "rows")

    def __init__(self, cols: int, rows: int):
        self.cols = cols
        self.rows = rows
        self.bits = np.zeros((rows, cols), dtype=np.uint8)
        self.color = np.full((rows, cols), -1, dtype=np.int32)
        self.hit = np.full((rows, cols), -1, dtype=np.int32)


def compute_frame(coords: np.ndarray,
                  rgb: np.ndarray,
                  cols: int,
                  rows: int,
                  azimuth: float,
                  tilt: float,
                  zoom: float = 1.0,
                  comet: Optional[np.ndarray] = None,
                  current_idx: Optional[int] = None,
                  selected_idx: Optional[int] = None,
                  shade: bool = True) -> Frame:
    """
    Project `coords` (N, 3) into a Braille grid of `cols`×`rows` character cells.

    Everything — every library point and every marker — is a **single dot**.
    Library points are **depth-shaded**: the `(N, 3)` colour is dimmed toward
    `SHADE_MIN` by how far the point sits from the camera this frame, so a turn of
    the cloud reads as depth without billboarded markers.  The rasteriser is
    vectorised: all dots become flat coordinate arrays, then a stable sort by
    depth + scatter gives each cell the colour and hit-index of its front-most dot
    (painter's algorithm).  The current track and the comet are *marks* — a bright
    contrasting colour with a large depth bump so they sit on top, and index −1 so
    they never claim a cell's hit.  The selected point is the point itself,
    recoloured to `COL_SELECTED` and lifted on top, keeping its own index so it
    stays clickable.
    """
    frame = Frame(cols, rows)
    if cols <= 0 or rows <= 0 or coords is None or len(coords) == 0:
        return frame

    dot_w, dot_h = 2 * cols, 4 * rows
    scale = (min(dot_w, dot_h) / 2.0) / BASE_HALF_EXTENT * zoom
    cx0, cy0 = dot_w / 2.0, dot_h / 2.0

    R = rotation_matrix(azimuth, tilt)          # once, reused for cloud + comet

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    depths: List[np.ndarray] = []
    colours: List[np.ndarray] = []
    idxs: List[np.ndarray] = []

    def splat(projected: np.ndarray, colour, idx, bump: float) -> None:
        """Add one dot per projected point (colour/idx scalar or per-point)."""
        if len(projected) == 0:
            return
        m = len(projected)
        xs.append(np.round(projected[:, 0] * scale + cx0).astype(np.int32))
        ys.append(np.round(cy0 - projected[:, 1] * scale).astype(np.int32))
        depths.append(projected[:, 2] + bump)
        colours.append(np.asarray(colour, dtype=np.int32) if np.ndim(colour)
                       else np.full(m, int(colour), dtype=np.int32))
        idxs.append(np.asarray(idx, dtype=np.int32) if np.ndim(idx)
                    else np.full(m, int(idx), dtype=np.int32))

    projected = coords @ R.T

    # Depth-shade the point colours by this frame's front-to-back range.
    z = projected[:, 2]
    if shade and len(z) and float(z.max() - z.min()) > 1e-9:
        t = (z - z.min()) / (z.max() - z.min())
        factor = SHADE_MIN + (1.0 - SHADE_MIN) * t
        color256 = _rgb_to_256_array(
            np.clip(np.asarray(rgb, dtype=np.float64) * factor[:, None], 0, 255))
    else:
        color256 = _rgb_to_256_array(rgb)

    splat(projected, color256, np.arange(len(coords), dtype=np.int32), 0.0)

    # The current track: a single bright-cyan dot on top (index −1: a mark).
    if current_idx is not None and 0 <= current_idx < len(coords):
        splat(projected[current_idx:current_idx + 1], COL_CURRENT, -1, 8.0)
    # The selected point: the point itself recoloured bright magenta and lifted on
    # top — no ring — keeping its own index so it stays clickable.
    if selected_idx is not None and 0 <= selected_idx < len(coords):
        splat(projected[selected_idx:selected_idx + 1], COL_SELECTED,
              selected_idx, 9.0)

    if comet is not None and len(comet) > 0:
        cproj = np.asarray(comet) @ R.T
        n = len(cproj)
        # Single dots along the trajectory: warm trail, white head, on top.
        frac = np.arange(n) / max(1, n - 1)
        trail = np.array(COL_COMET_TRAIL, dtype=np.int32)
        ccol = trail[np.minimum(len(trail) - 1, (frac * len(trail)).astype(int))]
        ccol[-1] = COL_COMET_HEAD
        splat(cproj, ccol, -1, 10.0)

    # ── Rasterise everything at once ──────────────────────────────────────────
    allx = np.concatenate(xs)
    ally = np.concatenate(ys)
    alld = np.concatenate(depths)
    allc = np.concatenate(colours)
    alli = np.concatenate(idxs)

    inb = (allx >= 0) & (allx < dot_w) & (ally >= 0) & (ally < dot_h)
    allx, ally, alld, allc, alli = allx[inb], ally[inb], alld[inb], allc[inb], alli[inb]
    if len(allx) == 0:
        return frame

    cell = (ally // 4) * cols + (allx // 2)
    bit = BRAILLE_LUT[(allx % 2) * 4 + (ally % 4)]
    np.bitwise_or.at(frame.bits.reshape(-1), cell, bit)

    # Colour: every dot, front-most (greatest depth) wins — so marks on top show.
    order = np.argsort(alld, kind="stable")     # far → near; near overwrites
    frame.color.reshape(-1)[cell[order]] = allc[order]

    # Hit: **library dots only** (idx ≥ 0).  A mark painted over a point must not
    # clear the point's hit — its dots share the point's character cell, so if the
    # marks took part here they would make the current/selected track unclickable.
    lib = alli >= 0
    if lib.any():
        lorder = np.argsort(alld[lib], kind="stable")
        frame.hit.reshape(-1)[cell[lib][lorder]] = alli[lib][lorder]
    return frame


def hit_test(frame: Frame, col: int, row: int, radius: int = 2) -> Optional[int]:
    """
    The library-point index nearest character cell `(col, row)`, or None.

    Click precision is one Braille cell, so a dense region needs "nearest within
    a small radius" rather than an exact-cell match (G3's stated caveat) — a ring
    of increasing radius, returning the first hit found.  The fatter splatted dots
    make this mostly moot (a click usually lands inside a point), but the radius
    search is kept for the gaps between them.
    """
    def at(c, r):
        if 0 <= r < frame.rows and 0 <= c < frame.cols:
            v = int(frame.hit[r, c])
            return v if v >= 0 else None
        return None

    exact = at(col, row)
    if exact is not None:
        return exact
    for r in range(1, radius + 1):
        best = None
        best_d2 = None
        for cy in range(row - r, row + r + 1):
            for cx in range(col - r, col + r + 1):
                if max(abs(cx - col), abs(cy - row)) != r:
                    continue
                idx = at(cx, cy)
                if idx is None:
                    continue
                d2 = (cx - col) ** 2 + (cy - row) ** 2
                if best_d2 is None or d2 < best_d2:
                    best_d2, best = d2, idx
        if best is not None:
            return best
    return None


# ── The urwid widget ──────────────────────────────────────────────────────────

if URWID_AVAILABLE:

    _READOUT_SPEC = urwid.AttrSpec("h250", "default", 256)

    class VibeCloudWidget(urwid.Widget):
        """
        A box widget that draws the rotating cloud and answers mouse hits.

        Keyboard is deliberately *not* handled here: every binding goes through
        the TUI's one shared dispatch table (`_handle_input`), which calls this
        widget's `orbit`/`zoom_by`/`reset_view` directly.  A `keypress` method
        here would be a second binding table, the exact §4 thing not to build.
        Mouse is not handled here either — the TUI translates a global click to a
        cell and calls `select_at`, so the hit map lives with the render that
        produced it.
        """

        _sizing = frozenset([urwid.BOX])

        def __init__(self, coords: Optional[np.ndarray],
                     rgb: Optional[np.ndarray],
                     track_files: Optional[List[str]],
                     label_fn: Optional[Callable[[str], str]] = None,
                     axis_labels: Optional[List[str]] = None):
            super().__init__()
            self.coords = (np.asarray(coords, dtype=np.float64)
                           if coords is not None and len(coords) else None)
            self.rgb = (np.asarray(rgb, dtype=np.float64)
                        if rgb is not None and len(rgb) else None)
            self.track_files = list(track_files) if track_files else []
            self.label_fn = label_fn or (lambda t: t)
            self.axis_labels = list(axis_labels) if axis_labels else ["x", "y", "z"]

            # Camera: `base_azimuth` is the ambient spin; the manual offset, tilt
            # and zoom each have a *target* the mouse sets and an *actual* that
            # eases toward it every frame (buttery motion, framerate-independent).
            self.base_azimuth = 0.0
            self.manual_azimuth = self.target_manual_azimuth = 0.0
            self.tilt = self.target_tilt = DEFAULT_TILT
            self.zoom = self.target_zoom = 1.0
            self.orbit_speed = ORBIT_SPEED_DEFAULT
            # The (azimuth, tilt, zoom) the last render actually drew, so the
            # animation alarm can skip a repaint when nothing moved enough.
            self._rendered_pose: Optional[Tuple[float, float, float]] = None

            self.comet: List[np.ndarray] = []
            self._comet_arr: Optional[np.ndarray] = None   # cached stack of comet
            self.current_idx: Optional[int] = None
            self.selected_idx: Optional[int] = None

            self._last_frame: Optional[Frame] = None
            # The orbit-speed slider's location in the last render, for hit-testing
            # a click on it: (widget-local row, first track column, track width).
            self._slider: Optional[Tuple[int, int, int]] = None
            self._spec_cache: Dict[int, "urwid.AttrSpec"] = {}
            self._track_index: Dict[str, int] = {
                t: i for i, t in enumerate(self.track_files)}

        @property
        def available(self) -> bool:
            return self.coords is not None and len(self.coords) > 0

        @property
        def azimuth(self) -> float:
            return self.base_azimuth + self.manual_azimuth

        # ── camera ────────────────────────────────────────────────────────────

        def advance(self, dt: float = 1.0 / 60.0) -> None:
            """
            Step the camera by real-time `dt` seconds: advance the ambient spin,
            then ease the manual offset, tilt and zoom toward their targets.

            Framerate-independent — both the spin (rad/s × dt) and the ease
            (`1 − e^(−dt/τ)`) scale with `dt`, so raising the panel framerate makes
            motion smoother without making it faster.
            """
            if self.orbit_speed:
                self.base_azimuth = (self.base_azimuth
                                     + self.orbit_speed * dt) % (2 * math.pi)
            ease = 1.0 - math.exp(-dt / CAMERA_TAU)
            self.manual_azimuth += (self.target_manual_azimuth
                                    - self.manual_azimuth) * ease
            self.tilt += (self.target_tilt - self.tilt) * ease
            self.zoom += (self.target_zoom - self.zoom) * ease

        def orbit(self, d_azimuth: float = 0.0, d_tilt: float = 0.0) -> None:
            """Nudge the camera *target*; `advance` eases the actual toward it."""
            self.target_manual_azimuth += d_azimuth
            self.target_tilt = max(TILT_MIN, min(TILT_MAX,
                                                 self.target_tilt + d_tilt))

        def zoom_by(self, factor: float) -> None:
            self.target_zoom = max(ZOOM_MIN, min(ZOOM_MAX,
                                                 self.target_zoom * factor))

        @property
        def orbit_fraction(self) -> float:
            return self.orbit_speed / ORBIT_SPEED_MAX

        def set_orbit_fraction(self, fraction: float) -> None:
            """Set orbit speed from a 0–1 slider position (0 = static)."""
            self.orbit_speed = max(0.0, min(1.0, fraction)) * ORBIT_SPEED_MAX

        def reset_view(self) -> None:
            """Recentre the camera (the actual eases back over a few frames).
            Leaves the orbit *speed* alone — that is the listener's standing
            preference, set on the slider, not part of a pose."""
            self.target_manual_azimuth = 0.0
            self.target_tilt = DEFAULT_TILT
            self.target_zoom = 1.0
            self.selected_idx = None

        def pose_changed(self) -> bool:
            """True if the camera has moved enough since the last render to be
            worth repainting — the animation alarm gates its redraw on this."""
            if self._rendered_pose is None:
                return True
            az, ti, zo = self._rendered_pose
            return (abs(self.azimuth - az) > POSE_EPS_ANGLE
                    or abs(self.tilt - ti) > POSE_EPS_ANGLE
                    or abs(self.zoom - zo) > POSE_EPS_ZOOM)

        # ── data motion ───────────────────────────────────────────────────────

        def set_current_track(self, track_file: Optional[str]) -> None:
            self.current_idx = self._track_index.get(track_file) if track_file else None

        def note_session(self, coords3: Optional[np.ndarray]) -> None:
            """
            Record the session vector's position, iff it has actually moved.

            Only real session events (full listen, skip) move the session vector,
            so this naturally traces data motion and never the ambient camera.
            Called from the 2 Hz session sync, not the animation alarm.
            """
            if coords3 is None:
                return
            coords3 = np.asarray(coords3, dtype=np.float64)
            if self.comet and np.linalg.norm(
                    coords3 - self.comet[-1]) < COMET_MOVE_EPSILON:
                return
            self.comet.append(coords3.copy())
            if len(self.comet) > COMET_MAX:
                del self.comet[:len(self.comet) - COMET_MAX]
            self._comet_arr = None                  # invalidate the cached stack

        def clear_comet(self) -> None:
            self.comet = []
            self._comet_arr = None

        def _comet_array(self) -> Optional[np.ndarray]:
            """The comet as one `(k, 3)` array, cached until the trail changes —
            the comet advances at ≤ 2 Hz but the render runs at ~30 fps."""
            if not self.comet:
                return None
            if self._comet_arr is None:
                self._comet_arr = np.array(self.comet)
            return self._comet_arr

        # ── mouse ─────────────────────────────────────────────────────────────

        def select_at(self, col: int, row: int) -> Optional[str]:
            """
            Resolve a click at local cell `(col, row)` to a track, and select it.

            Returns the track key (for the caller to inspect / queue), or None if
            nothing is near enough.  Uses the hit map from the last render, so it
            reflects exactly what the user sees.
            """
            if self._last_frame is None:
                return None
            # A generous radius: single-dot points need a forgiving click target,
            # and the search returns the nearest point, so a near-miss still lands.
            idx = hit_test(self._last_frame, col, row, radius=CLICK_RADIUS)
            if idx is None:
                return None
            self.selected_idx = idx
            return self.track_files[idx] if idx < len(self.track_files) else None

        def clear_selection(self) -> None:
            self.selected_idx = None

        def slider_at(self, col: int, row: int) -> Optional[float]:
            """
            If local cell `(col, row)` lands on the orbit-speed slider track,
            return the 0–1 fraction it points at; otherwise None.

            Uses the slider geometry recorded by the last render, so the hit
            follows exactly what is on screen.
            """
            if self._slider is None:
                return None
            srow, x0, width = self._slider
            if row != srow or width <= 0 or not (x0 <= col < x0 + width):
                return None
            return max(0.0, min(1.0, (col - x0) / max(1, width - 1)))

        # ── rendering ─────────────────────────────────────────────────────────

        def rows(self, size, focus: bool = False) -> int:      # pragma: no cover
            return size[1]

        def render(self, size, focus: bool = False):
            cols, rows = size
            self._slider = None
            if cols <= 0 or rows <= 0:
                return urwid.SolidCanvas(" ", max(cols, 0), max(rows, 0))

            if not self.available:
                return self._message_canvas(
                    cols, rows,
                    "Vibe cloud unavailable — no mood axes in the descriptor bank."
                    "  Rebuild:  python3 generate_embeddings.py --descriptors-only")

            # Reserve the bottom for the readout (track/coords) and the orbit
            # slider, as room allows: both need ≥ 4 rows, the readout alone ≥ 3.
            show_slider = rows >= 4
            show_readout = rows >= 3
            cloud_rows = rows - (1 if show_readout else 0) - (1 if show_slider else 0)

            frame = compute_frame(
                self.coords, self.rgb, cols, cloud_rows,
                self.azimuth, self.tilt, self.zoom,
                comet=self._comet_array(),
                current_idx=self.current_idx,
                selected_idx=self.selected_idx)
            self._last_frame = frame
            self._rendered_pose = (self.azimuth, self.tilt, self.zoom)

            markup = self._frame_markup(frame, cols, cloud_rows)
            if show_readout:
                markup.append("\n")
                markup.extend(self._readout_markup(cols))
            if show_slider:
                markup.append("\n")
                markup.extend(self._slider_markup(cols, cloud_rows + 1))

            text = urwid.Text(markup, wrap="clip")
            canvas = urwid.CompositeCanvas(text.render((cols,)))
            # Guarantee the exact box height the layout asked for.
            if canvas.rows() < rows:
                canvas.pad_trim_top_bottom(0, rows - canvas.rows())
            elif canvas.rows() > rows:
                canvas.trim(0, rows)
            return canvas

        def _spec(self, code: int):
            spec = self._spec_cache.get(code)
            if spec is None:
                spec = urwid.AttrSpec(f"h{code}", "default", 256)
                self._spec_cache[code] = spec
            return spec

        def _frame_markup(self, frame: Frame, cols: int, rows: int) -> list:
            """
            Build urwid text markup from the dense grids: run-length-encoded
            colour segments, empty cells collapsed into plain-space runs, exactly
            `cols` wide per row.
            """
            bits_grid = frame.bits
            color_grid = frame.color
            markup: list = []
            for cy in range(rows):
                row_bits = bits_grid[cy]
                row_color = color_grid[cy]
                run_text: List[str] = []
                run_code: Optional[int] = None
                gap = 0

                def flush_run():
                    if run_text:
                        markup.append((self._spec(run_code), "".join(run_text)))
                        run_text.clear()

                for cx in range(cols):
                    bits = int(row_bits[cx])
                    if bits == 0:
                        flush_run()
                        run_code = None
                        gap += 1
                        continue
                    if gap:
                        markup.append(" " * gap)
                        gap = 0
                    code = int(row_color[cx])
                    if code < 0:
                        code = 231
                    glyph = chr(BRAILLE_BASE + bits)
                    if run_code is None or code == run_code:
                        run_code = code
                        run_text.append(glyph)
                    else:
                        flush_run()
                        run_code = code
                        run_text.append(glyph)
                flush_run()
                if gap:
                    markup.append(" " * gap)
                if cy != rows - 1:
                    markup.append("\n")
            return markup

        def _readout_markup(self, cols: int) -> list:
            """The bottom line: the selected/current track and its coordinates,
            or a controls hint when nothing is picked."""
            idx = self.selected_idx if self.selected_idx is not None else self.current_idx
            if idx is not None and idx < len(self.track_files):
                track = self.track_files[idx]
                label = self.label_fn(track)
                c = self.coords[idx]
                coord = "  ".join(f"{lab[:1]} {c[k]:+.1f}"
                                  for k, lab in enumerate(self.axis_labels))
                tag = "▣" if self.selected_idx is not None else "♪"
                line = f"{tag} {label}   {coord}"
            else:
                line = ("right-drag orbit · scroll / +− zoom · click a point · "
                        "[T] panes")
            line = line[:cols].ljust(cols)
            return [(_READOUT_SPEC, line)]

        def _slider_markup(self, cols: int, row_index: int) -> list:
            """
            The orbit-speed slider: a label, a clickable track with a handle, and
            the speed as a percentage of the maximum (0 = static).

            Records `self._slider = (row, x0, width)` so `slider_at` can turn a
            click on the track back into a fraction — the same "geometry lives with
            the render that produced it" discipline as the point hit map.
            """
            label = "⟳ orbit "
            frac = self.orbit_fraction
            suffix = "  static" if frac <= 0.001 else f"  {int(round(frac * 100)):>3d}%"
            track_w = cols - len(label) - len(suffix) - 2      # 2 for the brackets
            if track_w < 3:
                # No room for a real track — just say the state.
                line = (label + suffix).strip()[:cols].ljust(cols)
                return [(_READOUT_SPEC, line)]

            handle = int(round(frac * (track_w - 1)))
            track = "".join("━" if i < handle else ("●" if i == handle else "─")
                            for i in range(track_w))
            x0 = len(label) + 1                                # inside the '['
            self._slider = (row_index, x0, track_w)
            return [(_READOUT_SPEC, label + "["),
                    (self._spec(51 if frac > 0.001 else 244), track),
                    (_READOUT_SPEC, "]" + suffix.ljust(
                        max(0, cols - len(label) - 1 - track_w - 1)))]

        def _message_canvas(self, cols: int, rows: int, message: str):
            """A centred, word-wrapped message filling the box (used when the
            cloud has no axes to draw)."""
            inner = urwid.Text(message, align="center")
            body = urwid.CompositeCanvas(inner.render((cols,)))
            top = max(0, (rows - body.rows()) // 2)
            canvas = urwid.CompositeCanvas(body)
            canvas.pad_trim_top_bottom(top, max(0, rows - body.rows() - top))
            if canvas.rows() > rows:                       # very short box
                canvas.trim(0, rows)
            elif canvas.rows() < rows:
                canvas.pad_trim_top_bottom(0, rows - canvas.rows())
            return canvas
