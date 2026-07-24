"""
The vibe cloud — Phase 4's centrepiece, a rotating 3-D point cloud of the library
in mood space, drawn in the terminal with Braille + 24-bit colour.

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
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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

# urwid's colour-depth selector for 24-bit.  The cloud's colours are packed
# 0xRRGGBB throughout (`Frame.color`, the marker constants) and are handed to
# `AttrSpec` as '#rrggbb'.  `tui.py` declares the same depth on the screen.
TRUECOLOR = 1 << 24

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

# ── The scaffold (exploration) ────────────────────────────────────────────────
#
# The readability problem this addresses: a cloud of specks has no reference
# frame, so orbiting reads as "the dots are swimming" rather than "the volume is
# turning".  A dim static frame gives the eye something to hold — the classic
# fix in every 3-D plot, and free here because the frame goes through the same
# projection and painter's sort as the points.
#
# Two constraints Braille imposes, and how each is met:
#   * **No alpha.** "Lower opacity" is (a) a genuinely dim colour — 24-bit means
#     a near-background grey is available, which the 256-cube could not do — and
#     (b) **stippling**: sampling a line every 2nd or 3rd dot instead of every
#     dot, so the line reads as a faint dotted rule rather than a hard wall.
#   * **Eight dots share one cell colour.**  So the scaffold is splatted with a
#     large *negative* depth bump: its dots light up, but any library point in
#     the same cell wins that cell's colour.  The frame can never recolour the
#     data, only sit behind it.
# The box is **±2σ**, not a number picked to look right: the coordinates are
# z-scored per axis (§6), so 2.0 is a stated fact about the library — most of it
# is inside the box and the outliers visibly are not.  It also happens to be the
# largest box that stays on the panel: at the fixed 0.5 tilt a cube corner
# reaches 1.556·e vertically against the panel's 3.0σ half-height.
SCAFFOLD_EXTENT = 2.0
SCAFFOLD_DIV = 4               # grid subdivisions per wall/floor
SCAFFOLD_SPACING = 2.0         # dots between line samples: 1 = solid, 2 = dotted
COL_SCAFFOLD_FAR = 0x0f141a    # the far side of the frame, barely above the bg
COL_SCAFFOLD_NEAR = 0x55697e   # the near side — still well under any point
COL_SCAFFOLD_LABEL = 0x7a8b9c  # axis names
# The gnomon is screen-anchored, so it is sized in **cells** rather than σ, and
# it never depth-shades: it is an instrument, not part of the scene.
GNOMON_CELLS = 5               # arm length, in character cells
GNOMON_MARGIN = 3
SHADOW_WEIGHT = 0.55           # the cast shadow, relative to a ruled line

# The presets `[B]` cycles, chosen by eye from the ten the offline explorer
# renders (`explore_cloud_scaffold.py`).  Tokens compose with '+', so a
# combination needs no new constant and the ones not in this list — `walls`,
# `gnomon` — stay reachable from the explorer without shipping as a stop.
#
#   ground   the horizon reading: a ruled floor and the library's cast shadow.
#            A point's distance from its own shadow is its height.
#   box      the enclosure reading: all twelve edges plus the ruled axes through
#            the origin, ticked every 1σ.
#   marks    the light-touch middle: crop-mark corners over the floor, with the
#            ticked axes, and no shadow mass under it.
SCAFFOLD_MODES: Tuple[str, ...] = (
    "off", "floor+shadow", "cage+triad", "corners+floor+triad")

# What the panel opens with.  A frame rather than `off`, because the reason the
# scaffold exists is that the bare cloud is hard to read while it turns — opening
# on the unreadable state and asking the listener to discover `[B]` would keep the
# defect as the default.  `off` stays one keypress away, first in the ring.
SCAFFOLD_DEFAULT = "floor+shadow"

# Bright, full-brightness contrasting colours for the markers — each drawn as a
# single dot on top of the (depth-shaded) library so it stands out.  Packed
# 0xRRGGBB, like everything else in `Frame.color`; the values are the exact RGB
# the xterm-256 codes these replace resolved to, so the palette is unchanged.
COL_COMET_HEAD = 0xFFFFFF     # white — the session's current position (was 231)
COL_COMET_TRAIL = (0xFFD7AF, 0xFFAF87, 0xFF875F, 0xFF5F00)  # amber → orange
COL_CURRENT = 0x00FFFF        # bright cyan — the playing track (was 51)
COL_SELECTED = 0xFF00FF       # bright magenta — a clicked point (was 201)


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


def _rgb_to_packed(rgb: np.ndarray) -> np.ndarray:
    """
    `(N, 3)` RGB → `(N,)` packed 0xRRGGBB, vectorised.

    This used to quantise to the xterm-256 cube (6 levels per channel), which
    collapsed the 648 distinct colours a depth-shaded library wants into 32 —
    a single point fading far→near got **six** steps, so the depth cue arrived
    banded.  24-bit keeps all of them; `VibeCloudWidget._spec` declares the
    capability and urwid still down-converts on a terminal that lacks it.
    """
    v = np.clip(np.asarray(rgb, dtype=np.float64), 0, 255).astype(np.int32)
    return (v[:, 0] << 16) | (v[:, 1] << 8) | v[:, 2]


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

    All four are `(rows, cols)` arrays: `bits` is the OR of the Braille dot bits
    lit in each character cell; `color` is the packed 0xRRGGBB of the front-most
    thing there (−1 = empty); `hit` is the index of the front-most *library* point
    whose dot falls there (−1 = none / a mark), so a click resolves to a track,
    never to the comet; `glyph` is a codepoint that **replaces** the Braille glyph
    in that cell (−1 = use the dots), which is how the axis labels are drawn into
    the same grid without a second widget.  Dense rather than dict because the
    rasteriser fills them with vectorised numpy, and the markup builder and
    hit-test read them directly.
    """

    __slots__ = ("bits", "color", "hit", "glyph", "cols", "rows")

    def __init__(self, cols: int, rows: int):
        self.cols = cols
        self.rows = rows
        self.bits = np.zeros((rows, cols), dtype=np.uint8)
        self.color = np.full((rows, cols), -1, dtype=np.int32)
        self.hit = np.full((rows, cols), -1, dtype=np.int32)
        self.glyph = np.full((rows, cols), -1, dtype=np.int32)


# ── Scaffold geometry ─────────────────────────────────────────────────────────
#
# Everything is expressed as world-space line segments and then **sampled into
# points**, so the scaffold rides the existing projection, depth sort and
# rasteriser rather than needing a line drawer of its own.  Projection is
# orthographic, so evenly spaced samples in world space stay evenly spaced in
# dot space — a sample step derived from `scale` gives an exact dot pitch.


Segment = Tuple[Tuple[float, float, float], Tuple[float, float, float]]


def _cube_edges(e: float) -> List[Segment]:
    """The 12 edges of the cube [−e, e]³."""
    corners = [(x, y, z) for x in (-e, e) for y in (-e, e) for z in (-e, e)]
    segs: List[Segment] = []
    for i, p in enumerate(corners):
        for q in corners[i + 1:]:
            if sum(abs(p[k] - q[k]) > 1e-9 for k in range(3)) == 1:
                segs.append((p, q))
    return segs


def _corner_brackets(e: float, fraction: float = 0.28) -> List[Segment]:
    """Only the ends of each edge — eight crop-mark corners.

    Reads as a box for a third of the ink of a full cage, and leaves the middle
    of the view (where the cloud is densest) completely clear.
    """
    d = 2 * e * fraction
    segs: List[Segment] = []
    for corner in [(x, y, z) for x in (-e, e) for y in (-e, e) for z in (-e, e)]:
        for k in range(3):
            end = list(corner)
            end[k] += -d if corner[k] > 0 else d
            segs.append((corner, tuple(end)))
    return segs


def _plane_grid(axis: int, value: float, e: float, div: int) -> List[Segment]:
    """A `div`×`div` grid ruled on the plane `axis = value`."""
    others = [k for k in range(3) if k != axis]
    steps = np.linspace(-e, e, div + 1)
    segs: List[Segment] = []
    for fixed, varying in (others, others[::-1]):
        for s in steps:
            a, b = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
            a[axis] = b[axis] = value
            a[fixed] = b[fixed] = float(s)
            a[varying], b[varying] = -e, e
            segs.append((tuple(a), tuple(b)))
    return segs


def _far_walls(R: np.ndarray, e: float, div: int) -> List[Segment]:
    """The three faces currently *behind* the cloud, ruled as grids.

    matplotlib's 3-D axes do exactly this and it is why they read as a box: the
    back walls give parallax and a horizon, and because they are chosen per frame
    by which side is farther from the camera they can never cross in front of the
    data.  A rotation past 90° swaps a wall — the switch is visible, and it is
    the same switch the eye already expects from the corner it is watching.
    """
    segs: List[Segment] = []
    for axis in range(3):
        # projected z of the +unit vector on this axis; farther = smaller z.
        sign = -1.0 if R[2, axis] > 0 else 1.0
        segs.extend(_plane_grid(axis, sign * e, e, div))
    return segs


def _axis_triad(e: float, tick: float = 0.12) -> List[Segment]:
    """Three ruled axes through the origin, with a tick every 1σ.

    The "3-axis graph" reading of the problem: instead of bounding the volume,
    name it.  Each tick is one z-score, so the frame doubles as the scale the
    readout line quotes.
    """
    segs: List[Segment] = []
    for axis in range(3):
        a, b = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        a[axis], b[axis] = -e, e
        segs.append((tuple(a), tuple(b)))
        for t in np.arange(-math.floor(e), math.floor(e) + 1):
            if abs(t) < 1e-9:
                continue
            for cross in [k for k in range(3) if k != axis]:
                p, q = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
                p[axis] = q[axis] = float(t)
                p[cross], q[cross] = -tick, tick
                segs.append((tuple(p), tuple(q)))
    return segs


def _sample_segments(segs: Sequence[Segment], scale: float,
                     spacing: float) -> np.ndarray:
    """Segments → `(M, 3)` sample points, one every `spacing` **dots**.

    `spacing = 1` is a solid line; 2 lights every other dot (the stipple that
    stands in for opacity); 3 is a faint dotted rule.
    """
    out: List[np.ndarray] = []
    for a, b in segs:
        p = np.asarray(a, dtype=np.float64)
        q = np.asarray(b, dtype=np.float64)
        length = float(np.linalg.norm(q - p))
        n = max(2, int(length * scale / max(0.25, spacing)) + 1)
        t = np.linspace(0.0, 1.0, n)[:, None]
        out.append(p + (q - p) * t)
    return np.concatenate(out) if out else np.zeros((0, 3))


def scaffold_points(mode: str,
                    R: np.ndarray,
                    scale: float,
                    coords: Optional[np.ndarray] = None,
                    extent: float = SCAFFOLD_EXTENT,
                    div: int = SCAFFOLD_DIV,
                    spacing: float = SCAFFOLD_SPACING
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    `(M, 3)` world-space dots for a scaffold `mode`, and their `(M,)` weights.

    The weight scales how bright a dot is allowed to get at the near end of the
    depth ramp — 1.0 for ruled lines, less for the shadow, which is a large solid
    mass and would otherwise out-shout the frame it sits in.

    `mode` is a '+'-joined set of tokens so combinations need no new constant:

        corners   the eight crop-mark cube corners  (least ink)
        cage      all twelve cube edges
        walls     the three rear faces, ruled as grids  (matplotlib's frame)
        floor     one ruled grid under the cloud
        shadow    the library flattened onto the floor plane — the strongest
                  depth cue available, since a point's distance from its own
                  shadow *is* its height
        triad     three ruled axes through the origin, ticked every 1σ
    """
    empty = (np.zeros((0, 3)), np.zeros(0))
    tokens = {t for t in str(mode).split("+") if t and t != "off"}
    if not tokens:
        return empty

    segs: List[Segment] = []
    if "corners" in tokens:
        segs += _corner_brackets(extent)
    if "cage" in tokens:
        segs += _cube_edges(extent)
    if "walls" in tokens:
        segs += _far_walls(R, extent, div)
    if "floor" in tokens:
        segs += _plane_grid(1, -extent, extent, div)
    if "triad" in tokens:
        segs += _axis_triad(extent)

    parts: List[np.ndarray] = []
    weights: List[np.ndarray] = []
    if segs:
        pts = _sample_segments(segs, scale, spacing)
        parts.append(pts)
        weights.append(np.ones(len(pts)))
    if "shadow" in tokens and coords is not None and len(coords):
        flat = np.asarray(coords, dtype=np.float64).copy()
        flat[:, 1] = -extent
        parts.append(flat)
        weights.append(np.full(len(flat), SHADOW_WEIGHT))
    if not parts:
        return empty
    return np.concatenate(parts), np.concatenate(weights)


def gnomon_dots(R: np.ndarray, cols: int, rows: int,
                labels: Optional[Sequence[str]] = None
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                           List[Tuple[int, int, str]]]:
    """
    A small screen-anchored axis cross for the bottom-left corner.

    The other scaffolds live in the scene; this one does not.  It is the
    orientation instrument every 3-D editor puts in a corner: three arms rotating
    with the camera, each shaded by whether it points **toward** the viewer or
    away, so "which way am I looking" is answerable without reading the cloud at
    all.  Sized in cells, unaffected by zoom, and it never occludes the data
    because the data is never in that corner.

    Returns `(dot_x, dot_y, packed_colour, [(col, row, letter)])`.
    """
    arm = GNOMON_CELLS * 4                       # dots; a dot is ~square on screen
    x0 = GNOMON_MARGIN * 2 + arm
    y0 = 4 * rows - GNOMON_MARGIN * 4 - arm
    xs, ys, cs = [], [], []
    text: List[Tuple[int, int, str]] = []
    names = list(labels) if labels else ["x", "y", "z"]
    for axis in range(3):
        vx, vy, vz = R[0, axis], R[1, axis], R[2, axis]
        t = np.linspace(0.0, 1.0, arm)
        dx = np.round(x0 + vx * arm * t).astype(np.int32)
        dy = np.round(y0 - vy * arm * t).astype(np.int32)
        # Depth along the arm, not per dot: the whole arm is one thing pointing
        # somewhere, and a gradient down its length would read as bending.
        shade = np.full(len(t), (float(vz) + 1.0) / 2.0)
        xs.append(dx)
        ys.append(dy)
        cs.append(_shade_between(shade, COL_SCAFFOLD_FAR, COL_SCAFFOLD_NEAR))
        tip_col = int(round(x0 + vx * (arm + 3))) // 2
        tip_row = int(round(y0 - vy * (arm + 3))) // 4
        letter = (names[axis][:1].upper() if axis < len(names) else "?")
        text.append((tip_col, tip_row, letter))
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(cs), text)


def _shade_between(t: np.ndarray, far: int, near: int) -> np.ndarray:
    """Packed colours interpolated far→near by `t` ∈ [0, 1], vectorised."""
    lo = np.array([(far >> 16) & 0xFF, (far >> 8) & 0xFF, far & 0xFF], float)
    hi = np.array([(near >> 16) & 0xFF, (near >> 8) & 0xFF, near & 0xFF], float)
    return _rgb_to_packed(lo[None, :] + (hi - lo)[None, :] * t[:, None])


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
                  shade: bool = True,
                  scaffold: str = "off",
                  axis_labels: Optional[Sequence[str]] = None) -> Frame:
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
    half_dots = min(dot_w, dot_h) / 2.0
    scale = half_dots / BASE_HALF_EXTENT
    cx0, cy0 = dot_w / 2.0, dot_h / 2.0

    tokens = {t for t in str(scaffold).split("+") if t and t != "off"}
    if tokens & {"cage", "corners", "walls", "floor"}:
        # Shrink just enough that the box **closes** on screen at zoom 1.  A cube
        # corner reaches `e·(|cos tilt| + √2·|sin tilt|)` from centre, which at the
        # default tilt is 3.11σ against the panel's 3.0 — so without this the top
        # and bottom corners are always cut and the frame reads as a wall, not a
        # box.  The budget is `half_dots − 1`, not `half_dots`: the far corner
        # sits at `cy0 + reach·scale`, and a dot rounded exactly onto `dot_h` is
        # one past the last index and silently vanishes — a corner missing from
        # the cube for one dot of greed.  Applied to the
        # base scale and *then* multiplied by zoom, so zooming in still overflows
        # the panel as it should — this fits the box, it does not cage the camera.
        reach = SCAFFOLD_EXTENT * (abs(math.cos(tilt))
                                   + math.sqrt(2.0) * abs(math.sin(tilt)))
        if reach * scale > half_dots - 1.0:
            scale = (half_dots - 1.0) / reach
    scale *= zoom

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

    def splat_dots(dx, dy, colour, depth: float) -> None:
        """Add dots already in dot coordinates (the screen-anchored gnomon)."""
        if len(dx) == 0:
            return
        xs.append(np.asarray(dx, dtype=np.int32))
        ys.append(np.asarray(dy, dtype=np.int32))
        depths.append(np.full(len(dx), depth, dtype=np.float64))
        colours.append(np.asarray(colour, dtype=np.int32))
        idxs.append(np.full(len(dx), -1, dtype=np.int32))

    # The scaffold first, and with a large *negative* bump: its dots are OR'd
    # into the Braille cells like anything else, but it loses every colour
    # contest, so a cell holding a library point shows the point's colour.  The
    # frame sits behind the data by construction, not by tuning.
    gnomon_text: List[Tuple[int, int, str]] = []
    if "gnomon" in tokens:
        gx, gy, gc, gnomon_text = gnomon_dots(R, cols, rows, axis_labels)
        splat_dots(gx, gy, gc, -99.0)

    scaf, scaf_w = scaffold_points(scaffold, R, scale, coords)
    if len(scaf):
        sproj = scaf @ R.T
        sz = sproj[:, 2]
        span = float(sz.max() - sz.min())
        t = ((sz - sz.min()) / span) if span > 1e-9 else np.full(len(sz), 0.5)
        splat(sproj, _shade_between(t * scaf_w, COL_SCAFFOLD_FAR,
                                    COL_SCAFFOLD_NEAR), -1, -100.0)

    projected = coords @ R.T

    # Depth-shade the point colours by this frame's front-to-back range.
    z = projected[:, 2]
    if shade and len(z) and float(z.max() - z.min()) > 1e-9:
        t = (z - z.min()) / (z.max() - z.min())
        factor = SHADE_MIN + (1.0 - SHADE_MIN) * t
        packed = _rgb_to_packed(
            np.clip(np.asarray(rgb, dtype=np.float64) * factor[:, None], 0, 255))
    else:
        packed = _rgb_to_packed(rgb)

    splat(projected, packed, np.arange(len(coords), dtype=np.int32), 0.0)

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

    for col, row, letter in gnomon_text:
        if 0 <= col < frame.cols and 0 <= row < frame.rows:
            frame.glyph[row, col] = ord(letter)
            frame.color[row, col] = COL_SCAFFOLD_LABEL
    if axis_labels is not None and tokens - {"gnomon"}:
        _place_axis_labels(frame, axis_labels, R, scale, cx0, cy0)
    return frame


def _place_axis_labels(frame: Frame, labels: Sequence[str], R: np.ndarray,
                       scale: float, cx0: float, cy0: float,
                       extent: float = SCAFFOLD_EXTENT) -> None:
    """Write each axis name at that axis's positive tip, in the same grid.

    A label is a glyph override on the cell, so it costs no extra widget and no
    extra pass — but it *does* replace whatever dots were there, which is why it
    is drawn last and kept to a short word.  Nothing is written where the tip
    falls off the panel.
    """
    tips = np.eye(3) * (extent * 1.22)
    for axis, name in enumerate(list(labels)[:3]):
        p = tips[axis] @ R.T
        col = int(round((p[0] * scale + cx0)) // 2)
        row = int(round((cy0 - p[1] * scale)) // 4)
        text = str(name)[:10]
        col = max(0, min(frame.cols - len(text), col - len(text) // 2))
        if not (0 <= row < frame.rows) or frame.cols < len(text):
            continue
        for i, ch in enumerate(text):
            frame.glyph[row, col + i] = ord(ch)
            frame.color[row, col + i] = COL_SCAFFOLD_LABEL


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

    _READOUT_SPEC = urwid.AttrSpec("#bcbcbc", "default", TRUECOLOR)

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

            # Which spatial frame is drawn behind the cloud (`[B]` cycles it).
            self.scaffold = SCAFFOLD_DEFAULT

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

        def cycle_scaffold(self, step: int = 1) -> str:
            """Advance to the next scaffold preset and return its name."""
            try:
                i = SCAFFOLD_MODES.index(self.scaffold)
            except ValueError:                             # a custom combination
                i = -1
            self.scaffold = SCAFFOLD_MODES[(i + step) % len(SCAFFOLD_MODES)]
            self._rendered_pose = None                     # force one repaint
            return self.scaffold

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
                selected_idx=self.selected_idx,
                scaffold=self.scaffold,
                axis_labels=self.axis_labels)
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
            """An `AttrSpec` for a packed 0xRRGGBB colour, cached.

            The cache key is quantised to 5 bits per channel.  With the 256-cube
            the key space was 256 entries; 24-bit colour is continuous, so an
            exact-keyed cache in a session that runs all night would grow without
            bound.  32 levels per channel caps it at 32,768 — still five times the
            cube's resolution per channel, and the step is imperceptible.
            """
            key = code & 0xF8F8F8
            spec = self._spec_cache.get(key)
            if spec is None:
                spec = urwid.AttrSpec(f"#{key:06x}", "default", TRUECOLOR)
                self._spec_cache[key] = spec
            return spec

        def _frame_markup(self, frame: Frame, cols: int, rows: int) -> list:
            """
            Build urwid text markup from the dense grids: run-length-encoded
            colour segments, empty cells collapsed into plain-space runs, exactly
            `cols` wide per row.
            """
            bits_grid = frame.bits
            color_grid = frame.color
            glyph_grid = frame.glyph
            markup: list = []
            for cy in range(rows):
                row_bits = bits_grid[cy]
                row_color = color_grid[cy]
                row_glyph = glyph_grid[cy]
                run_text: List[str] = []
                run_code: Optional[int] = None
                gap = 0

                def flush_run():
                    if run_text:
                        markup.append((self._spec(run_code), "".join(run_text)))
                        run_text.clear()

                for cx in range(cols):
                    bits = int(row_bits[cx])
                    over = int(row_glyph[cx])
                    if bits == 0 and over < 0:
                        flush_run()
                        run_code = None
                        gap += 1
                        continue
                    if gap:
                        markup.append(" " * gap)
                        gap = 0
                    code = int(row_color[cx])
                    if code < 0:
                        code = 0xFFFFFF
                    # A glyph override (an axis label) replaces the dots in its
                    # cell — drawn last in `compute_frame`, so it wins here too.
                    glyph = chr(over) if over >= 0 else chr(BRAILLE_BASE + bits)
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
                    (self._spec(0x00FFFF if frac > 0.001 else 0x808080), track),
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
