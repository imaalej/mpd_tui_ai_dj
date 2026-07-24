"""
The cloud's spatial scaffold — the dim frame drawn behind the points so that
orbiting reads as a volume turning rather than specks swimming.

Same discipline as `test_vibe_cloud.py`: the geometry is asserted on its own
output and the widget is rendered to real canvases.  Three properties get their
own tests because each is silent when it breaks:

  * **The frame can never recolour the data.**  Eight Braille dots share one cell
    colour, so a scaffold that won a colour contest would repaint tracks grey and
    nothing would raise.
  * **The frame can never steal a click.**  Scaffold dots carry index −1; if they
    reached the hit map, clicking a point over a grid line would select nothing.
  * **The box closes on screen.**  A cube corner reaches 1.556·e vertically at the
    default tilt against a 3.0σ half-panel, so without the fit shrink the top and
    bottom corners are always cut and the "box" reads as a wall.
"""

import numpy as np
import pytest

import vibe_cloud as vc
from tui import BODY_CLOUD, BODY_HISTORY


BOXED = [m for m in vc.SCAFFOLD_MODES if m != "off"]
SIZES = [(80, 24), (96, 30), (120, 40), (60, 20), (40, 12)]


@pytest.fixture
def scene(rng):
    coords = rng.standard_normal((40, 3))
    return coords, vc.point_rgb(coords), [f"a{i}/t{i:02d}.flac" for i in range(40)]


@pytest.fixture
def widget(scene):
    coords, rgb, tracks = scene
    return vc.VibeCloudWidget(coords, rgb, tracks,
                              axis_labels=["Tone", "Saturation", "Organic"])


# ── Geometry ──────────────────────────────────────────────────────────────────


def test_off_draws_nothing_at_all(scene):
    coords, rgb, _ = scene
    plain = vc.compute_frame(coords, rgb, 96, 30, 0.7, 0.5)
    off = vc.compute_frame(coords, rgb, 96, 30, 0.7, 0.5, scaffold="off")
    assert np.array_equal(plain.bits, off.bits)
    assert np.array_equal(plain.color, off.color)


def test_every_mode_lights_more_cells_than_bare(scene):
    coords, rgb, _ = scene
    bare = int((vc.compute_frame(coords, rgb, 96, 30, 0.7, 0.5).bits != 0).sum())
    for mode in BOXED:
        lit = int((vc.compute_frame(coords, rgb, 96, 30, 0.7, 0.5,
                                    scaffold=mode).bits != 0).sum())
        assert lit > bare, f"{mode} drew nothing"


def test_the_scaffold_never_wins_a_cell_a_point_owns(scene):
    """`triad` does not rescale the scene, so every point lands in exactly the
    cell it lands in with no scaffold — which makes the colours directly
    comparable.  Any cell owning a library point must keep that point's colour."""
    coords, rgb, _ = scene
    bare = vc.compute_frame(coords, rgb, 96, 30, 0.7, 0.5)
    ruled = vc.compute_frame(coords, rgb, 96, 30, 0.7, 0.5, scaffold="triad")
    owned = bare.hit >= 0
    assert owned.any()
    assert np.array_equal(ruled.color[owned], bare.color[owned]), \
        "a scaffold dot repainted a cell that belongs to a track"


def test_the_scaffold_never_claims_a_hit(scene):
    """The gnomon is screen-anchored and rescales nothing, so the hit map must
    come out bit-identical to the no-scaffold render."""
    coords, rgb, _ = scene
    bare = vc.compute_frame(coords, rgb, 96, 30, 0.7, 0.5)
    with_g = vc.compute_frame(coords, rgb, 96, 30, 0.7, 0.5, scaffold="gnomon",
                              axis_labels=["Tone", "Saturation", "Organic"])
    assert np.array_equal(bare.hit, with_g.hit)


def test_every_mode_keeps_hits_pointing_at_real_tracks(scene):
    coords, rgb, _ = scene
    for mode in BOXED:
        frame = vc.compute_frame(coords, rgb, 96, 30, 0.7, 0.5, scaffold=mode)
        seen = frame.hit[frame.hit >= 0]
        assert seen.size and seen.max() < len(coords)


def test_the_box_closes_on_screen_at_every_tilt():
    """Drive the real projection with the eight cube corners *as* the cloud: if a
    corner is clipped it never reaches the hit map, which is exactly the failure
    the fit shrink exists to prevent."""
    e = vc.SCAFFOLD_EXTENT
    corners = np.array([[x, y, z] for x in (-e, e) for y in (-e, e)
                        for z in (-e, e)], dtype=float)
    rgb = np.full((8, 3), 200, dtype=np.uint8)
    for tilt in np.linspace(vc.TILT_MIN, vc.TILT_MAX, 15):
        for azimuth in np.linspace(0, 2 * np.pi, 9):
            R = vc.rotation_matrix(float(azimuth), float(tilt))
            flat = np.round(corners @ R.T, 9)
            # Head-on, a back corner sits exactly behind a front one and loses the
            # cell — occlusion, not clipping.  So the assertion is over the corners
            # that *can* be seen: the front-most of each coincident screen pair.
            expected = set()
            for key in {tuple(p[:2]) for p in flat}:
                same = [i for i, p in enumerate(flat) if tuple(p[:2]) == key]
                expected.add(max(same, key=lambda i: flat[i][2]))

            frame = vc.compute_frame(corners, rgb, 96, 30, float(azimuth),
                                     float(tilt), scaffold="cage")
            visible = set(int(i) for i in frame.hit[frame.hit >= 0])
            assert expected <= visible, \
                f"corner clipped at tilt={tilt:.2f} azimuth={azimuth:.2f}"


def test_each_wall_is_the_far_face_of_its_pair():
    """`walls` picks, per axis and per frame, the face whose offset moves it
    *away* from the camera — which is what makes it unable to occlude anything
    inside the box.  (Its own far corners still project in front of the origin;
    that is perspective on a plane, not an occlusion.)"""
    e, div = vc.SCAFFOLD_EXTENT, vc.SCAFFOLD_DIV
    per_axis = 2 * (div + 1)
    for azimuth in np.linspace(0, 2 * np.pi, 12):
        for tilt in (-1.0, 0.0, 0.5, 1.2):
            R = vc.rotation_matrix(float(azimuth), float(tilt))
            segs = vc._far_walls(R, e, div)
            assert len(segs) == 3 * per_axis
            for axis in range(3):
                group = segs[axis * per_axis:(axis + 1) * per_axis]
                value = group[0][0][axis]
                assert all(a[axis] == value and b[axis] == value
                           for a, b in group), "a wall segment left its plane"
                assert value * R[2][axis] <= 0, \
                    f"axis {axis} chose the near face at tilt={tilt}"


def test_the_shadow_lies_on_the_floor(scene):
    coords, _, _ = scene
    R = vc.rotation_matrix(0.3, 0.5)
    pts, weights = vc.scaffold_points("shadow", R, 20.0, coords)
    assert len(pts) == len(coords)
    assert np.allclose(pts[:, 1], -vc.SCAFFOLD_EXTENT)
    assert np.allclose(weights, vc.SHADOW_WEIGHT), \
        "the shadow must be dimmer than a ruled line, or it out-shouts the frame"


def test_stippling_actually_thins_a_line():
    """Spacing is the stand-in for opacity, so it has to change the dot count."""
    R = vc.rotation_matrix(0.0, 0.0)
    solid, _ = vc.scaffold_points("cage", R, 20.0, spacing=1.0)
    dotted, _ = vc.scaffold_points("cage", R, 20.0, spacing=2.0)
    sparse, _ = vc.scaffold_points("cage", R, 20.0, spacing=4.0)
    assert len(solid) > len(dotted) > len(sparse)


def test_axis_labels_are_drawn_as_glyphs_not_dots(scene):
    coords, rgb, _ = scene
    frame = vc.compute_frame(coords, rgb, 96, 30, 0.4, 0.5, scaffold="cage",
                             axis_labels=["Tone", "Saturation", "Organic"])
    text = "".join("".join(chr(g) if g >= 0 else " " for g in row)
                   for row in frame.glyph)
    for name in ("Tone", "Saturation", "Organic"):
        assert name in text, f"{name} was not written into the grid"


def test_the_gnomon_is_three_arms_in_a_corner(scene):
    coords, rgb, _ = scene
    frame = vc.compute_frame(coords, rgb, 96, 30, 0.4, 0.5, scaffold="gnomon",
                             axis_labels=["Tone", "Saturation", "Organic"])
    letters = [chr(g) for g in frame.glyph.ravel() if g >= 0]
    assert sorted(letters) == ["O", "S", "T"]


def test_the_gnomon_shades_by_which_way_an_arm_points():
    """The arm pointing at the viewer must be brighter than the one pointing
    away — that shading *is* the orientation cue."""
    R = vc.rotation_matrix(0.0, 0.0)
    dots_x, _, colours, _ = vc.gnomon_dots(R, 96, 30, ["Tone", "Sat", "Org"])
    # At azimuth 0 / tilt 0, axis 2 points straight at the camera and axis 0 lies
    # flat across it, so axis 2's arm must be the brightest of the three.
    arm = len(dots_x) // 3
    brightness = [int(colours[i * arm]) & 0xFF for i in range(3)]
    assert brightness[2] > brightness[0]


# ── The widget ────────────────────────────────────────────────────────────────


def test_the_widget_renders_every_mode_at_every_size(widget):
    for mode in ("off",) + tuple(BOXED):
        widget.scaffold = mode
        for cols, rows in SIZES:
            canvas = widget.render((cols, rows))
            assert canvas.cols() == cols and canvas.rows() == rows, \
                f"{mode} broke the box at {cols}x{rows}"


def test_the_widget_survives_degenerate_sizes_with_a_scaffold(widget):
    widget.scaffold = "walls+floor+shadow"
    for rows in range(1, 8):
        assert widget.render((80, rows)).rows() == rows
    widget.render((0, 10))
    widget.render((10, 0))


def test_the_panel_opens_on_a_frame_not_on_the_bare_cloud(widget):
    """Opening on `off` would ship the unreadable state as the default and make
    the fix a discovery."""
    assert widget.scaffold == vc.SCAFFOLD_DEFAULT
    assert vc.SCAFFOLD_DEFAULT != "off"
    assert vc.SCAFFOLD_DEFAULT in vc.SCAFFOLD_MODES


def test_cycling_walks_the_presets_and_wraps(widget):
    start = vc.SCAFFOLD_MODES.index(widget.scaffold)
    order = list(vc.SCAFFOLD_MODES[start + 1:]) + list(vc.SCAFFOLD_MODES[:start + 1])
    assert [widget.cycle_scaffold() for _ in vc.SCAFFOLD_MODES] == order


def test_cycling_forces_a_repaint(widget):
    """The camera may be perfectly still when the frame changes, and the
    animation alarm only repaints when the *pose* moved — so the mode change has
    to invalidate the gate itself or the new frame never appears."""
    widget.render((96, 30))
    assert widget.pose_changed() is False
    widget.cycle_scaffold()
    assert widget.pose_changed() is True


def test_an_unknown_combination_still_renders(widget):
    widget.scaffold = "cage+nonsense+shadow"
    assert widget.render((96, 30)).rows() == 30


# ── Through the one binding table ─────────────────────────────────────────────


def _cloud_ready(tui):
    tui._show_pane(BODY_CLOUD)
    assert tui.cloud is not None and tui.cloud.available


def test_b_cycles_the_frame_only_over_the_cloud(tui):
    _cloud_ready(tui)
    start = tui.cloud.scaffold
    tui._handle_input("b")
    assert tui.cloud.scaffold != start
    moved = tui.cloud.scaffold

    tui._show_pane(BODY_HISTORY)
    tui._handle_input("b")
    assert tui.cloud.scaffold == moved, "[B] acted while the cloud was not up"


def test_b_is_case_insensitive_like_every_other_binding(tui):
    _cloud_ready(tui)
    tui._handle_input("B")
    assert tui.cloud.scaffold != vc.SCAFFOLD_DEFAULT


# ── The frame has to clear the ground it is drawn on ──────────────────────────


def test_the_frame_clears_the_terminal_background():
    """The first cut of these colours was picked against a near-black browser
    mock and came out at 0.93× the real terminal background's luminance — the far
    half of the frame was darker than the ground it sat on, which is invisible.
    The ratios are the claim; this is the test of it."""
    bg = vc._luminance(vc.TERMINAL_BACKGROUND)
    assert vc._luminance(vc.COL_SCAFFOLD_FAR) / bg == pytest.approx(2.0, abs=0.05)
    assert vc._luminance(vc.COL_SCAFFOLD_NEAR) / bg == pytest.approx(6.0, abs=0.05)
    assert vc._luminance(vc.COL_SCAFFOLD_LABEL) > vc._luminance(vc.COL_SCAFFOLD_NEAR)


def test_the_frame_stays_visible_on_any_background():
    """Derived, so it must re-derive — including on a pure black terminal, where
    a pure ratio collapses to zero and the floor is the only thing left."""
    for bg in (0x000000, 0x15141b, 0x1e1e2e, 0x2b3038):
        far = vc._lift(bg, 2.0, 26.0)
        near = vc._lift(bg, 6.0, 100.0)
        assert vc._luminance(far) >= max(1.9 * vc._luminance(bg), 25.0)
        assert vc._luminance(near) > vc._luminance(far) + 40
        for shift in (16, 8, 0):
            assert 0 <= (far >> shift) & 0xFF <= 255


def test_the_shading_ramp_never_dips_below_the_far_colour(scene):
    """`_shade_between` interpolates far→near, and the shadow's weight scales the
    *near* end down — it must not push a dot below the far end, which is the
    visibility floor everything else rests on."""
    for w in (vc.SHADOW_WEIGHT, 1.0):
        ramp = vc._shade_between(np.linspace(0, 1, 32) * w,
                                 vc.COL_SCAFFOLD_FAR, vc.COL_SCAFFOLD_NEAR)
        lums = [vc._luminance(int(c)) for c in ramp]
        assert min(lums) >= vc._luminance(vc.COL_SCAFFOLD_FAR) - 0.5
        assert lums == sorted(lums)
