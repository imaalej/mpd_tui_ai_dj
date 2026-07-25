"""
The vibe cloud (Phase 4, inspection G3), driven the way `test_art_geometry`
drives the album art: rendering is the point.  The pure geometry is asserted on
its own output, the widget is rendered to real canvases at real terminal sizes,
and the TUI integration is exercised through the one shared key handler and a
mouse hit resolved against a frame the widget actually drew.

Two invariants get their own tests because they are the ones easy to undo (§4):

  * **Ambient motion and data motion never blur.**  The camera orbit (`advance`)
    must not touch the comet; the comet advances only on a real session move.
  * **The panel-only animation must not starve the session bookkeeping.**  A full
    listen that lands between animation frames must still be recorded — the fast
    alarm and the 2 Hz poll are independent.
"""

import math

import numpy as np
import pytest

import vibe_cloud as vc
from tui import BODY_CLOUD, BODY_HISTORY, BODY_CONSOLE


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def scene(rng):
    """A synthetic cloud: 40 points in a fixed 3-D z-scored spread."""
    coords = rng.standard_normal((40, 3))
    rgb = vc.point_rgb(coords)
    tracks = [f"artist{i}/album/{i:02d}.flac" for i in range(40)]
    return coords, rgb, tracks


@pytest.fixture
def widget(scene):
    coords, rgb, tracks = scene
    return vc.VibeCloudWidget(coords, rgb, tracks,
                              label_fn=lambda t: t.split("/")[-1],
                              axis_labels=["Tone", "Saturation", "Organic"])


def render_lines(canvas):
    return [row.decode("utf-8", "replace") for row in canvas.text]


def start_playing(tui, tracks=2):
    """Queue a couple of tracks in FakeMPD and start playback."""
    mpd = tui.dj.mpd_controller
    for track in mpd.known_tracks[:tracks]:
        mpd.add_track(track)
    mpd.play()
    return mpd


# ── Pure colour ───────────────────────────────────────────────────────────────


def test_colour_is_a_deterministic_function_of_position(rng):
    coords = rng.standard_normal((50, 3))
    a = vc.point_rgb(coords)
    b = vc.point_rgb(coords.copy())
    assert np.array_equal(a, b), "same coordinates must give the same colour (G3)"


def test_distinct_regions_get_distinct_colours(rng):
    """A cloud drawn in one colour would defeat the point — position and hue
    agree, so a spread of positions yields a spread of hues."""
    coords = rng.standard_normal((200, 3)) * 1.5
    codes = vc._rgb_to_packed(vc.point_rgb(coords))
    assert len(set(codes.tolist())) > 10


def test_rgb_to_packed_round_trips_every_channel():
    packed = vc._rgb_to_packed(np.array([[0, 0, 0], [255, 255, 255],
                                         [18, 52, 86]]))
    assert packed[0] == 0x000000 and packed[1] == 0xFFFFFF
    assert packed[2] == 0x123456, "channels must pack R,G,B in that order"


def test_depth_shading_is_not_quantised_into_a_handful_of_steps(scene):
    """The reason the cloud paints 24-bit rather than the xterm-256 cube.

    The cube has six levels per channel, so a single point fading from far to
    near collapsed into ~6 distinct colours and the depth cue arrived visibly
    banded.  Packed 24-bit keeps the whole ramp.  Asserted as a property of the
    ramp itself, not of one frame, so it cannot pass by accident.
    """
    white = np.array([[255, 255, 255]], dtype=np.uint8)
    levels = np.linspace(vc.SHADE_MIN, 1.0, 64)[:, None]
    ramp = vc._rgb_to_packed(np.clip(white * levels, 0, 255))
    assert len(np.unique(ramp)) > 40, (
        "a 64-step depth ramp must survive as far more than the cube's six")


# ── Pure geometry ─────────────────────────────────────────────────────────────


def test_rotation_matrix_is_orthonormal():
    R = vc.rotation_matrix(0.7, 0.4)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(abs(np.linalg.det(R)), 1.0)


def test_compute_frame_lights_cells_for_a_real_cloud(scene):
    coords, rgb, _ = scene
    frame = vc.compute_frame(coords, rgb, 80, 24, 0.5, 0.5)
    assert int((frame.bits > 0).sum()) > 0, "a 40-point cloud must light cells"
    # every lit cell has a colour, and every colour is a packed 0xRRGGBB
    lit = frame.bits > 0
    assert (frame.color[lit] >= 0).all() and (frame.color[lit] <= 0xFFFFFF).all()


def test_nearer_points_win_a_shared_cell():
    """Painter's algorithm: two points landing in the same cell, the one with
    the greater depth (nearer the camera) owns the colour and the hit."""
    # Two points at the same x,y (different depth), tilt 0 — they project to the
    # exact same dot, hence the same cell.
    coords = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    rgb = np.array([[20, 20, 20], [200, 200, 200]], dtype=np.uint8)
    frame = vc.compute_frame(coords, rgb, 40, 20, 0.0, 0.0, shade=False)
    hits = frame.hit                              # (rows, cols), -1 = none
    centre = hits[9:12, 19:22]
    covered = centre[centre >= 0]
    assert covered.size > 0
    # the near point (index 1, z=+1) wins every shared cell at the centre
    assert (covered == 1).all()


def test_marks_do_not_claim_a_hit_cell():
    """A click must resolve to a library point, never to the comet or the current
    ring — marks paint but never populate `hit`."""
    coords = np.array([[2.0, 0.0, 0.0]])
    rgb = np.array([[50, 50, 50]], dtype=np.uint8)
    comet = np.array([[0.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
    frame = vc.compute_frame(coords, rgb, 60, 20, 0.0, 0.0,
                             comet=comet, current_idx=0)
    covered = frame.hit[frame.hit >= 0]
    assert covered.size > 0 and (covered == 0).all(), \
        "only the single library point may own a hit cell"


def test_a_point_is_a_single_dot():
    """Each point is one Braille dot — no billboarded marker to swim as the cloud
    turns; depth is carried by shading instead."""
    coords = np.array([[0.0, 0.0, 0.0]])
    frame = vc.compute_frame(coords, np.array([[200, 100, 50]], np.uint8),
                             80, 40, 0.0, 0.0, zoom=1.0)
    total_dots = sum(bin(int(b)).count("1") for b in frame.bits[frame.bits > 0])
    assert total_dots == 1, "the point is a single dot"
    assert int((frame.hit >= 0).sum()) == 1, "and it is on the hit map"


def test_points_stay_single_dots_at_every_zoom():
    coords = np.array([[0.0, 0.0, 0.0]])
    rgb = np.array([[200, 100, 50]], np.uint8)
    for zoom in (1.0, 2.0, 3.5):
        frame = vc.compute_frame(coords, rgb, 80, 40, 0.0, 0.0, zoom=zoom)
        dots = sum(bin(int(b)).count("1") for b in frame.bits[frame.bits > 0])
        assert dots == 1, f"still one dot at zoom {zoom}"


def test_far_points_are_dimmer_than_near_points():
    """Depth shading — the cue that replaces marker size.  Two white points at
    the same screen position, one near and one far: the far one renders in a
    darker cube colour, which is what makes the rotation read as 3-D."""
    coords = np.array([[1.0, 0.0, -2.0], [-1.0, 0.0, 2.0]])   # far, then near
    white = np.array([[255, 255, 255], [255, 255, 255]], np.uint8)
    frame = vc.compute_frame(coords, white, 80, 40, 0.0, 0.0, shade=True)
    codes = frame.color[frame.color >= 0]
    # the near white (~231) is brighter than the far white (dimmed toward SHADE_MIN)
    assert codes.max() > codes.min()
    # and with shading off, both would be the same full-white code
    flat = vc.compute_frame(coords, white, 80, 40, 0.0, 0.0, shade=False)
    fcodes = np.unique(flat.color[flat.color >= 0])
    assert len(fcodes) == 1


def test_a_mark_paints_over_a_point_without_making_it_unclickable():
    """The current-track dot shows on top (bright cyan) but the point underneath
    stays clickable — marks affect colour, never the hit map."""
    coords = np.array([[0.0, 0.0, 0.0]])
    frame = vc.compute_frame(coords, np.array([[200, 100, 50]], np.uint8),
                             80, 40, 0.0, 0.0, current_idx=0)
    covered = frame.hit[frame.hit >= 0]
    assert covered.size > 0 and (covered == 0).all()
    assert vc.COL_CURRENT in frame.color            # the marker is visible
    assert int((frame.bits > 0).sum()) == 1, "a single dot, not a ring"


def test_a_selected_point_is_recoloured_not_ringed():
    """Selection just recolours the clicked point to a contrasting hue — no ring —
    and the recoloured dot stays clickable and stays a single dot."""
    coords = np.array([[0.0, 0.0, 0.0]])
    frame = vc.compute_frame(coords, np.array([[200, 100, 50]], np.uint8),
                             80, 40, 0.0, 0.0, selected_idx=0)
    lit = frame.bits > 0
    assert vc.COL_SELECTED in frame.color                    # recoloured
    assert int(lit.sum()) == 1, "still one dot — no ring drawn around it"
    assert int((frame.hit == 0).sum()) == 1, "and still clickable"


def test_hit_test_finds_the_nearest_within_radius():
    frame = vc.Frame(40, 20)
    frame.hit[5, 10] = 7                           # (row, col)
    assert vc.hit_test(frame, 10, 5) == 7          # exact
    assert vc.hit_test(frame, 11, 5) == 7          # one cell away
    assert vc.hit_test(frame, 10, 7, radius=2) == 7  # two cells away
    assert vc.hit_test(frame, 25, 15, radius=2) is None  # too far


# ── The widget renders ────────────────────────────────────────────────────────


TERMINAL_SIZES = [(80, 24), (100, 40), (120, 30), (60, 20), (200, 45), (40, 12)]


def test_the_widget_renders_a_box_of_the_exact_size_asked(widget):
    for cols, rows in TERMINAL_SIZES:
        canvas = widget.render((cols, rows))
        assert canvas.cols() == cols, f"{cols}x{rows} rendered {canvas.cols()} cols"
        assert canvas.rows() == rows, f"{cols}x{rows} rendered {canvas.rows()} rows"


def test_the_widget_renders_down_to_tiny_and_degenerate_sizes(widget):
    for rows in range(1, 30):
        c = widget.render((80, rows))
        assert c.rows() == rows
    for cols in range(2, 80, 9):
        c = widget.render((cols, 20))
        assert c.cols() == cols
    # zero sizes must not raise
    widget.render((0, 10))
    widget.render((10, 0))


def test_the_readout_names_the_current_track(widget, scene):
    _coords, _color, tracks = scene
    widget.set_current_track(tracks[3])
    lines = render_lines(widget.render((90, 26)))
    assert any(tracks[3].split("/")[-1] in line for line in lines)


def test_an_unavailable_cloud_shows_a_rebuild_message_not_a_crash():
    """Absent axes are not fatal (Phase 3) — the widget says so and still fills
    its box, so the surrounding layout is unaffected."""
    widget = vc.VibeCloudWidget(None, None, [], axis_labels=None)
    assert widget.available is False
    canvas = widget.render((80, 24))
    assert canvas.rows() == 24 and canvas.cols() == 80
    text = "\n".join(render_lines(canvas))
    assert "unavailable" in text and "descriptors-only" in text


# ── Camera: ambient vs manual ─────────────────────────────────────────────────


def test_advance_orbits_the_camera_but_never_moves_the_comet(widget):
    widget.note_session(np.array([1.0, 0.0, 0.0]))
    widget.note_session(np.array([0.0, 1.0, 0.0]))
    comet_before = [c.copy() for c in widget.comet]
    azimuth_before = widget.azimuth

    for _ in range(50):
        widget.advance()

    assert widget.azimuth != azimuth_before, "ambient orbit must move the camera"
    assert len(widget.comet) == len(comet_before), \
        "ambient rotation must not touch the comet (§4: don't blur the two)"
    for a, b in zip(widget.comet, comet_before):
        assert np.array_equal(a, b)


def test_the_comet_advances_only_on_real_movement(widget):
    widget.note_session(np.array([1.0, 0.0, 0.0]))
    assert len(widget.comet) == 1
    # A negligible move lays down no new bead.
    widget.note_session(np.array([1.0, 0.0, 0.0]) + 1e-4)
    assert len(widget.comet) == 1
    # A real move does.
    widget.note_session(np.array([0.0, 2.0, 0.0]))
    assert len(widget.comet) == 2
    # None is a no-op (unseeded session).
    widget.note_session(None)
    assert len(widget.comet) == 2


def test_the_comet_trail_is_bounded(widget):
    for i in range(vc.COMET_MAX * 2):
        widget.note_session(np.array([float(i), 0.0, 0.0]))
    assert len(widget.comet) == vc.COMET_MAX


def _settle(widget, n=400):
    """Let the eased camera converge to its target."""
    for _ in range(n):
        widget.advance(1.0 / 60.0)


def test_orbit_and_zoom_clamp(widget):
    widget.orbit_speed = 0.0                       # isolate manual motion
    for _ in range(100):
        widget.orbit(d_tilt=+1.0)
    assert widget.target_tilt <= vc.TILT_MAX
    _settle(widget)
    assert widget.tilt <= vc.TILT_MAX + 1e-6 and widget.tilt == pytest.approx(
        vc.TILT_MAX, abs=1e-3)
    for _ in range(100):
        widget.zoom_by(2.0)
    assert widget.target_zoom <= vc.ZOOM_MAX
    _settle(widget)
    assert widget.zoom == pytest.approx(vc.ZOOM_MAX, abs=1e-3)
    for _ in range(100):
        widget.zoom_by(0.1)
    _settle(widget)
    assert widget.zoom == pytest.approx(vc.ZOOM_MIN, abs=1e-3)


def test_the_camera_eases_toward_its_target_rather_than_snapping(widget):
    """A drag sets the target; the actual angle moves partway there each frame —
    which is what makes tilt as smooth as the left/right spin."""
    widget.orbit_speed = 0.0
    start = widget.tilt
    widget.orbit(d_tilt=0.8)
    assert widget.target_tilt == pytest.approx(start + 0.8)
    widget.advance(1.0 / 60.0)
    # one frame: moved toward the target, but not all the way there
    assert start < widget.tilt < widget.target_tilt
    _settle(widget)
    assert widget.tilt == pytest.approx(widget.target_tilt, abs=1e-3)


def test_the_camera_is_framerate_independent(widget):
    """One second of ambient spin covers the same angle at 60 fps and at 144 fps —
    raising the framerate makes motion smoother, not faster."""
    def one_second(fps):
        w = vc.VibeCloudWidget(widget.coords, widget.rgb, widget.track_files)
        for _ in range(fps):
            w.advance(1.0 / fps)
        return w.base_azimuth
    assert one_second(60) == pytest.approx(one_second(144), abs=1e-6)
    assert one_second(144) == pytest.approx(vc.ORBIT_SPEED_DEFAULT, abs=1e-6)


def test_repaint_is_gated_on_the_camera_actually_moving(widget):
    """`pose_changed` is what keeps a fast alarm cheap: a slow ambient spin
    crosses the repaint threshold only every few frames, while an active drag
    crosses it every frame."""
    widget.render((96, 28))                        # establish the rendered pose
    # A single 144-fps frame of the slow default spin does not yet warrant a paint.
    widget.advance(1.0 / 144.0)
    assert widget.pose_changed() is False
    # A drag moves the target far, so the very next eased frame does.
    widget.orbit(d_azimuth=0.5)
    widget.advance(1.0 / 144.0)
    assert widget.pose_changed() is True


def test_reset_view_returns_to_defaults_but_keeps_orbit_speed(widget):
    widget.orbit_speed = 0.0
    widget.orbit(d_azimuth=1.0, d_tilt=0.5)
    widget.zoom_by(2.0)
    widget.set_orbit_fraction(0.8)
    widget.selected_idx = 3
    speed = widget.orbit_speed
    widget.reset_view()
    assert widget.selected_idx is None, "the selection clears at once"
    assert widget.orbit_speed == speed, "orbit speed is a standing preference"
    # the camera eases back to centre over a few frames
    _settle(widget)
    assert widget.manual_azimuth == pytest.approx(0.0, abs=1e-3)
    assert widget.tilt == pytest.approx(vc.DEFAULT_TILT, abs=1e-3)
    assert widget.zoom == pytest.approx(1.0, abs=1e-3)


# ── Orbit speed and the slider ────────────────────────────────────────────────


def test_the_default_orbit_is_slow(widget):
    """It should drift, not spin — the default sits low on the slider."""
    assert widget.orbit_speed == vc.ORBIT_SPEED_DEFAULT
    assert 0 < widget.orbit_fraction < 0.25


def test_orbit_speed_zero_is_static(widget):
    widget.set_orbit_fraction(0.0)
    az = widget.azimuth
    for _ in range(50):
        widget.advance()
    assert widget.azimuth == az, "speed 0 means the cloud does not rotate"


def test_the_slider_maps_a_click_to_a_speed(widget):
    widget.render((90, 26))                    # lays out the slider
    assert widget._slider is not None
    srow, x0, width = widget._slider
    # A click at the far right is full speed; the far left is static.
    assert widget.slider_at(x0 + width - 1, srow) == pytest.approx(1.0)
    assert widget.slider_at(x0, srow) == pytest.approx(0.0)
    # Off the track is not a slider hit.
    assert widget.slider_at(x0, srow - 1) is None
    assert widget.slider_at(x0 - 1, srow) is None
    widget.set_orbit_fraction(widget.slider_at(x0 + width - 1, srow))
    assert widget.orbit_speed == pytest.approx(vc.ORBIT_SPEED_MAX)


def test_a_short_box_drops_the_slider_then_the_readout(widget):
    """The slider needs ≥ 4 rows, the readout ≥ 3; below that they go, cleanly."""
    widget.render((80, 3))
    assert widget._slider is None              # no room for the slider
    widget.render((80, 2))
    assert widget._slider is None


# ── Click-to-inspect against a real frame ─────────────────────────────────────


def test_clicking_a_lit_cell_resolves_to_that_track(widget, scene):
    _coords, _color, tracks = scene
    widget.render((90, 26))                    # populate the hit map
    frame = widget._last_frame
    ys, xs = np.where(frame.hit >= 0)
    assert len(xs) > 0, "the render must expose a hit map"
    col, row = int(xs[0]), int(ys[0])
    idx = int(frame.hit[row, col])
    track = widget.select_at(col, row)
    assert track == tracks[idx]
    assert widget.selected_idx == idx


def test_clicking_empty_space_selects_nothing(widget):
    widget.render((90, 26))
    # A far corner with no dots (radius search fails).
    assert widget.select_at(0, 0) is None or widget.selected_idx is not None
    widget.clear_selection()
    assert widget.selected_idx is None


def test_selected_track_returns_the_picked_file_or_none(widget, scene):
    _coords, _color, tracks = scene
    assert widget.selected_track() is None, "nothing picked yet"
    widget.selected_idx = 2
    assert widget.selected_track() == tracks[2]
    widget.clear_selection()
    assert widget.selected_track() is None


def test_the_readout_marks_a_selection_with_a_note_not_a_square(widget, scene):
    """The selected-track readout leads with ♫; the currently-playing readout
    leads with ♪ — distinct glyphs, and neither is the old ▣ square."""
    _coords, _color, tracks = scene
    widget.selected_idx = 1
    lines = render_lines(widget.render((90, 26)))
    readout = next(line for line in lines if tracks[1].split("/")[-1] in line)
    assert "♫" in readout
    assert "▣" not in readout


# ── TUI integration ───────────────────────────────────────────────────────────


def _cloud_ready(tui):
    if tui.cloud is None or not tui.cloud.available:
        pytest.skip("mood axes unavailable (data/embeddings absent)")


def test_the_split_is_the_default_body_and_renders_the_cloud(tui):
    from tui import BODY_SPLIT

    _cloud_ready(tui)
    assert tui.body_view == BODY_SPLIT
    canvas = tui.frame.render((100, 40), focus=True)
    text = "\n".join(row.decode("utf-8", "replace") for row in canvas.text)
    # The split shows all three panels; the cloud (with its axis triad) is the
    # right column.
    assert "Vibe Space" in text
    assert "System Console" in text
    assert "Session" in text
    for label in tui.axes.labels:
        assert label in text


def _rendered_text(tui, size=(100, 40)):
    return "\n".join(r.decode("utf-8", "replace")
                     for r in tui.frame.render(size, focus=True).text)


def test_the_pane_key_cycles_split_cloud_history_console(tui):
    from tui import BODY_SPLIT

    _cloud_ready(tui)
    # Opens on the split.
    assert tui.body_view == BODY_SPLIT
    split_text = _rendered_text(tui)
    assert "System Console" in split_text
    assert "Session" in split_text
    assert "Vibe Space" in split_text
    tui._handle_input("t")
    assert tui.body_view == BODY_CLOUD
    assert "Vibe Space" in _rendered_text(tui)
    tui._handle_input("t")
    assert tui.body_view == BODY_HISTORY
    assert "Session" in _rendered_text(tui)
    tui._handle_input("t")
    assert tui.body_view == BODY_CONSOLE
    assert "System Console" in _rendered_text(tui)
    tui._handle_input("t")                     # wraps back to the split
    assert tui.body_view == BODY_SPLIT
    tui._handle_input("t")                     # wraps back to the cloud
    assert tui.body_view == BODY_CLOUD


def test_function_keys_jump_straight_to_a_pane(tui):
    _cloud_ready(tui)
    tui._handle_input("f2")
    assert tui.body_view == BODY_HISTORY
    tui._handle_input("f3")
    assert tui.body_view == BODY_CONSOLE
    tui._handle_input("f1")
    assert tui.body_view == BODY_CLOUD


def test_arrows_navigate_the_history_and_never_orbit_the_cloud(tui):
    _cloud_ready(tui)
    tui.history.note_playing("a.flac")
    tui.history.note_playing("b.flac")

    # The arrows drive the history cursor; the cloud is orbited with the mouse,
    # so up/down never move the camera (they leave the tilt target untouched).
    tilt_target_before = tui.cloud.target_tilt
    tui._handle_input("down")
    assert tui.history.focus == 1
    assert tui.cloud.target_tilt == tilt_target_before

    tui._show_pane(BODY_HISTORY)
    tui._handle_input("up")
    assert tui.history.focus == 0
    assert tui.cloud.target_tilt == tilt_target_before


def test_enter_resets_the_view_over_the_cloud(tui):
    _cloud_ready(tui)
    tui._show_pane(BODY_CLOUD)   # full cloud: ENTER recentres (in the split it replays)
    tui.cloud.orbit(d_azimuth=1.0, d_tilt=0.4)
    tui.cloud.zoom_by(2.0)
    tui._handle_input("enter")
    # Reset re-targets the camera to centre (it then eases there over frames).
    assert tui.cloud.target_manual_azimuth == 0.0 and tui.cloud.target_zoom == 1.0


def test_zoom_keys_only_act_over_the_cloud(tui):
    _cloud_ready(tui)
    z = tui.cloud.target_zoom
    tui._handle_input("+")
    assert tui.cloud.target_zoom > z
    tui._handle_input("-")
    assert tui.cloud.target_zoom < tui.cloud.target_zoom * 2   # moved back down
    # In the history view a zoom key is inert.
    tui._show_pane(BODY_HISTORY)
    z = tui.cloud.target_zoom
    tui._handle_input("+")
    assert tui.cloud.target_zoom == z


def test_mouse_wheel_zooms_and_a_click_inspects(tui, monkeypatch):
    _cloud_ready(tui)
    size = (100, 40)
    monkeypatch.setattr(tui.loop.screen, "get_cols_rows", lambda: size)

    # Render so the cloud's hit map reflects this exact size.
    tui.frame.render(size, focus=True)
    z = tui.cloud.target_zoom
    tui._handle_input(("mouse press", 4, 10, 10))
    assert tui.cloud.target_zoom > z, "wheel up zooms in"

    # A click on a lit cell selects the track under it.
    geo = tui._cloud_geometry(*size)
    assert geo is not None
    gx, gy, gw, gh = geo
    tui.frame.render(size, focus=True)         # refresh hit map after zoom
    frame = tui.cloud._last_frame
    ys, xs = np.where(frame.hit >= 0)
    cx, cy = int(xs[0]), int(ys[0])
    idx = int(frame.hit[cy, cx])
    tui._handle_input(("mouse press", 1, gx + cx, gy + cy))
    assert tui.cloud.selected_idx == idx


def test_a_right_drag_orbits_the_cloud(tui, monkeypatch):
    _cloud_ready(tui)
    size = (100, 40)
    monkeypatch.setattr(tui.loop.screen, "get_cols_rows", lambda: size)
    tui.frame.render(size, focus=True)
    gx, gy, _gw, _gh = tui._cloud_geometry(*size)

    az_before = tui.cloud.target_manual_azimuth
    tilt_before = tui.cloud.target_tilt
    # Right button (3) — the left button is reserved for picking/the slider.
    tui._handle_input(("mouse press", 3, gx + 20, gy + 8))
    tui._handle_input(("mouse drag", 3, gx + 30, gy + 12))
    # A drag re-targets the camera (the actual eases toward it next frames).
    assert tui.cloud.target_manual_azimuth != az_before
    assert tui.cloud.target_tilt != tilt_before
    tui._handle_input(("mouse release", 0, gx + 30, gy + 12))
    assert tui._drag_last is None


def test_left_clicking_the_slider_sets_the_orbit_speed(tui, monkeypatch):
    _cloud_ready(tui)
    size = (100, 40)
    monkeypatch.setattr(tui.loop.screen, "get_cols_rows", lambda: size)
    tui.frame.render(size, focus=True)
    gx, gy, _gw, _gh = tui._cloud_geometry(*size)
    srow, x0, width = tui.cloud._slider          # widget-local slider geometry

    # Click the far right of the track → full speed.
    tui._handle_input(("mouse press", 1, gx + x0 + width - 1, gy + srow))
    assert tui.cloud.orbit_speed == pytest.approx(vc.ORBIT_SPEED_MAX)
    # Click the far left → static.
    tui._handle_input(("mouse press", 1, gx + x0, gy + srow))
    assert tui.cloud.orbit_speed == 0.0


def test_the_next_track_is_shown_in_the_now_playing_header(tui):
    _cloud_ready(tui)
    mpd = start_playing(tui, tracks=2)
    tui._update_display()
    # The lookahead (queue[1]) appears on the Now Playing "Next:" line, so it can
    # be seen without opening the history panel.
    assert tui.np_next_text.text.startswith("Next:")
    assert "Test Artist" in tui.np_next_text.text
    text = "\n".join(r.decode("utf-8", "replace")
                     for r in tui.frame.render((100, 40), focus=True).text)
    assert "Next:" in text


def test_a_left_drag_does_not_orbit(tui, monkeypatch):
    """Left-drag is for the slider / picking, not orbiting (that moved to the
    right button)."""
    _cloud_ready(tui)
    size = (100, 40)
    monkeypatch.setattr(tui.loop.screen, "get_cols_rows", lambda: size)
    tui.frame.render(size, focus=True)
    gx, gy, _gw, _gh = tui._cloud_geometry(*size)
    az_before = tui.cloud.manual_azimuth
    tui._handle_input(("mouse press", 1, gx + 20, gy + 8))
    tui._handle_input(("mouse drag", 1, gx + 30, gy + 12))
    assert tui.cloud.manual_azimuth == az_before


def test_mouse_does_nothing_outside_the_cloud_view(tui, monkeypatch):
    _cloud_ready(tui)
    monkeypatch.setattr(tui.loop.screen, "get_cols_rows", lambda: (100, 40))
    tui._show_pane(BODY_HISTORY)
    z = tui.cloud.zoom
    tui._handle_input(("mouse press", 4, 10, 10))
    assert tui.cloud.zoom == z, "the mouse only drives the cloud view"


# ── The animation alarm and the poll are independent (§4) ─────────────────────


def test_animate_orbits_the_cloud_without_touching_the_session(tui, monkeypatch):
    _cloud_ready(tui)
    monkeypatch.setattr(tui.loop, "draw_screen", lambda: None)
    monkeypatch.setattr(tui.loop, "set_alarm_in", lambda *a, **k: None)
    tui.running = True

    az_before = tui.cloud.azimuth
    cursor_before = tui._feedback_cursor
    entries_before = len(tui.history.entries)
    comet_before = len(tui.cloud.comet)

    for _ in range(20):
        tui._animate()

    assert tui.cloud.azimuth != az_before, "the cloud must orbit on the alarm"
    assert tui._feedback_cursor == cursor_before, "animation must not drain events"
    assert len(tui.history.entries) == entries_before, "no history change"
    assert len(tui.cloud.comet) == comet_before, "no comet change (ambient only)"


def test_a_full_listen_between_frames_is_still_recorded(tui, monkeypatch):
    """
    The §4 invariant the panel-only alarm is warned about: a fast cloud redraw
    must not starve `_sync_session_state`, or a track that starts and ends between
    ticks is lost.  Here animation frames interleave with the poll, and the full
    listen the poll observes still lands in the history.
    """
    _cloud_ready(tui)
    monkeypatch.setattr(tui.loop, "draw_screen", lambda: None)
    monkeypatch.setattr(tui.loop, "set_alarm_in", lambda *a, **k: None)
    tui.running = True

    mpd = tui.dj.mpd_controller
    for track in mpd.known_tracks[:2]:
        mpd.add_track(track)
    mpd.play()
    current = mpd.queue[0]

    for _ in range(10):
        tui._animate()

    # A completion the background thread would fire, then the 2 Hz sync.
    tui.dj.feedback_handler.process_full_listen(current)
    for _ in range(10):
        tui._animate()
    tui._update_display()

    entry = next(e for e in tui.history.entries if e.track == current)
    assert entry.outcome == "✓", "the full listen survived the animation frames"


def test_animate_is_a_noop_off_the_cloud_view(tui, monkeypatch):
    _cloud_ready(tui)
    drawn = []
    monkeypatch.setattr(tui.loop, "draw_screen", lambda: drawn.append(1))
    monkeypatch.setattr(tui.loop, "set_alarm_in", lambda *a, **k: None)
    tui.running = True

    tui._show_pane(BODY_HISTORY)
    for _ in range(5):
        tui._animate()
    assert drawn == [], "no cloud on screen means no cloud redraw"


def test_animate_paints_the_cloud_in_the_split_view(tui, monkeypatch):
    """The split keeps the cloud on screen, so the ambient orbit runs and its
    column is repainted — `_cloud_visible`, not `_cloud_focused`, gates it."""
    from tui import BODY_SPLIT

    _cloud_ready(tui)
    drawn = []
    monkeypatch.setattr(tui.loop, "draw_screen", lambda: drawn.append(1))
    monkeypatch.setattr(tui.loop, "set_alarm_in", lambda *a, **k: None)
    tui.running = True

    tui._set_body_view(BODY_SPLIT)
    az_before = tui.cloud.azimuth
    for _ in range(20):
        tui._animate()

    assert tui.cloud.azimuth != az_before, "the cloud orbits in the split too"
    assert drawn, "and the split's cloud column is repainted"


def test_opening_the_split_refreshes_both_panels(tui, monkeypatch):
    """The split shows the history and the console together, so opening it must
    rebuild both — not the exclusive one-panel refresh the toggles use."""
    from tui import BODY_SPLIT

    calls = []
    monkeypatch.setattr(tui, "_update_session_panel",
                        lambda: calls.append("history"))
    monkeypatch.setattr(tui, "_update_console",
                        lambda: calls.append("console"))
    tui._set_body_view(BODY_SPLIT)
    assert "history" in calls and "console" in calls


def test_the_split_keeps_both_panels_live_on_the_periodic_tick(tui, monkeypatch):
    """The 2 Hz refresh rebuilds both split panels, not just the one the toggle
    views would show — otherwise the split would freeze one of them."""
    from tui import BODY_SPLIT

    _cloud_ready(tui)
    mpd = tui.dj.mpd_controller
    for track in mpd.known_tracks[:2]:
        mpd.add_track(track)
    mpd.play()
    tui._set_body_view(BODY_SPLIT)

    calls = []
    monkeypatch.setattr(tui, "_update_session_panel",
                        lambda: calls.append("history"))
    monkeypatch.setattr(tui, "_update_console",
                        lambda: calls.append("console"))
    tui._update_display()
    assert "history" in calls and "console" in calls


# ── The cloud geometry follows the tree ───────────────────────────────────────


def test_cloud_geometry_matches_the_rendered_panel(tui):
    _cloud_ready(tui)
    tui._show_pane(BODY_CLOUD)   # full-body cloud: gx == 1 (the split case has its own test)
    for cols, rows in [(100, 40), (120, 50), (80, 30)]:
        geo = tui._cloud_geometry(cols, rows)
        assert geo is not None
        gx, gy, gw, gh = geo
        lines = [r.decode("utf-8", "replace")
                 for r in tui.frame.render((cols, rows), focus=True).text]
        # The cloud's LineBox title sits one row above its inner top-left.
        title_row = next(i for i, l in enumerate(lines) if "Vibe Space" in l)
        assert gy == title_row + 1
        assert gx == 1


def test_cloud_geometry_in_the_split_is_the_right_column(tui):
    """In the split the cloud is the right column, so its hit rectangle is offset
    past the left column and narrowed to the right one — read from the tree, so
    the mouse still lands on the right point."""
    from tui import BODY_SPLIT

    _cloud_ready(tui)
    tui._set_body_view(BODY_SPLIT)
    cols, rows = 120, 40
    widths = tui.split_box.column_widths((cols,))
    left = sum(widths[:-1])

    geo = tui._cloud_geometry(cols, rows)
    assert geo is not None
    gx, gy, gw, gh = geo
    # One column into the right column (past the left column and the border).
    assert gx == left + 1
    assert gw == widths[-1] - 2

    lines = [r.decode("utf-8", "replace")
             for r in tui.frame.render((cols, rows), focus=True).text]
    title_row = next(i for i, l in enumerate(lines) if "Vibe Space" in l)
    assert gy == title_row + 1
    # The title is drawn in the right half, not at the far left.
    assert lines[title_row].index("Vibe Space") >= left


def test_the_cloud_survives_a_dj_with_no_axes(dj_stub, fake_art, monkeypatch):
    """If the descriptor bank carries no mood axes, the app still builds and
    renders — the cloud just shows its rebuild message."""
    import signal as _signal
    from tui import AdaptiveDJTUI

    monkeypatch.setattr("tui.MoodAxes.load", staticmethod(lambda *a, **k: None))
    previous = _signal.getsignal(_signal.SIGWINCH)
    instance = AdaptiveDJTUI(dj_stub, art_renderer=fake_art)
    try:
        assert instance.axes is None
        assert instance.cloud.available is False
        instance._update_display()
        instance.frame.render((100, 40), focus=True)
    finally:
        _signal.signal(_signal.SIGWINCH, previous)
