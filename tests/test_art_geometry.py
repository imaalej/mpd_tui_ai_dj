"""
Album-art geometry, derived from the widget tree (audit H8/L3).

These are the first tests in the project that construct the urwid widget tree.
That is the point: 311 tests covered everything *behind* the display, which is
how H1, C1 and C4 all shipped under a green suite — and how a `WidgetError` on
every terminal shorter than 33 rows shipped with them, unnoticed, for the whole
of Stages 0–2.

The load-bearing test here is `test_the_computed_rectangle_is_where_the_art_area
_actually_renders`.  Asserting that `_art_geometry` returns (1, 3, 33, 11) would
only prove the function agrees with itself — exactly the mistake the audit calls
out about tests that mirror the ordering they are checking.  Instead the art
placeholder is given a marker, the *real* frame is rendered to a canvas, and the
marker is located in the output.  If the geometry and the layout disagree, the
marker is not where the rectangle says it is.
"""

import urwid

from tui import ART_COLS, MIN_ART_COLS, MIN_ART_ROWS


TERMINAL_SIZES = [
    (80, 24),    # the classic default — this one used to raise WidgetError
    (80, 30),
    (80, 40),
    (100, 50),
    (120, 60),
    (60, 20),
    (200, 45),
]


def render_text(frame, cols, rows):
    canvas = frame.render((cols, rows), focus=True)
    return [row.decode('utf-8', 'replace') for row in canvas.text]


def find_marker(lines, marker):
    for row, text in enumerate(lines):
        if marker in text:
            return row, text.index(marker)
    return None


# ── The layout renders at all ────────────────────────────────────────────────


def test_the_frame_renders_at_every_terminal_size(tui):
    """
    The regression that motivates this whole file.

    Before Stage 3 the Now Playing box took `('weight', 3, …)` of the body.  Its
    content is a flow Pile wrapping a Columns whose left cell is a box Filler, so
    urwid resolved the Columns as a box widget and rendered it at its natural
    height while the Pile handed it the weighted height — a size mismatch that
    raises `WidgetError`.  Measured: it raised on every height from 20 to 32 rows
    inclusive, i.e. on an 80x24 terminal, and nothing in the suite noticed
    because nothing in the suite had ever called `render()`.
    """
    for cols, rows in TERMINAL_SIZES:
        canvas = tui.frame.render((cols, rows), focus=True)
        assert canvas.rows() == rows, f"{cols}x{rows} rendered {canvas.rows()} rows"
        assert canvas.cols() == cols, f"{cols}x{rows} rendered {canvas.cols()} cols"


def test_it_renders_down_to_absurdly_small_terminals(tui):
    """No size should crash; the panels clip instead."""
    for rows in range(6, 34):
        tui.frame.render((80, rows), focus=True)
    for cols in range(40, 90, 7):
        tui.frame.render((cols, 24), focus=True)


# ── The geometry matches the render ──────────────────────────────────────────


def test_the_computed_rectangle_starts_where_the_art_area_actually_renders(tui):
    """
    Locate the art placeholder in a real canvas and check the rectangle's
    top-left corner lands on it, at every terminal size.

    The placeholder is centre-aligned in the art cell, so its column is a
    deterministic function of `x` and `width`; urwid renders the cell's box
    Filler at its first row, so its row is `y`.  This asserts the geometry
    against the renderer's own arithmetic rather than against a repeat of the
    geometry's — an assertion of `(1, 3, 33, 11)` would prove only that the
    function agrees with itself.
    """
    marker = "COVER"
    tui.album_art_placeholder.set_text(marker)

    for cols, rows in TERMINAL_SIZES:
        geometry = tui._art_geometry(cols, rows)
        assert geometry is not None, f"no geometry at {cols}x{rows}"
        x, y, width, _height = geometry

        lines = render_text(tui.frame, cols, rows)
        found = find_marker(lines, marker)
        assert found is not None, f"placeholder not rendered at {cols}x{rows}"
        marker_row, marker_col = found

        assert marker_col == x + (width - len(marker)) // 2, (
            f"{cols}x{rows}: art column starts at "
            f"{marker_col - (width - len(marker)) // 2}, geometry says {x}")
        assert marker_row == y, (
            f"{cols}x{rows}: art rows start at {marker_row}, geometry says {y}")


def test_the_rectangle_spans_exactly_the_rows_of_the_track_info_column(tui):
    """
    The vertical extent, pinned to the column the art sits beside.

    This is the property the deleted `RIGHT_COL_ROWS` constant existed to
    guarantee — "even a very wide cover cannot overflow downward past the seek
    bar or vibe line" — restated as a fact about the render rather than as a
    comment above a hand-counted 10.  The first row of the right-hand Pile must
    be the art's first row, and its last row the art's last.
    """
    tui.status_text.set_text("FIRSTROW")
    tui.drift_text.set_text("LASTROW")

    for cols, rows in TERMINAL_SIZES:
        _x, y, _width, height = tui._art_geometry(cols, rows)
        lines = render_text(tui.frame, cols, rows)

        first = find_marker(lines, "FIRSTROW")
        last = find_marker(lines, "LASTROW")
        assert first is not None and last is not None, f"missing at {cols}x{rows}"
        assert first[0] == y, f"{cols}x{rows}: info starts {first[0]}, art {y}"
        assert last[0] == y + height - 1, (
            f"{cols}x{rows}: info ends {last[0]}, art ends {y + height - 1}")


def test_the_right_column_starts_exactly_where_the_art_area_ends(tui):
    """
    The other edge of the rectangle, checked against the neighbouring cell.

    The right-hand column is padded one column in from the art cell, so `Artist:`
    must begin at `x + width + 1`.  Together with the test above this pins all
    four sides of the rectangle to rendered output.
    """
    tui.artist_text.set_text("Artist:  Bathory")

    for cols, rows in TERMINAL_SIZES:
        x, _y, width, _height = tui._art_geometry(cols, rows)
        lines = render_text(tui.frame, cols, rows)
        found = find_marker(lines, "Artist:")
        assert found is not None, f"artist row missing at {cols}x{rows}"
        assert found[1] == x + width + 1, (
            f"{cols}x{rows}: artist column {found[1]}, art ends at {x + width}")


def test_the_art_area_stays_within_the_now_playing_box(tui):
    """
    The reason the old code pinned the height to a constant at all: a landscape
    cover scaled to the full column width could otherwise run down over the seek
    bar and out of the Now Playing box into whatever sits below it.  Now the
    height *is* the art cell's rendered height, so the bound holds by
    construction — this checks it holds in fact.

    Phase 4 (G3) made the panel below the box the vibe cloud (the old console
    moved to a toggle), and at very small sizes the weighted cloud can be
    squeezed to nothing while the fixed-height console never was — so the honest,
    size-robust assertion is that the art clears the Now Playing box's own bottom
    border, the first `└` line in the render, which the packed box always draws.
    """
    for cols, rows in TERMINAL_SIZES:
        _x, y, _w, height = tui._art_geometry(cols, rows)
        lines = render_text(tui.frame, cols, rows)
        np_bottom = next(i for i, line in enumerate(lines)
                         if line.startswith("└"))
        assert y + height <= np_bottom, (
            f"{cols}x{rows}: art occupies rows {y}..{y + height - 1}, "
            f"Now Playing box closes at {np_bottom}")


def test_the_width_is_the_declared_art_column(tui):
    """ART_COLS is a declaration the Columns cell is built to, so it survives."""
    for cols, rows in TERMINAL_SIZES:
        assert tui._art_geometry(cols, rows)[2] == ART_COLS


def test_the_art_x_offset_is_one_not_two(tui):
    """
    The hand-counted constant was wrong, and this is the correction.

    Its comment read "col 0 = terminal left edge, col 1 = LineBox border, col 2 =
    art inner start" — but the LineBox's left border *is* column 0, so the art
    column starts at column 1 and every cover was drawn one column right of its
    box.  A constant nothing derives is a constant nothing can check.
    """
    assert tui._art_geometry(80, 40)[0] == 1


# ── The geometry follows the tree ────────────────────────────────────────────


def test_adding_a_row_to_the_right_column_grows_the_art_by_one_row(tui):
    """
    The whole point of H8: the layout can move without editing a constant.

    The old code pinned the height to `RIGHT_COL_ROWS = 10`, hand-counted from
    the widgets in this Pile.  Stage 3 alone takes it to 11.
    """
    before = tui._art_geometry(80, 40)
    tui.right_col.contents.append(
        (urwid.Text("one more row"), tui.right_col.options('pack')))
    after = tui._art_geometry(80, 40)

    assert after[3] == before[3] + 1
    assert after[:3] == before[:3], "only the height should have moved"


def test_adding_a_row_above_the_columns_moves_the_art_down(tui):
    before = tui._art_geometry(80, 40)
    tui.now_playing_content.contents.insert(
        0, (urwid.Text("banner"), tui.now_playing_content.options('pack')))
    after = tui._art_geometry(80, 40)

    assert after[1] == before[1] + 1
    assert (after[0], after[2], after[3]) == (before[0], before[2], before[3])


def test_a_taller_footer_does_not_move_the_art(tui):
    """
    The footer is a flow widget so it wraps rather than truncating (L9).  It sits
    below everything the art is measured from, so wrapping must not shift the
    image — but it does shrink the body, so this also proves the clip below is
    measured from the right place.
    """
    tall = tui._art_geometry(80, 40)
    tui.frame.footer = urwid.AttrMap(
        urwid.Text("a very long footer " * 12), "footer")
    assert tui.frame.footer.rows((80,)) > 1
    assert tui._art_geometry(80, 40)[:3] == tall[:3]


def test_the_geometry_is_independent_of_terminal_height(tui):
    """
    A consequence of packing the Now Playing box rather than weighting it: the
    panel is the same height on an 80x24 as on an 80x60, so the art never moves
    when the window is resized vertically.  Under the old weighted layout the
    position depended on the terminal height, which is what made the hardcoded
    `y = 3` correct at one size and wrong at others.
    """
    at_forty = tui._art_geometry(80, 40)
    for rows in (34, 45, 60, 100):
        assert tui._art_geometry(80, rows) == at_forty


# ── Refusing rather than smearing ────────────────────────────────────────────


def test_a_terminal_too_short_for_art_gets_none_rather_than_a_smear(tui):
    """
    An image squeezed into two rows is a smear across the seek bar, not album
    art.  `_render_art` clears the renderer in this case.
    """
    tiny = [tui._art_geometry(80, rows) for rows in range(1, 12)]
    assert any(g is None for g in tiny)
    for geometry in tiny:
        if geometry is not None:
            assert geometry[3] >= MIN_ART_ROWS
            assert geometry[2] >= MIN_ART_COLS


def test_zero_and_negative_sizes_are_refused(tui):
    assert tui._art_geometry(0, 40) is None
    assert tui._art_geometry(80, 0) is None
    assert tui._art_geometry(-5, -5) is None


def test_a_terminal_narrower_than_the_art_column_is_refused(tui):
    """ART_COLS is fixed, so a narrow terminal squeezes the *right* column to
    nothing rather than the art — but below the art column's own width there is
    no image to draw."""
    assert tui._art_geometry(6, 40) is None
