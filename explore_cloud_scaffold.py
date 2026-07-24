#!/usr/bin/env python3
"""
THROWAWAY — look at the vibe cloud's candidate spatial scaffolds without a tty.

The question: orbiting the cloud is hard to read because a field of specks has no
reference frame.  Does bounding it (a dim cube), or naming it (a ruled 3-axis
frame), make the rotation legible — and does either survive Braille's 2×4 grid?

    python3 explore_cloud_scaffold.py               # stills + the preview's data
    node    explore_cloud_preview/verify.mjs        # prove the port matches Python
    python3 explore_cloud_preview/build_preview.py  # one self-contained HTML page

Everything is rendered through the **real** `vibe_cloud.compute_frame`, so what
this shows is what the TUI panel draws — not a prettier stand-in.  Outputs land in
`scratch_scaffold/` (gitignored by the `scratch_` prefix); the browser preview's
sources are tracked, in `explore_cloud_preview/`.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from mood_axes import MoodAxes                                    # noqa: E402
from vibe_cloud import (BRAILLE_BASE, SCAFFOLD_MODES, compute_frame,  # noqa: E402
                        point_rgb)

OUT = Path(__file__).parent / "scratch_scaffold"

# The *real* terminal background (§11: Alacritty, Aura theme, `#15141b`), not the
# near-black the archived browser mock used.  A still rendered on #05070b flatters
# every dim colour in the frame by a factor of three and hides the one question
# that matters — whether the faint end of the scaffold clears the background at
# all.  Override with --bg to check another theme.
BG = (0x15, 0x14, 0x1b)

# The presets to compare, plus a couple of combinations the cycle list does not
# carry — the point of an exploration is to see more than the shipping options.
MODES = list(SCAFFOLD_MODES) + ["gnomon", "corners", "cage", "walls",
                                "cage+gnomon", "corners+floor+shadow",
                                "walls+floor+shadow"]


def load_cloud():
    from embeddings_io import centre
    axes = MoodAxes.load()
    if axes is None:
        sys.exit("No mood axes in descriptors.npz — "
                 "python3 src/generate_embeddings.py --descriptors-only")
    with np.load(Path("data/embeddings/track_embeddings.npz"),
                 allow_pickle=False) as data:
        centred = centre(data["embeddings"], data["centroid"])
    coords = axes.coordinates(centred)
    return coords, point_rgb(coords), axes.labels


def simulate_comet(coords, steps=40, seed=7):
    """A short walk through nearby points — a stand-in session trajectory."""
    rng = np.random.default_rng(seed)
    idx = int(rng.integers(len(coords)))
    path = [idx]
    for _ in range(steps - 1):
        d = np.linalg.norm(coords - coords[idx], axis=1)
        d[path[-6:]] = 1e9
        idx = int(rng.choice(np.argpartition(d, 8)[:8]))
        path.append(idx)
    return coords[path]


def cells(frame, cols, rows):
    """Frame → [[(char, (r,g,b)) or None per cell]] — the honest cell grid."""
    grid = []
    for cy in range(rows):
        row = []
        for cx in range(cols):
            bits = int(frame.bits[cy, cx])
            over = int(frame.glyph[cy, cx])
            if bits == 0 and over < 0:
                row.append(None)
                continue
            code = int(frame.color[cy, cx])
            code = 0xFFFFFF if code < 0 else code
            ch = chr(over) if over >= 0 else chr(BRAILLE_BASE + bits)
            row.append((ch, ((code >> 16) & 0xFF, (code >> 8) & 0xFF, code & 0xFF)))
        grid.append(row)
    return grid


# A mono font that actually has U+2800–U+28FF.  DejaVu Sans **Mono** does not —
# it renders every Braille cell as tofu, which is worth knowing before trusting
# any screenshot of this cloud taken outside the real terminal.
PNG_FONT = "/home/gumibo/.local/share/fonts/IosevkaNerdFontMono-Regular.ttf"


def to_png(grid, path, font_px=22, bg=None):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(PNG_FONT, font_px)
    # Cell size from the font's own metrics, so the mock has the terminal's
    # aspect ratio (a Braille dot is 2× taller than wide) rather than a guess.
    cell_px = (int(round(font.getlength("M"))), int(round(font_px * 1.16)))
    w, h = cell_px[0] * len(grid[0]), cell_px[1] * len(grid)
    img = Image.new("RGB", (w, h), bg or BG)
    draw = ImageDraw.Draw(img)
    for cy, row in enumerate(grid):
        for cx, cell in enumerate(row):
            if cell is None:
                continue
            ch, rgb = cell
            draw.text((cx * cell_px[0], cy * cell_px[1]), ch, font=font, fill=rgb)
    img.save(path)
    return path


def to_html_frame(grid):
    out = []
    for row in grid:
        parts, cur = [], None
        for cell in row:
            if cell is None:
                if cur is not None:
                    parts.append("</span>")
                    cur = None
                parts.append(" ")
                continue
            ch, (r, g, b) = cell
            hexc = f"#{r:02x}{g:02x}{b:02x}"
            if hexc != cur:
                if cur is not None:
                    parts.append("</span>")
                parts.append(f'<span style="color:{hexc}">')
                cur = hexc
            parts.append("&lt;" if ch == "<" else "&amp;" if ch == "&" else ch)
        if cur is not None:
            parts.append("</span>")
        out.append("".join(parts))
    return "\n".join(out)


def export_data(coords, rgb, comet, labels, cols, rows, tilt):
    """The cloud as JSON, plus reference frames the JS preview is checked against.

    The preview re-implements `compute_frame` in the browser so the rotation can
    be dragged rather than watched.  A re-implementation is exactly the kind of
    mirror this project distrusts (§8: a double built from the same assumption
    reproduces the bug and passes), so it ships with reference frames: `verify.mjs`
    renders the same poses in JS and diffs them cell by cell against these.
    """
    import json
    refs = []
    for mode in MODES:
        for theta in (0.0, 0.9, 2.7):
            f = compute_frame(coords, rgb, cols, rows, theta, tilt, zoom=1.0,
                              comet=comet, current_idx=12, scaffold=mode,
                              axis_labels=labels)
            refs.append({"mode": mode, "azimuth": theta,
                         "bits": f.bits.ravel().tolist(),
                         "color": f.color.ravel().tolist(),
                         "glyph": f.glyph.ravel().tolist()})
    (OUT / "reference.json").write_text(json.dumps(
        {"cols": cols, "rows": rows, "tilt": tilt, "frames": refs}))
    (OUT / "data.json").write_text(json.dumps({
        # Full precision, not rounded: a 1e-4 nudge moves a point across a
        # dot-rounding boundary about twice per frame, and then the verify diff
        # blames the algorithm for what is really a lossy export.
        "coords": [[float(v) for v in p] for p in coords],
        "rgb": [[int(v) for v in c] for c in rgb],
        "comet": [[float(v) for v in p] for p in comet],
        "labels": list(labels),
        "modes": MODES,
    }))
    print("wrote data.json + reference.json "
          f"({len(refs)} reference frames, {len(MODES)} modes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=96)
    ap.add_argument("--rows", type=int, default=30)
    ap.add_argument("--frames", type=int, default=4, help="unused; kept for scripts")
    ap.add_argument("--tilt", type=float, default=0.5)
    ap.add_argument("--bg", default=None,
                    help="terminal background as #rrggbb (default: the Aura bg)")
    args = ap.parse_args()

    global BG
    if args.bg:
        v = int(args.bg.lstrip("#"), 16)
        BG = ((v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF)

    coords, rgb, labels = load_cloud()
    comet = simulate_comet(coords)
    print(f"{len(coords)} points · axes {' · '.join(labels)}")
    OUT.mkdir(exist_ok=True)

    def frame_at(mode, theta):
        return compute_frame(coords, rgb, args.cols, args.rows, theta, args.tilt,
                             zoom=1.0, comet=comet, current_idx=12,
                             scaffold=mode, axis_labels=labels)

    # Stills for eyeballing: one mid-orbit angle per mode, plus a two-angle strip
    # for the two most promising, because the whole question is what *rotation*
    # looks like.
    for mode in MODES:
        grid = cells(frame_at(mode, 0.9), args.cols, args.rows)
        to_png(grid, OUT / f"scaffold-{mode.replace('+', '_')}.png")
    print(f"wrote {len(MODES)} stills to {OUT}/")

    export_data(coords, rgb, comet, labels, args.cols, args.rows, args.tilt)


if __name__ == "__main__":
    main()
