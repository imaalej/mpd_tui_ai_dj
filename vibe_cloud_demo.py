#!/usr/bin/env python3
"""
THROWAWAY prototype — the rotating vibe cloud, rendered in the terminal.

Not part of the app.  Safe to delete.  This is the faithful preview of what the
TUI widget would look like: the real library projected onto three legible mood
axes (Intensity · Tone · Organic, chosen by explore_mood_axes.py), drawn in
Braille, auto-rotating, with a simulated "session comet" tracing a trajectory.

    python3 vibe_cloud_demo.py --live               # animate until Ctrl-C
    python3 vibe_cloud_demo.py --frames 3 --no-color # static frames (for capture)
    python3 vibe_cloud_demo.py --export-json out.json

Honesty notes, carried from the design discussion:
  • Point *positions* are real: centred CLAP embeddings projected onto three
    axes built from descriptor poles, z-scored against this library.
  • Point *colours* are the same three numbers as HSV (Value←Intensity,
    Hue←Tone, Sat←Organic), so colour and position never disagree.
  • The camera orbits (ambient); the comet moves only along its path (data).
  • The comet here is a *simulated* nearest-neighbour walk, labelled as such —
    the real widget would drive it from the live session vector.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# Recommended by explore_mood_axes.py on this library: one of the collinear
# energy/saturation pair, plus the two independent axes.
AXES = {
    "Intensity": (["calm", "gentle", "mellow", "serene"],
                  ["aggressive", "intense", "frenetic", "energetic"]),
    "Tone":      (["melancholic", "sombre", "wistful", "menacing"],
                  ["uplifting", "joyful", "triumphant", "sunlit"]),
    "Organic":   (["acoustic", "orchestral", "guitar-driven", "piano-led"],
                  ["electronic", "synth-heavy"]),
}

# Braille: 2×4 dots per glyph, base U+2800.  Bit per (col, row).
_BRAILLE = {(0, 0): 0x01, (0, 1): 0x02, (0, 2): 0x04, (0, 3): 0x40,
            (1, 0): 0x08, (1, 1): 0x10, (1, 2): 0x20, (1, 3): 0x80}


def load_centred_library(path):
    with np.load(path, allow_pickle=False) as d:
        emb = np.asarray(d["embeddings"], dtype=np.float64)
        centroid = np.asarray(d["centroid"], dtype=np.float64)
    x = emb - centroid
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return x / n


def load_bank(path):
    with np.load(path, allow_pickle=False) as d:
        labels = [str(x) for x in d["labels"]]
        text = np.asarray(d["text_embeddings"], dtype=np.float64)
    n = np.linalg.norm(text, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return {lab: (text / n)[i] for i, lab in enumerate(labels)}


def axis_dir(low, high, bank):
    lo = np.mean([bank[w] for w in low if w in bank], axis=0)
    hi = np.mean([bank[w] for w in high if w in bank], axis=0)
    d = hi - lo
    return d / np.linalg.norm(d)


def project(library, bank):
    """(N,3) z-scored coordinates on the three axes."""
    cols = []
    for name, (low, high) in AXES.items():
        p = library @ axis_dir(low, high, bank)
        cols.append((p - p.mean()) / (p.std() + 1e-12))
    return np.column_stack(cols)


def hsv_to_rgb(h, s, v):
    i = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r, g, b = [(v, t, p), (q, v, p), (p, v, t),
               (p, q, v), (t, p, v), (v, p, q)][i]
    return int(r * 255), int(g * 255), int(b * 255)


def colours(coords):
    """Intensity→Value, Tone→Hue (cool↔warm), Organic→Saturation."""
    def unit(a, lo=-2.5, hi=2.5):
        return np.clip((a - lo) / (hi - lo), 0, 1)
    inten, tone, org = coords[:, 0], coords[:, 1], coords[:, 2]
    val = 0.45 + 0.55 * unit(inten)
    hue = 0.62 - 0.55 * unit(tone)            # dark→blue(0.62), bright→red/orange
    hue = np.mod(hue, 1.0)
    sat = 0.35 + 0.6 * unit(org)
    return np.array([hsv_to_rgb(h, s, v) for h, s, v in zip(hue, sat, val)])


def rot(theta, tilt):
    cy, sy, cx, sx = math.cos(theta), math.sin(theta), math.cos(tilt), math.sin(tilt)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return rx @ ry


def rgb_to_256(r, g, b):
    return 16 + 36 * round(r / 255 * 5) + 6 * round(g / 255 * 5) + round(b / 255 * 5)


def simulate_session(library, coords3, steps=60, seed=7):
    """A nearest-neighbour walk through the library — a stand-in trajectory."""
    rng = np.random.default_rng(seed)
    idx = int(rng.integers(len(library)))
    path = [idx]
    for _ in range(steps - 1):
        sims = library @ library[idx]
        sims[path[-6:]] = -2                      # avoid immediate backtracking
        nbrs = np.argpartition(sims, -8)[-8:]
        idx = int(rng.choice(nbrs))
        path.append(idx)
    return coords3[path]


def render(coords, cols256, W, H, theta, tilt, comet=None, use_colour=True):
    dotW, dotH = 2 * W, 4 * H
    scale = (min(dotW, dotH) / 2) / 3.0
    P = coords @ rot(theta, tilt).T
    order = np.argsort(P[:, 2])                    # far → near (painter's)

    bits = {}
    cellcol = {}
    cx0, cy0 = dotW / 2, dotH / 2

    def plot(x, y, depth, colour, bump=0):
        dx = int(round(x * scale + cx0))
        dy = int(round(cy0 - y * scale))
        if not (0 <= dx < dotW and 0 <= dy < dotH):
            return
        cellx, celly = dx // 2, dy // 4
        key = (cellx, celly)
        bits[key] = bits.get(key, 0) | _BRAILLE[(dx % 2, dy % 4)]
        if key not in cellcol or depth + bump >= cellcol[key][1]:
            cellcol[key] = (colour, depth + bump)

    for i in order:
        plot(P[i, 0], P[i, 1], P[i, 2], cols256[i])

    if comet is not None:
        C = comet @ rot(theta, tilt).T
        n = len(C)
        for j, (x, y, z) in enumerate(C):
            frac = (j + 1) / n
            head = j == n - 1
            colour = 231 if head else int(230 - (1 - frac) * 6)   # white→grey trail
            plot(x, y, z, colour, bump=10)                        # always on top
            if head:                                              # fatten the head
                for ddx, ddy in ((0.4, 0), (-0.4, 0), (0, 0.4), (0, -0.4)):
                    plot(x + ddx / scale, y + ddy / scale, z, 231, bump=10)

    lines = []
    for cy in range(H):
        row = []
        for cx in range(W):
            key = (cx, cy)
            if key not in bits:
                row.append(" ")
                continue
            ch = chr(0x2800 + bits[key])
            if use_colour:
                row.append(f"\x1b[38;5;{cellcol[key][0]}m{ch}")
            else:
                row.append(ch)
        lines.append("".join(row) + ("\x1b[0m" if use_colour else ""))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default="data/embeddings/track_embeddings.npz")
    ap.add_argument("--descriptors", default="data/embeddings/descriptors.npz")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--height", type=int, default=26)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--no-color", dest="colour", action="store_false")
    ap.add_argument("--export-json", default=None)
    args = ap.parse_args()

    library = load_centred_library(Path(args.embeddings))
    bank = load_bank(Path(args.descriptors))
    coords = project(library, bank)
    rgb = colours(coords)
    cols256 = np.array([rgb_to_256(*c) for c in rgb])
    comet = simulate_session(library, coords)

    if args.export_json:
        # Track identity for click-to-inspect.  Order matches `library`/`coords`.
        with np.load(Path(args.embeddings), allow_pickle=False) as d:
            names = [str(t) for t in d["track_files"]]

        def split(p):
            parts = p.split("/")
            title = parts[-1].rsplit(".", 1)[0]
            artist = parts[0] if len(parts) > 1 else ""
            return [artist, title]

        pts = [{"x": float(coords[i, 0]), "y": float(coords[i, 1]),
                "z": float(coords[i, 2]),
                "r": int(rgb[i, 0]), "g": int(rgb[i, 1]), "b": int(rgb[i, 2])}
               for i in range(len(coords))]
        meta = [split(n) for n in names]
        path = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
                for p in comet]
        Path(args.export_json).write_text(json.dumps(
            {"axes": list(AXES), "points": pts, "meta": meta, "comet": path}))
        print(f"wrote {args.export_json}: {len(pts)} points, {len(path)} comet steps")
        return

    tilt = 0.5
    W, H = args.width, args.height
    if args.live:
        try:
            print("\x1b[2J\x1b[?25l", end="")
            t0 = time.time()
            while True:
                theta = (time.time() - t0) * 0.6
                cpos = int((time.time() - t0) * 4) % len(comet)
                trail = comet[max(0, cpos - 10):cpos + 1]
                lines = render(coords, cols256, W, H, theta, tilt,
                               comet=trail, use_colour=args.colour)
                print("\x1b[H" + "\n".join(lines))
                print("\x1b[0m  Intensity → x · Tone → colour · Organic → saturation"
                      "   (simulated comet)   Ctrl-C to quit")
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\x1b[?25h\x1b[0m")
            return

    for f in range(args.frames):
        theta = f / max(1, args.frames) * 2 * math.pi
        cpos = int((f + 1) / args.frames * len(comet))
        trail = comet[max(0, cpos - 10):cpos]
        print(f"\n── frame {f + 1}/{args.frames}  (rotation {math.degrees(theta):.0f}°)")
        for ln in render(coords, cols256, W, H, theta, tilt,
                         comet=trail, use_colour=args.colour):
            print(ln)


if __name__ == "__main__":
    main()
