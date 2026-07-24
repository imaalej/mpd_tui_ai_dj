#!/usr/bin/env python3
"""
THROWAWAY — render the vibe cloud exactly as the terminal would (coloured
Braille, 2×4 dots per glyph), and bake the frames into an HTML "terminal window"
so the real thing can be seen without a tty.  Safe to delete.

    python3 render_terminal_frames.py            # writes vibe_terminal.html

This is NOT the webapp's smooth canvas — it is the honest cell grid: every
character is one Braille glyph, colour is per-cell (front point wins), and this
is what the TUI panel would actually show.
"""

import math
from pathlib import Path

import numpy as np

from vibe_cloud_demo import (load_centred_library, load_bank, project, colours,
                             simulate_session, rot, _BRAILLE)

OUT = Path(__file__).with_name("scratch_vibe_terminal.html")
COLS, ROWS = 92, 30          # terminal character cells
FRAMES = 44
TILT = 0.5


def render_cells(coords, rgb, comet, theta):
    dotW, dotH = 2 * COLS, 4 * ROWS
    scale = (min(dotW, dotH) / 2) / 3.0
    cx0, cy0 = dotW / 2, dotH / 2
    P = coords @ rot(theta, TILT).T
    order = np.argsort(P[:, 2])

    bits = {}
    col = {}                 # cell -> (rgb, depth)

    def plot(x, y, depth, colour, bump=0.0):
        dx = int(round(x * scale + cx0))
        dy = int(round(cy0 - y * scale))
        if not (0 <= dx < dotW and 0 <= dy < dotH):
            return
        cell = (dx // 2, dy // 4)
        bits[cell] = bits.get(cell, 0) | _BRAILLE[(dx % 2, dy % 4)]
        if cell not in col or depth + bump >= col[cell][1]:
            col[cell] = (colour, depth + bump)

    for i in order:
        plot(P[i, 0], P[i, 1], P[i, 2], tuple(int(v) for v in rgb[i]))

    # comet: interpolate along the path so the trail reads as a line
    C = comet @ rot(theta, TILT).T
    head = int(theta / (2 * math.pi) * len(C)) % len(C)
    trail = 24
    warm = (255, 214, 150)
    for k in range(trail, 0, -1):
        j = head - k
        if j < 1:
            continue
        a, b = C[j - 1], C[j]
        for s in np.linspace(0, 1, 4):
            p = a + (b - a) * s
            plot(p[0], p[1], p[2], warm, bump=10)
    hx, hy, hz = C[head]
    for ddx, ddy in ((0, 0), (0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5)):
        plot(hx + ddx / scale, hy + ddy / scale, hz, (255, 255, 255), bump=11)

    return bits, col


def frame_html(bits, col):
    rows = []
    for cy in range(ROWS):
        parts, cur = [], None
        for cx in range(COLS):
            cell = (cx, cy)
            if cell not in bits:
                if cur is not None:
                    parts.append("</span>"); cur = None
                parts.append(" ")
                continue
            ch = chr(0x2800 + bits[cell])
            r, g, b = col[cell][0]
            hexc = f"#{r:02x}{g:02x}{b:02x}"
            if hexc != cur:
                if cur is not None:
                    parts.append("</span>")
                parts.append(f'<span style="color:{hexc}">')
                cur = hexc
            parts.append(ch)
        if cur is not None:
            parts.append("</span>")
        rows.append("".join(parts))
    return "\n".join(rows)


def main():
    lib = load_centred_library(Path("data/embeddings/track_embeddings.npz"))
    bank = load_bank(Path("data/embeddings/descriptors.npz"))
    coords = project(lib, bank)
    rgb = colours(coords)
    comet = simulate_session(lib, coords)

    frames = []
    for f in range(FRAMES):
        theta = f / FRAMES * 2 * math.pi
        bits, col = render_cells(coords, rgb, comet, theta)
        frames.append(frame_html(bits, col))
    print(f"{FRAMES} frames, ~{sum(len(x) for x in frames)//1024} KB of glyph markup")

    payload = "[\n" + ",\n".join("`" + fr.replace("\\", "\\\\").replace("`", "\\`")
                                 + "`" for fr in frames) + "\n]"

    html = TEMPLATE.replace("__FRAMES__", payload).replace(
        "__COLS__", str(COLS)).replace("__ROWS__", str(ROWS))
    OUT.write_text(html)
    print("wrote", OUT, f"({len(html)//1024} KB)")


TEMPLATE = r"""<title>Vibe cloud — the terminal render</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  :root { color-scheme: dark;
    --bg:#05070b; --term:#080b11; --chrome:#12161d; --ink:#8b98a6;
    --accent:#4fe3c4; --mono: ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace; }
  * { box-sizing:border-box; }
  html,body { height:100%; }
  body { margin:0; background:
      radial-gradient(900px 500px at 50% -10%, #101a24 0%, transparent 60%), var(--bg);
    color:var(--ink); font-family:var(--mono); display:flex; flex-direction:column;
    align-items:center; justify-content:center; gap:18px; padding:24px; min-height:100vh; }
  .cap { max-width:720px; text-align:center; font-size:12.5px; line-height:1.6; color:var(--ink); }
  .cap b { color:var(--accent); font-weight:600; letter-spacing:.04em; }
  .cap .dim { color:#4a5661; }
  .window { border:1px solid #1b2734; border-radius:9px; overflow:hidden;
    box-shadow:0 24px 60px rgba(0,0,0,.5); background:var(--term); max-width:96vw; }
  .titlebar { display:flex; align-items:center; gap:8px; padding:8px 12px;
    background:var(--chrome); border-bottom:1px solid #1b2734; }
  .dot { width:11px; height:11px; border-radius:50%; }
  .dot.r{background:#e05a53;} .dot.y{background:#e0b24d;} .dot.g{background:#4fbf6b;}
  .tt { margin-left:8px; font-size:11px; color:#5c6b78; letter-spacing:.03em; }
  .screen { padding:12px 14px; background:var(--term); overflow-x:auto; }
  pre { margin:0; font-family:var(--mono); font-size:11px; line-height:1.02;
    letter-spacing:0; white-space:pre; color:#26313c; }
  .statusline { display:flex; justify-content:space-between; gap:16px; padding:6px 14px;
    background:var(--chrome); border-top:1px solid #1b2734; font-size:10.5px; color:#5c6b78;
    letter-spacing:.04em; }
  .statusline b { color:var(--accent); font-weight:500; }
  .controls { display:flex; gap:10px; align-items:center; }
  button { font-family:var(--mono); font-size:11px; color:var(--ink); background:#0d131c;
    border:1px solid #1b2734; border-radius:6px; padding:6px 12px; cursor:pointer;
    letter-spacing:.05em; }
  button:hover { color:#d6e3ee; border-color:var(--accent); }
  input[type=range]{ accent-color:var(--accent); }
  .lg { font-size:11px; color:#4a5661; }
</style>

<p class="cap">
  This is the <b>actual terminal rendering</b> — the same 674-track cloud, drawn
  in <b>Braille glyphs</b> (each character is a 2×4 dot cell) with 256-colour.
  Not the webapp's smooth canvas: this is what the TUI panel would show, at
  <span class="dim">__COLS__×__ROWS__ characters</span>.
</p>

<div class="window">
  <div class="titlebar">
    <span class="dot r"></span><span class="dot y"></span><span class="dot g"></span>
    <span class="tt">vibe_cloud_demo.py — live · __COLS__×__ROWS__</span>
  </div>
  <div class="screen"><pre id="screen"></pre></div>
  <div class="statusline">
    <span>♪ <b>Boards of Canada</b> — Roygbiv &nbsp; I <b>+0.4</b> · T <b>−0.3</b> · O <b>+1.1</b></span>
    <span>comet: simulated session · orbit 0.10×</span>
  </div>
</div>

<div class="controls">
  <button id="play">⏸ PAUSE</button>
  <label class="lg">fps <input type="range" id="fps" min="4" max="24" step="1" value="12" /></label>
  <span class="lg" id="fps-n">12</span>
</div>

<script>
const FRAMES = __FRAMES__;
const screen = document.getElementById("screen");
let i = 0, playing = true, fps = 12, acc = 0, last = performance.now();

function loop(now) {
  const dt = now - last; last = now;
  if (playing) {
    acc += dt;
    const step = 1000 / fps;
    while (acc >= step) { i = (i + 1) % FRAMES.length; acc -= step; }
    screen.innerHTML = FRAMES[i];
  }
  requestAnimationFrame(loop);
}
screen.innerHTML = FRAMES[0];
requestAnimationFrame(loop);

document.getElementById("play").onclick = (e) => {
  playing = !playing;
  e.target.textContent = playing ? "⏸ PAUSE" : "▶ PLAY";
};
const fpsEl = document.getElementById("fps"), fpsN = document.getElementById("fps-n");
fpsEl.oninput = () => { fps = +fpsEl.value; fpsN.textContent = fps; };
</script>
"""

if __name__ == "__main__":
    main()
