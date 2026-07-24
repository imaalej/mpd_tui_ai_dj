// THROWAWAY — a browser port of vibe_cloud.compute_frame, so the candidate
// scaffolds can be *dragged* rather than watched as a baked loop.
//
// This is a mirror of Python, which this project distrusts on principle, so it
// is checked rather than trusted: `verify.mjs` renders the poses in
// reference.json here and diffs every cell against what the real renderer
// produced.  If the two ever disagree, the preview is wrong, not the app.

export const BRAILLE_LUT = [0x01, 0x02, 0x04, 0x40, 0x08, 0x10, 0x20, 0x80];
export const SHADE_MIN = 0.4;
export const BASE_HALF_EXTENT = 3.0;
export const SCAFFOLD_EXTENT = 2.0;
export const SCAFFOLD_MARGIN = 1.04;
export const SCAFFOLD_EXTENT_FLOOR = 0.5;
const WORLD_SCAFFOLDS = ["cage", "corners", "walls", "floor", "triad", "shadow"];
export const SCAFFOLD_DIV = 4;
export const SCAFFOLD_SPACING = 2.0;
export const SHADOW_WEIGHT = 0.55;
export const GNOMON_CELLS = 5;
export const GNOMON_MARGIN = 3;
// Derived in Python from the terminal background (see `_lift`) and pasted here;
// verify.mjs fails loudly if the two ever drift.
export const COL_SCAFFOLD_FAR = 0x292931;
export const COL_SCAFFOLD_NEAR = 0x777d87;
export const COL_SCAFFOLD_LABEL = 0x8a929c;
export const COL_COMET_HEAD = 0xffffff;
export const COL_COMET_TRAIL = [0xffd7af, 0xffaf87, 0xff875f, 0xff5f00];
export const COL_CURRENT = 0x00ffff;
export const COL_SELECTED = 0xff00ff;

export function rotationMatrix(azimuth, tilt) {
  const cy = Math.cos(azimuth), sy = Math.sin(azimuth);
  const cx = Math.cos(tilt), sx = Math.sin(tilt);
  // rx @ ry, written out.
  return [
    [cy, 0, sy],
    [sx * sy, cx, -sx * cy],
    [-cx * sy, sx, cx * cy],
  ];
}

// numpy rounds half to **even**; JS rounds half up.  One dot in a few hundred
// lands exactly on .5, and that is enough to make the verify diff non-zero and
// waste an hour — so match numpy here rather than explain the mismatch later.
function rint(v) {
  const f = Math.floor(v);
  if (v - f !== 0.5) return Math.round(v);
  return f % 2 === 0 ? f : f + 1;
}

function project(p, R) {
  return [
    p[0] * R[0][0] + p[1] * R[0][1] + p[2] * R[0][2],
    p[0] * R[1][0] + p[1] * R[1][1] + p[2] * R[1][2],
    p[0] * R[2][0] + p[1] * R[2][1] + p[2] * R[2][2],
  ];
}

function pack(r, g, b) {
  // `_rgb_to_packed` clips then `.astype(np.int32)` — which **truncates**.
  // Rounding here would drift one level per channel against the real render.
  const c = (v) => Math.trunc(Math.min(255, Math.max(0, v)));
  return (c(r) << 16) | (c(g) << 8) | c(b);
}

function shadeBetween(t, far, near) {
  const lo = [(far >> 16) & 255, (far >> 8) & 255, far & 255];
  const hi = [(near >> 16) & 255, (near >> 8) & 255, near & 255];
  return pack(lo[0] + (hi[0] - lo[0]) * t,
              lo[1] + (hi[1] - lo[1]) * t,
              lo[2] + (hi[2] - lo[2]) * t);
}

// ── scaffold geometry (mirrors the Python segment builders) ──────────────────

// The box is measured off the cloud, one extent per axis — a cube would be
// mostly empty volume on a lopsided library, and the box is fitted to the panel,
// so that emptiness is paid for in how small the cloud gets drawn.
export function libraryExtent(coords, margin = SCAFFOLD_MARGIN) {
  if (!coords || !coords.length) return [SCAFFOLD_EXTENT, SCAFFOLD_EXTENT, SCAFFOLD_EXTENT];
  const e = [0, 0, 0];
  for (const p of coords)
    for (let k = 0; k < 3; k++) e[k] = Math.max(e[k], Math.abs(p[k]));
  return e.map((v) => Math.max(v * margin, SCAFFOLD_EXTENT_FLOOR));
}

function asExtent(extent, coords) {
  if (extent == null) return libraryExtent(coords);
  return typeof extent === "number" ? [extent, extent, extent] : extent;
}

function cubeEdges(e) {
  const corners = [];
  for (const x of [-e[0], e[0]]) for (const y of [-e[1], e[1]]) for (const z of [-e[2], e[2]])
    corners.push([x, y, z]);
  const segs = [];
  for (let i = 0; i < corners.length; i++)
    for (let j = i + 1; j < corners.length; j++) {
      let diff = 0;
      for (let k = 0; k < 3; k++)
        if (Math.abs(corners[i][k] - corners[j][k]) > 1e-9) diff++;
      if (diff === 1) segs.push([corners[i], corners[j]]);
    }
  return segs;
}

function cornerBrackets(e, fraction = 0.28) {
  const segs = [];
  for (const x of [-e[0], e[0]]) for (const y of [-e[1], e[1]]) for (const z of [-e[2], e[2]]) {
    const c = [x, y, z];
    for (let k = 0; k < 3; k++) {
      const end = c.slice();
      end[k] += (c[k] > 0 ? -1 : 1) * 2 * e[k] * fraction;
      segs.push([c, end]);
    }
  }
  return segs;
}

function planeGrid(axis, value, e, div) {
  const others = [0, 1, 2].filter((k) => k !== axis);
  const segs = [];
  for (const [fixed, varying] of [others, [others[1], others[0]]]) {
    for (let i = 0; i <= div; i++) {
      // numpy's linspace: step-multiply with an exact endpoint.
      const step = (2 * e[fixed]) / div;
      const s = i === div ? e[fixed] : -e[fixed] + step * i;
      const a = [0, 0, 0], b = [0, 0, 0];
      a[axis] = b[axis] = value;
      a[fixed] = b[fixed] = s;
      a[varying] = -e[varying]; b[varying] = e[varying];
      segs.push([a, b]);
    }
  }
  return segs;
}

function farWalls(R, e, div) {
  let segs = [];
  for (let axis = 0; axis < 3; axis++) {
    const sign = R[2][axis] > 0 ? -1 : 1;
    segs = segs.concat(planeGrid(axis, sign * e[axis], e, div));
  }
  return segs;
}

function axisTriad(e, tick = 0.12) {
  const segs = [];
  for (let axis = 0; axis < 3; axis++) {
    const a = [0, 0, 0], b = [0, 0, 0];
    a[axis] = -e[axis]; b[axis] = e[axis];
    segs.push([a, b]);
    for (let t = -Math.floor(e[axis]); t <= Math.floor(e[axis]); t++) {
      if (Math.abs(t) < 1e-9) continue;
      for (const cross of [0, 1, 2].filter((k) => k !== axis)) {
        const p = [0, 0, 0], q = [0, 0, 0];
        p[axis] = q[axis] = t;
        p[cross] = -tick; q[cross] = tick;
        segs.push([p, q]);
      }
    }
  }
  return segs;
}

function sampleSegments(segs, scale, spacing) {
  const pts = [];
  for (const [p, q] of segs) {
    const dx = q[0] - p[0], dy = q[1] - p[1], dz = q[2] - p[2];
    // `Math.hypot` is *more* accurate than numpy's sqrt-of-sum-of-squares, and
    // one ulp is enough to flip `n` and shift a whole line by a dot.  Mirror
    // numpy: same norm, same linspace (step multiply, exact endpoint).
    const length = Math.sqrt(dx * dx + dy * dy + dz * dz);
    const n = Math.max(2, Math.trunc((length * scale) / Math.max(0.25, spacing)) + 1);
    const step = 1 / (n - 1);
    for (let i = 0; i < n; i++) {
      const t = i === n - 1 ? 1 : i * step;
      pts.push([p[0] + dx * t, p[1] + dy * t, p[2] + dz * t]);
    }
  }
  return pts;
}

export function scaffoldPoints(mode, R, scale, coords, extent) {
  const tokens = new Set(String(mode).split("+").filter((t) => t && t !== "off"));
  if (!tokens.size) return { pts: [], weights: [] };
  const e = asExtent(extent, coords);
  let segs = [];
  if (tokens.has("corners")) segs = segs.concat(cornerBrackets(e));
  if (tokens.has("cage")) segs = segs.concat(cubeEdges(e));
  if (tokens.has("walls")) segs = segs.concat(farWalls(R, e, SCAFFOLD_DIV));
  if (tokens.has("floor")) segs = segs.concat(planeGrid(1, -e[1], e, SCAFFOLD_DIV));
  if (tokens.has("triad")) segs = segs.concat(axisTriad(e));

  const pts = segs.length ? sampleSegments(segs, scale, SCAFFOLD_SPACING) : [];
  const weights = pts.map(() => 1.0);
  if (tokens.has("shadow") && coords && coords.length) {
    for (const p of coords) { pts.push([p[0], -e[1], p[2]]); weights.push(SHADOW_WEIGHT); }
  }
  return { pts, weights };
}

export function gnomonDots(R, cols, rows, labels) {
  const arm = GNOMON_CELLS * 4;
  const x0 = GNOMON_MARGIN * 2 + arm;
  const y0 = 4 * rows - GNOMON_MARGIN * 4 - arm;
  const dots = [], text = [];
  const names = labels && labels.length ? labels : ["x", "y", "z"];
  for (let axis = 0; axis < 3; axis++) {
    const vx = R[0][axis], vy = R[1][axis], vz = R[2][axis];
    const colour = shadeBetween((vz + 1) / 2, COL_SCAFFOLD_FAR, COL_SCAFFOLD_NEAR);
    const step = 1 / (arm - 1);
    for (let i = 0; i < arm; i++) {
      const t = i === arm - 1 ? 1 : i * step;
      dots.push([rint(x0 + vx * arm * t), rint(y0 - vy * arm * t), colour]);
    }
    text.push([Math.floor(rint(x0 + vx * (arm + 3)) / 2),
               Math.floor(rint(y0 - vy * (arm + 3)) / 4),
               (names[axis] || "?").slice(0, 1).toUpperCase()]);
  }
  return { dots, text };
}

// ── the frame ────────────────────────────────────────────────────────────────

export function computeFrame(data, cols, rows, azimuth, tilt, zoom, scaffold, opts = {}) {
  const n = cols * rows;
  const bits = new Uint8Array(n);
  const color = new Int32Array(n).fill(-1);
  const glyph = new Int32Array(n).fill(-1);
  if (cols <= 0 || rows <= 0) return { bits, color, glyph, cols, rows };

  const dotW = 2 * cols, dotH = 4 * rows;
  const halfDots = Math.min(dotW, dotH) / 2;   // base scale only
  let scale = halfDots / BASE_HALF_EXTENT;
  const cx0 = dotW / 2, cy0 = dotH / 2;
  const tokens = new Set(String(scaffold).split("+").filter((t) => t && t !== "off"));
  const ext = asExtent(opts.extent, data.coords);
  if (WORLD_SCAFFOLDS.some((t) => tokens.has(t))) {
    // Worst case over azimuth, not this frame's — a scale that tracked the
    // current angle would make the cloud pulse as it spins.
    const diagonal = Math.hypot(ext[0], ext[2]);
    const reachX = Math.max(diagonal, 1e-6);
    const reachY = Math.max(Math.abs(Math.cos(tilt)) * ext[1]
                            + Math.abs(Math.sin(tilt)) * diagonal, 1e-6);
    scale = Math.min((dotW / 2 - 1) / reachX, (dotH / 2 - 1) / reachY);
  }
  scale *= zoom;
  const R = rotationMatrix(azimuth, tilt);

  // Every dot as {x, y, depth, colour} — one array, one stable sort, exactly
  // like the numpy path's concatenate + argsort(kind="stable").
  const D = [];
  const push = (x, y, depth, colour) => D.push([x, y, depth, colour]);
  const splat = (p, colour, bump) =>
    push(rint(p[0] * scale + cx0), rint(cy0 - p[1] * scale),
         p[2] + bump, colour);

  let gtext = [];
  if (tokens.has("gnomon")) {
    const g = gnomonDots(R, cols, rows, data.labels);
    for (const [x, y, c] of g.dots) push(x, y, -99, c);
    gtext = g.text;
  }

  const { pts, weights } = scaffoldPoints(scaffold, R, scale, data.coords, ext);
  if (pts.length) {
    const proj = pts.map((p) => project(p, R));
    let lo = Infinity, hi = -Infinity;
    for (const p of proj) { if (p[2] < lo) lo = p[2]; if (p[2] > hi) hi = p[2]; }
    const span = hi - lo;
    proj.forEach((p, i) => {
      const t = span > 1e-9 ? (p[2] - lo) / span : 0.5;
      splat(p, shadeBetween(t * weights[i], COL_SCAFFOLD_FAR, COL_SCAFFOLD_NEAR), -100);
    });
  }

  const proj = data.coords.map((p) => project(p, R));
  let lo = Infinity, hi = -Infinity;
  for (const p of proj) { if (p[2] < lo) lo = p[2]; if (p[2] > hi) hi = p[2]; }
  const span = hi - lo;
  proj.forEach((p, i) => {
    const t = span > 1e-9 ? (p[2] - lo) / span : 0.5;
    const f = span > 1e-9 ? SHADE_MIN + (1 - SHADE_MIN) * t : 1.0;
    const c = data.rgb[i];
    splat(p, pack(c[0] * f, c[1] * f, c[2] * f), 0);
  });

  if (opts.currentIdx != null && opts.currentIdx >= 0 && opts.currentIdx < proj.length)
    splat(proj[opts.currentIdx], COL_CURRENT, 8);
  if (opts.selectedIdx != null && opts.selectedIdx >= 0 && opts.selectedIdx < proj.length)
    splat(proj[opts.selectedIdx], COL_SELECTED, 9);

  if (data.comet && data.comet.length) {
    const m = data.comet.length;
    data.comet.forEach((p, j) => {
      const frac = j / Math.max(1, m - 1);
      let c = COL_COMET_TRAIL[Math.min(COL_COMET_TRAIL.length - 1,
                                       Math.trunc(frac * COL_COMET_TRAIL.length))];
      if (j === m - 1) c = COL_COMET_HEAD;
      splat(project(p, R), c, 10);
    });
  }

  const inside = D.filter(([x, y]) => x >= 0 && x < dotW && y >= 0 && y < dotH);
  for (const [x, y] of inside)
    bits[Math.trunc(y / 4) * cols + Math.trunc(x / 2)] |=
      BRAILLE_LUT[(x % 2) * 4 + (y % 4)];
  // Stable sort far → near, then write: the last write per cell is the nearest.
  inside.map((d, i) => [d, i])
    .sort((a, b) => (a[0][2] - b[0][2]) || (a[1] - b[1]))
    .forEach(([[x, y, , c]]) => {
      color[Math.trunc(y / 4) * cols + Math.trunc(x / 2)] = c;
    });

  for (const [c, r, ch] of gtext)
    if (c >= 0 && c < cols && r >= 0 && r < rows) {
      glyph[r * cols + c] = ch.codePointAt(0);
      color[r * cols + c] = COL_SCAFFOLD_LABEL;
    }
  const boxed = new Set([...tokens].filter((t) => t !== "gnomon"));
  if (data.labels && boxed.size) {
    for (let axis = 0; axis < 3 && axis < data.labels.length; axis++) {
      const tip = [0, 0, 0];
      tip[axis] = ext[axis] * 1.10;
      const p = project(tip, R);
      let col = Math.floor(rint(p[0] * scale + cx0) / 2);
      const row = Math.floor(rint(cy0 - p[1] * scale) / 4);
      const text = String(data.labels[axis]).slice(0, 10);
      col = Math.max(0, Math.min(cols - text.length, col - Math.floor(text.length / 2)));
      if (row < 0 || row >= rows || cols < text.length) continue;
      for (let i = 0; i < text.length; i++) {
        glyph[row * cols + col + i] = text.codePointAt(i);
        color[row * cols + col + i] = COL_SCAFFOLD_LABEL;
      }
    }
  }
  return { bits, color, glyph, cols, rows };
}
