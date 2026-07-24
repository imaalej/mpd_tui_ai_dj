// THROWAWAY — prove the JS preview draws what the Python renderer draws.
//
//     node explore_cloud_preview/verify.mjs
//
// Diffs every cell of every reference pose: dot bits, colour and glyph.  A
// preview that has drifted from `vibe_cloud.compute_frame` is worse than no
// preview, because the whole point is to judge the real thing.

import { readFileSync } from "node:fs";
import { computeFrame } from "./core.js";

// Sources live here and tracked; what the explorer renders lives in
// `scratch_scaffold/` and is gitignored.
const out = new URL("../scratch_scaffold/", import.meta.url).pathname;
const data = JSON.parse(readFileSync(out + "data.json", "utf8"));
const ref = JSON.parse(readFileSync(out + "reference.json", "utf8"));

let worst = null;
let totalCells = 0, totalBad = 0;
// Classify, don't just count: "0.06% of cells differ" is useless without
// knowing whether a dot moved (a real divergence) or a channel landed one level
// apart on a truncation boundary (invisible).
let bitsBad = 0, glyphBad = 0, maxChannel = 0;
for (const f of ref.frames) {
  const got = computeFrame(data, ref.cols, ref.rows, f.azimuth, ref.tilt, 1.0,
                           f.mode, { currentIdx: 12 });
  let bad = 0, firstBad = null;
  for (let i = 0; i < f.bits.length; i++) {
    const same = got.bits[i] === f.bits[i] && got.color[i] === f.color[i]
                 && got.glyph[i] === f.glyph[i];
    if (!same) {
      bad++;
      if (got.bits[i] !== f.bits[i]) bitsBad++;
      if (got.glyph[i] !== f.glyph[i]) glyphBad++;
      if (got.color[i] !== f.color[i] && got.color[i] >= 0 && f.color[i] >= 0)
        for (const sh of [16, 8, 0])
          maxChannel = Math.max(maxChannel,
            Math.abs(((got.color[i] >> sh) & 255) - ((f.color[i] >> sh) & 255)));
      if (!firstBad)
        firstBad = { cell: i, py: [f.bits[i], f.color[i], f.glyph[i]],
                     js: [got.bits[i], got.color[i], got.glyph[i]] };
    }
  }
  totalCells += f.bits.length;
  totalBad += bad;
  const pct = (100 * bad) / f.bits.length;
  if (!worst || pct > worst.pct)
    worst = { mode: f.mode, azimuth: f.azimuth, pct, bad, firstBad };
}
const pct = (100 * totalBad) / totalCells;
console.log(`${ref.frames.length} poses · ${totalCells} cells · `
            + `${totalBad} differing (${pct.toFixed(4)}%)`);
console.log(`  dots differing: ${bitsBad} · glyphs differing: ${glyphBad}`
            + ` · largest colour-channel gap: ${maxChannel}/255`);
if (bitsBad || glyphBad || maxChannel > 1) {
  console.log("DIVERGED — worst pose:", JSON.stringify(worst, null, 1));
  process.exit(1);
}
console.log("Every dot and glyph identical; colour within one level "
            + "(float64 truncation boundaries) ✓");
