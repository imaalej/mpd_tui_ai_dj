#!/usr/bin/env python3
"""
THROWAWAY exploration — pick three "mood axes" for the vibe point cloud.

Not part of the app.  Safe to delete.  Run it whenever the library changes:

    python3 explore_mood_axes.py
    python3 explore_mood_axes.py --embeddings data/embeddings/track_embeddings.npz \
                                 --descriptors data/embeddings/descriptors.npz

What it answers
---------------
The vibe cloud needs a 3-D coordinate frame that is (a) legible — each axis is a
named perceptual dimension, not an abstract principal component — and (b) a real
3-D shape rather than a pancake, i.e. the three axes must actually be
*decorrelated over this library*.  Intensity and Saturation are the usual
collision risk (loud music tends to be dense/distorted).

Every axis is a genuine direction in the same centred CLAP space the app scores
in:  axis = normalise( mean(text-vecs of the HIGH pole) − mean(text-vecs of the
LOW pole) ).  Projecting a centred track onto it is exactly the descriptor
operation the app already trusts (`DescriptorBank.z_scores`), just expressed as a
difference of two poles.  So this is not a new model — it is the bank, read as a
compass.

Because the axis directions come from the (fixed) text tower while the spread,
the z-score calibration and the *choice of the best triad* are measured against
whatever library is on disk, this is inherently per-user and re-fits itself when
the library grows.  In production the chosen axes + their mean/std would be
computed once at embedding-generation time and stored beside the descriptors, so
the runtime just reads them.  This script is the offline proof of that path.

Output: per-axis signal strength, the full correlation matrix, the auto-selected
most-orthogonal triad, the hand-picked Intensity/Tone/Saturation triad for
comparison, how flat each cloud is, and small ASCII scatters to eyeball it.
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np


# ── Candidate axes, poles drawn only from the descriptor bank's own words ──────
#
# (low pole → high pole).  Missing words are dropped with a warning; an axis
# survives as long as each pole keeps at least one word.
CANDIDATE_AXES = {
    "Intensity":  (["calm", "gentle", "mellow", "serene"],
                   ["aggressive", "intense", "frenetic", "energetic"]),
    "Tone":       (["melancholic", "sombre", "wistful", "menacing"],
                   ["uplifting", "joyful", "triumphant", "sunlit"]),
    "Saturation": (["sparse", "spacious", "polished"],
                   ["dense", "distorted", "gritty"]),
    "Organic":    (["acoustic", "orchestral", "guitar-driven", "piano-led"],
                   ["electronic", "synth-heavy"]),
    "Warmth":     (["cold"],
                   ["warm"]),
    "Groove":     (["free-time", "halftime"],
                   ["danceable", "groovy", "motorik"]),
    "Brightness": (["nocturnal", "cavernous"],
                   ["sunlit", "shimmering"]),
    "Drama":      (["intimate"],
                   ["cinematic", "cavernous"]),
}

# The triad the humans currently like — reported alongside the auto pick.
PREFERRED = ("Intensity", "Tone", "Saturation")


def load_centred_library(path: Path):
    """Replicate the app's load: subtract the stored centroid, re-normalise."""
    with np.load(path, allow_pickle=False) as data:
        emb = np.asarray(data["embeddings"], dtype=np.float64)
        centroid = np.asarray(data["centroid"], dtype=np.float64)
    x = emb - centroid
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return x / norms


def load_bank(path: Path):
    """Return {label: unit text-tower vector}."""
    with np.load(path, allow_pickle=False) as data:
        labels = [str(x) for x in data["labels"]]
        text = np.asarray(data["text_embeddings"], dtype=np.float64)
    norms = np.linalg.norm(text, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    text = text / norms
    return {lab: text[i] for i, lab in enumerate(labels)}


def build_axis(name, low, high, bank):
    """Unit direction for one axis, or None if a pole has no words in the bank."""
    def pole(words):
        vecs = [bank[w] for w in words if w in bank]
        missing = [w for w in words if w not in bank]
        if missing:
            print(f"  [{name}] not in bank, skipped: {', '.join(missing)}",
                  file=sys.stderr)
        return (np.mean(vecs, axis=0) if vecs else None)

    lo, hi = pole(low), pole(high)
    if lo is None or hi is None:
        print(f"  [{name}] dropped — a pole ended up empty", file=sys.stderr)
        return None
    d = hi - lo
    n = np.linalg.norm(d)
    return d / n if n > 1e-12 else None


def ascii_scatter(x, y, xlabel, ylabel, w=46, h=15):
    """A tiny density scatter of two z-scored coordinates, clamped to ±3 σ."""
    ramp = " .:-=+*#%@"
    gx = np.clip((x + 3) / 6, 0, 1) * (w - 1)
    gy = np.clip((3 - y) / 6, 0, 1) * (h - 1)      # flip so +y is up
    grid = np.zeros((h, w), dtype=int)
    for cx, cy in zip(gx.astype(int), gy.astype(int)):
        grid[cy, cx] += 1
    peak = max(1, grid.max())
    lines = []
    for row in grid:
        lines.append("".join(ramp[min(len(ramp) - 1,
                      int(v / peak * (len(ramp) - 1)))] for v in row))
    print(f"    {ylabel} ↑  (x = {xlabel} →, each axis ±3σ)")
    for ln in lines:
        print("    |" + ln)
    print("    +" + "-" * w)


def flatness(coords):
    """Participation ratio of the 3-axis covariance: 3.0 = round, →1 = a line."""
    cov = np.cov(coords, rowvar=False)
    ev = np.clip(np.linalg.eigvalsh(cov), 0, None)
    s = ev.sum()
    if s <= 0:
        return 1.0, ev
    pr = (s ** 2) / np.square(ev).sum()
    return float(pr), ev / s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings", default="data/embeddings/track_embeddings.npz")
    ap.add_argument("--descriptors", default="data/embeddings/descriptors.npz")
    args = ap.parse_args()

    emb_path, desc_path = Path(args.embeddings), Path(args.descriptors)
    if not emb_path.exists():
        sys.exit(f"No embeddings at {emb_path}")
    if not desc_path.exists():
        sys.exit(f"No descriptor bank at {desc_path}")

    library = load_centred_library(emb_path)          # (N, 512)
    bank = load_bank(desc_path)
    print(f"Library: {library.shape[0]} tracks, {library.shape[1]}-d, centred")
    print(f"Bank:    {len(bank)} descriptors\n")

    # Build every candidate axis and its raw (uncalibrated) projection.
    axes, dirs, raw = {}, {}, {}
    for name, (low, high) in CANDIDATE_AXES.items():
        d = build_axis(name, low, high, bank)
        if d is None:
            continue
        p = library @ d
        axes[name] = d
        dirs[name] = d
        raw[name] = p

    names = list(axes)
    if len(names) < 3:
        sys.exit("Fewer than three usable axes — cannot form a triad.")

    # Signal strength = spread of the raw projection.  A tiny std means the axis
    # barely distinguishes this library (the descriptor variance gate, applied to
    # a direction).  Reported as a fraction of the median across candidates.
    raw_std = {n: float(raw[n].std()) for n in names}
    median_std = float(np.median(list(raw_std.values())))

    print("── Axis signal strength (raw projection std, higher = more separating)")
    for n in sorted(names, key=lambda k: -raw_std[k]):
        bar = "█" * int(raw_std[n] / max(raw_std.values()) * 30)
        flag = "  ⚠ weak" if raw_std[n] < 0.5 * median_std else ""
        print(f"  {n:11s} {raw_std[n]:.4f}  {bar}{flag}")

    # z-score each axis, then correlate.  Orthogonality is what makes a 3-D cloud.
    z = {n: (raw[n] - raw[n].mean()) / (raw[n].std() + 1e-12) for n in names}
    Z = np.column_stack([z[n] for n in names])
    R = np.corrcoef(Z, rowvar=False)

    print("\n── Correlation matrix (0 = independent, ±1 = collinear)")
    print("            " + " ".join(f"{n[:6]:>6}" for n in names))
    for i, n in enumerate(names):
        print(f"  {n:11s}" + " ".join(f"{R[i, j]:+6.2f}" for j in range(len(names))))

    # Score every triad by the volume it spans: det of its correlation submatrix
    # (1 = mutually orthogonal, →0 = flat), tie-broken toward stronger signal.
    def triad_score(tri):
        idx = [names.index(t) for t in tri]
        sub = R[np.ix_(idx, idx)]
        det = float(np.linalg.det(sub))
        strength = sum(raw_std[t] for t in tri)
        return det, strength

    triads = list(itertools.combinations(names, 3))
    ranked = sorted(triads, key=lambda t: triad_score(t)[0], reverse=True)

    # Orthogonality alone rewards weak-but-independent axes; a usable frame also
    # needs *contrast*.  Balanced score = orthogonality × mean signal strength.
    def balanced(tri):
        det, strength = triad_score(tri)
        return det * (strength / 3.0)

    ranked_bal = sorted(triads, key=balanced, reverse=True)
    recommended = ranked_bal[0]

    print("\n── Most-orthogonal triads (det of correlation submatrix)")
    for tri in ranked[:5]:
        det, strength = triad_score(tri)
        print(f"  det {det:.3f}   Σsignal {strength:.3f}   {' · '.join(tri)}")

    print("\n── Best balanced triads (orthogonality × mean signal) — the usable pick")
    for tri in ranked_bal[:5]:
        det, strength = triad_score(tri)
        tag = "   ← recommended" if tri == recommended else ""
        print(f"  score {balanced(tri):.3f}   det {det:.3f}   Σsignal {strength:.3f}   "
              f"{' · '.join(tri)}{tag}")

    have_preferred = all(p in names for p in PREFERRED)
    if have_preferred:
        det, strength = triad_score(PREFERRED)
        rank = ranked.index(PREFERRED) + 1
        print(f"\n  preferred {' · '.join(PREFERRED)}: det {det:.3f} "
              f"(orthogonality rank {rank} of {len(ranked)}, Σsignal {strength:.3f})")

    # Reference ceiling: how much variance three PCs would capture, vs a triad.
    cov = np.cov(library, rowvar=False)
    pca_ev = np.sort(np.clip(np.linalg.eigvalsh(cov), 0, None))[::-1]
    pca3 = pca_ev[:3].sum() / pca_ev.sum()
    print(f"\n  (reference: top-3 PCA captures {pca3:.1%} of total variance — "
          "the abstract ceiling a legible frame trades against)")

    # Show the balanced recommendation and, if different, the preferred triad.
    to_show = [("RECOMMENDED (balanced)", recommended)]
    if have_preferred and PREFERRED != recommended:
        to_show.append(("PREFERRED", PREFERRED))

    for title, tri in to_show:
        coords = np.column_stack([z[t] for t in tri])
        pr, share = flatness(coords)
        print(f"\n{'='*60}\n{title}: {' · '.join(tri)}")
        print(f"  cloud roundness (participation ratio, 3=round, 1=line): {pr:.2f}")
        print(f"  variance split across the three axes: "
              + " / ".join(f"{s:.0%}" for s in share))
        a, b, c = tri
        ascii_scatter(z[a], z[b], a, b)
        ascii_scatter(z[a], z[c], a, c)

    print("\nThrowaway — delete when done.  Re-run after regenerating embeddings.")


if __name__ == "__main__":
    main()
