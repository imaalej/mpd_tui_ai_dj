# Archive — do not edit

Reference material from the vibe-cloud planning phase. These are frozen design
artifacts, not shipping code. Leave them as-is.

## `vibe_space_webapp.html`

A **browser** design preview of the vibe cloud — the "feel the concept" tool,
not the delivery target. Self-contained (real data embedded), opens in any
browser. It renders the real 674-track library projected onto the three mood
axes **Intensity · Tone · Organic**, coloured by those same three numbers as
HSV, auto-rotating, with a simulated session comet, drag-orbit, scroll-zoom, and
click-to-inspect.

It is deliberately smoother and prettier than what ships: the **shipping target
is the terminal Braille widget**, which is coarser (2×4 dots per cell, 256
colour, no anti-aliasing/glow). Keep this only as the reference for what the
concept is reaching toward.

Why the axes are Intensity·Tone·Organic and not the intuitive
Intensity·Tone·Saturation: measured on this library, Intensity and Saturation
correlate **0.98** (the cloud collapses to a plane). The axis triad is
data-driven per library — see `explore_mood_axes.py`.

## Live prototypes (kept in the repo root, untracked, throwaway)

- `explore_mood_axes.py` — picks the best three axes for a given library and
  prints the correlation/orthogonality analysis.
- `vibe_cloud_demo.py` — the faithful terminal Braille render (`--live`), and the
  JSON exporter that feeds the previews.
- `render_terminal_frames.py` — earlier baked-frame terminal mock (superseded by
  the live one).

## Snapshot

Planning phase, July 2026. `../inspection_findings.md` held the design record and
the staged work plan; it and `../project_state.md` were distilled into
`../PROJECT.md` and now live only in git history
(`git show c599ab7:inspection_findings.md`).
