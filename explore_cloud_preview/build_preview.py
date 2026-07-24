#!/usr/bin/env python3
"""
THROWAWAY — assemble the self-contained scaffold preview page.

Inlines `core.js` (the verified port of `compute_frame`) and `data.json` (the
real 674-track cloud) into one HTML file, because the artifact host blocks every
external request.  Run `explore_cloud_scaffold.py` first, then:

    python3 explore_cloud_preview/build_preview.py
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).parent            # tracked sources
OUT = HERE.parent / "scratch_scaffold"  # gitignored output

# (mode, label, note, in the [B] cycle).  The three chosen stops come first —
# the rail is a decision, not a catalogue.
MODE_NOTES = [
    ("off", "No frame", "How it is today. A field of specks: the orbit reads as "
     "swimming rather than as a volume turning.", True),
    ("floor+shadow", "Ground", "A ruled floor plus the library flattened onto it. "
     "A point's distance from its own shadow is its height — the strongest depth "
     "cue available on this grid.", True),
    ("cage+triad", "Box + axes", "All twelve edges, stippled every second dot, with "
     "the ruled axes through the origin ticked every 1σ.", True),
    ("corners+floor+triad", "Marks + ground + axes", "Crop-mark corners over the "
     "floor grid, axes ticked, no shadow mass underneath. The light-touch middle.",
     True),
    ("gnomon", "Corner instrument", "A 5-cell axis cross anchored to the panel "
     "corner, each arm shaded by whether it points at you. 60 dots, never touches "
     "the cloud — but it tells you the orientation without describing the volume.",
     False),
    ("corners", "Crop marks alone", "Reads as a box while leaving the middle of the "
     "panel — where the cloud is densest — completely clear.", False),
    ("cage", "Full box alone", "The most explicit frame, at the price of one near "
     "edge crossing the data.", False),
    ("walls", "Room", "Only the three faces currently behind the cloud, ruled as "
     "grids, re-chosen every frame. Cannot occlude anything inside the box; the "
     "wall that swaps as you pass 90° is a cue of its own.", False),
    ("cage+gnomon", "Box + instrument", "Volume from the box, orientation from the "
     "corner cross.", False),
    ("corners+floor+shadow", "Marks + ground", "The least ink that still gives both "
     "a bounded volume and a horizon.", False),
    ("walls+floor+shadow", "Everything", "Where it tips into busy — kept visible so "
     "the ceiling is known, not because it should ship.", False),
]


def main():
    core = (HERE / "core.js").read_text()
    core = re.sub(r"^export ", "", core, flags=re.M)
    data = json.loads((OUT / "data.json").read_text())
    notes = [n for n in MODE_NOTES if n[0] in data["modes"]]
    missing = set(data["modes"]) - {n[0] for n in notes}
    assert not missing, f"the preview would silently drop {missing}"

    html = (HERE / "preview.template.html").read_text()
    html = (html.replace("/*__CORE__*/", core)
                .replace("/*__DATA__*/", json.dumps(data))
                .replace("/*__NOTES__*/", json.dumps(notes)))
    out = OUT / "preview.html"
    out.write_text(html)
    print(f"wrote {out} ({len(html) // 1024} KB, {len(notes)} modes)")


if __name__ == "__main__":
    main()
