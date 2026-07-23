"""
The vibe readout — naming the session with the CLAP descriptor bank.

Audit H1/H1b, Stage 3.  The bank has existed since Stage 1 and been unused since;
this is the display layer that finally reads it.  Nothing here touches the
player: it takes a session vector, a seeded flag and a track count, and returns
two lines of text.

Three things here are load-bearing, and each one is a defect this project has
already shipped once, wearing a different costume.

**1 · A zero session vector produces a confident-looking readout.**  Stage 2 made
"nothing has played yet" a real state (L7): the session vector starts at zero.
`DescriptorBank.top(zeros)` does not refuse — every similarity is exactly 0, so
every z-score collapses to `−mean_d / std_d`, a finite, deterministic, entirely
plausible ranking determined by the bank's own baselines and nothing else.  On
the real bank it reads `shimmering · orchestral · serene`, about no music at all.
That is H1's original defect — a mood word pinned to a constant — re-derived from
a different formula, and the bank cannot defend against it because the arithmetic
is valid.  Every entry point here is gated on `seeded`.

**2 · The drift figure is a count, not a cosine, and that was a measurement.**
H1 specified "the cosine between the session's descriptor z-vector now and five
tracks ago".  Measured over 40 real sessions on the 674-track library (§10d),
that cosine has a median of 0.988 and a p10 of 0.947: 90% of ordinary listening
lands in the top 5% of its nominal range, so the number reads as "0.99" forever
and carries almost nothing.  It is a compressed scale — precisely C5's shape.
What *does* occupy its range is how many of the three words on screen were also
on screen five tracks ago: measured min 0, median 2, max 3.  So the readout
reports that count.  It needs no threshold, no calibration and no vocabulary,
and it is a statement about something the listener actually saw.  The cosine is
still computed and still shown, but in `[I]` where there is room to label it.

**3 · The comparison window is in tracks, and says how far back it reached.**
Snapshots are recorded once per full listen — the event that increments
`tracks_played` — so "5 tracks ago" is literally five tracks.  Early in a session
there are fewer than five, so the line reports the distance it actually reached
rather than implying a full window.
"""

from collections import deque
from typing import List, Optional, Sequence, Tuple

import numpy as np

from config import config

# How many of the top descriptors the line names.
TOP_N = 3

# Shown when there is no vector to describe, or no bank to describe it with.
UNSEEDED_LINE = "♪ —  nothing has played yet"
NO_BANK_LINE = "♪ —  no descriptor bank (see startup log)"

SEPARATOR = " · "


class VibeReadout:
    """
    Descriptor readout for the session vector, plus a rolling store of past
    z-vectors so drift can be measured against something.

    The store lives here, in the display layer, deliberately.  `SessionState`
    keeps `recent_tracks` — the last five *track* embeddings — which is not what
    a drift figure compares; adding a second history to it would put a display
    concern inside the closed and tested player (audit §8, Stage 3 trap 3).
    Nothing in this class is read by selection.
    """

    def __init__(self, bank, window: Optional[int] = None, top_n: int = TOP_N):
        self.bank = bank
        self.window = window if window is not None else config.session_influence_window
        self.top_n = top_n
        # (tracks_played, z_vector, top_labels).  One more than the window so the
        # oldest entry is exactly `window` tracks back once it has filled.
        self._snapshots: deque = deque(maxlen=self.window + 1)

    # ── Recording ────────────────────────────────────────────────────────────

    def observe(self, vector: np.ndarray, tracks_played: int, seeded: bool):
        """
        Record a snapshot if this is a track boundary we have not seen.

        Called on every display refresh; a no-op on all but the first refresh
        after each full listen.  An unseeded vector is never recorded — a zero
        z-vector would be a baseline artifact, not a vibe (see the module note).
        """
        if self.bank is None or not seeded:
            return

        if self._snapshots:
            last_count = self._snapshots[-1][0]
            if tracks_played == last_count:
                return
            if tracks_played < last_count:
                # A new session started under the same TUI.  The old snapshots
                # describe a different evening.
                self._snapshots.clear()

        z = self.bank.z_scores(vector)
        self._snapshots.append((tracks_played, z, self._labels_from_z(z)))

    def reset(self):
        """Forget every snapshot — a new session is not a continuation."""
        self._snapshots.clear()

    # ── Reading ──────────────────────────────────────────────────────────────

    def descriptors(self, vector: np.ndarray, seeded: bool) -> List[Tuple[str, float]]:
        """Top-n descriptors by z-score, or [] when there is nothing to name."""
        if self.bank is None or not seeded:
            return []
        return self.bank.top(vector, self.top_n)

    def drift(self, vector: np.ndarray, seeded: bool) -> Optional[dict]:
        """
        How much the character of the session has moved, measured two ways.

        `held` / `of` is the headline: how many of the words on screen now were
        also on screen `distance` tracks ago.  `cosine` is H1's original
        statistic, kept for `[I]` — it is a real measurement, it is just a
        compressed one (see the module note).

        None when there is no earlier snapshot to compare against.
        """
        if self.bank is None or not seeded or not self._snapshots:
            return None

        oldest_count, oldest_z, oldest_labels = self._snapshots[0]
        z_now = self.bank.z_scores(vector)
        labels_now = self._labels_from_z(z_now)

        # `tracks_played` is not passed in here — the snapshot carries its own
        # count, and the newest snapshot's count is the current one.
        distance = self._snapshots[-1][0] - oldest_count
        if distance < 1:
            return None

        return {
            'held': len(set(labels_now) & set(oldest_labels)),
            'of': len(labels_now),
            'distance': distance,
            'cosine': _cosine(z_now, oldest_z),
        }

    # ── Formatting ───────────────────────────────────────────────────────────

    def descriptor_line(self, vector: np.ndarray, seeded: bool) -> str:
        """`♪ hypnotic · nocturnal · sparse`, or an honest refusal."""
        if self.bank is None:
            return NO_BANK_LINE
        if not seeded:
            return UNSEEDED_LINE
        words = [label for label, _z in self.descriptors(vector, seeded)]
        if not words:
            return NO_BANK_LINE
        return "♪ " + SEPARATOR.join(words)

    def drift_line(self, vector: np.ndarray, seeded: bool, tracks_played: int) -> str:
        """`⟳ 2 of 3 held over 5 tracks · 14 played`, degrading to just the count."""
        played = f"{tracks_played} played"
        moved = self.drift(vector, seeded)
        if moved is None:
            return f"⟳ {played}"
        tracks = "track" if moved['distance'] == 1 else "tracks"
        return (f"⟳ {moved['held']} of {moved['of']} held over "
                f"{moved['distance']} {tracks} · {played}")

    def lines(self, vector: np.ndarray, seeded: bool,
              tracks_played: int) -> Tuple[str, str]:
        """The two rows the Now Playing panel shows."""
        return (self.descriptor_line(vector, seeded),
                self.drift_line(vector, seeded, tracks_played))

    # ── Internals ────────────────────────────────────────────────────────────

    def _labels_from_z(self, z: np.ndarray) -> Tuple[str, ...]:
        order = np.argsort(z)[::-1][:self.top_n]
        return tuple(self.bank.labels[i] for i in order)


def format_descriptors(pairs: Sequence[Tuple[str, float]]) -> str:
    """`aggressive +1.0 · gritty +0.8 · intense +0.7` — the `[I]` form, with the
    z-scores visible, because the inspector's job is to show the numbers."""
    if not pairs:
        return "—"
    return SEPARATOR.join(f"{label} {z:+.1f}" for label, z in pairs)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
