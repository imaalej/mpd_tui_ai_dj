# Adaptive Session AI DJ

A terminal DJ for MPD that selects tracks from CLAP audio embeddings and adapts to feedback within a
listening session. Python + urwid TUI, MPD driven via `mpc` subprocesses.

## Read this first

**`PROJECT_AUDIT.md` is the governing document.** It is a full codebase audit plus the design decisions
for an in-progress rewrite. Before changing anything:

- **§0** — the decision record. It changes the meaning of about half the findings.
- **§0b** — the implementation log: what has shipped, and where doing the work contradicted the plan.
- **§8** — the ordered work plan, with a definition of done per stage.
- **§7** — the `.npz` schemas. Both artifacts now exist and are validated on load; build to these.
- **§10c** — the Stage 2 measurements, taken by driving the shipped code. Supersedes §10b's two
  "Re-measured for Stage 2" tables, which were simulations written before the code existed.

Findings are tagged `OPEN` / `NEW` / `ELEVATED` (work required), `DONE` / `PARTIALLY DONE` (shipped),
or `DISSOLVED` / `SUPERSEDED` (no work — the design change removed them). Don't fix a dissolved
finding. §6 is the current status table.

## Project state

**Stages 0, 1 and 2 are complete. Stage 3 is next.** Line references in the audit's §2–§5 predate
Stage 0 and are stale for every file those stages touched.

**The application plays.** It runs continuously one track ahead, adapts on the very next song,
forces and restores MPD's playback modes on every exit path, and escalates `[N]` by a magnitude
solved for a measured pool-turnover target. Verified live: 36 tracks, no stall, no repeat inside the
replay gap, modes restored byte-identically after `kill -TERM`. Every critical finding is closed.

**Stage 3 is display only.** Read §8's Stage 3 section first — it opens with what is settled, what
Stage 2 left in place for it, and the one item in the stage that risks inventing a threshold. The
short version:

- **Do H8 first.** The album art is pinned to hand-counted row constants (`RIGHT_COL_ROWS = 10`,
  `x=2`, `y=3`). Stage 2 deliberately left the panel geometry untouched so those constants are still
  correct — and every other Stage 3 item moves the layout out from under them.
- The files in scope are `tui.py` and `album_art.py`, plus `descriptor_bank.py` as a *reader*. If a
  display change seems to need `queue_manager.py`, `track_selector.py`, `session_state.py` or
  `manifold.py`, stop and work out why — the player is closed and tested.
- `DescriptorBank` already loads as `self.descriptor_bank` on the orchestrator; `z_scores(vector)`
  and `top(vector, n)` are the whole API H1b and H1d need. It has been sitting unused since Stage 1.
- **Gate the vibe readout on `session_state.is_seeded()`.** Stage 2 made the session vector start at
  zero, and `bank.top(zeros)` does not refuse — every similarity is 0, so it returns
  `−mean_d / std_d`, which reads as `shimmering · orchestral · serene` about nothing at all. This is
  H1's original defect in a new costume and the bank cannot defend against it. A test pins the
  hazard.
- **The history panel needs a different metadata path.** `get_playlist_metadata()` reads
  `mpc playlist`, and under `consume on` played tracks are gone from it — so it covers exactly the
  tracks the history panel does not show. `MPDController._fetch_track_tags()` is the per-track,
  cached one (mutagen → `mpc search` → filename).
- **Nothing stores what H1's consistency word compares against.** `recent_tracks` holds the last five
  *track embeddings*, not past session-vector z-vectors.
- `↑↓` and `ENTER` are unbound and absent from the footer, so H1d rebinds free keys. **Re-derive the
  history indices from scratch** — the old ones counted from the top of the MPD playlist, which is
  why `ENTER` used to replay the session's first track.
- `[I]` already reports τ, the drawn rank, β earned and the next skip's turnover target. H1d adds the
  descriptor rows; the overlay is sized to its content and will need to **scroll** once they land.
- **Nothing in the suite constructs the widget tree.** 311 tests cover everything *behind* the
  display, so Stage 3 starts with no coverage in its own area — which is how H1, C1 and C4 all shipped
  under a green suite. H8's geometry is arithmetic and testable.

The vector space is settled: 674 tracks, full-coverage deterministic windows, centred on load, plus a
49-word CLAP descriptor bank. So is the player. Use §10c's numbers for anything about skips,
selection or the manifold; §10b for the embeddings and the descriptor bank.

Where the code and the audit disagree, the audit wins. Several of the audit's own empirical claims
turned out to be wrong when the work was done; those findings have been **rewritten to carry the
correct claim**, so you can read any finding as current rather than hunting for a later correction.
§0b records only what did not belong inside a finding.

**Nothing in `data/state/` is worth preserving.** The taste vector, exploration state, feedback
history and play history are all regenerated by listening (§0, D3), and none of it is committed. Do
not write migration code, backward-compat shims, or "reset carefully" logic for any of it. `data/embeddings/` is different now — it is a five-minute rebuild, not
a throwaway, and regenerating it is `python3 generate_embeddings.py --force`.

**Significant refactoring is in scope** (§0, D7). Where a finding's fix is "patch this" but the honest
answer is "delete this and rewrite it smaller," take the second. Roughly 15% of the codebase is being
deleted outright.

## The governing principle

Most defects in this project share one cause: **numbers chosen against a scale nobody measured.**
Entropy thresholds calibrated for a smaller dimension, a novelty formula assuming a range the data
never occupies, a "50% vibe shift" that turns over 9% of what you will actually hear.

So: **derive constants from the library's actual distribution, or delete them.** A constant may stay if
it shapes behaviour without asserting a fact. It goes if it produces a claim the user reads that the
system cannot back up.

Concretely, prefer scale-invariant formulations (rank-based sampling over score-based; z-scores over
raw similarities; measured pool-turnover over a declared magnitude) so nothing needs recalibration when
the weights or the embedding space move.

The audit's empirical claims are reproducible — **§10c** is the live record for skips, selection and
the manifold, **§10b** for the embeddings and the descriptor bank, §10 the pre-Stage-1 provenance. If
you change a formula that a number in there depends on, re-measure rather than assuming. Each of the
last three stages did that and each time it changed something material — most recently, a skip
turnover the audit reported as a constant 0.3% turned out to be **2.9%**, and to vary tenfold with
how settled the session vector is.

## Running it

```sh
./start.sh                                            # the only entry point; launches main_tui.py
python3 generate_embeddings.py --help
python3 generate_embeddings.py --stats                # raw vs centred similarity distributions
python3 generate_embeddings.py --describe "Bathory"   # top descriptors for a track
python3 generate_embeddings.py --compare-templates    # re-run the prompt-template measurement
```

A full regeneration takes ~5.5 minutes on the 3070 and must not be run casually while testing — it
rewrites the artifact every downstream number depends on.

`main_tui.py` is the only orchestrator; `main.py` and `setup_check.py` were deleted in Stage 0 (M2).
There is no demo/random-embedding path anywhere — it was removed on purpose (M2/M4).

**Driving it for real.** urwid needs a tty, so end-to-end verification runs it under `pty.fork()` —
and the pty needs an explicit `TIOCSWINSZ`, or it reports 0×0, urwid draws nothing, and the run looks
like it worked while every captured frame is empty. Playback can be accelerated with `mpc seek 99%`
per track. **Always snapshot the user's MPD queue, modes and volume first and restore them after**;
the app itself only restores the modes.

## Environment

- Fedora, Python 3.14.6, numpy 1.26.4, urwid 3.0.5, transformers 5.1.0, torch/torchaudio
- MPD's real `music_directory` is `/mnt/storage/music`, now read from `~/.config/mpd/mpd.conf` by
  `music_directory.py` rather than assumed (M3, done). `/var/lib/mpd/music` is a symlink to it — the
  reason the old hardcoded default worked, and the reason it hid
- MPD's database holds 692 entries; 690 are audio, 674 have embeddings. The 16 without are one corrupt
  album, listed with their exceptions in `data/embeddings/failed.txt`
- Embedding generation runs on an RTX 3070 in ~5.5 minutes; the bottleneck is CPU mel extraction, not
  the GPU, so `--workers` matters more than `--batch-size`. CPU-only is the untested fallback
- Album art works only via ueberzugpp; the kitty and sixel paths are non-functional (L2)

## Constraints

- **MPD playback modes are the user's state.** The app forces `random`/`repeat`/`single` off and
  `consume` on, but it must log what it changed and restore the originals on *every* exit path
  including SIGTERM (C2 + H3). Leaving a user's MPD in consume mode is a real-world side effect.
- **`mpc listall` is the single source of truth for track keys** (M4). Do not enumerate the filesystem
  for anything MPD will be asked to play.
- **A skip must add the replacement before it advances.** Verified against the live MPD: `mpc next`
  off the last remaining track empties the queue and stops, and a subsequent `mpc add` does *not*
  restart it. Advance-then-add therefore kills the session silently, and the only recovery is a
  `play()` call that C4 forbids in a skip path. The full verified consume-mode semantics table is in
  the audit under M1, including four rows measured in Stage 2 — notably that **`mpc next` while
  paused advances *and resumes playing***, contrary to what C4 originally assumed. `FakeMPD` is built
  to that table; extend it there rather than reasoning from the protocol.
- **Never blend an audio-space vector toward a text embedding, or toward a random direction.** CLAP's
  towers don't share a cone, and a random 512-d direction is 0.085-similar to real music where a
  session vector is 0.787 and a real track is 0.748. `manifold.py` owns this: any large displacement
  is projected back with `snap()` = `normalise(mean(top-25 library embeddings by dot(E, v)))`. Three
  things about it that were learned the hard way and are easy to undo:
  - **`snap()` is a move, not a projection.** It has a turnover floor of its own (~8%), so applying
    it to a single small skip overshoots the schedule. It is gated at run length ≥ 2.
  - **Solve *through* the snap, not before it.** λ chosen against the un-snapped vector and snapped
    afterwards let a second consecutive skip land back where the run started — measured live at 1%
    turnover against a 20% target. `solve_repulsion(..., snap_result=True)` is not an optimisation.
  - **"Still music" has no fixed threshold**, only the library's own distribution: real tracks span
    0.427 to 0.961 on this measure. The assertion that means something is "no worse than the least
    typical real track" (p1 = 0.463).
- **stderr is swallowed while the TUI runs.** `data/dj.log` is the durable copy (L5, shipped in
  Stage 0) — read it, not the 5-line console panel.

## Testing

`python3 -m pytest tests` — 311 tests, green, about 17 seconds. The three `test_phase*.py` files were
deleted rather than repaired (M1a); roughly thirty of their 66 "passing" checks were hardcoded `True`
literals.

`tests/test_clap_pipeline.py` loads the real CLAP checkpoint and **skips** if it is not already in the
HF cache — a test that silently downloads 700 MB is not a test anyone can run. It also skips the
library assertions if `data/embeddings/` is empty.

The suite is behavioural, not existence checks — C1 and C4 both survived under a green suite of the
latter. What that means in practice, and what to preserve:

- **`FakeMPD` is itself under test.** `tests/test_fake_mpd.py` asserts the double against the
  verified semantics table row by row, because a double built on the assumptions that produced C1
  would reproduce C1 and pass. It has already earned that: a fixture defaulting to `consume off`
  silently put every component back in C1's world, and the replay-gap test caught it.
- **Tests drive the real methods.** `test_skip_path.py` calls the actual
  `AdaptiveDJWithTUI.skip_current_track` against a stand-in and asserts on FakeMPD's **call log** —
  one `next`, no `play`, every `add` before the advance. A test that mirrors the ordering it is
  checking proves only that the mirror is self-consistent.
- **Claims about the library are tested against the library.** `test_skip_escalation.py` skips if
  `data/embeddings/` is absent, because whether a turnover target is reachable at all is a property
  of the collection's structure, not of the solver.
- Stage 1's three acceptance properties stay asserted: bit-deterministic embedding (for a fixed batch
  size), self-similarity exactly 1.0, post-centring random pairs at +0.011.

Fixtures live in `tests/conftest.py`: `rng` (seeded), `library` (in-memory `TrackLibrary`),
`make_artifact` (schema-correct `.npz` with any field overridable), `fake_mpd` (**consume on**, the
state the DJ forces) and `dj_parts` (the real selection stack wired to `FakeMPD`).
