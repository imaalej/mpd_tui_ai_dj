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
- **§10b** — the Stage 1 re-measurement. Supersedes §10 wherever they overlap.

Findings are tagged `OPEN` / `NEW` / `ELEVATED` (work required), `DONE` / `PARTIALLY DONE` (shipped),
or `DISSOLVED` / `SUPERSEDED` (no work — the design change removed them). Don't fix a dissolved
finding. §6 is the current status table.

## Project state

**Stages 0 and 1 are complete. Stage 2 is next.** Line references in the audit's §2–§5 predate
Stage 0 and are stale for every file those stages touched.

**The application starts and reports a clean library load.** It does not yet play continuously — the
queue is ten deep and never refills (C1), MPD's `random on` still discards the ordering (C2), and
selection is a strict argmax (H6). That is Stage 2, and D1 + H6 must land together.

**Starting Stage 2:** read §8's Stage 2 section first — it opens with what Stage 1 settled, what moved
and by how much, and the numbers not to trust. Then M1's verified MPD semantics table, because the
`FakeMPD` is the first thing to build and the assumptions that produced C1 would reproduce it. The
files in scope are `queue_manager.py` (mostly deleted and rewritten), `feedback_handler.py`,
`session_state.py`, `exploration_controller.py`, `track_selector.py`, `main_tui.py`, `persistence.py`
and the key bindings in `tui.py`. Nothing under `data/embeddings/`, `embeddings_io.py`,
`embedding_generator.py`, `descriptor_bank.py` or `music_directory.py` should need to change; the
descriptor bank loads as `self.descriptor_bank` on the orchestrator and stays unused until Stage 3.

The vector space is now trustworthy and should be treated as settled: 674 tracks, full-coverage
deterministic windows, centred on load, plus a 49-word CLAP descriptor bank. §10b is the
re-measurement; use those numbers, not §10's pre-C3 ones.

Where the code and the audit disagree, the audit wins. Several of the audit's own empirical claims
turned out to be wrong when the work was done; those findings have been **rewritten to carry the
correct claim**, so you can read any finding as current rather than hunting for a later correction.
§0b records only what did not belong inside a finding.

**Nothing in `data/state/` is worth preserving.** The taste vector, exploration state and feedback
history are discarded (§0, D3). Do not write migration code, backward-compat shims, or "reset
carefully" logic for any of it. `data/embeddings/` is different now — it is a five-minute rebuild, not
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

The audit's empirical claims are reproducible — **§10b** gives the live measurements and the
conditions they were taken under (§10 is the pre-Stage-1 record, kept as provenance). If you change a
formula that a number in there depends on, re-measure rather than assuming. Stage 1 did exactly that
and found that a skip nudge which used to move 3.9% of the candidate pool now moves 0.3%.

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
  the audit under M1 — build `FakeMPD` to it, not to intuition.
- **Never blend an audio-space vector toward a text embedding, or toward a random direction.** CLAP's
  towers don't share a cone, and a random 512-d direction is 0.085-similar to real music where a
  session vector is 0.641 and a real track is 0.729. After any large session-vector manipulation,
  project back onto the manifold: `normalise(mean(top-25 library embeddings by dot(E, v)))` (H9) —
  but only for displacements large enough to warrant it; `snap()` is itself a move, and applied to a
  small nudge it overshoots (§10b).
- **stderr is swallowed while the TUI runs.** `data/dj.log` is the durable copy (L5, shipped in
  Stage 0) — read it, not the 5-line console panel.

## Testing

`python3 -m pytest tests` — 179 tests, green, about 11 seconds. The three `test_phase*.py` files were
deleted rather than repaired (M1a); roughly thirty of their 66 "passing" checks were hardcoded `True`
literals.

`tests/test_clap_pipeline.py` loads the real CLAP checkpoint and **skips** if it is not already in the
HF cache — a test that silently downloads 700 MB is not a test anyone can run. It also skips the
library assertions if `data/embeddings/` is empty.

The suite covers Stages 0 and 1 and **nothing in it touches MPD**. The `FakeMPD` that models real
semantics including consume mode is Stage 2 (M1b) and has not been written. Behavioural tests, not
existence checks: C1 and C4 both survived under a green suite.

Stage 1's three acceptance properties are asserted and stay asserted: embedding is bit-deterministic
(for a fixed batch size — §0b), self-similarity is exactly 1.0, and the post-centring random-pair
distribution sits at +0.011.

Fixtures live in `tests/conftest.py`: `rng` (seeded), `library` (in-memory `TrackLibrary`),
`make_artifact` (writes a schema-correct `.npz` with any field overridable, for testing the loader's
refusals). `FakeMPD` belongs beside them.
