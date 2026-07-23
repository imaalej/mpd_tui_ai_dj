# Adaptive Session AI DJ

A terminal DJ for MPD that selects tracks from CLAP audio embeddings and adapts to feedback within a
listening session. Python + urwid TUI, MPD driven via `mpc` subprocesses.

## Read this first

**`PROJECT_AUDIT.md` is the governing document.** It is a full codebase audit plus the design decisions
for an in-progress rewrite. Before changing anything:

- **§0** — the decision record. It changes the meaning of about half the findings.
- **§0b** — the implementation log: what has shipped, and where doing the work contradicted the plan.
- **§8** — the ordered work plan, with a definition of done per stage.
- **§7** — the target `.npz` schemas. Build to these.

Findings are tagged `OPEN` / `NEW` / `ELEVATED` (work required), `DONE` / `PARTIALLY DONE` (shipped),
or `DISSOLVED` / `SUPERSEDED` (no work — the design change removed them). Don't fix a dissolved
finding. §6 is the current status table.

## Project state

**Stage 0 is complete. Stage 1 is next.** Line references in the audit's §2–§5 predate Stage 0 and
are stale for every file it touched.

**The application does not currently start.** Stage 0 deleted the embeddings (D3) and removed the
random-vector fallback; `main_tui.py` exits pointing at `generate_embeddings.py`. Regenerating them
*is* Stage 1 — do it with the C3/C5/M8 changes in place, not before, or you will run it twice.

Where the code and the audit disagree, the audit wins.

**Nothing in `data/` is worth preserving.** Embeddings, taste vector, exploration state and feedback
history are all being regenerated or discarded (§0, D3). Do not write migration code, backward-compat
shims, or "reset carefully" logic for any of it.

**Significant refactoring is in scope** (§0, D7). Where a finding's fix is "patch this" but the honest
answer is "delete this and rewrite it smaller," take the second. Roughly 15% of the codebase is being
deleted outright.

## The governing principle

Most defects in this project share one cause: **numbers chosen against a scale nobody measured.**
Entropy thresholds calibrated for a smaller dimension, a novelty formula assuming a range the data
never occupies, a "50% vibe shift" that moves 29%.

So: **derive constants from the library's actual distribution, or delete them.** A constant may stay if
it shapes behaviour without asserting a fact. It goes if it produces a claim the user reads that the
system cannot back up.

Concretely, prefer scale-invariant formulations (rank-based sampling over score-based; z-scores over
raw similarities; measured pool-turnover over a declared magnitude) so nothing needs recalibration when
the weights or the embedding space move.

The audit's empirical claims are reproducible — §10 gives the measurements and the conditions. If you
change a formula that a number in there depends on, re-measure rather than assuming.

## Running it

```sh
./start.sh              # the only entry point; launches main_tui.py
python3 generate_embeddings.py --help
```

`main_tui.py` is the only orchestrator; `main.py` and `setup_check.py` were deleted in Stage 0 (M2).
There is no demo/random-embedding path anywhere — it was removed on purpose (M2/M4).

## Environment

- Fedora, Python 3.14.6, numpy 1.26.4, urwid 3.0.5, transformers 5.1.0, torch/torchaudio
- MPD's real `music_directory` is `/mnt/storage/music`; `/var/lib/mpd/music` is a symlink to it, which
  is the only reason the hardcoded config default works (M3)
- Embedding generation runs on an RTX 3070; CPU is the untested fallback
- Album art works only via ueberzugpp; the kitty and sixel paths are non-functional (L2)

## Constraints

- **MPD playback modes are the user's state.** The app forces `random`/`repeat`/`single` off and
  `consume` on, but it must log what it changed and restore the originals on *every* exit path
  including SIGTERM (C2 + H3). Leaving a user's MPD in consume mode is a real-world side effect.
- **`mpc listall` is the single source of truth for track keys** (M4). Do not enumerate the filesystem
  for anything MPD will be asked to play.
- **Never blend an audio-space vector toward a text embedding, or toward a random direction.** CLAP's
  towers don't share a cone, and a random 512-d direction is 0.105-similar to real music where a
  session vector is 0.697. After any large session-vector manipulation, project back onto the manifold:
  `normalise(mean(top-25 library embeddings by dot(E, v)))` (H9).
- **stderr is swallowed while the TUI runs.** `data/dj.log` is the durable copy (L5, shipped in
  Stage 0) — read it, not the 5-line console panel.

## Testing

`python3 -m pytest tests` — 67 tests, green. The three `test_phase*.py` files were deleted rather than
repaired (M1a); roughly thirty of their 66 "passing" checks were hardcoded `True` literals.

The current suite covers Stage 0 only and **nothing in it touches MPD**. The `FakeMPD` that models
real semantics including consume mode is Stage 2 (M1b) and has not been written. Behavioural tests,
not existence checks: C1 and C4 both survived under a green suite.

Two properties are newly assertable and should be tested from the start: embedding generation is
bit-deterministic, and post-centring the random-pair similarity distribution sits near 0.
