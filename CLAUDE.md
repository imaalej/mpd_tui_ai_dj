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
- **§10d** — the Stage 3 measurements: the descriptor drift distribution (which changed what the
  readout ships), the derived-versus-hand-counted art geometry, and the terminal sizes at which the
  pre-Stage-3 layout crashed.
- **§10e** — the Stage 4 measurements: why un-liking is a replay and not a subtraction, what a first
  run costs on disk and on CPU, and the urwid source showing that **L1's conclusion was wrong**.

Findings are tagged `OPEN` / `NEW` / `ELEVATED` (work required), `DONE` / `PARTIALLY DONE` (shipped),
or `DISSOLVED` / `SUPERSEDED` (no work — the design change removed them). Don't fix a dissolved
finding. §6 is the current status table.

## Project state

**Stages 0–4 are complete. The rewrite is done and every finding is closed.** Line references in the
audit's §2–§5 predate Stage 0 and are stale for every file those stages touched.

**The application plays, and now says what it is playing.** It runs continuously one track ahead,
adapts on the very next song, forces and restores MPD's playback modes on every exit path, and
escalates `[N]` by a magnitude solved for a measured pool-turnover target. The vibe line names three
CLAP descriptors z-scored against this library, the Session panel lists what actually played with
`♥` / `⏭` / `✓` marks, `↑↓` and `ENTER` scroll and replay it, and the album art's position is derived
from the widget tree. Verified live at 120×45 and 80×24, with the user's MPD queue, modes and volume
restored byte-identically.

**Stage 4 made it durable.** 542 tests. The fallback text mode is driven through a pty and shares one
binding table with the urwid mode; all four state files have round-trip coverage; `[L]` is a toggle;
the kitty and sixel art paths are deleted; the ueberzugpp child dies on every exit path; the
`.gitkeep` scaffolding ships; and the setup numbers are measured and held together by a test.

**The most important thing Stage 4 learned is about this document.** Acting on **L1's fix direction
exactly as written broke the application** — urwid 3.0.5 *chains* to the previous SIGWINCH handler
rather than replacing it, so re-installing on top closed a recursion. Nine unit tests passed, because
they ran against a double built from L1's own assumption. A live `SIGWINCH` found it in seconds.
L1 has been rewritten to carry the correct claim. **Where a finding states a conclusion drawn from an
observation, check the observation still supports it before acting.**

Three things later stages established that are easy to undo:

- **The Now Playing box is `('pack', …)`, not `('weight', 3, …)`.** Weighting it raises `WidgetError`
  on every terminal shorter than 33 rows — a defect that shipped through three stages and 311 tests
  because nothing had ever called `render()` (audit **N1**). Packing also makes the panel's height
  independent of the terminal, which is what H8's derived geometry rests on.
- **The drift figure is a count of held words, not a cosine.** H1 specified the cosine; measured over
  40 real sessions it has p10 = 0.948 and median = 0.989, so it reads as "0.99" forever. That is the
  same compressed-scale defect H1 and C5 are both about. The cosine is still computed and shown in
  `[I]` with its distribution beside it. See §10d before changing this back.
- **Un-liking is a replay of the feedback history, not a `−0.1` update.** Measured: the subtraction is
  within 10⁻⁴ of correct in every ordinary case and **totally wrong in one** — a subtraction cannot
  un-seed a model, so retracting your only like leaves the taste vector pinned at unit strength to the
  track you just rejected. It is gated on `UserTaste.explains()` because the history is capped at 1000
  events and a replay past that cap moves the vector by 0.077 for unrelated reasons. See §10e.

**The display layer owns its own state, deliberately.** `vibe_readout.py` (the z-vector drift store)
and `session_history.py` (what played, the marks, the `↑↓` cursor) are pure modules that nothing
behind the display reads. Putting either behind the display would split a display concern across a
component that cannot see it, which is the shape of C4. `SessionState`, `TrackSelector`,
`QueueManager` and `manifold.py` are closed and tested — `QueueManager.requeue_next()` was the one
Stage 3 addition, and §0b records why it belongs there rather than in `tui.py`.

The vector space is settled: 674 tracks, full-coverage deterministic windows, centred on load, plus a
49-word CLAP descriptor bank. So are the player and the display. Use §10c's numbers for anything
about skips, selection or the manifold; §10b for the embeddings and the descriptor bank; §10d for
the readout, the drift distribution and the widget-tree geometry; **§10e for the taste model's
retraction arithmetic and the setup figures**.

Two things Stage 4 measured and left alone, both recorded in §10e rather than fixed: the first `[N]`
press turns over **100%** of the pool against a 5% target when the session vector is cold (§9's item
5, at its extreme — and the first time the escalation has been driven from an empty `data/state/`),
and the Session panel draws no content rows at 80×24 because the packed layout consumes all of them.

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

`python3 -m pytest tests` — 542 tests, green, about 18 seconds. The three `test_phase*.py` files were
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

- **The display is under test now, and rendering is the point.** Stage 3 added the first tests that
  construct the widget tree, and the first one to call `render()` found a crash on every terminal
  under 33 rows. `test_art_geometry.py` renders the real frame at seven terminal sizes and *locates
  the art placeholder in the canvas* rather than asserting `_art_geometry`'s arithmetic back at it —
  a test that repeats the calculation it is checking proves only that it agrees with itself.

- **The fallback text mode is under test, through a pty.** `test_simple_mode.py`. It stopped being
  guesswork when the mode stopped having its own binding table: `decode_key()`/`decode_keys()` turn
  terminal bytes into urwid's key names and `_handle_input` dispatches for both interfaces, so most of
  it is testable without a terminal and a binding cannot exist in one interface and not the other.
  Two traps if you extend it: `tty.setcbreak` uses `TCSAFLUSH`, so keys written to the pty *before*
  the loop starts are discarded — send them from the first tick; and a burst is consumed by one read,
  so a key meant to dismiss `[I]` must arrive after the page opens.
- **The suite must never write to `data/state/`.** It did, for some time — `process_like()` saves to
  `config.taste_file` — and a green run replaced a real taste model with a fixture's. An autouse
  fixture in `conftest.py` redirects all four paths. Do not remove it, and do not add a state file
  without adding it there.

Fixtures live in `tests/conftest.py`: `rng` (seeded), `library` (in-memory `TrackLibrary`),
`make_artifact` (schema-correct `.npz` with any field overridable), `fake_mpd` (**consume on**, the
state the DJ forces), `dj_parts` (the real selection stack wired to `FakeMPD`), and for the display
`fake_art` / `stub_bank` / `dj_stub` / `tui` (the real `AdaptiveDJTUI` with an injected art renderer,
so building the tree does not spawn a ueberzugpp child).
