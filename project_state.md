# Adaptive Session AI DJ — Project State

A terminal DJ for MPD that selects tracks from CLAP audio embeddings and adapts to feedback within a
listening session. Python + urwid TUI, MPD driven via `mpc` subprocesses.

**Status: the five-stage rewrite is complete.** 542 tests, green, ~18 s. Every finding from the
original audit is closed. The application plays continuously, adapts on the next track, says what it
is playing, restores the user's MPD state on every exit path, and has coverage on all three
interfaces (player, urwid display, fallback text mode).

---

## What this document is

This replaces `PROJECT_AUDIT.md` (a 3,900-line audit of defects that are now all fixed) and
`CLAUDE.md`. It carries forward only what is still load-bearing:

- the invariants that are **easy to undo and were learned the hard way** (§4) — the single most
  valuable section
- the **verified MPD semantics** that `FakeMPD` is built to (§5)
- the **measured numbers** downstream work depends on (§7)
- the **finding-ID glossary** (§10), because ~450 code comments cite IDs like `C4` and `L9`
- what is deliberately **deferred or left open** (§11)

It does **not** narrate the defects that were fixed — they are fixed, and the narrative was the bloat.
If you need it, both files are in git history:

```sh
git log --oneline --follow -- PROJECT_AUDIT.md   # every version, newest first
git show 82abf60:PROJECT_AUDIT.md                # FINAL, 4,005 lines — Stage 4 log, §10e, corrected L1
git show 82abf60:CLAUDE.md                       # FINAL, 234 lines
git show dcb1a64:PROJECT_AUDIT.md                # through Stage 3, 3,531 lines
git show 16c4286:PROJECT_AUDIT.md                # the original audit, 1,768 lines
```

`82abf60` is the Stage 4 commit — the last one that contained either file.

---

## 1 · The governing principle

Nearly every defect this project has had shared one cause: **numbers chosen against a scale nobody
measured.** An entropy threshold calibrated for a smaller dimension. A novelty formula assuming a
range the data never occupies. A "50% vibe shift" that turned over 9% of what you would actually
hear. A similarity scale compressed into its top third.

So: **derive constants from the library's actual distribution, or delete them.** A constant may stay
if it shapes behaviour without asserting a fact. It goes if it produces a claim the user reads that
the system cannot back up.

Concretely, prefer scale-invariant formulations — rank-based sampling over score-based, z-scores over
raw similarities, measured pool-turnover over a declared magnitude — so nothing needs recalibration
when the weights or the embedding space move.

**The corollary, learned in Stage 4 and worth more than the principle itself: an observation is not a
conclusion.** The audit recorded a correct measurement of the SIGWINCH handler and drew the wrong
inference from it. Acting on that inference broke the application, under nine unit tests that all
passed because they ran against a double built from the same assumption. Before trusting any claim in
this file, check that the observation behind it still supports it.

---

## 2 · How the system works

Everything is driven by 512-dimensional CLAP audio embeddings, one per track, stored L2-normalised in
`data/embeddings/track_embeddings.npz`. All similarity is a dot product **on centred vectors**. No
genre tags or metadata enter the selection logic.

**All application modules live under `src/`** (audit G2); the module names in the diagram below are
`src/…` on disk. `data/`, `tests/`, `README.md`, `start.sh` and this file stay at the repo root, and
`config.data_dir` is anchored to the repo root (the parent of `src/`) so the layout move changed no
path. `start.sh` invokes `python3 src/main_tui.py`.

```
start.sh  ──▶ src/main_tui.py (AdaptiveDJWithTUI)  the only entry point
                   │        owns: MPD mode force/restore, signal handling,
                   │               the one skip path, periodic checkpoint
   ┌───────────────┼────────────────────────────────┐
   ▼               ▼                                ▼
tui.py         background thread                mpd_controller.py
(urwid loop,   (polls MPD 2×/sec, detects        (every op = an `mpc`
 0.5s redraw)   track change, fires               subprocess, ~0.8 ms)
   │            full-listen, refills, seeds
   │            the session vector)
   ├──▶ vibe_readout.py     descriptor_bank.py → top-3 z-scores + drift
   ├──▶ session_history.py  what played · ♥ ⏭ ✓ · the ↑↓ cursor
   │        (both display-only: nothing behind the display reads them)
   ▼
feedback_handler.py ──┬──▶ session_state.py   short-term vibe vector
                      │        └──▶ manifold.py   turnover · solved λ · snap
                      ├──▶ user_taste.py      long-term taste → data/state/*.npz
                      ├──▶ exploration_controller.py   scalar 0.1–0.7 → weights + τ
                      └──▶ queue_manager.py ──▶ track_selector.py ──▶ track_library.py
                             (one ahead)          (rank-Boltzmann over ~100 candidates)
```

### The selection loop, precisely

1. `TrackLibrary.get_candidate_pool()` takes the top ~150 tracks nearest the session vector and the
   top ~150 nearest the taste vector, interleaves them, truncates to 100.
2. `TrackSelector._calculate_score()` scores each as
   `α·session_sim + β·taste_sim + γ·novelty + δ·anti_repetition`. Those four are the whole score;
   `config.validate()` enforces that they sum to 1.0 and raises `ValueError` (not `assert`) if not.
3. Weights come from `ExplorationController.get_weights()`, which shifts mass from session/taste into
   novelty as the exploration scalar rises. β is additionally ramped from 0 as taste updates
   accumulate, with the unearned share going to the session term.
4. **One track is drawn by Boltzmann sampling over rank** — `p(i) ∝ exp(−i/τ)`, τ set by the
   exploration scalar. Rank-based, so no recalibration when the score scale moves.
5. `QueueManager.ensure_one_ahead()` keeps exactly one track ahead, with `consume on` so MPD pops
   finished tracks itself — making the refill condition `len(playlist) < 2`.

Before anything has played both vectors are zero and step 1 returns an empty pool, so the first track
is a uniform random draw rather than an arbitrary `argpartition` ordering.

### What feeds back

| Event | Trigger | Effect |
|---|---|---|
| **Full listen** | ≥75% of duration played (`config.full_listen_fraction`) | Session vector updated (the primary driver). Taste +0.02. Exploration −0.02. The only thing that increments `tracks_played`. |
| **Skip `[N]`** | Keypress | Taste −0.05. Exploration +0.05. Session vector repelled from the consecutive-skip-run centroid by a magnitude **solved for a pool-turnover target that escalates with run length** (5% → 20% → 50% → 85%), projected back onto the manifold from the second consecutive press. Lookahead replaced, then exactly one advance. Measured turnover printed to the console. |
| **Pass `[V]`** | Keypress | **Nothing to the model** (audit G1): no session repel, no taste penalty, no exploration change, no escalation counter. Advances into the already-queued lookahead — add-before-advance, exactly one advance, no `play()` — and marks the passed track `»`. "Not this song, keep the vibe." |
| **Like `[L]`** | Keypress | Taste +0.10, saved immediately. Adds `♥`. |
| **Un-like `[L]`** | Keypress on an already-liked track | Removes the like from `feedback_history` and **recomputes** the taste model from what remains. See §4. |

### What persists

`user_taste.npz` (on every like and at exit), `exploration_state.json`, `feedback_history.json`,
`play_history.json` (carrying `play_history`, `current_index` and `recent_history`, so
anti-repetition survives a restart). The session vector is intentionally ephemeral. All of it is
checkpointed every `config.checkpoint_every_n_tracks` (5) full listens, not only at exit.

**Nothing in `data/state/` is worth preserving.** It is all regenerated by listening and none of it is
committed. Do not write migration code, backward-compat shims, or "reset carefully" logic for it.
`data/embeddings/` is different — it is a five-minute rebuild (`python3 src/generate_embeddings.py --force`),
not a throwaway.

---

## 3 · The vector space

Settled and not to be re-derived without re-measuring everything downstream:

- **674 tracks**, 512-d, full-coverage deterministic windows (consecutive 10 s, last aligned to the
  end), mean-pooled. Embedding the same file twice gives a bit-identical vector for a fixed batch size.
- **Centred on load.** CLAP's vectors occupy a narrow cone: two *unrelated* tracks sat at cosine 0.67
  raw. Subtracting the library centroid moves random pairs to **+0.011** and gives the scoring weights
  the range they always assumed. The centroid ships in the artifact; centring is applied at load.
- **49-word CLAP descriptor bank** with per-descriptor mean/std over the centred library, so the
  readout reports z-scores rather than raw text-tower dot products.
- 690 of MPD's 692 entries are audio; 674 have embeddings. The 16 without are one corrupt album
  (Jimi Hendrix, *Electric Ladyland*), listed with their exceptions in `data/embeddings/failed.txt`.

---

## 4 · Invariants — what is easy to undo

**This is the section to read before changing anything.** Each of these cost real debugging to
establish, and each would silently regress.

### MPD is the user's machine

- **Playback modes are the user's state.** The app forces `random`/`repeat`/`single` off and
  `consume` on, but it must log what it changed and restore the originals on *every* exit path
  including SIGTERM. Wired into `_shutdown()`, the signal handler **and** an `atexit` hook. Leaving
  someone's MPD in consume mode is a real side effect on their system.
- **`mpc listall` is the single source of truth for track keys.** Do not enumerate the filesystem for
  anything MPD will be asked to play. The embeddings are stored under these exact strings.
- **A skip must add the replacement *before* it advances.** `mpc next` off the last remaining track
  empties the queue and stops MPD, and a later `mpc add` does *not* restart it. Advance-then-add kills
  the session silently, and the only recovery is a `play()` call that the skip path forbids. There is
  no `play()` anywhere in `skip_current_track()`, which is what makes the double-advance impossible by
  construction rather than by care.
- **The album-art child process must be terminated explicitly.** `clear()` removes the image and
  leaves `ueberzugpp` running. `AlbumArtRenderer.shutdown()` is called from `_shutdown()` and from the
  signal handler, and **cannot raise** — it runs before `request_exit()`, the call that unblocks urwid
  and makes the rest of the shutdown reachable.

### The manifold

**Never blend an audio-space vector toward a text embedding, or toward a random direction.** CLAP's
towers don't share a cone: a random 512-d direction is 0.090-similar to real music where a session
vector is 0.787 and a real track is 0.748. `manifold.py` owns this — any large displacement is
projected back with `snap()` = `normalise(mean(top-25 library embeddings by dot(E, v)))`.

- **`snap()` is a move, not a projection.** It has a turnover floor of its own (~8%), so applying it
  to a single small skip overshoots a 5% target. Gated at run length ≥ 2.
- **Solve *through* the snap, not before it.** λ chosen against the un-snapped vector and snapped
  afterwards let a second consecutive skip land back where the run started — measured live at 1%
  turnover against a 20% target. `solve_repulsion(..., snap_result=True)` is not an optimisation.
- **"Still music" has no fixed threshold**, only the library's distribution: real tracks span 0.427 to
  0.961 on this measure. The assertion that means something is "no worse than the least typical real
  track" (p1 = 0.463).

### The display

- **The Now Playing box is `('pack', …)`, not `('weight', 3, …)`.** Weighting it raises `WidgetError`
  on every terminal shorter than 33 rows — a defect that shipped through three stages and 311 tests
  because nothing had ever called `render()`. Packing also makes the panel's height independent of the
  terminal, which is what the derived album-art geometry rests on.
- **The album-art geometry is derived from the widget tree,** not hand-counted. Two of the four
  original constants were wrong. Same for the `[I]` overlay's inner size, which asks
  `Overlay.calculate_padding_filler()` / `top_w_size()` rather than recomputing the `("relative", 70)`
  arithmetic.
- **The drift figure is a count of held words, not a cosine.** The cosine has p10 = 0.948 and median
  = 0.989 over 40 real sessions, so it reads as "0.99" forever — the same compressed-scale defect the
  readout exists to fix. The cosine is still computed and shown in `[I]` with its distribution beside
  it. See §7 before changing this back.
- **The display layer owns its own state, deliberately.** `vibe_readout.py` and `session_history.py`
  are pure modules that nothing behind the display reads. Putting either behind the display would
  split a display concern across a component that cannot see it.
- **Any code path that blocks the loop stops observing the session.** `[I]` handles this with
  `screen.set_input_timeouts(max_wait=0.5)` plus `_sync_session_state()` on every timeout wake —
  without which a track that starts *and* finishes while the overlay is open never reaches the history
  panel. **A new modal must do the same.**
- **Both interfaces share one binding table.** `decode_key()`/`decode_keys()` turn terminal bytes into
  urwid's key names and `_handle_input` dispatches for both. A binding cannot exist in one interface
  and not the other. Do not reintroduce a second `if key == …` ladder.

### The taste model

- **Un-liking is a replay, not a negative update.** `_update` is a normalised EMA, so subtracting
  `taste_update_like` is not its inverse. The asymmetry is ~10⁻⁴ and irrelevant; the real failure is
  that **a subtraction cannot un-seed a model**. From zero, one like normalises to the track itself;
  subtracting `0.1·e` from `e` gives `0.9·e`, which normalises back to `e`. Retract your only like and
  a subtraction leaves your long-term taste pinned at unit strength to the track you just rejected —
  and that is the first retraction any new listener makes.
- **The replay is gated on `UserTaste.explains()`.** It is exact only if the feedback history is a
  complete account of the model, and `_record_feedback` caps the history at 1000 events. Past the cap
  a blind replay moves the vector by up to 0.077 for reasons unrelated to the retraction. When the
  account is incomplete the retraction is display-only and the console says so.
- **A skip cannot seed the taste model.** From zero, one negative update normalises to `−track` at
  unit length: a full-strength claim from a single rejection. This did **not** become redundant when
  the β ramp landed — β gates the *score*, never *retrieval*, and the candidate pool opens its taste
  half on `np.any(taste_vector)`. Retiring the guard would hand half the pool to "the tracks least
  like the one song you rejected".
- **A zero taste vector is inert in retrieval, not just scoring.** An all-zero query to `find_similar`
  returns an arbitrary slice of the library. Do not reverse this when β ramps in.
- **Loads are atomic.** `ExplorationController.load()` and `UserTaste.load()` read every field before
  assigning any. Assigning as they went left a truncated file half-applied while returning `False`,
  so the caller believed nothing had been read.

### Diagnostics

**stderr is swallowed while the TUI runs.** `data/dj.log` is the durable copy — read it, not the
5-line console panel.

---

## 5 · Verified MPD semantics

`FakeMPD` is built to this table, and `tests/test_fake_mpd.py` asserts the double against it row by
row — because a double built from the assumptions that caused the original queue bug would reproduce
the bug and pass. **Extend the table by measuring against a live MPD, not by reasoning from the
protocol.** Verified against MPD 0.24.0 / mpc 0.35, `consume on`, `random`/`repeat`/`single` off.

| Behaviour | Verified result |
|---|---|
| The **currently playing track stays in the queue** | `mpc status` reads `#1/4` while playing the first of four. Consume removes a track when you *leave* it, not when you start it. |
| Position is always `#1` | After each removal the new current track is `#1/N`. "How many ahead" is `len(playlist) − 1`; there is no position to parse. |
| Natural end consumes | 3 → 2, the next track begins at `#1/2`. |
| `mpc next` consumes | 4 → 3. A skip and a completion look identical to the queue. |
| `mpc del 2` removes the **lookahead** | The current track keeps playing, uninterrupted, at the same position. |
| **`mpc next` on the last remaining track empties the queue and stops** | `playlist` 1 → 0, state `stopped`. |
| **Adding to a stopped queue does not start it** | `mpc add` on an empty stopped queue leaves the state `stopped`. Nothing recovers on its own. |
| **`mpc next` while *paused* consumes and resumes *playing*** | It does **not** stay paused. The shipped skip path advances and re-pauses. |
| `mpc pause` is idempotent, not a toggle | Which is what makes that re-pause safe. |
| `mpc next` on a stopped player | `MPD error: Not playing`; the queue is unchanged. The skip path guards on state. |
| `mpc del N` past the end of the queue | Exits 1 (`song number does not exist`). `replace_next()` relies on this when the queue holds only the current track. |

During playback `len(playlist)` is **2** (current + lookahead), so refill when it is `< 2`. Measured
at depth 2 in 30 of 30 mid-track samples during a live 30-track run.

---

## 6 · Data artifacts

Both produced by one generation run and validated on load. `embeddings_io.validate_embeddings()`
enforces every row; `TrackLibrary` refuses a file that fails it; `tests/test_embeddings_io.py` breaks
each field in turn to prove the check is real.

### `data/embeddings/track_embeddings.npz` — 45.5 MB

| Key | Shape / type | Notes |
|---|---|---|
| `schema_version` | `int` | Currently `2`. Refuse `< 2` — a v1 file has no centroid and would be scored on an uncentred space. |
| `track_files` | `(N,)` unicode | Keys exactly as `mpc listall` returns them. |
| `embeddings` | `(N, 512)` float32 | Pooled, **uncentred**, L2-normalised. Kept raw so the centroid can be recomputed if the library grows. |
| `centroid` | `(512,)` float32 | `mean(embeddings, axis=0)`. Applied at load: `normalise(E − centroid)`. |
| `window_offsets` | `(N+1,)` int32 | CSR-style index into `windows` — track *i* owns `windows[offsets[i]:offsets[i+1]]`. |
| `windows` | `(ΣW, 512)` float32 | Per-window embeddings, L2-normalised. 24,494 rows. Lets pooling be re-decided without regenerating. |
| `metadata` | JSON string | Model, transformers/torch versions, date, device, window scheme, timing, window-RMS percentiles. **JSON, not a dict** — a dict in an `.npz` is a pickled object array and forces `allow_pickle=True` on every read. |

Also written: `data/embeddings/failed.txt`, one line per failure with its exception.

### `data/embeddings/descriptors.npz` — 93 KB

Generated in the same run, because the z-score baselines are measured against the centred library and
building it separately invites the two to drift apart.

| Key | Shape / type | Notes |
|---|---|---|
| `schema_version` | `int` | |
| `labels` | `(D,)` unicode | Post-validation; near-zero-variance descriptors already dropped. 49 of 49 survived. |
| `prompts` | `(D,)` unicode | The rendered prompt (`"This is a recording of {} music."`), kept so the template is auditable. |
| `text_embeddings` | `(D, 512)` float32 | CLAP text-tower output, L2-normalised. |
| `mean`, `std` | `(D,)` float32 | Per-descriptor over the centred library — the z-score baseline. Stored rather than computed at startup so the correct path is the only path. |

**No `anchors` key.** Nothing needs them. They are one line to derive if free-text steering is ever
built; do not add them speculatively.

---

## 7 · Measured reference numbers

Reproduce rather than inherit. Every stage that re-measured found something material.

### Similarity scale

```
  random pairs, raw          0.670       random pairs, centred        0.011
  self-similarity            1.000       a random 512-d direction     0.090
```

### On-manifold quality across all 674 real tracks

```
  p0 0.427   p1 0.463   p5 0.541   p25 0.667   p50 0.752   p75 0.883   p100 0.961
  an ordinary session vector 0.787
```

### `[N]` turnover depends on how settled the session is

Any single number for "what does a skip do" is a statement about one point on this curve.

```
  tracks played before the skip    mean turnover    cos(new, old)
             1                         25.8%            0.989
             3                          7.3%            0.990
             6                          4.6%            0.992
            12                          2.9%            0.996
            24                          2.7%            0.997
```

### The escalation as it runs (λ solved against the post-snap vector)

```
  press   target   median λ    turnover p10 / median / p90    min quality
    1       5%       0.28         5.0%    5.0%    7.0%           0.588
    2      20%       0.75        23.9%   29.0%   64.8%           0.576
    3      50%       0.90        72.0%   83.0%  100.0%           0.534
    4      85%       1.28       100.0%  100.0%  100.0%           0.563
    5      85%       1.83       100.0%  100.0%  100.0%           0.619

  presses that moved BACKWARDS vs the press before: 0 of 160
  no press went below the library p1 floor (0.463)
```

The 85% row is why `snap()` is not optional above the first press; the 5% row is why it is not applied
*to* the first press.

### The τ map

```
  exploration    τ      p(rank 0)
     0.1        1.0        63%
     0.4        7.5        12%
     0.7       15.0         6%
```

### Descriptor drift over a real session (40 sessions × 30 tracks)

```
  words held (of 3)     0: 1%    1: 12%    2: 42%    3: 45%      mean 2.31
  cos(z_now, z_5ago)    min 0.721   p10 0.948   p50 0.989   p90 0.997   max 0.999
```

Ninety per cent of ordinary listening sits in the top 5% of the cosine's nominal range. **This is why
the readout ships the word count.**

### Un-liking: subtraction vs replay

```
  scenario                        cos(subtract-0.1, truth)
  like is the 2nd event             min 0.999900   median 0.999953
  settled model, 20 events after    min 0.999871   median 0.999959
  like is the only event            min 0.000000   median 0.000000   ← the finding

  replay of the full history        bit-identical at 10 / 50 / 200 / 1000 events

  history truncation (cap = 1000 events)
   999 lifetime events,  999 retained: cos = 1.000000000000   complete
  1001 lifetime events, 1000 retained: cos = 0.994142650376   truncated
  1400 lifetime events, 1000 retained: cos = 0.923200176638   truncated
```

The `explains()` discriminator needs no calibration: it is exact reproduction against ≤ 0.994, six
orders of magnitude of margin.

### Generation cost

```
  GPU (RTX 3070)   5 min 23 s   674 tracks, 24,494 windows, 75.8 windows/s
  CPU (12 threads) ≈ 25–35 min  audio encoder 17.0 win/s vs the GPU's 333

  model cache after a first run   1,232,327,859 B = 1.15 GiB
```

The cache is twice the model's size because the repo's `main` carries only `pytorch_model.bin`
(614.5 MB) and transformers ≥ 5 also fetches the safetensors conversion from `refs/pr/3` (614.4 MB).
Both stay cached. `EmbeddingGenerator.MODEL_CACHE_MB` / `ARTIFACT_MB` are derived from this, and
`tests/test_documented_numbers.py` holds the README, `start.sh` and the pre-flight check together —
they originally stated three different sizes because nothing bound them.

Throughput detail: batching buys ~30%, threading the mel extraction buys 2×. On GPU the bottleneck is
CPU mel extraction, so `--workers` matters more than `--batch-size`; on CPU the encoder dominates.

---

## 8 · Testing

`python3 -m pytest tests` — **542 tests, green, ~18 s.**

The suite is behavioural, not existence checks. Two of the worst defects in this project's history
survived a green suite of the latter. What that means in practice, and what to preserve:

- **`FakeMPD` is itself under test.** `tests/test_fake_mpd.py` asserts the double against §5 row by
  row. It has already earned that: a fixture defaulting to `consume off` silently put every component
  back in the broken world, and the replay-gap test caught it.
- **Tests drive the real methods.** `test_skip_path.py` calls the actual
  `AdaptiveDJWithTUI.skip_current_track` and asserts on FakeMPD's **call log** — one `next`, no
  `play`, every `add` before the advance. A test that mirrors the ordering it is checking proves only
  that the mirror is self-consistent.
- **Claims about the library are tested against the library.** `test_skip_escalation.py` skips if
  `data/embeddings/` is absent, because whether a turnover target is reachable at all is a property of
  the collection's structure, not of the solver.
- **Rendering is the point.** `test_art_geometry.py` renders the real frame at seven terminal sizes
  and *locates the art placeholder in the canvas* rather than asserting the arithmetic back at itself.
  The first test ever to call `render()` found a crash on every terminal under 33 rows.
- **The fallback text mode is driven through a pty.** Two traps if you extend it: `tty.setcbreak` uses
  `TCSAFLUSH`, so keys written before the loop starts are **discarded** — send them from the first
  tick; and a burst is consumed by one read, so a key meant to dismiss `[I]` must arrive *after* the
  page opens. A watchdog plus a "did every key get sent" assertion stops the harness passing vacuously
  on a timeout.
- **The suite must never write to `data/state/`.** It did, for a long time — `process_like()` saves to
  `config.taste_file` — and a green run replaced a real listener's taste model with a fixture's. The
  autouse `_isolate_state_files` fixture redirects all four paths. **Do not remove it, and do not add
  a state file without adding it there.**
- **Deletions are guarded.** `test_deletions.py` fails if a removed symbol returns as live code
  (comments explaining a removal are exempt), if `tests/` stops being tracked, if track keys are
  enumerated from the filesystem, or if anything under `data/` other than `.gitkeep` gets committed.

Acceptance properties that must stay asserted: bit-deterministic embedding for a fixed batch size,
self-similarity exactly 1.0, post-centring random pairs at +0.011.

`tests/test_clap_pipeline.py` loads the real CLAP checkpoint and **skips** if it is not already in the
HF cache — a test that silently downloads 700 MB is not a test anyone can run.

### Fixtures (`tests/conftest.py`)

`rng` (seeded) · `library` (in-memory `TrackLibrary`) · `make_artifact` (schema-correct `.npz` with
any field overridable) · `fake_mpd` (**consume on**, the state the DJ forces) · `dj_parts` (the real
selection stack wired to `FakeMPD`) · `fake_art` / `stub_bank` / `dj_stub` / `tui` (the real
`AdaptiveDJTUI` with an injected art renderer, so building the tree does not spawn a ueberzugpp
child) · `_isolate_state_files` and `_restore_stderr` (both autouse).

---

## 9 · Running and verifying it

```sh
./start.sh                                                # the only entry point; launches src/main_tui.py
python3 src/generate_embeddings.py --help
python3 src/generate_embeddings.py --stats                # raw vs centred similarity distributions
python3 src/generate_embeddings.py --describe "Bathory"   # top descriptors for a track
python3 src/generate_embeddings.py --compare-templates    # re-run the prompt-template measurement
```

A full regeneration takes ~5.5 minutes and **must not be run casually while testing** — it rewrites
the artifact every downstream number depends on.

`main_tui.py` is the only orchestrator. There is no demo/random-embedding path anywhere; it was
removed on purpose, because random vectors make every similarity, novelty score and learned preference
meaningless while the interface keeps presenting them as insight.

### Driving it for real

urwid needs a tty, so end-to-end verification runs under `pty.fork()` — **and the pty needs an
explicit `TIOCSWINSZ`**, or it reports 0×0, urwid draws nothing, and the run looks like it worked
while every captured frame is empty. Playback can be accelerated with `mpc seek 99%` per track.

**Always snapshot the user's MPD queue, modes and volume first and restore them after** — the app
itself only restores the modes. Save the queue with `mpc -f %file% playlist`, not the default format;
the default prints `Artist - Title`, which you then have to search back into file paths.

Delete `data/state/` before measuring anything about a cold start.

### Controls

| Key | Action |
|---|---|
| `Space` | Play / Pause |
| `N` | Skip — a rejection; escalates if you keep pressing it |
| `V` | Pass — advance without changing the vibe; marks the track `»`, moves nothing in the model |
| `L` | Like; press again on a liked track to un-like |
| `↑` / `↓` | Move the cursor through the session history |
| `Enter` | Replay the track under the cursor — it becomes `↓ next:` |
| `,` / `.` | Volume down / up |
| `←` / `→` | Seek ∓10 s |
| `I` | Model state (descriptors, sampling, taste, exploration, weights); `↑↓` scrolls |
| `Q` | Quit |

The footer, the README table and both interfaces advertise exactly this list, and tests drive every
row through the real key handler.

---

## 10 · Finding-ID glossary

Roughly 450 code comments cite these IDs. This is what each one meant. **All are closed.**

| ID | What it was |
|---|---|
| **C1** | The queue never refilled; playback stopped dead after 10 tracks. |
| **C2** | MPD's `random` mode silently discarded every ordering decision the DJ made. |
| **C3** | Audio fingerprints were non-deterministic ~10 s random crops, not track representations. |
| **C4** | `[V]` always threw away the first track of the new vibe (double advance). Dissolved with `[V]`; its "one advance, no `play()`" constraint is still enforced and tested. |
| **C5** | The similarity scale was compressed (anisotropy); every scoring constant was calibrated against a range that did not exist. |
| **H1** | The mood word in every vibe description was mathematically pinned to "eclectic". |
| **H2** | The "Upcoming Queue" panel listed tracks that had already played. |
| **H3** | Ctrl-C and SIGTERM neither exited nor saved. |
| **H4** | Skipping did not change what played next. Dissolved by D1. |
| **H5** | The day-of-week exploration modifier was dead code. |
| **H6** | Selection was strictly greedy — "exploration" never explored. |
| **H7** | `mpd_controller.py` defined eight methods twice; the surviving `add_track` swallowed failures. |
| **H8** | Album-art geometry was hardcoded to a layout that was about to change. |
| **H9** | Neither `[V]` nor `[N]` changed much of what you heard, and `[V]` aimed off the manifold. |
| **N1** | The layout raised `WidgetError` on any terminal shorter than 33 rows. Found in Stage 3; predates the audit. |
| **M1** | The test suite was green theatre and not in the repository. (a: untracked/phase files, b: no `FakeMPD`, c: no round-trips, no fallback-mode coverage.) |
| **M2** | Two divergent orchestrators; the stale one was what the setup helper told you to run. |
| **M3** | `mpd_music_directory` was an undocumented, unvalidated requirement that worked by accident. |
| **M4** | Two different sources of truth for track keys, with no reconciliation. |
| **M5** | No embedding-dimension validation on load. |
| **M6** | (a) State file misnamed; (b) anti-repetition history never persisted despite a comment claiming it was. |
| **M7** | Setup documentation contradicted itself on every number; macOS was claimed but unsupported. |
| **M8** | `--batch-size` was advertised, accepted, and completely ignored. |
| **L1** | "The SIGWINCH handler is never invoked." **The finding's conclusion was wrong** — urwid 3.0.5 chains to the previous handler rather than replacing it, so it had been running all along. Acting on the fix direction closed a recursion. |
| **L2** | Kitty and sixel album art fought urwid for the screen; only ueberzug works. Both paths deleted. |
| **L3** | Album-art geometry hardcoded. Elevated to H8. |
| **L4** | Hearts vanished on restart even though likes were already on disk. |
| **L5** | No log file, and stderr was swallowed while the TUI ran. |
| **L6** | Polling instead of `mpc idle`. **Declined**, not fixed — see §11. |
| **L7** | Cold start injected a random direction at 30% weight. |
| **L8** | Dead API surface across most modules; also, no way to un-like. |
| **L9** | Assorted small traps: `validate()` built on `assert`, the weight invariant, a dead `[I]`, `select_track` mutating its caller's set, disagreeing keybinding docs, `.gitkeep` scaffolding that did not ship, and an orphaned ueberzugpp child. |
| **D1–D8** | Design decisions taken before the rewrite: queue depth 10 → 1 (D1), force `consume on` (D2), discard existing embeddings and learned state (D3), delete invented heuristics rather than recalibrate them (D4), use the CLAP text encoder (D5), remove the time-context subsystem (D6), significant refactoring is in scope (D7), delete `[V]` and escalate `[N]` instead (D8). |

---

## 11 · Deferred, declined, and open observations

Nothing here is a defect with a known fix being ignored. These are the things a future reviewer should
know were seen and decided on.

### Deferred by choice

1. **Free-text steering.** "Something nocturnal and sparse": embed the text, take its top-20 library
   tracks, blend toward that centroid. The descriptor machinery already does this mechanically. **The
   same trap applies** — blend toward the *audio* centroid of matching tracks, never toward the raw
   text vector.
2. **Per-window representations.** The full window matrix is persisted specifically so this can be
   revisited without regenerating. Medoid clustering or max-over-windows would let a track match on
   one section — better recall, more whiplash. Worth an A/B.
3. **Re-tuning the control constants.** Exploration step sizes and taste update rates were chosen
   against the compressed similarity scale and behave differently now it is centred. The constants
   that made *claims* are already deleted; these only shape behaviour. Listen first, then tune once
   with real data.
4. **τ_max = 15.** The one genuinely new constant, still uncalibrated by listening. Raise it until
   unattended sessions start feeling incoherent, then back off. `[I]` reports the τ in force.
5. **`previous_track`.** Impossible via MPD under `consume on`; would need re-adding from the app's
   own history. No binding exists, so nothing regresses.

### Declined

- **L6 — `mpc idle`.** Steady state is ~12 `mpc` spawns/sec at 0.8 ms each, about 1% of one core. It
  matters only against a remote `MPD_HOST`, where each call is a TCP round trip. At depth 1 the one
  queued track gives the 2 s poll minutes of runway.

### Open observations — measured, not fixed

- **The turnover schedule means little until the session settles.** The first `[N]` press turns over
  **100%** of the pool against its 5% target on a cold session. Consistent with the settling curve in
  §7, but it is the extreme of it, and it was the first time the escalation had been driven from a
  genuinely empty `data/state/`. **Worth a measurement of its own before the schedule is ever
  retuned.**
- **The Session panel draws no content rows at 80×24.** The header, the packed Now Playing box, the
  console and the two-row footer consume all 24. The layout renders — nothing crashes — but the panel
  is a border. Fixing it means re-weighting a tree whose packing *is* the fix for N1.
- **macOS is untested.** `start.sh` no longer uses any bash 4+ syntax, so it would probably run, but
  nothing has been run there and album art cannot work (ueberzug/ueberzugpp are X11/Wayland only). The
  README says Linux and says macOS is untested rather than claiming support.
- **`[I]`'s fallback page and the urwid overlay are two pieces of code.** They agree on content; only
  the urwid one runs the session bookkeeping while open, because the fallback's own loop is the thing
  that would be blocked.

### Reversal notes for the non-obvious decisions

| Decision | Why | Cost to reverse |
|---|---|---|
| Force `consume on` | Makes the refill condition `len(queue) < 2` with no `#N/M` parsing. | Moderate — the alternative is parsing `mpc status` and leaving consume alone: more code, no side effects. |
| Rank-Boltzmann sampling | Argmax reproduces a byte-identical session; score-softmax needs recalibration whenever the score scale moves. Rank is scale-invariant. | Trivial — one function. |
| Turnover reported against the skip run's start, not the previous press | "How much has changed since I started skipping" is the question actually being asked. | Trivial — one stored anchor vector. |
| Drift as a word count, not the cosine | The cosine's p10 is 0.948 over 40 real sessions. | Trivial — `drift()` returns both. Re-read §7 first: the reason is the distribution, not the presentation. |
| Drift and history stores live in the display layer | Neither is read by selection. Putting a display concern behind the display splits state across a component that cannot see it. | Low, but there is no reason to. |
| `QueueManager.requeue_next()` adds before it deletes | It already knows its track, so it has nothing to exclude. Appending first means a refusal from MPD leaves the session untouched. | Trivial, but it reintroduces a window where a failed add leaves the queue one deep. `replace_next()` **cannot** use this order — it has to exclude the entry it is dropping. |
| `skip_turnover_schedule = (0.05, 0.20, 0.50, 0.85)` | The only *input* to the escalation — λ is solved for it at every press. Stated in units the listener can verify. | Trivial, and it is the right knob: add a row rather than fudging λ. The console prints what each press measured. |
| `skip_snap_from_run_length = 2` | Both directions measured: below it `snap()` overshoots; at and above it an unguarded λ for the 85% target lands below the library's p1 floor 100% of the time. | Low, but re-measure before moving it — both failure modes are silent. |
| `RMS_GATE = 0.01` | Not chosen — the window-RMS distribution is bimodal and the gate sits in the empty band. | Trivial, and re-measurable: the distribution is printed by the run and stored in the artifact metadata. |
| `STD_FLOOR_FRACTION = 0.5` | Relative to the library's own median std, so it needs no recalibration when the model, template or collection changes. | Trivial. The run prints every descriptor's std. |
| `taste_ramp_updates = 20`, `minimum_sampled_pool = 4`, `checkpoint_every_n_tracks = 5`, `minimum_mpd_coverage = 0.5` | Control constants: they shape behaviour without asserting a fact. | Trivial. |

---

## 12 · Environment

- Fedora, Python 3.14.6, numpy 1.26.4, urwid 3.0.5, transformers 5.1.0, torch/torchaudio
- MPD's real `music_directory` is `/mnt/storage/music`, read from `~/.config/mpd/mpd.conf` by
  `music_directory.py` rather than assumed. `/var/lib/mpd/music` is a symlink to it — the reason the
  old hardcoded default worked, and the reason it hid.
- Embedding generation runs on an RTX 3070 in ~5.5 minutes.
- **Album art works only via ueberzugpp** (or classic ueberzug). The kitty and sixel paths are
  deleted; that user is told what to install.

---

## 13 · How it got here

| Stage | What it did |
|---|---|
| **0** | Cleared the ground: deleted the second orchestrator, the time-context subsystem, the invented heuristics and the demo-embedding paths; tracked `tests/`; added `data/dj.log`. |
| **1** | Rebuilt the signal: full-coverage deterministic embeddings, the centroid, schema validation, `mpc listall` as the only key source, the 49-word descriptor bank. |
| **2** | Made it play: depth-1 queue with `consume on`, the one skip path, mode force/restore across every exit, rank-Boltzmann sampling, the solved-λ escalation. |
| **3** | Made it legible: descriptor readout, Session panel, derived album-art geometry, the first tests that render — which immediately found N1. |
| **4** | Made it durable: persistence round-trips, the fallback mode under a pty with a shared binding table, un-like as a replay, kitty/sixel deleted, the ueberzugpp child terminated, the setup numbers measured and bound by a test. |

Each stage re-measured rather than inheriting, and each time it changed something material. Stage 4
was the first where the thing that turned out to be wrong was a *finding* rather than a number — see
§1's corollary.
