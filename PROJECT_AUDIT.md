# Adaptive Session AI DJ — Audit & Continuation Brief

A terminal DJ for MPD that learns your taste from audio embeddings. This document records what
the system actually does versus what it claims, the defects that block its core promise, the design
decisions taken since the first audit, and where to pick the work back up.

| | |
|---|---|
| **Repository** | `/home/gumibo/misc/programming/projects/mpd_tui_ai_dj` · audited at `master` HEAD `8dc4275` |
| **Scope** | 23 tracked files, ~6,200 lines Python + Bash. All source read in full; claims verified by execution where possible. |
| **Environment tested** | Fedora (Linux 7.1.3), Python 3.14.6, numpy 1.26.4, transformers 5.1.0, torch 2.10.0+cu128, urwid 3.0.5, MPD/MPC present, ueberzugpp present, RTX 3070. Library: 692 MPD entries, 674 embedded (was 616 at audit time). |
| **Audit date** | 22 July 2026 |
| **Revision date** | 22 July 2026 — design decisions folded in, see §0 |
| **Progress** | **Stages 0–3 complete** (Stage 3: 23 July 2026). Stage 4 outstanding. See §0b for what landed and where it deviated from the plan. |

---

## How to use this document

This is the working brief for the rewrite. It is written to be picked up cold, with no prior
conversation in context.

- **§0** is the decision record. Read it first — it changes the meaning of roughly half the findings
  below, and several findings' "fix direction" blocks were rewritten because of it.
- **§0b** is the implementation log, one entry per completed stage. It records what has actually
  shipped and — more usefully — the places where doing the work turned up something the plan had not
  anticipated, including several of the audit's own empirical claims that turned out to be wrong.
  **Stage 4's entry is the one to read if you are about to trust a finding**: acting on L1's fix
  direction exactly as written broke the application.
- **§2–§5** are the findings, each tagged with a status: `OPEN`, `NEW`, `DISSOLVED`, `RESOLVED by
  deletion`, `SUPERSEDED`, `ELEVATED`, `DONE`. Only `OPEN`, `NEW` and `ELEVATED` require work.
- **§6** is the status table — the fastest way to see what is still live. **As of Stage 4 nothing
  is: every finding is closed**, one of them (L1) with its own claim corrected, and one (L6) declined
  with a reason.
- **§7** specifies the target data artifacts (the `.npz` schemas). Build to these.
- **§8** is the ordered plan. Each stage has a definition of done. **Each stage leaves the
  application runnable** — there is no point in the sequence where it is half-migrated and broken.
- **§9** lists the judgement calls the plan rests on and what was deliberately deferred. Nothing is
  undecided.
- **§10 / §10b / §10c / §10d / §10e** are the evidence appendix, one per stage that measured anything,
  **newest wins where they overlap**. §10 is the pre-Stage-1 library and is provenance only. §10b is
  the embeddings, the centring and the descriptor bank. **§10c is the live record for anything about
  skips, selection or the manifold** — it re-took §10b's two "Re-measured for Stage 2" tables by
  driving the shipped code, and four of their figures did not reproduce. **§10d is the live record
  for the display**: the descriptor drift distribution, the widget-tree geometry, and the terminal
  sizes at which the pre-Stage-3 layout crashed. **§10e is Stage 4**: why un-liking is a replay and
  not a subtraction, what a first run actually costs on disk and on CPU, and the urwid source that
  shows L1's conclusion was wrong.

Where this document and the code disagree, the document describes the target. **Line references
throughout §2–§5 are to `HEAD 8dc4275`, the pre-Stage-0 tree**, and are now stale for every file
Stage 0 touched — they are kept as provenance for the finding, not as navigation. §0b lists what
moved.

---

## Bottom line

The architecture is sound and the code is unusually well commented for a personal project. The
intelligence core — session vector, taste vector, exploration controller, multi-factor scoring — is
all implemented and all wired together correctly.

But defects sit directly on the critical path, and each one independently prevents the product from
being what the README describes. The queue stops refilling after ten tracks. MPD's own random
mode silently discards the ordering the DJ computes. The audio fingerprints that everything else is
built on are non-deterministic ten-second crops. The vibe descriptor's mood word is mathematically
pinned to a single constant value.

Two more were found while planning the rewrite. The entire similarity scale the scoring constants were
tuned against is compressed into the top third of its nominal range (**C5**), which is why the weights
never seemed to do much. And neither key that is supposed to change direction actually does: `[N]`
turns over 0.3% of what you will hear per press, and `[V]` — the "give me something different" key —
buys about ten skips' worth of movement while pointing the session vector a third of the way into
noise (**H9**). `[V]` is deleted; `[N]` escalates on consecutive skips instead.

There is a pattern connecting almost all of them: **numbers chosen against a scale that was never
measured.** Entropy thresholds picked for a smaller dimension, a novelty formula assuming a range the
data never occupies, a "50% shift" that moves 29%, a temperature that would need re-tuning every time
the weights move. The rewrite's organising principle is to derive constants from the library's actual
distribution, or delete them.

None of this is architectural. The reason it survived is that the test suite reports 66 passing checks
while roughly half of them are hardcoded `True` literals.

**Where it stands now.** Stages 0–3 are done, and **the application does what the README describes.**
The ground is cleared, the invented vocabulary is gone, the vector space underneath everything is
trustworthy — 674 tracks embedded from full coverage, bit-reproducible, centred, named by a 49-word
CLAP descriptor bank — the player plays, and the display finally reads the bank.

Stage 2 closed every critical finding. The queue is one deep and refills as each track ends; MPD's
modes are forced, logged and restored on every exit path including SIGTERM; selection draws by
Boltzmann sampling over rank rather than taking the argmax; `[V]` is deleted and `[N]` escalates on
consecutive presses by a magnitude *solved* for an observable pool-turnover target. Verified against
the live MPD: a 30-track unattended run with no stall, no repeat inside the replay gap, and a queue
depth of exactly 2 in all 30 samples; four consecutive skips reporting 5% → 20% → 70% → 100% turnover
and moving from Björk to Watain; the user's playback modes restored byte-identically after a
`kill -TERM`.

Stage 3 closed the display. The vibe line names three descriptors z-scored against this library, the
Session panel lists what actually played with `♥` / `⏭` / `✓` marks, `↑↓` and `ENTER` scroll and
replay it, `[I]` reports both vectors' descriptors and scrolls, and the album art's position is
derived from the widget tree rather than hand-counted.

It also found a defect nobody was looking for, which is the argument for the stage's first line of
work. **The application raised `WidgetError` on every terminal shorter than 33 rows** — including the
default 80×24 — and had done since before the audit began. Nothing caught it because, across 311
tests, nothing had ever constructed the widget tree. It is fixed, reproduced live against the
pre-Stage-3 tree, and pinned by a test that renders at seven terminal sizes (**N1**).

What remains is Stage 4: durability, the rest of the README's contradictions, and the four small
open items in §6.

---

## 0 · Revision log — decisions taken 22 July 2026

These decisions were made after reading the original audit. They change what several findings mean,
dissolve four of them outright, and reorder the work. Everything downstream of this section reflects
them.

### D1 · Queue depth drops from 10 to 1

`config.queue_buffer_size` goes `10 → 1`. Exactly one track sits in MPD ahead of the current one.
The TUI shows it as `next:`.

**Why.** The ten-track buffer was the single largest obstacle to the app being adaptive. Every one of
those ten tracks was scored under the weights that existed at the moment of generation, so any
feedback — skip, like, vibe shift — was inaudible for up to ten songs. It also made the "Upcoming
Queue" panel dishonest: it displayed a future that one `[V]` press would erase.

Depth 1 is the shallowest depth that still gives gapless playback. Depth 0 would mean MPD hits
`stopped` at every track boundary and waits for the 2 Hz poller to notice and shell out to `mpc add`
— roughly half a second to a second of silence between every song.

**What it dissolves.** H2 and H4 stop existing. The 5% trajectory-blend loop
(`queue_manager.py:119–129`) becomes unnecessary — the real session vector does that job one track
at a time, which is what it was always for. `planned_queue` / `currently_queued_in_mpd` dual
bookkeeping collapses to a single "is there one ahead?" check.

**What it makes worse.** H6 (strictly greedy argmax) goes from a quality issue to a blocker — see
that finding. C4 goes from "you lose one track of ten" to "playback stops dead," and its published fix
inverts: `[V]` must now advance, so the defect is `recalculate()` existing at all rather than the
`next_track()` call.

**What it removes from scope.** The queue is no longer a *display* concern. Its visibility role passes
entirely to the descriptor readout (D5) and the session-history panel. Depth 1 exists purely to keep
MPD from stalling between tracks.

### D2 · MPD `consume` mode goes on

Combined with D1 this makes queue management nearly trivial: with `consume on`, MPD removes each
track from the queue as it finishes, so `len(mpc playlist)` is exactly 2 during normal playback
(current + next) and the refill condition is `< 2`. No `#N/M` position parsing needed.

This is a change to the user's MPD state, so it is forced at session start alongside `random off` /
`repeat off` / `single off`, logged to the console panel, and restored on exit (see C2).

**Tradeoff:** with consume on, `mpc prev` cannot work — played tracks are gone from MPD's queue.
The app has no previous-track binding today (L8), so nothing regresses, but if one is wanted later it
must be implemented by re-adding from the app's own history rather than by delegating to MPD.

### D3 · The existing embeddings and all learned state are discarded

616 embeddings, `user_taste.npz` (15 updates), `exploration_state.json`, `feedback_history.json`,
`time_context.npz`. There is no accumulated value here — one 38-minute session on one evening — and
C3 means the taste vector was trained in a space that is being replaced. Delete all of it and
regenerate. This removes the "reset state carefully" caveat that appears throughout the original
audit's stage 2.

### D4 · Invented heuristics are deleted rather than recalibrated

Anything that dresses an arbitrary constant up as an insight comes out. Specifically:

| Heuristic | Location | Disposition |
|---|---|---|
| `vector_entropy` → `eclectic`/`diverse`/`cohesive` | `session_state.py:200–207` | **Delete.** Replaced by the CLAP descriptor bank (D5). |
| momentum thresholds `0.85 / 0.7 / 0.5` → `focused`/`flowing`/`drifting`/`exploring` | `session_state.py:186–193` | **Delete.** Replaced by measured descriptor drift (D5). |
| `stage` word: `warming up` / `building` / `deep in the zone` | `session_state.py:210–215` | **Delete.** It is a `tracks_played` counter wearing a costume. Show the counter. |
| `0.15 · time_sim` additive bonus | `track_selector.py:129–132` | **Delete** with the whole time-context subsystem (D6). |
| Random-direction cold-start taste vector at β=0.3 | `user_taste.py:27–34` | **Delete.** Ramp β from 0 to its configured value over the first ~20 updates instead (see L7). |

Also deleted, but for a different reason — it is provably wrong rather than merely arbitrary:
`vibe_shift_magnitude = 0.5`, which turns out to move the session vector less than the distance
between two random songs, in a direction that is not music. See **H9**.

Deliberately **kept**: the exploration control loop (±0.02 / ±0.05, bounds 0.1–0.7),
`penalize_similar`'s 0.15 nudge, the anti-repetition log decay. These are control constants, not truth
claims — they don't assert anything false to the user. They do need re-tuning after C5 changes the
similarity scale underneath them.

**The rule this follows:** a constant may stay if it shapes behaviour without asserting a fact. It
goes if it produces a claim the user reads ("eclectic", "deep in the zone", "50% shift") that the
system cannot back up.

### D5 · The CLAP text encoder gets used

The half of the model the project never touched. A bank of ~50 descriptor prompts is embedded once
with CLAP's text tower and cached; the session vector is scored against it and the top descriptors
become the vibe readout. This is the actual answer to "I want a clearer idea of what's being played"
— it was the reason the queue panel existed. Full specification in **H1**.

The bank drives the vibe readout and the `[I]` model inspector. §8 builds it in **Stage 1**, with the
embeddings, rather than in Stage 3 with the UI — it is a data artifact, and building it early retires
the risk that CLAP descriptors simply do not discriminate this library. If that turns out to be true,
H1's whole design needs rethinking, and you want to learn it from a script before a TUI is built on
top of it.

**The one trap:** CLAP's audio and text towers do not share a cone, so raw cross-modal dot products are
incomparable across descriptors — a naive top-3 returns the same three words forever. Two defences,
both mandatory and both enforced as Stage 1 steps: per-descriptor z-scoring against the library's own
distribution, and a build-time variance gate that drops descriptors carrying no information about
*your* collection. Specified in H1 and §8/Stage 1.

### D8 · `[V]` is deleted; `[N]` escalates instead

Measured against the real library, `force_shift`'s "50% vibe shift" turns over **9.3%** of the
candidate pool — about what ten presses of `[N]` achieve — and lands the session vector at **0.450**
similarity to the nearest real music, against 0.641 for an ordinary session vector and 0.085 for a
random direction. `[V]` is not a bigger gesture than the skip key; it is a comparably small one
pointed a third of the way at noise.

Its real job was clearing the ten-track queue, which D1 removes. The intent behind it survives and is
absorbed into `[N]`, which escalates on consecutive skips — repelling from the skip-run centroid by a
magnitude *solved for* an observable pool-turnover target, then projected back onto the manifold.
Full derivation and measurements in **H9**.

This deletes a key binding, two `SessionState`/`FeedbackHandler` methods, `set_high_exploration()`,
one config constant, and the descriptor `anchors` that a redesigned `[V]` would have needed.

### D6 · The time-context subsystem is removed

`time_context.py` (9 KB), `config.enable_time_context`, `config.time_context_weight`, the
weekday/weekend modifiers, the `[I]` overlay in its current form.

**Why.** It is the sole source of four separate findings (H5 dead modifier, M6a misnamed state file,
and two L9 items), its only live effect is an unvalidated 0.15 additive bonus that breaks the
weight-sum invariant, and it cannot be evaluated until the rest of the system works. It is a feature
built before its foundation was trustworthy.

The idea is not wrong and is worth revisiting once there is real listening history to test it
against. It is recorded in §8 as deferred, not rejected.

`[I]` is repurposed — see the TUI section of H1.

### D7 · Significant refactoring is in scope

There is no attachment to the existing code. Where a finding's fix is "patch this," but the honest
answer is "delete this and rewrite it smaller," take the second. This applies most to
`queue_manager.py` (which loses most of its reason to exist), `session_state.get_vibe_description`,
`main.py` / `setup_check.py` (M2 — just delete them, don't extract a `build_dj()` factory), and the
dead API surface in L8.

---

## 0b · Implementation log

### Stage 0 — complete, 22 July 2026

Net effect: **−1,462 lines of application code, +734 lines of tests**. Three whole files deleted
(`main.py`, `setup_check.py`, `time_context.py`) plus in-file removals across nine modules.

| Plan item | Landed as |
|---|---|
| **L5** | `_ConsoleCapture` tees every completed line to `config.log_file` (`data/dj.log`), append mode with a per-session banner. Opening the log is failure-tolerant — an unwritable path degrades to no log rather than killing the app, and a write error drops the handle instead of recursing through `stderr`. |
| **D6 / H5 / M6a** | `time_context.py` deleted. Ten config keys removed. `get_exploration_factor()` removed. The `time_context=` parameters removed from `TrackSelector.select_track` / `_calculate_score` / `QueueManager._generate_tracks`, the update hooks removed from `FeedbackHandler`, and the save/load hooks removed from `Persistence` (which no longer takes `session_state` at all). |
| **D4** | `get_vibe_description()`, `_update_vibe_trajectory()` and `vibe_trajectory` deleted, with a comment in their place recording *why* so it cannot be reintroduced by someone who thinks it was an oversight. `UserTaste._initialize_taste_vector()` now returns zeros. |
| **M2 / D7** | `main.py` and `setup_check.py` deleted. `start.sh`'s demo-embeddings branch deleted; `generate_dummy_embeddings()` deleted from `track_library.py`; `main_tui.py`'s "generate dummy embeddings now? (y/n)" prompt replaced with a hard exit pointing at `generate_embeddings.py`. |
| **H7** | All eight duplicate method bodies removed, bool-returning versions kept. `add_track` checks `returncode` and logs MPD's `stderr` on refusal. A test parses the class with `ast` and fails on any duplicate `FunctionDef` name, so the condition cannot silently return. |
| **L8** | Deleted: `MPDController.toggle` / `seek` / `get_all_tracks` / `get_track_metadata`, `TrackLibrary.has_track`, `SessionState.get_recent_average` / `get_similarity` / `reset`, `UserTaste.get_similarity` / `reset`, `TrackSelector.clear_history`, `Persistence.reset_all`. Retained per the plan: `previous_track` (with the consume-mode caveat written into its docstring), `save_embeddings`. |
| **M1a** | `test_*.py` removed from `.gitignore`; the three phase-test files deleted; `tests/` created and tracked — 9 files, 67 tests, green. |
| **cfg** | `validate()` raises `ValueError` with a message naming the offending key. Nine orphaned config keys removed; `log_file` added. |
| **D3** | `data/embeddings/*` and `data/state/*` deleted. Only the three `.gitkeep` files remain. |

**Verification.** 67 pytest tests pass. The application was additionally driven end to end against
the live MPD in a pty — launch, `[SPACE]` play, `[N]` skip, `[L]` like, `[I]` overlay, `[N]` skip,
`[Q]` quit — exiting 0, with the rendered frames captured and read. `data/dj.log` filled correctly,
feedback history and taste model round-tripped, and the taste vector was seeded by the like and
moved away by the skip. The user's MPD queue, playback modes and volume were snapshotted before the
run and restored after.

### Four things the Stage 0 plan did not anticipate

**1 · Zero taste vector needs two guards, not one.** D4 says "delete the random taste seed" and L7
defers the β ramp to Stage 2. Doing only that produces two new defects, both worse than what was
removed:

- `TrackLibrary.get_candidate_pool()` queries `find_similar(taste_vector)` for half the pool. With an
  all-zero vector every dot product is 0, so the ordering is whatever `argpartition` happens to
  produce — half the candidate pool becomes an arbitrary slice of the library, presented as
  preference. **Fix:** skip the taste half entirely while the vector is zero and fill from the
  session vector alone. This is exactly what L7 argues for in prose ("driven purely by what they are
  listening to right now"); it just also has to be true of the *pool*, not only the score.
- `UserTaste._update()` with a negative weight subtracts from zero and re-normalises, so the *first
  skip of a new user's life* sets the taste vector to `−track` at unit length — a full-strength claim
  from a single rejection, and a stronger one than the random seed ever made. **Fix:** negative
  updates are a no-op while unseeded. The counters still move; the vector does not. The first
  positive signal establishes it.

Both guards become redundant once L7's β ramp lands in Stage 2, but the pool guard should stay
regardless — β never gates which candidates are *retrieved*.

**2 · `[I]` could not wait for Stage 3.** D6 empties the time-context overlay, but the key stays
bound and the footer, the README and `start.sh` all advertise it. Leaving a key that silently does
nothing is the same class of dishonesty the rest of Stage 0 removes. It is now a minimal model
inspector — library size, session/taste/exploration/selector counters, and the live scoring weights —
which asserts nothing that is not measured. **H1d still stands**: Stage 3 adds the session and taste
top-descriptors and the effective τ ("choosing from ~top 7") to the same overlay.

**3 · Stage 0's definition of done contradicts D3.** "The app launches, plays" is not reachable after
"delete `data/embeddings/*`" — there is nothing to select from until Stage 1 regenerates. Verified
instead with a throwaway random-vector file, created outside the repo, used for one scripted session,
and deleted immediately. **This is not a licence to keep such a file around**; it proves the plumbing,
and nothing about selection quality. For Stages 1–4 the criterion should read "launches and reports a
clean library load"; playback verification belongs to Stage 2's 30-track unattended run.

**4 · One new finding, folded into M5.** `TrackLibrary.load_embeddings()` decides a file is CLAP with
`'clap' in model_name.lower()` — a substring match on an arbitrary metadata string. The smoke-test
file was named `SMOKE-TEST-RANDOM-NOT-CLAP` and was greeted with `✓ Loading CLAP embeddings`. M5
already calls for `schema_version` and dimension validation on load; add "model identity is checked
by equality against the expected checkpoint, not by substring."

### Stage 0 — deviations from the plan's letter

- **The vibe line shows `Session: N tracks played`, not a blank.** D4's own text says the stage word
  "is a `tracks_played` counter wearing a costume — show the counter," so the counter is what it
  shows. Same widget, same row count, so H8's geometry problem is unchanged and still Stage 3's.
- **`ExplorationController.reset()` was deleted too.** L8 did not list it because `Persistence.reset_all()`
  called it; deleting `reset_all` orphaned it. D7 covers this.
- **README surgery, not the M7 rewrite.** Only the claims Stage 0 falsified were touched: the
  day-of-week modifier, the Time Context section, the "Mood & Narrative" description, the demo-embeddings
  option, the `[I]` row, and the screenshot's vibe line. `[V]`, queue depth 10 and the contradictory
  model-size numbers still describe the current code and stay for Stages 2 and 4.
- **One M7 item taken early.** `start.sh`'s embeddings prompt was being rewritten anyway, so
  `${EMB_CHOICE,,}` — the bash 4+ expansion that is a hard syntax error on macOS's bash 3.2 — became
  `tr '[:upper:]' '[:lower:]'` in the same edit. The rest of M7 is untouched.

### What Stage 0 did **not** do

`[V]` and `force_shift()` still exist, the queue is still ten deep and still never refills, selection
is still strict argmax, MPD modes are still unasserted, and SIGTERM still neither exits nor saves.
Those are Stages 1–2 and are unchanged by anything above.

---

### Stage 1 — complete, 23 July 2026

Net effect: **three new modules** (`music_directory.py`, `embeddings_io.py`, `descriptor_bank.py`),
`embedding_generator.py` and `generate_embeddings.py` substantially rewritten, and **+112 tests**
(67 → 179, all green). Two data artifacts exist for the first time: 674 embeddings with their
per-window matrix and centroid, and a 49-word descriptor bank.

| Plan item | Landed as |
|---|---|
| **M3** | `music_directory.py`: detection reads `MPD_MUSIC_DIR`, then MPD's own config (`$XDG_CONFIG_HOME/mpd/mpd.conf`, `~/.config/mpd/mpd.conf`, `~/.mpdconf`, `~/.mpd/mpd.conf`, `/etc/mpd.conf`, `/etc/mpd/mpd.conf`), then the legacy default *marked as unverified*. `config.mpd_music_directory_source` records which, and it is printed at startup — a wrong directory is now diagnosable rather than mysterious. Validation resolves five probes spread across `mpc listall`, so a half-mounted library fails too. `start.sh` gained a step that prompts when detection fails. On this machine it now reads `/mnt/storage/music` from `~/.config/mpd/mpd.conf` instead of relying on the `/var/lib/mpd/music` symlink. |
| **M4** | `generate_embeddings.py` enumerates from `mpc listall` and nothing else; the `rglob` walk is gone and a test fails if it returns anywhere. `TrackLibrary.reconcile_with_mpd()` logs coverage on load, drops embeddings MPD cannot play, and raises below `config.minimum_mpd_coverage`. Live result: `674 of 674 embeddings match MPD (100.0%); 0 stale, 16 MPD tracks have no embedding`. |
| **C3** | Full-coverage non-overlapping 10 s windows with an end-aligned tail, RMS-gated, mean-pooled, per-window matrix persisted. `truncation` pinned to `rand_trunc` — see C3, which now records why that is not inheritable. 24,494 windows over 674 tracks. Embedding one file twice is now **bit-identical**, and re-embedding a track reproduces the vector already on disk. `failed.txt` records every failure with its exception. |
| **M8** | Real batching (32 windows per forward pass, filled **per track** — item 1 below) plus a decode/mel worker pool, which is where the cost actually sat; M8 now carries the throughput table. Sustained 75.8 windows/s; the whole library took **5m 23s** on the RTX 3070. The resumed-count and divide-by-zero bugs noted alongside M8 are fixed too. |
| **C5** | Centroid computed at generation, stored in the artifact, applied by `TrackLibrary.load_embeddings()` as `normalise(E − centroid)`. Everything downstream now lives in the centred space. Measured: random pairs move from **+0.670** to **+0.011**. |
| **D5a** | `descriptor_bank.py`: 49 prompts across six axes, CLAP text-tower embeddings, per-descriptor mean/std over the **centred** library, stored in `descriptors.npz` (93 KB). |
| **D5b** | Variance gate at build time, floor = 0.5 × the library's own median std. Both lists are printed. On this library it dropped **nothing** — the weakest descriptor (`frenetic`, std 0.096) still clears a floor of 0.093. That is a pass, not a no-op; see item 2 below for why it is nonetheless only proven by unit tests. |
| **M5** | `embeddings_io.validate_embeddings()` checks required keys, schema version, track/embedding/window/offset agreement, centroid shape and normalisation. The loader adopts the *file's* dimension over the config's, and checks model identity **by equality**. `TrackLibrary` raises `LibraryError` instead of degrading to an empty library. |

**Verification.** 179 pytest tests pass, including the three acceptance assertions the plan named:
same file embedded twice is bit-identical, self-similarity is exactly 1.0, and the post-centring
random-pair distribution sits at +0.011 rather than +0.670. The application was launched against the
live MPD in a pty and exited 0, reporting a clean library load, 100% coverage and a 49-descriptor
bank; the user's MPD queue, modes and volume were snapshotted before and restored after. Every
measurement quoted in this section is reproduced in **§10b**, including the descriptor spot-check that
is H1's own acceptance step.

### Two things the Stage 1 plan did not anticipate

*The rest of what the work turned up — the truncation mode, where the generation cost actually sits,
the sharper anisotropy — has been folded into C3, C5 and M8 themselves, replacing the claims that
turned out to be wrong. Nothing here needs cross-referencing to be read.*

**1 · Bit-determinism is a property of the file *and the batch size*, not of the file alone.** GPU
reductions depend on how many rows are in the batch, so the same track embedded at `--batch-size 32`
and `--batch-size 1` agrees to ~2 × 10⁻⁸ but not bit for bit. Two consequences, both shipped: batches
are filled from one track at a time (see M8 for why that beats window-level packing), and `batch_size`
is recorded in the artifact. Both facts are tests — bit-identity at the recorded batch size, agreement
to 1e-6 at a different one — so the limit of the guarantee is asserted rather than assumed.

**2 · The variance gate did not fire, so it is only proven by unit tests.** All 49 descriptors clear
the floor (weakest `frenetic` at std 0.096 against a floor of 0.093, median 0.185). That is the
outcome you want, but it means the *real* evidence for the gate is synthetic: a fixture where one
descriptor points along an axis the library has no variance on, which is dropped, and a degenerate
library where every std is float noise, which is kept whole. Building the gate was still right — the
cost of discovering the need for it in Stage 3, through a TUI, is exactly what D5's "build it early"
argument is about.

A related measurement worth recording, because it is the number that would have signalled H1's failure
shape returning: the bank's **effective rank is 2.5 of 49** — the descriptors are heavily correlated on
a metal-and-rock-heavy library. The readout is nonetheless not degenerate: every one of the 49 words
appears in some track's top-3, with a perplexity of 34.8. Correlation between descriptors is not the
same as a pinned readout, and the z-scoring is what separates them.

### Stage 1 — deviations from the plan's letter

- **A wrong music directory refuses *generation* but only warns at *startup*.** M3 says "refuse to
  launch". At generation time the cost of a wrong directory is total and expensive — a five-minute run
  that decodes nothing — so it is fatal there. At runtime, with M4 in place, playback does not touch
  the music directory at all: MPD plays by its own relative paths. All that breaks is album art and
  the in-process mutagen tag read, and the latter already falls back to `mpc search`. Refusing to
  start a working player because artwork will not render is a worse trade than saying so on the
  console.
- **`embeddings_io.py` was split out of `embedding_generator.py`.** Reading, validating and centring
  the artifact is pure numpy; generation needs torch. Without the split, `track_library` — imported by
  the TUI — would drag torch into every launch for three helper functions.
- **`TrackLibrary.save_embeddings()` was deleted**, having been explicitly retained in Stage 0. Stage 1
  made it dangerous rather than merely dead: it wrote `track_files` and `embeddings` only, so a round
  trip produced a file the new loader must refuse, and the vectors it held were already centred and
  would have been centred a second time on the next load. M5 offers this as an option; D7 settles it.
- **`_assess_embedding_quality()` / `_quality_interpretation()` were deleted.** They graded a library
  "Excellent — Embeddings show good discrimination" down to "Very Poor" against thresholds (mean
  0.3–0.5, std > 0.15) that nothing had ever measured, and which the *correct* raw distribution of a
  centred-on-load library fails by construction. This is D4's rule applied to a number instead of a
  word. `--stats` now prints the raw and centred distributions side by side and says nothing else.
- **The prompt template was chosen by measurement, not preference**, as H1 asks. All four candidates
  scored against the real library; `recording` won on spread and is what CLAP was trained on. Table in
  §10b.
- **README surgery, not the M7 rewrite.** Only what Stage 1 falsified: the "one pass over your
  library" line (it is now ~36 passes per track), the missing generation-time figure (now measured),
  the music-directory step, the windowing and centring paragraphs, and the descriptor bank going from
  "specified" to "built, with a command to try it". The queue-depth-10 and `[V]` claims still describe
  the current code and stay for Stage 2.

### What Stage 1 did **not** do

Nothing in the player changed. `[V]` and `force_shift()` still exist, the queue is still ten deep and
still never refills, selection is still a strict argmax, MPD modes are still unasserted, and SIGTERM
still neither exits nor saves. The descriptor bank is built and loads at startup but nothing displays
it — that is Stage 3. `SessionState._initialize_session_vector()` still seeds from `randn` (L7,
Stage 2), which is why the first queue of a fresh session is an arbitrary neighbourhood.

**One thing Stage 1 did that was not asked for, because Stage 2 needs it.** Every quantity H6 and H9
are specified against was measured on the crop-based embeddings in a space that no longer exists, and
the changes are large enough to mislead — a `[N]` press that moved 3.9% of the pool then moves 0.3%
now. Those two findings have been re-measured against the built library and carry the new numbers;
the raw data is in §10b. **The designs both survived**, which is the point: they target observable
quantities (pool turnover, rank) rather than vector-space magnitudes, so re-measurement adjusted the
reference values without touching the specification.

---

### Stage 2 — complete, 23 July 2026

Net effect: **one new module** (`manifold.py`, 190 lines of pure numpy), `queue_manager.py` cut from
169 lines to 180 of which almost none are the old ones, and **+131 tests** (179 → 310, all green).
The application plays continuously for the first time.

| Plan item | Landed as |
|---|---|
| **M1b** | `FakeMPD` in `tests/conftest.py`, built to the semantics table in M1 — which was **re-verified against the live MPD** (0.24.0 / mpc 0.35) rather than taken from the audit's prose. All seven original rows reproduced; four more were measured and three of them were new (below). `tests/test_fake_mpd.py` asserts the double row by row, so the harness is itself under test, and it caught a real fixture bug: a `FakeMPD` defaulting to `consume off` silently put every component under test back in the world C1 lived in. |
| **C2** | `MPDController.get_modes()` / `set_mode()`; modes carried as **raw strings** because `single` is off/on/`oneshot` and restoring a user's `oneshot` as `off` would be a silent change of their setting. `start_session()` forces `random`/`repeat`/`single` off and `consume` on, logs each change (`random on → off, consume off → on`) and says they will be restored. |
| **H3** | The signal handler now sets both flags, restores the MPD modes directly, and unblocks urwid through a **self-pipe** (`loop.watch_pipe`) rather than raising `ExitMainLoop` from a signal context. Plus an `atexit` hook for the mode restore specifically, a lock-and-flag so running it three times is harmless, and `_maybe_checkpoint()` writing state every `config.checkpoint_every_n_tracks` full listens. |
| **C1 / D1** | `config.queue_lookahead = 1`. `QueueManager` is now `ensure_one_ahead()` / `replace_next()` / `get_next_track()`. Deleted: `planned_queue`, `currently_queued_in_mpd`, `_sync_to_mpd`, `_generate_tracks`, `get_upcoming_tracks`, `recalculate`, `initialize_queue`, `on_track_started`, the 5% trajectory blend, `queue_buffer_size` and `queue_low_threshold`. |
| **H6** | `p(i) ∝ exp(−i/τ)` over rank, τ linear in the exploration scalar and floored at `tau_min`. The map reproduces H6's own published table exactly — p(rank 0) of 63% / 12% / 6% at exploration 0.1 / 0.4 / 0.7 — and a test recomputes those from the shipped code. Below `minimum_sampled_pool` candidates it falls back to a uniform draw. The generator is injectable, so sampling did not cost testability. |
| **C4** | One skip path, on the orchestrator: `skip_current_track()` — feedback, `replace_next()`, then exactly one `next_track()`. No `play()` anywhere in it. `tests/test_skip_path.py` drives **the real method** and asserts the ordering against the call log, not against the end state. |
| **H9 / D8** | `[V]`, `force_shift()`, `process_vibe_skip()`, `set_high_exploration()` and `vibe_shift_magnitude` deleted. `[N]` repels from the skip-run centroid by a λ solved for the turnover schedule (5/20/50/85%), snapping back onto the manifold from the second consecutive press. The console reports what each press *measured*, not what it targeted. |
| **H4-repl** | Falls out of D1: `replace_next()` drops the lookahead and re-picks under the post-skip vector, so a skip is audible on the very next song. |
| **L7** | β ramps from 0 over the first 20 taste updates, with the unearned weight going to the session term. `SessionState` starts at **zero** rather than `randn`, seeds from the first track that plays, and `get_candidate_pool` now guards *both* halves — so a fresh session draws its first track uniformly at random, which is the honest answer to "no information". |
| **M6b** | `TrackSelector.save()` / `load()` → `data/state/play_history.json`, wired into `Persistence` and the periodic checkpoint. `recent_history` is persisted alongside the two the finding names, because it is the half that does the actual excluding. |

**Verification.** 310 pytest tests pass. The application was then driven end to end against the live
MPD in a pty, with the user's queue, modes and volume snapshotted before and restored after. Against
the stage's own definition of done:

| Criterion | Result |
|---|---|
| 30+ tracks, no stall | **36 distinct tracks, 0 stalls** |
| no repeat inside the replay gap | **0 violations** |
| queue holds one ahead | **depth 2 in 30 of 30 mid-track samples** |
| one `[N]` changes the next track | yes — the queued lookahead was dropped and replaced |
| four consecutive `[N]` exceed 80% turnover | **5% → 20% → 70% → 100%**, each meeting its target |
| …and audibly change the *kind* of music | Björk / Arcane OST → Watain |
| `mpc status` after `kill -TERM` | modes restored byte-identically |

### Four things the Stage 2 plan did not anticipate

**1 · Three of MPD's semantics were not what the audit recorded, and one of them mattered.**
Re-running M1's table against the live server reproduced all seven published rows and turned up four
more:

- **`mpc next` while paused consumes the track *and resumes playback*.** The audit assumed it stayed
  paused, and C4's fix direction was written around that ("if MPD is paused, step 3 is skipped").
  Following that would have meant a skip while paused replayed the very track the user rejected.
  The shipped path advances and then re-pauses — which is safe because **`mpc pause` is idempotent
  rather than a toggle**, itself verified.
- **`mpc next` on a stopped player is an error** (`MPD error: Not playing`) that changes nothing, so
  the skip path guards on state rather than discovering this at runtime.
- **`mpc del N` past the end of the queue exits non-zero.** `replace_next()` relies on that when the
  queue holds only the current track.

**2 · Solving λ and *then* snapping does not deliver the schedule — and only a live run showed it.**
H9 specifies "solve for the turnover target, then `snap()` for n ≥ 2". Built that way, the offline
medians looked right (5.5% → 15.7% → 53.2% → 99.3% over 60 simulated runs). The first live session
printed:

```
Skip #1: λ=0.60, 5% of what you would have heard is now different (target 5%)
Skip #2: λ=0.80 + snap, 1% of what you would have heard is now different (target 20%)
```

The second press **undid the first**. `snap()` relocates to the centroid of the 25 nearest tracks,
and after a modest λ those 25 are largely the neighbourhood the vector started in, so the result
lands back where the run began. The median hid a bad tail.

The fix applies the plan's own principle one level more consistently: **the target is stated on the
vector that actually selects**, so the snap belongs *inside* the objective rather than applied to its
result. `solve_repulsion(..., snap_result=True)` snaps every candidate on the λ grid before measuring
its turnover — two matrix products instead of one, still microseconds. Measured after the change,
over 40 simulated runs of five presses: **0 of 160 presses moved backwards**, against a schedule
every press meets. The live session now reads 5% → 20% → 70% → 100%.

**3 · The reason to gate `snap()` at n ≥ 2 is stronger than the audit's, and the audit's is also
true.** §10b justified the gate by overshoot at the 5% target. That holds — post-snap turnover has a
floor around 8%. But the load-bearing argument is the other direction: an **unguarded** repulsion
solved for the 85% target lands below the 1st percentile of the library's own on-manifold quality
**100% of the time** (and at the 50% target, 87% of the time), while a single skip's λ never does.
Gating at n ≥ 3 instead leaves press 2 at quality 0.42 — worse than the deleted `[V]`'s 0.56.

**4 · "Is this vector still music?" has no fixed threshold, only the library's own distribution.**
The first attempt asserted snapped vectors stayed above ~95% of a typical real track's on-manifold
quality, and it failed — because real tracks themselves span **0.43 to 0.96** on this library. A track
in a sparse corner legitimately scores low, and so does a legitimate centroid near it. The assertion
that means something is *no worse than the least typical real track*: snapped vectors never fall below
the library's 1st percentile, and unguarded ones at the escalated targets always do. This is D4's rule
applied to a test rather than to a feature.

### Stage 2 — deviations from the plan's letter

- **The "negative updates are a no-op while unseeded" guard was kept, not retired.** §8 and §9 both
  say to retire it once the β ramp lands, on the grounds that the ramp makes it redundant. It does
  not. From zero, one skip normalises to `−track` at unit length; β would damp that in the *score*,
  but **β never gates retrieval**, and `get_candidate_pool` opens its taste half on
  `np.any(taste_vector)`. So retiring the guard would hand half the candidate pool, at full strength,
  to "the tracks least like the one song you rejected" — reintroducing in retrieval exactly the defect
  Stage 0 removed from scoring. The plan's own reasoning ("β gates the score, not the pool") is the
  argument for keeping it.
- **`get_random_track()` no longer draws from numpy's global RNG.** It is the "no evidence yet" path
  (L7), so it decides the first track of every session; leaving it on the global state made that
  untestable for reasons unrelated to selection. It takes an optional generator, and the selector
  passes its own.
- **`TrackSelector.forget_selection()` is new, and was not in the plan.** A selection is normally a
  play at depth 1, but two paths break that: a skip drops the lookahead and re-picks it, and MPD can
  refuse an `add`. Left recorded, those tracks would sit inside the 20-track replay gap having never
  been heard, and the `[I]` inspector's "tracks played" would claim more music than the session
  played.
- **The queue panel became a one-line "Up Next" rather than going blank.** `get_upcoming_tracks()` is
  deleted as the plan says, and with it the ↑↓/ENTER bindings and their footer entries — they indexed
  into `mpc playlist`, which with consume off is the session's *history*, so ENTER on "1." replayed
  the first track of the evening. The panel's geometry is deliberately untouched: the album art is
  still pinned to hand-counted row constants, and H8 must land before the layout moves.
- **`[I]` gained the sampling and skip rows now rather than in Stage 3.** Stage 2 ships a sampler and
  an escalation schedule; an inspector that could not show either would be advertising less than the
  machine does. Everything added is measured — τ, the drawn rank, β earned, the next skip's target.
  Doing so overflowed the overlay's fixed 70% height, so it is now sized to its content; **Stage 3's
  descriptor rows (H1d) will need it to scroll rather than merely fit.**
- **`MPDController.next_track()` now checks its return code.** It returned `True` unconditionally,
  which is the H7 pattern; the stopped-player error made it observable.
- **`FeedbackHandler` no longer takes a `queue_manager`.** Once the skip path moved to the
  orchestrator it stopped using it, and carrying the reference would have left the door open to
  re-splitting the ordering across two objects — which is the shape of C4. The class now updates the
  models and records history; it does not touch MPD or the queue, and a comment in its constructor
  says why the parameter is absent.
- **README and `start.sh` surgery, not the M7 rewrite.** Only what Stage 2 falsified: the `[V]` row
  and its two prose paragraphs, the queue-navigation bindings, "the queue is kept at 10 tracks",
  "picks the best", `vibe_shift_magnitude`, and the screenshot's queue panel. New: the skip-escalation
  table and the rank-sampling paragraph. The rest of M7 — the three contradictory download sizes, the
  untested macOS claim — is still Stage 4.

### What Stage 2 did **not** do

Nothing in the descriptor display changed. `descriptor_bank` still loads at startup and is still
unused; the vibe line still shows a track count; the session-history panel does not exist; album-art
geometry is still hand-counted constants (H8). Those are Stage 3. `[L]` is still not a toggle and
`liked_tracks` still does not rehydrate from `feedback_history.json` (L4).

### Stage 3 — complete, 23 July 2026

Net effect: **two new modules** (`vibe_readout.py`, `session_history.py` — both pure, no urwid, no
MPD), `tui.py` rebuilt around a derived geometry and a Session panel, and **+105 tests** (311 → 416,
all green). The descriptor bank built in Stage 1 is finally read by something.

| Plan item | Landed as |
|---|---|
| **H8 / L3** | `AdaptiveDJTUI._art_geometry(cols, rows)` walks the widget tree — `Frame.header.rows()`, `Pile.get_item_rows()`, `Pile.contents`, `Columns.column_widths()`, `Columns.rows()` — and returns `(x, y, width, height)` or None. `RIGHT_COL_ROWS` and `NP_BORDER_ROWS` are deleted and guarded by `test_deletions.py`. Two of the four hand-counted constants were **wrong**: `x` was 2 where the art column starts at 1, and the height was pinned to a Pile that Stage 3 takes from 10 rows to 11. Doing this first was correct and load-bearing — see item 1 below. |
| **H1b** | `vibe_readout.VibeReadout`: top-3 descriptors by z-score on line 1, drift and the track count on line 2, both gated on `SessionState.is_seeded()`. The consistency figure is a **count of held words, not a cosine** — a measurement, not a preference (item 2). |
| **H1c** | `session_history.SessionHistory` plus the `Session` panel: `↓ next:`, a divider, then plays newest-first with `♥` / `⏭` / `✓` / `♪`. Order comes from watching MPD's current track; outcomes come from draining `FeedbackHandler.feedback_history` from a cursor taken at `run()`. Neither required the player to grow a store. |
| **L4** | `SessionHistory.rehydrate_likes()` reads the same `feedback_history.json` at construction. `TUI.liked_tracks` is gone; `history.liked` is the one set, and it spans sessions. |
| **H1d** | `↑↓` and `ENTER` bound through `unhandled_input`; both list panels wrapped in `urwid.WidgetDisable` so the body Pile stops swallowing arrow keys. `[I]` gains a `DESCRIPTORS` block (session **and** taste, each gated on its own vector) and is now a `ListBox` that scrolls. `QueueManager.requeue_next()` is the one player-side addition — reasoning in the deviations below. |
| **L9** | Footer, README table and fallback-mode bindings reconciled to one list. A test drives every advertised key through the real `_handle_input` and requires a distinct action, so a footer entry cannot outlive its binding. |
| **N1** *(not in the plan)* | The `('weight', 3, …)` → `('pack', …)` layout fix for the sub-33-row `WidgetError`. |

**Verification.** 416 pytest tests pass. The application was then driven end to end against the live
MPD in a pty at 120×45 and at 80×24, with the user's queue, modes and volume snapshotted before and
restored byte-identically after.

| Criterion | Result |
|---|---|
| the vibe line names three recognisable descriptors | yes — Fleshgod Apocalypse's *King* read `piano-led · halftime · triumphant`; the seven §10b spot-checks reproduce exactly |
| `[I]` explains the model without invented vocabulary | yes — every row is a number the system computed, including the drift cosine with its measured scale printed beside it |
| the readout refuses before anything has played | yes — `♪ —  nothing has played yet`, where the bank alone answers `shimmering · orchestral · serene` |
| the Session panel marks what happened | yes — `♥♪ Yes – America (single version)` above `⏭ Arctic Monkeys – D Is for Dangerous` |
| `↑↓` / `ENTER` | yes — `Replay: Yes – America (single version) queued next`, one `add` before one `del`, no `play`, no `next` |
| album art at the derived geometry | yes — ueberzugpp, geometry asserted equal to `_art_geometry` at seven terminal sizes |
| **an 80×24 terminal** | **renders** — the same run against the pre-Stage-3 tree raises `WidgetError` and dumps a traceback over the UI |
| modes restored after `kill -TERM` | byte-identical, queue byte-identical |

### Three things the Stage 3 plan did not anticipate

**1 · The layout had been crashing on short terminals the whole time.** Not a Stage 3 regression and
not in any finding: the `Now Playing` box took `('weight', 3, …)` of the body, its content is a flow
`Pile` wrapping a `Columns` whose left cell is a box `Filler`, so urwid resolved the `Columns` as a
box widget, handed it the weighted height, and raised when it rendered its natural one instead.

```
  80x20 … 80x32   WidgetError: <Columns …> rendered (80 x 12) canvas when passed size (80, 6)!
  80x33 … 80x45   OK
```

Live, against `HEAD 3558b88` in a pty at 80×24, the traceback prints over the interface and the
session is unusable. `('pack', …)` gives the box exactly `rows()`, which fixes it at every size down
to 80×6 — and has a second effect the geometry depends on: **the panel is now the same height at
80×24 as at 80×60, so the album art's position stops varying with the terminal's height at all.**
That is why the hardcoded `y = 3` was right at one size and wrong at others.

The finding is recorded as **N1**. The thing worth carrying forward is not the bug; it is that a
suite of 311 behavioural tests said nothing about it, because none of them called `render()`. §8's
Stage 3 preamble warned that this stage started with no coverage in its own area. It was right, and
the cost was already on the ground before the stage began.

**2 · H1's consistency statistic is compressed, and the measurement changed the design.** H1
specifies "the cosine between the session's descriptor z-vector now and five tracks ago". Driven
through the shipped `SessionState` and `VibeReadout` over 40 sessions on the real library (§10d):

```
  cos(z_now, z_5ago)     min 0.721   p10 0.948   p50 0.989   p90 0.997
  words held (0 of 3 … 3 of 3)   1%    12%    42%    45%      mean 2.31
```

Ninety per cent of ordinary listening sits in the top 5% of the cosine's nominal range, so the
readout would print "0.99" essentially forever. **That is C5's shape and the entropy heuristic's
shape**: a number whose scale the data never occupies, presented as information. What does occupy its
range is how many of the three words on screen were also on screen five tracks ago — 0 through 3,
median 2, and 0 or 1 after a skip run.

So the line reports the count. It needs no threshold, no calibration and no vocabulary, and it is a
statement about something the listener actually read. This is §8's trap 4 ("the only item in this
stage that invents a threshold") resolved by its second option — ship the measurement — after the
first option, calibrating a threshold, turned out to be calibrating one against a scale that does not
discriminate. The cosine is still computed and still shown in `[I]`, with its measured distribution
printed beside it so the number can be read.

**2b · Making `[I]` scroll quietly broke the history panel, and only a deliberate check found it.**
Stage 2's inspector dismissed on *any* key, so it was momentary. H1d makes ↑↓ scroll it instead, which
means it can now be held open across a whole track — and the `[I]` loop owns the thread that records
history. Measured: a track that started **and** finished while the overlay was up never reached the
panel, and its `✓` was dropped, because `apply_event` had no entry to attach it to. The panel's whole
claim is that it lists what actually played.

The fix is `screen.set_input_timeouts(max_wait=0.5)` around the overlay loop, so it wakes without a
keypress and runs `_sync_session_state()` — the bookkeeping half of `_update_display`, extracted for
exactly this. Measured cost: 0.02 s of CPU over 6 s with the overlay open, against 0.04 s for an idle
TUI. Pinned by `test_the_session_keeps_being_observed_while_the_inspector_is_open`, whose input script
includes an empty `[]` wake-up because that is the case that matters.

Worth naming the general shape, because Stage 4 will meet it: **the display now holds state that only
the display's own tick maintains.** That is the right place for it (§9), but it means any code path
that blocks the urwid loop is a path that stops observing the session. `[I]` was the only one; a
Stage 4 modal — a confirmation prompt, a help overlay — would be another.

**3 · The history panel needed no new store behind the display, but it did need a cursor.** Trap 3
said "nothing stores what H1's consistency word compares against" and offered two options. Both the
z-vector store and the play history ended up in the display layer, which is the honest place for
them: neither is read by selection, and `SessionState` is closed. But `FeedbackHandler.feedback_history`
is loaded from disk at startup and spans every previous session, so draining it naively back-fills
tonight's panel with last week's skips. The cursor taken at `run()` is what makes it a *session*
history, and it is tested directly.

### Stage 3 — deviations from the plan's letter

- **One method was added to `queue_manager.py`, which §8 says to stop and justify.** `ENTER` has to
  put a chosen track in the lookahead slot, and that is a queue operation: doing it from `tui.py`
  would put a second copy of the add/delete ordering — the thing C4 and M1 are about — in the display
  layer. `requeue_next(track)` adds **then** deletes, the inverse of `replace_next()`, because it
  already knows its track and has nothing to exclude; appending first means the queue is momentarily
  three deep rather than one deep, so a refusal from MPD leaves the session untouched. Six tests
  against `FakeMPD`'s call log.
- **`get_playlist_metadata()` was deleted rather than left alone.** Trap 2 says to use
  `_fetch_track_tags()` and promote it to public. Doing so left `get_playlist_metadata` with no
  callers — it existed only for the panel that Stage 2 had already cut to one line — and it shelled
  out an extra `mpc playlist` on every 0.5 s refresh to build a map keyed on tracks that, under
  `consume on`, are exactly the ones no longer in the playlist. `fetch_track_tags` is public and
  carries the cache the audit already described it as having.
- **The vibe readout is two rows, not one.** H1's mock-up puts descriptors, consistency and counts on
  a single line. At the real panel width the right-hand column is ~45 columns, and one line either
  truncates or wraps unpredictably — which would then change the art height under H8. Two declared
  rows are stable, and the geometry follows them automatically.
- **`_session_line()` is deleted, not rewritten.** It reported the track count, which was the only
  honest thing available while the bank was unwired. The count moved onto the drift line.
- **The header and footer became flow widgets.** They were `Filler`s, which are box widgets that
  `Frame` accepted by accident; a flow footer wraps its hints onto a second row on a narrow terminal
  instead of clipping them, and both can now report `rows()` honestly, which `_art_geometry` reads
  rather than assuming 1.
- **The art renderer is injectable** (`AdaptiveDJTUI(dj, art_renderer=…)`). Constructing the widget
  tree in a test otherwise detects image protocols and leaves a `ueberzugpp` child behind.

### What Stage 3 did **not** do

- **`[L]` is still not a toggle.** L8's status line says un-liking is Stage 3, but it is not a row in
  §8's Stage 3 table, and it is not a display change: un-liking means subtracting a like from the
  taste model, which is the player. Left for Stage 4, where the finding is now recorded.
- **The `[I]` overlay's ListBox is sized by arithmetic, not by the tree.** `_show_model_info` computes
  its inner size from the 70% relative width it declares. That is a hand-derived number of exactly
  the kind H8 removed, in a place where being wrong costs a slightly-off scroll page rather than a
  misplaced image. It is noted rather than fixed.
- **Nothing in the *simple* fallback mode was tested.** Its bindings were reconciled with the urwid
  mode and the README (L9), but there is no harness that drives a non-urwid terminal, so that
  reconciliation is asserted only by reading. L2's non-functional art paths are untouched (Stage 4).
- **No Stage 4 work was pulled forward.** M7's remaining README contradictions, L1's SIGWINCH
  handler, L6's polling and the `.gitkeep` scaffolding are all still open.

### Stage 4 — complete, 23 July 2026

Net effect: **+126 tests** (416 → 542, all green, ~18 s), four new test files, the kitty and sixel
album-art paths deleted, `[L]` made a toggle on a retraction model that is a replay rather than a
guessed inverse, and **three defects found that were not in any finding** — one of them in the test
suite itself, and one created by following L1's own fix direction.

| Plan item | Landed as |
|---|---|
| **M1c** | `test_persistence_round_trip.py` (+22): the three files with no round-trip coverage, each as round-trip → behavioural consequence → missing file → corrupt file. `test_simple_mode.py` (+37): the fallback mode driven through a real pty, plus `decode_keys()` as a pure function. |
| **M7** | Every figure re-measured and `test_documented_numbers.py` (+13) holds the three places together, because M7 is a consistency finding and rewriting the numbers once would not have fixed it. |
| **L1** | **The finding's conclusion was wrong on this urwid** — see item 1 below. The handler was already being reached; what shipped is the check that keeps it reachable without closing a loop, and the first tests that assert any of it. |
| **L2** | `KittyProtocol` and `SixelProtocol` deleted (D7), guarded by `test_deletions.py`. `_warn_about_unsupported_terminal()` tells a kitty or sixel user why there is no art and what would give them some — the only part of those ~100 lines that was ever true. |
| **L8** | `UserTaste.replay()` + `UserTaste.explains()` + `FeedbackHandler.process_unlike()`. The retraction is a *deletion plus a recomputation*, gated on the history being able to account for the model. Reasoning and both measurements in item 2. |
| **L9** | `.gitignore` switched from excluding the directories to excluding their contents, so the `.gitkeep` files ship — `git ls-files data` now lists all three. `AlbumArtRenderer.shutdown()` ends the child, wired into `_shutdown()` **and** the signal path. |
| *(optional)* | The `[I]` overlay's ListBox is sized from `Overlay.calculate_padding_filler()` / `top_w_size()` rather than from our own copy of the `("relative", 70)` arithmetic, and re-read per keypress so it survives a resize while the page is open. |
| **L6** | *(optional)* Not done. Still only worth it against a remote `MPD_HOST`, and Stage 4's brief is to change observable behaviour as little as possible. |

**Verification.** 542 pytest tests pass. The application was then driven end to end against the live
MPD in a pty at 120×45 and at 80×24, from a cold `data/state/`, with the user's queue, modes and
volume snapshotted before and restored byte-identically after.

| Criterion | Result |
|---|---|
| plays, one ahead, modes forced and restored | yes — `random on → off, consume off → on` at start, `random → on, consume → off` after `SIGTERM` |
| `[L]` likes, `[L]` again retracts | yes — `♥` and `❤` drawn, then gone; `Un-liked; taste model rebuilt from 0 feedback events` at 120×45 and `from 2 feedback events` at 80×24 |
| the retraction's fallback branch | **fired on real data** — see item 3 |
| `[I]` opens, scrolls and dismisses | yes, at both sizes |
| SIGWINCH during a live session | no traceback, no `RecursionError`, still rendering — **this is the check that found item 1** |
| skip escalation | `Skip #1 λ=1.05`, `#2 λ=0.05 + snap`, `#3 λ=1.00 + snap`, `#4 λ=0.80 + snap` |
| `80×24` | renders; no `WidgetError` |
| ueberzugpp after `SIGTERM` | none |
| `data/dj.log` | 0 tracebacks across both runs |

### Three things the Stage 4 plan did not anticipate

**1 · L1's measurement is right, its conclusion is wrong, and following its fix direction broke the
application.** The finding says urwid replaces `_on_sigwinch` at startup, and cites `getsignal`
returning `Screen._sigwinch_handler` after `Screen.start()`. That output is real. The inference from
it is not: on urwid 3.0.5 `_posix_raw_display.Screen` **captures** whatever handler was installed
into `_prev_sigwinch_handler` (line 129), **calls it** after its own work (line 98), and **restores
it** in `stop()` (line 142). urwid wraps our handler rather than displacing it — which is
indistinguishable from replacement through `getsignal`, and is not the same thing at all. The
handler had been running the whole time.

Re-installing on top of that, which is exactly what L1 says to do, closes a cycle: ours → urwid's →
ours (as urwid's `_prev`) → urwid's → … The first live run produced precisely that, and only because
it was a live run:

```
  File "tui.py", line 932, in _on_sigwinch
    chained(signum, frame)
  File "urwid/display/_posix_raw_display.py", line 98, in _sigwinch_handler
    self._prev_sigwinch_handler(signum, frame)
  … × ~500 …
RecursionError: maximum recursion depth exceeded
```

Nine unit tests said the chaining was correct. All nine were consistent with each other and with the
finding, and none of them knew what urwid does, because the double they ran against was one I wrote
from the same assumption. **That is the shape of C1 and of `FakeMPD`, arriving in a stage whose whole
subject is durability** — a harness built on the belief that produced the defect will reproduce the
defect and pass. What caught it was sending a real `SIGWINCH` to a real process.

The shipped code asks the screen object whether we are already in its chain
(`_prev_sigwinch_handler is self._on_sigwinch`) and installs only when we are not — which is the
world older urwid is in, and the world the finding describes. Reading the object rather than a
version number keeps it a property of what is actually there. Both branches are now tested, including
one that fails if the cycle comes back.

**L1 is therefore recorded as done, but not as "the handler now works" — it already did.** What Stage
4 added is the first test of any kind for it, and safety on urwid builds that do not chain.

**2 · L8's trap warns against the wrong failure, and the right design needed two measurements rather
than a decision.** §8 says subtracting `taste_update_like` is not symmetric because the update is a
normalised EMA. True, but measured on the real library the asymmetry is *tiny* — cos 0.9999 against
the truth in every ordinary case:

```
  like is the 2nd event         cos(subtract-0.1, truth)   min 0.999900   median 0.999953
  like early, 3 events after                               min 0.999876   median 0.999942
  settled model, 20 after                                  min 0.999871   median 0.999959
  like is the only event                                   min 0.000000   median 0.000000   ← 
```

The last row is the finding. From zero, one like normalises to the track itself; subtracting 0.1·e
from e gives 0.9·e, which normalises straight **back to e**. A subtraction cannot un-seed a model, so
retracting your only like leaves your long-term taste pinned at unit strength to the track you just
rejected — and that is the state a new listener is in, so it is the first retraction anyone performs.
No magnitude fixes it, because the defect is structural rather than numerical. The audit's own
argument would not have found this; the measurement did.

Replaying the feedback history without the like has no such case, and reproduces the incrementally
built model **bit for bit** (§10e). But a second measurement changed the design again: the replay is
only exact if the history is a *complete* account of the model, and `_record_feedback` caps it at
1000 events.

```
   999 lifetime events,  999 retained: cos = 1.000000000000   complete
  1000 lifetime events, 1000 retained: cos = 1.000000000000   complete
  1001 lifetime events, 1000 retained: cos = 0.994142650376   truncated
  1400 lifetime events, 1000 retained: cos = 0.923200176638   truncated
```

A blind replay would move a long-time listener's taste by 0.077 for reasons unrelated to the track
they un-liked — an order of magnitude more than the retraction itself, and silently. So
`UserTaste.explains()` checks first, and when the history cannot account for the model the retraction
is display-only and **says so in the console**. That is §8's second option, taken only where the
first cannot be honest, and the README states which is which. Note there is no threshold here to
calibrate: the discriminator is exact reproduction against ≤ 0.994, six orders of magnitude of
margin, not a cut anyone chose.

**3 · The test suite was writing to the developer's live `data/state/`, and had been for some time.**
Found while writing M1c's round-trips. `process_like()` calls `user_taste.save()` with no argument,
which resolves to `config.taste_file` — so a single green run replaced a real listener's taste model
with a fixture's and emptied `feedback_history.json`, the file Stage 3 made the `♥` marks depend on
(L4). Confirmed pre-existing by stashing every Stage 4 source change and watching the old code do it
again.

A suite that destroys the state it exists to protect is not green in a sense worth having, so the fix
is an autouse fixture redirecting all four paths, rather than a monkeypatch in the tests that
happened to notice. The leak was three layers below the test that triggered it, and the next one will
be too.

It also cost something real and is worth recording honestly: the taste model on this machine was
already overwritten by the baseline run taken at the start of the stage, before there was any reason
to snapshot it. Per D3 that is regenerable rather than lost, which is the only reason it is a
footnote.

**3b · Two smaller things the pty found, both pre-existing.** The fallback mode read one character
per tick and `select()`ed on `sys.stdin` between reads — but `read()` on a buffered text stream
drains the descriptor into Python's buffer, so `select()` then reported nothing pending and the rest
of a burst sat unread until another keypress dislodged one character of it. It reads the raw
descriptor now and splits the buffer with `decode_keys()`. And `[I]`'s fallback page blocked on an
unbounded `select`, which nothing could interrupt — `request_exit()` sets `running` false from the
signal handler and the page would have sat there through a SIGTERM. That is H3's shape in the one
interface H3 was never driven through; it polls now.

### Stage 4 — deviations from the plan's letter

- **The fallback mode's bindings were not merely reconciled, they were made the same code.** §8 asks
  for a harness for the mode as it stands. Testing it as it stood would have pinned a *second*
  binding table beside the urwid one — the thing L9 is about — so `decode_key()`/`decode_keys()` turn
  terminal bytes into urwid's key names and `_handle_input` does the dispatch for both interfaces. A
  binding can no longer exist in one and not the other, which is a stronger claim than any test of
  the old structure could have made. Most of the mode is now testable without a terminal; the pty
  covers what genuinely needs one.
- **`[I]` was bound in the fallback mode.** §8 trap 4 calls its absence "a loose end rather than an
  intended asymmetry" without asking for it to be fixed. It is one line once the dispatch is shared,
  and leaving a key advertised in the README and the urwid footer but dead in the other interface is
  the dishonesty Stage 0 exists to remove.
- **`ExplorationController.load()` and `UserTaste.load()` were made atomic.** Not in any finding.
  Writing M1c's "file missing a key" case showed both assigned field by field, so a truncated file
  left the object carrying some values from disk and the rest at their defaults — while returning
  `False`, so the caller believed nothing had been read. `UserTaste` is the worse of the two: the β
  ramp reads `total_updates`, so a vector loaded beside default counters arrives with none of its
  weight earned. Both read every field before assigning any.
- **`process_like()` returns a bool.** A track with no embedding produces no feedback event, and the
  TUI was drawing a `♥` for it anyway — a heart that `[L]` could then never take off, because there
  was nothing in the history to retract. Harmless before `[L]` was a toggle; a dead end after.
- **The kitty and sixel paths were deleted rather than marked.** L2 offers either. D7 says take the
  deletion when the honest answer is "delete this", and a branch that cannot work is not a feature
  with a caveat. What replaced them is a sentence telling that user what to install.

### What Stage 4 did **not** do

- **L6 (`mpc idle`)** — optional in §8, and still only worth it against a remote `MPD_HOST`.
- **The Session panel is empty at 80×24.** The layout renders (N1 stays fixed) but the Now Playing
  box, the console and the footer consume all 24 rows, so the panel draws its border and no content.
  Pre-existing — Stage 4 changed no layout — and fixing it means re-weighting a tree whose packing is
  N1's fix. Recorded rather than touched.
- **The first-press skip turnover is 100% against a 5% target on a cold session.** §9's item 5 already
  notes the middle rows overshoot; this is the extreme of it, at press 1 from an essentially unseeded
  vector, where the pool being compared against is a uniform draw. No file in the selection stack was
  touched this stage (`session_state.py`, `manifold.py`, `track_selector.py`, `queue_manager.py` are
  all unchanged), so it is an observation about a cold start rather than a regression — but it is the
  first time the escalation has been driven from a genuinely empty `data/state/`.
- **`[I]`'s fallback page and the urwid overlay are still two pieces of code.** They agree on
  content; only the urwid one runs the session bookkeeping while open, because the fallback's own
  loop is the thing that would be blocked.

---

## 1 · How the system actually works

*Read this first if you are picking the project up cold. **As of Stage 3 this describes the system as
it is**, not as it is planned to be — the bracketed "currently…" notes that used to qualify each step
are gone because the behaviour they described is gone. No part of this section is now ahead of the
code.*

Everything is driven by 512-dimensional CLAP audio embeddings, one per track, stored L2-normalised
in `data/embeddings/track_embeddings.npz`. All similarity is a dot product **on centred vectors**
(C5). No genre tags or metadata enter the selection logic.

```
start.sh  ──▶ main_tui.py (AdaptiveDJWithTUI)
                   │        owns: MPD mode force/restore, signal handling,
                   │               the one skip path, periodic checkpoint
   ┌───────────────┼────────────────────────────────┐
   ▼               ▼                                ▼
tui.py         background thread                mpd_controller.py
(urwid loop,   (polls MPD 2×/sec, detects        (every op = an `mpc`
 0.5s redraw)   track change, fires               subprocess)
   │            full-listen, refills, seeds
   │            the session vector)
   ├──▶ vibe_readout.py     descriptor_bank.py → top-3 z-scores + drift
   ├──▶ session_history.py  what played · ♥ ⏭ ✓ · the ↑↓ cursor
   │        (both display-only: nothing behind the display reads them)
   │               │
   ▼               ▼
feedback_handler.py ──┬──▶ session_state.py   short-term vibe vector
                      │        └──▶ manifold.py   turnover · solved λ · snap
                      ├──▶ user_taste.py      long-term taste vector → data/state/*.npz
                      ├──▶ exploration_controller.py   scalar 0.1–0.7 → weights + τ
                      └──▶ queue_manager.py ──▶ track_selector.py ──▶ track_library.py
                             (one ahead)          (rank-Boltzmann over ~100 candidates)
```

### The selection loop, precisely

1. `TrackLibrary.get_candidate_pool()` takes the top ~150 tracks nearest the session vector and the
   top ~150 nearest the taste vector, interleaves them, and truncates to 100.
2. `TrackSelector._calculate_score()` scores each candidate as
   `α·session_sim + β·taste_sim + γ·novelty + δ·anti_repetition`. Those four are the whole score.
3. The weights come from `ExplorationController.get_weights()`, which shifts mass from session/taste
   into novelty as the exploration scalar rises. β is additionally ramped from 0 as taste updates
   accumulate, and the unearned share goes to the session term (L7).
4. **One track is drawn by Boltzmann sampling over rank** — `p(i) ∝ exp(−i/τ)` with τ set by the
   exploration scalar. Rank-based rather than score-based, so it needs no recalibration when the score
   scale moves.
5. `QueueManager.ensure_one_ahead()` keeps exactly one track ahead of the current one, with
   `consume on` so MPD pops finished tracks itself — making the refill condition `len(playlist) < 2`.

Before anything has played, both vectors are zero and step 1 returns an empty pool, so the first
track is drawn uniformly at random rather than from an arbitrary `argpartition` ordering (L7).

### What feeds back

| Event | Trigger | Effect |
|---|---|---|
| **Full listen** | ≥90% of duration played | Session vector updated (the primary driver). Taste +0.02. Exploration −0.02. This is the only thing that increments `tracks_played`. |
| **Skip `[N]`** | Keypress | Taste −0.05. Exploration +0.05. Session vector repelled from the consecutive-skip-run centroid by a magnitude **solved for a pool-turnover target that escalates with the run length** (5% → 85%), projected back onto the manifold from the second consecutive press. Lookahead replaced, then exactly one advance. The measured turnover is reported to the console. |
| ~~Vibe skip `[V]`~~ | — | **Deleted (D8/H9).** It turned over less of the candidate pool than `[N]`×5 while pointing off-manifold. Its role is covered by `[N]`'s escalation. |
| **Like `[L]`** | Keypress | Taste +0.10. Taste file saved immediately. |

### What persists

`user_taste.npz` (on every like and at exit), `exploration_state.json`, `feedback_history.json` and
`play_history.json` — the last carrying `play_history`, `current_index` and `recent_history`, so
anti-repetition survives a restart (M6b). The session vector is intentionally ephemeral. All of it is
checkpointed every `config.checkpoint_every_n_tracks` full listens, not only at exit (H3).

### Current state on disk

*(Stage 1, 23 July 2026.)* `data/embeddings/` holds the two artifacts of §7:

```
track_embeddings.npz   674 tracks × 512 d, uncentred + centroid + 24,494 windows   45.5 MB
descriptors.npz        49 descriptors × 512 d, with per-descriptor mean/std        93 KB
failed.txt             16 files, all one corrupt album, each with its exception
```

`data/state/` now fills during a session: `user_taste.npz`, `exploration_state.json`,
`feedback_history.json` and `play_history.json`. None of it is committed (D3, and
`tests/test_deletions.py` fails if any of it is).

---

## 2 · Critical findings

*Each of these independently defeats the project's stated goal.*

---

### C1 · The queue never refills. Playback stops dead after 10 tracks. `DONE — Stage 2`

> **Status.** `config.queue_lookahead = 1`, `consume on` forced, and `QueueManager` reduced to
> `ensure_one_ahead()` / `replace_next()` / `get_next_track()`. The refill condition is
> `len(mpc playlist) < 2`, which holds because consume makes MPD pop each finished track itself.
> Everything the depth existed to support is deleted — `planned_queue`, `currently_queued_in_mpd`,
> `_sync_to_mpd`, `_generate_tracks`, `recalculate`, `initialize_queue`, `on_track_started`, the 5%
> trajectory blend, `queue_buffer_size` and `queue_low_threshold`.
>
> Live: a 30-track unattended run with **0 stalls** and a queue depth of exactly 2 in all 30
> mid-track samples. `tests/test_queue_manager.py` drives the real manager against `FakeMPD` and
> plays 40 tracks through, so C1's own scenario is now a regression test. One case the original code
> only pretended to handle is now real and tested: if the queue does run dry and MPD stops, adding
> alone will not restart it, so the refill explicitly plays — but only when the queue was empty on
> entry and playback had already begun, which is what keeps it clear of C4's skip path.


**What.** `QueueManager.check_and_refill()` decides whether to top up the queue by comparing
`len(mpd_controller.get_queue())` against `queue_low_threshold` (3). But `get_queue()` runs
`mpc playlist`, which returns the entire MPD queue — including every track already played. MPD's
consume mode is off (verified on this machine) and the application never enables it, so that count
only ever grows. After the initial fill it is pinned at 10, the condition `10 < 3` is never true, and
no track is ever added again.

**Evidence.** Simulated against a queue model with correct MPD semantics, driving the real
`QueueManager` and the real 616-track library:

```
after initialize_queue: mpd queue=10
*** PLAYBACK STOPPED after 9 tracks (queue len=10, pos=10) ***
final: mpd queue len=10 planned_queue=1 state=stopped
```

The dead-code branch immediately below the condition — `if mpd_state == "stopped":
self.mpd_controller.play()` — is nested inside the refill block, so it never runs either.
Corroborating real-world evidence: the entire feedback history is one 38-minute session, which is
roughly ten tracks.

**Where.** `queue_manager.py:33–58` (condition at line 48); `queue_manager.py:159–165`

> **Fix direction (revised per D1/D2).** With `consume on` forced at session start, MPD removes
> finished tracks itself, so `len(get_queue())` is exactly 2 during playback (current + next).
> Refill when it drops below 2. The `#N/M` position parsing the original audit called for is no
> longer needed — it existed only to compensate for consume being off.
>
> `QueueManager` shrinks to roughly two methods: `ensure_one_ahead()` and `replace_next()`. Delete
> `planned_queue`, `currently_queued_in_mpd`, `_sync_to_mpd`, and the 5% trajectory blend.

---

### C2 · MPD's random mode silently discards every ordering decision the DJ makes. `DONE — Stage 2`

> **Status.** `start_session()` reads the modes, forces `random`/`repeat`/`single` off and `consume`
> on, and logs exactly what it changed (`random on → off, consume off → on`) with a note that they
> are restored on exit. The restore is wired into `_shutdown()`, the signal handler **and** an
> `atexit` hook, guarded by a lock and a flag so running three times is harmless.
>
> Modes are carried as raw strings, not booleans: `single` is `off`/`on`/`oneshot` in modern MPD, and
> restoring a user's `oneshot` as `off` would be a silent change of their setting. Verified live —
> after `kill -TERM` the machine's original `random on / consume off` was back byte-identically.


**What.** The application never asserts MPD's playback modes. If `random` is on, MPD picks an
arbitrary queue entry on auto-advance and on `mpc next` — so the ordering `QueueManager` computes is
thrown away at the last step. The same applies to `repeat` and `single`.

**Evidence.** This is not hypothetical; it is the live state of the development machine:

```
$ mpc status
volume:100%   repeat: off   random: on   single: off   consume: off
```

The failure is completely silent. Nothing in the TUI, the console panel, or the logs indicates that
selection is being overridden.

At depth 1 the failure mode changes character: with only two entries in the queue, `random on` means
MPD may replay the current track instead of advancing to the chosen one — so the DJ appears to be
stuck rather than merely mis-ordered.

**Where.** `mpd_controller.py` (no mode management anywhere); `main_tui.py:138–159` `start_session()`

> **Fix direction.** In `start_session()`, read the current modes, force `random off` / `repeat off` /
> `single off` / **`consume on`** (D2), log what was changed to the console panel, and restore the
> originals in `_shutdown()`. Silently clobbering a user's MPD config is worse than telling them.
> Because `_shutdown()` is currently unreachable on SIGTERM (H3), the restore must be wired into the
> fixed signal path — otherwise an abnormal exit leaves the user's MPD in consume mode.

---

### C3 · The audio fingerprints are non-deterministic ~10-second crops, not track representations. `DONE — Stage 1`

> **Status.** Full-coverage 10 s windows with an end-aligned tail, RMS-gated, mean-pooled, per-window
> matrix persisted: 24,494 windows over 674 tracks in 5m 23s. Embedding the same file twice is
> bit-identical, and re-embedding reproduces the vector already on disk — both asserted in
> `tests/test_clap_pipeline.py`. `failed.txt` names the 16 failures. Actual figures in §10b.

**What.** `CLAPEmbeddingGenerator.generate_embedding()` loads the entire waveform and hands it to
`ClapProcessor` with default arguments. The effective defaults for this checkpoint are
`max_length_s=10` and `truncation="rand_trunc"` — the feature extractor takes a **uniformly random
ten-second crop** of anything longer. The result: each track's embedding describes roughly ten
seconds sampled at random from it, and re-running generation produces a different vector for the same
file.

*(`truncation` deserves care. `ClapFeatureExtractor`'s class default is `"fusion"`, but
`laion/clap-htsat-unfused` ships a `preprocessor_config.json` that sets `"rand_trunc"`, and the
checkpoint's value is the one that applies. `"fusion"` is not merely the wrong mode for an unfused
model — it stacks four mel crops into a 4-channel tensor that this model's single-channel patch
embedding cannot consume at all, and raises. So the mode is not a free choice, and it is not
inherited: see the `Determinism` bullet below.)*

**Evidence.** A synthetic three-minute signal with three musically distinct sections, embedded 10×
through the project's own code path, compared against the real 616-track library's similarity
distribution:

| Measurement | min | median | max |
|---|---|---|---|
| One track embedded 10 times — similarity to itself | 0.354 | 0.884 | 0.998 |
| Two different random library tracks | −0.175 | 0.582 | 0.994 |

36% of same-track pairs scored below the median similarity of two unrelated tracks. More than a third
of the time, a song is less recognisable as itself than a stranger is. Every downstream number —
taste vector, session vector, novelty, the vibe description — is computed on top of this.

Additionally: 16 of 632 tracks failed during the recorded generation run and no list of which ones
was retained, so those files are permanently unreachable by the DJ with no diagnostic.

**Where.** `embedding_generator.py:219–270` (`generate_embedding`, processor call at 234–238);
`embedding_generator.py:179–217` (`load_audio`)

#### Fix direction — resolved

**The window length is not a tunable parameter.** `laion/clap-htsat-unfused` uses an HTSAT audio
encoder with a fixed input size: 10 seconds at 48 kHz. `max_length_s=10` is not a default you can
raise — feeding it 30 seconds only changes *which* 10 seconds survive. So "what is the ideal number
of seconds" has one answer, 10, and it was never the real question.

The real question is **how many windows, placed where, pooled how.** And the right move is to delete
the "how many" parameter rather than tune it:

**Cover the entire track with non-overlapping 10-second windows.**

- **Placement.** Windows at `[0:10], [10:20], [20:30], …`. The final window is *end-aligned* — it
  covers the last 10 seconds of the track, overlapping its predecessor — so the tail is neither
  dropped nor zero-padded. Zero-padding is actively harmful: CLAP maps silence to a consistent
  direction that then contaminates the pooled vector.
- **Determinism.** `generate_embedding()` slices the already-loaded waveform tensor into exact
  480,000-sample chunks before calling the processor, instead of handing it the whole track. At
  exactly `max_length_s` the random-crop branch inside `ClapFeatureExtractor` cannot fire — there is
  nothing left to crop — and neither can the padding branch.

  **The truncation mode must still be pinned to `rand_trunc` explicitly.** At exactly `max_length_s`
  the two modes do not converge: `rand_trunc` yields one channel from the slaney-normalised mel
  filters, `fusion` stacks four copies computed from the htk filters, and this model's patch embedding
  accepts one channel. Relying on the checkpoint's config to supply the right value works today and
  would break silently if a future `transformers` stopped honouring that override, so
  `CLAPEmbeddingGenerator.TRUNCATION` sets it and a test asserts both it and the checkpoint's value.

  > **This is a code change, not a manual one.** `load_audio()` already returns the full waveform as a
  > single tensor; the slicing is a few lines of `torch` indexing inside the generator, applied
  > automatically to every file by `generate_embeddings.py`. Nothing is edited per-track and no audio
  > files are touched. The library size is irrelevant to the effort — it is the same code path whether
  > you have 6 tracks or 6,000.
- **Silence gate.** Drop windows whose RMS falls below a threshold (lead-in, run-out, gaps between
  movements). If *every* window is below threshold — a genuinely quiet ambient track — keep them all
  rather than returning nothing. A binary gate, not continuous energy weighting: one threshold is
  easier to reason about and to defend than a weighting curve.
- **Pooling.** Mean-pool the surviving windows, then re-normalise.
- **Persist the window matrix.** Store the per-window embeddings in the `.npz` alongside the pooled
  vector. Built: 674 tracks × 36.3 windows average × 512 dims × float32, **45.5 MB** for the whole
  artifact. This is the important part: it turns pooling from a regeneration-cost decision into a
  load-time knob. If you later want to try 8 evenly-spaced windows, or medoid clustering, or
  max-over-windows similarity, you experiment in seconds instead of re-running generation.

**Why mean-pool rather than max-over-windows.** Max similarity would let a seven-minute track with one
matching ten-second passage score as high as a track that matches throughout — musical whiplash. Mean
answers "does this *whole track* fit where the session is," which is the question a DJ selecting whole
tracks is actually asking. Persisting the windows keeps the alternative available without committing
to it.

**Cost — measured.** 24,494 forward passes for 674 tracks, **5m 23s** on the 3070 at 75.8 windows/s.
The bottleneck is not the GPU and not audio decode: it is `ClapFeatureExtractor`'s single-threaded
numpy mel extraction, which caps a batched pipeline at ~39 windows/s no matter how large the batch.
M8 is a prerequisite, but the part of it that pays is running the *feature extraction* on a worker
pool, not the batching. See M8 and §10b.

**Also.** Persist the failed-track list to `data/embeddings/failed.txt` with the exception per file.
Built: all 16 failures are one album of corrupt FLACs — the same 16 the original run lost silently.

---

### C4 · Pressing `[V]` always throws away the first track of the new vibe. `DISSOLVED by D8 — constraint enforced and tested in Stage 2`

> **Status.** `DISSOLVED — constraint now enforced and tested (Stage 2).` `[V]`, `_skip_vibe()`,
> `process_vibe_skip()` and `recalculate()` are all deleted (D8/H9), so this code path no longer
> exists. The constraint it established governs the replacement skip path and is asserted directly:
> `tests/test_skip_path.py` drives the real `AdaptiveDJWithTUI.skip_current_track` and checks the
> **call log** — exactly one `next`, no `play` anywhere, and every `add` before the advance.
>
> **One correction from the live MPD.** The fix direction below says the paused case "resolves
> itself" because a paused player would not advance. It does: `mpc next` while paused consumes the
> track *and resumes playback*. Skipping the advance would therefore have replayed the very track
> being rejected. The shipped path advances and then re-pauses, which is safe because `mpc pause` is
> idempotent rather than a toggle — both verified. A stopped player is guarded instead, since
> `mpc next` there is an error that changes nothing.

**What.** A double-advance. `TUI._skip_vibe()` calls `process_vibe_skip()`, which calls
`QueueManager.recalculate()`. That method clears the MPD queue, generates fresh tracks, syncs them,
and — if playback was active — calls `play()`, which starts MPD at queue position 1. Control returns
to `_skip_vibe()`, which then calls `mpd_controller.next_track()`, jumping to position 2.

```python
tui.py:364   def _skip_vibe(self):
                 t = self.current_status.get("track_file")
                 if t:
                     self.dj.feedback_handler.process_vibe_skip(t)   # → recalculate() → play() [pos 1]
                 self.dj._last_skip_time = time.time()
                 self.dj.mpd_controller.next_track()                 # ← skips past it        [pos 2]
```

The single highest-scoring track for the newly chosen direction — the one the whole vibe shift exists
to surface — is never heard.

**At depth 1 this becomes fatal rather than wasteful.** `recalculate()` produces a one-track queue;
`next_track()` then advances past the only entry and MPD goes to `stopped`. Every `[V]` press would
kill playback.

**Secondary case:** if `[V]` is pressed while paused, `recalculate()` sees `was_playing == False` and
never calls `play()`, so the subsequent `next_track()` acts on a stopped MPD and the session stalls.

**Where.** `tui.py:364–370`; `queue_manager.py:60–84`; same pattern in `main.py:266–270` *(file being
deleted, D7)*

#### Fix direction — corrected for depth 1

The original audit said "drop the `next_track()` call from `_skip_vibe()`". **That is wrong under D1.**
It was correct only because `recalculate()` cleared the queue and called `play()`, which restarted
MPD at position 1. Once `recalculate()` no longer exists, `[V]` *must* advance — the currently playing
track is part of the vibe being rejected.

The real defect is that `recalculate()` conflates two operations. Delete it. With `[V]` also gone
(D8/H9), there is exactly **one** skip path:

1. Adjust the session vector — repel from the skip-run centroid, magnitude solved for the escalating
   turnover target, projected back onto the manifold (H9).
2. Delete queue position 2, re-pick under the new vector, add.
3. `next_track()` — one advance, into the new pick.

No `play()` call anywhere in it, so no double-advance is possible by construction. The paused case
resolves itself: if MPD is paused, step 3 is skipped and the new lookahead simply becomes what plays
on resume.

`_last_skip_time` must still be stamped before the advance so the track-change detector does not
count the abandoned track as a full listen.

---

### C5 · The similarity scale is compressed; every scoring constant is calibrated against a range that does not exist. `DONE — Stage 1`

> **Status.** The centroid is computed at generation, stored in the artifact and applied by
> `TrackLibrary.load_embeddings()`; there is no path that scores on the raw space, and a file without
> a centroid is refused rather than silently loaded (M5). On the rebuilt library, random pairs move
> from **+0.670 to +0.011** (median 0.675 → −0.049, p05 +0.363 → −0.562).

**What.** CLAP's embedding space is strongly anisotropic — embeddings occupy a narrow cone rather than
spreading over the unit sphere. Two *unrelated* tracks of this library sit at a mean cosine similarity
of **0.670** (median 0.675, std 0.183, 5th percentile 0.363).

That figure is *worse* than the 0.577 the first audit measured, and for an instructive reason: 0.577
was measured on single random ten-second crops. Mean-pooling ~36 windows per track (C3) averages each
vector toward the library mean, so full coverage buys a truer representation of the track at the cost
of an even narrower cone. **Fixing C3 made C5 more necessary, not less.**

Nothing in the codebase accounted for this, and several places assume the full `[−1, 1]` range is in
play:

| Site | Assumption | Reality, uncentred |
|---|---|---|
| `track_selector.py` | `(sim + 1) / 2` maps similarity to `[0, 1]` | Real values land in ~`[0.68, 0.99]` — two thirds of the scale is never used |
| `track_selector.py` | `novelty = (1 − max_sim) / 2` spans `[0, 1]` | Clusters near 0.16 with very little spread; the novelty weight γ has almost nothing to act on |
| ~~`session_state.py` momentum thresholds~~ | ~~`0.85 / 0.7 / 0.5` discriminate~~ | Deleted in Stage 0 (D4) — they were unreachable branches for the same reason |
| `config.py` weights α/β/γ/δ | terms are comparable in magnitude | session/taste terms vary over a ~0.3-wide band, novelty over ~0.1 — so γ is effectively smaller than its nominal value |

This is why the weights never felt like they did much: three of the four scoring terms are nearly
constant across candidates, so ranking is dominated by whichever term happens to retain variance.

**Fix direction.** At load time, subtract the library centroid from every embedding and re-normalise:

```
E_centred = normalise(E − mean(E, axis=0))
```

Store the centroid in the `.npz` so the same transform can be applied to anything computed later.
This is standard practice for contrastive embedding spaces ("all-but-the-top" / mean-centring) and it
costs one line. Post-centring, random pairs sit at +0.011 (p05 −0.562, p75 +0.297) and the full range
is usable, so:

- `(sim + 1) / 2` means what it claims
- novelty actually spans its range
- the weight constants can be tuned against something real
- the descriptor bank in H1 becomes far more discriminative

**Order matters:** do this before re-tuning any constant, or you will tune twice.

**What Stage 2 inherits from this.** Every similarity in §10 was measured in the uncentred space, and
the space the code now runs in is a different one — a skip nudge that moved 3.9% of the pool there
moves 0.3% here, because the neighbourhood is tighter. The re-measurements are in **§10b**, and the
findings that depend on them (H6, H9) carry the new numbers. Anything still quoting a raw-space
figure is provenance, not a target.

**Consequence for L7.** Post-centring, the library mean is the zero vector, so "seed the taste vector
from the library centroid" — the original audit's suggestion — is degenerate. See L7 for the
replacement.

---

## 3 · High-severity findings

---

### H1 · The mood word in every vibe description is mathematically pinned to "eclectic". `DONE — heuristic deleted (Stage 0); bank built (Stage 1); display shipped (Stage 3)`

> **Status.** Closed. The defective code went in Stage 0 (D4), the **data** was built in Stage 1
> (D5a/D5b) — `descriptors.npz`, 49 descriptors across the six axes below, their CLAP text-tower
> embeddings and each one's mean and std over the centred library — and Stage 3 wired the display:
> the vibe readout (H1b), the Session panel (H1c) and the `[I]` extension (H1d) all ship.
> `vibe_readout.VibeReadout` owns the readout; `session_history.SessionHistory` owns the panel.
>
> **The zero-vector hazard is gated, and the gate is where it has to be.** Stage 2 made the session
> vector start at **zero** (L7), a state that did not exist when H1 was specified — and z-scoring a
> zero vector does not fail loudly. Every `sim` is exactly 0, so `z = −mean_d / std_d`: a finite,
> deterministic, plausible-looking readout determined entirely by the bank's own baselines. On the
> real bank it reads `shimmering · orchestral · serene`, about nothing at all. **That is H1's original
> defect in a new costume**, and the bank cannot defend against it because the arithmetic is valid.
> Every entry point in `VibeReadout` is gated on `SessionState.is_seeded()`, and `[I]` gates the taste
> row on `UserTaste.is_seeded()` separately. Pinned in three places:
> `test_descriptor_bank.py::test_z_scoring_a_zero_vector_produces_a_confident_looking_readout` (the
> hazard is real), `test_vibe_readout.py::test_a_zero_session_vector_is_refused_even_though_the_bank_answers`
> (both halves at once), and `test_tui_display.py::test_the_vibe_line_refuses_before_anything_has_played`
> (through the real `SessionState`).
>
> **Point 6's consistency word shipped as a count, not a cosine, and not a word.** See the amended
> point 6 below: the cosine H1 specified turns out to be compressed into the top 5% of its range for
> 90% of ordinary listening, which is the same failure this finding is about. The measurement is in
> §10d and the reasoning in §0b.
>
> The spot-check the acceptance step asks for is in §10b and reproduces exactly under the shipped
> code: Bathory's orchestral intro comes back as `cavernous · orchestral · dense`, the Arcane score as
> `cinematic · tense · menacing`. Live, Fleshgod Apocalypse's *King* read
> `piano-led · halftime · triumphant`. The words are also still reachable from the command line —
> `python3 generate_embeddings.py --describe "Arctic Monkeys"`.

**What.** `SessionState.get_vibe_description()` derives its mood word from the entropy-like quantity
`−Σ|v|·log(|v|)` over the session vector, then branches on `> 5.0 → "eclectic"`, `> 4.0 → "diverse"`,
else `"cohesive"`. For a 512-dimensional unit vector that quantity is always around 55. The thresholds
appear to have been chosen for a much smaller dimension.

**Evidence.** Computed over all 616 real embeddings, and over 200 simulated session vectors built by
the actual EMA update rule:

```
vector_entropy over 616 real CLAP vectors: min 53.048  max 56.836  mean 54.840
simulated SESSION vectors:                 min 53.103  max 56.599  mean 54.849

fraction > 5.0 ("eclectic"):    100.0%
fraction > 4.0 ("diverse"):       0.0%
fraction ≤ 4.0 ("cohesive"):      0.0%
```

The `"diverse"` and `"cohesive"` branches are unreachable. The README's own example screenshot shows
`"focused cohesive vibe, deep in the zone"` — a string the program cannot produce.

**Where.** `session_state.py:197–217`

#### Fix direction — superseded by D5

The original audit suggested replacing the entropy heuristic with mean pairwise cosine distance over
`recent_tracks`. That is more honest but still produces a number nobody can read, calibrated against
thresholds you would have to invent. Replace the whole function instead.

**Embed a bank of descriptor prompts with CLAP's text encoder and name the vibe.**

**1. The bank.** ~50 prompts across complementary axes, embedded once and cached to
`data/embeddings/descriptors.npz`:

- *Energy* — calm, gentle, mellow, driving, energetic, aggressive, intense, frenetic
- *Affect* — melancholic, sombre, wistful, uplifting, joyful, triumphant, tense, menacing, serene, romantic
- *Texture* — sparse, dense, lo-fi, polished, distorted, warm, cold, shimmering, gritty, spacious
- *Rhythm* — hypnotic, danceable, groovy, syncopated, free-time, motorik, halftime
- *Setting* — nocturnal, sunlit, cinematic, intimate, cavernous, dreamlike
- *Instrumentation* — acoustic, electronic, orchestral, vocal-led, instrumental, guitar-driven, piano-led, synth-heavy

**2. The prompt template matters.** CLAP was trained on audio captions, not bare adjectives. Use
`"This is a recording of {} music."` rather than `"{}"`. Worth a one-off comparison of two or three
templates, scored by how well the bank separates the library (below).

**3. The modality gap is the trap.** CLAP's audio and text towers are contrastively aligned but do not
share a cone — raw audio·text dot products are **not comparable across descriptors**. "electronic"
may score 0.31 against everything while "motorik" scores 0.08 against everything, and a naive top-3
would return the same three words forever. This is the same failure shape as the entropy bug, and it
is easy to walk into.

**The fix is the same as C5: normalise per descriptor against the library's own distribution.** For
each descriptor *d*, compute its similarity to all N library tracks once, keep `mean_d` and `std_d`,
then report

```
z(d, v) = (sim(d, v) − mean_d) / std_d
```

Now "hypnotic" means *"this session is unusually hypnotic **for your library**"* — which is the
correct semantics for a personal collection, and is robust to the modality gap.

**4. Validate the bank empirically.** Any descriptor whose `std_d` over the library is near zero
carries no information about *your* music — drop it. This is a concrete calibration step with a real
pass/fail, not a guess, and it is the reason to build the bank before wiring the display.

**5. Build it in Stage 1, not with the UI.** It is a data artifact derived from the embeddings, and
building it early retires a real risk: if the descriptors do not separate *this* library, H1's design
fails and you want to discover that from a script, not midway through a TUI rewrite.

**6. The readout.** Top-3 descriptors by z-score for the session vector. As shipped, two rows:

```
♪ hypnotic · nocturnal · sparse
⟳ 2 of 3 held over 5 tracks · 14 played
```

> **Amended by measurement, Stage 3.** This point originally specified the consistency figure as *the
> cosine between the session's descriptor z-vector now and five tracks ago*, with thresholds "to be
> calibrated against observed drift rather than invented". Driven through the shipped code over 40
> sessions on the real library (§10d), that cosine has p10 = 0.948 and median = 0.989 — **90% of
> ordinary listening sits in the top 5% of its range**, so any word derived from it would be pinned to
> one branch. That is precisely this finding's own defect, and C5's, arriving a third time.
>
> What occupies the range is *how many of the three words on screen were also on screen five tracks
> ago*: measured 0 in 1% of readings, 1 in 12%, 2 in 42%, 3 in 45%, and 0 or 1 after a skip run. So
> the shipped readout reports that count. It invents no threshold, needs no calibration, has no
> vocabulary to justify, and is a statement about something the listener actually read.
>
> The cosine is still measured and still shown — in `[I]`, with its own distribution printed beside it
> so a reader can tell 0.99 from 0.95. Neither line renders at all until `is_seeded()` is true.

#### TUI consequences `ALL SHIPPED — Stage 3`

The vibe readout is the payoff of D1 — it replaces the queue panel as the window into what the system
is doing. The layout changed accordingly:

- **Delete the "Upcoming Queue" panel.** There is nothing to list. — *Done: Stage 2 cut it to one
  line, Stage 3 replaced it.*
- **Add a "Session" panel** in its place: one `↓ next: <artist — title>` line at the top, a divider,
  then session history newest-first with feedback marks (`♥` liked, `⏭` skipped, `✓` full listen).
  This is what you actually wanted visibility into — it is truthful, because it happened. — *Done,
  plus `♪` for the track playing now, which would otherwise be the one row with no mark at all.*
- **Repurpose `↑↓` + `ENTER`.** They currently index into the queue list and replay the session's first
  track (H2). Rebind to: scroll history, `ENTER` on a history entry **requeues that track as `next`**.
  A "play that again" action the app currently lacks, reusing existing plumbing. — *Done. The indices
  are `SessionHistory`'s own, newest-first, sharing no numbering with MPD; the plumbing reused is
  `QueueManager`, which gained `requeue_next()` (§0b).*
- **Repurpose `[I]`.** With time context gone (D6) the overlay has no content. Make it the model
  inspector: top descriptors for the **session** vector *and* the **taste** vector side by side,
  current exploration value, tracks played, taste update count. That is the honest answer to "what
  does the system think it has learned," and it reuses the descriptor bank. — *Done in two parts:
  Stage 0 built the inspector, Stage 3 added the `DESCRIPTORS` block and made the overlay scroll.*
- **L3 becomes a blocker.** Album-art geometry is hardcoded to `RIGHT_COL_ROWS = 10`, the exact row
  count of the current Now-Playing pile. The vibe line is changing shape, so this constant is about to
  be wrong. See L3. — *Done first, as instructed. It was already wrong; see H8.*
- **Update the footer and the README keybinding table together** — they already disagree (L9). —
  *Done, together with the fallback mode, which disagreed with both.*

---

### H2 · The "Upcoming Queue" panel lists tracks that already played. `DONE — panel removed in Stage 2, replaced in Stage 3`

> **Closed.** Stage 2 removed the panel, `get_upcoming_tracks()`, and the `↑↓`/`ENTER` bindings that
> indexed into `mpc playlist`. Stage 3 put a **Session** panel in its place (H1c) which lists the same
> tracks on purpose — as history, newest-first, marked with what happened to them — and rebound `↑↓`
> and `ENTER` to indices derived from this session's own plays. `test_session_history.py::test_enter_
> targets_the_row_under_the_cursor_not_the_first_track` is the direct regression guard.

**What.** Same root cause as C1. `QueueManager.get_upcoming_tracks()` returns the whole MPD playlist,
so the panel shows the session's history above the current track, numbered as if it were the future.
The numbering counts from the top of the queue rather than from the current song, and `↑↓` navigation
plus `ENTER`-to-play index into that same list — so selecting "1." plays the first track of the
session again.

**Status.** `DONE — Stage 2.` `get_upcoming_tracks()`, `_queue_navigate()` and
`_queue_play_selected()` are deleted, along with the ↑↓/ENTER bindings and their footer entries. The
panel is now a one-line `↓ next:` readout titled "Up Next"; its geometry is deliberately unchanged
because the album art is still pinned to hand-counted row constants and **H8 must land before the
layout moves**. Stage 3 replaces it with the session-history panel — **re-derive those indices from
scratch rather than porting these**, which is the whole lesson of the `ENTER` misbehaviour.

---

### H3 · Ctrl-C and SIGTERM neither exit nor save. `DONE — Stage 2`

> **Status.** The handler sets both flags, restores the MPD modes directly — the one thing that
> alters state outside this process, so it must happen even if urwid never yields — and unblocks the
> main loop through a **self-pipe** (`loop.watch_pipe`) rather than raising `ExitMainLoop` from a
> signal context, which urwid cannot receive from an arbitrary point. An `atexit` hook covers paths
> the handler misses, including an unhandled exception. State is additionally checkpointed every
> `config.checkpoint_every_n_tracks` full listens rather than only at exit.
>
> Verified live: `kill -TERM` mid-session exited cleanly, wrote all four state files, and left the
> user's MPD exactly as it was found.


**What.** `AdaptiveDJWithTUI._signal_handler` sets `self.running = False` and nothing else. That stops
the background thread, but the urwid `MainLoop` is unaware of it — `_periodic_update` reschedules on
the TUI's own separate `self.running` flag, which is still `True`. The result is a live-looking UI
with no MPD polling, no track-change detection and no queue management behind it.

Because `persistence.save_all()` only runs in `_shutdown()`, which is only reached after the main loop
exits, a SIGTERM (terminal closed, `systemctl stop`, session logout) discards the entire session's
exploration state and feedback history.

**Newly relevant:** C2 forces `consume on` and restores it in `_shutdown()`. Under the current signal
handling that restore never runs, so an abnormal exit leaves the user's MPD in consume mode — a
visible, confusing side effect on their next manual use of `mpc`. This upgrades H3 from "you lose a
session of learning" to "you leave the user's system altered."

**Where.** `main_tui.py:245–247`, `main_tui.py:249–261`; `tui.py:487–490`

> **Fix direction.** Have the signal handler set both flags and unblock urwid —
> `loop.set_alarm_in(0, lambda *_: (_ for _ in ()).throw(urwid.ExitMainLoop()))` or write a byte to a
> self-pipe the loop watches. Register an `atexit` hook for the MPD mode restore specifically, so it
> survives paths the signal handler misses. Separately, checkpoint state every N tracks rather than
> only at exit.

---

### H4 · Skipping does not change what plays next. `DISSOLVED by D1`

**What.** `process_skip()` raises exploration and nudges the session vector, but a comment explicitly
declines to rebuild the queue: *"[N] = skip one track, keeps the current queue direction"*. Since the
queue holds ten tracks all chosen under the pre-skip weights, up to ten more songs of the rejected
direction play before any adaptation is audible. The README promises the opposite.

**Status.** At depth 1 there is exactly one lookahead track. `[N]` drops it and re-picks under the new
weights, so adaptation is audible on the *very next song*. The original audit's fix — "regenerate the
tail after N consecutive skips" — was a workaround for a problem that only existed because the queue
was deep, and is not needed.

`ExplorationController.consecutive_skips` turned out to have the opposite fate: H9 makes it the
*evidence* driving the skip escalation, so it is now load-bearing rather than unused. It is also
reported in `[I]`, alongside the turnover the next press will target.

---

### H5 · The day-of-week exploration modifier is dead code. `DONE — deleted in Stage 0 (D6)`

**What.** `ExplorationController.get_exploration_factor()` is the only place the weekday/weekend
modifier is applied. Nothing in the running application ever calls it — a grep across all source finds
references only inside test files. `get_weights()`, the method actually used for scoring, reads raw
`self.exploration` and ignores the modifier entirely. The README documents the feature as live.
`test_phase3.py` tests the modifier in isolation and passes, which is how it went unnoticed.

**Where.** `exploration_controller.py:71–90` (uncalled), `exploration_controller.py:92–128` (used);
`config.py:89–91`

**Status.** Done in Stage 0. `time_context.py`, `get_exploration_factor()`, the config keys and the
README paragraph are all gone; `tests/test_deletions.py` fails if any of the names reappear outside an
explanatory comment.

---

### H6 · Selection is strictly greedy — "exploration" never actually explores. `DONE — Stage 2`

> **Status.** `p(i) ∝ exp(−i/τ)` over rank, with τ mapped linearly from the exploration scalar and
> floored at `config.tau_min`. The shipped map reproduces the table below exactly — a test recomputes
> p(rank 0) from the code and requires 63% / 12% / 6% at exploration 0.1 / 0.4 / 0.7 — and another
> samples 40,000 draws and compares the empirical distribution against `exp(−i/τ)`. Below
> `config.minimum_sampled_pool` candidates it falls back to a uniform draw, per the guard below.
>
> The generator is injectable, so sampling did not cost reproducibility: one test asserts the same
> seed gives the identical session, which is what makes the *different*-seeds assertion meaningful.
>
> **τ_max = 15 remains uncalibrated by listening.** It is the one genuinely new constant and it is
> still a starting point, not a finding.


**What.** `TrackSelector.select_track()` sorts the candidates and takes index 0. There is no
epsilon-greedy step, no softmax sampling, no temperature, no tie-breaking jitter. The exploration
scalar only reshuffles the weights of a deterministic argmax, so a higher exploration value produces a
*different deterministic answer*, not a more varied one.

**Where.** `track_selector.py:83–89`

**Why this is now a blocker rather than a quality issue.** At depth 10, the within-batch exclusion set
forced ten *distinct* tracks, which masked the greediness — you got variety as a side effect of
bookkeeping. At depth 1 that mask is gone. Each pick is an independent argmax over a candidate pool
that barely moved (the session vector shifts by `1 − decay` per track) with weights that barely moved.
Consecutive picks will cluster hard, and the only thing preventing literal repeats is the
anti-repetition term. **D1 cannot ship without H6.** They are one change.

#### Fix direction — Boltzmann sampling over **rank**, not over score

The obvious fix is a softmax over the scores with a temperature. Do not do that. Score-softmax needs a
temperature calibrated to the *width of the score distribution*, and that width moves constantly here:
it changes when C5 re-centres the space, when the exploration controller reshuffles α/β/γ/δ, when β
ramps in (L7), and it would change again on a different library. You would be re-tuning a constant
against a moving scale — the exact failure pattern behind H1, C5 and the momentum thresholds.

**Sample over rank instead.** Sort the candidates as now, then draw index *i* with

```
p(i) ∝ exp(−i / τ)          i = 0, 1, 2, … over the scored candidate list
```

τ is the only parameter, and it is **scale-invariant** — it does not care what the scores look like,
only their ordering. It never needs recalibration.

**τ is driven by the exploration scalar, and means something legible:** τ is roughly the effective
number of candidates in play.

| exploration | τ | p(rank 0) | behaviour |
|---|---|---|---|
| 0.1 (floor) | 1 | 63% | near-argmax; the best track usually wins |
| 0.4 (mid) | ~7 | 13% | genuine spread over the top ~15 |
| 0.7 (ceiling) | ~15 | 6% | broad; the top ~40 are all live |

Linear map from the existing `exploration_min`/`exploration_max` bounds, so no new constants beyond
`τ_max`. Implementation is three lines — `np.exp(-np.arange(n)/tau)`, normalise, `np.random.choice`.

**Why this is the right shape for this project specifically:**

- It never fully collapses to determinism, even at the exploration floor. "Dynamic yet working" is the
  goal; a strict argmax at low exploration would still give you the identical session twice from the
  same state.
- The candidate list is already bounded (top-100 from `get_candidate_pool`), so even the tail of the
  distribution is a reasonable track, not a random one.
- It is directly reportable in the `[I]` inspector: *"choosing from ~top 7"* is a true statement about
  what the machine is doing, derived rather than invented.
- It composes with the anti-repetition term rather than fighting it — repeats are still suppressed by
  score, so sampling cannot resurrect a track inside the replay gap.

**One guard:** if the candidate list is shorter than a few entries (small library, heavy exclusions),
fall back to uniform choice rather than letting τ dominate a 2-element list.

#### What this actually buys — measured, so the claim is not oversold

30-track sessions simulated from one identical starting state on the rebuilt 674-track library,
centred space, with the 20-track anti-repetition gap applied:

| Selection rule | mean pairwise similarity within the session | distinct sessions from one start state (5 runs) | mean overlap with run #1 |
|---|---|---|---|
| argmax (current) | 0.539 | **1 / 5** | 100% |
| rank-Boltzmann τ=1 | 0.618 | 5 / 5 | 43% |
| rank-Boltzmann τ=7 | 0.811 | 5 / 5 | 31% |
| rank-Boltzmann τ=15 | 0.759 | 5 / 5 | 16% |
| *(library baseline: two random tracks)* | *0.009* | | |

Two things worth reading carefully:

- **Within-session coherence is not the cost — it is a small gain.** Every sampled rule is *more*
  internally coherent than argmax (0.54 → 0.62–0.81), against a library baseline of 0.009. The reason
  is the anti-repetition gap: strict argmax, forbidden from repeating, keeps stepping to the next-best
  unplayed track and walks steadily away from where it started, while sampling circles a
  neighbourhood. Sampling still should not be *sold* as buying diversity inside a session — the effect
  is small and the ordering among τ values is within noise at five runs.
- **Run-to-run variety is the whole gain.** Argmax returns the byte-identical 30 tracks from the same
  state, every time; sampling returns 16–43% overlap. For a system whose premise is an evolving
  session, reproducing the same evening from the same starting point is the failure mode — and it is
  the one argmax guarantees.

The ordering among τ values is within noise at five runs; **τ_max ≈ 15 is a starting point, not a
finding.** Calibrate it in use: raise it until unattended sessions start feeling incoherent, then back
off. It is the one genuinely new constant this plan introduces, and it is flagged here so it does not
quietly become another uncalibrated threshold of the kind C5 and H1 document.

---

### H7 · `mpd_controller.py` defines eight methods twice; the surviving `add_track` swallows failures. `DONE — Stage 0`

> **Status.** All eight duplicates removed, bool-returning versions kept, `add_track` checks
> `returncode` and logs MPD's `stderr` on refusal. `tests/test_mpd_controller.py` parses the class
> with `ast` and fails on any repeated method name, and monkeypatches `subprocess.run` to assert the
> refusal path returns `False`. The original text follows for provenance.

**What.** Eight methods are defined twice in the same class. Python keeps the second; the first is
unreachable but still reads as live code.

| Method | Dead (1st) | Live (2nd) | Behaviour differs? |
|---|---|---|---|
| `add_track` | 259 | 504 | **Yes — see below** |
| `play` | 128 | 452 | returns bool |
| `pause` | 139 | 465 | returns bool |
| `next_track` | 161 | 478 | returns bool |
| `clear_queue` | 272 | 491 | returns bool |
| `get_queue` | 283 | 520 | identical |
| `get_queue_length` | 299 | 537 | identical |
| `update_database` | 319 | 572 | drops the log line |

The one that matters: the dead `add_track` at 259 returns `result.returncode == 0`. The live one at
504 discards the result and returns `True` unconditionally. So `QueueManager._sync_to_mpd()`'s
`if success:` check is decorative — a track that MPD refuses (removed file, stale database, path
mismatch) is recorded as successfully queued, and the "Failed to add track" branch can never fire.

**Where.** `mpd_controller.py` as tabulated; consumed at `queue_manager.py:131–146`

> **Fix direction.** Delete the dead copies, keep the bool-returning versions, restore the return-code
> check in `add_track`. At depth 1 a swallowed `add_track` failure means the queue silently runs dry
> and playback stops — so this is load-bearing for D1, not just hygiene. It is also the diagnostic M3
> and M4 need.

---

### H8 · Album-art geometry is hardcoded to a layout that is about to change. `DONE — Stage 3, first item`

> **Status.** `AdaptiveDJTUI._art_geometry(cols, rows)` derives the rectangle from the widget tree —
> `Frame.header.rows()` and `.footer.rows()`, `Pile.get_item_rows()`, `Pile.contents`,
> `Columns.column_widths()`, `Columns.rows()` — and returns `(x, y, width, height)` or None when the
> area is too small to be an image rather than a smear. `RIGHT_COL_ROWS` and `NP_BORDER_ROWS` are
> deleted and guarded in `test_deletions.py`.
>
> **Two of the four constants were already wrong**, which is the argument for deriving rather than
> re-counting:
>
> - `x = 2`. Its comment read "col 0 = terminal left edge, col 1 = LineBox left border, col 2 = art
>   inner start" — but the LineBox's left border *is* column 0, so the art column starts at **1** and
>   every cover was drawn one column right of its box.
> - `height = RIGHT_COL_ROWS = 10`. Correct for the pile as it stood; Stage 3's two-row vibe readout
>   takes it to **11**, which is exactly the change the finding predicted and exactly what a
>   hand-counted comment cannot notice.
>
> `y = 3` and `width = 33` were right, and remain right — the derivation reproduces them.
>
> **The geometry is now independent of the terminal's height**, because N1's `('pack', …)` fix gives
> the Now Playing box its natural height rather than a share of the body. Under the old weighted
> layout the art's position varied with the terminal, which is why a single hardcoded `y` could be
> correct at one size and wrong at another.
>
> Tested by rendering the real frame at seven terminal sizes and locating the art placeholder in the
> canvas (`tests/test_art_geometry.py`), rather than by asserting the function's own arithmetic back
> at it. The right-hand column's first and last rendered rows are required to be the rectangle's first
> and last, which is the property the deleted constant existed to guarantee.

*See L3 for the original text.* Promoted because the H1 TUI rework changes the Now-Playing pile's row
count, which is exactly the constant `_render_art()` pinned the image against. Fixed as part of the
rework, first, as planned.

---

### N1 · The layout raises `WidgetError` on any terminal shorter than 33 rows. `NEW — DONE Stage 3`

> **Found while doing H8, not by looking for it.** Not a Stage 3 regression: it predates the audit and
> shipped through Stages 0–2 under a green suite of 311 tests, none of which had ever called
> `render()`.

**What.** `main_pile` gave the Now Playing box `('weight', 3, …)` of the body. Its content is a flow
`Pile` wrapping a `Columns` whose left cell is a box `Filler`, so urwid resolves the `Columns` as a
box widget, hands it the weighted height, and raises when it renders its natural height instead.

**Evidence.** Rendering the shipped tree at `HEAD 3558b88` across terminal heights:

```
  80x20 … 80x32   WidgetError: Widget <Columns …> rendered (80 x 12) canvas
                  when passed size (80, 6)!
  80x33 … 80x45   OK
```

Driven live in a pty at 80×24 against that tree, the traceback prints over the interface and the
session is unusable. The same driver against the Stage 3 tree renders, plays, skips and opens `[I]`.

**Why nothing caught it.** §8's Stage 3 preamble says "311 tests pass and not one constructs the
widget tree — the suite covers everything *behind* the display". This is what that costs. The suite
was behavioural and honest about the player; the display was simply outside it.

> **Fix.** `('pack', self.now_playing_box)`. The box is a flow widget, so packing gives it exactly
> `rows()` and the size mismatch cannot arise. Two consequences worth keeping:
> - It renders down to 80×6, clipping the panels rather than raising.
> - The Now Playing panel is the same height at every terminal height, which is what makes H8's
>   derived geometry independent of the terminal.
>
> `tests/test_art_geometry.py::test_the_frame_renders_at_every_terminal_size` and
> `::test_it_renders_down_to_absurdly_small_terminals` cover heights 6–33 and widths 40–90.

---

### H9 · Neither `[V]` nor `[N]` changes much of what you hear, and `[V]` aims off the manifold. `DONE — Stage 2`

> **Status.** `[V]` and everything that served it are deleted. `[N]` repels from the skip-run
> centroid by a λ **solved** for the turnover schedule, snapping onto the manifold from the second
> consecutive press. Live: 5% → 20% → 70% → 100% across four presses, each meeting its target, moving
> from Björk to Watain.
>
> **One correction to the design, found by running it.** The plan says "solve λ for the target, then
> `snap()`". Built that way the offline medians looked right, but the first live session printed
> `Skip #2: … 1% … (target 20%)` — the second press *undid* the first, because `snap()` relocates to
> a 25-track centroid that after a modest λ is largely the neighbourhood the vector started in. The
> snap now lives **inside** the objective (`solve_repulsion(..., snap_result=True)`), so λ is chosen
> against the turnover of the vector that actually selects. After the change, **0 of 160 simulated
> presses moved backwards**. Full account in §0b, item 2, and the numbers in §10c.


**What.** `SessionState.force_shift()` blends the session vector 50% toward a **random 512-dimensional
direction** (`session_state.py:107–127`). The stated purpose of `[V]` is a decisive change of
direction. It is measurably neither decisive nor a direction.

**Problem 1 — the destination is not music, and this is what decides it.** In 512 dimensions a random
unit vector is near-orthogonal to every real embedding (expected cosine 0, σ ≈ `1/√512` ≈ 0.044).
Measured against the rebuilt library, as mean similarity to the 25 nearest real tracks:

| | on-manifold quality |
|---|---|
| a real track | 0.729 |
| an ordinary session vector | 0.641 |
| **after `[V]`** (`force_shift` 0.5) | **0.450** |
| a random direction | 0.085 |

`[V]` blends half of the last row into the second. The candidate pool is then drawn nearest to a point
that is halfway to meaningless, so `[V]` does not mean "different vibe" — it means "weaker vibe, plus
noise." This is the argument that survives every re-measurement, because it is structural rather than
numerical.

**Problem 2 — it moves less than the constant implies.** With `r ⊥ v`, `|0.5v + 0.5r| = 0.5√2`, so
`cos(v', v) = 0.707` — measured at 0.703 across 40 simulated sessions. A "50% vibe shift" that leaves
the new direction 70% aligned with the old one is not what the label promises.

**Problem 3 — measured as candidate-pool turnover, the only thing that determines what you actually
hear next, both keys are tiny:**

| Action | cos(new, old) | pool turnover |
|---|---|---|
| `[N]` ×1 | 0.999 | 0.3% |
| `[N]` ×3 | 0.989 | 1.3% |
| `[N]` ×5 | 0.958 | 2.4% |
| **`[V]` ×1** (`force_shift` 0.5) | 0.703 | **9.3%** |
| `[N]` ×10 | 0.665 | 10.7% |
| `[N]` ×20 | 0.243 | 87.9% |

*(674-track library, centred, 40 simulated sessions, top-100 pool, session-only argmax — method and
caveats in §10b. §10 has the same table measured on the 616 crop-based embeddings, where `[V]`×1 sat
*below* `[N]`×5. That ordering did not survive the embeddings changing — which is itself a reason not
to rest a decision on it, and why the argument above rests on problem 1 instead.)*

**A `[V]` press turns over 9.3% of what you will hear.** Pressing `[N]` ten times turns over 10.7%. So
the honest statement is not that one key is weaker than the other — it is that **neither key changes
direction at all**, and the one advertised as a hard reset buys about ten skips' worth of movement
while pointing a third of the way into noise. The large `cos` figure is what made `[V]` *look*
decisive; turnover is what the listener experiences.

**Where.** `session_state.py:107–127`; `config.vibe_shift_magnitude`;
`feedback_handler.process_vibe_skip()`; `exploration_controller.set_high_exploration()`; `tui.py:364`

#### Resolution — delete `[V]`; make `[N]` escalate

`[V]` was introduced alongside the ten-track queue, where its real job was *clearing the queue*: with
ten tracks committed in advance, rejecting a direction meant ten presses of `[N]`, so a queue-nuking
key earned its place. D1 removes the queue, and with it the only thing `[V]` did that `[N]` could not.
The vector mathematics it carried was never sound — problem 1 is not a tuning issue, it is a
statement about where the vector lands. It is a queue-era affordance and it goes.

But the *intent* behind it — "the problem is the direction, not this song" — is real, and the table
above shows `[N]` does not serve it either: 0.3% turnover per press, and twenty presses before
anything moves. Neither key delivers a change of direction today. So `[N]` is rebuilt to cover both.

**One key. Escalation driven by evidence, not by a second keypress.**

`ExplorationController.consecutive_skips` is already tracked and currently unused — it is exactly the
evidence needed. *n* consecutive rejections is the system observing that the neighbourhood is wrong;
that is strictly better information than a user asserting the same thing, because it arrives without
the user having to diagnose their own dissatisfaction before acting.

On every `[N]`:

1. Taste −0.05, exploration +0.05, `consecutive_skips += 1` *(unchanged)*.
2. Repel the session vector from the **centroid of the current consecutive-skip run**, not from the
   single last track — more evidence, less sensitivity to one atypical song.
3. **Solve for the repulsion magnitude λ** rather than declaring it. Target a pool turnover that rises
   with *n*:

   | consecutive skips | turnover target | λ that reaches it | reads as |
   |---|---|---|---|
   | 1 | 5% | ~0.25 | "not this song" |
   | 2 | 20% | ~0.60 | "not this corner either" |
   | 3 | 50% | ~0.75 | "this is the wrong direction" |
   | ≥4 | 85% | ~0.95 | full reset |

   λ is found by increasing it until the target is met — a few dot-product passes over 674 vectors,
   microseconds. **The λ column is a sanity check on the solver, not an input to it**; the schedule is
   the turnover column, which is why this survived the space changing underneath it (the same solve
   against the uncentred library returned 0.23 / 0.55 / 0.80 / 1.05). The current code uses a fixed
   λ = 0.15 at every step, which is why twenty presses were needed to get anywhere.

4. **Project back onto the manifold** for escalated skips: replace the vector with the normalised
   centroid of its 25 nearest real embeddings.

   ```
   snap(v) = normalise( mean( top-25 library embeddings by dot(E, v) ) )
   ```

   Measured on the rebuilt library: at the 85% target it preserves the turnover (90.2% → 93.3%) while
   holding on-manifold quality at 0.765 — against 0.641 for an ordinary session vector, 0.450 after a
   `[V]`, and 0.085 for a random direction. **This is a structural guarantee, not a tuning:** however
   large λ grows, the session vector cannot leave the region your music occupies. It costs one 674×512
   matrix-vector product.

   **Apply it only for *n* ≥ 2.** `snap()` is a move in its own right, not a projection that leaves
   small displacements alone: at the 5% target it *raises* turnover from 5.7% to 17.8%, overshooting
   the schedule by more than three times. At 50% and 85% it is close to neutral. The audit's original
   "*n* ≥ 2, where λ is large enough to matter" now has a measurement behind it rather than an
   intuition.

5. Replace the lookahead, advance *(as C4)*.

A full listen resets `consecutive_skips` to 0, so escalation decays naturally the moment you stop
skipping — no separate cooldown.

**Why this is the sound version rather than the convenient one:**

- **λ is solved, not chosen.** The only tunable is the turnover schedule, expressed in units a listener
  can verify ("85% of what I would have heard is now different") rather than in vector-space units
  nobody can reason about. This is the same move as C5 and H6.
- **The escalation is evidence-driven.** It responds to accumulated rejections rather than to a mode
  the user has to select in advance.
- **It cannot go off-manifold.** Step 4 makes that impossible by construction, which is what neither
  `force_shift` nor an unguarded large repulsion can promise.
- **It is continuous.** `[N]` and the old `[V]` become two points on one curve instead of two code
  paths with two sets of constants.

**And it stays legible.** After an escalated skip, the descriptor readout (H1) already names where you
landed — `→ sparse · cavernous · cold`. That is the explanation the redesigned `[V]` would have
provided, obtained from the bank that exists for the vibe line anyway. No descriptor *anchors* are
needed, which is why §7's schema does not carry them.

**Deletions.** `force_shift()`, `process_vibe_skip()`, `set_high_exploration()`,
`config.vibe_shift_magnitude`, the `[V]` binding and its footer/README entries.

---

## 4 · Medium-severity findings

---

### M1 · The test suite is green theatre, and it is not in the repository. `M1a/M1b DONE — M1c OPEN`

> **Status (M1a, Stage 0).** `test_*.py` is out of `.gitignore`, the three phase-test files are
> deleted, and `tests/` is tracked: 9 files, 67 tests, green. They cover what Stage 0 changed — the
> weight-sum invariant raising `ValueError` rather than `AssertionError`, the score being exactly the
> four weighted terms, the zero taste vector being inert and unmovable by a lone skip, the candidate
> pool falling back to session-only, `add_track` reporting refusals, the log tee surviving
> `tui_active`, and every deleted symbol staying deleted.
>
> **Status (M1b, Stage 2).** `DONE.` `FakeMPD` lives in `tests/conftest.py`, built to the verified
> table below, and `tests/test_fake_mpd.py` asserts the double against it row by row — the harness is
> itself under test, because a double written from the assumptions that produced C1 would reproduce
> C1 and pass. That paid immediately: a fixture defaulting to `consume off` silently put every
> component back in C1's world, and the replay-gap test caught it.
>
> Behavioural suites landed for refill (`test_queue_manager.py`), the skip path
> (`test_skip_path.py`, driving the *real* orchestrator method and asserting on the call log), mode
> force/restore and the signal path (`test_mpd_modes.py`), rank sampling
> (`test_selection_sampling.py`), skip escalation against the real library
> (`test_skip_escalation.py`), the geometry (`test_manifold.py`), the β ramp (`test_taste_ramp.py`)
> and anti-repetition persistence (`test_play_history_persistence.py`). 179 → **310 tests**.
>
> **Status (M1c, Stage 4).** `DONE.` 416 → **542 tests**. `test_persistence_round_trip.py` covers the
> three state files that had none — `feedback_history.json` (never called directly by the suite at
> all, and load-bearing since L4), `exploration_state.json`, and a *seeded* `user_taste.npz` — each as
> round-trip, then the behavioural consequence, then a missing file, then a corrupt one. Writing the
> "file missing a key" cases found that `ExplorationController.load()` and `UserTaste.load()` both
> assigned field by field, leaving a truncated file half-applied while reporting failure; both are
> atomic now.
>
> `test_simple_mode.py` gives the fallback text mode its first coverage, through a real pty — and the
> mode was changed to deserve it: `decode_key()`/`decode_keys()` turn terminal bytes into urwid's key
> names and `_handle_input` dispatches for both interfaces, so the second binding table L9 is about no
> longer exists. The pty found two pre-existing defects (a buffered-read/`select` mismatch that
> swallowed bursts, and an uninterruptible `[I]` page) and, in a mutation check, catches the exact
> regression L9 records: rebinding `↑` to volume fails two tests.
>
> **And the suite was writing to the developer's live `data/state/`** — `process_like()` saves to
> `config.taste_file` — so a green run replaced a real taste model with a fixture's. Confirmed
> pre-existing by stashing every Stage 4 change. An autouse fixture now redirects all four paths.
> §0b, item 3.

**What.** `test_phase2.py` reports "Passed: 66". Roughly thirty of those are a literal list of
hardcoded pass tuples:

```python
requirements = [
    ("Handle MPD disconnects", True),
    ("Handle empty queue",     True),
    ("Handle terminal resize", True),
    ...
]
for requirement, implemented in requirements:
    test_result(requirement, implemented)
```

The remainder are import checks, `hasattr(tui, '_skip_vibe')`-style existence checks, and
`Path.exists()` checks. No test exercises queue refill, the `[V]` path, scoring, or persistence
round-trips under realistic MPD semantics. That is precisely why C1, C4 and H1 survived.

Current actual status: `test_phase2.py` exits 1 (asserts a `PHASE2_DOCUMENTATION.md` that does not
exist), `test_phase3.py` exits 0, `test_phase3_integration.py` exits 1. And `.gitignore` contains
`test_*.py`, so none of the three files are tracked — `git ls-files` lists no test at all. A fresh
clone has zero tests.

**Where.** `test_phase2.py:300–352`; `.gitignore` ("# Test scripts")

> **Fix direction.** Remove `test_*.py` from `.gitignore`. Delete the hardcoded requirements block and
> the phase-based test files outright — they test a structure that is being rewritten. Write a
> `FakeMPD` that models real semantics (**including consume mode**, per D2) and behavioural tests for:
> one-ahead refill, `[N]` replacing the lookahead, exactly one advance per skip with no `play()` call
> (C4's retained constraint), skip escalation reaching its turnover target, mode force/restore across
> an abnormal exit (C2+H3), and embedding determinism (C3 — assert that embedding the same file twice
> gives an identical vector, which is now a *testable property*).
> Move to pytest so exit codes are meaningful.

#### The MPD semantics `FakeMPD` must model — verified, not remembered

C1 exists because nobody checked how the queue actually behaves. A `FakeMPD` built on the same
assumptions would reproduce the bug and pass. So these were run against the live MPD (23 July 2026,
mpc 0.35, consume on, random/repeat/single off) and are the specification.

**Re-verified in Stage 2** against MPD 0.24.0 before `FakeMPD` was written. All seven rows below
reproduced exactly; four further behaviours were measured and are in the second table.

| Behaviour | Verified result |
|---|---|
| The **currently playing track stays in the queue** | With consume on, `mpc status` reads `#1/4` while playing the first of four. Consume removes a track when you *leave* it, not when you start it. |
| Position is always `#1` | After each removal the new current track is `#1/N`. There is no position to parse — "how many ahead" is `len(playlist) − 1`. |
| Natural end consumes | A track that plays to its end is removed: `playlist` 3 → 2, the next track begins at `#1/2`. |
| `mpc next` consumes | Skipping also removes the abandoned track: 4 → 3. So a skip and a completion look identical to the queue. |
| `mpc del 2` removes the **lookahead** | The current track keeps playing, uninterrupted, at the same position. This is exactly C4's "delete queue position 2, re-pick, add". |
| **`mpc next` on the last remaining track empties the queue and stops** | `playlist` 1 → 0, state `stopped`. |
| **Adding to a stopped queue does not start it** | `mpc add` on an empty stopped queue leaves the state `stopped`. Nothing recovers on its own. |

The last two rows are one trap and they are worth stating as a rule, because they interact with C4's
"no `play()` in a skip path" constraint:

> **The skip path must add the replacement *before* it advances.** Advance-then-add empties the queue,
> MPD stops, and the subsequent `add` will not restart it — so the session dies silently and the only
> way back is a `play()` call that C4 forbids in that path. Add-then-advance never sees an empty
> queue. At depth 1 this is not a nicety; it is the difference between a working skip and a dead
> session.

#### Four more, measured in Stage 2 — three of them new, one contradicting the audit

| Behaviour | Verified result |
|---|---|
| **`mpc next` while *paused* consumes and resumes *playing*** | It does **not** stay paused. C4's fix direction assumed it did, and following that would have made a paused skip replay the rejected track. The shipped path advances and re-pauses. |
| `mpc pause` is idempotent, not a toggle | Which is what makes the re-pause above safe. |
| `mpc next` on a stopped player | `MPD error: Not playing`; the queue is unchanged. The skip path guards on state. |
| `mpc del N` past the end of the queue | Exits 1 (`song number does not exist`). `replace_next()` relies on this when the queue holds only the current track. |

The refill condition follows directly: during playback `len(playlist)` is **2** (current + lookahead),
so refill when it is `< 2`. That is D2's claim, and it holds — measured at depth 2 in 30 of 30
mid-track samples during a live 30-track run.

---

### M2 · Two divergent orchestrators; the stale one is what the setup helper tells you to run. `DONE — Stage 0`

> **Status.** Both files deleted, along with `generate_dummy_embeddings()` and `start.sh`'s demo
> branch. `main_tui.py` now exits with a message pointing at `generate_embeddings.py` rather than
> offering random vectors.

**What.** `main.py` ("Phase 1, headless") and `main_tui.py` ("Phase 2") duplicate about 120 lines of
identical component wiring. They have already drifted apart:

- `main.py` auto-plays on start; `main_tui.py` deliberately waits for `[SPACE]`.
- `main.py` has no `_last_skip_time` guard, so skipping near the end of a track is also counted as a
  full listen — the exact double-count the TUI path was fixed for.
- `main.py`'s `_check_input()` reads whole lines, so every command needs a trailing Enter.
- It carries the same `[V]` double-skip bug as C4.

Nothing runs `main.py` — `start.sh` launches `main_tui.py`. But `setup_check.py` ends by printing
*"Setup complete! You can now run: `python main.py`"*, pointing new users at the stale path.
`setup_check.py` is itself orphaned: not referenced by `start.sh` or the README, and it offers only
random test embeddings.

**Where.** `main.py` (whole file); `setup_check.py:156–159`

> **Fix direction (revised per D7).** Delete both files. The original audit suggested extracting a
> `build_dj()` factory to share wiring — but there is only one consumer, so the factory would exist to
> serve a file that is being removed. `setup_check.py`'s only unique offering is random test
> embeddings, which are actively harmful now that C3 makes real generation fast and reproducible.

---

### M3 · `mpd_music_directory` is an undocumented, unvalidated requirement that works here by accident. `DONE — Stage 1`

> **Status.** `music_directory.py` detects it from `MPD_MUSIC_DIR`, then MPD's own config files, then
> the legacy default *labelled as unverified*; `config.mpd_music_directory_source` says which, and
> startup prints it. Validation resolves five probes spread across `mpc listall`, so both a wrong
> directory and a half-mounted library fail. `start.sh` prompts when detection fails. This machine now
> reads `/mnt/storage/music` from `~/.config/mpd/mpd.conf` rather than depending on the symlink.
>
> **One deviation:** generation refuses on a bad directory, startup only warns — at runtime nothing
> but album art and the mutagen tag read touches the path, and the latter already falls back to
> `mpc search`. Reasoning in §0b, Stage 1, deviations.

**What.** `config.mpd_music_directory` defaults to `/var/lib/mpd/music`. It is used for mutagen tag
reads and album-art lookup, and it is never validated, never prompted for by `start.sh`, and never
mentioned in the README's setup section.

**Evidence.** On this machine MPD's real `music_directory` is `/mnt/storage/music`, and
`MPD_MUSIC_DIR` is unset — so the config value is wrong. It works only because `/var/lib/mpd/music`
happens to be a symlink:

```
$ ls -la /var/lib/mpd/music
lrwxrwxrwx. root root /var/lib/mpd/music -> /mnt/storage/music
```

Without that symlink, album art disappears entirely and every tag lookup falls back from an in-process
mutagen read to a per-track `mpc search` subprocess. More seriously, `generate_embeddings.py`
enumerates tracks by walking this directory with `rglob` — if it is wrong, generation either finds
nothing or produces a file whose keys match no MPD path, and the DJ silently has zero usable tracks.

**Where.** `config.py:17`; `generate_embeddings.py:124–141`; `album_art.py:545–591`;
`mpd_controller.py:375–400`

> **Fix direction.** Read it from MPD's own configuration instead of guessing. (`mpc` has no way to
> report it — it is a server-side path the protocol never exposes, and `mpc --verbose status` only adds
> protocol chatter. Parse `music_directory` out of the config files instead:
> `$XDG_CONFIG_HOME/mpd/mpd.conf`, `~/.config/mpd/mpd.conf`, `~/.mpdconf`, `~/.mpd/mpd.conf`,
> `/etc/mpd.conf`, `/etc/mpd/mpd.conf`, user before system, as MPD itself resolves them.) Fall back to
> prompting in `start.sh`. Validate by resolving real track paths from `mpc listall` — several, spread
> across the list, so a half-mounted library fails too. Do this **before** the C3 regeneration run, not
> after: a wrong music dir wastes the whole run.

---

### M4 · Two different sources of truth for track keys, with no reconciliation. `DONE — Stage 1`

> **Status.** `mpc listall` is the only enumeration left. `generate_embeddings.py` takes its track
> list from `MPDController.list_all_tracks()`; the `rglob` walk is deleted and
> `tests/test_deletions.py` fails if it reappears. `TrackLibrary.reconcile_with_mpd()` logs coverage
> on every load, drops embeddings MPD cannot play, and raises below `config.minimum_mpd_coverage`
> (0.5). Live: `674 of 674 embeddings match MPD (100.0%); 0 stale, 16 MPD tracks have no embedding` —
> those 16 being the corrupt album in `failed.txt`.

**What.** Embedding keys come from two incompatible enumerations depending on which path the user
takes:

| Path | Key source |
|---|---|
| Real CLAP (`generate_embeddings.py`) | Filesystem `rglob` under `mpd_music_directory` |
| Demo embeddings (`start.sh` option 2) | `mpc listall` |
| Fallback prompt (`main*.py`) | `MPDController.list_all_tracks()` → `mpc listall` |

These diverge whenever MPD excludes files, resolves symlinks differently, or applies its own path
normalisation. Nothing ever checks the overlap between the embedding keyspace and MPD's database, so a
mismatch presents as "the DJ picks nothing" or "tracks never play" rather than as an error. Compounded
by H7, where failed `mpc add` calls are reported as successes.

**Where.** `generate_embeddings.py:132–141`; `start.sh:161–179`; `track_library.py:26–82`

> **Fix direction.** Make `mpc listall` the single enumeration source for all paths — it is
> authoritative for what MPD will actually play, which is the only thing that matters. Resolve each
> listed path against the validated music dir (M3) for decoding. At load time, intersect the embedding
> keys with `mpc listall` and log coverage ("612 of 616 embeddings match MPD; 4 stale"). Refuse to
> start below a sane threshold. The demo-embeddings path in `start.sh` should be deleted alongside
> `setup_check.py` (M2).

---

### M5 · No embedding-dimension validation on load. `DONE — Stage 1`

> **Status.** `embeddings_io.validate_embeddings()` checks the required keys, the schema version, that
> the track count matches the embedding count, that the window offsets end where the windows do, that
> the windows share the embeddings' dimension, the centroid's shape, and normalisation. The loader
> adopts the *file's* dimension over the config's, and checks the model **by equality** against
> `config.clap_model_name` — the `SMOKE-TEST-RANDOM-NOT-CLAP` case from §0b is now a test.
> `TrackLibrary` raises `LibraryError` rather than degrading to an empty library. `save_embeddings()`
> was deleted rather than taught to carry metadata (§0b, deviations).

**What.** `TrackLibrary.load_embeddings()` never checks `embeddings.shape[1]` against
`config.embedding_dimension` (512). Meanwhile `UserTaste` and `SessionState` size their vectors from
the config value. Loading a file generated by a different model surfaces as a numpy broadcast error
deep inside `_calculate_score` rather than as "these embeddings are 768-d, expected 512-d".

Related: `TrackLibrary.save_embeddings()` writes only `track_files` and `embeddings`, dropping the
metadata block. A CLAP file round-tripped through it is silently downgraded to "placeholder
embeddings" by the loader's own warning logic on the next run.

**Where.** `track_library.py:26–82`, `track_library.py:231–247`; `config.py:64`

**Also (found during Stage 0).** The CLAP check is `'clap' in metadata['model'].lower()` — a substring
match on an arbitrary string. A throwaway smoke-test file named `SMOKE-TEST-RANDOM-NOT-CLAP` was
greeted with `✓ Loading CLAP embeddings`. Any file whose model string happens to contain those four
letters passes, and any legitimately-CLAP file whose string does not gets the "placeholder embeddings"
warning.

> **Fix direction.** Validate shape on load and adopt the file's dimension rather than the config's —
> the embeddings are the authority. **The `.npz` schema is changing** (C3 adds the per-window matrix,
> C5 adds the centroid), so add a `schema_version` key and refuse to load anything that lacks the
> centroid rather than silently scoring on an uncentred space. Check model identity by **equality**
> against the expected checkpoint name, not by substring. Carry metadata through `save_embeddings` —
> or delete `save_embeddings`, which nothing calls (L8).

---

### M6 · State file misnamed; anti-repetition history never persisted despite a comment claiming it is. `M6a/M6b DONE`

**(a) `DONE — deleted in Stage 0 (D6).`** `config.context_file` was named `time_context.npz`, but `TimeContext.save()`
writes JSON. Loading it as an npz raises `UnpicklingError: invalid load key, '{'` — verified against
the live file. Resolved by deleting the time-context subsystem; delete the file and the config key
rather than renaming.

**(b) `DONE — Stage 2.`** *(`clear_history()` and its misleading comment were deleted in Stage 0 per
L8.)* `TrackSelector.save()` / `load()` write `data/state/play_history.json`, wired into `Persistence`
and the periodic checkpoint. `recent_history` is persisted alongside `play_history` and
`current_index`, because it is the half that actually does the excluding — `play_history` only shapes
the score. `tests/test_play_history_persistence.py` asserts the behavioural version: a track played
just before a restart is still excluded after it.
The method carried the comment *"Don't clear
play_history to maintain long-term anti-repetition"* — but nothing ever saves or loads `play_history`,
and `clear_history()` itself was never called. Both `recent_history` and `play_history` are rebuilt
empty on every launch, so the README's "Recently played tracks are excluded for at least 20 songs"
resets each time you start the program.

This becomes visible rather than merely wrong once the session history panel exists (H1): the panel
shows what played, and the user will notice a track reappearing that the panel says played twenty
minutes ago.

**Where.** `config.py:94` vs `time_context.py:272–283`; `track_selector.py:198–202`

> **Fix direction.** Persist `play_history` and `current_index` alongside the other state, checkpointed
> per H3. Delete `clear_history()` and its misleading comment.

---

### M7 · Setup documentation contradicts itself on every number, and macOS is claimed but unsupported. `DONE — Stage 4`

> **Status (Stage 4).** Every figure re-measured, and — more to the point — **held together by a
> test**, because M7 is a consistency finding: the three numbers drifted apart because nothing bound
> them, so rewriting them once would have fixed nothing.
> `tests/test_documented_numbers.py` asserts the pre-flight check covers what is actually downloaded,
> that both user-facing places state the same figure, that no stale claim survives on a line a user
> reads, that a CPU estimate exists at all, and that `start.sh` uses no bash 4+ syntax.
>
> | | Measured, 23 July 2026 |
> |---|---|
> | Model cache after a first run | **1,232,327,859 B = 1.15 GiB** |
> | Why it is twice the model | `main` carries only `pytorch_model.bin` (614,525,833 B); transformers 5.1.0 also fetches the safetensors conversion from `refs/pr/3` (614,431,440 B). Both `refs/main` and `refs/refs/pr/3` are present in the cache. |
> | Pre-flight check | `MODEL_CACHE_MB = 1176` + `ARTIFACT_MB = 46`, derived rather than chosen. The old `700` passed on a disk the download then filled — worse than no check, because it reads as an endorsement. |
> | GPU run | 5 min 23 s, 674 tracks, 24,494 windows, 75.8 windows/s (§10b, unchanged) |
> | **CPU only**, 12 threads | audio encoder **17.0 windows/s** against the GPU's 333 → **≈ 25–35 min** for the same library. The figure that never existed, for the path that is the default fallback. |
>
> **macOS: the claim is dropped, not tested.** Stage 0 removed the `${VAR,,}` that made `start.sh` a
> hard syntax error there, and a scan finds no other bash 4+ construct — so it would probably now
> *run*. But nothing has been run there, and album art cannot work: the only two surviving renderers
> are ueberzug and ueberzugpp, both X11/Wayland. The README says Linux, and says macOS is untested and
> why. §8's trap 5 was observed — nothing was regenerated to obtain any of this.

> **Done in Stage 0, opportunistically.** `${EMB_CHOICE,,}` is now
> `tr '[:upper:]' '[:lower:]'` — that block was being rewritten to remove the demo-embeddings option
> anyway, so the bash 3.2 syntax error went with it. The README's claims about time context, the
> day-of-week modifier, the mood description and demo embeddings were also removed, because Stage 0
> made them false. **Everything else stands**: the three contradictory download sizes, the runtime
> estimate, the missing CPU figure, the untested macOS claim, and the full rewrite the queue and vibe
> changes will require.

**What.** The CLAP model download size is stated three different ways:

| Source | Claim |
|---|---|
| `README.md:56` | "requires ~1 GB model download" |
| `start.sh:133` | "requires ~4 GB download on first run" |
| `embedding_generator.py:44` | Pre-flight check demands 700 MB free |

Runtime is similarly vague: the README says "~1–60 min depending on library size"; the actual recorded
run was 198 seconds for 616 tracks on an RTX 3070. A CPU-only estimate is never given, and CPU is the
default fallback.

Separately, the README claims "A terminal (Linux or macOS)". `start.sh:138` uses `${EMB_CHOICE,,}`, a
bash 4+ lowercase expansion. macOS ships bash 3.2, where this is a hard syntax error — the script
cannot run there at all. Album art also has no macOS path in practice: ueberzug and ueberzugpp are
X11/Wayland, and sixel needs `img2sixel`.

**Where.** `README.md:34–71`; `start.sh:118–190`; `embedding_generator.py:44`

> **Fix direction.** Measure the real cache size once and use that number everywhere. Re-measure
> generation time after C3 (it will be several times longer per track, and the disk requirement grows
> by the 45.5 MB artifact — measured, §10b). Replace `${VAR,,}` with `tr '[:upper:]' '[:lower:]'`. Either test on
> macOS or drop the claim. **The README needs a rewrite regardless** — D1, D6 and H1 change the
> described behaviour of the queue, the time-context feature, and the vibe display.

---

### M8 · `--batch-size` is advertised, accepted, and completely ignored. `DONE — Stage 1, with a correction`

> **Status.** `--batch-size` now does what it says, and the adjacent bugs are fixed: the resumed count
> is reported separately from what the run did, and the speed line cannot divide by zero.
>
> The adjacent bugs below are fixed too: the resumed count is reported separately from what the run
> actually did, and the speed line cannot divide by zero.

**What.** `CLAPEmbeddingGenerator.__init__` stores `batch_size` and `generate_embeddings.py` exposes
`--batch-size` with a documented default of 16, printing it in the run banner.
`generate_embeddings_batch()` then processes tracks strictly one at a time in a Python loop. On a GPU
this leaves most of the throughput unused — the recorded run averaged 3.1 tracks/sec on an RTX 3070.

Minor adjacent issues in the same file: `stats['successful'] / stats['duration']` divides by zero if
generation completes instantly, and on `--resume` the successful counter is pre-seeded with the
resumed count so the final summary overstates this run's work.

**Where.** `embedding_generator.py:46–68`, `embedding_generator.py:272–376`;
`generate_embeddings.py:339–344`

#### Fix direction — and where the cost actually is

C3's full-coverage windowing produces ~36 windows per track — 24,494 forward passes for the library
instead of 674. Running those one at a time is not viable, so batching stops being an optimisation and
becomes a prerequisite for C3.

**But batching is not what makes it fast.** Measured on the 3070, throughput barely responds to batch
size — 29.2 windows/s at batch 1, 38.2 at batch 32 — because the GPU was never the constraint.
`ClapFeatureExtractor` computes its mel spectrograms in single-threaded numpy and caps the pipeline at
~39 windows/s on its own. Running *that* on a worker pool is what pays: 37 → 59 → 75 → 83 windows/s at
1 / 2 / 4 / 8 threads. The shipped generator overlaps a decode-and-mel thread pool with a batching GPU
loop and sustains 75.8 windows/s.

**Fill batches from one track at a time, not across tracks.** Cross-track packing is marginally more
efficient and costs the property that matters more: GPU reductions depend on batch composition, so a
track's embedding would depend on which tracks happened to sit beside it in the run — and a library
could not be extended without every existing vector shifting under the taste model. Per-track chunking
still batches 32 windows per forward pass; it forfeits one partial batch per track and buys
bit-identical re-embedding, which C3's acceptance criterion asserts and the suite checks.

Because batch composition is load-bearing, `batch_size` is recorded in the artifact's metadata.

---

## 5 · Lower-severity findings and friction

---

### L1 · The SIGWINCH handler is never invoked. `DONE — Stage 4, and the finding was wrong`

> **This finding's claim did not survive being acted on. It is rewritten here to carry the correct
> one.**
>
> The measurement is real and reproduces:
>
> ```
> after _setup_urwid()    <bound method AdaptiveDJTUI._on_sigwinch …>
> after Screen.start()    <bound method Screen._sigwinch_handler …>
> ```
>
> **But "urwid replaces it" is the wrong reading of that output on urwid 3.0.5.**
> `_posix_raw_display.Screen.start()` *captures* whatever handler was installed into
> `_prev_sigwinch_handler` (line 129), calls it after its own work (line 98), and restores it in
> `stop()` (line 142). urwid **wraps** our handler; it does not displace it. That is
> indistinguishable through `getsignal`, which is why the original inference looked safe. The
> handler had been running since Stage 0.
>
> **Following the fix direction below breaks the application.** Re-installing on top of a chain that
> already reaches us closes a cycle — ours → urwid's → ours (as urwid's `_prev`) → urwid's → … — and
> the first live `SIGWINCH` after doing so raised `RecursionError` about 500 frames deep, printing a
> traceback over the interface. Nine unit tests written against a hand-made double agreed the
> chaining was correct, because the double was built from this finding's assumption. §0b, item 1.
>
> **What shipped** asks the screen object whether we are already in its chain and installs only when
> we are not, which is the case on urwid builds without `_prev_sigwinch_handler`. Both branches are
> tested, including one that fails if the cycle returns. So the outcome of this finding is *safety on
> other urwid versions plus the first tests of any kind for it* — not a behaviour change here.
>
> **What the handler is for, when it is needed, is narrower than the finding reads.** `_art_geometry()`
> is re-derived from `get_cols_rows()` on every 0.5 s tick and `AlbumArtRenderer.render()` skips only
> when its key — `(path, x, y, width, height)` — is unchanged. So a **resize** changes the geometry,
> changes the key, and re-sends within half a second on its own. Only a **window move at unchanged
> size** needs `force_redraw()`: same key, send skipped, stale image.

`tui._setup_urwid()` installs `_on_sigwinch` during `__init__`. urwid's `MainLoop` /
`raw_display.Screen.start()` installs its own SIGWINCH handler afterwards, when `loop.run()` is
called. On urwid 3.0.5 that handler delegates back; on builds that do not, re-install after
`loop.run()` starts — but check which world you are in first, because doing it unconditionally is a
recursion, not a fix.

---

### L2 · Kitty and sixel album art fight urwid for the screen; only ueberzug works. `DONE — Stage 4`

> **Status.** Both classes deleted (D7: the honest answer was "delete this", and a branch that cannot
> work is not a feature with a caveat). `test_deletions.py` guards `KittyProtocol`, `SixelProtocol`
> and `img2sixel`. Detection is now ueberzugpp → ueberzug and stops there.
>
> What replaced them is the only part of those ~100 lines that was ever true:
> `_warn_about_unsupported_terminal()` tells a kitty or sixel user why there is no art and that
> `ueberzugpp` would give them some. Terminal detection for kitty is therefore still live and is
> deliberately *not* in the deleted-symbols list — what is gone is the pretence of drawing into it.

Both `KittyProtocol.render` and `SixelProtocol.render` write escape sequences straight to
`sys.__stdout__` while urwid owns the terminal. urwid's next full redraw — every 0.5 s — paints over
them. The ueberzug/ueberzugpp overlay protocols work because they draw in a separate X11/Wayland
surface. Detection order puts ueberzug first, so this mostly hides, but the kitty and sixel branches
are effectively non-functional and should be marked as such or removed. `album_art.py:234–325`

**Restoring them is a different design, not a repair**: it needs urwid to stop owning the screen for
the region the image occupies.

---

### L3 · Album-art geometry is hardcoded to the current layout. `ELEVATED to H8 — DONE Stage 3`

`_render_art()` pins the image at `x=2, y=3` with `width=33, height=10`. Those constants encode
"header is 1 row, LineBox border is 1 row, Divider is 1 row" and the exact row count of the right-hand
Pile (`RIGHT_COL_ROWS = 10`). Any change to the header, the console height, or the Now-Playing
contents silently misplaces the image, and on a short terminal the fixed 10-row art can overlap the
console panel. The comment block documents the arithmetic carefully but nothing enforces it.
`tui.py:563–602`, constants at `tui.py:113–131`

**Why elevated:** H1's rework changes the Now-Playing pile — the vibe line becomes a descriptor list,
and the deleted queue panel frees vertical space. `RIGHT_COL_ROWS = 10` is about to be wrong.

> **Fix direction.** Derive the offsets from the widget tree at render time rather than from constants,
> or at minimum compute every constant from one declared source (`RIGHT_COL_ROWS` should be
> `len(right_col.contents)`, not a hand-counted comment).

---

### L4 · Hearts vanish on restart even though likes are already on disk. `DONE — Stage 3`

> **Status.** `TUI.liked_tracks` is deleted. `SessionHistory.liked` is the one set, rehydrated from
> `feedback_history.json` when the TUI is constructed — which is after `persistence.load_all()`, so
> the file is loaded by then. It drives the `♥` in the Session panel, the `❤` on the Now Playing
> track line and the `❤` on the `↓ next:` line, all from the same place.
>
> The distinction the panel needed anyway: `♥` is persistent and spans sessions ("you like this
> track"), while `⏭` and `✓` are this session's outcomes. They occupy separate mark slots, so a track
> can show `♥⏭`. Tested in `test_session_history.py` and end to end in `test_tui_display.py`.

`TUI.liked_tracks` was an in-memory set, populated only by pressing `[L]` during the current run. Every
like is already recorded in `feedback_history.json` with a track path, so the set could be rehydrated
at startup in three lines. `tui.py:144–145`, `feedback_handler.py:160–171`

**Status.** The new session-history panel (H1) needs per-track feedback marks anyway, so rehydration
from `feedback_history.json` became part of building it rather than a separate fix.

---

### L5 · No log file, and stderr is swallowed while the TUI runs. `DONE — Stage 0, first item`

> **Status.** `_ConsoleCapture` tees every completed line to `data/dj.log` (append mode, per-session
> banner, `config.log_file`). Failure-tolerant in both directions: an unwritable path degrades to no
> log rather than aborting startup, and a write error drops the handle rather than recursing back
> through `stderr`. Five tests in `tests/test_console_log.py`, including the case that matters — a
> line written while `tui_active` reaches the log and does **not** reach the terminal.
>
> The 5-line console panel and the 200-line ring buffer are unchanged; the log is the durable copy.

`_ConsoleCapture` replaces `sys.stderr` at import and, once `tui_active` is set, stops forwarding to
the real terminal. It keeps the last 200 lines in a ring buffer that is never written to disk, and the
console panel shows only the last 5. A traceback from the background thread flashes past and is then
unrecoverable. This is the single biggest obstacle to diagnosing anything reported from a real session.
`tui.py:44–106`

**Why elevated.** The revision rewrites the queue manager, the selector's sampling, the embedding
pipeline and the TUI layout. Debugging that with a 5-line ring buffer and no log is a poor trade for
the ten minutes it takes to tee the buffer to `data/dj.log`. Do it **first**, before any other change.

---

### L6 · Polling architecture instead of `mpc idle`. `OPEN`

Steady state is roughly 12 `mpc` subprocess spawns per second — the background thread and the TUI each
poll at 2 Hz, and each poll issues 2–4 separate `mpc` invocations. Measured locally at 0.8 ms per
call, this is about 1% of one core: not a local performance problem, and worth stating plainly rather
than overselling. It does matter in two ways. Against a remote `MPD_HOST` every one of those is a TCP
round trip. And queue refill only reacts on a 2-second timer.

At depth 1 the refill latency is still comfortable — the one queued track provides minutes of runway,
not seconds — so this stays low priority. `main_tui.py:185–243`; `tui.py:487–560`

---

### L7 · Cold start injects a random direction at 30% weight. `DONE — Stage 0 + Stage 2 (one guard deliberately kept)`

> **Status.** `DONE — Stage 2.` β now ramps from 0 to its configured value over the first
> `config.taste_ramp_updates` (20) updates, with the unearned weight going to the session term. The
> ramp is applied *after* the exploration shift and its `max(0.1, …)` floors, which would otherwise
> stop the taste term reaching the zero it should hold with no evidence.
>
> `SessionState` no longer seeds from `randn` at all: it starts at zero, adopts the first track that
> actually plays, and `get_candidate_pool` guards **both** halves — so a fresh session's first pick is
> a uniform draw, which is the honest answer to "no information".
>
> **One deviation.** The plan says to retire the "negative updates are a no-op while unseeded" guard
> once the ramp lands. It was kept. β gates the *score*, never *retrieval*, and the pool opens its
> taste half on `np.any(taste_vector)` — so retiring it would let one skip hand half the candidate
> pool, at full strength, to "the tracks least like the one song you rejected". Reasoning in §0b.

`UserTaste._initialize_taste_vector()` returns a normalised `randn` vector. The taste term carries
β = 0.3 from the very first track, so a brand-new user's "long-term taste" is a random direction in
CLAP space with meaningful influence over selection until dozens of updates accumulate. With
`taste_update_full_listen = 0.02`, that is a long time. `user_taste.py:27–34`,
`session_state.py:34–40`

**The original audit's fix does not survive C5.** "Seed from the library centroid" is degenerate once
the space is centred — the centroid *is* the zero vector, which has no direction.

> **Fix direction.** Delete the seed rather than replace it. Start the taste vector at zero and **ramp
> β from 0 to its configured value** as updates accumulate (e.g. `β_eff = β · min(1, n_updates / 20)`),
> redistributing the freed weight to the session term. A new user is then driven purely by what they
> are listening to right now, which is correct — there is nothing else to know about them yet. This
> also removes one more invented constant (D4).
>
> The same argument applies to `SessionState._initialize_session_vector()`, which seeds the session
> from `randn` at every startup. Since `start_session()` can seed from the first track's embedding, the
> random path should only survive as the "no track playing yet" case — and there, the honest behaviour
> is to pick the first track uniformly at random rather than pretend a random vector is a vibe.

---

### L8 · Dead API surface across most modules. `DONE — Stage 0 (two exceptions deferred by design)`

> **Status.** All of the below deleted except `previous_track` and `save_embeddings`, which the plan
> says to keep for now. `ExplorationController.reset()` went too — it was not on the list only because
> `Persistence.reset_all()` called it, and that is gone. Taste-model inspection landed early as the
> `[I]` overlay (§0b, item 2) and was extended with the descriptor rows in Stage 3.
>
> **Un-like (`[L]` as a toggle) — `DONE — Stage 4`.** It is *not* a subtraction, and the reason is
> stronger than the asymmetry §8's trap 1 warns about. Measured on the real library, subtracting
> `taste_update_like` lands at cos 0.9999 from the truth in every ordinary case — and at **0.000** in
> the one that matters: from zero a single like normalises to the track, and subtracting 0.1·e from e
> gives 0.9·e, which normalises back to e. **A subtraction cannot un-seed a model**, so retracting
> your only like would leave the taste vector pinned at unit strength to the track you just rejected,
> which is the first retraction any new listener performs.
>
> So `process_unlike()` deletes the like from `feedback_history` and calls `UserTaste.replay()` over
> what remains — asserting nothing, restating the definition the model already has. The replay
> reproduces the incrementally built vector bit for bit (§10e).
>
> **Gated on `UserTaste.explains()`.** The replay is exact only if the history can account for the
> model, and `_record_feedback` caps it at 1000 events; one event past the cap a replay lands at cos
> 0.994 from the vector it replaced, and at 1400 events, 0.923. When the account is incomplete the
> retraction is display-only — the `♥` goes, the like leaves the history, the taste vector is left
> alone — and the console says which of the two happened. §8's second option, taken only where the
> first cannot be honest, and the README states which is which. Both branches were exercised live.

This line used to say Stage 3. It is not a row in §8's Stage 3 table and it is not a display change:
un-liking means changing the taste model, which is the player, and Stage 3's whole constraint was
that the player is closed. The display half was already there — `SessionHistory.liked` is one set,
rehydrated from disk (L4).

Never called anywhere in the running system: `MPDController.toggle` / `previous_track` / `seek` /
`get_all_tracks` / `get_track_metadata`, `TrackLibrary.has_track` / `save_embeddings`,
`SessionState.get_recent_average` / `get_similarity` / `reset`, `UserTaste.get_similarity` / `reset`,
`TrackSelector.clear_history`, `TimeContext.reset_period`, `Persistence.reset_all`.

Notably there is no key bound to `previous_track` and no way to un-like, reset taste, or inspect the
taste model from the UI.

**Status.** Delete them (D7), with three exceptions worth *wiring up* instead:

- **Taste model inspection** — the `[I]` rework (H1) provides it.
- **Un-like** — trivially useful; bind it to `[L]` on an already-liked track (toggle).
- **`previous_track`** — note that `consume on` (D2) makes MPD-side prev impossible. If wanted, it
  must re-add from the app's own history. Defer.

---

### L9 · Assorted small traps.

- **`select_track` mutates its caller's set.** `exclude_tracks.update(self.recent_history)` modified
  the set its caller passed in. — `DONE` (Stage 2): it copies first, and
  `tests/test_selection_sampling.py` passes a set in and requires it unchanged.
- **`config.validate()` is all `assert`.** Under `python -O` every check vanishes, including the
  weight-sum-to-1.0 invariant. — `DONE` (Stage 0): raises `ValueError` naming the offending key;
  `tests/test_config.py` asserts the exception type, so a regression to `assert` fails the suite.
- **`.gitignore` intent not achieved.** It excludes `data/state/` and `data/embeddings/` as
  directories, then tries to re-include `!data/state/.gitkeep` — which git cannot do inside an
  excluded directory. `git ls-files data` returns only `data/.gitkeep`. Harmless because
  `Config.__init__` mkdirs them, but the scaffolding does not ship. — `DONE` (Stage 4): the patterns
  are `data/state/*` and `data/embeddings/*`, so the directories stay visible to git and the
  negations underneath them work. `git ls-files data` now lists all three `.gitkeep` files, and
  `test_deletions.py::test_no_learned_state_or_embeddings_are_committed` still requires that they are
  the *only* thing under `data/` that is tracked.
- **Time-context bonus breaks the weight invariant.** `config.validate()` enforces that the four
  weights sum to 1.0, then `_calculate_score` adds `0.15·time_sim` on top. — `DONE` (Stage 0): the
  bonus is gone, the invariant is true again and is now enforced outside `assert`.
  `tests/test_scoring.py` recomputes the formula by hand and requires an exact match, so a fifth term
  cannot be added silently.
- **`[I]` silently does nothing when `enable_time_context` is false.** —
  `DONE` (Stage 0): `[I]` is a model inspector and always has content. Extended in Stage 3 (H1d).
- **Orphaned ueberzugpp process.** `_shutdown` calls `clear()` but never terminates the child; cleanup
  relies on `__del__` firing at interpreter exit, which is not guaranteed. `album_art.py:118–124` —
  `DONE` (Stage 4). `AlbumArtRenderer.shutdown()` clears *and* ends the child, and is called from
  `_shutdown()` **and** from the signal handler — which is the half that was actually broken, since
  `[Q]` always reached `_shutdown()` and a SIGTERM never did. `_terminate()` escalates to `kill()`
  when the child ignores `terminate()`, which the old `wait(timeout=1)` swallowed. The art cleanup
  runs *before* `request_exit()` and cannot raise, because an exception there would trade this
  finding for H3; a test asserts that the shutdown still reaches `request_exit()` when the renderer
  throws.
- **Keybinding docs disagree.** README lists volume as `,` / `.`; the footer reads `Vol - [<,>]`. In
  the non-urwid fallback mode, `↑↓` are bound to volume rather than queue navigation, contradicting
  both. — `DONE` (Stage 3): one list of bindings, in the footer, the README table and the fallback
  mode. `↑↓` is history everywhere and volume nowhere; volume is `,`/`.` everywhere (`<`/`>` still
  accepted, since they are the same physical keys shifted).
  `tests/test_tui_display.py::test_the_footer_advertises_exactly_the_keys_that_are_bound` drives every
  advertised key through the real `_handle_input` and requires a distinct action, so a footer entry
  cannot outlive its binding. The fallback mode's bindings were reconciled by reading only — there is
  no harness that drives a non-urwid terminal (§0b).

---

## 6 · Findings status

*As of Stage 4 complete, 23 July 2026. `Done` means the code is in the tree and a test guards it —
which now includes the MPD path, via the `FakeMPD` built to verified semantics (M1b), **the
display**, via tests that construct and render the real widget tree (Stage 3), and **the fallback
text interface**, via a pty (Stage 4). **Every finding is now closed.** Three are closed as
`Dissolved`, one — L1 — is closed with its own claim corrected, and L6 is closed as declined with a
reason rather than left open.*

| ID | Finding | Status | Stage |
|---|---|---|---|
| **C1** | Queue never refills | **Done** — depth 1 + `consume on`; 30-track live run, 0 stalls, depth 2 in 30/30 samples | ~~2~~ |
| **C2** | MPD random mode discards ordering | **Done** — forced, logged, restored on `_shutdown` / signal / `atexit`; modes carried as raw strings | ~~2~~ |
| **C3** | Non-deterministic 10 s crop embeddings | **Done** — full coverage, bit-reproducible, window matrix persisted, failures listed. One stated cause corrected (§0b) | ~~1~~ |
| **C4** | `[V]` double-advance | **Dissolved** by D8 — code path deleted. Constraint retained and asserted on the call log of the real skip method. Its paused-case reasoning was corrected against the live MPD | ~~2~~ |
| **C5** | Compressed similarity scale (anisotropy) | **Done** — centroid stored and applied on load; random pairs 0.670 → 0.011 | ~~1~~ |
| **H1** | Mood word pinned to "eclectic" | **Done** — heuristic deleted (Stage 0), 49-descriptor bank built and gated (Stage 1), readout / Session panel / `[I]` shipped (Stage 3). The consistency figure shipped as a word count, not the specified cosine — the cosine was measured and found compressed (§10d) | ~~0~~ / ~~1~~ / ~~3~~ |
| **H2** | Queue panel shows history as future | **Done** — panel, bindings and `get_upcoming_tracks()` removed in Stage 2; Session panel shows the same tracks *as history* in Stage 3, on indices derived from this session's plays | ~~2~~ / ~~3~~ |
| **H3** | Ctrl-C/SIGTERM neither exit nor save | **Done** — self-pipe unblocks urwid, `atexit` covers the mode restore, state checkpointed every N tracks | ~~2~~ |
| **H4** | Skipping doesn't change what plays next | **Dissolved** by D1 — one lookahead track, dropped and re-picked. `consecutive_skips` became load-bearing rather than unused | ~~2~~ |
| **H5** | Day-of-week modifier is dead code | **Done** — deleted (D6) | ~~0~~ |
| **H6** | Strictly greedy selection | **Done** — rank-Boltzmann; the shipped τ map reproduces the published p(rank 0) table, and 40k draws are checked against `exp(−i/τ)`. τ_max = 15 still uncalibrated by listening | ~~2~~ |
| **H7** | Eight duplicate methods; `add_track` swallows failures | **Done** — duplicates removed, return code checked, `ast` test guards it | ~~0~~ |
| **H8** | Album-art geometry hardcoded | **Done** — derived from the widget tree; two of the four constants were already wrong (`x`, and the height Stage 3 changes). Asserted against a real render at seven terminal sizes | ~~3~~ |
| **N1** | `WidgetError` on any terminal under 33 rows | **Done** — `('pack', …)` instead of `('weight', 3, …)`. Predates the audit; found while doing H8, reproduced live at 80×24 against the pre-Stage-3 tree | ~~3~~ |
| **H9** | Neither `[V]` nor `[N]` changes direction; `[V]` aims off-manifold | **Done** — `[V]` deleted, `[N]` escalates on a solved λ. The design needed one correction found only by running it: the snap must be *inside* the solve (§0b) | ~~2~~ |
| **M1a** | Tests untracked; phase files are theatre | **Done** — `.gitignore` fixed, phase files deleted, `tests/` tracked (67 green) | ~~0~~ |
| **M1b/c** | No behavioural suite, no `FakeMPD` | **Done** — `FakeMPD` built to re-verified semantics and itself under test (2); the display's own coverage, where N1 had been hiding (3); persistence round-trips and the fallback mode under a pty (4). 67 → 542. Stage 4 also found the suite writing to the developer's live `data/state/` | ~~2~~ / ~~3~~ / ~~4~~ |
| **M2** | Two divergent orchestrators | **Done** — both deleted, plus the dummy-embedding paths | ~~0~~ |
| **M3** | `mpd_music_directory` unvalidated | **Done** — read from MPD's config, source reported, probes resolved; fatal for generation, a warning at startup | ~~1~~ |
| **M4** | Two sources of truth for track keys | **Done** — `mpc listall` only; coverage logged and enforced on load | ~~1~~ |
| **M5** | No embedding-dimension validation | **Done** — full schema validation, file dimension wins, model checked by equality | ~~1~~ |
| **M6a** | `time_context.npz` is JSON | **Done** — deleted (D6) | ~~0~~ |
| **M6b** | `play_history` never persisted | **Done** — `play_history.json` carries `play_history`, `current_index` and `recent_history`, checkpointed with the rest | ~~2~~ |
| **M7** | Contradictory setup docs; macOS unsupported | **Done** — cache re-measured at 1.15 GiB and the reason it is twice the model's size established; the pre-flight check derived from it rather than chosen; the missing CPU figure measured at 17.0 windows/s; the macOS claim dropped with its reasons. A test holds the three places together, since the finding is that they drifted | ~~2~~ / ~~3~~ / ~~4~~ |
| **M8** | `--batch-size` ignored | **Done** — batching plus a decode/mel worker pool, which is where the cost actually was (§0b) | ~~1~~ |
| **L1** | SIGWINCH handler never invoked | **Done — and the finding was wrong.** urwid 3.0.5 *chains* to the previous handler rather than replacing it, so it had been running since Stage 0; acting on the fix direction closed a cycle and raised `RecursionError` on the first live resize. What shipped is the check that keeps it safe on urwid builds that do not chain, plus the first tests for it | ~~4~~ |
| **L2** | Kitty/sixel art fights urwid | **Done** — both classes deleted (D7) and guarded; a kitty or sixel user is told why there is no art and what to install | ~~4~~ |
| **L3** | Album-art geometry hardcoded | **Elevated → H8**, done | ~~3~~ |
| **L4** | Hearts vanish on restart | **Done** — `SessionHistory.liked` rehydrates from `feedback_history.json`; `TUI.liked_tracks` deleted | ~~3~~ |
| **L5** | No log file; stderr swallowed | **Done** — teed to `data/dj.log`, 5 tests | ~~0~~ |
| **L6** | Polling instead of `mpc idle` | **Declined** — optional in §8, and still only worth it against a remote `MPD_HOST`. At depth 1 the one queued track gives the 2 s poll minutes of runway, and Stage 4's brief is to change observable behaviour as little as possible | — |
| **L7** | Random cold-start taste vector at β=0.3 | **Done** — β ramps over 20 updates; session vector starts at zero and seeds from the first real track. One guard deliberately *kept* against the plan (§0b) | ~~0~~ / ~~2~~ |
| **L8** | Dead API surface | **Done** — deleted except the two the plan defers (0). Un-like shipped in Stage 4 as a *replay* of the feedback history rather than a negative update, gated on the history being able to account for the model — the measurement that decided it is that a subtraction cannot un-seed a model, so retracting your only like would leave the taste vector pinned to it | ~~0~~ / ~~4~~ |
| **L9** | Assorted small traps | **Done** — `validate()` off `assert`, the weight invariant, the dead `[I]`, `select_track`'s set mutation and the keybinding docs (0–3); the `.gitkeep` scaffolding now ships and the ueberzugpp child is terminated on both exit paths (4) | ~~0~~ / ~~2~~ / ~~3~~ / ~~4~~ |

---

## 7 · Target data artifacts

Build to these. Both are produced by the generation run in Stage 1 and are the contract everything
downstream reads.

> **Built, 23 July 2026.** Both exist and match the tables below.
> `embeddings_io.validate_embeddings()` enforces every row, `TrackLibrary` refuses to load a file that
> fails it, and `tests/test_embeddings_io.py` breaks each field in turn to prove the check is real.

### `data/embeddings/track_embeddings.npz`

| Key | Shape / type | Notes |
|---|---|---|
| `schema_version` | `int` | Start at `2`. Refuse to load `< 2` — a v1 file has no centroid and would be scored on an uncentred space (C5). |
| `track_files` | `(N,)` unicode | Keys as returned by `mpc listall` — the single enumeration source (M4). |
| `embeddings` | `(N, 512)` float32 | Pooled, **uncentred**, L2-normalised. Keep raw so the centroid can be recomputed if the library grows. |
| `centroid` | `(512,)` float32 | `mean(embeddings, axis=0)`. Applied at load: `normalise(E − centroid)` (C5). |
| `window_offsets` | `(N+1,)` int32 | CSR-style index into `windows` — track *i* owns `windows[offsets[i]:offsets[i+1]]`. Tracks have different window counts. |
| `windows` | `(ΣW, 512)` float32 | Per-window embeddings, L2-normalised. 24,494 rows for 674 tracks; 45.5 MB for the whole artifact. Lets pooling be re-decided without regenerating (C3). |
| `metadata` | JSON string | Model name, transformers/torch versions, date, device, window scheme (`length_s`, `hop_s`, `tail`, `rms_gate`, `truncation`, `pooling`, `batch_size`), timing, window-RMS percentiles. Must survive round-trips (M5). **JSON rather than a dict**: a dict in an `.npz` is a pickled object array and forces `allow_pickle=True` on every read. `truncation` is recorded because the mode is load-bearing (C3) and `batch_size` because bit-reproduction depends on it (M8). |

Also written: `data/embeddings/failed.txt` — one line per file that failed, with the exception. The
original run lost 16 tracks silently (C3); they turn out to be one album of corrupt FLACs.

### `data/embeddings/descriptors.npz`

Generated after the embeddings, in the same run — the z-score baselines are measured against the
centred library, so building it separately invites the two to drift apart. 93 KB.

| Key | Shape / type | Notes |
|---|---|---|
| `schema_version` | `int` | |
| `labels` | `(D,)` unicode | The descriptor words, post-validation — near-zero-variance ones already dropped (H1). Built: 49 of 49 survived. |
| `prompts` | `(D,)` unicode | The full rendered prompt (`"This is a recording of {} music."`), kept so the template is auditable. |
| `text_embeddings` | `(D, 512)` float32 | CLAP text-tower output, L2-normalised. |
| `mean` | `(D,)` float32 | Per-descriptor mean similarity over the centred library — the z-score baseline (H1). |
| `std` | `(D,)` float32 | Per-descriptor std. The validation gate drops any descriptor below a floor here. |

**Why `mean`/`std` are stored rather than computed at startup:** they are the fix for the modality
gap, and computing them lazily invites someone to skip them "just this once" and ship raw dot
products. Baking them into the artifact makes the correct path the only path.

**No `anchors` key.** An earlier draft carried per-descriptor audio-space anchors so a redesigned `[V]`
could steer toward a named direction. `[V]` is deleted (D8/H9) and nothing else needs them. They are
one line to derive from `text_embeddings` + `embeddings` if free-text steering is ever built (§9).
Do not add them speculatively.

---

## 8 · Suggested order of work

Restructured around the revision. Two governing changes from the original ordering:

- **Deletions come first.** Roughly 15% of the codebase is being removed; there is no reason to carry
  it through a refactor.
- **The embedding rebuild moves from Stage 2 to Stage 1.** The old embeddings are being discarded
  anyway (D3), and every tuning decision downstream — softmax temperature, weights, exploration
  bounds, descriptor thresholds — is meaningless until the vector space is trustworthy. The descriptor
  bank builds here too: it is a data artifact derived from the embeddings, not a UI feature, and
  building it early retires the risk that CLAP descriptors do not discriminate this library at all.

**Every stage leaves the application runnable.** After Stage 0 it runs with a blank vibe line. After
Stage 1 it runs with honest embeddings but still stalls at ten tracks. After Stage 2 it genuinely
works. Stages 3–4 are polish and durability.

> **One caveat discovered in Stage 0:** "runnable" is not literally true *between* Stage 0 and
> Stage 1, because D3 deletes the embeddings Stage 1 regenerates. That is the one gap in the sequence,
> and it is intentional — the alternative is regenerating twice with a generator C3 is about to
> replace. **Stage 1 is not optional homework; the app does not play until it lands.**
>
> *Closed 23 July 2026. The gap lasted one working session; the embeddings exist and the app starts.*

---

### Stage 0 — Clear the ground *(an hour, mostly mechanical)* · ✅ **COMPLETE 22 July 2026**

*Every row landed. Full account, including four things this table did not anticipate, in §0b.*

| ID | Change | Why first | |
|---|---|---|---|
| **L5** | Tee the console ring buffer to `data/dj.log` | Ten minutes, and everything after this is a refactor you will need to debug through a 5-line panel otherwise. | ✅ |
| **D6 / H5 / M6a** | Delete `time_context.py`, `enable_time_context`, `time_context_weight`, the weekday/weekend modifiers, `get_exploration_factor()`, `time_context.npz` | Removes four findings and restores the weight-sum-to-1.0 invariant. | ✅ |
| **D4** | Delete `get_vibe_description()`'s entropy mood word, momentum thresholds and stage word; delete the random taste seed | Nothing after this should be built on them. Leave the vibe line blank until Stage 3 — a blank line is honest, the current string is not. | ✅ *(shows the track count, per D4's own "show the counter")* |
| **M2 / D7** | Delete `main.py`, `setup_check.py`, and `start.sh`'s demo-embedding path | Stops every later fix from needing to be applied twice. | ✅ |
| **H7** | Delete the eight duplicate methods in `mpd_controller.py`; restore the `add_track` return-code check | Makes MPD failures visible — load-bearing for Stage 2, where a swallowed `add_track` means silent stall. | ✅ |
| **L8** | Delete the dead API surface | Smaller surface to refactor. Keep `previous_track` and `save_embeddings` for now — see L8's exceptions. | ✅ |
| **M1a** | Remove `test_*.py` from `.gitignore`; delete the three phase test files | They test a structure being replaced. Green currently means nothing; delete rather than repair. | ✅ |
| **cfg** | Prune `config.py` of every key the above orphans; convert `validate()` off bare `assert` (L9) | Dead config keys are how H5 hid for months. | ✅ |
| **D3** | Delete `data/embeddings/*` and `data/state/*` | No value to preserve; removes the "reset state carefully" caveat from every later step. | ✅ |

**Done when:** ~~the app launches, plays,~~ and shows an empty vibe line; `git ls-files` includes a
tests directory; `grep -ri time_context` returns nothing; `data/dj.log` fills during a session.

> **Correction to this criterion.** "The app launches, plays" is unreachable once D3 deletes the
> embeddings — there is nothing to select from until Stage 1 regenerates. It was verified with a
> throwaway random-vector file created outside the repo, used for one scripted end-to-end session
> against the live MPD, and deleted (§0b). **For Stage 1 the criterion is "launches and reports a
> clean library load"; the playback criterion belongs to Stage 2's 30-track unattended run.**

**Actual result:** −1,462 lines of application code, +734 lines of tests. `git ls-files tests` lists
9 files; 67 tests pass. `grep -ril time_context` matches only this document, `CLAUDE.md` and the
tests that assert the deletion.

---

### Stage 1 — Rebuild the signal *(half a day plus a regeneration run)* · ✅ **COMPLETE 23 July 2026**

Nothing about selection quality can be judged until the vectors mean something. Build to the schemas
in §7.

*Every row landed. Full account, including five things this table did not anticipate and two
corrections to the audit's own claims, in §0b.*

| ID | Change | Notes | |
|---|---|---|---|
| **M3** | Detect `mpd_music_directory` from MPD's config files; prompt in `start.sh` as fallback; refuse to run generation if known track paths will not resolve | **Before** the regeneration run — a wrong music dir wastes the whole thing. | ✅ *(fatal for generation, a warning at startup — see §0b)* |
| **M4** | `mpc listall` as the single enumeration source; log embedding↔MPD coverage on load; refuse below a threshold | Verifies the regenerated file matches what MPD will actually play. | ✅ *(100% coverage on the rebuilt library)* |
| **C3** | Full-coverage deterministic 10 s windows, end-aligned tail, RMS-gated, mean-pooled; persist the per-window matrix and the failed-track list | The change everything else sits on. | ✅ *(24,494 windows; bit-identical re-embedding)* |
| **M8** | Batch the windows, and thread the mel extraction | Prerequisite, not an optimisation — ~36× the forward passes, and the feature extractor is the real bottleneck. | ✅ *(batches filled per track, for reproducibility — see M8)* |
| **C5** | Compute and store the library centroid; centre + re-normalise on load | Do it *with* C3 so the `.npz` carries the centroid from day one. | ✅ *(0.670 → 0.011)* |
| **D5a** | Build the descriptor bank: ~50 prompts, text-tower embeddings, per-descriptor `mean`/`std` over the centred library | Data artifact, not UI. Both H1 and H9 consume it. | ✅ *(49 descriptors, template chosen by measurement)* |
| **D5b** | **Validation gate** — drop any descriptor whose `std` over the library is below a floor; log what was dropped and what survived | **This is the anti-trap step.** See below. | ✅ *(floor = 0.5 × the library's median std; nothing dropped — §0b)* |
| **M5** | Validate `schema_version`, dimension and required keys on load; adopt the file's dimension over the config's | Guards the new schema against silent mismatch. | ✅ |

#### The descriptor-bank trap, and the gate that catches it

CLAP's audio and text towers are contrastively aligned but do not share a cone — the **modality gap**.
Raw `audio · text` dot products are not comparable across descriptors: one word may score 0.31 against
your entire library while another scores 0.08 against all of it. A naive top-3 then returns the same
three words forever, which is H1's failure repeated in a new costume.

Two defences, both mandatory:

1. **Per-descriptor z-scoring.** `z(d, v) = (sim(d, v) − mean_d) / std_d`, with `mean_d` / `std_d`
   computed over the centred library and **stored in the artifact** (§7). This makes "hypnotic" mean
   *"unusually hypnotic for this library"* — the correct semantics, and immune to the gap.
2. **Variance gate at build time.** A descriptor whose `std_d` is near zero does not discriminate
   anything in *your* collection — it is a word that describes all your music equally, so it can never
   be informative. Drop it, and log both lists.

**Acceptance for the gate:** print the surviving descriptors with their `std`, and the dropped ones
with theirs. Then spot-check five tracks you know well against their top-3 descriptors. If the words
are wrong, the problem is the prompt template, not the method — try two or three templates and keep
the one with the best library separation. Do this before wiring anything to the display; debugging a
bad bank through the TUI is far harder than through a script.

**Acceptance for the stage:** embed the same file twice, assert **bit-identical** vectors. Re-run the
audit's measurement: same-track self-similarity must be exactly 1.0, and post-centring the random-pair
distribution must sit near 0 rather than the 0.577 measured before the rebuild. All three are one-line
assertions and all three belong in the test suite.

> **Met, 23 July 2026.** All three are in `tests/test_clap_pipeline.py`, which skips rather than
> downloads if the checkpoint is not already cached:
>
> | Criterion | Result |
> |---|---|
> | same file embedded twice | bit-identical (pooled *and* per-window) |
> | same-track self-similarity | 1.0 to float precision |
> | post-centring random pairs | **+0.011** (was +0.670 raw) |
>
> With one qualification the plan did not state: bit-identity holds for a fixed `--batch-size`. A
> different batch size agrees to ~2 × 10⁻⁸, because GPU reductions depend on batch composition. Both
> facts are asserted, and `batch_size` is recorded in the artifact (§0b, item 1).
>
> **Acceptance for the gate** — the descriptor spot check — is in §10 and passed: the words are
> recognisable on tracks whose character is not in dispute. The gate itself dropped nothing, so it
> rests on unit tests rather than on the library (§0b, item 2).
>
> **Stage-level criterion** ("launches and reports a clean library load"): verified in a pty against
> the live MPD, exit 0, `674 embeddings … centred`, `100.0%` coverage, `49 descriptors`. The user's
> MPD queue, modes and volume were snapshotted and restored.

---

### Stage 2 — Play continuously, one track ahead *(a few hours)* · ✅ **COMPLETE 23 July 2026**

*Every row landed. Full account, including four things this table did not anticipate and one design correction that only a live run exposed, in §0b.*

> **Historical.** This block briefed Stage 2 before it was written and is kept as provenance. Two of
> its warnings proved decisive — H6 really was the blocker it says, and τ_max = 15 really is still
> uncalibrated. One number in it is wrong: `[N]`×1 does not turn over 0.3% of the pool, it turns over
> about 2.9% on a settled session vector. See §10c.

> **Read this before writing anything.** Stage 1 changed the ground these items stand on, and the
> shortest path through Stage 2 is knowing what is now settled and what is now different.
>
> **Settled — do not re-derive, do not re-tune.** The embeddings are final for this library: 674
> tracks, deterministic, `data/embeddings/track_embeddings.npz`, regenerated with
> `python3 generate_embeddings.py --force` in ~5.5 minutes if you ever need to. `TrackLibrary` centres
> them on load and validates the schema; every vector you touch is already in the centred space, and
> nothing should centre anything a second time. `mpc listall` is the keyspace, reconciled at startup.
> The descriptor bank loads as `self.descriptor_bank` on the orchestrator and is unused until Stage 3
> — leave it alone.
>
> **Different — the numbers moved, and by a lot.** The centred space is far more concentrated than the
> one the original measurements were taken in: the session vector now sits at cosine **0.971** to the
> track it is about to play (was 0.790), so a `[N]` press turns over **0.3%** of the candidate pool
> (was 3.9%). Two consequences for the work below. **H6 is more of a blocker than it looked** — argmax
> over a pool this tight is effectively deterministic. And **the fixed λ = 0.15 nudge is now
> negligible**, which strengthens H9's case for solving λ rather than declaring it. Reference figures
> in §10b; §10 is the pre-Stage-1 record and its H6/H9 tables should not be used as targets.
>
> **Still true, and worth knowing you can rely on it.** Both designs survived re-measurement unchanged
> because they target observable quantities — pool turnover, candidate rank — rather than vector-space
> magnitudes. If you find yourself reaching for a constant expressed in cosines, that is the signal to
> stop and express it as turnover or rank instead.
>
> **The one number you must not trust from anywhere:** τ_max ≈ 15. It is the only genuinely new
> constant in the plan and it has never been calibrated by listening. Ship it, listen, adjust.
>
> **Tests.** 179 pass and none of them touch MPD. `tests/conftest.py` has the fixtures (`rng`,
> `library`, `make_artifact`); M1b's `FakeMPD` belongs beside them and is the first item below.

D1 and H6 are one change and must land together. Write the test harness **first** — this is the stage
where the original audit's four critical defects lived, all under a green suite.

| ID | Change | Notes |
|---|---|---|
| **M1b** | `FakeMPD` modelling real semantics **including consume mode**, on the pytest scaffold Stage 0 created | First, not last. Every item below gets a behavioural test as it lands. **The semantics are specified and verified against the live MPD in M1** — build to that table, not to intuition; a `FakeMPD` built on the assumptions that produced C1 would reproduce C1 and pass. |
| **C2** | Force `random`/`repeat`/`single` off and `consume` **on**; log each change to the console panel; restore originals on exit | Enables the simplified C1. |
| **H3** | Signal handling that exits, saves, and restores MPD modes; `atexit` hook for the mode restore; periodic state checkpoint | Must land **with** C2 — otherwise an abnormal exit strands the user's MPD in consume mode. |
| **C1 / D1** | `lookahead = 1`; rewrite `QueueManager` to `ensure_one_ahead()` / `replace_next()` | Delete `planned_queue`, `currently_queued_in_mpd`, `_sync_to_mpd`, `get_upcoming_tracks`, `recalculate`, `initialize_queue`, the 5% trajectory blend, and `queue_low_threshold`. |
| **H6** | Boltzmann sampling over **rank**: `p(i) ∝ exp(−i/τ)`, τ linear in the exploration scalar | **Blocker for D1.** Rank-based, not score-based — scale-invariant, so it survives C5 and every weight change without recalibration. |
| **C4** | Delete `recalculate()`; one skip path: *adjust vector → replace lookahead → advance once* | No `play()` call anywhere in it, so the double-advance cannot recur by construction. **Order is load-bearing: add the replacement before advancing.** Verified — `mpc next` off the last track empties the queue and stops, and a later `add` will not restart it (M1). |
| **H9 / D8** | Delete `[V]`, `force_shift()`, `process_vibe_skip()`, `set_high_exploration()`, `vibe_shift_magnitude`. Rebuild `[N]`: repel from the skip-run centroid, λ **solved** for an escalating turnover target (5/20/50/85% by run length), then `snap()` to the 25-NN centroid for n≥2 | Measured: `[N]` turns over 0.3% of the pool per press and `[V]` lands a third of the way into noise. The `snap()` guard makes leaving the manifold structurally impossible — but it is a move in its own right, so **n≥2 is a requirement, not a nicety** (§10b). |
| **H4-repl** | `[N]` drops and re-picks the lookahead under the new weights | Falls out of D1 — skips become audible on the very next song. |
| **L7** | Ramp β from 0 over the first ~20 taste updates; make the random session seed the explicit "nothing playing yet" case | Same theme as C5: stop injecting noise as if it were signal. The taste seed itself was zeroed in Stage 0; when the ramp lands, retire the "negative updates are a no-op while unseeded" guard but **keep the candidate-pool guard** — β gates scoring, not retrieval (§0b). |
| **M6b** | Persist `play_history` / `current_index`; checkpoint with the rest of the state | Anti-repetition finally survives a restart. |

**Done when:** a 30+ track unattended run completes with no stall and no repeat inside the replay gap.
Two runs from the same starting state produce materially different track sets (argmax produces
identical ones — see H6). One `[N]` visibly changes the next track; four consecutive `[N]` presses
audibly change the *kind* of music, and the reported turnover exceeds 80%. `mpc status` after a
`kill -TERM` shows the user's original modes restored.

> **Met, 23 July 2026**, in a pty against the live MPD with the user's queue, modes and volume
> snapshotted and restored:
>
> | Criterion | Result |
> |---|---|
> | 30+ tracks, no stall | **36 distinct tracks, 0 stalls** |
> | no repeat inside the replay gap | **0 violations** |
> | queue holds one ahead | **depth 2 in 30 of 30 mid-track samples** |
> | different track sets from one state | 5/5 distinct runs, ≤ 95% overlap (unit-tested) |
> | one `[N]` changes the next track | yes |
> | four `[N]` exceed 80% turnover | **5% → 20% → 70% → 100%**, each meeting its target |
> | …and change the *kind* of music | Björk / Arcane OST → Watain |
> | modes restored after `kill -TERM` | byte-identical |
>
> 310 pytest tests pass, and `data/dj.log` from the run contains no error, no refusal and no dry-queue
> recovery.

---

### Stage 3 — Make it legible *(half a day)* · ✅ **COMPLETE 23 July 2026**

*Every row landed. Full account, including three things this table did not anticipate — one of them a
crash that predates the audit — in §0b.*

The stage where the queue's original purpose finally gets served. All display; the data it reads was
built in Stage 1 and had been sitting unused since.

> **Read this before writing anything.** *(Kept as written. Every warning below was live at the time
> and four of the five were load-bearing; the annotations record how each turned out.)*
>
> **Settled — do not re-derive.** The player works and is not in scope. The embeddings, the centring
> and the 49-word descriptor bank are final; the bank already loads as `self.descriptor_bank` on the
> orchestrator and `DescriptorBank` will z-score a vector against it and return the top *n* — that is
> the entire API H1b and H1d need. Selection, the queue, the skip path and MPD mode handling are all
> closed and tested; if a display change requires touching `queue_manager.py`, `track_selector.py`,
> `session_state.py` or `manifold.py`, stop and check why.
>
> **Do H8 first, and actually first.** The album art is pinned to hand-counted constants
> (`RIGHT_COL_ROWS = 10`, `x=2`, `y=3`) that encode the exact current row layout. Every other item in
> this stage moves that layout. Stage 2 deliberately left the panel geometry untouched for this
> reason, so the constants are still correct *right now* — which is the last moment they will be.
>
> **What Stage 2 left you.** The `Up Next` panel is a one-line `↓ next:` readout where the queue list
> used to be; replace it with the session panel (H1c) rather than adding beside it. `↑↓` and `ENTER`
> are unbound and absent from the footer, so H1d is rebinding free keys, not repurposing live ones —
> and **re-derive the history indices from scratch**, since the old ones counted from the top of the
> MPD playlist and made `ENTER` replay the session's first track (H2).
>
> **`[I]` already exists and already reports the Stage 2 machinery** — τ and "choosing from ~top N",
> the drawn rank, β earned, the next skip's turnover target. H1d *adds* the session and taste
> descriptors to it. Note that the overlay is now sized to its content and will need to **scroll**
> once those rows land; it is already close to a 40-row terminal's height.
>
> **Four traps, three of which Stage 2 created.**
>
> 1. **A zero session vector produces a confident-looking readout.** Nothing has played yet is now a
>    real state (L7), and `bank.top(zeros)` answers `shimmering · orchestral · serene` rather than
>    refusing. Gate the vibe line on `SessionState.is_seeded()`. See H1's status block; there is a
>    test pinning the hazard.
> 2. **The history panel cannot get its tags the way the queue panel did.** `get_playlist_metadata()`
>    is built from `mpc playlist`, and under `consume on` a played track is *gone from the playlist* —
>    so the one existing metadata path covers exactly the tracks the history panel does not show. Use
>    `MPDController._fetch_track_tags()`, which is per-track, cached, and falls back mutagen → `mpc
>    search` → filename; promote it to public rather than reaching into a private.
> 3. **Nothing stores what H1's consistency word compares against.** "Cosine between the session's
>    descriptor z-vector now and five tracks ago" needs a rolling store of past z-vectors, and none
>    exists — `SessionState.recent_tracks` holds the last five *track embeddings*, not session-vector
>    z-vectors. Either add the store or derive the word from `recent_tracks` and say which.
> 4. **The consistency word is the only item in this stage that invents a threshold.** Calibrate it
>    against observed drift on a real session, or ship the cosine as a number, or leave it out. Do not
>    pick 0.85/0.7/0.5 the way the deleted momentum words did (D4).
>
> **You are starting with no coverage in your own area.** 311 tests pass and not one constructs the
> widget tree — the suite covers everything *behind* the display. That is not an argument for leaving
> it that way: H8's geometry is arithmetic and testable, and the readout's gating condition above is
> exactly the kind of thing that ships broken under a green suite (H1, C1 and C4 all did).
>
> *This one was the most valuable warning in the section, and it understated the situation. The first
> test that rendered the tree found a `WidgetError` on every terminal under 33 rows — not a Stage 3
> risk but a defect that had already shipped through three stages (**N1**). Trap 4 was also right
> that the consistency word was the item at risk, though not in the way it expected: the specified
> cosine turned out to be the compressed quantity (§10d). Traps 1, 2 and 3 all landed as described.*

| ID | Change | Notes | Landed |
|---|---|---|---|
| **H8 / L3** | Derive album-art geometry from the widget tree rather than hand-counted constants | **First** — the layout changes below, and the art is pinned to `RIGHT_COL_ROWS = 10`. | ✅ `_art_geometry()`; two of the four constants were already wrong |
| **H1b** | Vibe readout = top-3 descriptors by z-score; consistency word = cosine between the session's descriptor z-vector now and five tracks ago | Replaces the deleted heuristics with something measured against a real distribution. | ✅ `vibe_readout.py`; the consistency figure ships as a **word count**, the cosine having measured p10 = 0.95 |
| **H1c** | Replace the queue panel with a **Session** panel: `↓ next:` line, divider, history newest-first with `♥` / `⏭` / `✓` marks | The actual answer to "what is being played". | ✅ `session_history.py`, plus `♪` for the track playing now |
| **L4** | Rehydrate `liked_tracks` from `feedback_history.json` at startup | Falls out of building the history panel. | ✅ `SessionHistory.liked`; `TUI.liked_tracks` deleted |
| **H1d** | Rebind `↑↓` to scroll history and `ENTER` to requeue a history entry as `next`; **extend** `[I]` | `[I]` already exists as a model inspector (Stage 0, §0b item 2) showing library size, taste/exploration/selector counters and the live weights. Add: session **and** taste top descriptors, and the effective τ ("choosing from ~top 7"). | ✅ plus `QueueManager.requeue_next()` and a scrolling `[I]`; τ was already there from Stage 2 |
| **L9** | Reconcile the footer, the README keybinding table, and the fallback-mode bindings | They already disagree three ways; do it while the bindings are open. | ✅ one list, with a test that drives every advertised key |
| **N1** | *(not planned)* `('pack', …)` for the Now Playing box | Found by the first test that rendered the tree. | ✅ renders down to 80×6 |

**Done when:** the vibe line names three descriptors that a listener would recognise as accurate, and
`[I]` explains what the machine currently believes without any invented vocabulary.

> **Met.** Live at 120×45 and 80×24: Fleshgod Apocalypse's *King* read
> `piano-led · halftime · triumphant`, and §10b's seven spot-checks reproduce exactly under the
> shipped readout. `[I]` reports the bank size, both vectors' descriptors (each gated on its own
> seeding), the drift count *and* the drift cosine with its measured distribution beside it, on top of
> Stage 2's τ, rank, β and turnover rows — every one a number the system computed. 416 tests green.

---

### Stage 4 — Make it durable *(half a day)* · ✅ **COMPLETE 23 July 2026**

> **Read this before writing anything.**
>
> **Settled — do not re-derive.** The player closed in Stage 2 and the display in Stage 3. The
> embeddings, the centring, the 49-word bank, selection, the queue, the skip path, MPD mode handling,
> the descriptor readout, the Session panel and the album-art geometry are all done and tested. Stage 4
> is durability and documentation: **it should change observable behaviour as little as possible.**
> Exactly one item below (L8) adds a feature, and it is the one that needs a decision rather than an
> edit.
>
> **What Stage 3 left you.** 416 tests, green, ~17 s. The display is under test for the first time —
> `test_art_geometry.py`, `test_vibe_readout.py`, `test_session_history.py`, `test_tui_display.py` —
> and the fixtures for it are `fake_art`, `stub_bank`, `dj_stub` and `tui` in `conftest.py`. Use `tui`
> rather than constructing `AdaptiveDJTUI` directly: it injects the art renderer, so building the tree
> does not detect image protocols and leave a `ueberzugpp` child behind, and it restores the SIGWINCH
> handler afterwards.
>
> **One structural fact Stage 3 introduced.** The display owns state that only the display's own tick
> maintains — `SessionHistory` (what played, the marks, the cursor) and `VibeReadout`'s z-vector
> store. That is the right place for both (§9), but it means **any code path that blocks the urwid
> loop stops observing the session.** `[I]` is currently the only one, and it handles this with
> `screen.set_input_timeouts(max_wait=0.5)` plus a call to `_sync_session_state()` on every timeout
> wake — without which a track that starts *and* finishes while the overlay is open never reaches the
> history panel (§0b, item 2b). If Stage 4 adds a modal — a confirm prompt, a help overlay — it must
> do the same thing.
>
> **Five traps.**
>
> 1. **L8 is the only item in this stage that risks inventing a constant.** "Un-like" is not a
>    keybinding problem — the display half is already done, `SessionHistory.liked` is one set
>    rehydrated from disk. It is a *modelling* question: what does `UserTaste` do with a retraction?
>    `taste_update_like` is +0.1, so the obvious answer is to apply −0.1 — but the update is a
>    normalised EMA, not an accumulator, so subtracting the same magnitude does **not** return the
>    vector to where it was, and the error depends on how many updates have happened since. Do not
>    pick a number and call it symmetry. Either derive the retraction from the recorded like (the
>    embedding is in `feedback_history.json`), or make `[L]` a display-only toggle that removes the
>    heart and the history entry without touching the taste vector, and **say in the README which one
>    it is**. D4's rule applies: a constant may stay if it shapes behaviour without asserting a fact.
> 2. **M1c's "persistence round-trips" is narrower than it sounds, and one gap matters more than the
>    others.** `play_history.json` has eight tests including corruption and a missing file.
>    `user_taste.npz` has exactly one, and only of an **unseeded** model. `exploration_state.json` and
>    `feedback_history.json` have **no round-trip test at all** — `save_feedback_history()` and
>    `load_feedback_history()` are never called directly by the suite. That last one is now
>    load-bearing: Stage 3 made the `♥` marks depend on that file surviving a restart (L4), so it is
>    the round-trip to write first.
> 3. **L1 is now measured, and its impact is narrower than the finding reads.** Confirmed by driving
>    it: after `Screen.start()`, `signal.getsignal(SIGWINCH)` is urwid's `Screen._sigwinch_handler`,
>    not `AdaptiveDJTUI._on_sigwinch`. But a **resize** already self-heals within 0.5 s, because
>    `_art_geometry()` is re-derived from `get_cols_rows()` on every tick and the renderer's key
>    changes with it. What does *not* self-heal is a **window move at unchanged size** — same key, so
>    the send is skipped — which is precisely what `force_redraw()` exists for. Fix the handler, but
>    do not describe it as fixing resize; it fixes monitor moves.
> 4. **The fallback text mode cannot be tested the way the urwid mode was.** `_run_simple_mode()`
>    reads real stdin through `termios` / `tty` / `select`, so a harness needs a pty. Its bindings were
>    reconciled with the footer and the README by *reading* in Stage 3, not by testing, and they are
>    the only interface claim in the project with nothing behind it. Note also that `[I]` has no
>    binding there while `_show_model_info()` returns its lines for a non-urwid caller — a loose end
>    Stage 3 left rather than an intended asymmetry.
> 5. **Do not re-run the generation to satisfy M7.** A full regeneration is ~5.5 minutes and rewrites
>    the artifact every downstream number depends on. M7's outstanding items are the three
>    contradictory download sizes, the runtime estimate and the untested macOS claim — all
>    documentation. `--stats`, `--describe` and `--compare-templates` answer questions without
>    rewriting anything. §10b already holds the generation figures.
>
> **`data/state/` currently holds real data** from Stage 3's live verification runs — a taste model
> with updates, exploration state, feedback and play history. None of it is committed and none of it
> is worth preserving (D3). Delete it before measuring anything about a cold start.

> **What Stage 3 changed about this stage.** M1c is partly done — the display now has its own
> coverage, which is where N1 was hiding — so what is left of it is narrower and named above. Two new
> items arrived: un-liking moved here from Stage 3 (it is a taste-model change, not a display one),
> and the fallback text mode's bindings were reconciled by reading rather than by testing. L9's
> keybinding row is closed.

| ID | Change | Why |
|---|---|---|
| **M1c** | Finish the suite: persistence round-trips, and a harness for the non-urwid fallback mode | Stage 3 covered the widget tree and immediately found a three-stage-old crash (N1). The fallback mode is now the only interface with no coverage at all, and Stage 3 changed its bindings. |
| **M7** | Rewrite the README against actual behaviour; re-measure model cache size, generation time and disk requirement | The three contradictory download sizes, the runtime estimate and the untested macOS claim. Stage 3 rewrote the controls table, the screenshot and the "Describing the session" section. |
| **L1** | Re-install the SIGWINCH handler after `loop.run()` | urwid replaces it at startup. Now more visible than before: with the art geometry derived per render, a missed resize is the one path that still leaves the image stale. |
| **L2** | Remove or explicitly mark the kitty/sixel art paths as non-functional | They fight urwid for the terminal and lose every 0.5 s. |
| **L8** | Bind `[L]` on an already-liked track as un-like | Moved from Stage 3. The display half is done (`SessionHistory.liked`); what remains is deciding what `UserTaste` does with a retraction. |
| **L9** | Fix the `.gitignore` `.gitkeep` scaffolding; terminate the ueberzugpp child in `_shutdown` | The ueberzugpp leak shares a root cause with H3 — fix them together. |
| **L6** | *(optional)* Move to `mpc idle` | Only worth it against a remote `MPD_HOST`. At depth 1 the 2 s poll has minutes of runway. |
| — | *(optional)* Size the `[I]` overlay's ListBox from the widget tree | It computes its inner size from the 70% relative width it declares — a hand-derived number of exactly the kind H8 removed, in a place where being wrong costs a slightly-off scroll page (§0b). |

**Done when:** a fresh clone runs the suite green, and green means something.

> **Complete, 23 July 2026.** 542 tests, green, ~18 s; driven live at 120×45 and 80×24 from a cold
> `data/state/`, with the user's MPD queue, modes and volume restored byte-identically. Every row
> above is done except L6, which is declined with a reason. §0b records what the stage found that
> this plan did not anticipate — including that **trap 3's own fix direction, applied as written,
> broke the application**, and that the suite had been overwriting the developer's live taste model.

---

## 9 · Decisions taken, and what is deliberately deferred

**No decisions are outstanding.** This section records the judgement calls the plan rests on so that a
future reader can reverse one without re-deriving why it was made, and lists what was consciously left
for later.

### Judgement calls, and how to reverse each

| Decision | Rationale | Reversal cost |
|---|---|---|
| **Remove time context (D6)** | Unvalidated rather than provably wrong, but the sole source of four findings, and unevaluable until the rest works. | Low. The concept is sound; re-add against honest embeddings and real listening history. Do not restore the old code. |
| **Force `consume on` (D2)** | Makes the refill condition `len(queue) < 2` with no position parsing. Guarded by log-on-change and restore-on-every-exit-path. | Moderate. The alternative is parsing `#N/M` from `mpc status` and leaving consume alone — more code, no side effects. |
| **Delete `[V]` (D8)** | Measured: it lands the session vector at 0.450 on-manifold quality against 0.641 for an ordinary one — halfway to noise — for 9.3% pool turnover. Its queue-clearing job no longer exists. | Low, but do not restore `force_shift`. If a distinct "change subject" gesture is wanted later, build it as a named-descriptor jump, not a random rotation. |
| **Rank-Boltzmann sampling (H6)** | Argmax reproduces the byte-identical session from a given state. Score-softmax needs recalibration every time the score scale moves. Rank is scale-invariant. | Trivial — one function. |
| **The snap is solved *through*, not applied after (Stage 2)** | λ chosen against the un-snapped vector and snapped afterwards let a second consecutive skip land back where the run started — measured live at 1% turnover against a 20% target. Solving against the post-snap vector is the same principle the schedule already rests on: state the target on the thing that actually selects. | Trivial — one keyword argument — but it reintroduces a defect that offline medians do not show. |
| **Turnover is reported against the skip run's start, not the previous press (Stage 2)** | "How much has changed since I started skipping" is the question a listener is actually asking; per-press deltas understate an escalation by construction. | Trivial — one stored anchor vector. |
| **`ENTER` on history requeues (H1d)** | Replaces the removed queue navigation with something useful, reusing existing plumbing. | Trivial. If unwanted, `↑↓` becomes pure scrolling and `ENTER` unbinds. |
| **The drift figure is a word count, not the specified cosine (Stage 3)** | Measured: the cosine's p10 over 40 real sessions is 0.948 and its median 0.989, so it prints "0.99" for nine readings in ten. The count of held words spans 0–3 with a median of 2. Reporting the compressed one would repeat the defect H1 exists to fix. | Trivial — `drift()` returns both. But re-read §10d first: the reason is the distribution, not the presentation. |
| **The drift and history stores live in the display layer (Stage 3)** | Neither is read by selection, and `SessionState` is closed and tested. Putting a display concern behind the display would be the shape of C4 — state split across a component that cannot see the thing it is about. | Low, but there is no reason to: the only cost is that both reset when the TUI does, which is correct for a session-scoped readout. |
| **`QueueManager.requeue_next()` adds before it deletes (Stage 3)** | It already knows its track, so it has nothing to exclude and no reason to delete first. Appending first means the queue is momentarily three deep rather than one deep, so a refusal from MPD leaves the session untouched. | Trivial, but it would reintroduce a window where a failed add leaves the queue one deep. `replace_next()` cannot use this order — it has to exclude the entry it is dropping. |
| **The Now Playing box is packed, not weighted (Stage 3)** | It is the fix for N1, and it makes the panel's height — and therefore the art's position — independent of the terminal's. | Do not reverse. Weighting it raises `WidgetError` below 33 rows and reintroduces a terminal-height dependency into H8's geometry. |
| **Zero taste vector is inert in *retrieval*, not just scoring (Stage 0)** | An all-zero query to `find_similar` returns an arbitrary slice of the library, so half the candidate pool would be noise presented as preference. Skipping the taste half is what L7's own reasoning implies. | Trivial — one `if np.any(...)`. Do **not** reverse it when β ramps in; β gates the score, not the pool. |
| **A skip cannot seed the taste model (Stage 0, reaffirmed Stage 2)** | From zero, one negative update normalises to `−track` at unit length: a full-strength claim from a single rejection, stronger than the random seed it replaced. **It did *not* become redundant when the β ramp landed** — β gates the score, never retrieval, and the candidate pool opens its taste half on `np.any(taste_vector)`. Retiring it would hand half the pool to "the tracks least like the one song you rejected". | Trivial to reverse, but do not: the reason is about the magnitude of a claim from one event, which no weight can damp. |
| **`[I]` became a model inspector in Stage 0 rather than Stage 3** | D6 emptied the overlay while the key, the footer, the README and `start.sh` all still advertised it. A key that silently does nothing is the same dishonesty Stage 0 exists to remove. | None — Stage 3 extends the same overlay rather than building one. |
| **Batches are filled per track, not per window (Stage 1)** | Cross-track packing makes a track's embedding depend on which tracks sat next to it in the run, so a library could not be extended without every existing vector shifting under the taste model. Costs one partial batch per track. | Trivial, but it forfeits the bit-reproduction guarantee and the test that asserts it. |
| **`RMS_GATE = 0.01` (Stage 1)** | Not chosen: the window-RMS distribution over 40 tracks is bimodal, 1.8% at ~4 × 10⁻⁵ and real content from ~0.04. Every threshold in between drops the same windows to within half a percent. The gate sits in the empty band. | Trivial, and re-measurable — the distribution is printed by the run and stored in the artifact's metadata. |
| **`STD_FLOOR_FRACTION = 0.5` for the descriptor gate (Stage 1)** | Relative to the library's own median std, so it needs no recalibration when the model, template or collection changes — the same move as rank-Boltzmann and z-scoring. | Trivial. If it ever drops something you wanted, the run prints every descriptor's std, so the decision is inspectable rather than mysterious. |
| **`skip_turnover_schedule = (0.05, 0.20, 0.50, 0.85)` (Stage 2)** | The only *input* to the escalation — λ is solved for it at every press, so this is the one place a number is chosen. It is stated in units the listener can verify ("85% of what I would have heard is now different") rather than in vector-space magnitudes nobody can reason about, which is why it survived the embedding space changing underneath it. | Trivial, and it is the right knob: if three consecutive skips feel too close to a full reset, add a row rather than fudging λ. The console prints what each press *measured*, so the effect of a change is observable. |
| **`skip_snap_from_run_length = 2` (Stage 2)** | Both directions are measured, not assumed. Below it, `snap()` overshoots — it is a move with a turnover floor of ~8% against a 5% target. At and above it, `snap()` is required: an unguarded λ solved for the 85% target lands below the library's 1st-percentile on-manifold quality **100% of the time**. Gating at 3 instead leaves press 2 at 0.42, worse than the deleted `[V]`. | Low, but re-measure before moving it — §10c has the table, and both failure modes are silent. |
| **`taste_ramp_updates = 20` (Stage 2)** | How long β takes to earn its configured weight. A control constant: it shapes how fast the taste model starts counting, and asserts nothing to the user. | Trivial. Shorter makes a new listener's first few likes dominate sooner; longer keeps them on session-only for longer. `[I]` reports "β earned" as a percentage, so the effect is visible. |
| **`minimum_sampled_pool = 4` (Stage 2)** | Below it the rank distribution is replaced by a uniform draw, because with two or three candidates τ decides the outcome and the "choice" is a formality. | Trivial. Only reachable on a tiny library or under heavy exclusion. |
| **`checkpoint_every_n_tracks = 5` (Stage 2)** | How often learned state is written during a session (H3). Four small files, so the cost is negligible; the number only trades write frequency against how much a hard kill can lose. | Trivial. |
| **`minimum_mpd_coverage = 0.5` (Stage 1)** | A control constant, not a truth claim: it decides when to refuse rather than what to tell the user. The actual coverage is always logged, so the number the user reads is measured. | Trivial — one config key. |

### Deferred, with the door left open

1. **Free-text steering.** "Something nocturnal and sparse" typed by the user: embed the text, take its
   top-20 library tracks, blend toward that centroid. Mechanically the descriptor machinery already
   built, driven by arbitrary input instead of a fixed bank. The natural next feature once the player
   works. **The same trap applies** — blend toward the *audio* centroid of matching tracks, never
   toward the raw text vector (H9).

2. **Per-window representations.** Stage 1 persists the full window matrix specifically so this can be
   revisited without regenerating. Medoid clustering or max-over-windows similarity would let a track
   match on one section — better recall, more whiplash. Worth an A/B once there is a working player to
   judge it against.

3. **Re-tuning the control constants.** Exploration step sizes and the taste update rates were chosen
   against the compressed similarity scale and behave differently now that it is centred. H9 replaced
   the worst of them — `penalize_similar`'s fixed 0.15, which moved about 2.9% of the candidate pool
   per press (§10c) — with a solved λ. The rest are left deliberately: listen first, then tune once
   with real data rather than guess new numbers now. The constants that made *claims* are already
   deleted (D4); these only shape behaviour.

4. **τ_max = 15.** The one genuinely new constant, now shipped and still uncalibrated by listening.
   It is documented in H6 with the measurement that motivated it and an explicit note that it is a
   starting point, so it does not quietly become the next uncalibrated threshold. Raise it until
   unattended sessions start feeling incoherent, then back off. `[I]` reports the τ in force, so the
   effect of a change is observable rather than guessed at.

5. **The turnover schedule's middle rows overshoot.** Post-snap turnover is a set-overlap count and
   moves in jumps, so the smallest λ meeting the 50% target typically delivers ~70–83%. The schedule
   is a floor and the *measured* value is what the console reports, so nothing dishonest is shown —
   but three consecutive skips are closer to a reset than the table's "wrong direction" suggests. If
   that proves too steep in use, the fix is another row in the schedule, not a fudged λ.

6. **`previous_track`.** Impossible via MPD under `consume on`; would need re-adding from the app's own
   history. No binding exists today, so nothing regresses.

7. **The Session panel starts empty on every launch, and that is deliberate.** It is a *session*
   history: the plays come from watching MPD's current track, and the cursor into them is
   session-scoped. `feedback_history.json` holds enough to reconstruct a previous evening, so this
   will read like an unfixed bug to someone who finds it — it is not. What *does* survive a restart is
   `♥`, because "you like this track" is a persistent fact about the track, while `⏭` and `✓` are facts
   about tonight (L4). If a cross-session history is ever wanted, it is a different panel with a
   different claim, not a wider version of this one — and note that `[N]`'s escalation, the drift
   window and the replay cursor all read "this session" as their unit.

---

## 10 · Evidence appendix — the pre-Stage-1 library

*Raw measurements behind the original findings, taken on the 616 crop-based embeddings that Stage 0
deleted and Stage 1 replaced. They are kept because they are what each fix was designed against.*

> **For anything you are about to build on, use §10b instead.** Both the embeddings and the space
> changed: full coverage replaced random crops, and the library is centred on load. Where the same
> quantity appears in both sections, §10b is the live one and the finding above carries its number.

### C3 — embedding non-determinism

Ten embeddings of one synthetic three-minute signal through the project's own
`CLAPEmbeddingGenerator`, compared to 300 random library tracks:

```
SELF-SIMILARITY of ONE track embedded 10x:
  min 0.354  median 0.884  max 0.998  mean 0.740
LIBRARY inter-track similarity: mean 0.577  median 0.582
fraction of self-pairs BELOW library median: 36%

# root cause — transformers 5.1.0
# max_length_s = 10, and this checkpoint's preprocessor_config.json sets
# truncation = rand_trunc: anything longer is cropped at a uniformly random offset
```

### H1 — vibe entropy is constant

```
vector_entropy = -Σ|v|·log(|v|)  over 616 real CLAP vectors
  min 53.048  max 56.836  mean 54.840  std 0.711

code thresholds: >5.0 "eclectic"  >4.0 "diverse"  else "cohesive"
  fraction >5.0: 100.0%   >4.0: 0.0%   ≤4.0: 0.0%
```

### C1 — queue exhaustion

Real `QueueManager` + real 616-track library, driven against a queue model with correct MPD semantics
(played tracks retained, consume off):

```
after initialize_queue: mpd queue=10
*** PLAYBACK STOPPED after 9 tracks (queue len=10, pos=10) ***
final: mpd queue len=10 planned_queue=1 state=stopped
```

### C2 / M3 — live environment state

```
$ mpc status
volume:100%   repeat: off   random: on   single: off   consume: off

$ grep music_directory /etc/mpd.conf
music_directory "/mnt/storage/music"

$ ls -la /var/lib/mpd/music          # the old hardcoded config.py default
lrwxrwxrwx. root root -> /mnt/storage/music   # worked only by symlink

$ echo $MPD_MUSIC_DIR                 # unset
```

*(M3 is fixed: the path is now parsed from `~/.config/mpd/mpd.conf` and proved against `mpc listall`
before anything expensive runs. `mpc` itself cannot report it — `music_directory` is a server-side
path the protocol never exposes.)*

### C5 — the compressed similarity scale

Derived from the library distribution below. Two unrelated crop-based embeddings sit at median cosine
**0.582**; the 1st percentile is 0.074. The scoring code's `(sim + 1) / 2` normalisation therefore
produces values in roughly `[0.54, 0.98]`, and `novelty = (1 − max_sim) / 2` clusters near 0.21.

*(On the full-coverage embeddings this is worse — median 0.675 — because mean-pooling averages each
track toward the library mean. §10b.)*

### H9 / D8 — `[V]` versus `[N]`, measured

40 simulated sessions on the real 616-track library, driven through the project's own EMA update and a
session+taste scoring approximation. "Turnover" = fraction of the top-100 candidate pool that changed.

```
                                   cos(new,old)   pool turnover
  [N] x1   (penalize_similar 0.15)      0.995          3.9%
  [N] x3                                0.977          7.2%
  [N] x5                                0.948         11.9%
  [V] x1   (force_shift 0.5 random)     0.710          7.4%
  [N] x10                               0.818         23.1%
  [N] x20                               0.474         58.1%

  cos(session vector, the track it is about to play): mean 0.790  max 0.959
```

Repulsion magnitude required from the 3-skip centroid, solved per session:

```
  target turnover     median lambda     turnover after snap()    on-manifold quality
        5%                0.23                  11.2%                   0.775
       20%                0.55                  13.9%                   0.746
       50%                0.80                  40.5%                   0.688
       85%                1.05                  86.9%                   0.656

  reference — mean similarity to the 25 nearest real tracks:
    ordinary session vector : 0.697
    random direction        : 0.105
```

Re-measured on the Stage 1 library in §10b. The schedule survived — it targets turnover, not λ — and
the solved λ values landed within 0.1 of these.

### H6 — argmax versus rank-Boltzmann sampling

30-track sessions from an identical starting state, centred space, 616 crop-based embeddings.
Re-measured on the Stage 1 library in §10b; the conclusion held, the numbers moved:

```
  selection rule        within-session sim   distinct sessions (5 runs)   overlap w/ run #1
  argmax (current)             0.603                   1 / 5                   100%
  rank-Boltzmann tau=1         0.596                   5 / 5                    42%
  rank-Boltzmann tau=7         0.599                   5 / 5                    53%
  rank-Boltzmann tau=15        0.570                   5 / 5                    31%
  (library baseline)           0.014

  # coherence is unchanged; run-to-run variety is the entire gain
  # ordering among tau values is within noise at n=5 — calibrate tau_max in use
```

### C5 — centring the space

```
  library random-pair similarity, 616 crop-based embeddings
    raw     : mean 0.569   p75 0.737     ← scoring constants were tuned against this
    centred : mean 0.014   p75 0.254     ← the range the formulas actually assume
```

### Library and state inventory *(all of this is being deleted per D3)*

```
track_embeddings.npz   616 tracks × 512 dims, all L2-normalised (0.9999996 – 1.0000005)
  metadata: laion/clap-htsat-unfused, 2026-02-17, CUDA (RTX 3070)
  stats: 632 attempted, 616 successful, 16 failed, 198.05 s

user_taste.npz         15 updates — 3 likes, 8 full listens, 4 skips
exploration_state.json exploration 0.59 (ceiling 0.7), 4 skips, 8 listens
feedback_history.json  16 events — 8 full_listen, 4 skip, 3 like, 1 vibe_skip
                       all within 19:02–19:40 on 2026-02-17
time_context.npz       JSON despite the extension; np.load raises UnpicklingError

library similarity     mean 0.577  std 0.208  median 0.582
                       percentiles [1,5,25,50,75,95,99] =
                         0.074  0.224  0.431  0.583  0.738  0.901  0.946
```

### Test suite as it stands

```
# at audit time
$ python3 test_phase2.py             → exit 1   ("Passed: 66, Failed: 1")
$ python3 test_phase3.py             → exit 0   (9 passed)
$ python3 test_phase3_integration.py → exit 1   (4 passed, 1 failed)
$ git ls-files | grep test           → (nothing — .gitignore excludes test_*.py)

# after Stage 1
$ python3 -m pytest tests -q         → 179 passed
```

---

## 10b · Stage 1 measurements

*Taken 23 July 2026 on the machine in the header, against the rebuilt 674-track library. These
supersede the pre-C3 figures above wherever they overlap; the originals are kept because they are what
the fix was designed against.*

> **The H9 and H6 tables at the end of this section were re-measured in Stage 2 with the shipped
> code, and several figures did not reproduce — see §10c, which is the live record for anything
> about skips or selection.** The generation, determinism, centring, silence-gate, template and
> descriptor measurements above are unaffected and remain current.

### The generation run

```
690 tracks in MPD  ·  674 embedded  ·  16 failed  ·  24,494 windows kept, 213 gated as silence
5m 23s wall clock   2.09 tracks/s   75.8 windows/s   45.5 MB
laion/clap-htsat-unfused, transformers 5.1.0, torch 2.10.0+cu128, RTX 3070

all 16 failures are one album — Jimi Hendrix, "Electric Ladyland" — every track:
  RuntimeError: Failed to decode audio samples: Could not receive frame from
  decoder: Invalid data found when processing input
```

The original run also lost 16 tracks (§10, inventory). Same files: the FLACs are corrupt, and the
audit's complaint was never that they failed but that nothing recorded *which*. `failed.txt` now does.

### Throughput — where the cost actually is (M8)

```
  batch  1: 29.2 windows/s        batch 16: 31.9 windows/s
  batch  4: 32.0 windows/s        batch 32: 38.2 windows/s
  batch  8: 32.3 windows/s        batch 48: 36.8 windows/s

  ClapFeatureExtractor alone, no GPU:   39.0 windows/s   ← the bottleneck
  same, threaded:  1 → 37.4    2 → 59.5    4 → 75.1    8 → 83.0 windows/s
```

Batching buys ~30%; threading the mel extraction buys 2×. The GPU is nearly idle either way.

### Determinism (C3)

```
  same file, same batch size, twice     → bit-identical (pooled and per-window)
  re-embedded vs the vector on disk     → bit-identical
  batch 32 vs batch 1, three tracks     → max |Δ| 2.2e-8, cos 1.0000000
  truncation="fusion" at exactly 10 s   → RuntimeError (4 channels into a 1-channel patch embed)
```

### The similarity scale, rebuilt (C5)

```
  random-pair similarity, 674 pooled full-coverage embeddings, n=400 sampled
    raw     : mean +0.670  median +0.675  p05 +0.363  p75 +0.819   std 0.183
    centred : mean +0.011  median −0.049  p05 −0.562  p75 +0.297   std 0.412
  centroid norm: 0.821     ← how far off the origin the library sits

  for comparison, the pre-C3 crop-based figures from §10:
    raw     : mean 0.569   p75 0.737
```

Pooling ~36 windows per track pulls every vector toward the library mean, so full coverage makes the
raw space *more* anisotropic (0.569 → 0.670), not less. The centred distribution is what the scoring
formulas always assumed: the negative half of the range is now occupied.

### The silence gate (C3)

Window RMS over 1,475 windows from 40 randomly chosen tracks, peak-normalised per track:

```
  p0.5  0.00004     p5   0.04239     p50  0.18635
  p1    0.00004     p10  0.07214     p75  0.27812
  p2    0.00550     p25  0.12682     p100 0.48096

  gate 0.001 → drops 1.76%     gate 0.010 → drops 2.24%     gate 0.030 → drops 3.53%
  gate 0.003 → drops 1.90%     gate 0.020 → drops 2.78%     gate 0.050 → drops 6.58%
```

Bimodal with an empty band between ~5 × 10⁻⁵ and ~0.04. `RMS_GATE = 0.01` sits in it. On the full run
213 of 24,707 windows (0.9%) were gated.

### Prompt template selection (H1, step 2)

Each template's 49 descriptors scored against all 674 centred embeddings. `mean std` is how far apart
the bank spreads the library; `eff. rank` is the participation ratio of the descriptor correlation
matrix — roughly how many independent things the bank measures; `top-3 perplexity` is the effective
number of distinct words the readout actually produces across the library.

```
  template     mean std   min std   eff. rank   distinct in top-3   top-3 perplexity
  recording      0.1798    0.0958         2.5          49 / 49              34.8
  bare           0.1435    0.0806         4.0          49 / 49              38.2
  sounds         0.1539    0.0773         2.5          48 / 48              33.9
  genre          0.1489    0.0897         2.0          49 / 49              30.2
```

`recording` ("This is a recording of {} music.") wins on spread, is within noise of the best on
readout diversity, and matches what CLAP was trained on — audio captions, not bare adjectives. The low
effective rank is a property of a metal-and-rock-heavy library, not a degenerate readout: every
descriptor still surfaces for some track.

### The variance gate (D5b)

```
  49 descriptors, floor 0.0927 = 0.50 × median std 0.1853
  strongest: mellow 0.2387  romantic 0.2356  melancholic 0.2327  dreamlike 0.2308
  weakest:   frenetic 0.0958  motorik 0.1008  driving 0.1135  triumphant 0.1172
  dropped:   none
```

### Descriptor spot check (H1, acceptance for the gate)

Top-3 by z-score, session-vector-free — these are the tracks' own embeddings:

```
  Bathory — Odens Ride Over Nordland      cavernous (+1.8) · orchestral (+1.0) · dense (+0.7)
  Arcane OST — Heavy Is the Crown         cinematic (+1.7) · tense (+1.5) · menacing (+1.5)
  Gojira — Ocean Planet                   aggressive (+1.0) · gritty (+0.8) · intense (+0.7)
  bye2 — onionfriends2004 (breakcore)     electronic (+0.8) · hypnotic (+0.7) · danceable (+0.6)
  Tame Impala — Be Above It               motorik (+1.5) · halftime (+0.5) · groovy (+0.4)
  The Smiths — The Queen Is Dead          aggressive (+1.2) · energetic (+1.2) · gritty (+1.1)
  Sabah — Dakhlak La Tealeqny Feek        motorik (+1.0) · halftime (+0.9) · joyful (+0.6)
```

Six of the seven are recognisable descriptions — Bathory's track is a reverberant orchestral intro,
"Be Above It" is built on a repeating motorik drum loop, and "The Queen Is Dead" is the loudest thing
on the record. The Arabic pop entry is the weak one: `joyful` fits, the rhythm words do not obviously.
The rhythm axis carries the four weakest descriptors in the bank, which is consistent.

### Re-measured for Stage 2 — H9's skip mechanics `SUPERSEDED by §10c`

> **Provenance only.** These came from a simulation written alongside the audit, before the code
> existed. §10c re-took them by driving the shipped `manifold.py` and `SessionState`, and four
> figures did not reproduce — including the `[N]`×1 turnover quoted below as 0.3%, which is about
> 2.9% and varies tenfold with how settled the session vector is. **Use §10c.**

*The numbers Stage 2's `[N]` rebuild was specified against. §10's versions were taken on the 616
crop-based embeddings; these are the space the code now runs in.*

**Method.** 40 sessions per row on the 674-track centred library. Each session starts from a random
unit vector and plays 12 tracks by argmax under the session vector alone, applying the project's own
EMA update (`decay = 0.85`). Turnover = fraction of the top-100 pool that changed. Session-only, so no
taste, novelty or anti-repetition term is in play — the real scorer will move the pool *more* than
this at every row, so treat these as the tight end of the range and the ordering as the finding.

```
  on-manifold quality — mean similarity to the 25 nearest real tracks
    a real track            0.729
    session vector          0.641
    after [V] force_shift   0.450     ← half a random direction blended in
    random direction        0.085

  candidate-pool turnover
                                     cos(new,old)   turnover
    [N] x1   (penalize_similar 0.15)     0.999         0.3%
    [N] x3                               0.989         1.3%
    [N] x5                               0.958         2.4%
    [V] x1   (force_shift 0.5)           0.703         9.3%
    [N] x10                              0.665        10.7%
    [N] x20                              0.243        87.9%

  cos(session vector, the track it is about to play): mean 0.971
```

The session vector locks onto its neighbourhood far harder in the centred space (0.971 against 0.790
before), which is why a fixed 0.15 nudge now moves almost nothing. It is also, independently, the
clearest argument for H6: an argmax over a pool this concentrated is close to deterministic.

**Solving λ from the 3-skip centroid, then `snap()`:**

```
  target    median lambda    turnover    after snap()   on-manifold after snap()
    5%          0.25            5.7%         17.8%              0.826
   20%          0.60           24.5%         20.8%              0.806
   50%          0.75           58.9%         56.5%              0.709
   85%          0.95           90.2%         93.3%              0.765
```

Two things to carry into the implementation:

- **The schedule survived the space changing.** λ is solved against a turnover target, so it
  re-derives itself; the values it lands on (0.25 / 0.60 / 0.75 / 0.95) are within 0.1 of the
  pre-Stage-1 solve. This is the payoff of specifying the target in observable units.
- **`snap()` is a move, not a projection.** At the 5% target it nearly *quadruples* turnover
  (5.7% → 17.8%), because relocating to the 25-NN centroid displaces the vector by more than a small
  λ did. At 50% and 85% it is roughly neutral. H9's "*n* ≥ 2 only" is therefore load-bearing, not a
  nicety — applying it to a single skip would overshoot the schedule badly.

### Re-measured for Stage 2 — H6's selection rule `SUPERSEDED by §10c`

> **Provenance only** — a pre-implementation simulation. §10c checks the *shipped* τ map against
> H6's own published p(rank 0) table instead, and the run-to-run variety claim is now a unit test.

30-track sessions from one fixed start state on the 674-track centred library, with the 20-track
anti-repetition gap applied:

```
  selection rule        within-session sim   distinct sessions (5 runs)   overlap w/ run #1
  argmax (current)             0.539                   1 / 5                   100%
  rank-Boltzmann tau=1         0.618                   5 / 5                    43%
  rank-Boltzmann tau=7         0.811                   5 / 5                    31%
  rank-Boltzmann tau=15        0.759                   5 / 5                    16%
  (library baseline)           0.009
```

The conclusion held and one detail inverted: sampling is now slightly *more* coherent than argmax
rather than slightly less. Strict argmax, forbidden from repeating within 20 tracks, is pushed to the
next-best unplayed candidate every time and walks away from where it started; sampling circles a
neighbourhood. The ordering among τ values remains within noise at five runs — **τ_max ≈ 15 is still a
starting point requiring calibration in use, not a finding.**

### Subprocess overhead — measured, not assumed

```
one `mpc status` subprocess: 0.8 ms
steady state ≈ 12 mpc calls/sec → ~1% of one core

# a real cost only against a remote MPD_HOST, where each call is a TCP round trip
```

### Projected cost of the C3 rebuild — and what it actually cost

```
projected:  616 tracks × ~24 windows ≈ 14,800 forward passes, ~30 MB, "low single-digit minutes"
actual:     674 tracks × 36.3 windows =  24,494 forward passes, 45.5 MB, 5m 23s

# the projection was low on window count (tracks are longer than 4 minutes on average)
# and wrong about the bottleneck: audio decode did not dominate, mel extraction did (M8)
```

---

## 10c · Stage 2 measurements

*Taken 23 July 2026 by driving the **shipped** `manifold.py`, `SessionState` and `TrackSelector`
over the built 674-track centred library, plus one end-to-end session against the live MPD. Where
these overlap §10b's "Re-measured for Stage 2" tables — which were produced by a simulation written
alongside the audit rather than by the code — **this section is the live one**.*

### What did not reproduce from §10b

```
                                        §10b said    measured with shipped code
  [N] x1, fixed 0.15 nudge                  0.3%              2.9%
  on-manifold quality, session vector       0.641             0.787
  on-manifold quality, after [V]            0.450             0.561
  [V] x1 pool turnover                      9.3%              6.6%
```

None of these changes a conclusion, and two of them strengthen one. `[V]` still lands far below a real
track and is still beaten by two escalated `[N]` presses, so D8 holds — and it holds for the
structural reason H9 always rested on rather than for the numeric one. The fixed 0.15 nudge at 2.9%
per press is still nowhere near a change of direction, which is H9's case for solving λ.

**One figure in §10b was materially misleading and is the reason to re-measure rather than inherit:**
`[N]`×1 turnover depends almost entirely on how settled the session vector is, which no earlier table
records.

```
  tracks played before the skip    mean turnover    cos(new, old)
             1                         25.8%            0.989
             3                          7.3%            0.990
             6                          4.6%            0.992
            12                          2.9%            0.996
            24                          2.7%            0.997
```

A skip early in a session moves nearly ten times as much of the pool as the same skip an hour in. Any
single number for "what does `[N]` do" is a statement about one point on this curve.

### Stage 2 — turnover and λ vs the repulsion magnitude

40 simulated sessions, each settled over 12 tracks, repelled from the 3-track skip-run centroid.
`raw` is the plain repulsion; `snapped` replaces it with the centroid of its 25 nearest real tracks.

```
  lambda   turnover raw   turnover snapped   quality raw   quality snapped
   0.05        0.9%             8.9%            0.793          0.861
   0.20        3.2%             8.4%            0.774          0.859
   0.40        8.9%             8.1%            0.727          0.855
   0.60       19.6%            11.4%            0.629          0.848
   0.80       44.0%            29.3%            0.448          0.816
   1.00       88.8%            88.1%            0.304          0.762
   1.20       98.5%            98.7%            0.397          0.818
   2.00      100.0%           100.0%            0.552          0.837

  reference — mean similarity to the 25 nearest real tracks
    a real track (median of all 674)   0.748       [V], force_shift 0.5   0.561
    an ordinary session vector         0.787       a random direction     0.090
```

Two things this settles, both of which the plan asserted on weaker evidence:

- **The snap is what keeps large λ usable.** Unguarded quality falls off a cliff past λ ≈ 0.5 and
  bottoms at 0.30 — *below* the deleted `[V]*. Snapped, it never leaves 0.76–0.86.
- **The snap has a turnover floor of its own,** about 8%, against a 5% target for a single skip. That
  is the audit's stated reason for gating it at n ≥ 2, and it is correct.

### Stage 2 — on-manifold quality has no fixed threshold, only a distribution

```
  on-manifold quality across all 674 real tracks
    p0 0.427   p1 0.463   p5 0.541   p25 0.667   p50 0.752   p75 0.883   p100 0.961
```

A real track in a sparse corner scores 0.43. So "still music" cannot mean "above some number" — the
assertion that means something is *no worse than the least typical real track*, and that is what the
suite checks. Measured against it:

```
  solved for target    unguarded quality    below the p1 floor    snapped quality
        5%                   0.731                  0%                0.820
       20%                   0.579                 13%                0.860
       50%                   0.335                 87%                0.777
       85%                   0.279                100%                0.771
```

The 85% row is why `snap()` is not optional above the first press, and the 5% row is why it is not
applied *to* the first press.

### Stage 2 — the escalation as it actually runs

Solving λ against the **post-snap** vector (§0b, item 2). 40 sessions, five consecutive presses each,
turnover measured against the session vector as it stood when the run began:

```
  press   target   median λ    turnover p10 / median / p90    min quality
    1       5%       0.28         5.0%    5.0%    7.0%           0.588
    2      20%       0.75        23.9%   29.0%   64.8%           0.576
    3      50%       0.90        72.0%   83.0%  100.0%           0.534
    4      85%       1.28       100.0%  100.0%  100.0%           0.563
    5      85%       1.83       100.0%  100.0%  100.0%           0.619

  presses that moved BACKWARDS vs the press before: 0 of 160
  library p1 on-manifold floor: 0.463 — no press went below it
```

Before the correction, the same measurement had presses that *reduced* turnover relative to the one
before, and a live session printed `Skip #2: … 1% … (target 20%)` immediately after Skip #1 achieved
5%. The medians looked acceptable throughout; only the live run and the per-press monotonicity check
exposed it.

Live, from `data/dj.log` of the accepted run:

```
Skip #1: λ=0.40, 5% of what you would have heard is now different (target 5%)
Skip #2: λ=0.80 + snap, 20% ... (target 20%)
Skip #3: λ=0.85 + snap, 70% ... (target 50%)
Skip #4: λ=1.45 + snap, 100% ... (target 85%)
Skip #5: λ=1.95 + snap, 100% ... (target 85%)
```

### Stage 2 — the τ map, checked against H6's own table

τ is linear in the exploration scalar over `[exploration_min, exploration_max]`, floored at `tau_min`:

```
  exploration    τ      p(rank 0)     H6's published claim
     0.1        1.0        63%              63%
     0.4        7.5        12%              13%
     0.7       15.0         6%               6%
```

The shipped map reproduces the table H6 was written around, so the behaviour a reader was promised is
the behaviour that ships. A test recomputes these from the code rather than restating them.

---

## 10d · Stage 3 measurements

*Taken 23 July 2026 by driving the **shipped** `VibeReadout`, `SessionState` and `AdaptiveDJTUI` over
the built 674-track centred library and the 49-word bank, plus two end-to-end sessions against the
live MPD in a pty. This is the live record for anything about the display.*

### The zero-vector hazard, on the real bank

```
  bank.top(zeros)            shimmering · orchestral · serene
  VibeReadout, unseeded      ♪ —  nothing has played yet
```

Every similarity to a zero vector is exactly 0, so every z-score is `−mean_d / std_d` — finite,
deterministic and determined entirely by the bank's own baselines. The bank does not refuse and
should not: the arithmetic is valid, and the same code path is correct for a real vector. The gate
belongs in the readout, and there is one at every entry point (H1).

### Descriptor drift over a real session

40 sessions of 30 tracks each on the 674-track library, each starting from a random real track,
selecting by argmax under the session vector and applying the shipped EMA. 1,040 readings taken at a
full five-track window.

```
  words held (of 3)     0: 1%    1: 12%    2: 42%    3: 45%      mean 2.31
  cos(z_now, z_5ago)    min 0.721   p10 0.948   p50 0.989   p90 0.997   max 0.999
```

**This is why H1's specified cosine did not ship as the readout.** Ninety per cent of ordinary
listening sits above 0.947 — the top 5% of the quantity's nominal range — so a line reporting it
prints "0.99" essentially forever, and any word derived from it would be pinned to one branch. That
is the same failure as the entropy heuristic H1 exists to remove (always ≈ 55, one reachable branch)
and the same failure as C5 (a similarity scale compressed into its top third). It is the third
distinct place this project has put a number in front of a user without checking which part of its
range the data occupies.

The word count occupies its range, needs no threshold, and is a statement about something the
listener actually read. The cosine is still computed, and shown in `[I]` with these percentiles
printed beside it.

### What a skip run does to the readout

Four settled sessions (12 tracks), then three consecutive `[N]` presses through the shipped
`repel_from_skip_run`:

```
  gritty · aggressive · intense   →  menacing · cold · cinematic      0 of 3 held, cos −0.29
  aggressive · gritty · intense   →  driving · tense · cold           0 of 3 held, cos −0.65
  joyful · halftime · motorik     →  shimmering · warm · vocal-led    1 of 3 held, cos +0.30
  groovy · energetic · lo-fi      →  guitar-driven · gritty · driving  0 of 3 held, cos +0.17
```

The readout moves when the session moves, which is the property that makes it worth showing. Note the
cosine's behaviour here against its behaviour above: it discriminates fine at the extremes and not at
all in the middle, which is exactly what a compressed scale looks like.

### The pre-Stage-3 layout, by terminal height (N1)

Rendering the shipped widget tree at `HEAD 3558b88`:

```
  80x20 … 80x32   WidgetError: <Columns …> rendered (80 x 12) canvas when passed size (80, 6)!
  80x33 … 80x45   OK

  after ('pack', …):  every height from 80x6 to 80x100 renders; widths 40–90 render at 24 rows
```

Driven live in a pty at 80×24 against `HEAD`, the traceback prints over the interface. The same
driver against the Stage 3 tree renders, plays, skips, scrolls the history and opens `[I]`.

### Album-art geometry, derived versus hand-counted

```
                       hand-counted        derived (80x40)     derived (80x24)
  x                         2                    1                   1
  y                         3                    3                   3
  width                    33                   33                  33
  height       RIGHT_COL_ROWS = 10              11                  11
```

`x` was wrong by one column: the comment counted a "terminal left edge" column before the LineBox
border, but the border *is* column 0. The height was right for the pile as it stood and is wrong the
moment Stage 3's two-row vibe readout lands — which is what the finding predicted. `y` and `width`
were right and the derivation reproduces them.

The derived geometry is now **independent of the terminal's height**, because the Now Playing box is
packed rather than weighted (N1). Under the old layout the position varied with the terminal, which
is how one hardcoded `y` could be correct at 80×40 and wrong at 80×30.

### Suite growth

```
  Stage 0 →  67      Stage 1 → 179      Stage 2 → 311      Stage 3 → 416
```

Stage 3's 105 are the first to construct the widget tree: `test_art_geometry.py` (15, rendering at
seven terminal sizes), `test_vibe_readout.py` (17), `test_session_history.py` (27),
`test_tui_display.py` (32, driving the real TUI against `FakeMPD`), plus 6 for `requeue_next` and 8
deletion guards.

---

## 10e · Stage 4 measurements

*Taken 23 July 2026 against the shipped code and the real 674-track library. Two of the three changed
a design decision, and one of them contradicted the finding that prompted the work.*

### L8 — can a subtraction undo a like?

The question §8's trap 1 poses. `_update` is `v ← normalise((1 − w)·v + w·e)` with `w = 0.1` for a
like; the proposed inverse is a `−0.1` update. Measured against the truth (the same event sequence
with the like never applied), 40 seeds per row:

```
  scenario                        cos(subtract-0.1, truth)
  like is the 2nd event             min 0.999900   median 0.999953
  like early, 3 events after        min 0.999876   median 0.999942
  settled model, 20 events after    min 0.999871   median 0.999959
  5 likes, one retracted            min 0.999900   median 0.999923
  like is the only event            min 0.000000   median 0.000000
```

**The asymmetry the trap warns about is real and negligible — 10⁻⁴. The failure is the last row, and
it is total.** From zero, `(1−w)·0 + w·e` normalises to `e`; subtracting `0.1·e` gives `0.9·e`, which
normalises back to `e`. A subtraction cannot un-seed a model. The truth there is the zero vector, so
the cosine is 0 by construction — retract your only like and your long-term taste stays pinned, at
unit strength, to the track you just rejected. That is the first retraction any new listener makes.

### L8 — does a replay reproduce the model it replaces?

```
     10 events: bit-identical=True  max|Δ|=0.000e+00  cos=1.000000000000  counts match
     50 events: bit-identical=True  max|Δ|=0.000e+00  cos=1.000000000000  counts match
    200 events: bit-identical=True  max|Δ|=0.000e+00  cos=1.000000000000  counts match
   1000 events: bit-identical=True  max|Δ|=0.000e+00  cos=1.000000000000  counts match
```

Exactly, including `total_updates`, `like_count`, `skip_count` and `full_listen_count`, because the
replay drives the same `update_from_*` methods in the same order.

### L8 — but only while the history is a complete account

`_record_feedback` caps the history at 1000 events. Simulated lifetimes, comparing a replay of what
was *retained* against the vector built incrementally from everything:

```
     50 lifetime events,   50 retained: cos = 1.000000000000   complete
    500 lifetime events,  500 retained: cos = 1.000000000000   complete
    999 lifetime events,  999 retained: cos = 1.000000000000   complete
   1000 lifetime events, 1000 retained: cos = 1.000000000000   complete
   1001 lifetime events, 1000 retained: cos = 0.994142650376   truncated
   1400 lifetime events, 1000 retained: cos = 0.923200176638   truncated
```

This is what `UserTaste.explains()` tests, and the reason it needs no calibration: the discriminator
is **exact reproduction versus ≤ 0.994**, six orders of magnitude of margin, not a threshold anyone
chose. A blind replay past the cap would move the taste vector by 0.077 for reasons unrelated to the
retraction — more than the retraction itself, and invisibly.

### M7 — what the first run actually costs

```
  du -sb ~/.cache/huggingface/hub/models--laion--clap-htsat-unfused
  1232327859                                    # 1.15 GiB

  refs/main         → 8fa0f1c…   pytorch_model.bin        614,525,833 B
  refs/refs/pr/3    → 79b58ed…   model.safetensors        614,431,440 B
  (tokenizer, vocab, merges, configs)                       3,369,610 B
```

The repository's `main` carries **no** safetensors — `HfApi.model_info` lists ten files and
`pytorch_model.bin` is the only weight file among them — so transformers 5.1.0 additionally pulls the
conversion from `refs/pr/3`. Both stay cached, which is why the honest figure is 1.15 GiB rather than
the ~590 MiB the model's own size suggests. The old `REQUIRED_DISK_SPACE_MB = 700` sat between the
two and passed on a disk the download then filled.

### M7 — the CPU figure that never existed

CLAP audio-tower throughput, batch 32, 10-second windows, on the machine in the header
(12 logical cores, torch defaulting to 6 threads):

```
        mel extraction    encode        serial pipeline    24,494 windows
  cuda      33.3 win/s     333.1 win/s      30.3 win/s        13.5 min
  cpu       40.5 win/s      17.0 win/s      12.0 win/s        34.1 min
```

The encoder is **20× slower on CPU** and becomes the bottleneck, where on GPU mel extraction is (which
is M8's finding, still true). The shipped run overlaps mel with a worker pool and achieves 75.8
windows/s on GPU — 5 min 23 s — so the honest CPU projection for the same library is the encoder
ceiling, 24,494 / 17.0 ≈ 24 min, up to 34 min unoverlapped. The README states 25–35 min. Note the
GPU column here is *not* the shipped 75.8 win/s: this measures a serial pipeline with no worker pool,
so it understates both devices equally and is useful for the ratio, which is what a reader needs.

### L1 — urwid chains, it does not replace

```python
# urwid/display/_posix_raw_display.py
129:  self._prev_sigwinch_handler = self.signal_handler_setter(signal.SIGWINCH, self._sigwinch_handler)
 97:  if callable(self._prev_sigwinch_handler):
 98:      self._prev_sigwinch_handler(signum, frame)
142:  self.signal_handler_setter(signal.SIGWINCH, self._prev_sigwinch_handler or signal.SIG_DFL)
```

L1's `getsignal` output is correct and its conclusion is not: urwid's handler is on the *outside*,
and calls ours. Installing on top of that closes a cycle, which a live `SIGWINCH` demonstrated
immediately:

```
  File "tui.py", line 932, in _on_sigwinch          →  chained(signum, frame)
  File "urwid/…/_posix_raw_display.py", line 98     →  self._prev_sigwinch_handler(signum, frame)
  … ~500 frames …
  RecursionError: maximum recursion depth exceeded
```

Nine unit tests passed against this code. They were written from the finding's assumption and ran
against a double built from it — the same failure mode `FakeMPD` exists to prevent, in the stage
whose subject is durability.

### The live runs

Both from a cold `data/state/`, under `pty.fork()` with an explicit `TIOCSWINSZ`, MPD snapshotted and
restored.

```
  120×45   started ✓  playing ✓  ♥ drawn ✓  un-like exact ("rebuilt from 0 feedback events") ✓
           [I] open/scroll/dismiss ✓   SIGWINCH: no traceback ✓   no orphaned ueberzugpp ✓
           Skip #1 λ=1.05        100% turnover (target  5%)
           Skip #2 λ=0.05 + snap 100% turnover (target 20%)

   80×24   renders, no WidgetError ✓   un-like exact ("rebuilt from 2 feedback events") ✓
           Skip #3 λ=1.00 + snap 100% turnover (target 50%)
           Skip #4 λ=0.80 + snap 100% turnover (target 85%)

  modes    random on → off, consume off → on   at start
           random → on,     consume → off      after SIGTERM       (byte-identical to the snapshot)
  dj.log   0 tracebacks across both runs
```

Two observations, neither a Stage 4 change (no file in the selection stack was touched):

- **Turnover is 100% from press 1 on a cold session**, against a 5% target. §9's item 5 already notes
  the middle rows overshoot; this is its extreme, and consistent with §10c's finding that turnover
  varies tenfold with how settled the session vector is. At press 1 from an unseeded vector the pool
  being compared against is a uniform draw, so everything in it turns over. **This is the first time
  the escalation has been driven from a genuinely empty `data/state/`**, and it suggests the schedule
  means nothing until the session has settled — worth a measurement of its own before the schedule is
  ever retuned.
- **The Session panel draws no content rows at 80×24.** The header, the packed Now Playing box, the
  console and the two-row footer consume all 24. The layout renders — N1 stays fixed — but the panel
  is a border. Pre-existing; fixing it means re-weighting a tree whose packing *is* N1's fix.

### Suite growth, Stage 4

```
  Stage 0 →  67   Stage 1 → 179   Stage 2 → 311   Stage 3 → 416   Stage 4 → 542
```

Stage 4's 126: `test_simple_mode.py` (37, the fallback mode's first coverage — a pty plus
`decode_keys()` as a pure function), `test_shutdown_and_resize.py` (34, the ueberzugpp child, the
SIGWINCH chain and the `[I]` overlay's derived size), `test_persistence_round_trip.py` (22),
`test_unlike.py` (22), `test_documented_numbers.py` (13), plus 3 deletion guards for L2 — and one
autouse fixture that stops the whole suite writing to the developer's live `data/state/`.

A mutation check on the part that matters: rebinding `↑` from history to volume — the exact
regression L9 records — fails `test_the_arrow_keys_mean_the_same_thing_in_both_interfaces` and
`test_the_arrow_keys_scroll_the_history_at_a_real_terminal`, one through the shared dispatch and one
through the pty.

---

*All line references in §2–§5 are against `HEAD 8dc4275` plus the then-uncommitted `tui.py` change
(footer key-hint rewording, +10/−10, cosmetic). They predate Stage 0 and are stale for every file it
touched — see §0b. Measurements were taken on the machine described in the header; the
embedding-determinism figures use the project's own code path against the cached
`laion/clap-htsat-unfused` weights.*

*The library inventory in §10 describes state that has since been deleted (D3). The similarity
distributions it records were measured on those 616 embeddings; **§10b is the re-measurement** Stage 1
owed, taken on the 674 embeddings C3 and C5 produced. Where the two disagree, §10b is the live
library and §10 is the record of what the fix was designed against.*

*§10b's two "Re-measured for Stage 2" tables were produced by a simulation written alongside the
audit, before the code existed. **§10c re-took them by driving the shipped modules**, and four
figures did not reproduce — most consequentially `[N]`×1's pool turnover, which is not a constant at
all but a function of how settled the session vector is. For anything concerning skips, selection or
the manifold, §10c is the live record. This is the third stage in a row where re-measuring rather
than inheriting a number changed something material, which is the argument for the habit.*

*§10d is the fourth. Stage 3 measured the consistency statistic H1 had specified in prose and found
it compressed into the top 5% of its range for 90% of ordinary listening — so the readout ships a
word count instead. It also rendered the widget tree for the first time in the project's history and
found a crash on every terminal shorter than 33 rows (**N1**), which had survived three stages and
311 behavioural tests. The habit that keeps paying is not re-measuring specifically; it is refusing
to inherit a claim — a number, or a green suite — without checking what it actually covers.*

***§10e is the fifth, and it is the one where the audit was wrong about itself.*** *L1 was recorded as
"confirmed by measurement" — and the measurement was right while the conclusion drawn from it was
not: urwid wraps our SIGWINCH handler rather than replacing it, so acting on the fix direction closed
a recursion, under nine unit tests that all agreed because they ran against a double built from the
same assumption. L8's trap named the wrong failure mode, and the real one — that a subtraction cannot
un-seed a taste model — was found only by measuring the case the trap did not think to name. And the
suite itself turned out to have been overwriting the developer's live taste model for some time.*

*Every stage of this rewrite has ended the same way: the thing that was assumed rather than driven is
the thing that was wrong. Four times it was a number. The fifth time it was a finding in this
document, and the correction is written into L1 itself rather than left here.*
