# Adaptive Session AI DJ — Audit & Continuation Brief

A terminal DJ for MPD that learns your taste from audio embeddings. This document records what
the system actually does versus what it claims, the defects that block its core promise, the design
decisions taken since the first audit, and where to pick the work back up.

| | |
|---|---|
| **Repository** | `/home/gumibo/misc/programming/projects/mpd_tui_ai_dj` · branch `master` · HEAD `8dc4275` |
| **Scope** | 23 tracked files, ~6,200 lines Python + Bash. All source read in full; claims verified by execution where possible. |
| **Environment tested** | Fedora (Linux 7.1.3), Python 3.14.6, numpy 1.26.4, urwid 3.0.5, MPD/MPC present, ueberzugpp present, 616 CLAP embeddings on disk. |
| **Audit date** | 22 July 2026 |
| **Revision date** | 22 July 2026 — design decisions folded in, see §0 |

---

## How to use this document

This is the working brief for the rewrite. It is written to be picked up cold, with no prior
conversation in context.

- **§0** is the decision record. Read it first — it changes the meaning of roughly half the findings
  below, and several findings' "fix direction" blocks were rewritten because of it.
- **§2–§5** are the findings, each tagged with a status: `OPEN`, `NEW`, `DISSOLVED`, `RESOLVED by
  deletion`, `SUPERSEDED`, `ELEVATED`. Only `OPEN`, `NEW` and `ELEVATED` require work.
- **§6** is the status table — the fastest way to see what is still live.
- **§7** specifies the target data artifacts (the `.npz` schemas). Build to these.
- **§8** is the ordered plan. Each stage has a definition of done. **Each stage leaves the
  application runnable** — there is no point in the sequence where it is half-migrated and broken.
- **§9** lists what is still undecided.
- **§10** is the evidence appendix — raw measurements behind every empirical claim.

Where this document and the code disagree, the document describes the target and the code describes
`HEAD 8dc4275`. Line references are to the current code.

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
never seemed to do much. And `[V]` — the "give me something different" key — changes less of what
you will actually hear than pressing `[N]` five times, while pointing the session vector somewhere that
is not music (**H9**). It is deleted; `[N]` escalates on consecutive skips instead.

There is a pattern connecting almost all of them: **numbers chosen against a scale that was never
measured.** Entropy thresholds picked for a smaller dimension, a novelty formula assuming a range the
data never occupies, a "50% shift" that moves 29%, a temperature that would need re-tuning every time
the weights move. The rewrite's organising principle is to derive constants from the library's actual
distribution, or delete them.

None of this is architectural. The reason it survived is that the test suite reports 66 passing checks
while roughly half of them are hardcoded `True` literals.

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

Measured against the real library, `force_shift`'s "50% vibe shift" turns over **7.4%** of the
candidate pool — *less* than pressing `[N]` five times (11.9%), and in a direction that is 0.105
similar to real music where an ordinary session vector is 0.697. `[V]` is not a bigger gesture than
the skip key; it is an equally small one pointed at noise.

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

## 1 · How the system actually works

*Read this first if you are picking the project up cold. This section describes the system **as it will
be after the revision**; where it differs from `HEAD 8dc4275`, the current behaviour is noted in
brackets.*

Everything is driven by 512-dimensional CLAP audio embeddings, one per track, stored L2-normalised
in `data/embeddings/track_embeddings.npz`. All similarity is a dot product **on centred vectors**
(C5). No genre tags or metadata enter the selection logic.

```
start.sh  ──▶ main_tui.py (AdaptiveDJWithTUI)      [ main.py = stale headless twin — deleting ]
                   │
   ┌───────────────┼────────────────────────────────┐
   ▼               ▼                                ▼
tui.py         background thread                mpd_controller.py
(urwid loop,   (polls MPD 2×/sec,                (every op = an `mpc`
 0.5s redraw)   detects track change,             subprocess; 8 methods
   │            fires full-listen)                 are defined twice)
   │               │
   ▼               ▼
feedback_handler.py ──┬──▶ session_state.py   short-term vibe vector
                      ├──▶ user_taste.py      long-term taste vector → data/state/*.npz
                      ├──▶ exploration_controller.py   scalar 0.1–0.7 → scoring weights
                      └──▶ queue_manager.py ──▶ track_selector.py ──▶ track_library.py
                                                 (softmax-sampled over ~100 candidates)
```

### The selection loop, precisely

1. `TrackLibrary.get_candidate_pool()` takes the top ~150 tracks nearest the session vector and the
   top ~150 nearest the taste vector, interleaves them, and truncates to 100.
2. `TrackSelector._calculate_score()` scores each candidate as
   `α·session_sim + β·taste_sim + γ·novelty + δ·anti_repetition`.
   *[Currently also adds `0.15·time_sim` — removed per D6, restoring the weight-sum-to-1.0 invariant.]*
3. The weights come from `ExplorationController.get_weights()`, which shifts mass from session/taste
   into novelty as the exploration scalar rises. β is additionally ramped from 0 as taste updates
   accumulate (L7).
4. **One track is drawn by Boltzmann sampling over rank** — `p(i) ∝ exp(−i/τ)` with τ set by the
   exploration scalar. Rank-based rather than score-based, so it needs no recalibration when the score
   scale moves. *[Currently strict argmax — H6.]*
5. `QueueManager` ensures exactly one track sits ahead of the current one in MPD, with `consume on`
   so MPD pops finished tracks itself.
   *[Currently generates ten, blends 5% of each into a working session vector, and never refills — C1.]*

### What feeds back

| Event | Trigger | Effect |
|---|---|---|
| **Full listen** | ≥90% of duration played | Session vector updated (the primary driver). Taste +0.02. Exploration −0.02. This is the only thing that increments `tracks_played`. |
| **Skip `[N]`** | Keypress | Taste −0.05. Exploration +0.05. Session vector repelled from the consecutive-skip-run centroid by a magnitude **solved for a pool-turnover target that escalates with the run length** (5% → 85%), then projected back onto the manifold. Lookahead replaced, advance. *[Currently: fixed 0.15 nudge, queue not rebuilt — H4, H9.]* |
| ~~Vibe skip `[V]`~~ | — | **Deleted (D8/H9).** It turned over less of the candidate pool than `[N]`×5 while pointing off-manifold. Its role is covered by `[N]`'s escalation. |
| **Like `[L]`** | Keypress | Taste +0.10. Taste file saved immediately. |

### What persists

`user_taste.npz` (on every like and at exit), `exploration_state.json`, `feedback_history.json`, and
— newly — `play_history` for anti-repetition (M6b). The session vector is intentionally ephemeral.
State is checkpointed periodically, not only at exit (H3).

### Current state on disk

616 real CLAP embeddings (`laion/clap-htsat-unfused`, generated 2026-02-17 on an RTX 3070, 632 files
attempted, 16 failed silently). Taste model has 15 updates. Exploration sits at 0.59. Feedback history
holds 16 events spanning 19:02–19:40 on a single evening.

**All of this is being deleted (D3).** The 38-minute span is itself a symptom — see C1.

---

## 2 · Critical findings

*Each of these independently defeats the project's stated goal.*

---

### C1 · The queue never refills. Playback stops dead after 10 tracks. `OPEN — fix simplified`

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

### C2 · MPD's random mode silently discards every ordering decision the DJ makes. `OPEN — scope expanded`

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

### C3 · The audio fingerprints are non-deterministic ~10-second crops, not track representations. `OPEN — fix specified`

**What.** `CLAPEmbeddingGenerator.generate_embedding()` loads the entire waveform and hands it to
`ClapProcessor` with default arguments. Those defaults are `max_length_s=10` and
`truncation="fusion"`, where fusion stacks randomly positioned mel-spectrogram crops. The result:
each track's embedding describes roughly ten seconds sampled at random from it, and re-running
generation produces a different vector for the same file.

Additionally, `truncation="fusion"` is the wrong mode for this checkpoint — feature fusion is the
mechanism the `-fused` variant carries an adapter for. `laion/clap-htsat-unfused` is getting the
fusion crop behaviour without the module meant to consume it.

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
  exactly `max_length_s`, neither the truncation nor the padding branch inside `ClapFeatureExtractor`
  fires, so the `fusion` / `rand_trunc` question becomes irrelevant — both are random, and neither
  runs.

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
  vector. ~616 tracks × ~24 windows × 512 dims × float32 ≈ **30 MB**. This is the important part:
  it turns pooling from a regeneration-cost decision into a load-time knob. If you later want to try
  8 evenly-spaced windows, or medoid clustering, or max-over-windows similarity, you experiment in
  seconds instead of re-running generation.

**Why mean-pool rather than max-over-windows.** Max similarity would let a seven-minute track with one
matching ten-second passage score as high as a track that matches throughout — musical whiplash. Mean
answers "does this *whole track* fit where the session is," which is the question a DJ selecting whole
tracks is actually asking. Persisting the windows keeps the alternative available without committing
to it.

**Cost.** ~15,000 forward passes for 616 tracks. Audio decode dominated the original 198-second run
(632 tracks, one pass each); with batching (M8) the added GPU work is on the order of a minute, so the
whole regeneration should land in the **low single-digit minutes** on the 3070. M8 stops being an
optimisation and becomes a prerequisite: 24× the forward passes, one at a time, is not acceptable.

**Also.** Persist the failed-track list to `data/embeddings/failed.txt` with the exception per file.

---

### C4 · Pressing `[V]` always throws away the first track of the new vibe. `DISSOLVED by D8 — constraint retained`

> **Status.** `[V]`, `_skip_vibe()`, `process_vibe_skip()` and `recalculate()` are all deleted (D8/H9),
> so this code path ceases to exist. The finding is retained because the *constraint* it establishes
> governs the replacement skip path, and because the reasoning below corrects a fix the original audit
> got wrong. **The constraint: exactly one advance per keypress, and no `play()` call anywhere in a
> skip path.** Verify it against the new unified `[N]` in Stage 2.

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

### C5 · The similarity scale is compressed; every scoring constant is calibrated against a range that does not exist. `NEW — added in revision`

**What.** CLAP's embedding space is strongly anisotropic — embeddings occupy a narrow cone rather than
spreading over the unit sphere. The audit measured this without naming it: two *unrelated* library
tracks sit at a mean cosine similarity of **0.577** (median 0.582, std 0.208, 1st percentile 0.074).

Nothing in the codebase accounts for this, and several places assume the full `[−1, 1]` range is in
play:

| Site | Assumption | Reality |
|---|---|---|
| `track_selector.py:110,114` | `(sim + 1) / 2` maps similarity to `[0, 1]` | Real values land in ~`[0.54, 0.98]` — the bottom half of the scale is never used |
| `track_selector.py:163` | `novelty = (1 − max_sim) / 2` spans `[0, 1]` | Clusters around 0.21 with very little spread; the novelty weight γ has almost nothing to act on |
| `session_state.py:186–193` | momentum thresholds `0.85 / 0.7 / 0.5` discriminate | 0.5 is below the 25th percentile of *random* pairs — the "exploring" branch is nearly unreachable |
| `config.py` weights α/β/γ/δ | terms are comparable in magnitude | session/taste terms vary over a ~0.44-wide band, novelty over ~0.1 — so γ is effectively smaller than its nominal value |

This is why the weights never felt like they did much: three of the four scoring terms are nearly
constant across candidates, so ranking is dominated by whichever term happens to retain variance.

**Fix direction.** At load time, subtract the library centroid from every embedding and re-normalise:

```
E_centred = normalise(E − mean(E, axis=0))
```

Store the centroid in the `.npz` so the same transform can be applied to anything computed later.
This is standard practice for contrastive embedding spaces ("all-but-the-top" / mean-centring) and it
costs one line. Post-centring, random pairs sit near 0 and the full range becomes usable, at which
point:

- `(sim + 1) / 2` means what it claims
- novelty actually spans its range
- the weight constants can be tuned against something real
- the descriptor bank in H1 becomes far more discriminative

**Order matters:** do this before re-tuning any constant, or you will tune twice.

**Consequence for L7.** Post-centring, the library mean is the zero vector, so "seed the taste vector
from the library centroid" — the original audit's suggestion — is degenerate. See L7 for the
replacement.

---

## 3 · High-severity findings

---

### H1 · The mood word in every vibe description is mathematically pinned to "eclectic". `OPEN — fix superseded`

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

**6. The readout.** Top-3 descriptors by z-score, for the session vector:

```
♪ hypnotic · nocturnal · sparse                    ⟳ drifting   ·   14 tracks · 52 min
```

The consistency word — replacing the deleted momentum thresholds (D4) — becomes the cosine between
the session's descriptor z-vector now and five tracks ago. That is a measurement of "how much has the
character of this session moved," which is what the word was always trying to say, and its thresholds
can be calibrated against observed drift rather than invented.

#### TUI consequences

The vibe readout is the payoff of D1 — it replaces the queue panel as the window into what the system
is doing. The layout changes accordingly:

- **Delete the "Upcoming Queue" panel.** There is nothing to list.
- **Add a "Session" panel** in its place: one `↓ next: <artist — title>` line at the top, a divider,
  then session history newest-first with feedback marks (`♥` liked, `⏭` skipped, `✓` full listen).
  This is what you actually wanted visibility into — it is truthful, because it happened.
- **Repurpose `↑↓` + `ENTER`.** They currently index into the queue list and replay the session's first
  track (H2). Rebind to: scroll history, `ENTER` on a history entry **requeues that track as `next`**.
  A "play that again" action the app currently lacks, reusing existing plumbing.
- **Repurpose `[I]`.** With time context gone (D6) the overlay has no content. Make it the model
  inspector: top descriptors for the **session** vector *and* the **taste** vector side by side,
  current exploration value, tracks played, taste update count. That is the honest answer to "what
  does the system think it has learned," and it reuses the descriptor bank.
- **L3 becomes a blocker.** Album-art geometry is hardcoded to `RIGHT_COL_ROWS = 10`, the exact row
  count of the current Now-Playing pile (`tui.py:113–131, 563–602`). The vibe line is changing shape,
  so this constant is about to be wrong. See L3.
- **Update the footer and the README keybinding table together** — they already disagree (L9).

---

### H2 · The "Upcoming Queue" panel lists tracks that already played. `DISSOLVED by D1`

**What.** Same root cause as C1. `QueueManager.get_upcoming_tracks()` returns the whole MPD playlist,
so the panel shows the session's history above the current track, numbered as if it were the future.
The numbering counts from the top of the queue rather than from the current song, and `↑↓` navigation
plus `ENTER`-to-play index into that same list — so selecting "1." plays the first track of the
session again.

**Status.** The panel is being removed (H1, TUI consequences). `get_upcoming_tracks()` is deleted
along with it. No fix required — but the `ENTER` misbehaviour is a reminder to re-derive the new
history panel's indices from scratch rather than porting the old ones.

---

### H3 · Ctrl-C and SIGTERM neither exit nor save. `OPEN — now also affects MPD state`

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

`ExplorationController.consecutive_skips` remains tracked and is now genuinely unused. Either delete
it or keep it purely for the `[I]` inspector; do not invent a use for it.

---

### H5 · The day-of-week exploration modifier is dead code. `RESOLVED by deletion (D6)`

**What.** `ExplorationController.get_exploration_factor()` is the only place the weekday/weekend
modifier is applied. Nothing in the running application ever calls it — a grep across all source finds
references only inside test files. `get_weights()`, the method actually used for scoring, reads raw
`self.exploration` and ignores the modifier entirely. The README documents the feature as live.
`test_phase3.py` tests the modifier in isolation and passes, which is how it went unnoticed.

**Where.** `exploration_controller.py:71–90` (uncalled), `exploration_controller.py:92–128` (used);
`config.py:89–91`

**Status.** Resolved by removing the time-context subsystem (D6). Delete
`get_exploration_factor()`, the two config keys, and the README paragraph.

---

### H6 · Selection is strictly greedy — "exploration" never actually explores. `OPEN — ELEVATED to blocker`

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

30-track sessions simulated from an identical starting state on the real 616-track library, centred
space:

| Selection rule | mean pairwise similarity within the session | distinct sessions from one start state (5 runs) | mean overlap with run #1 |
|---|---|---|---|
| argmax (current) | 0.603 | **1 / 5** | 100% |
| rank-Boltzmann τ=1 | 0.596 | 5 / 5 | 42% |
| rank-Boltzmann τ=7 | 0.599 | 5 / 5 | 53% |
| rank-Boltzmann τ=15 | 0.570 | 5 / 5 | 31% |
| *(library baseline: two random tracks)* | *0.014* | | |

Two things worth reading carefully:

- **Within-session coherence is unaffected** — 0.603 → 0.599 at τ=7, against a library baseline of
  0.014. The session stays as musically tight as argmax makes it. Sampling does *not* buy diversity
  inside a session, and it should not be sold as though it does.
- **Run-to-run variety is the whole gain.** Argmax returns the byte-identical 30 tracks from the same
  state, every time; sampling returns 31–53% overlap. For a system whose premise is an evolving
  session, reproducing the same evening from the same starting point is the failure mode — and it is
  the one argmax guarantees.

The ordering among τ values is within noise at five runs; **τ_max ≈ 15 is a starting point, not a
finding.** Calibrate it in use: raise it until unattended sessions start feeling incoherent, then back
off. It is the one genuinely new constant this plan introduces, and it is flagged here so it does not
quietly become another uncalibrated threshold of the kind C5 and H1 document.

---

### H7 · `mpd_controller.py` defines eight methods twice; the surviving `add_track` swallows failures. `OPEN`

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

### H8 · Album-art geometry is hardcoded to a layout that is about to change. `ELEVATED from L3`

*See L3 for the full text.* Promoted because the H1 TUI rework changes the Now-Playing pile's row
count, which is exactly the constant `_render_art()` pins the image against. Fix it as part of the
rework, not after.

---

### H9 · `[V]` is a smaller gesture than `[N]`×5, in a direction that is not music. `NEW — resolved by deletion`

**What.** `SessionState.force_shift()` blends the session vector 50% toward a **random 512-dimensional
direction** (`session_state.py:107–127`). The stated purpose of `[V]` is a decisive change of
direction. It is measurably neither decisive nor a direction.

**Problem 1 — the destination is not music.** In 512 dimensions a random unit vector is
near-orthogonal to every real embedding (expected cosine 0, σ ≈ `1/√512` ≈ 0.044). Measured against
this library: an ordinary session vector sits at **0.697** mean similarity to its 25 nearest tracks; a
random direction sits at **0.105**. `[V]` blends half of the second thing into the first. The candidate
pool is then drawn nearest to a point that is largely meaningless, so `[V]` does not mean "different
vibe" — it means "weaker vibe, plus noise."

**Problem 2 — it moves less than the constant implies.** With `r ⊥ v`, `|0.5v + 0.5r| = 0.5√2`, so
`cos(v', v) = 0.707` — measured at 0.705–0.712 across 40 simulated sessions. Against this library,
where two *unrelated* tracks average 0.577 and p75 is 0.738, the "hard reset" leaves the new direction
closer to its own former self than ~70% of random track pairs are to each other.

**Problem 3 — and this is the one that decides it.** Measured as **candidate-pool turnover**, the only
thing that determines what you actually hear next:

| Action | cos(new, old) | pool turnover |
|---|---|---|
| `[N]` ×1 | 0.995 | 3.9% |
| `[N]` ×3 | 0.977 | 7.2% |
| `[N]` ×5 | 0.948 | **11.9%** |
| **`[V]` ×1** (`force_shift` 0.5) | 0.710 | **7.4%** |
| `[N]` ×10 | 0.818 | 23.1% |
| `[N]` ×20 | 0.474 | 58.1% |

*(centred space, 40 simulated sessions, top-100 pool; the raw-space figures are within a point or two)*

**`[V]` changes less of what you will hear than pressing `[N]` five times.** It is not a bigger
gesture than the skip key — it is an equally small one pointed at noise. The large `cos` figure is what
made it *look* decisive; turnover is what the listener experiences, and by that measure it barely
registers.

**Where.** `session_state.py:107–127`; `config.vibe_shift_magnitude`;
`feedback_handler.process_vibe_skip()`; `exploration_controller.set_high_exploration()`; `tui.py:364`

#### Resolution — delete `[V]`; make `[N]` escalate

`[V]` was introduced alongside the ten-track queue, where its real job was *clearing the queue*: with
ten tracks committed in advance, rejecting a direction meant ten presses of `[N]`, so a queue-nuking
key earned its place. D1 removes the queue, and with it the only thing `[V]` did that `[N]` could not.
The vector mathematics it carried was never sound. It is a queue-era affordance and it goes.

But the *intent* behind it — "the problem is the direction, not this song" — is real, and the table
above shows `[N]` does not currently serve it either: 3.9% turnover per press, and 20 presses to reach
58%. Neither key delivers a change of direction today. So `[N]` is rebuilt to cover both.

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

   | consecutive skips | turnover target | measured λ | reads as |
   |---|---|---|---|
   | 1 | 5% | ~0.23 | "not this song" |
   | 2 | 20% | ~0.55 | "not this corner either" |
   | 3 | 50% | ~0.80 | "this is the wrong direction" |
   | ≥4 | 85% | ~1.05 | full reset |

   λ is found by increasing it until the target is met — a few dot-product passes over 616 vectors,
   microseconds. The current code uses a fixed λ = 0.15 at every step, which is why twenty presses were
   needed to get anywhere.

4. **Project back onto the manifold** for escalated skips (*n* ≥ 2, where λ is large enough to matter):
   replace the vector with the normalised centroid of its 25 nearest real embeddings.

   ```
   snap(v) = normalise( mean( top-25 library embeddings by dot(E, v) ) )
   ```

   Measured: at the 85% target this preserves the turnover almost exactly (87.2% → 86.9%) while holding
   on-manifold quality at 0.656 — against 0.697 for an ordinary session vector and 0.105 for a random
   direction. **This is a structural guarantee, not a tuning:** however large λ grows, the session
   vector cannot leave the region your music occupies. It is the general fix for the entire class of
   bug this finding describes, and it costs one 616×512 matrix-vector product.

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

### M1 · The test suite is green theatre, and it is not in the repository. `OPEN`

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

---

### M2 · Two divergent orchestrators; the stale one is what the setup helper tells you to run. `OPEN — resolution changed to deletion`

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

### M3 · `mpd_music_directory` is an undocumented, unvalidated requirement that works here by accident. `OPEN`

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

> **Fix direction.** Read it from MPD instead of guessing — `mpc --verbose` or the MPD config — and
> fall back to prompting in `start.sh`. Validate at startup by resolving one known track path and
> refusing to launch (with a clear message) if it does not exist. Do this **before** the C3
> regeneration run, not after — a wrong music dir wastes the whole run.

---

### M4 · Two different sources of truth for track keys, with no reconciliation. `OPEN`

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

### M5 · No embedding-dimension validation on load. `OPEN — scope widened by C3`

**What.** `TrackLibrary.load_embeddings()` never checks `embeddings.shape[1]` against
`config.embedding_dimension` (512). Meanwhile `UserTaste` and `SessionState` size their vectors from
the config value. Loading a file generated by a different model surfaces as a numpy broadcast error
deep inside `_calculate_score` rather than as "these embeddings are 768-d, expected 512-d".

Related: `TrackLibrary.save_embeddings()` writes only `track_files` and `embeddings`, dropping the
metadata block. A CLAP file round-tripped through it is silently downgraded to "placeholder
embeddings" by the loader's own warning logic on the next run.

**Where.** `track_library.py:26–82`, `track_library.py:231–247`; `config.py:64`

> **Fix direction.** Validate shape on load and adopt the file's dimension rather than the config's —
> the embeddings are the authority. **The `.npz` schema is changing** (C3 adds the per-window matrix,
> C5 adds the centroid), so add a `schema_version` key and refuse to load anything that lacks the
> centroid rather than silently scoring on an uncentred space. Carry metadata through
> `save_embeddings` — or delete `save_embeddings`, which nothing calls (L8).

---

### M6 · State file misnamed; anti-repetition history never persisted despite a comment claiming it is. `PARTIALLY RESOLVED`

**(a) `RESOLVED by D6.`** `config.context_file` is named `time_context.npz`, but `TimeContext.save()`
writes JSON. Loading it as an npz raises `UnpicklingError: invalid load key, '{'` — verified against
the live file. Resolved by deleting the time-context subsystem; delete the file and the config key
rather than renaming.

**(b) `OPEN — now user-visible.`** `TrackSelector.clear_history()` carries the comment *"Don't clear
play_history to maintain long-term anti-repetition"* — but nothing ever saves or loads `play_history`,
and `clear_history()` itself is never called. Both `recent_history` and `play_history` are rebuilt
empty on every launch, so the README's "Recently played tracks are excluded for at least 20 songs"
resets each time you start the program.

This becomes visible rather than merely wrong once the session history panel exists (H1): the panel
shows what played, and the user will notice a track reappearing that the panel says played twenty
minutes ago.

**Where.** `config.py:94` vs `time_context.py:272–283`; `track_selector.py:198–202`

> **Fix direction.** Persist `play_history` and `current_index` alongside the other state, checkpointed
> per H3. Delete `clear_history()` and its misleading comment.

---

### M7 · Setup documentation contradicts itself on every number, and macOS is claimed but unsupported. `OPEN — scope widened`

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
> by the ~30 MB window matrix). Replace `${VAR,,}` with `tr '[:upper:]' '[:lower:]'`. Either test on
> macOS or drop the claim. **The README needs a rewrite regardless** — D1, D6 and H1 change the
> described behaviour of the queue, the time-context feature, and the vibe display.

---

### M8 · `--batch-size` is advertised, accepted, and completely ignored. `OPEN — ELEVATED to prerequisite`

**What.** `CLAPEmbeddingGenerator.__init__` stores `batch_size` and `generate_embeddings.py` exposes
`--batch-size` with a documented default of 16, printing it in the run banner.
`generate_embeddings_batch()` then processes tracks strictly one at a time in a Python loop. On a GPU
this leaves most of the throughput unused — the recorded run averaged 3.1 tracks/sec on an RTX 3070.

Minor adjacent issues in the same file: `stats['successful'] / stats['duration']` divides by zero if
generation completes instantly, and on `--resume` the successful counter is pre-seeded with the
resumed count so the final summary overstates this run's work.

**Where.** `embedding_generator.py:46–68`, `embedding_generator.py:272–376`;
`generate_embeddings.py:339–344`

> **Fix direction.** C3's full-coverage windowing produces ~24 windows per track — roughly 15,000
> forward passes for the library instead of 616. Running those one at a time is not viable, so
> batching stops being an optimisation and becomes a prerequisite for C3. Batch at the *window* level,
> not the track level: fill a batch from whichever tracks' windows are ready, so short and long tracks
> pack evenly.

---

## 5 · Lower-severity findings and friction

---

### L1 · The SIGWINCH handler is almost certainly never invoked. `OPEN`

`tui._setup_urwid()` installs `_on_sigwinch` at line 310, during `__init__`. urwid's `MainLoop` /
`raw_display.Screen.start()` installs its own SIGWINCH handler afterwards, when `loop.run()` is called
— replacing it. So `force_redraw()` on resize or monitor-move likely never fires, which is exactly the
bug the handler's comment says it exists to prevent. Register through urwid's screen hooks, or
re-install after `loop.run()` starts.

---

### L2 · Kitty and sixel album art fight urwid for the screen; only ueberzug works. `OPEN`

Both `KittyProtocol.render` and `SixelProtocol.render` write escape sequences straight to
`sys.__stdout__` while urwid owns the terminal. urwid's next full redraw — every 0.5 s — paints over
them. The ueberzug/ueberzugpp overlay protocols work because they draw in a separate X11/Wayland
surface. Detection order puts ueberzug first, so this mostly hides, but the kitty and sixel branches
are effectively non-functional and should be marked as such or removed. `album_art.py:234–325`

---

### L3 · Album-art geometry is hardcoded to the current layout. `ELEVATED to H8`

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

### L4 · Hearts vanish on restart even though likes are already on disk. `OPEN — folded into H1`

`TUI.liked_tracks` is an in-memory set, populated only by pressing `[L]` during the current run. Every
like is already recorded in `feedback_history.json` with a track path, so the set could be rehydrated
at startup in three lines. `tui.py:144–145`, `feedback_handler.py:160–171`

**Status.** The new session-history panel (H1) needs per-track feedback marks anyway, so rehydration
from `feedback_history.json` becomes part of building it rather than a separate fix.

---

### L5 · No log file, and stderr is swallowed while the TUI runs. `OPEN — ELEVATED to M`

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

### L7 · Cold start injects a random direction at 30% weight. `OPEN — fix changed by C5`

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

### L8 · Dead API surface across most modules. `OPEN`

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

- **`select_track` mutates its caller's set.** `exclude_tracks.update(self.recent_history)` modifies
  the set `QueueManager._generate_tracks` passes in. Harmless today, a trap later.
  `track_selector.py:44–47` — `OPEN`
- **`config.validate()` is all `assert`.** Under `python -O` every check vanishes, including the
  weight-sum-to-1.0 invariant. `config.py:101–117` — `OPEN`
- **`.gitignore` intent not achieved.** It excludes `data/state/` and `data/embeddings/` as
  directories, then tries to re-include `!data/state/.gitkeep` — which git cannot do inside an
  excluded directory. `git ls-files data` returns only `data/.gitkeep`. Harmless because
  `Config.__init__` mkdirs them, but the scaffolding does not ship. — `OPEN`
- **Time-context bonus breaks the weight invariant.** `config.validate()` enforces that the four
  weights sum to 1.0, then `_calculate_score` adds `0.15·time_sim` on top. — `RESOLVED by D6`; the
  invariant becomes true again, so it is worth actually enforcing outside `assert`.
- **`[I]` silently does nothing when `enable_time_context` is false.** `tui.py:427–432` —
  `RESOLVED by D6/H1`; `[I]` is repurposed and always has content.
- **Orphaned ueberzugpp process.** `_shutdown` calls `clear()` but never terminates the child; cleanup
  relies on `__del__` firing at interpreter exit, which is not guaranteed. `album_art.py:118–124` —
  `OPEN`, and it shares a root cause with H3: `_shutdown` is unreachable on SIGTERM. Fix both together.
- **Keybinding docs disagree.** README lists volume as `,` / `.`; the footer reads `Vol - [<,>]`. In
  the non-urwid fallback mode, `↑↓` are bound to volume rather than queue navigation, contradicting
  both. — `OPEN`, folded into the H1 footer/README update.

---

## 6 · Findings status after the revision

| ID | Finding | Status |
|---|---|---|
| **C1** | Queue never refills | **Open** — fix simplified by D1/D2 (consume mode, no position parsing) |
| **C2** | MPD random mode discards ordering | **Open** — expanded to also force `consume on`; restore now depends on H3 |
| **C3** | Non-deterministic 10 s crop embeddings | **Open** — fix now fully specified (full-coverage windows, persisted matrix) |
| **C4** | `[V]` double-advance | **Dissolved** by D8 — the code path is deleted. Its constraint (one advance per keypress, no `play()` in a skip path) is retained and tested |
| **C5** | Compressed similarity scale (anisotropy) | **New** — added in revision; blocks meaningful tuning of everything else |
| **H1** | Mood word pinned to "eclectic" | **Open** — original fix superseded by the CLAP descriptor bank; drives the TUI rework |
| **H2** | Queue panel shows history as future | **Dissolved** by D1 — panel removed |
| **H3** | Ctrl-C/SIGTERM neither exit nor save | **Open** — now also leaves MPD in consume mode |
| **H4** | Skipping doesn't change what plays next | **Dissolved** by D1 — one lookahead track, dropped and re-picked |
| **H5** | Day-of-week modifier is dead code | **Resolved by deletion** (D6) |
| **H6** | Strictly greedy selection | **Open — elevated to blocker.** D1 cannot ship without it. Fix changed: rank-Boltzmann, not score-softmax |
| **H7** | Eight duplicate methods; `add_track` swallows failures | **Open** — load-bearing for D1 |
| **H8** | Album-art geometry hardcoded | **Elevated from L3** — blocks the TUI rework |
| **H9** | `[V]` turns over less pool than `[N]`×5, pointing off-manifold | **New — resolved by deletion.** `[V]` goes; `[N]` escalates on consecutive skips instead |
| **M1** | Test suite is green theatre, untracked | **Open** — phase tests to be deleted, not repaired |
| **M2** | Two divergent orchestrators | **Open** — resolution changed to deletion (D7) |
| **M3** | `mpd_music_directory` unvalidated | **Open** — must land before the C3 regeneration run |
| **M4** | Two sources of truth for track keys | **Open** — `mpc listall` becomes the single source |
| **M5** | No embedding-dimension validation | **Open** — widened: the `.npz` schema is changing |
| **M6a** | `time_context.npz` is JSON | **Resolved by deletion** (D6) |
| **M6b** | `play_history` never persisted | **Open** — becomes user-visible with the history panel |
| **M7** | Contradictory setup docs; macOS unsupported | **Open** — README needs a rewrite regardless |
| **M8** | `--batch-size` ignored | **Open — elevated to prerequisite** for C3 |
| **L1** | SIGWINCH handler never invoked | Open |
| **L2** | Kitty/sixel art fights urwid | Open |
| **L3** | Album-art geometry hardcoded | **Elevated → H8** |
| **L4** | Hearts vanish on restart | Open — folded into H1's history panel |
| **L5** | No log file; stderr swallowed | **Elevated to M** — do it first |
| **L6** | Polling instead of `mpc idle` | Open — still low priority at depth 1 |
| **L7** | Random cold-start taste vector at β=0.3 | Open — fix changed by C5 (ramp β from 0) |
| **L8** | Dead API surface | Open — delete, except three items worth wiring |
| **L9** | Assorted small traps | Mixed — two resolved by D6, rest open |

---

## 7 · Target data artifacts

Build to these. Both are produced by the generation run in Stage 1 and are the contract everything
downstream reads.

### `data/embeddings/track_embeddings.npz`

| Key | Shape / type | Notes |
|---|---|---|
| `schema_version` | `int` | Start at `2`. Refuse to load `< 2` — a v1 file has no centroid and would be scored on an uncentred space (C5). |
| `track_files` | `(N,)` unicode | Keys as returned by `mpc listall` — the single enumeration source (M4). |
| `embeddings` | `(N, 512)` float32 | Pooled, **uncentred**, L2-normalised. Keep raw so the centroid can be recomputed if the library grows. |
| `centroid` | `(512,)` float32 | `mean(embeddings, axis=0)`. Applied at load: `normalise(E − centroid)` (C5). |
| `window_offsets` | `(N+1,)` int32 | CSR-style index into `windows` — track *i* owns `windows[offsets[i]:offsets[i+1]]`. Tracks have different window counts. |
| `windows` | `(ΣW, 512)` float32 | Per-window embeddings, L2-normalised. ~30 MB. Lets pooling be re-decided without regenerating (C3). |
| `metadata` | dict | Model name, transformers version, date, device, window scheme (`length_s`, `hop_s`, `rms_gate`), timing. Must survive round-trips (M5). |

Also written: `data/embeddings/failed.txt` — one line per file that failed, with the exception. The
original run lost 16 tracks silently (C3).

### `data/embeddings/descriptors.npz`

Generated after the embeddings, in the same run.

| Key | Shape / type | Notes |
|---|---|---|
| `schema_version` | `int` | |
| `labels` | `(D,)` unicode | The descriptor words, post-validation — near-zero-variance ones already dropped (H1). |
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

---

### Stage 0 — Clear the ground *(an hour, mostly mechanical)*

| ID | Change | Why first |
|---|---|---|
| **L5** | Tee the console ring buffer to `data/dj.log` | Ten minutes, and everything after this is a refactor you will need to debug through a 5-line panel otherwise. |
| **D6 / H5 / M6a** | Delete `time_context.py`, `enable_time_context`, `time_context_weight`, the weekday/weekend modifiers, `get_exploration_factor()`, `time_context.npz` | Removes four findings and restores the weight-sum-to-1.0 invariant. |
| **D4** | Delete `get_vibe_description()`'s entropy mood word, momentum thresholds and stage word; delete the random taste seed | Nothing after this should be built on them. Leave the vibe line blank until Stage 3 — a blank line is honest, the current string is not. |
| **M2 / D7** | Delete `main.py`, `setup_check.py`, and `start.sh`'s demo-embedding path | Stops every later fix from needing to be applied twice. |
| **H7** | Delete the eight duplicate methods in `mpd_controller.py`; restore the `add_track` return-code check | Makes MPD failures visible — load-bearing for Stage 2, where a swallowed `add_track` means silent stall. |
| **L8** | Delete the dead API surface | Smaller surface to refactor. Keep `previous_track` and `save_embeddings` for now — see L8's exceptions. |
| **M1a** | Remove `test_*.py` from `.gitignore`; delete the three phase test files | They test a structure being replaced. Green currently means nothing; delete rather than repair. |
| **cfg** | Prune `config.py` of every key the above orphans; convert `validate()` off bare `assert` (L9) | Dead config keys are how H5 hid for months. |
| **D3** | Delete `data/embeddings/*` and `data/state/*` | No value to preserve; removes the "reset state carefully" caveat from every later step. |

**Done when:** the app launches, plays, and shows an empty vibe line; `git ls-files` includes a tests
directory; `grep -ri time_context` returns nothing; `data/dj.log` fills during a session.

---

### Stage 1 — Rebuild the signal *(half a day plus a regeneration run)*

Nothing about selection quality can be judged until the vectors mean something. Build to the schemas
in §7.

| ID | Change | Notes |
|---|---|---|
| **M3** | Detect `mpd_music_directory` from MPD (`mpc --verbose` / MPD config); prompt in `start.sh` as fallback; refuse to launch if a known track path will not resolve | **Before** the regeneration run — a wrong music dir wastes the whole thing. |
| **M4** | `mpc listall` as the single enumeration source; log embedding↔MPD coverage on load; refuse below a threshold | Verifies the regenerated file matches what MPD will actually play. |
| **C3** | Full-coverage deterministic 10 s windows, end-aligned tail, RMS-gated, mean-pooled; persist the per-window matrix and the failed-track list | The change everything else sits on. |
| **M8** | Batch at the **window** level, not the track level | Prerequisite, not an optimisation — ~24× the forward passes. Window-level packing keeps short and long tracks even. |
| **C5** | Compute and store the library centroid; centre + re-normalise on load | Do it *with* C3 so the `.npz` carries the centroid from day one. |
| **D5a** | Build the descriptor bank: ~50 prompts, text-tower embeddings, per-descriptor `mean`/`std` over the centred library | Data artifact, not UI. Both H1 and H9 consume it. |
| **D5b** | **Validation gate** — drop any descriptor whose `std` over the library is below a floor; log what was dropped and what survived | **This is the anti-trap step.** See below. |
| **M5** | Validate `schema_version`, dimension and required keys on load; adopt the file's dimension over the config's | Guards the new schema against silent mismatch. |

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
distribution must sit near 0 rather than 0.577. All three are one-line assertions and all three belong
in the test suite.

---

### Stage 2 — Play continuously, one track ahead *(a few hours)*

D1 and H6 are one change and must land together. Write the test harness **first** — this is the stage
where the original audit's four critical defects lived, all under a green suite.

| ID | Change | Notes |
|---|---|---|
| **M1b** | pytest scaffold + a `FakeMPD` modelling real semantics **including consume mode** | First, not last. Every item below gets a behavioural test as it lands. |
| **C2** | Force `random`/`repeat`/`single` off and `consume` **on**; log each change to the console panel; restore originals on exit | Enables the simplified C1. |
| **H3** | Signal handling that exits, saves, and restores MPD modes; `atexit` hook for the mode restore; periodic state checkpoint | Must land **with** C2 — otherwise an abnormal exit strands the user's MPD in consume mode. |
| **C1 / D1** | `lookahead = 1`; rewrite `QueueManager` to `ensure_one_ahead()` / `replace_next()` | Delete `planned_queue`, `currently_queued_in_mpd`, `_sync_to_mpd`, `get_upcoming_tracks`, `recalculate`, `initialize_queue`, the 5% trajectory blend, and `queue_low_threshold`. |
| **H6** | Boltzmann sampling over **rank**: `p(i) ∝ exp(−i/τ)`, τ linear in the exploration scalar | **Blocker for D1.** Rank-based, not score-based — scale-invariant, so it survives C5 and every weight change without recalibration. |
| **C4** | Delete `recalculate()`; one skip path: *adjust vector → replace lookahead → advance once* | No `play()` call anywhere in it, so the double-advance cannot recur by construction. |
| **H9 / D8** | Delete `[V]`, `force_shift()`, `process_vibe_skip()`, `set_high_exploration()`, `vibe_shift_magnitude`. Rebuild `[N]`: repel from the skip-run centroid, λ **solved** for an escalating turnover target (5/20/50/85% by run length), then `snap()` to the 25-NN centroid for n≥2 | Measured: `[V]` turned over less pool than `[N]`×5. The `snap()` guard makes leaving the manifold structurally impossible. |
| **H4-repl** | `[N]` drops and re-picks the lookahead under the new weights | Falls out of D1 — skips become audible on the very next song. |
| **L7** | Ramp β from 0 over the first ~20 taste updates; delete the random taste seed; make the random session seed the explicit "nothing playing yet" case | Same theme as C5: stop injecting noise as if it were signal. |
| **M6b** | Persist `play_history` / `current_index`; checkpoint with the rest of the state | Anti-repetition finally survives a restart. |

**Done when:** a 30+ track unattended run completes with no stall and no repeat inside the replay gap.
Two runs from the same starting state produce materially different track sets (argmax produces
identical ones — see H6). One `[N]` visibly changes the next track; four consecutive `[N]` presses
audibly change the *kind* of music, and the reported turnover exceeds 80%. `mpc status` after a
`kill -TERM` shows the user's original modes restored.

---

### Stage 3 — Make it legible *(half a day)*

The stage where the queue's original purpose finally gets served. All display; the data it reads was
built in Stage 1.

| ID | Change | Notes |
|---|---|---|
| **H8 / L3** | Derive album-art geometry from the widget tree rather than hand-counted constants | **First** — the layout changes below, and the art is pinned to `RIGHT_COL_ROWS = 10`. |
| **H1b** | Vibe readout = top-3 descriptors by z-score; consistency word = cosine between the session's descriptor z-vector now and five tracks ago | Replaces the deleted heuristics with something measured against a real distribution. |
| **H1c** | Replace the queue panel with a **Session** panel: `↓ next:` line, divider, history newest-first with `♥` / `⏭` / `✓` marks | The actual answer to "what is being played". |
| **L4** | Rehydrate `liked_tracks` from `feedback_history.json` at startup | Falls out of building the history panel. |
| **H1d** | Rebind `↑↓` to scroll history and `ENTER` to requeue a history entry as `next`; repurpose `[I]` as the model inspector | `[I]` shows session **and** taste top descriptors, exploration value, effective τ ("choosing from ~top 7"), tracks played, taste update count. |
| **L9** | Reconcile the footer, the README keybinding table, and the fallback-mode bindings | They already disagree three ways; do it while the bindings are open. |

**Done when:** the vibe line names three descriptors that a listener would recognise as accurate, and
`[I]` explains what the machine currently believes without any invented vocabulary.

---

### Stage 4 — Make it durable *(half a day)*

| ID | Change | Why |
|---|---|---|
| **M1c** | Broaden the suite: scoring, persistence round-trips, embedding determinism, descriptor-gate regression, skip-escalation turnover | C1 and C4 both survived under a green suite. Until the tests are behavioural, green means nothing. |
| **M7** | Rewrite the README against actual behaviour; re-measure model cache size, generation time and disk requirement | Three described features no longer exist as described; the numbers contradict each other three ways today. |
| **L1** | Re-install the SIGWINCH handler after `loop.run()` | urwid replaces it at startup. |
| **L2** | Remove or explicitly mark the kitty/sixel art paths as non-functional | They fight urwid for the terminal and lose every 0.5 s. |
| **L9** | Fix the `.gitignore` `.gitkeep` scaffolding; terminate the ueberzugpp child in `_shutdown`; stop `select_track` mutating its caller's set | The ueberzugpp leak shares a root cause with H3 — fix them together. |
| **L6** | *(optional)* Move to `mpc idle` | Only worth it against a remote `MPD_HOST`. At depth 1 the 2 s poll has minutes of runway. |

**Done when:** a fresh clone runs the suite green, and green means something.

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
| **Delete `[V]` (D8)** | Measured: 7.4% pool turnover versus 11.9% for `[N]`×5, in a direction 0.105-similar to real music. Its queue-clearing job no longer exists. | Low, but do not restore `force_shift`. If a distinct "change subject" gesture is wanted later, build it as a named-descriptor jump, not a random rotation. |
| **Rank-Boltzmann sampling (H6)** | Argmax reproduces the byte-identical session from a given state. Score-softmax needs recalibration every time the score scale moves. Rank is scale-invariant. | Trivial — one function. |
| **`ENTER` on history requeues (H1d)** | Replaces the removed queue navigation with something useful, reusing existing plumbing. | Trivial. If unwanted, `↑↓` becomes pure scrolling and `ENTER` unbinds. |

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
   against the compressed similarity scale and will behave differently once centred. The plan is to
   leave them, listen, then tune once with real data rather than guess new numbers now. The constants
   that made *claims* are already deleted (D4); these only shape behaviour.

4. **τ_max ≈ 15.** The one genuinely new constant. Documented in H6 with the measurement that motivated
   it and an explicit note that it is a starting point requiring calibration in use, so it does not
   become the next uncalibrated threshold.

5. **`previous_track`.** Impossible via MPD under `consume on`; would need re-adding from the app's own
   history. No binding exists today, so nothing regresses.

---

## 10 · Evidence appendix

*Reproduction commands and raw measurements for the empirical claims.*

### C3 — embedding non-determinism

Ten embeddings of one synthetic three-minute signal through the project's own
`CLAPEmbeddingGenerator`, compared to 300 random library tracks:

```
SELF-SIMILARITY of ONE track embedded 10x:
  min 0.354  median 0.884  max 0.998  mean 0.740
LIBRARY inter-track similarity: mean 0.577  median 0.582
fraction of self-pairs BELOW library median: 36%

# root cause — transformers 5.1.0
ClapFeatureExtractor defaults: truncation = fusion, max_length_s = 10
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

$ ls -la /var/lib/mpd/music          # config.py default
lrwxrwxrwx. root root -> /mnt/storage/music   # works only by symlink

$ echo $MPD_MUSIC_DIR                 # unset
```

### C5 — the compressed similarity scale

Derived from the library distribution below. Two unrelated tracks sit at median cosine **0.582**; the
1st percentile is 0.074. The scoring code's `(sim + 1) / 2` normalisation therefore produces values in
roughly `[0.54, 0.98]`, and `novelty = (1 − max_sim) / 2` clusters near 0.21. Centring the space is
what restores the range these formulas assume.

### H9 / D8 — `[V]` versus `[N]`, measured

40 simulated sessions on the real 616-track library, driven through the project's own EMA update and a
session+taste scoring approximation. "Turnover" = fraction of the top-100 candidate pool that changed.

```
                                   cos(new,old)   pool turnover
  [N] x1   (penalize_similar 0.15)      0.995          3.9%
  [N] x3                                0.977          7.2%
  [N] x5                                0.948         11.9%
  [V] x1   (force_shift 0.5 random)     0.710          7.4%     ← smaller than [N] x5
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
    random direction        : 0.105     ← what force_shift blends 50% of
```

`snap()` preserves the turnover at the target that matters (87.2% raw → 86.9% projected) while keeping
on-manifold quality comparable to an ordinary session vector. Current code uses a fixed λ = 0.15, which
is why twenty presses were needed to reach 58%.

### H6 — argmax versus rank-Boltzmann sampling

30-track sessions from an identical starting state, centred space:

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
  library random-pair similarity
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
$ python3 test_phase2.py             → exit 1   ("Passed: 66, Failed: 1")
$ python3 test_phase3.py             → exit 0   (9 passed)
$ python3 test_phase3_integration.py → exit 1   (4 passed, 1 failed)

$ git ls-files | grep test           → (nothing — .gitignore excludes test_*.py)
```

### Subprocess overhead — measured, not assumed

```
one `mpc status` subprocess: 0.8 ms
steady state ≈ 12 mpc calls/sec → ~1% of one core

# a real cost only against a remote MPD_HOST, where each call is a TCP round trip
```

### Projected cost of the C3 rebuild

```
616 tracks × ~24 non-overlapping 10 s windows ≈ 14,800 forward passes
storage: 616 × 24 × 512 × float32 ≈ 30 MB for the per-window matrix
audio decode: 616 files, once each — dominated the original 198 s run
expected wall clock, batched, RTX 3070: low single-digit minutes
```

---

*All line references are against `HEAD 8dc4275` plus the uncommitted `tui.py` change (footer key-hint
rewording, +10/−10, cosmetic). Measurements were taken on the machine described in the header; the
embedding-determinism figures use the project's own code path against the cached
`laion/clap-htsat-unfused` weights.*
