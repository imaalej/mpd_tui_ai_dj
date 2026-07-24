# Independent Inspection — Observations

An outside review of the post-rewrite codebase (branch `rewrite/stages-0-4`), read against
`project_state.md` but not bound by it. Scope: correctness, architecture, and drift from the
project's own stated spirit ("derive constants from the data; an observation is not a conclusion;
never leave the user's machine changed; the display tells the truth"). Tests are green (542 passed,
~24 s) at time of writing.

These are **observations**, not verdicts. Each is a place a future planning phase should look, with a
concrete scenario and a file:line anchor. Severity is this inspector's estimate of blast radius, not a
decision.

---

## A. Concurrency — insurance, not a necessity

**A1 · The two threads share mutable state without a lock — but correctness does not depend on one.**
`main_tui.py:308` runs `_background_event_loop` as a daemon thread that calls
`queue_manager.ensure_one_ahead()` → `track_selector.select_track()` (mutating `recent_history`,
`play_history`, `current_index`) and processes full-listens (mutating `session_state`, `user_taste`,
`exploration_controller`, `feedback_history`). The input thread mutates the *same* objects via
`skip_current_track()` / `replace_next()` / `requeue_next()` and the like/unlike path. The only locks
in the app are `_restore_lock` (mode restore, `main_tui.py:60`) and the console ring buffer
(`tui.py:86`).

The reason this is filed as low priority rather than major:
- CPython's GIL makes the individual `deque`/`dict`/reference operations atomic, so no structure is
  corrupted into a crash. The worst a race does to `current_index += 1` (`track_selector.py:269`) is
  *lose* one increment — a one-track anti-repetition drift that self-corrects.
- The one fatal outcome (advance into an empty queue → MPD stops silently, the C4 failure) is
  prevented by construction, not by timing: `skip_current_track` advances only after
  `replace_next()` returns non-None, which happens only after a successful `add`. Add-before-advance
  holds under any interleaving.
- The realistic residual race is a queue briefly going `1 + queue_lookahead + 1` deep when the 2 s
  refill and a `[N]` press both add at a boundary. The extra lookahead is scored under near-current
  vectors and consumed normally; the next refill no-ops.

So a single lock around selector/queue mutation is **cheap insurance and makes the behaviour
deterministic under test**, but it is not required for the app to be correct. Recommended only if a
real misbehaviour is observed, or opportunistically. Severity: **low** (was over-weighted in the first
pass).

---

## B. Feedback / playback correctness

**B1 · The full-listen threshold is 90 %, which is stricter than intended — reduce it to 75 %.**
`main_tui.py:356` computes `completion_threshold = max(0.9·duration, duration − 10)`. Requiring 90 % of
a track before it counts as "listened through" is too high; a track heard for three-quarters of its
length is a genuine full listen. The intended change is to lower the fraction to **0.75**. Note the
interaction with the `duration − 10` term: on long tracks that term dominates (a 10-minute track needs
9:50 either way), so planning should decide whether the floor becomes `max(0.75·duration, duration−10)`
or simply `0.75·duration`. Seeking is explicitly *not* a concern here — the user may seek however they
like, and a completion reached by a forward seek counting as a listen is acceptable by design.
Severity: **tuning** (single constant; re-measure the settling curve in §7 only if it changes felt
behaviour).

**B2 · Poll-based completion can silently drop a full listen.**
Same block (`main_tui.py:352-364`). Completion is detected only if a 0.5 s poll lands in the
`[threshold, end]` window *while* `state == 'playing'`. If polls are delayed (thread contention, an
`mpc` timeout, load) and the track ends + MPD auto-advances between two polls, the next poll sees a new
`current_track`, resets the flag (`main_tui.py:338-341`), and the finished track is never counted:
no session update, no `tracks_played`, and a blank (outcome-less) row in the history panel for a track
that fully played. Lowering the threshold to 75 % (B1) *widens* the catch window, which incidentally
makes this rarer. Severity: **minor** (probabilistic).

**B3 · The first *two* tracks of a cold session are both random, not one.**
`start_session` pre-queues `1 + queue_lookahead = 2` tracks (`main_tui.py:241`, `queue_manager.py:91`)
while both session and taste vectors are still zero, so the lookahead (track 2) is a uniform draw
unrelated to track 1. `project_state.md:108` states only "the first track is a uniform random draw."
Selection first reflects the seeded session at track 3. Severity: **minor** (design nuance; may feel
jarring on cold start).

**B4 · On a warm restart the "first track is uniform" invariant does not hold.**
`project_state.md:108` says the opening pool is empty because "both vectors are zero." After a restart
with a persisted taste model, `user_taste.load()` seeds `taste_vector`, so `get_candidate_pool` opens
its taste half (`track_library.py:310`) and the first track is drawn from taste neighbours, not
uniformly. This is arguably the *desirable* behaviour (README:154 promises taste anchoring), but it
contradicts the stated invariant. Severity: **minor** (doc/behaviour mismatch, not a defect).

**B5 · `taste_ramp` / "β earned" is driven by skips, not only positive signal.**
`user_taste._update` increments `total_updates` even on an unseeded skip (`user_taste.py:111`), and
`taste_ramp` reads `total_updates` (`exploration_controller.py:78`). A listener who only skips ramps
their displayed "β earned" upward (`tui.py:784`) though no *positive* evidence exists. Harmless while
the taste vector is zero (the term is a constant 0.5 and reorders nothing), but the counter overstates
earned taste weight. Severity: **minor**.

**B6 · Un-like decrements this session's like counter by whole-history matches.**
`feedback_handler.py:169,189`: `removed` counts like-events for the track across the *entire* loaded
history (all sessions), then subtracts that from `session_feedback_count['likes']` (this session only).
Un-liking a track liked in a previous session decrements tonight's like count (clamped at 0, so no
negative, but inaccurate). Severity: **trivial** (stat only).

---

## C. Album-art lifecycle and rendering

**C1 · A single BrokenPipe permanently disables album art, and the self-heal is unreachable.**
`album_art.py:114-115` (and identically `:207-208`): on a broken stdin write the handler sets
`self.process = None`. The next `render()` hits the guard `if not self.process ... : return`
(`:96`/`:189`) and returns *before* the respawn branch `if self.process.poll() is not None:
_start_layer()` (`:98`/`:191`) can run — so the code that exists specifically to resurrect a dead
ueberzugpp child can never execute after a pipe break, while `is_available()` still reports True and
nothing is logged. This is the most likely mechanism behind the observed "finagling until it comes
back" behaviour: once the child dies, only a *new* renderer (or a code path that clears
`self.process` **and** falls through to respawn) restores it. Severity: **major**.

**C2 · Resizing the terminal breaks the cover; only repeated resizing restores it.**
Observed live: one resize leaves the image broken or gone, and continuous resizing toggles it back and
away until it happens to land. The relevant machinery: `_on_sigwinch` calls `force_redraw()`
(`tui.py:932`), and `_render_art` re-derives geometry from `get_cols_rows()` and re-sends every tick
(`tui.py:1180-1200`), while `_art_geometry` returns `None` at some sizes (the `MIN_ART_*` floors and
the `visible` clip, `tui.py:1174-1178`), in which case the renderer is *cleared*. A resize also risks
the C1 pipe break if ueberzugpp chokes on a mid-resize command. The net effect — an image whose
presence depends on which frame a jittering geometry/among-clear/among-respawn sequence settles on — is
consistent with the "finagle until it's back" symptom. This is app-adjacent (the render/clear/respawn
interplay under SIGWINCH), not purely a ueberzugpp limitation, so it is worth reproducing
deterministically. Severity: **major** (directly user-visible).

**C3 · Under tmux with multiple attached clients, the cover appears in only one terminal and is
disturbed by the others.**
Observed: with two terminals attached to the same tmux server, the cover renders in the first only;
opening a second client, then closing it, removes the image from the first until a resize restores it.
ueberzugpp positions overlays in absolute terminal coordinates against a single output surface, and
tmux multiplexing across clients of differing geometry has no single truth for "where is cell (x, y)."
This is largely an **inherent ueberzug/tmux limitation** rather than a fixable app bug — the honest
outcome is probably to document it (as the README already documents the X11/Wayland-only constraint)
rather than promise a fix. Worth noting the closing-a-client-removes-it symptom overlaps with C1/C2:
the disturbance likely triggers a clear or a pipe event the app then fails to recover from. Severity:
**minor** (constraint to document; partial overlap with C1/C2).

**C4 · The cover-cache temp dir leaks on SIGTERM.**
Cleanup of `adaptive_dj_covers_*` is registered via `atexit` (`album_art.py:301-303`), which does not
run on default SIGTERM — the same failure class the shutdown refactor fixed for the ueberzugpp *child*
(wired into the signal handler) but left unaddressed for the cache dir. Temp dirs accumulate across
signal-terminated runs. Severity: **minor**.

**C5 · Cover-cache writes are non-atomic.**
`album_art.py:379-385` writes cached covers with `write_bytes`; a truncated file (crash mid-write) is
then served forever from the `cached.exists()` fast path. Temp-then-rename would be safe. Severity:
**minor**. (The embedding pipeline has the same pattern for its artifacts — see D3.)

---

## D. Embedding generation — the determinism story is empirical, not enforced

**D1 · "Bit-identical embeddings" is not enforced on the documented CUDA path.**
`embedding_generator.py:249-250` sets only `torch.backends.cudnn.benchmark = False`. That fixes
algorithm *selection*, not determinism of the chosen kernel; `torch.use_deterministic_algorithms(True)`
/ `cudnn.deterministic = True` are absent. The headline invariant (`project_state.md` §3, §8
"bit-deterministic for a fixed batch size") rests on the measurement having held on one RTX 3070, not
on the code guaranteeing it. `tests/test_clap_pipeline.py` skips unless the model is cached, so CI
never checks it. Severity: **medium** (a core stated invariant is unguarded).

**D2 · Resume does not pin batch size, and the docstring says batch size changes the vector.**
`_save_partial` (`embedding_generator.py:611-622`) stores no batch size; `--resume` never validates it.
Generate with `--batch-size 32`, interrupt, resume with `--batch-size 16`, and the artifact holds rows
computed under two regimes the code itself claims are non-equivalent — while metadata records only the
final run's `batch_size` (`:664`). Severity: **medium** (silent, subtly inconsistent artifact).

**D3 · Non-atomic artifact writes; corrupt partial is swallowed into a silent full restart.**
`_save_partial`/`_save_embeddings` (`:616`, `:682`) write straight to the destination — a kill or full
disk mid-write corrupts the file in place, destroying a completed run. `_load_partial` (`:624-638`)
catches *every* exception, emits only a `warnings.warn`, and returns empty — so a truncated
`.partial.npz` silently discards all prior work and looks like a normal slow run. Severity: **medium**.

**D4 · Schema validation is shape-only for the two things that would silently poison scoring.**
`embeddings_io.validate_embeddings` checks `centroid`'s *shape* but never that it equals
`embeddings.mean(0)` — a stale/edited centroid passes and every downstream `centre()` uses the wrong
origin (the exact C5 failure the schema exists to prevent). It also never checks that `window_offsets`
is monotonic non-decreasing, nor that `windows` rows are L2-normalised (only pooled `embeddings` are
checked). Severity: **minor–medium**.

*(Lower-severity generation notes from the sub-inspection: over-strict disk check on cached re-runs,
checking the HF-cache filesystem rather than the artifact's; resumed tracks under-count the window-RMS
metadata percentiles; duplicate MPD keys would yield duplicate rows undetected; validation accepts any
schema ≥ 2 with no upper bound. All low-probability; listed for completeness.)*

---

## E. Documentation / consistency drift (the project's own stated pet peeve)

**E1 · `start.sh`'s control banner is a third, drifted binding list.**
`start.sh:207-214` prints SPACE / N / L / ,. / ←→ / I / Q but **omits `[↑↓] History` and
`[ENTER] Replay`**. `project_state.md` §4 and README:112 assert "the footer, the README table and both
interfaces advertise exactly this list," and a test binds the README/start.sh *numbers* together — but
the *controls* list in the launcher is unguarded and has drifted, the exact L9/M7 class of defect the
rewrite prides itself on closing. (A new neutral-skip binding, G1, must be added here too.) Severity:
**minor**.

**E2 · README "excluded for at least 20 songs" vs. an effective 50-track exclusion, and a dead branch.**
`select_track` hard-excludes the last `recent_history_size = 50` selections (`track_selector.py:114`),
so any *scored* candidate was last played >50 selections ago. The `if tracks_since < minimum_replay_gap
(20): return 0.1` penalty branch (`track_selector.py:255`) is therefore unreachable via normal
selection, and the real exclusion window is 50, not 20. README:235 and the `save()` docstring both say
"20." "At least 20" is technically true, but the 20-gap penalty is dead code. Severity: **minor**.

**E3 · `music_directory.validate_music_directory` proves *existence*, not *containment*.**
`music_directory.py:143` uses `(music_dir / t).exists()`. Because `Path` join discards the left side
when `t` is absolute and `..` escapes, the "resolve under it" claim in the module docstring is stronger
than what is enforced. MPD reports relative paths, so real-world risk is low. Related minor
parsing-fidelity gaps: the `#`-as-comment regex truncates a directory containing `#` (`:37`);
`expandvars` expands `$VAR` that MPD takes literally (`:59`); detection falls through to a
lower-priority config file where MPD would stop at the first existing one (`:85`). Severity: **minor**.

**E4 · `MPD_MUSIC_DIR` "permanent" wording** — `start.sh:140-141` `export`s it for the current launch
only while calling it "permanent." Severity: **trivial**.

---

## F. Robustness edges (low severity, noted once)

- **F1 · No MPD reconnection after the first connect.** Once `MPDController.connected` is True
  (`mpd_controller.py:38`) it never reverts; if MPD restarts mid-session every call times out forever.
  Severity: **minor**.
- **F2 · Position/duration regex `(\d+):(\d+)/(\d+):(\d+)`** (`mpd_controller.py:112`) does not match an
  `H:MM:SS` display for tracks over an hour; position/duration read 0 and completion never fires for
  such a track. Severity: **minor**.
- **F3 · `[I]` overlay leaves the ueberzugpp image on screen** — the overlay box draws over the TUI but
  the art layer is a separate surface not cleared while the inspector is open. Cosmetic.

---

## G. Requested changes — features and structure

**G1 · Neutral skip: advance to the queued track without moving any vector.**
Desired: a skip that means "not this particular song right now, but keep the vibe" — it should just
play the already-queued lookahead and touch *none* of the maths (no session repel, no taste penalty, no
exploration change, no escalation counter). Design notes for planning:
- It is simpler than the real skip, not harder: the lookahead already exists at depth 1, so a neutral
  skip does **not** call `feedback_handler.process_skip` and does **not** `replace_next`. It only needs
  to guarantee the add-before-advance invariant (if the queue is momentarily 1-deep at a boundary, add
  one lookahead first), then `next_track()`, then re-pause if paused — mirroring `skip_current_track`
  (`main_tui.py:250`) minus the feedback call.
- History marking: the neutrally-skipped track was heard partially; it is already recorded by
  `note_playing`. Decide its outcome mark — leave it blank, or introduce a distinct neutral glyph
  (not `⏭`, which reads as rejection, and not `✓`, which is a full listen). This is a
  `session_history.py` decision.
- Binding: needs a new key added to the *one* shared table — the urwid `_handle_input` dispatch
  (`tui.py:555`), the footer, the README table, and `start.sh` (see E1) — so the interfaces cannot
  drift. Pick a key that does not collide with the existing set.
Severity: **feature**.

**G2 · Move all module files into a `src/` folder.**
Desired for a cleaner top level. This is mechanical but has real consequences a cold-start agent must
handle in one pass, because the current layout is flat and several things assume it:
- **Imports are flat** (`from config import config`, `from track_library import TrackLibrary`) in both
  the modules and `tests/` (e.g. `tests/conftest.py`). Moving to `src/` requires either running with
  `src/` on `sys.path`, making `src/` a package, or adjusting every import and the test discovery path.
- **`config.data_dir = Path(__file__).parent / 'data'`** (`config.py:135`) and every path derived from
  it (`embeddings_file`, `taste_file`, `log_file`, …) would relocate `data/` under `src/data/` unless
  re-anchored to the repo root (`Path(__file__).parent.parent / 'data'`). The same applies to any
  `__file__`-relative path in `generate_embeddings.py` / `embedding_generator.py`.
- **`start.sh`** invokes `python3 main_tui.py`, `python3 music_directory.py`, and
  `python3 generate_embeddings.py` (`start.sh:125,189,218`) — all paths change to `src/…`.
- **`tests/test_deletions.py`** guards that `tests/` stays tracked and that track keys are not
  enumerated from the filesystem; verify its path assumptions survive the move.
- **`project_state.md`** and comments reference bare module names (mostly filenames, not paths), so most
  prose survives, but the "the only entry point" wiring diagram should be re-checked.
Do this as an isolated commit with the suite green before and after, so no behavioural change hides
inside the move. Severity: **refactor**.

**G3 · Vibe cloud as the centrepiece: a full-body 3-D mood cloud, panels on demand.**
Settled direction, validated against the real library. Prototypes live in the repo root
(`explore_mood_axes.py`, `vibe_cloud_demo.py`, `render_terminal_frames.py`) and a frozen browser preview
is in `archive/` (see `archive/NOTES.md`). The header stays — album art · track details · seek bar · the
three per-track descriptor words — and the footer stays. The **whole body below the header becomes the
vibe cloud**: an auto-rotating 3-D point cloud of the library in mood-space with a session comet tracing
the trajectory. History and console move *off* the default view onto **toggle keybinds**, which is what
lets the cloud have the full cell budget it needs to be legible.
- **Axes are data-driven, not hardcoded.** Measured on this library, the intuitive colour triad
  Intensity·Tone·Saturation collapses — Intensity and Saturation correlate **0.98**, so the cloud is a
  pancake (participation ratio 1.7). The usable triad here is **Intensity · Tone · Organic** (one of the
  collinear energy/saturation pair, plus the two independent axes). Because another user's library can
  favour a different triad, the axes **and** their z-score calibration must be selected and stored at
  embedding-generation time (like the descriptor mean/std already are, §6) and re-fit as the library
  grows — never hardcoded. Colour is each point's own three axis values as HSV, so position and hue
  cannot disagree.
- **Rendering is Braille (2×4 dots per cell) + 256-colour, computed live per tick.** A dedicated
  animation alarm (~20–30 fps) drives *only the cloud widget*, decoupled from the 2 Hz MPD poll — so
  orbit speed and framerate are independent. The camera orbits (ambient); the comet moves only on real
  session events (data).
- **Interaction:** mouse via urwid's mouse events — which the code already receives (`tui.py`'s `[I]`
  loop tests `isinstance(key, tuple)` for them) — giving drag-orbit, scroll-zoom, and click-a-point-to-
  inspect (→ the track key the app already holds), plus keyboard orbit/zoom through the one shared
  binding table. Since the cloud wants the arrow keys, revealing history is a **focus mode**: cloud
  focus → arrows orbit; toggle → history panel where arrows navigate and ENTER replays.
- **Hiding the console loses nothing** — `data/dj.log` is already the durable copy (§4), so the 5-line
  panel was never the source of truth.
- **Honest caveats, none a blocker:** braille + 256-colour is coarser than the browser preview (no
  anti-aliasing, no glow); a full urwid redraw at 30 fps is heavier than a browser `<pre>`, hence the
  panel-only alarm; click precision is per-braille-cell, so dense regions need "nearest within a small
  radius" plus a hover highlight; and 256-colour in urwid needs a registered high-colour palette rather
  than raw ANSI.
Severity: **feature (centrepiece)**.

---

## H. What holds up well (so it isn't re-litigated)

The skip ordering (add-before-advance, no `play()` in the skip path), the mode force/restore across
every exit path, the un-like replay + `explains()` gate, the zero-vector retrieval guard, the
rank-Boltzmann sampling rationale, the derived album-art *geometry*, the packed Now-Playing box, and
the shared binding table were all inspected closely and match their documentation. The
FakeMPD-under-test discipline is real. None of the findings above undermine those; they are the edges
around them.

---

# Proposed order of work — four phases

Regrouped from the earlier seven stages so that work sharing a context (and a set of files) is done by
one agent, not handed between several. Each phase is scoped to fit one cold-start agent comfortably.

**Every phase, before touching code:** read `project_state.md` §4 (the invariants that are easy to undo)
and §5 (verified MPD semantics), plus the findings sections this phase cites — they carry the file:line
anchors and the reasoning. Run `python3 -m pytest tests` first for the green baseline (**542**). Never
remove the autouse `_isolate_state_files` fixture, and never add a state file without registering it
there (§8).

Dependency graph:  **Phase 1 → Phase 2**,  **Phase 1 → Phase 3**,  **(Phase 1 + Phase 3) → Phase 4**.
Phase 2 is independent of 3 and 4, so 2 and 3 can run in either order (or in parallel) once 1 lands.

---

### Phase 1 — Foundation, feedback, and consistency
**Closes G2, B1, G1, B2, E1–E4, F2; optionally A1.** One agent. Everything here lives in the same
context — `config.py`, `main_tui.py`, `feedback_handler.py`, `session_history.py`, `tui.py`,
`track_selector.py`, `start.sh`, `README.md` — and the edits are small. Grouped because the `src/` move,
the neutral skip, and the doc-consistency fixes all touch the binding table and the launcher; doing them
in one pass avoids re-reading the same files three times.

Order within the phase:
1. **`src/` move (G2)** — first, as its own commit, suite green either side, zero behavioural change.
   Re-anchor the `__file__`-relative `data/` paths (`config.py:135`) to the repo root, fix the flat
   imports and test discovery, update `start.sh`'s three invocations. Everything after lands in `src/`.
2. **Completion threshold → 75 % (B1)** at `main_tui.py:356`; decide the `duration − 10` floor. It also
   widens B2's catch window.
3. **Neutral skip (G1)** — a feedback-free variant of `skip_current_track` (`main_tui.py:250`): it does
   *not* call `feedback_handler.process_skip` and does *not* `replace_next` — it plays the existing
   lookahead. Preserve add-before-advance: if `get_queue()` is 1-deep at the boundary, add a lookahead
   *before* `next_track()`; re-pause if paused. New mark in `session_history.py` (marks at `:34-44`; not
   `⏭`/`✓`). New key through the *one* shared table (`tui.py:555`) and added to the footer, README table
   and `start.sh` banner together.
4. **Doc / dead-code reconciliation (E1–E4, F2)** — fold the `start.sh` banner into the documented-
   numbers test; reconcile "20 vs 50" and remove-or-revive the dead anti-repetition branch
   (`track_selector.py:255`); tighten the music-directory containment/parsing claims; note the `H:MM:SS`
   regex limit.
5. **(Optional) concurrency lock (A1)** — only if you want the insurance while already in these files.
6. **Minor sweep, same files** — while you are in here, decide per item whether to fix or explicitly
   accept (don't silently drop): **B3** cold-start double-random, **B4** correct the `project_state.md`
   "first track is uniform" wording, **B5** β-ramp counting skips, **B6** un-like session-counter,
   **F1** no MPD reconnection. Each is a one-line edit or a doc correction.

Invariants: **no `play()` in any skip variant, and add-before-advance** (§4); **one shared binding
table** (§4). Verify: full suite green; a neutral-skip test on the real methods asserting one `next`, no
`play`, add-before-advance, and *no vector/taste/exploration change*; documented-numbers test extended to
the banner.

---

### Phase 2 — Album-art lifecycle and rendering  ✅ DONE (branch `rewrite/phase-2`, suite 591 → 603)
**Closes C1, C2, C4, C5; documents C3.** Depends on Phase 1's `src/` move. Separate agent: a
self-contained subsystem (`album_art.py` + the SIGWINCH/render path in `tui.py`), and C2 needs
interactive/visual reproduction — a different mode from Phase 1's edits.

> **What Phase 2 actually did** (so Phase 3/4 don't re-derive it):
> - **C1** — the two protocol classes (`UeberzugppProtocol`, `UeberzugProtocol`) were near-identical
>   copies carrying the pipe-break bug *twice*; they now share one `ImageProtocol` base and set only
>   `binary`/`launch`. `ImageProtocol.render()` **returns a bool** and, on `BrokenPipeError`, drops the
>   dead child and retries once against a fresh layer in the same call. `is_alive()`/`_ensure_process()`
>   are the new helpers.
> - **C1/C2 honesty** — `protocol.available` flips False **only** on a confirmed-missing binary
>   (`FileNotFoundError`), never on a transient immediate-exit, so one mid-resize hiccup can't
>   permanently disable art. `AlbumArtRenderer.render` reads that bool: on failure it forgets the render
>   key (so the next tick retries) and disables + logs only when `protocol.available` is False. The
>   renderer's flicker-skip is now gated on `self.protocol.is_alive()` — skipping a re-send is safe only
>   while a child is holding the image.
> - **C2** — the "toggle until it lands" symptom is the C1 pipe-break plus the flicker-guard wedge;
>   both are fixed above. Active-resize flicker can still occur while dragging (many rapid re-sends at
>   changing geometry) but **converges within one 0.5 s tick** once resizing stops. `_on_sigwinch` was
>   already correct (signal-safe `force_redraw()` only) and is unchanged. No deterministic *visual*
>   repro was added — the mechanism is reproduced at the unit level in `test_album_art_lifecycle.py`
>   (pipe break → respawn → paint; dead child → key no longer skipped).
> - **C4** — `cleanup_cover_cache()` (module fn, idempotent, clears the global) is called from
>   `AlbumArtRenderer.shutdown()`, which the signal path and `_shutdown()` both reach; `atexit` stays as
>   a backstop.
> - **C5** — `_atomic_write_bytes(path, data)` (temp-then-rename via `os.replace`) replaces the cache
>   `write_bytes`. **Phase 3's D3 wants the same property** for the `.npz` artifacts — see the handoff
>   note below.
> - **F3** — `_show_model_info()` clears the art on open and `force_redraw()`s on close.
> - **C3** — documented in `README.md` (Requirements) as an inherent overlay-under-tmux limitation.

Work: make the BrokenPipe path reach the respawn (clear `process` *and* fall through to `_start_layer`,
or gate on a `needs_restart` flag) — `album_art.py:96/98/115`; reproduce the resize toggling
deterministically and stabilise the render/clear/respawn interplay under SIGWINCH (`_on_sigwinch`
`tui.py:921`, `_render_art`/`_art_geometry`); wire cover-cache cleanup into the signal/`_shutdown` path
(not just `atexit`); cache write temp-then-rename; document the tmux-multi-client limit in the README.
Also clear the art layer while the `[I]` overlay is open (**F3**), so the image does not sit under it.

Invariants: `AlbumArtRenderer.shutdown()` **cannot raise** and runs before `request_exit()` (§4). Verify:
a test that kills the child and asserts the next `render` respawns and `is_available()` stays honest.

---

> **⚠ Phase 2 → Phase 3 handoff — read before you start (three trip-hazards):**
>
> 1. **The finding-ID letters in *this document* collide with `project_state.md` §10's glossary.**
>    The glossary defines its own `C1–C5` and `D1–D8` (e.g. glossary `D1` = "queue depth 10→1",
>    glossary `C5` = "similarity scale compressed") — **different findings** from this doc's inspection
>    `C1–C5`/`D1–D4`. Phase 2 disambiguated its code comments by writing **"(inspection C1)"** rather
>    than "(audit C1)". **Phase 3 must do the same for its D-series** — cite **"(inspection D1)"**, never
>    "(audit D1)", or a future reader will map your comment to the wrong glossary entry. (Phase 1's
>    citations happened not to collide because it only touched A/B/E/F/G letters, which the glossary
>    doesn't use — so it wrote bare "(audit G1)". That convention is unsafe for C and D.)
>
> 2. **The atomic-write pattern D3 needs already exists — but not reusably for `.npz`.** Phase 2 added
>    `album_art._atomic_write_bytes(path, bytes)` (write-`.tmp`-then-`os.replace`). It is **bytes-only**,
>    so it does **not** wrap `np.savez`/`np.savez_compressed` directly. For D3, mirror the *pattern*:
>    save to `<dest>.tmp.npz` (or a `NamedTemporaryFile` in the same dir) then `os.replace` onto the
>    destination — same-directory rename is the atomic part. Consider factoring one shared helper if you
>    want, but don't try to call the bytes helper on an npz path.
>
> 3. **Branch/baseline bookkeeping.** Phase 2 lives on `rewrite/phase-2` (off Phase 1's tip) and the
>    suite is **603 green** there. Phase 3 depends on Phase 1, **not** Phase 2 — the two are independent
>    siblings. So Phase 3's green baseline is **591** if you branch from Phase 1's tip, or **603** if you
>    branch from a tree that already has Phase 2 merged. Confirm which base you're on before trusting a
>    count; the `_isolate_state_files` autouse fixture and the "read `src/<module>.py`, not
>    `<module>.py`" source-reading rule (§8) apply unchanged.
>
> Nothing in Phase 2 touched `embedding_generator.py`, `embeddings_io.py`, `generate_embeddings.py`, or
> the artifact schema, so there are **no file-level conflicts** with Phase 3's scope.

### Phase 3 — Generation pipeline: durability and the vibe-cloud axes
**Closes D1–D4; produces the stored axes Phase 4 needs.** Depends on Phase 1's move. **Gate: run only
when already regenerating embeddings** — §9 warns a regeneration must not be run casually. One agent:
all of this is `embedding_generator.py`, `embeddings_io.py`, `generate_embeddings.py` and the artifact
schema. The axis work is grouped here because it is generation-time, so it and the D-fixes land in a
single regeneration rather than two.

Work: D1 enforce CUDA determinism (or downgrade the docstring to "empirically bit-identical on the
tested device"); D2 pin/validate batch size across `--resume`; D3 atomic writes (temp-then-rename) for
`_save_partial`/`_save_embeddings` (`:640`) and a hard, visible error on a corrupt partial
(`_load_partial` `:625`); D4 extend `validate_embeddings` (`embeddings_io.py:71`) to check centroid ==
`mean(embeddings)`, `window_offsets` monotonic, and window-row norms. **Plus**: compute the mood-axis
selection + per-axis z-score calibration at generation time (the `explore_mood_axes.py` logic) and store
them in the artifact (new keys; extend the schema), so the cloud is per-library and re-fits as it grows.

Invariants: schema-version discipline — refuse older, metadata stays JSON not pickle, the file is the
authority on dimension (§6). Verify: extend `test_embeddings_io` to break each new check; a test that the
stored axes reproduce `explore_mood_axes.py`'s pick on the real library (skip if `data/embeddings/`
absent).

---

### Phase 4 — Vibe-cloud widget and TUI restructure
**Closes G3.** Depends on **Phase 1** (final `src/` layout + shared binding table) and **Phase 3** (the
stored axes). The largest single piece, rightly its own agent: a new widget module plus a body
restructure in `tui.py`. References: `vibe_cloud_demo.py` (braille render logic), `explore_mood_axes.py`
(axis selection), `archive/` (target feel).

Work: (a) a dedicated cloud widget rendering Braille + 256-colour from the live session/library vectors,
on its **own ~30 fps alarm (panel-only redraw)** alongside the 2 Hz poll; (b) restructure the body so the
cloud owns the full space below the header, with history and console on **toggle keybinds** and a
**focus mode** for the arrow keys (cloud focus → orbit; history focus → navigate + ENTER replays);
(c) mouse — drag-orbit, scroll-zoom, click-to-inspect (urwid already delivers mouse events as tuples,
`tui.py:905`) — and keyboard, through the shared table, with bindings added to footer/README/`start.sh`
together.

Invariants: **camera orbits / comet moves only on data** — never blur ambient and data motion; **a loop
that blocks must keep observing the session** (§4) — the panel-only alarm must not starve
`_sync_session_state`, or a track that starts and ends between ticks is lost; the packed Now-Playing box
and derived album-art geometry stay (§4); 256-colour needs a registered high-colour palette, not raw
ANSI. Verify: render the widget at several terminal sizes (the `test_art_geometry` "rendering is the
point" discipline); assert the animation alarm coexists with the 2 Hz poll without dropping full-listens;
a hit-test unit test. Python is adequate (674 points is trivial); revisit C/Rust only for very large
libraries.
