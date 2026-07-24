# Adaptive Session AI DJ

A terminal DJ for MPD that picks tracks from CLAP audio embeddings and adapts to your feedback
*within* a listening session. Python + urwid TUI; MPD driven through `mpc` subprocesses.

**Goal.** Put on music and have it follow you — no genre tags, no playlists, no rules you have to
write. Everything is derived from what the audio actually sounds like, and everything the interface
tells you has to be something the system can back up.

**Status.** Feature-complete and stable. 668 tests, green, ~20 s. It plays continuously, adapts on
the next track, says what it is playing, restores your MPD state on every exit path, and shows the
whole library as a rotating 3-D mood cloud with a session comet. Coverage on all three interfaces
(player, urwid display, fallback text mode).

> This file replaces `project_state.md` and `inspection_findings.md`, which recorded a rewrite and an
> inspection whose findings are all closed. Both are in git history at `c599ab7` — the last commit
> that held them — if you ever need the narrative:
> `git show c599ab7:project_state.md`, `git show c599ab7:inspection_findings.md`.
>
> **Reading the code comments:** roughly 450 of them cite finding IDs — `(audit C4)`, `(audit L9)`,
> `(inspection D3)`, `(G3)`. All are closed; they are provenance, not open work. The decoder tables
> are in the two archived files (`project_state.md` §10 for `audit`, `inspection_findings.md` for
> `inspection`). **The two ID spaces collide** — `audit C1` and `inspection C1` are different findings
> — which is why the prefix in the comment matters.

---

## 1 · The governing principle

Nearly every defect this project has had shared one cause: **numbers chosen against a scale nobody
measured.** An entropy threshold calibrated for a smaller dimension. A novelty formula assuming a
range the data never occupies. A "50% vibe shift" that turned over 9% of what you would actually
hear. A similarity scale compressed into its top third.

So: **derive constants from the library's actual distribution, or delete them.** A constant may stay
if it shapes behaviour without asserting a fact. It goes if it produces a claim the user reads that
the system cannot back up. Prefer scale-invariant formulations — rank-based sampling over
score-based, z-scores over raw similarities, measured pool-turnover over a declared magnitude — so
nothing needs recalibration when the weights or the embedding space move.

**The corollary, worth more than the principle: an observation is not a conclusion.** A correct
measurement of the SIGWINCH handler once produced the wrong inference, and acting on it broke the
application under nine unit tests that all passed — because they ran against a double built from the
same assumption. Before trusting any claim in this file, check that the observation behind it still
supports it.

---

## 2 · How the system works

Everything is driven by 512-dimensional CLAP audio embeddings, one per track, stored L2-normalised in
`data/embeddings/track_embeddings.npz`. All similarity is a dot product **on centred vectors**. No
genre tags or metadata enter the selection logic.

All application modules live under `src/`; imports are flat (`from config import config`) with `src/`
on `sys.path`. `data/`, `tests/`, `README.md`, `start.sh` and this file stay at the repo root, and
`config.data_dir` is anchored to the repo root (the parent of `src/`).

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
   ├──▶ session_history.py  what played · ♥ ⏭ ✓ » · the ↑↓ cursor
   ├──▶ vibe_cloud.py       mood_axes.py → rotating 3-D Braille point cloud +
   │        session comet; single-dot depth-shaded points, eased camera; own alarm
   │        (all three display-only: nothing behind the display reads them)
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

**Cold start.** Both vectors are zero, step 1 returns an empty pool, and the opening picks are uniform
random draws. Two known nuances:

- **The first *two* tracks are both uniform.** `start_session` pre-queues `1 + queue_lookahead` (= 2)
  while the session vector is still zero, so the lookahead is drawn independently of track 1.
  Selection first reflects the seeded session at track 3. Inherent to depth-1 gapless startup and
  accepted: deferring the lookahead until track 1 seeds the session improves exactly one track at the
  cost of a startup-gap edge case.
- **A *warm* restart is not uniform at all.** A persisted taste model seeds `taste_vector`, so
  `get_candidate_pool` opens its taste half and the first track is drawn from taste neighbours. This
  is the desired behaviour (taste anchoring across sessions).

### What feeds back

| Event | Trigger | Effect |
|---|---|---|
| **Full listen** | ≥75% of duration played (`config.full_listen_fraction`) | Session vector updated (the primary driver). Taste +0.02. Exploration −0.02. The only thing that increments `tracks_played`. |
| **Skip `[N]`** | Keypress | Taste −0.05. Exploration +0.05. Session vector repelled from the consecutive-skip-run centroid by a magnitude **solved for a pool-turnover target that escalates with run length** (5% → 20% → 50% → 85%), projected back onto the manifold from the second consecutive press. Lookahead replaced, then exactly one advance. Measured turnover printed to the console. |
| **Pass `[V]`** | Keypress | **Nothing to the model**: no session repel, no taste penalty, no exploration change, no escalation counter. Advances into the already-queued lookahead — add-before-advance, exactly one advance, no `play()` — and marks the track `»`. "Not this song, keep the vibe." |
| **Like `[L]`** | Keypress | Taste +0.10, saved immediately. Adds `♥`. |
| **Un-like `[L]`** | Keypress on an already-liked track | Removes the like from `feedback_history` and **recomputes** the taste model from what remains. See §4. |

### What persists

`user_taste.npz` (on every like and at exit), `exploration_state.json`, `feedback_history.json`,
`play_history.json` (carrying `play_history`, `current_index` and `recent_history`, so
anti-repetition survives a restart). The session vector is intentionally ephemeral. All of it is
checkpointed every `config.checkpoint_every_n_tracks` (5) full listens, not only at exit.

**Nothing in `data/state/` is worth preserving.** It is all regenerated by listening and none of it is
committed. Do not write migration code, backward-compat shims, or "reset carefully" logic for it.
`data/embeddings/` is different — it is a five-minute rebuild
(`python3 src/generate_embeddings.py --force`), not a throwaway.

---

## 3 · The vector space

Settled; do not re-derive without re-measuring everything downstream.

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

**Read this before changing anything.** Each cost real debugging to establish, and each would silently
regress.

### MPD is the user's machine

- **Playback modes are the user's state.** The app forces `random`/`repeat`/`single` off and
  `consume` on, but it must log what it changed and restore the originals on *every* exit path
  including SIGTERM. Wired into `_shutdown()`, the signal handler **and** an `atexit` hook.
- **`mpc listall` is the single source of truth for track keys.** Do not enumerate the filesystem for
  anything MPD will be asked to play. The embeddings are stored under these exact strings.
- **A skip must add the replacement *before* it advances.** `mpc next` off the last remaining track
  empties the queue and stops MPD, and a later `mpc add` does *not* restart it. Advance-then-add kills
  the session silently, and the only recovery is a `play()` call that the skip path forbids. There is
  no `play()` anywhere in `skip_current_track()`, which makes the double-advance impossible by
  construction rather than by care.

### Album art

- **The `ueberzugpp` child must be terminated explicitly.** `clear()` removes the image and leaves the
  process running. `AlbumArtRenderer.shutdown()` is called from `_shutdown()` and from the signal
  handler, and **cannot raise** — it runs before `request_exit()`, the call that unblocks urwid and
  makes the rest of the shutdown reachable. It also tears down the extracted-cover cache dir
  (`cleanup_cover_cache()`), because `atexit` does not fire on a default SIGTERM.
- **A broken pipe to the overlay child must respawn, not disable art.** One long-lived `ueberzugpp`
  child sits on a stdin pipe. `ImageProtocol.render()` drops a dead child and retries once against a
  fresh layer *in the same call*, and returns a bool the renderer reads. **`protocol.available` goes
  False only on a confirmed-missing binary (`FileNotFoundError`), never on a transient spawn failure**
  — otherwise one mid-resize hiccup permanently disables art. The renderer's flicker-skip guard
  (`key == self._render_key`) is gated on `protocol.is_alive()`: skipping a re-send is only safe while
  a child is actually holding the image, or a dead frame freezes on screen.
- **The `[I]` overlay takes the cover down while open.** The art is a separate X11/Wayland surface
  urwid does not paint over. `_show_model_info()` clears it on open and `force_redraw()`s on close.
- **Only ueberzugpp / classic ueberzug work.** The kitty and sixel paths were deleted — they fought
  urwid for the screen. Under tmux with multiple attached clients the cover renders in one terminal
  only; that is inherent to absolute-coordinate overlays, documented in the README, not a bug to fix.

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
- **The album-art geometry is derived from the widget tree,** not hand-counted. Two of four original
  hand-counted constants were wrong. Same for the `[I]` overlay's inner size, which asks
  `Overlay.calculate_padding_filler()` / `top_w_size()` rather than recomputing the arithmetic.
  **The footer is a flow widget and its `rows()` feeds `_art_geometry()`**, so its wrapped height —
  which varies with terminal width — is part of the derivation.
- **The drift figure is a count of held words, not a cosine.** The cosine has p10 = 0.948 and median
  = 0.989 over 40 real sessions, so it reads as "0.99" forever — the same compressed-scale defect the
  readout exists to fix. The cosine is still computed and shown in `[I]` with its distribution beside
  it. See §6 before changing this back.
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

### The vibe cloud

- **The camera orbits ambient; the comet moves only on data.** `_animate` (a fast alarm, up to ~144 fps,
  separate from the 0.5 s poll) advances only the camera and redraws just the cloud, and **only when
  the camera moved enough to matter** (`cloud.pose_changed()`). The camera **eases** toward a mouse-set
  target rather than snapping, and both the spin and the ease are in real time (`advance(dt)`), so the
  framerate is free to change without changing the feel. urwid's canvas cache makes it a panel-only
  redraw, so it never starves `_sync_session_state`. `note_session` lays a comet bead only when the
  session vector actually moved. **Never drive the comet from the animation alarm, and never put
  session bookkeeping on it.**
- **The camera is a mouse instrument; the arrows are for the history.** Orbit is **right-drag** (left
  is free for picking), zoom is **scroll**, and an on-panel **orbit-speed slider** (0 = static, starts
  slow) is set by left-click/drag. So `↑/↓` always move the history cursor. `ENTER` is the one
  focus-dependent key: `_handle_input` routes it to a single method (`_enter_action`) that resets the
  cloud when the cloud is up and replays the focused track when the history is up — still one binding
  table. `[T]` cycles the body pane (cloud → history → console); `F1/F2/F3` jump straight to one
  (unadvertised). `←/→` stay seek; `+/−` zoom.
- **The cloud paints 24-bit `AttrSpec`s and needs `set_terminal_properties(colors=TRUECOLOR)`.**
  Without it urwid down-converts every colour to the nearest of 16 and the cloud goes muddy — declare
  the capability, never emit raw ANSI into the text. **256 is not enough**: its 6×6×6 cube collapsed
  the 648 colours a depth-shaded library wants into 32, so a point fading far→near arrived in six
  visible steps and the depth cue read as banding. Colours are packed `0xRRGGBB` throughout
  (`Frame.color`, the `COL_*` marker constants). `_spec()` quantises its **cache key** to 5 bits per
  channel — 24-bit is continuous, so an exact-keyed cache would grow without bound in a long session;
  32 levels per channel caps it at 32,768 and is imperceptible.
- **Absent mood axes are not fatal.** `MoodAxes.load()` returns `None` and the widget draws a rebuild
  message while the rest of the app plays on.
- **The scaffold sits behind the data by construction, not by tuning.** The frame
  `[B]` draws (ground · box + axes · marks + ground + axes) is splatted with a large
  *negative* depth bump, so its dots light their Braille cells but **lose every colour
  contest** — a cell holding a track keeps the track's colour. It carries index −1, so
  it can never claim a click either. Both are silent when broken: a frame that won
  would repaint tracks grey and nothing would raise.
- **The frame's colours are derived from the terminal background, not chosen.**
  They were first picked against the near-black `#05070b` of an archived browser
  mock, which put `COL_SCAFFOLD_FAR` at **0.93×** the luminance of the real
  terminal's background (`#15141b`) — the far half of the frame was *darker than
  the ground it sat on*, i.e. invisible, and on this terminal it was. `_lift()`
  now solves the blend exactly (luminance is linear in it) for a stated contrast
  ratio over `TERMINAL_BACKGROUND`, with an absolute floor so a pure-black
  terminal still gets a visible frame. **Change `TERMINAL_BACKGROUND` and the
  three colours re-derive.** The cloud's own points were checked at the same time
  and need nothing: the library's darkest track is luminance 75.9, so even
  depth-shaded to `SHADE_MIN` it clears this background by 1.47×.
- **"Lower opacity" is a dim colour plus stippling.** Braille has no alpha and eight
  dots share one cell colour. So a faint line is 24-bit's near-background grey
  (`COL_SCAFFOLD_FAR/NEAR`, depth-interpolated) *and* `SCAFFOLD_SPACING` — one sample
  every two dots. The cast shadow additionally carries `SHADOW_WEIGHT`, because a
  674-point mass at line brightness out-shouts the frame it sits in.
- **The box is the library's own bounding box, and it must close on screen.** It began
  as a ±2σ cube, which left **55 of 674 tracks outside it** — and a cube is the wrong
  shape here: Tone spans ±3.74σ, Saturation ±1.94, Organic ±2.82, so a cube containing
  all of it is **61% volume the library never occupies**. That matters because the box
  is *fitted to the panel*, so empty volume is paid for in how small the cloud is drawn:
  a containing cube costs 52% of the old scale, the measured box 76%. `library_extent()`
  measures it once per library (`max|coord| × 1.04`, floored so a degenerate cloud cannot
  divide by zero) and it re-fits as the collection grows.
  The fit uses **both** panel dimensions and the worst case *over azimuth* — never this
  frame's angle, which would make the cloud pulse as it spins: `reach_x` is the x–z
  diagonal, `reach_y` adds the tilted y. Fitting per axis is what makes a non-cubic box
  worth having — a wide flat box uses the panel's width (73%, was 56%) instead of paying
  for it in height. The budget is `half − 1`, not `half`: a dot rounded exactly onto
  `dot_h` is one index past the end and vanishes, a corner missing for one dot of greed.
  Zoom is applied after, so zooming in still overflows the panel, as it should.
- **The panel opens on a frame, not on `off`.** Opening bare would ship the unreadable
  state as the default and make the fix a discovery. `off` is the first stop in the
  `[B]` ring.
- **Markers are single dots distinguished by colour, never by size.** Fatter billboarded markers
  (squares, discs, rings) always face the camera, which flattens the cloud into a swimming plane. Depth
  is carried by shading; clicking uses a nearest-within-radius search, not a bigger target.

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
  half on `np.any(taste_vector)`.
- **A zero taste vector is inert in retrieval, not just scoring.** An all-zero query to `find_similar`
  returns an arbitrary slice of the library. Do not reverse this when β ramps in.
- **Loads are atomic.** `ExplorationController.load()` and `UserTaste.load()` read every field before
  assigning any. Assigning as they went left a truncated file half-applied while returning `False`.

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
each field in turn to prove the check is real. All artifact writes go through
`embeddings_io.atomic_savez` (temp-then-rename), so a killed or full-disk write never corrupts a
completed file in place.

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

Validation is behavioural, not shape-only: the **centroid must equal `mean(embeddings)`** (a stale one
would silently centre on the wrong origin), `window_offsets` must be **monotonic non-decreasing** (a
backwards index would hand one track's windows to another and still validate), and **per-window rows
must be L2-normalised**, not only the pooled embeddings.

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
| `axis_labels` | `(3,)` unicode | **Mood axes** for the vibe cloud. The auto-selected legible triad — on this library **Tone · Saturation · Organic**. |
| `axis_directions` | `(3, D)` float32 | Unit directions in the centred audio space, each `normalise(mean(high-pole text) − mean(low-pole text))`. |
| `axis_mean`, `axis_std` | `(3,)` float32 | Raw-projection calibration over the library, so a runtime coordinate is `(v·dir − mean)/std` — the vibe's z-score on that axis. |

The four `axis_*` keys are **additive**: `descriptor_bank.SCHEMA_VERSION` is unchanged, an older bank
without them still loads for the readout, and `MoodAxes.load()` returns `None` (reported) when they
are absent. They are selected and calibrated **against this library** and re-fit with a cheap
`--descriptors-only` rebuild — seconds, text tower only, no audio re-embedding. The choice of triad is
per-user: another library can favour a different one (here Intensity and Saturation correlate 0.98, so
only one of the pair can be used). `explore_mood_axes.py` imports the same selection from
`mood_axes.py`, so the offline diagnostic and the stored artifact cannot drift.

**Reading the axes:** `from mood_axes import MoodAxes; axes = MoodAxes.load()` → `axes.labels`,
`axes.directions`, `axes.coordinates(v)`. `coordinates` takes **either** a single `(512,)` centred
vector → `(3,)` (the session comet) **or** the whole centred library `(N, 512)` → `(N, 3)` in one call
(the point cloud). It unit-normalises rows and returns **z-scored** coordinates, so the cloud is
already centred on the origin with ~unit spread per axis.

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
`EmbeddingGenerator.MODEL_CACHE_MB` / `ARTIFACT_MB` are derived from this, and
`tests/test_documented_numbers.py` holds the README, `start.sh` and the pre-flight check together —
they originally stated three different sizes because nothing bound them.

Throughput detail: batching buys ~30%, threading the mel extraction buys 2×. On GPU the bottleneck is
CPU mel extraction, so `--workers` matters more than `--batch-size`; on CPU the encoder dominates.

### Vibe-cloud render cost (96×30 panel, 674 points, Braille)

```
  compute_frame (numpy raster)     0.11 ms
  markup build (python per-cell)   0.26 ms
  urwid.Text(markup).render()      1.08 ms   ← the largest single stage
  widget.render() total            1.71 ms
  loop.draw_screen() end-to-end    2.40 ms   -> ~415 fps ceiling
  terminal bytes                   6.8 KB/frame  (~400 KiB/s at 60 fps)
```

Scaling: 1.5 ms at 96×28 → 7.1 ms at 300×80 (cells dominate); `compute_frame` is 0.1 ms at 674 points
and 9.4 ms at 50,000 (so the library size is nowhere near a limit).

The scaffold's cost is the cells it lights, not its arithmetic (96×30, `widget.render()`):

```
  off                    2.25 ms    353 lit cells        0 scaffold dots
  floor+shadow           2.80 ms    544                1084
  cage+triad             3.32 ms    549                 788
  corners+floor+triad    3.38 ms    577                1088
  walls+floor+shadow     3.58 ms    766                2314   (explored, not shipped)
```

So the frame costs ~25–40% of a repaint and leaves the ~144 fps animation alarm two
thirds of its budget. `compute_frame` itself stays under 0.8 ms in every mode.

---

## 8 · Testing

`python3 -m pytest tests` — **668 tests, green, ~20 s** (measured 2026-07-24; the archived docs
said 651, which had drifted).

The suite is behavioural, not existence checks. Two of the worst defects in this project's history
survived a green suite of the latter. What that means in practice, and what to preserve:

- **Modules live in `src/`.** `conftest.py` puts `src/` on `sys.path`, so the flat imports still work.
  But a test that reads a module's **source** off disk — for an AST or regex check — must resolve
  `parent.parent / "src" / "<module>.py"`.
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
  `test_vibe_cloud.py` follows the same discipline: real canvases at six sizes including degenerate.
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
./start.sh                                                # the only entry point
python3 src/generate_embeddings.py --help
python3 src/generate_embeddings.py --stats                # raw vs centred similarity distributions
python3 src/generate_embeddings.py --describe "Bathory"   # top descriptors for a track
python3 src/generate_embeddings.py --descriptors-only     # re-fit descriptors + mood axes (seconds)
```

A full regeneration takes ~5.5 minutes and **must not be run casually while testing** — it rewrites
the artifact every downstream number depends on. `--descriptors-only` is safe and cheap.

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
| `V` | Pass — advance without changing the vibe; marks the track `»` |
| `L` | Like; press again on a liked track to un-like |
| `↑` / `↓` | Move the cursor through the session history |
| `+` / `−` | Zoom the cloud in / out |
| `Enter` | Over the cloud: reset the view. Over the history: replay the focused track |
| `B` | Cycle the frame behind the cloud: none → ground → box + axes → marks + ground + axes |
| `T` | Cycle the body pane: cloud → history → console → cloud |
| `F1`/`F2`/`F3` | Jump straight to cloud / history / console (unadvertised in the footer) |
| `,` / `.` | Volume down / up |
| `←` / `→` | Seek ∓10 s |
| `I` | Model state (descriptors, sampling, taste, exploration, weights); `↑↓` scrolls |
| `Q` | Quit |
| mouse | Over the cloud: right-drag to orbit, scroll to zoom, left-click a point to inspect it, drag the orbit-speed slider |

The footer, the README table, `start.sh`'s launch banner and both interfaces advertise this list
(F1–F3 are a deliberately unadvertised convenience). Tests drive every row through the real key
handler, and `test_documented_numbers` guards the `start.sh` banner so it cannot drift.

---

## 10 · Open, deferred, and declined

Nothing here is a defect with a known fix being ignored. These are things that were seen and decided
on.

### Deferred by choice

1. **Free-text steering.** "Something nocturnal and sparse": embed the text, take its top-20 library
   tracks, blend toward that centroid. The descriptor machinery already does this mechanically. **The
   manifold trap applies** — blend toward the *audio* centroid of matching tracks, never toward the
   raw text vector.
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

- **`mpc idle` instead of polling.** Steady state is ~12 `mpc` spawns/sec at 0.8 ms each, about 1% of
  one core. It matters only against a remote `MPD_HOST`, where each call is a TCP round trip. At depth
  1 the one queued track gives the 2 s poll minutes of runway.
- **A lock around selector/queue mutation.** The two threads share mutable state, but correctness does
  not rest on a lock: the GIL makes each `deque`/`dict`/reference op atomic, and the one fatal outcome
  (advance into an empty queue) is prevented *by construction* — every skip variant adds before it
  advances, which holds under any interleaving. The realistic residual race briefly makes the queue
  one entry deeper and self-corrects. A lock is cheap insurance, not a fix; adding one now risks a
  deadlock between the skip path and the background loop for no observed misbehaviour.
- **MPD reconnection.** `MPDController.connected` is a one-time startup gate that never reverts. This
  reads like a missing reconnect, but there is no persistent connection to lose: **every operation is
  a fresh `mpc` subprocess**, so if MPD restarts mid-session the calls fail only while it is down and
  succeed again on their own once it returns. The flag is a startup check, not a liveness signal.

### Open observations — measured, not fixed

- **The turnover schedule means little until the session settles.** The first `[N]` press turns over
  **100%** of the pool against its 5% target on a cold session. Consistent with the settling curve in
  §7, but it is the extreme of it. **Worth a measurement of its own before the schedule is retuned.**
- **The Session panel draws no content rows at 80×24** when it is the visible pane. The header, the
  packed Now Playing box and the two-row footer consume all 24. The layout renders — nothing crashes —
  but the panel is a border.
- **macOS is untested.** `start.sh` uses no bash 4+ syntax so it would probably run, but nothing has
  been run there and album art cannot work (ueberzug is X11/Wayland only). The README says Linux.
- **`[I]`'s fallback page and the urwid overlay are two pieces of code.** They agree on content; only
  the urwid one runs the session bookkeeping while open, because the fallback's own loop is the thing
  that would be blocked.
- **The Braille grid is the floor on render quality, and it was measured.** At 96×26 the grid is
  192×104 dots; at the default 0.15 rad/s orbit **95% of points are frozen in any 60 fps frame** and
  the 5% that move teleport a whole dot, because the median point moves 0.10 dots per repaint. More
  frames cannot subdivide a step the grid will not subdivide, and Braille cannot be anti-aliased in
  space — a dot is on or off, and eight dots share one colour (**53% of on-screen points render in
  another point's colour**). Solid block characters (half blocks, sextants) buy AA and per-subpixel
  colour and **look far worse** — a filled rectangle is a mosaic tile, and the cloud's appeal is small
  crisp specks. **Braille has both the finest grid and the smallest mark; it is the right primitive
  and close to the ceiling.** The browser preview in `archive/` is ~20–40× more pixels; only real
  pixels close that, and Alacritty has neither sixel nor the kitty protocol, so the sole route is an
  `ueberzugpp` overlay — at which point the cloud stops being a urwid widget (no clipping, no
  z-order, broken under tmux multi-client). Not pursued. Do not re-derive this by switching
  primitives.

### Reversal notes for the non-obvious decisions

| Decision | Why | Cost to reverse |
|---|---|---|
| Force `consume on` | Makes the refill condition `len(queue) < 2` with no `#N/M` parsing. | Moderate — the alternative is parsing `mpc status` and leaving consume alone: more code, no side effects. |
| Rank-Boltzmann sampling | Argmax reproduces a byte-identical session; score-softmax needs recalibration whenever the score scale moves. Rank is scale-invariant. | Trivial — one function. |
| Turnover reported against the skip run's start, not the previous press | "How much has changed since I started skipping" is the question actually being asked. | Trivial — one stored anchor vector. |
| Drift as a word count, not the cosine | The cosine's p10 is 0.948 over 40 real sessions. | Trivial — `drift()` returns both. Re-read §7 first: the reason is the distribution, not the presentation. |
| Drift and history stores live in the display layer | Neither is read by selection. Putting a display concern behind the display splits state across a component that cannot see it. | Low, but there is no reason to. |
| `QueueManager.requeue_next()` adds before it deletes | It already knows its track, so it has nothing to exclude. Appending first means a refusal from MPD leaves the queue untouched. | Trivial, but it reintroduces a window where a failed add leaves the queue one deep. `replace_next()` **cannot** use this order — it has to exclude the entry it is dropping. |
| `skip_turnover_schedule = (0.05, 0.20, 0.50, 0.85)` | The only *input* to the escalation — λ is solved for it at every press. Stated in units the listener can verify. | Trivial, and it is the right knob: add a row rather than fudging λ. |
| `skip_snap_from_run_length = 2` | Both directions measured: below it `snap()` overshoots; at and above it an unguarded λ for the 85% target lands below the library's p1 floor 100% of the time. | Low, but re-measure before moving it — both failure modes are silent. |
| `RMS_GATE = 0.01` | Not chosen — the window-RMS distribution is bimodal and the gate sits in the empty band. | Trivial, and re-measurable: the distribution is printed by the run and stored in the artifact metadata. |
| `STD_FLOOR_FRACTION = 0.5` | Relative to the library's own median std, so it needs no recalibration when the model, template or collection changes. | Trivial. The run prints every descriptor's std. |
| `taste_ramp_updates = 20`, `minimum_sampled_pool = 4`, `checkpoint_every_n_tracks = 5`, `minimum_mpd_coverage = 0.5` | Control constants: they shape behaviour without asserting a fact. | Trivial. |

---

## 11 · Environment

- Fedora, Python 3.14.6, numpy 1.26.4, urwid 3.0.5, transformers 5.1.0, torch/torchaudio.
- Terminal: **Alacritty 0.17** inside **tmux 3.7b**. tmux reports the client as RGB-capable
  (`terminal-features` includes `RGB`) even though the inner `TERM` is `screen-256color`, so 24-bit
  colour passes through — verified, not assumed. Alacritty supports neither sixel nor the kitty
  graphics protocol — which is why album art needs an `ueberzugpp` X11 overlay.
- **Background `#15141b`** (Alacritty + the Aura theme). It is not decoration: the vibe cloud's
  frame is derived from it (§4), and a screenshot taken on a darker ground flatters every dim
  colour in the panel by a factor of three.
- **The terminal font must cover U+2800–U+28FF**, or every cell of the cloud is drawn by whatever
  fontconfig falls back to. Two common defaults do **not** cover it — `Noto Sans Mono` (the system
  monospace default here, so an Alacritty with no `font.normal.family` set renders the whole cloud
  through a fallback) and `DejaVu Sans Mono`. `Iosevka Nerd Font Mono` does, and its 0.5 em advance
  against Noto's 0.6 em also packs the dots ~17% tighter, which is most of the difference between a
  cloud that reads as a mass and one that reads as scattered specks.
- MPD's real `music_directory` is `/mnt/storage/music`, read from `~/.config/mpd/mpd.conf` by
  `music_directory.py` rather than assumed. `/var/lib/mpd/music` is a symlink to it — the reason the
  old hardcoded default worked, and the reason it hid.
- Embedding generation runs on an RTX 3070 in ~5.5 minutes.

---

## 12 · Repo layout

```
start.sh                  the only entry point
PROJECT.md                this file
README.md                 user-facing setup + usage
src/                      every application module (flat imports, src/ on sys.path)
tests/                    668 tests; conftest.py owns the fixtures and the state isolation
data/embeddings/          track_embeddings.npz, descriptors.npz, failed.txt  (not committed)
data/state/               learned state; regenerated by listening  (not committed)
data/dj.log               the durable diagnostic copy
archive/                  frozen design references — do not edit
explore_mood_axes.py      offline: pick and analyse the mood-axis triad for a library
explore_cloud_scaffold.py offline: render every candidate cloud frame to stills
explore_cloud_preview/    offline: the browser preview — a JS port of compute_frame
                            (core.js) that verify.mjs proves cell-identical to it
scratch_*                 anything those two render; gitignored, safe to delete
vibe_cloud_demo.py        offline: the Braille render as a standalone script
render_terminal_frames.py offline: baked-frame terminal mock
```
