# Adaptive Session AI DJ

A terminal-based DJ that learns your taste in real time and curates a continuously evolving queue from your own music library — no streaming service, no account, no ads. Just MPD and your files.

```
╔════════════════════════════════════════════════════════════════════╗
║                  🎵 Adaptive Session AI DJ                         ║
╠════════════════════════════════════════════════════════════════════╣
║  ♪ Now Playing                                                     ║
║  ┌─────────────┐   ▶ Playing        Vol: 72%                       ║
║  │             │                                                   ║
║  │  [cover]    │   Artist:  Floating Points                        ║
║  │             │   Album:   Promises                               ║
║  │             │   Track:   ❤ LesAlpx                              ║
║  └─────────────┘   [████████████░░░░░░░░░░░░]  3:12 / 8:44         ║
║                    Session: 14 tracks played                       ║
╠════════════════════════════════════════════════════════════════════╣
║  System Console                                                    ║
║  [14:32:01] Exploration decreased to 0.18 (6 consecutive listens)  ║
╠════════════════════════════════════════════════════════════════════╣
║  Up Next                                                           ║
║    ↓ next:  Pharoah Sanders – Karma – The Creator Has a Master Plan ║
║                                                                    ║
╠════════════════════════════════════════════════════════════════════╣
║ SPACE=Play/Pause  N=Next  L=Like  <,>=Vol  ←→=Seek  I=Info  Q=Quit ║
╚════════════════════════════════════════════════════════════════════╝
```

---

## Requirements

- **Python 3.9+**
- **MPD** (Music Player Daemon) + **MPC** — must be running with your music library indexed
- A terminal (Linux or macOS)

---

## Setup

Everything is handled by a single script:

```bash
git clone https://github.com/imaalej/mpd_tui_ai_dj
cd mpd_tui_ai_dj
bash start.sh
```

`start.sh` will check your Python version, install pip dependencies, verify MPD is reachable, locate MPD's music directory, and walk you through generating embeddings if this is your first run. After that it launches the TUI automatically.

The music directory is read from MPD's own config (`~/.config/mpd/mpd.conf`, `/etc/mpd.conf`, …). If that cannot be found you will be asked for it; set `MPD_MUSIC_DIR` to skip the question permanently.

**First run only — embeddings:** The DJ needs audio fingerprints of your library to find musically similar tracks. These come from [CLAP](https://github.com/LAION-AI/CLAP), and generating them is the one slow step: a model download, then a decode-and-encode pass over every track. Measured on this machine — 674 tracks, RTX 3070 — that pass took **5 minutes 23 seconds** and produced a 45 MB file. On CPU expect substantially longer.

Tracks are enumerated from `mpc listall`, so the embedding keys are exactly the paths MPD will be asked to play. Files MPD cannot decode are skipped and listed in `data/embeddings/failed.txt` with the reason, rather than disappearing silently.

There is no "demo mode" with random vectors. Random embeddings make every similarity, every novelty score and every learned preference meaningless while the interface keeps presenting them as insight — it looks like it works, and none of it does.

You only ever run this once. The result is saved to `data/embeddings/`.

### Manual MPD setup (if needed)

```bash
# Ubuntu/Debian
sudo apt install mpd mpc


# Point MPD at your music and build the database
mpc update
mpc status   # should show your track count
```

---

## Controls

| Key | Action |
|-----|--------|
| `Space` | Play / Pause |
| `N` | Skip track — escalates if you keep pressing it (see below) |
| `L` | Like current track |
| `,` / `.` | Volume down / up |
| `←` / `→` | Seek backward / forward 10s |
| `I` | Show model state (sampling, taste, exploration, scoring weights) |
| `Q` | Quit |

---

## How It Works

### Audio Fingerprints

Every track in your library is encoded into a 512-dimensional embedding vector that represents its sonic character — timbre, texture, energy, harmonic content. These come from [CLAP](https://github.com/LAION-AI/CLAP), a model trained to understand audio similarity. Two songs that *feel* similar will have embeddings that point in roughly the same direction in that space. All selection logic operates on these vectors; no genre tags or metadata are used.

CLAP's audio encoder takes exactly ten seconds of audio, so a track is covered by consecutive ten-second windows — the last one aligned to the end so nothing is dropped or padded with silence — and the window vectors are averaged into one. The whole track is represented, not a sample of it, and embedding the same file twice gives the identical vector. Near-silent windows are dropped; the per-window vectors are kept in the file so the pooling decision can be revisited without regenerating.

One more step matters more than it looks. CLAP's vectors occupy a narrow cone rather than the whole space: on this library, two *completely unrelated* tracks sat at a cosine similarity of 0.67. "Similar" and "unrelated" were nearly the same number, which is why the scoring weights never seemed to do much. Subtracting the library's own centroid at load time moves unrelated pairs to ≈ 0 and gives the rest of the system the range it always assumed it had.

### Two Layers of Preference

The system keeps two separate models of what you like, operating on different timescales:

**Session state** is short-term and lives only for the current listening session. It's a single vector that shifts with every track you hear — pulled toward songs you listen through and nudged away from songs you skip. It represents the *vibe right now*: where the session has been and the direction it's heading. It starts empty — before anything has played there is nothing to know, so the first track is drawn at random rather than from a direction the system invented.

**User taste** is long-term and persists between sessions. It accumulates slowly from everything you've heard across all sessions — strong pull toward explicit likes (`L`), weaker pull from full listens, and gentle pushback from skips. It doesn't reflect what you want *today*, it reflects what you've consistently come back to *across time*. New sessions start from a fresh vibe but are still anchored to your taste history.

### Track Selection

To choose the next track, the system scores a pool of candidates across four factors:

```
score = α · session_similarity
      + β · taste_similarity
      + γ · novelty
      + δ · anti_repetition_penalty
```

The weights (α, β, γ, δ) shift dynamically based on your behavior. Skip a few songs and the system increases `γ` (novelty) to try something different. Listen through several tracks and it lowers `γ`, leaning harder on what it already knows you like.

`β` is also *earned*. A new listener has no taste history, so the taste term starts at zero weight and ramps up over the first 20 updates, with the unearned weight going to the session term. Until then you are driven purely by what you are playing right now, which is the only thing known about you.

### Skipping, and what it means

`N` is the only steering key. A single press says "not this song" — the session vector is nudged away from it and the queued track behind it is dropped and re-picked immediately.

Keep pressing and it escalates, because *n* consecutive rejections is the system observing that the neighbourhood is wrong — better evidence than a separate key you would have to reach for after diagnosing your own dissatisfaction. Each press targets a fraction of the candidate pool that must come out different:

| consecutive skips | turns over | reads as |
|---|---|---|
| 1 | 5% | "not this song" |
| 2 | 20% | "not this corner either" |
| 3 | 50% | "this is the wrong direction" |
| 4+ | 85% | full reset |

The important part is that those are *targets*, and the magnitude of the move is solved for them at each press rather than being a constant someone picked. "85% of what I would have heard is now different" is something you can check; a rotation of 0.15 radians is not. The console reports what each press actually achieved.

From the second press onward the session vector is projected back onto your library — replaced with the centre of its 25 nearest real tracks — so however hard you push, it cannot drift into a region no music occupies. A full listen ends the run and the escalation resets.

*(An earlier `V` key existed for this. It blended the session vector halfway toward a random direction, which measurably landed it outside the music and turned over less than two presses of `N` do. It is gone.)*

### Exploration vs Exploitation

A single **exploration value** (0.1–0.7) controls how adventurous the DJ is. It increases with every skip and decreases with every full listen, meaning it self-calibrates to your engagement. If you're in a zone and letting tracks run, it narrows in. If you're skipping around, it opens up and reaches further from your established taste.

### Describing the session

The session line currently shows a track count and nothing else.

It used to show a mood phrase — *"focused cohesive vibe, deep in the zone"* — assembled from three heuristics. All three were invented against scales nobody measured: the mood word came from an entropy-like quantity that is always ≈ 55 for a 512-dimensional unit vector, so it returned *eclectic* every time and its other two branches were unreachable; the *warming up → building → deep in the zone* stage word was a track counter in a costume. A blank line is honest and a counter is a fact; the phrase was neither.

The replacement is half built. The data exists: 49 descriptor prompts — energy, affect, texture, rhythm, setting, instrumentation — embedded with CLAP's *text* encoder and stored with each word's mean and standard deviation over this library, so a score reads as a z-score against your own collection and *"hypnotic"* means "unusually hypnotic **for your music**". You can already ask for it from the command line:

```bash
python3 generate_embeddings.py --describe "Arctic Monkeys"
```

Wiring it to the session line is the next piece of work. See `PROJECT_AUDIT.md` §H1.

Exactly **one** track sits in the queue ahead of the current one, refilled as each song ends so playback never stops. That depth is deliberate: with ten queued tracks, every one of them had been scored under the weights that existed ten songs ago, so a skip or a like was inaudible until they drained. At depth one, feedback changes what plays *next*.

Candidates are drawn from a pool of 100 nearest neighbours in embedding space and re-ranked by the full scoring function. The winner is not simply the top-scoring track — one is drawn by Boltzmann sampling over **rank**, `p(i) ∝ exp(−i/τ)`, with `τ` set by the exploration value and readable in `[I]` as "choosing from ~top 8". A strict argmax would replay the identical evening from the same starting state every time, which for a system built around an evolving session is the failure mode. Recently played tracks are excluded for at least 20 songs, and that exclusion now survives a restart.

### What Persists Between Sessions

- Your **taste vector** — accumulated from all your likes, listens, and skips
- Your **exploration level** — picks up roughly where you left off
- Your **feedback history** — a log of every like, skip, and listen event

The session vector itself resets each time. Every session begins fresh but informed by everything before it.

---

## Configuration

All parameters live in `config.py`. Notable ones:

```python
# Scoring weights (must sum to 1.0)
weight_session_similarity = 0.4   # pull toward current session vibe
weight_taste_similarity   = 0.3   # pull toward long-term taste
weight_novelty            = 0.2   # preference for unheard territory
weight_anti_repetition    = 0.1   # penalty for recently played tracks

# Exploration range
exploration_min = 0.1   # floor — always some novelty
exploration_max = 0.7   # ceiling — never fully random

# Session evolution speed
session_decay_factor  = 0.85   # how quickly old tracks fade from session context

# Selection temperature — the effective number of candidates in play
tau_max = 15.0   # at the exploration ceiling; ~1 at the floor

# Skip escalation: the fraction of the candidate pool each consecutive
# press of N must turn over.  The repulsion magnitude is solved for these,
# not declared, so they need no recalibration if the library changes.
skip_turnover_schedule = (0.05, 0.20, 0.50, 0.85)

# Queue depth ahead of the current track
queue_lookahead = 1

# Taste update rates
taste_update_like         =  0.10   # explicit like
taste_update_full_listen  =  0.02   # passive full listen
taste_update_skip_penalty = -0.05   # skip
```

You can also set `MPD_HOST` and `MPD_PORT` as environment variables if your MPD isn't on localhost:

```bash
MPD_HOST=192.168.1.10 bash start.sh
```
