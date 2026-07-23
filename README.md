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
║  Upcoming Queue  [↑↓ navigate · ENTER play]                        ║
║    ▶ Floating Points – Promises – LesAlpx                          ║
║    1. Pharoah Sanders – Karma – The Creator Has a Master Plan      ║
║  » 2. Nils Frahm – Spaces – Said and Done                          ║
║    3. ❤ Jon Hopkins – Immunity – Open Eye Signal                   ║
║    4. Four Tet – There Is Love In You – Love Cry                   ║
╠════════════════════════════════════════════════════════════════════╣
║ SPACE=Play/Pause  N=Next  V=Vibe  L=Like  <,>=Vol  ←→=Seek  Q=Quit ║
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

`start.sh` will check your Python version, install pip dependencies, verify MPD is reachable, and walk you through generating embeddings if this is your first run. After that it launches the TUI automatically.

**First run only — embeddings:** The DJ needs audio fingerprints of your library to find musically similar tracks. These come from [CLAP](https://github.com/LAION-AI/CLAP) and take a while to generate on first run (a model download plus one pass over your library).

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
| `N` | Skip track (stays in current vibe) |
| `V` | Skip vibe (hard shift to a new direction) |
| `L` | Like current track |
| `,` / `.` | Volume down / up |
| `←` / `→` | Seek backward / forward 10s |
| `↑` / `↓` | Navigate the queue |
| `Enter` | Play selected queue item immediately |
| `I` | Show model state (taste, exploration, current scoring weights) |
| `Q` | Quit |

---

## How It Works

### Audio Fingerprints

Every track in your library is encoded into a 512-dimensional embedding vector that represents its sonic character — timbre, texture, energy, harmonic content. These come from [CLAP](https://github.com/LAION-AI/CLAP), a model trained to understand audio similarity. Two songs that *feel* similar will have embeddings that point in roughly the same direction in that space. All selection logic operates on these vectors; no genre tags or metadata are used.

### Two Layers of Preference

The system keeps two separate models of what you like, operating on different timescales:

**Session state** is short-term and lives only for the current listening session. It's a single vector that shifts with every track you hear — pulled toward songs you listen through and nudged away from songs you skip. It represents the *vibe right now*: where the session has been and the direction it's heading. When you press `V` (skip vibe), the session vector is rotated by a large random angle, immediately breaking from the current trajectory and forcing the queue to recalculate from a new starting point.

**User taste** is long-term and persists between sessions. It accumulates slowly from everything you've heard across all sessions — strong pull toward explicit likes (`L`), weaker pull from full listens, and gentle pushback from skips. It doesn't reflect what you want *today*, it reflects what you've consistently come back to *across time*. New sessions start from a fresh vibe but are still anchored to your taste history.

### Track Selection

For each slot in the queue, the system scores every candidate track across four factors, then picks the best:

```
score = α · session_similarity
      + β · taste_similarity
      + γ · novelty
      + δ · anti_repetition_penalty
```

The weights (α, β, γ, δ) shift dynamically based on your behavior. Skip a few songs and the system increases `γ` (novelty) to try something different. Listen through several tracks and it lowers `γ`, leaning harder on what it already knows you like. Press `V` and it sets novelty to near-maximum and rebuilds the queue from scratch.

### Exploration vs Exploitation

A single **exploration value** (0.1–0.7) controls how adventurous the DJ is. It increases with every skip and decreases with every full listen, meaning it self-calibrates to your engagement. If you're in a zone and letting tracks run, it narrows in. If you're skipping around, it opens up and reaches further from your established taste.

### Describing the session

The session line currently shows a track count and nothing else.

It used to show a mood phrase — *"focused cohesive vibe, deep in the zone"* — assembled from three heuristics. All three were invented against scales nobody measured: the mood word came from an entropy-like quantity that is always ≈ 55 for a 512-dimensional unit vector, so it returned *eclectic* every time and its other two branches were unreachable; the *warming up → building → deep in the zone* stage word was a track counter in a costume. A blank line is honest and a counter is a fact; the phrase was neither.

The replacement is real and specified: a bank of descriptor prompts embedded with CLAP's *text* encoder, scored against the session vector as a z-score relative to this library's own distribution, so *"hypnotic"* means "unusually hypnotic **for your collection**". See `PROJECT_AUDIT.md` §H1.

The queue is kept at 10 tracks and refilled dynamically so playback never stops. Each candidate is drawn from a pool of 100 nearest-neighbor tracks in embedding space, then re-ranked by the full scoring function. Recently played tracks are excluded for at least 20 songs to prevent repetition.

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
vibe_shift_magnitude  = 0.5    # how hard V rotates the session vector

# Taste update rates
taste_update_like         =  0.10   # explicit like
taste_update_full_listen  =  0.02   # passive full listen
taste_update_skip_penalty = -0.05   # skip
```

You can also set `MPD_HOST` and `MPD_PORT` as environment variables if your MPD isn't on localhost:

```bash
MPD_HOST=192.168.1.10 bash start.sh
```
