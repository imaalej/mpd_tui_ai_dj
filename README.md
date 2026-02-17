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
║                    Vibe: focused cohesive vibe, deep in the zone   ║
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

**First run only — embeddings:** The DJ needs audio fingerprints of your library to find musically similar tracks. You'll be offered two options:

- **Real embeddings** via [CLAP](https://github.com/LAION-AI/CLAP) — audio-based, best quality, slow (~1–60 min depending on library size, requires ~4 GB model download)
- **Demo embeddings** — random vectors, instant, good enough to try the interface

You only ever run this once. The result is saved to `data/embeddings/`.

### Manual MPD setup (if needed)

```bash
# Ubuntu/Debian
sudo apt install mpd mpc

# macOS
brew install mpd mpc

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
| `I` | Show time-context stats |
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

The exploration level also picks up a **day-of-week modifier**: slightly more conservative on weekdays, slightly more adventurous on weekends.

### Mood & Narrative

The session vector's trajectory is tracked over time and used to generate the vibe description you see on screen. If recent tracks have been highly similar to each other (cosine similarity > 0.85), the vibe reads as *focused*. More varied and it shifts to *flowing*, *drifting*, or *exploring*. The description also reflects how deep you are into the session (*warming up → building → deep in the zone*) and whether the overall spread of your session is *cohesive*, *diverse*, or *eclectic*.

The queue is kept at 10 tracks and refilled dynamically so playback never stops. Each candidate is drawn from a pool of 100 nearest-neighbor tracks in embedding space, then re-ranked by the full scoring function. Recently played tracks are excluded for at least 20 songs to prevent repetition.

### Time Context

The system tracks which kinds of music you tend to listen to at different times of day — morning, midday, afternoon, evening, night — and adds a small time-similarity bonus to tracks that fit the current period based on your history. This builds up passively over many sessions and has a gentle influence (weighted at 15%) so it never overrides your explicit preferences.

### What Persists Between Sessions

- Your **taste vector** — accumulated from all your likes, listens, and skips
- Your **exploration level** — picks up roughly where you left off
- Your **time context** — which music you listen to at which hours
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
