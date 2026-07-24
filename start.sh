#!/usr/bin/env bash
# =============================================================================
#  start.sh — Adaptive Session AI DJ  ·  One-click setup and launch
# =============================================================================
#  Run this script to install dependencies, generate embeddings (if needed),
#  and start the TUI. You don't need to know Python — just run:
#
#      bash start.sh
#
#  Requirements:
#    • Python 3.9+  (check with: python3 --version)
#    • MPD + MPC    (install: sudo apt install mpd mpc  OR  brew install mpd mpc)
#    • Your music library already configured in MPD
# =============================================================================

set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'
info() { echo -e "${CYAN}ℹ  $*${RESET}"; }
success() { echo -e "${GREEN}✓  $*${RESET}"; }
warn() { echo -e "${YELLOW}⚠  $*${RESET}"; }
error() { echo -e "${RED}✗  $*${RESET}"; }
header() { echo -e "\n${BOLD}${CYAN}$*${RESET}"; }

# ── Locate the project directory (same folder as this script) ─────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║   🎵  Adaptive Session AI DJ  ·  Launcher   ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════╝${RESET}"
echo ""

# ── 1. Python check ───────────────────────────────────────────────────────────
header "Step 1/5 · Checking Python"

if ! command -v python3 &>/dev/null; then
  error "Python 3 is not installed."
  echo "  • Ubuntu: sudo apt install python3 python3-pip"
  exit 1
fi

PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 9 ]; }; then
  error "Python 3.9+ is required (you have $PY_VER)."
  exit 1
fi
success "Python $PY_VER found"

# ── 2. pip dependencies ───────────────────────────────────────────────────────
header "Step 2/5 · Installing dependencies"

if ! python3 -c "import numpy, urwid, PIL, mutagen" &>/dev/null 2>&1; then
  info "Installing Python packages from requirements.txt …"
  # Try normal install first; fall back to --break-system-packages on newer
  # Debian/Ubuntu systems that enforce PEP 668.
  if python3 -m pip install -r requirements.txt -q 2>/dev/null; then
    success "Packages installed"
  elif python3 -m pip install -r requirements.txt -q --break-system-packages 2>/dev/null; then
    success "Packages installed (system Python)"
  else
    # Last resort: user scheme
    python3 -m pip install -r requirements.txt -q --user
    success "Packages installed (user scheme)"
  fi
else
  success "All dependencies already installed"
fi

# ── 3. MPD / MPC check ───────────────────────────────────────────────────────
header "Step 3/5 · Checking MPD"

MPD_HOST="${MPD_HOST:-localhost}"
MPD_PORT="${MPD_PORT:-6600}"

if ! command -v mpc &>/dev/null; then
  error "MPC (MPD client) is not installed."
  echo "  • Ubuntu/Debian: sudo apt install mpd mpc"
  echo "  • Arch:          sudo pacman -S mpd mpc"
  echo ""
  echo "  After installing, add your music directory to MPD's config,"
  echo "  start MPD, and run 'mpc update' to build the database."
  exit 1
fi

if ! mpc -h "$MPD_HOST" -p "$MPD_PORT" status &>/dev/null; then
  error "Cannot connect to MPD at $MPD_HOST:$MPD_PORT"
  echo ""
  echo "  Make sure MPD is running:"
  echo "    • systemctl start mpd   (Linux with systemd)"
  echo "    • mpd                   (manual)"
  echo ""
  echo "  Then verify with: mpc status"
  echo "  You can also set MPD_HOST / MPD_PORT environment variables:"
  echo "    MPD_HOST=192.168.1.10 bash start.sh"
  exit 1
fi

TRACK_COUNT=$(mpc -h "$MPD_HOST" -p "$MPD_PORT" listall 2>/dev/null | wc -l | tr -d ' ')
success "Connected to MPD ($TRACK_COUNT tracks in library)"

if [ "$TRACK_COUNT" -eq 0 ]; then
  warn "No tracks found in MPD database."
  echo "  Run 'mpc update' after pointing MPD at your music folder."
  exit 1
fi

# ── 4. Music directory ────────────────────────────────────────────────────────
# Audit M3: this used to default to /var/lib/mpd/music, unvalidated, and worked
# on the developer's machine only because that path happened to be a symlink.
# Detection reads MPD's own config; if that fails, ask rather than guess — a
# wrong directory wastes an entire embedding-generation run.
header "Step 4/5 · Locating MPD's music directory"

if MUSIC_DIR=$(python3 src/music_directory.py 2>/dev/null); then
  success "Music directory: $MUSIC_DIR"
else
  warn "Could not read MPD's music_directory from its config."
  echo ""
  echo "  This is the folder MPD plays from — the one its config calls"
  echo "  music_directory. Album art and tag reading need it, and embedding"
  echo "  generation cannot decode a single file without it."
  echo ""
  read -rp "  Path to your music folder: " MUSIC_DIR
  MUSIC_DIR="${MUSIC_DIR/#\~/$HOME}"
  if [ ! -d "$MUSIC_DIR" ]; then
    error "Not a directory: $MUSIC_DIR"
    exit 1
  fi
  export MPD_MUSIC_DIR="$MUSIC_DIR"
  success "Using $MUSIC_DIR for this run"
  info "This only applies to the current launch. To make it stick, add"
  info "  export MPD_MUSIC_DIR=\"$MUSIC_DIR\"   to your shell profile."
fi

# ── 5. Embeddings check ───────────────────────────────────────────────────────
header "Step 5/5 · Checking audio embeddings"

EMBED_FILE="$SCRIPT_DIR/data/embeddings/track_embeddings.npz"

if [ ! -f "$EMBED_FILE" ]; then
  echo ""
  warn "No embeddings file found."
  echo ""
  echo "  Embeddings capture the 'sound fingerprint' of each song so the AI"
  echo "  can find musically similar tracks. You need to generate them once."
  echo ""
  # Measured, not estimated — this script used to say "~4 GB download" while the
  # README said "~1 GB" and the pre-flight check demanded 700 MB (audit M7).
  echo "  What this costs, measured on a 674-track library:"
  echo "    • 1.15 GB one-time model download (cached in ~/.cache/huggingface)"
  echo "    • 46 MB of embeddings in data/embeddings/"
  echo "    • about 5 minutes on an RTX 3070, or 25-35 minutes on CPU"
  echo ""
  echo "  There is no demo/random option: random vectors make every similarity,"
  echo "  every novelty score and every learned preference meaningless while the"
  echo "  interface keeps presenting them as insight."
  echo ""
  # Lowercase via tr, not ${VAR,,} — the latter is a bash 4+ expansion and a
  # hard syntax error on the bash 3.2 that macOS ships (audit M7).
  read -rp "  Generate CLAP embeddings now? [y/N]: " EMB_CHOICE
  EMB_CHOICE=$(printf '%s' "$EMB_CHOICE" | tr '[:upper:]' '[:lower:]')

  case "$EMB_CHOICE" in
  y | yes)
    echo ""
    info "Checking for CLAP dependencies …"
    if ! python3 -c "import transformers, torch, torchaudio" &>/dev/null 2>&1; then
      info "Installing CLAP packages (this may take several minutes) …"
      PIP_ARGS="transformers>=4.30.0 torch>=2.0.0 torchaudio>=2.0.0"
      if python3 -m pip install $PIP_ARGS -q 2>/dev/null; then
        success "CLAP packages installed"
      else
        python3 -m pip install $PIP_ARGS -q --break-system-packages 2>/dev/null ||
          python3 -m pip install $PIP_ARGS -q --user
        success "CLAP packages installed"
      fi
    fi
    echo ""
    info "Generating real embeddings — this will take a while …"
    python3 src/generate_embeddings.py
    success "Embeddings generated!"
    ;;
  *)
    echo ""
    echo "  Nothing to play against. Run this when you are ready:"
    echo "      python3 src/generate_embeddings.py"
    exit 0
    ;;
  esac
else
  success "Embeddings file found"
fi

# ── Launch ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}Everything looks good — starting the DJ! 🎶${RESET}"
echo ""
echo "  Controls inside the TUI:"
echo "    SPACE         Play / Pause"
echo "    N             Skip track (escalates on consecutive presses)"
echo "    V             Pass — move on without changing the vibe"
echo "    L             Like current song"
echo "    ↑ / ↓         Move the history cursor"
echo "    + / −         Zoom the cloud in / out"
echo "    ENTER         Reset the cloud view  (or Replay the focused track)"
echo "    T             Cycle panes: cloud / history / console"
echo "    F1 / F2 / F3  Jump straight to cloud / history / console"
echo "    mouse         Right-drag orbit · scroll zoom · click a point · slider sets orbit speed"
echo "    , / .         Volume down / up"
echo "    ← / →         Seek backward / forward"
echo "    I             Model info"
echo "    Q             Quit"
echo ""
sleep 1

python3 src/main_tui.py
