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
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
info()    { echo -e "${CYAN}ℹ  $*${RESET}"; }
success() { echo -e "${GREEN}✓  $*${RESET}"; }
warn()    { echo -e "${YELLOW}⚠  $*${RESET}"; }
error()   { echo -e "${RED}✗  $*${RESET}"; }
header()  { echo -e "\n${BOLD}${CYAN}$*${RESET}"; }

# ── Locate the project directory (same folder as this script) ─────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║   🎵  Adaptive Session AI DJ  ·  Launcher   ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════╝${RESET}"
echo ""

# ── 1. Python check ───────────────────────────────────────────────────────────
header "Step 1/4 · Checking Python"

if ! command -v python3 &>/dev/null; then
    error "Python 3 is not installed."
    echo "  • macOS:  brew install python"
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
header "Step 2/4 · Installing dependencies"

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
header "Step 3/4 · Checking MPD"

MPD_HOST="${MPD_HOST:-localhost}"
MPD_PORT="${MPD_PORT:-6600}"

if ! command -v mpc &>/dev/null; then
    error "MPC (MPD client) is not installed."
    echo "  • Ubuntu/Debian: sudo apt install mpd mpc"
    echo "  • macOS:         brew install mpd mpc"
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

# ── 4. Embeddings check ───────────────────────────────────────────────────────
header "Step 4/4 · Checking audio embeddings"

EMBED_FILE="$SCRIPT_DIR/data/embeddings/track_embeddings.npz"

if [ ! -f "$EMBED_FILE" ]; then
    echo ""
    warn "No embeddings file found."
    echo ""
    echo "  Embeddings capture the 'sound fingerprint' of each song so the AI"
    echo "  can find musically similar tracks. You need to generate them once."
    echo ""
    echo "  Choose an option:"
    echo "    [1] Generate REAL embeddings using CLAP (best quality, slow,"
    echo "        requires ~4 GB download on first run)"
    echo "    [2] Generate DEMO embeddings (fast, random — good for testing)"
    echo "    [Q] Quit"
    echo ""
    read -rp "  Your choice [1/2/Q]: " EMB_CHOICE

    case "${EMB_CHOICE,,}" in
        1)
            echo ""
            info "Checking for CLAP dependencies …"
            if ! python3 -c "import transformers, torch, torchaudio" &>/dev/null 2>&1; then
                info "Installing CLAP packages (this may take several minutes) …"
                PIP_ARGS="transformers>=4.30.0 torch>=2.0.0 torchaudio>=2.0.0"
                if python3 -m pip install $PIP_ARGS -q 2>/dev/null; then
                    success "CLAP packages installed"
                else
                    python3 -m pip install $PIP_ARGS -q --break-system-packages 2>/dev/null || \
                    python3 -m pip install $PIP_ARGS -q --user
                    success "CLAP packages installed"
                fi
            fi
            echo ""
            info "Generating real embeddings — this will take a while …"
            python3 generate_embeddings.py
            success "Embeddings generated!"
            ;;
        2)
            echo ""
            info "Generating demo embeddings …"
            python3 - <<'PYEOF'
import sys
sys.path.insert(0, '.')
from config import config
from track_library import generate_dummy_embeddings
import subprocess, re

result = subprocess.run(
    ['mpc', 'listall'],
    capture_output=True, text=True
)
tracks = [l.strip() for l in result.stdout.splitlines()
          if l.strip() and re.search(r'\.(mp3|flac|ogg|m4a|wav|opus)$', l, re.I)]
if not tracks:
    print("No music tracks found!", file=sys.stderr)
    sys.exit(1)
generate_dummy_embeddings(tracks, config.embeddings_file)
print(f"Generated demo embeddings for {len(tracks)} tracks.")
PYEOF
            success "Demo embeddings ready"
            ;;
        q|quit)
            echo "Bye!"
            exit 0
            ;;
        *)
            error "Invalid choice."
            exit 1
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
echo "    N             Skip track"
echo "    V             Change vibe (new direction)"
echo "    L             Like current song"
echo "    , / .         Volume down / up"
echo "    ← / →         Seek backward / forward"
echo "    ↑ / ↓         Navigate queue"
echo "    ENTER         Play selected queue item"
echo "    I             Time-context info"
echo "    Q             Quit"
echo ""
sleep 1

python3 main_tui.py
