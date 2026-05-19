#!/bin/bash
# ═══════════════════════════════════════════════════════
#  VERITY — Deception Analysis System
#  Launch script for macOS
# ═══════════════════════════════════════════════════════

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BOLD="\033[1m"
GREEN="\033[0;32m"
AMBER="\033[0;33m"
RED="\033[0;31m"
CYAN="\033[0;36m"
RESET="\033[0m"

echo ""
echo "${BOLD}══════════════════════════════════════════════════════${RESET}"
echo "${BOLD}  VERITY — Deception Analysis System${RESET}"
echo "  Ekman–Friesen Micro-Expression Framework"
echo "${BOLD}══════════════════════════════════════════════════════${RESET}"
echo ""

# ── Check Python ──────────────────────────────────────
if ! command -v python3 &> /dev/null; then
    echo "${RED}✗ Python 3 not found.${RESET}"
    echo "  Install via: brew install python3"
    exit 1
fi
echo "${GREEN}✓${RESET} Python $(python3 --version | awk '{print $2}') found"

# ── Virtual environment ───────────────────────────────
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "${AMBER}→ Creating virtual environment...${RESET}"
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
echo "${GREEN}✓${RESET} Virtual environment active"

# ── Install / verify dependencies ─────────────────────
echo "${AMBER}→ Checking dependencies...${RESET}"
pip install -q -r requirements.txt
echo "${GREEN}✓${RESET} Dependencies ready"

# ── Camera permission check (macOS) ───────────────────
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "${CYAN}ℹ${RESET}  macOS: if camera prompt appears, click Allow"
fi

# ── Launch server ──────────────────────────────────────
PORT=5050
echo ""
echo "${GREEN}▶ Starting VERITY on http://localhost:${PORT}${RESET}"
echo "  Press Ctrl+C to stop"
echo "${BOLD}══════════════════════════════════════════════════════${RESET}"
echo ""

# Auto-open browser after 1.5s
(sleep 1.5 && open "http://localhost:${PORT}" 2>/dev/null || xdg-open "http://localhost:${PORT}" 2>/dev/null) &

python3 app.py
