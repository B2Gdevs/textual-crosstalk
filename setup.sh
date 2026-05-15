#!/usr/bin/env bash
# conduit_proj — one-shot setup + run script (Git Bash / WSL / macOS / Linux)
# Usage:
#   ./setup.sh           # setup + run
#   ./setup.sh --setup-only

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

setup_only=0
[[ "${1:-}" == "--setup-only" ]] && setup_only=1

echo ""
echo "=== conduit_proj setup ==="

# 1. Python check
if ! command -v python >/dev/null 2>&1; then
  echo "ERROR: python not on PATH. Install Python 3.11+." >&2
  exit 1
fi
echo "  ✓ $(python --version)"

# 2. venv
if [[ ! -d ".venv" ]]; then
  echo "  → creating .venv..."
  python -m venv .venv
fi
echo "  ✓ .venv ready"

# 3. Activate (Git Bash on Windows uses Scripts, *nix uses bin)
if [[ -f ".venv/Scripts/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/Scripts/activate
else
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "  → installing requirements..."
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt --quiet
echo "  ✓ deps installed"

# 4. Env validation
if [[ ! -f ".env.local" ]]; then
  echo "  ! .env.local missing — copying from .env.example"
  cp .env.example .env.local
  echo "  Edit .env.local then re-run." >&2
  exit 1
fi

missing=()
for k in DEEPGRAM_API_KEY ELEVENLABS_API_KEY; do
  if ! grep -qE "^${k}=." .env.local; then
    missing+=("$k")
  fi
done

has_llm=0
for k in OPENROUTER_API_KEY GROQ_API_KEY OPENAI_API_KEY; do
  if grep -qE "^${k}=." .env.local; then has_llm=1; fi
done
[[ $has_llm == 0 ]] && missing+=("(one of: OPENROUTER_API_KEY, GROQ_API_KEY, OPENAI_API_KEY)")

if [[ ${#missing[@]} -gt 0 ]]; then
  echo "  ! Missing env keys in .env.local:" >&2
  for m in "${missing[@]}"; do echo "      $m" >&2; done
  exit 1
fi
echo "  ✓ .env.local validated"

# 5. data dir
mkdir -p data
echo "  ✓ data/ ready"

if [[ $setup_only == 1 ]]; then
  echo ""
  echo "Setup complete. Run: python main.py"
  exit 0
fi

# 6. Run
echo ""
echo "=== Starting Conduit TUI ==="
echo "  q = quit · r = reset turn · s = stop TTS"
echo ""
python main.py
