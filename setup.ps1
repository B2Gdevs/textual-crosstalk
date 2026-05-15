# conduit_proj — one-shot setup + run script (Windows PowerShell)
# Usage:
#   .\setup.ps1            # setup + run
#   .\setup.ps1 -SetupOnly  # setup only

param(
  [switch]$SetupOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot

Write-Host ""
Write-Host "=== conduit_proj setup ===" -ForegroundColor Cyan

# 1. Python version check
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { throw "python not on PATH. Install Python 3.11+ from python.org" }
$pyver = & python --version 2>&1
Write-Host "  ✓ $pyver"

# 2. Create venv if missing
$venvPath = Join-Path $RepoRoot ".venv"
if (-not (Test-Path $venvPath)) {
  Write-Host "  → creating .venv..."
  & python -m venv .venv
}
Write-Host "  ✓ .venv ready"

# 3. Activate + pip install
$activate = Join-Path $venvPath "Scripts\Activate.ps1"
. $activate

Write-Host "  → installing requirements..."
& python -m pip install --upgrade pip --quiet
& python -m pip install -r requirements.txt --quiet
Write-Host "  ✓ deps installed"

# 4. Verify .env.local
$envLocal = Join-Path $RepoRoot ".env.local"
if (-not (Test-Path $envLocal)) {
  Write-Host "  ! .env.local missing — copy .env.example and fill in keys" -ForegroundColor Yellow
  Copy-Item ".env.example" ".env.local"
  Write-Host "    Created .env.local from .env.example — edit it before running"
  exit 1
}

# Check required keys are non-empty
$envContent = Get-Content $envLocal
$required = @("DEEPGRAM_API_KEY", "ELEVENLABS_API_KEY")
$oneOf = @("OPENROUTER_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY")
$missing = @()
foreach ($k in $required) {
  $line = $envContent | Where-Object { $_ -match "^$k=." }
  if (-not $line) { $missing += $k }
}
$hasLLM = $false
foreach ($k in $oneOf) {
  $line = $envContent | Where-Object { $_ -match "^$k=." }
  if ($line) { $hasLLM = $true }
}
if (-not $hasLLM) { $missing += "(one of: $($oneOf -join ', '))" }

if ($missing.Count -gt 0) {
  Write-Host "  ! Missing env keys in .env.local:" -ForegroundColor Yellow
  $missing | ForEach-Object { Write-Host "      $_" -ForegroundColor Yellow }
  Write-Host "  Edit .env.local then re-run." -ForegroundColor Yellow
  exit 1
}
Write-Host "  ✓ .env.local validated"

# 5. Ensure data dir exists
$dataDir = Join-Path $RepoRoot "data"
if (-not (Test-Path $dataDir)) { New-Item -ItemType Directory -Path $dataDir | Out-Null }
Write-Host "  ✓ data/ ready"

if ($SetupOnly) {
  Write-Host ""
  Write-Host "Setup complete. Run: python main.py" -ForegroundColor Green
  exit 0
}

# 6. Run
Write-Host ""
Write-Host "=== Starting Conduit TUI ===" -ForegroundColor Cyan
Write-Host "  q = quit · r = reset turn · s = stop TTS"
Write-Host ""
& python main.py
