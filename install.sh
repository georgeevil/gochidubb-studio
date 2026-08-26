#!/usr/bin/env bash
set -e

G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; C='\033[0;36m'; B='\033[1m'; N='\033[0m'
ok()   { echo -e "${G}[✓]${N} $*"; }
warn() { echo -e "${Y}[!]${N} $*"; }
err()  { echo -e "${R}[✗]${N} $*"; }
step() { echo -e "\n${C}${B}━━ $* ━━${N}"; }

cd "$(dirname "$0")"

echo -e "${B}"
echo "╔══════════════════════════════════════════════╗"
echo "║    GoChiDUBB Studio — AI Video Dubbing       ║"
echo "║    by George Chigrichenko (@georgeevil)      ║"
echo "╚══════════════════════════════════════════════╝"
echo -e "${N}"

# ── Python ──
step "Checking Python"
PY_V=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
PY_MAJ=${PY_V%.*}; PY_MIN=${PY_V#*.}
# 3.10 is a real floor (voxcpm and torch both declare >=3.10). The ceiling is
# 3.14, and it comes from `spaces` (<3.15), not from VoxCPM — which declares no
# upper bound at all. 3.13+ resolves the same dependency set as 3.12, but has
# had less real-world use here, so it warns rather than blocks.
if [[ "$PY_MAJ" != "3" || "$PY_MIN" -lt 10 || "$PY_MIN" -gt 14 ]]; then
    err "Python $PY_V — need 3.10-3.14"
    exit 1
fi
if [[ "$PY_MIN" -gt 12 ]]; then
    warn "Python $PY_V works but is less tested here — 3.11 or 3.12 if you hit trouble"
fi
ok "Python $PY_V"

# ── FFmpeg ──
step "FFmpeg"
if command -v ffmpeg &>/dev/null; then 
    ok "Found: $(ffmpeg -version | head -1)"
else
    warn "Installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then 
        if command -v brew &>/dev/null; then
            brew install ffmpeg
        else
            err "Homebrew not found. Install ffmpeg manually: https://ffmpeg.org"
            exit 1
        fi
    elif command -v apt-get &>/dev/null; then 
        sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg
    elif command -v dnf &>/dev/null; then 
        sudo dnf install -y ffmpeg
    elif command -v pacman &>/dev/null; then
        sudo pacman -S --noconfirm ffmpeg
    else 
        err "Install FFmpeg manually: https://ffmpeg.org"; exit 1
    fi
    ok "Installed"
fi

# ── venv + core packages ──
step "Python environment"
[ ! -d "venv" ] && python3 -m venv venv && ok "venv created" || ok "venv exists"
source venv/bin/activate
pip install --upgrade pip wheel setuptools -q

step "Core packages"
pip install -q fastapi "uvicorn[standard]" python-multipart httpx soundfile numpy pydub nltk yt-dlp edge-tts
ok "Core installed"

# ── PyTorch ──
step "PyTorch"
if command -v nvidia-smi &>/dev/null; then
    ok "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    if ! python3 -c "import torch" 2>/dev/null; then
        pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    warn "macOS detected — installing CPU PyTorch (MPS fallback enabled)"
    pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    # Set MPS fallback environment variable
    export PYTORCH_ENABLE_MPS_FALLBACK=1
else
    warn "No GPU — CPU PyTorch"
    pip install -q torch torchvision torchaudio
fi
ok "PyTorch ready"

# ── Whisper ──
step "Whisper"
# Install both whisperx and faster-whisper for flexibility
pip install -q faster-whisper 2>/dev/null && ok "faster-whisper installed" || warn "faster-whisper install failed"
# Try whisperx but don't fail if it doesn't install
pip install -q whisperx 2>/dev/null || warn "whisperx install failed (optional, using faster-whisper)"
# Install additional dependencies
pip install -q transformers==4.36.0 2>/dev/null || warn "transformers install failed"

# ── VoxCPM2 ──
step "VoxCPM2 (voice cloning)"
if [[ "$OSTYPE" == "darwin"* ]]; then
    # voxcpm 2.0.3 fixed the MPS audio-quality problems this recommendation was
    # written for, but that has not been re-confirmed on Apple Silicon here, so
    # the safer option stays the default. See CLD-188.
    warn "macOS detected — VoxCPM2's MPS issues were fixed in voxcpm 2.0.3, but"
    warn "that is unconfirmed on Apple Silicon. F5-TTS remains the safe default."
    read -p "$(echo -e ${Y}'Install F5-TTS instead? (the tested path on macOS) [Y/n]: '${N})" -n 1 -r; echo ""
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        pip install -q f5-tts 2>/dev/null && ok "F5-TTS installed" || warn "F5-TTS install failed"
    else
        read -p "$(echo -e ${Y}'Install VoxCPM2? Downloads ~5GB on first use [y/N]: '${N})" -n 1 -r; echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            pip install -q voxcpm && ok "VoxCPM2 installed" || warn "VoxCPM2 install failed — edge-tts fallback"
        fi
    fi
else
    read -p "$(echo -e ${Y}'Install VoxCPM2? Downloads ~5GB on first use [Y/n]: '${N})" -n 1 -r; echo ""
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        pip install -q voxcpm && ok "VoxCPM2 installed" || warn "VoxCPM2 install failed — edge-tts fallback"
    fi
fi

# ── LM Studio (replaces Ollama) ──
step "Translation Backend"
echo -e "${C}${B}Choose translation backend:${N}"
echo "  1) LM Studio  — RECOMMENDED (OpenAI-compatible, supports latest models)"
echo "  2) Ollama     — Legacy (still supported)"
read -p "$(echo -e ${C}'Select [1, default=1]: '${N})" -n 1 -r; echo ""

if [[ "${REPLY:-1}" == "1" ]]; then
    ok "Using LM Studio"
    echo ""
    echo -e "${C}${B}LM Studio Setup:${N}"
    echo "  1. Download LM Studio from: https://lmstudio.ai"
    echo "  2. Open LM Studio and go to Settings → Developer"
    echo "  3. Enable 'Local Server' (default port: 1234)"
    echo "  4. Download a model (e.g., Qwen 2.5 14B Instruct)"
    echo ""
    read -p "$(echo -e ${Y}'Is LM Studio running with local server enabled? [Y/n]: '${N})" -n 1 -r; echo ""
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        warn "Please start LM Studio and enable local server before running GoChiDUBB"
    else
        # Test connection to LM Studio
        if curl -s http://localhost:1234/v1/models &>/dev/null; then
            ok "LM Studio connection successful!"
            # Get available models
            MODELS=$(curl -s http://localhost:1234/v1/models | python3 -c "import sys, json; data=json.load(sys.stdin); print('\n'.join([m['id'] for m in data.get('data', [])]))" 2>/dev/null)
            if [ -n "$MODELS" ]; then
                echo -e "${G}Available models:${N}"
                echo "$MODELS" | head -10
                if [ $(echo "$MODELS" | wc -l) -gt 10 ]; then
                    echo "... and $(($(echo "$MODELS" | wc -l) - 10)) more"
                fi
            else
                warn "No models found in LM Studio. Please download a model."
            fi
        else
            warn "Could not connect to LM Studio. Make sure it's running with local server enabled."
        fi
    fi
    # Set default .env for LM Studio
    cat > .env.example <<'EOF'
# LM Studio Configuration
USE_LM_STUDIO=1
LM_STUDIO_URL=http://localhost:1234/v1
# Auto-detect model by leaving empty, or specify:
# LM_STUDIO_MODEL=Qwen/Qwen2.5-14B-Instruct

# macOS compatibility
PYTORCH_ENABLE_MPS_FALLBACK=1
TORCHCODEC_USE_TORCHAUDIO=1
GOCHIDUBB_USE_WHISPERX=0
WHISPER_MODEL=base
GOCHIDUBB_WARMUP=0
EOF
else
    ok "Using Ollama"
    # ── Ollama ──
    if command -v ollama &>/dev/null; then 
        ok "Ollama found: $(ollama --version)"
    else
        warn "Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
        ok "Installed"
    fi

    # Start Ollama if not running
    if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
        ollama serve &>/dev/null & sleep 2
    fi

    step "Pull translation model"
    echo ""
    echo -e "${B}Choose a model to download (or skip and pull later from UI):${N}"
    echo "  1) qwen3:8b   — Recommended (5GB)"
    echo "  2) qwen3:14b  — Best (8GB)"
    echo "  3) qwen3:4b   — Lightweight (3GB)"
    echo "  4) aya-expanse:8b — Multilingual specialist"
    echo "  5) Skip"
    read -p "$(echo -e ${C}'Select [1-5, default=1]: '${N})" -n 1 -r; echo ""
    case "${REPLY:-1}" in
        1) ollama pull qwen3:8b ;;
        2) ollama pull qwen3:14b ;;
        3) ollama pull qwen3:4b ;;
        4) ollama pull aya-expanse:8b ;;
        *) ok "Skipping model pull" ;;
    esac

    # Set default .env for Ollama
    cat > .env.example <<'EOF'
# Ollama Configuration
USE_LM_STUDIO=0
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:8b

# macOS compatibility
PYTORCH_ENABLE_MPS_FALLBACK=1
TORCHCODEC_USE_TORCHAUDIO=1
GOCHIDUBB_USE_WHISPERX=0
WHISPER_MODEL=base
GOCHIDUBB_WARMUP=0
EOF
fi

# ── NLTK data ──
step "NLTK data"
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)" 2>/dev/null
ok "NLTK data ready"

# ── Create directories ──
mkdir -p uploads outputs jobs_db presets/voices

# ── Copy .env.example to .env if not exists ──
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    ok ".env created from .env.example"
    echo -e "${Y}Please edit .env to configure your settings${N}"
fi

# ── Create start.sh launcher ──
step "Creating launcher"
cat > start.sh <<'EOF'
#!/usr/bin/env bash
cd "$(dirname "$0")"
source venv/bin/activate
[ -f .env ] && export $(grep -v '^#' .env | xargs)

# Check if using LM Studio
if [ "${USE_LM_STUDIO:-1}" = "1" ]; then
    echo "Using LM Studio: ${LM_STUDIO_URL:-http://localhost:1234/v1}"
    # Check if LM Studio is running
    if ! curl -s --max-time 2 ${LM_STUDIO_URL:-http://localhost:1234/v1}/models &>/dev/null; then
        echo -e "\033[1;33m[!] LM Studio not running. Start LM Studio and enable local server.\033[0m"
        echo "    Settings → Developer → Local Server → Start Server"
        echo ""
    fi
else
    # Start Ollama if needed
    if ! curl -s --max-time 2 http://localhost:11434/api/tags &>/dev/null; then
        command -v ollama &>/dev/null && { echo "Starting Ollama..."; ollama serve &>/dev/null & sleep 2; }
    fi
fi

# Set macOS environment variables
if [[ "$OSTYPE" == "darwin"* ]]; then
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    export TORCHCODEC_USE_TORCHAUDIO=1
    export GOCHIDUBB_USE_WHISPERX=0
fi

echo ""
echo "╔═════════════════════════════════════════════════════════╗"
echo "║  GoChiDUBB Studio — http://localhost:8910              ║"
echo "╠═════════════════════════════════════════════════════════╣"
if [ "${USE_LM_STUDIO:-1}" = "1" ]; then
    echo "║  Translation: LM Studio (${LM_STUDIO_URL:-http://localhost:1234/v1})  ║"
else
    echo "║  Translation: Ollama                                  ║"
fi
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "║  Platform: macOS (MPS fallback enabled)              ║"
fi
echo "╚═════════════════════════════════════════════════════════╝"
echo ""
python server.py
EOF
chmod +x start.sh

# ── Check for speaker diarization token ──
step "Speaker diarization"
echo ""
echo -e "${Y}For speaker diarization, you need a Hugging Face token:${N}"
echo "  1. Create account: https://huggingface.co/join"
echo "  2. Get token: https://huggingface.co/settings/tokens"
echo "  3. Accept terms at:"
echo "     - https://huggingface.co/pyannote/speaker-diarization-3.1"
echo "     - https://huggingface.co/pyannote/speaker-diarization-community-1"
echo "  4. Add to .env: HF_TOKEN=your_token_here"
echo ""

# ── Final message ──
step "Done!"
echo ""
echo -e "   ${G}${B}./start.sh${N}   →   http://localhost:8910"
echo ""
echo -e "${C}${B}Quick Setup:${N}"
if grep -q "USE_LM_STUDIO=1" .env 2>/dev/null; then
    echo "  1. Make sure LM Studio is running with local server enabled"
    echo "  2. Download a model in LM Studio (recommended: Qwen 2.5 14B Instruct)"
    echo "  3. Run: ./start.sh"
    echo ""
    echo -e "${Y}If LM Studio is not running, the app will use fallback translation.${N}"
else
    echo "  1. Make sure Ollama is running: ollama serve"
    echo "  2. Run: ./start.sh"
fi
echo ""