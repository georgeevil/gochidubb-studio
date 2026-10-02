#!/bin/bash
# macOS setup script for GoChiDUBB

echo "Setting up GoChiDUBB for macOS..."

# 1. Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
if [ "$PYTHON_VERSION" != "3.10" ] && [ "$PYTHON_VERSION" != "3.11" ] && [ "$PYTHON_VERSION" != "3.12" ]; then
    echo "⚠️  Python version $PYTHON_VERSION detected. Recommend Python 3.12."
    echo "Install with: brew install python@3.12"
fi

# 2. Install ffmpeg
echo "Installing ffmpeg..."
brew install ffmpeg

# 3. Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# 4. Activate environment
source venv/bin/activate

# 5. Install PyTorch (CPU version for stability)
echo "Installing PyTorch..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 6. Install core dependencies
echo "Installing dependencies..."
pip install -r requirements.txt 2>/dev/null || echo "No requirements.txt found"

# 7. Install whisperx without forcing GPU
pip install whisperx --no-deps
pip install faster-whisper

# 7b. Apple Silicon: GPU (Metal) Whisper via mlx-whisper. ASR_BACKEND=auto
# (the default) picks it up automatically; it is skipped on Intel Macs.
if [ "$(uname -m)" = "arm64" ]; then
    echo "Installing mlx-whisper (Apple Silicon GPU transcription)..."
    pip install -r requirements-mac.txt
fi

# 8. Set environment variables
echo "export PYTORCH_ENABLE_MPS_FALLBACK=1" >> .env
echo "export TORCHCODEC_USE_TORCHAUDIO=1" >> .env
echo "export GOCHIDUBB_USE_WHISPERX=0" >> .env  # Use faster-whisper for stability

echo "✅ Setup complete!"
echo "Run: ./start.sh"
