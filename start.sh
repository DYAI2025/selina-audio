#!/bin/bash
# Start Selina Audio Service (faster-whisper ASR + F5-TTS on GPU)
cd "$(dirname "$0")"
echo "[Selina-Audio] Starting on port 8100..."
exec uv run python server.py
