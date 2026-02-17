"""Selina Audio Service - ASR + TTS on local GPU.

ASR: faster-whisper German large-v3-turbo (int8)
TTS: Qwen3-TTS-0.6B with preset voices
"""

import io
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load both models at startup (together ~3.9GB, fits in 8GB VRAM)
    from asr import load_model as load_asr
    from tts import load_model as load_tts

    load_asr()
    print("[Server] ASR model ready.")

    load_tts()
    print("[Server] TTS model ready.")

    yield


app = FastAPI(title="Selina Audio Service", lifespan=lifespan)


@app.get("/health")
def health():
    from asr import _model as asr_model
    from tts import _model as tts_model
    return {
        "status": "healthy",
        "models": {
            "asr": asr_model is not None,
            "tts": tts_model is not None,
        },
    }


@app.post("/transcribe")
async def transcribe_endpoint(
    file: UploadFile = File(...),
    language: str | None = Form(None),
):
    """Transcribe audio to text."""
    from asr import transcribe

    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(400, "Empty audio file")

    start = time.time()
    result = transcribe(audio_bytes, language)
    elapsed_ms = int((time.time() - start) * 1000)

    return JSONResponse({
        "text": result["text"],
        "language": result["language"],
        "language_probability": result["language_probability"],
        "duration_ms": elapsed_ms,
        "provider": "faster-whisper-german-v3-turbo",
    })


@app.post("/synthesize")
async def synthesize_endpoint(
    text: str = Form(...),
    speaker: str = Form("Serena"),
    language: str = Form("de"),
    instruct: str = Form(""),
):
    """Synthesize speech from text using Qwen3-TTS preset voices."""
    from tts import synthesize

    if not text.strip():
        raise HTTPException(400, "Empty text")

    start = time.time()
    try:
        audio_bytes = synthesize(
            text=text,
            speaker=speaker,
            language=language,
            instruct=instruct,
        )
    except Exception as e:
        raise HTTPException(500, f"TTS synthesis failed: {e}")

    elapsed_ms = int((time.time() - start) * 1000)

    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/wav",
        headers={
            "X-Duration-Ms": str(elapsed_ms),
            "X-Provider": "qwen3-tts",
        },
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8100)
