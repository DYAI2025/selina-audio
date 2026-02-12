"""Selina Audio Service - ASR + TTS on local GPU.

ASR: faster-whisper large-v3-turbo (int8)
TTS: F5-TTS with voice cloning
"""

import io
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load ASR at startup (fast, ~1.5GB VRAM)
    from asr import load_model as load_asr
    load_asr()
    print("[Server] ASR model ready.")

    # TTS is lazy-loaded on first request (saves VRAM if unused)
    yield


app = FastAPI(title="Selina Audio Service", lifespan=lifespan)


@app.get("/health")
def health():
    from asr import _model as asr_model
    from tts import _model as tts_model, _model_de as tts_model_de
    return {
        "status": "healthy",
        "models": {
            "asr": asr_model is not None,
            "tts_en": tts_model is not None,
            "tts_de": tts_model_de is not None,
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
        "provider": "faster-whisper-large-v3-turbo",
    })


@app.post("/synthesize")
async def synthesize_endpoint(
    text: str = Form(...),
    ref_text: str = Form(""),
    ref_audio: UploadFile | None = File(None),
    language: str = Form("de"),
    speed: float = Form(1.0),
    nfe_step: int = Form(32),
):
    """Synthesize speech from text. Optionally provide reference audio for voice cloning."""
    from tts import synthesize

    if not text.strip():
        raise HTTPException(400, "Empty text")

    ref_audio_bytes = None
    if ref_audio is not None:
        ref_audio_bytes = await ref_audio.read()
        if len(ref_audio_bytes) == 0:
            ref_audio_bytes = None

    start = time.time()
    try:
        audio_bytes = synthesize(
            text=text,
            ref_audio_bytes=ref_audio_bytes,
            ref_text=ref_text,
            language=language,
            speed=speed,
            nfe_step=nfe_step,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    elapsed_ms = int((time.time() - start) * 1000)

    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/wav",
        headers={
            "X-Duration-Ms": str(elapsed_ms),
            "X-Provider": "f5-tts",
        },
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8100)
