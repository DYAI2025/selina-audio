"""faster-whisper ASR wrapper for speech-to-text."""

import io
from faster_whisper import WhisperModel

_model: WhisperModel | None = None

MODEL_SIZE = "cstr/whisper-large-v3-turbo-german-int8_float32"
COMPUTE_TYPE = "int8"


def load_model():
    global _model
    if _model is not None:
        return
    print(f"[ASR] Loading faster-whisper {MODEL_SIZE} ({COMPUTE_TYPE})...")
    _model = WhisperModel(
        MODEL_SIZE,
        device="cuda",
        compute_type=COMPUTE_TYPE,
        device_index=0,
    )
    print("[ASR] Model loaded.")


def transcribe(audio_bytes: bytes, language: str | None = None) -> dict:
    """Transcribe audio bytes to text.

    Args:
        audio_bytes: Raw audio file bytes (MP3, WAV, OGG, etc.)
        language: Optional language code (e.g. "en", "de"). None = auto-detect.

    Returns:
        dict with text, language, language_probability
    """
    load_model()
    assert _model is not None

    audio_stream = io.BytesIO(audio_bytes)

    segments, info = _model.transcribe(
        audio_stream,
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    # segments is a generator - must iterate to trigger transcription
    full_text = " ".join(segment.text.strip() for segment in segments)

    return {
        "text": full_text,
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
    }
