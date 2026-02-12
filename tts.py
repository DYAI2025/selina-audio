"""F5-TTS wrapper for text-to-speech with voice cloning."""

import io
import tempfile
from pathlib import Path

import soundfile as sf

_model = None
_model_de = None

DEFAULT_REF_DIR = Path(__file__).parent / "voices"


def load_model(lang: str = "en"):
    """Load F5-TTS model. Uses German fine-tune for 'de', base model otherwise."""
    global _model, _model_de
    from f5_tts.api import F5TTS

    if lang == "de":
        if _model_de is not None:
            return _model_de
        print("[TTS] Loading F5-TTS German model...")
        try:
            from cached_path import cached_path
            _model_de = F5TTS(
                model="F5TTS_v1_Base",
                ckpt_file=str(cached_path("hf://hvoss-techfak/F5-TTS-German/model_f5tts_german.pt")),
                vocab_file=str(cached_path("hf://hvoss-techfak/F5-TTS-German/vocab.txt")),
                device="cuda",
            )
        except Exception as e:
            print(f"[TTS] German model failed ({e}), falling back to base model")
            if _model is None:
                _model = F5TTS(model="F5TTS_v1_Base", device="cuda")
            _model_de = _model
        print("[TTS] German model loaded.")
        return _model_de
    else:
        if _model is not None:
            return _model
        print("[TTS] Loading F5-TTS base model...")
        _model = F5TTS(model="F5TTS_v1_Base", device="cuda")
        print("[TTS] Base model loaded.")
        return _model


def get_default_ref_audio(lang: str = "en") -> tuple[str, str] | None:
    """Get default reference audio for a language, if available."""
    ref_dir = DEFAULT_REF_DIR
    for suffix in (".wav", ".mp3", ".flac"):
        ref_path = ref_dir / f"selina_{lang}{suffix}"
        if ref_path.exists():
            txt_path = ref_path.with_suffix(".txt")
            ref_text = txt_path.read_text().strip() if txt_path.exists() else ""
            return str(ref_path), ref_text
    return None


def synthesize(
    text: str,
    ref_audio_bytes: bytes | None = None,
    ref_text: str = "",
    language: str = "en",
    speed: float = 1.0,
    nfe_step: int = 32,
) -> bytes:
    """Synthesize text to WAV audio bytes.

    Args:
        text: Text to speak.
        ref_audio_bytes: Optional reference audio for voice cloning.
        ref_text: Transcript of the reference audio.
        language: "en" or "de" to select model.
        speed: Speech speed multiplier.
        nfe_step: ODE solver steps (16=fast, 32=default, 64=high quality).

    Returns:
        WAV audio bytes.
    """
    model = load_model(language)

    # Handle reference audio
    ref_audio_path = None

    if ref_audio_bytes is not None:
        # Save uploaded reference to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(ref_audio_bytes)
            ref_audio_path = tmp.name
    else:
        # Try default reference voice
        default_ref = get_default_ref_audio(language)
        if default_ref:
            ref_audio_path, ref_text = default_ref

    if ref_audio_path is None:
        raise ValueError(
            "No reference audio available. Upload ref_audio or place a "
            "default voice file in voices/selina_en.wav"
        )

    # Auto-transcribe if ref_text not provided
    if not ref_text and ref_audio_path:
        ref_text = model.transcribe(ref_audio_path, language=language)

    # Run inference
    wav, sr, _spec = model.infer(
        ref_file=ref_audio_path,
        ref_text=ref_text,
        gen_text=text,
        speed=speed,
        nfe_step=nfe_step,
        file_wave=None,
        file_spec=None,
    )

    # Convert numpy array to WAV bytes
    buffer = io.BytesIO()
    sf.write(buffer, wav, sr, format="WAV")
    buffer.seek(0)
    return buffer.read()
