"""Qwen3-TTS wrapper for text-to-speech."""

import io

import soundfile as sf
import torch

_model = None

DEFAULT_SPEAKER = "Serena"
VALID_SPEAKERS = {
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
    "Ryan", "Aiden", "Ono_Anna", "Sohee",
}
LANG_MAP = {
    "de": "German", "en": "English", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "fr": "French",
    "ru": "Russian", "pt": "Portuguese", "es": "Spanish",
    "it": "Italian",
}


def load_model():
    global _model
    if _model is not None:
        return
    from qwen_tts import Qwen3TTSModel

    print("[TTS] Loading Qwen3-TTS-12Hz-0.6B-CustomVoice (bf16)...")
    _model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    print("[TTS] Model loaded.")


def synthesize(
    text: str,
    speaker: str = DEFAULT_SPEAKER,
    language: str = "de",
    instruct: str = "",
) -> bytes:
    """Synthesize text to WAV audio bytes.

    Args:
        text: Text to speak.
        speaker: Preset voice name (e.g. Serena, Vivian, Ryan).
        language: Language code (e.g. "de", "en", "zh").
        instruct: Optional style/emotion instruction.

    Returns:
        WAV audio bytes.
    """
    load_model()

    if speaker not in VALID_SPEAKERS:
        speaker = DEFAULT_SPEAKER

    lang_name = LANG_MAP.get(language, "German")

    kwargs = dict(text=text, language=lang_name, speaker=speaker)
    if instruct:
        kwargs["instruct"] = instruct

    wavs, sr = _model.generate_custom_voice(**kwargs)

    buffer = io.BytesIO()
    sf.write(buffer, wavs[0], sr, format="WAV")
    buffer.seek(0)
    return buffer.read()
