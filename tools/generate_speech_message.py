"""Generate the fixed guard-mode warning WAV once."""

from __future__ import annotations

import os
from pathlib import Path

import soundfile as sf
from dotenv import load_dotenv
from kokoro_onnx import Kokoro


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


model_path = project_path(
    os.getenv(
        "KOKORO_MODEL_PATH",
        "models/kokoro-v1.0.onnx",
    )
)

voices_path = project_path(
    os.getenv(
        "KOKORO_VOICES_PATH",
        "models/voices-v1.0.bin",
    )
)

output_path = (
    ROOT
    / "assets"
    / "audio"
    / "guard_warning.wav"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

kokoro = Kokoro(
    str(model_path),
    str(voices_path),
)

samples, sample_rate = kokoro.create(
    (
        "Warning. You are not the authorised user. "
        "Please step back from the desk."
    ),
    voice=os.getenv(
        "KOKORO_VOICE",
        "af_heart",
    ),
    speed=0.95,
    lang=os.getenv(
        "KOKORO_LANGUAGE",
        "en-us",
    ),
)

sf.write(
    output_path,
    samples,
    sample_rate,
    subtype="PCM_16",
)

print(
    f"Guard warning created: {output_path}"
)