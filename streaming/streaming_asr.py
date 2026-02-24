"""
Phase 4 — Incremental streaming ASR.

Runs Whisper on buffer segments (or full buffer) for partial transcript updates.
Does NOT run dual ASR or full CTC/tajweed; that remains in the batch pipeline.
"""

import io
import tempfile
import time
from typing import Any, Dict, Optional, Tuple

from streaming.audio_buffer import bytes_to_duration_ms


def run_streaming_asr(
    whisper_model: Any,
    audio_bytes: bytes,
    language: str = "ar",
) -> Tuple[str, Dict[str, Any]]:
    """
    Run Whisper on a segment of audio (incremental / partial result).

    Args:
        whisper_model: Loaded OpenAI Whisper model (transcribe(path, language=...)).
        audio_bytes: Raw 16 kHz 16-bit mono audio bytes.
        language: Language code for Whisper (default ar).

    Returns:
        (partial_transcript: str, meta: dict with keys duration_ms, inference_ms, etc.)
    """
    meta: Dict[str, Any] = {
        "duration_ms": bytes_to_duration_ms(len(audio_bytes)),
        "inference_ms": None,
        "bytes": len(audio_bytes),
    }
    if not audio_bytes or len(audio_bytes) < 1600:  # < 50 ms
        return "", meta

    start = time.perf_counter()
    try:
        with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
            f.write(audio_bytes)
            path = f.name
        try:
            # Whisper expects file path; raw PCM may need to be wrapped or use a format it accepts.
            # Many setups use WAV; we use a small WAV header so Whisper can read it.
            wav_bytes = _raw_to_wav(audio_bytes)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(wav_bytes)
                wav_path = f.name
            try:
                result = whisper_model.transcribe(wav_path, language=language)
                text = (result.get("text") or "").strip()
            finally:
                import os
                if os.path.exists(wav_path):
                    os.unlink(wav_path)
        finally:
            import os
            if os.path.exists(path):
                os.unlink(path)
        meta["inference_ms"] = round((time.perf_counter() - start) * 1000)
        return text, meta
    except Exception as e:
        meta["inference_ms"] = round((time.perf_counter() - start) * 1000)
        meta["error"] = str(e)
        return "", meta


def _raw_to_wav(raw: bytes, sample_rate: int = 16000, sample_width: int = 2) -> bytes:
    """Wrap raw PCM (16 kHz, 16-bit mono) in a minimal WAV header."""
    import struct
    n = len(raw)
    # WAV header: 44 bytes
    header = bytearray(44)
    header[0:4] = b"RIFF"
    struct.pack_into("<I", header, 4, 36 + n)
    header[8:12] = b"WAVE"
    header[12:16] = b"fmt "
    struct.pack_into("<I", header, 16, 16)  # fmt chunk size
    struct.pack_into("<H", header, 20, 1)   # PCM
    struct.pack_into("<H", header, 22, 1)   # mono
    struct.pack_into("<I", header, 24, sample_rate)
    struct.pack_into("<I", header, 28, sample_rate * sample_width)
    struct.pack_into("<H", header, 32, sample_width * 8 // 8)
    struct.pack_into("<H", header, 34, sample_width * 8)
    header[36:40] = b"data"
    struct.pack_into("<I", header, 40, n)
    return bytes(header) + raw
