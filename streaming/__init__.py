"""
Phase 4 — Real-time streaming layer.

- audio_buffer: Per-connection buffer for 200–500 ms chunks.
- streaming_asr: Incremental Whisper on buffer segments.
- websocket_server: WebSocket handler for /ws/recite (import separately to avoid pulling FastAPI).
"""

from streaming.audio_buffer import AudioBuffer, bytes_to_duration_ms, duration_ms_to_bytes
from streaming.streaming_asr import run_streaming_asr

__all__ = [
    "AudioBuffer",
    "bytes_to_duration_ms",
    "duration_ms_to_bytes",
    "run_streaming_asr",
]
