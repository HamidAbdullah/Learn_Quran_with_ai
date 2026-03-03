"""
Phase 4 — Per-connection audio buffer for streaming ASR.

Accumulates incoming audio chunks (200–500 ms typical), maintains a sliding window
suitable for incremental Whisper inference. Designed so session state can later
move to Redis for horizontal scaling.
"""

import time
from typing import Optional, Tuple

# 16 kHz mono, 16-bit = 32000 bytes/sec
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
BYTES_PER_SECOND = SAMPLE_RATE * BYTES_PER_SAMPLE


def bytes_to_duration_ms(num_bytes: int) -> float:
    """Convert raw audio byte count to duration in milliseconds."""
    if num_bytes <= 0:
        return 0.0
    return (num_bytes / BYTES_PER_SECOND) * 1000.0


def duration_ms_to_bytes(ms: float) -> int:
    """Convert duration in ms to byte count for 16 kHz 16-bit mono."""
    return int((ms / 1000.0) * BYTES_PER_SECOND)


class AudioBuffer:
    """
    Per-user/session audio buffer for streaming.

    - Accepts chunks of any size (typical 200–500 ms).
    - Accumulates until at least `window_duration_ms` is available, then
      exposes a segment for ASR via `get_ready_segment()`.
    - Optional overlap: keep last `overlap_duration_ms` for next window (reduces
      boundary artifacts in Whisper).
    """

    def __init__(
        self,
        window_duration_ms: float = 2000.0,
        overlap_duration_ms: float = 500.0,
        min_chunk_bytes: int = 0,
    ):
        """
        Args:
            window_duration_ms: Minimum duration (ms) to consider buffer "ready" for ASR.
            overlap_duration_ms: Duration to keep at end of segment for next window (0 = no overlap).
            min_chunk_bytes: Ignore chunks smaller than this (0 = accept all).
        """
        self.window_duration_ms = max(200.0, window_duration_ms)
        self.overlap_duration_ms = max(0.0, min(overlap_duration_ms, self.window_duration_ms * 0.5))
        self.min_chunk_bytes = min_chunk_bytes
        self._buffer = bytearray()
        self._total_appended = 0
        self._created_at = time.monotonic()

    def append(self, chunk: bytes) -> None:
        """Append a raw audio chunk (16 kHz, 16-bit mono)."""
        if not chunk:
            return
        if self.min_chunk_bytes and len(chunk) < self.min_chunk_bytes:
            return
        self._buffer.extend(chunk)
        self._total_appended += len(chunk)

    def duration_ms(self) -> float:
        """Current buffered duration in milliseconds."""
        return bytes_to_duration_ms(len(self._buffer))

    def has_ready_segment(self) -> bool:
        """True if at least one window_duration_ms of audio is available."""
        return self.duration_ms() >= self.window_duration_ms

    def get_ready_segment(self) -> Optional[bytes]:
        """
        If buffer has at least window_duration_ms, return a segment of that length
        (as bytes) and optionally retain overlap at the end for next call.
        Returns None if not enough data.
        """
        if not self.has_ready_segment():
            return None
        window_bytes = duration_ms_to_bytes(self.window_duration_ms)
        overlap_bytes = duration_ms_to_bytes(self.overlap_duration_ms)
        take = min(len(self._buffer), window_bytes)
        segment = bytes(self._buffer[:take])
        # Keep overlap for next window; drop the rest
        keep = min(overlap_bytes, len(self._buffer) - take)
        self._buffer = self._buffer[take : take + keep]
        return segment

    def peek_all(self) -> bytes:
        """Return all buffered bytes without consuming (e.g. for final full inference)."""
        return bytes(self._buffer)

    def clear(self) -> None:
        """Clear the buffer (e.g. on session end)."""
        self._buffer.clear()

    def total_appended_bytes(self) -> int:
        """Total bytes ever appended (for stats)."""
        return self._total_appended

    def age_seconds(self) -> float:
        """Seconds since buffer was created (for session TTL)."""
        return time.monotonic() - self._created_at
