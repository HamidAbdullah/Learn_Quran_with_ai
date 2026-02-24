"""
Phase 4.1 — Streaming observability metrics.

Thread-safe counters and latency samples for /ws/recite.
Exposed via GET /metrics/streaming (JSON snapshot).
Used by websocket_server for queue depth, latency, failures, and adaptive window.
"""

import os
import threading
from collections import deque
from typing import Any, Dict, List

# ----- Shared state (module-level for singleton behavior) -----
_lock = threading.Lock()
_active_connections = 0
_latency_samples: deque = deque(maxlen=1000)  # last N total_ms values for avg/p95
_asr_failure_count = 0
_dropped_segments = 0

# Adaptive window (seconds); updated when avg_latency crosses thresholds
_adaptive_window_seconds = float(os.environ.get("WS_WINDOW_SECONDS", "2.5"))
_adaptive_window_min = float(os.environ.get("WS_WINDOW_MIN", "1.0"))
_adaptive_window_max = float(os.environ.get("WS_WINDOW_MAX", "5.0"))
_adaptive_latency_high_threshold_ms = float(os.environ.get("ADAPTIVE_LATENCY_HIGH_MS", "800"))
_adaptive_latency_low_threshold_ms = float(os.environ.get("ADAPTIVE_LATENCY_LOW_MS", "300"))


def record_connection_open() -> None:
    """Call when a WebSocket connection is accepted."""
    with _lock:
        global _active_connections
        _active_connections += 1


def record_connection_close() -> None:
    """Call when a WebSocket connection closes."""
    with _lock:
        global _active_connections
        _active_connections = max(0, _active_connections - 1)


def record_latency_ms(total_ms: float) -> None:
    """Record one end-to-end latency sample (total_ms) and update adaptive window."""
    with _lock:
        global _adaptive_window_seconds
        _latency_samples.append(total_ms)
        n = len(_latency_samples)
        if n < 5:
            return
        avg = sum(_latency_samples) / n
        if avg > _adaptive_latency_high_threshold_ms:
            _adaptive_window_seconds = min(_adaptive_window_max, _adaptive_window_seconds + 0.5)
        elif avg < _adaptive_latency_low_threshold_ms:
            _adaptive_window_seconds = max(_adaptive_window_min, _adaptive_window_seconds - 0.5)


def record_asr_failure() -> None:
    """Call when incremental ASR raises or returns error."""
    with _lock:
        global _asr_failure_count
        _asr_failure_count += 1


def record_dropped_segment() -> None:
    """Call when a chunk is skipped due to queue full (server_busy)."""
    with _lock:
        global _dropped_segments
        _dropped_segments += 1


def get_queue_depth() -> int:
    """Current number of pending ASR segments (maintained by caller; this is for consistency)."""
    return 0


def get_adaptive_window_seconds() -> float:
    """Current adaptive window length in seconds (for new connections)."""
    with _lock:
        return _adaptive_window_seconds


def get_snapshot() -> Dict[str, Any]:
    """
    Return a JSON-serializable snapshot of streaming metrics.
    Used by GET /metrics/streaming.
    """
    with _lock:
        samples = list(_latency_samples)
        adaptive_sec = _adaptive_window_seconds
    n = len(samples)
    if n == 0:
        avg_latency_ms = None
        p95_latency_ms = None
    else:
        avg_latency_ms = round(sum(samples) / n, 2)
        sorted_s = sorted(samples)
        idx = max(0, int(0.95 * n) - 1)
        p95_latency_ms = round(sorted_s[idx], 2)
    return {
        "active_connections": _active_connections,
        "avg_latency_ms": avg_latency_ms,
        "p95_latency_ms": p95_latency_ms,
        "asr_failure_count": _asr_failure_count,
        "dropped_segments": _dropped_segments,
        "latency_sample_count": n,
        "adaptive_window_seconds": round(adaptive_sec, 2),
    }
