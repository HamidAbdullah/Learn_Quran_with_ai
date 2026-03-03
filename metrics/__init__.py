"""
Observability and streaming metrics (Phase 4.1).
"""

from metrics.streaming_metrics import (
    get_snapshot,
    get_adaptive_window_seconds,
    record_connection_open,
    record_connection_close,
    record_latency_ms,
    record_asr_failure,
    record_dropped_segment,
)

__all__ = [
    "get_snapshot",
    "get_adaptive_window_seconds",
    "record_connection_open",
    "record_connection_close",
    "record_latency_ms",
    "record_asr_failure",
    "record_dropped_segment",
]
