"""
Phase 4 — WebSocket server for /ws/recite.
Phase 4.1 — Queue depth (server_busy), feedback smoothing, metrics, adaptive window.

- Accepts audio chunks (200–500 ms), per-connection buffer.
- Incremental ASR → partial transcript + lightweight word feedback + latency_ms.
- If queue_depth > MAX_QUEUE: send server_busy, skip ASR, record dropped_segments.
- Word feedback smoothed: wrong only after 2 consecutive windows.
- Records active_connections, latency, asr_failure, dropped_segments.
- Full scoring only when user stops (or on explicit end message).
"""

import asyncio
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from fastapi import WebSocket, WebSocketDisconnect

from streaming.audio_buffer import AudioBuffer, bytes_to_duration_ms
from streaming.streaming_asr import run_streaming_asr


def build_ws_recite_handler(
    get_whisper_model: Callable[[], Any],
    get_reference_text: Callable[[str], str],
    get_inference_lock: Callable[[], Any],
    run_batch_verification: Optional[Callable[[str, str, bytes], Dict[str, Any]]] = None,
    window_duration_ms: float = 2000.0,
    overlap_duration_ms: float = 500.0,
    max_queue_depth: int = 3,
    get_adaptive_window_ms: Optional[Callable[[], float]] = None,
    get_metrics: Optional[Callable[[], Any]] = None,
) -> Callable:
    """
    Build the async WebSocket handler for /ws/recite.

    Args:
        get_whisper_model: Callable that returns the Whisper model.
        get_reference_text: Callable(verse_key) -> reference text (Uthmani).
        get_inference_lock: Callable that returns the threading lock for GPU/inference.
        run_batch_verification: Optional (verse_key, reference_text, audio_bytes) -> full result.
        window_duration_ms: Buffer window before running incremental ASR (used if get_adaptive_window_ms not set).
        overlap_duration_ms: Overlap between consecutive windows.
        max_queue_depth: Max pending ASR segments; if exceeded, send server_busy and skip chunk (Phase 4.1).
        get_adaptive_window_ms: Optional callable returning current window in ms for new connections (Phase 4.1).
        get_metrics: Optional module with record_connection_open/close, record_latency_ms, record_asr_failure, record_dropped_segment.

    Returns:
        Async function (websocket: WebSocket) -> None for use with FastAPI.
    """
    metrics = get_metrics

    async def handle_ws_recite(websocket: WebSocket) -> None:
        params = websocket.query_params
        surah, ayah = params.get("surah"), params.get("ayah")
        if not surah or not ayah:
            await websocket.close(code=1008, reason="Missing surah or ayah")
            return

        verse_key = f"{surah}:{ayah}"
        reference_text = get_reference_text(verse_key)
        if not reference_text:
            await websocket.close(code=1008, reason=f"Verse {verse_key} not found")
            return

        await websocket.accept()
        if metrics and hasattr(metrics, "record_connection_open"):
            metrics.record_connection_open()

        # Adaptive window: use get_adaptive_window_ms() for this connection if provided
        window_ms = window_duration_ms
        if get_adaptive_window_ms:
            try:
                window_ms = get_adaptive_window_ms()
            except Exception:
                pass
        buffer = AudioBuffer(
            window_duration_ms=window_ms,
            overlap_duration_ms=overlap_duration_ms,
        )
        last_partial_transcript = ""
        inference_lock = get_inference_lock() if get_inference_lock else None
        whisper_model = get_whisper_model()
        pending_tasks: List[asyncio.Task] = []

        try:
            from core.scoring import get_lightweight_word_feedback
        except ImportError:
            get_lightweight_word_feedback = None

        try:
            from streaming.feedback_smoothing import WordFeedbackSmoother
            smoother = WordFeedbackSmoother(history_size=2)
        except ImportError:
            smoother = None

        async def send_partial(partial: str, word_feedback: list, latency_ms: dict, server_busy: bool = False) -> None:
            payload = {
                "type": "partial_result",
                "partial_transcript": partial,
                "word_feedback": word_feedback,
                "latency_ms": latency_ms,
            }
            if server_busy:
                payload["server_busy"] = True
            try:
                await websocket.send_json(payload)
            except (WebSocketDisconnect, RuntimeError):
                pass

        async def run_incremental(segment: bytes) -> None:
            t0 = time.perf_counter()
            latency_ms = {"buffer_ms": round(bytes_to_duration_ms(len(segment)))}
            partial = ""
            word_feedback: List[Dict[str, Any]] = []
            try:
                loop = asyncio.get_event_loop()
                if inference_lock:

                    def do_asr():
                        with inference_lock:
                            return run_streaming_asr(whisper_model, segment)

                    partial, meta = await loop.run_in_executor(None, do_asr)
                else:
                    partial, meta = await loop.run_in_executor(None, lambda: run_streaming_asr(whisper_model, segment))
                if meta.get("inference_ms") is not None:
                    latency_ms["inference_ms"] = meta["inference_ms"]
                if meta.get("error"):
                    latency_ms["error"] = meta["error"]
                    if metrics and hasattr(metrics, "record_asr_failure"):
                        metrics.record_asr_failure()
                nonlocal last_partial_transcript
                if partial:
                    last_partial_transcript = (last_partial_transcript + " " + partial).strip()
                if get_lightweight_word_feedback and reference_text:
                    word_feedback = get_lightweight_word_feedback(reference_text, last_partial_transcript)
                    if smoother:
                        word_feedback = smoother.update(word_feedback)
                latency_ms["total_ms"] = round((time.perf_counter() - t0) * 1000)
                if metrics and hasattr(metrics, "record_latency_ms"):
                    metrics.record_latency_ms(latency_ms["total_ms"])
                await send_partial(last_partial_transcript, word_feedback, latency_ms)
            except Exception as e:
                latency_ms["total_ms"] = round((time.perf_counter() - t0) * 1000)
                latency_ms["error"] = str(e)
                if metrics and hasattr(metrics, "record_asr_failure"):
                    metrics.record_asr_failure()
                await send_partial(last_partial_transcript, word_feedback, latency_ms)

        try:
            while True:
                try:
                    data = await asyncio.wait_for(websocket.receive(), timeout=300.0)
                except asyncio.TimeoutError:
                    break
                if data.get("type") == "websocket.disconnect":
                    break
                if data.get("type") == "websocket.receive":
                    msg = data.get("text") or data.get("bytes")
                    if msg is None:
                        continue
                    if isinstance(msg, str):
                        if msg.strip().lower() in ("end", "stop", "final"):
                            break
                        continue
                    buffer.append(msg)
                    # Prune done tasks to get current queue depth
                    pending_tasks[:] = [t for t in pending_tasks if not t.done()]
                    queue_depth = len(pending_tasks)
                    if buffer.has_ready_segment():
                        if queue_depth >= max_queue_depth:
                            if metrics and hasattr(metrics, "record_dropped_segment"):
                                metrics.record_dropped_segment()
                            await send_partial(
                                last_partial_transcript,
                                [],
                                {"buffer_ms": 0, "total_ms": 0, "skipped": "queue_full"},
                                server_busy=True,
                            )
                        else:
                            segment = buffer.get_ready_segment()
                            if segment:
                                task = asyncio.create_task(run_incremental(segment))
                                pending_tasks.append(task)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                    "fallback": "Use POST /verify with full audio for complete scoring.",
                })
            except Exception:
                pass
        finally:
            if metrics and hasattr(metrics, "record_connection_close"):
                metrics.record_connection_close()

        # Wait for any in-flight tasks before final result (optional)
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)

        if run_batch_verification and buffer.peek_all():
            try:
                loop = asyncio.get_event_loop()
                full_result = await loop.run_in_executor(
                    None,
                    lambda: run_batch_verification(verse_key, reference_text, bytes(buffer.peek_all())),
                )
                await websocket.send_json({"type": "final_result", "result": full_result})
            except Exception as e:
                try:
                    await websocket.send_json({
                        "type": "final_result_error",
                        "message": str(e),
                        "fallback": "Use POST /verify with full audio.",
                    })
                except Exception:
                    pass

        buffer.clear()

    return handle_ws_recite
