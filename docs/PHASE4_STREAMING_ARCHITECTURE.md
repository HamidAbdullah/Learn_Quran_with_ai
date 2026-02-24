# Phase 4 — Real-Time Streaming Architecture

## Overview

Phase 4 adds a **new streaming layer** alongside the existing batch APIs. It does not replace `/verify` or `/ws/verify`.

- **New endpoint:** `GET /ws/recite?surah=1&ayah=1` (WebSocket).
- **Input:** Binary audio chunks (200–500 ms, 16 kHz 16-bit mono).
- **Output per chunk:** `{ partial_transcript, word_feedback, latency_ms }`.
- **On session end:** Optional full batch verification (same pipeline as `/verify`) for final score; if streaming fails, client falls back to `POST /verify`.

---

## Components

| Component | Role |
|-----------|------|
| **streaming/audio_buffer.py** | Per-connection buffer; accumulates chunks until a window (e.g. 2 s) is ready for ASR. |
| **streaming/streaming_asr.py** | Runs Whisper on a segment (raw → WAV → transcribe); returns partial text + timing. |
| **streaming/websocket_server.py** | WebSocket handler: receive chunks → buffer → incremental ASR → lightweight word alignment → send `partial_result` with `latency_ms`. On disconnect/end, optionally run full batch. |
| **core/scoring.get_lightweight_word_feedback** | Text-only word-level status (correct/wrong/minor_mistake/missing) for streaming; no CTC/tajweed. |

---

## Concurrency & Scaling Design

### Current (single-node)

- **Async FastAPI:** WebSocket connections are async; I/O (receive/send) does not block.
- **Background ASR:** Each “ready” segment is run via `run_in_executor` with a **single global `inference_lock`**, so only one Whisper forward runs at a time. This avoids GPU OOM and keeps latency predictable.
- **Session state:** In-memory per connection (`AudioBuffer` + `last_partial_transcript`). No Redis yet.

### Scaling strategy

| Scale | Approach | Notes |
|-------|----------|------|
| **~1K concurrent users** | Single app instance, keep `inference_lock`, tune `window_duration_ms` and overlap. | One GPU or CPU; queueing is implicit (lock). Add horizontal replication behind a load balancer for more connections; each instance still one inference at a time. |
| **~10K concurrent users** | **Redis session storage:** Store `AudioBuffer` state (and optional `last_partial_transcript`) keyed by `connection_id` so connections can be handed off or resumed. **GPU inference queue:** Replace single lock with a **bounded queue** and a small pool of worker processes/threads that pull segments and run Whisper; multiple workers allow concurrent inference up to GPU memory. | Sticky sessions or Redis-backed state; multiple workers per node or dedicated inference nodes. |
| **~100K concurrent users** | **Dedicated ASR tier:** WebSocket servers (stateless or with Redis session store) only do buffering and send segments to a **message queue** (e.g. Redis Streams, RabbitMQ, Kafka). **Worker pool** (multiple nodes) consumes from the queue, runs Whisper, and pushes results to Redis or back to the client via a separate channel (e.g. WebSocket reply channel keyed by `connection_id`). **Auto-scaling** workers based on queue depth. | Separation of “connection handling” and “inference”; session state in Redis; GPU nodes scale independently. |

### Redis session storage (design)

- **Key:** `streaming:session:{connection_id}`.
- **Value:** Serialized buffer (e.g. base64 bytes) + `last_partial_transcript` + `verse_key` + `created_at`.
- **TTL:** e.g. 10 minutes; extend on each chunk.
- **Use:** On reconnect or for serverless/edge, restore buffer and continue; or for “final” step, read full buffer and run batch verification.

### GPU inference queue (design)

- **Producer:** WebSocket handler, when `buffer.get_ready_segment()` returns, push `{ connection_id, segment, verse_key }` to a queue (e.g. Redis list or Stream).
- **Consumers:** N worker processes (or threads) each holding a Whisper model (or sharing one if thread-safe). Each worker blocks on queue pop, runs `run_streaming_asr`, then pushes result to a “results” structure keyed by `connection_id` (or sends via WebSocket if the connection is still on the same node).
- **Back-pressure:** Bounded queue size; if full, drop segment or return “overloaded” to client and suggest batch `/verify`.

---

## Fail-Safe

- **During streaming:** If incremental ASR or alignment throws, the handler catches, sends `partial_result` with `latency_ms.error` (or an `error` message) and suggests: “Use POST /verify with full audio for complete scoring.”
- **On session end:** If the optional full batch run fails, send `final_result_error` with the same fallback message.
- **Client:** Can always record full audio and call `POST /verify` for full scoring (Phase 1–3 unchanged).

---

## Latency

Each `partial_result` includes:

- **latency_ms.buffer_ms:** Duration of the segment (audio length).
- **latency_ms.inference_ms:** Whisper forward time.
- **latency_ms.total_ms:** End-to-end for this chunk (receive → ASR → send).

Per-stage latency is tracked so you can tune window size and overlap and monitor bottlenecks.

---

## Configuration (env)

- `WS_WINDOW_SECONDS` — buffer window before running ASR (default 2.5 s).
- `WS_OVERLAP_SECONDS` — overlap between consecutive windows (default 0.5 s).

Existing `WS_BUFFER_THRESHOLD`, `WS_PARTIAL_RESULT_INTERVAL` apply to `/ws/verify`; Phase 4 `/ws/recite` uses the new window/overlap above.

---

## Phase 4.1 — Product Hardening

### Streaming stability (queue depth)

- **queue_depth:** Number of pending ASR segments per connection (tasks not yet done).
- **MAX_QUEUE** (`WS_MAX_QUEUE`, default 3): If `queue_depth >= MAX_QUEUE`, the server sends `{ server_busy: true }` in the next `partial_result` and **skips** running incremental ASR for that chunk (segment is not consumed; client can retry or slow down).
- **Dropped segments** are counted in metrics (`dropped_segments`).

**Design:** Prevents one fast client from flooding the inference lock; gives back-pressure without closing the connection.

### Feedback smoothing

- **streaming/feedback_smoothing.py:** `WordFeedbackSmoother` keeps the last 2 `word_feedback` states.
- A word is only marked **"wrong"** (or **"minor_mistake"**) if it was already wrong/minor in the **previous** window. Otherwise the previous status (or "correct") is shown.
- **Design:** Reduces flicker when the partial transcript briefly misaligns (e.g. one window says wrong, next says correct); only show wrong after 2 consecutive windows agree.

### Observability metrics

- **metrics/streaming_metrics.py:** Thread-safe counters and latency samples.
  - `active_connections` — incremented on accept, decremented on close.
  - `avg_latency_ms` / `p95_latency_ms` — from last N `total_ms` samples (e.g. 1000).
  - `asr_failure_count` — when ASR raises or returns error.
  - `dropped_segments` — when a chunk is skipped due to queue full.
- **GET /metrics/streaming:** Returns a JSON snapshot of the above plus `latency_sample_count` and `adaptive_window_seconds`. No auth; use for internal dashboards or behind a proxy.

### Adaptive window

- When `record_latency_ms(total_ms)` is called, metrics update a rolling average. If **avg_latency_ms > ADAPTIVE_LATENCY_HIGH_MS** (default 800): increase `adaptive_window_seconds` by +0.5 s (cap `WS_WINDOW_MAX`, default 5). If **avg_latency_ms < ADAPTIVE_LATENCY_LOW_MS** (default 300): decrease by 0.5 s (floor `WS_WINDOW_MIN`, default 1).
- **New connections** use `get_adaptive_window_seconds()` (in seconds, converted to ms for the buffer) so each new WebSocket gets the current adaptive window. Existing connections keep their initial window.
- **Design:** Under load, longer windows reduce the number of ASR runs and ease latency; when load is low, shorter windows give faster feedback.

### Phase 4.1 configuration (env)

- `WS_MAX_QUEUE` — max pending ASR segments per connection (default 3).
- `WS_WINDOW_MIN` / `WS_WINDOW_MAX` — bounds for adaptive window in seconds (default 1, 5).
- `ADAPTIVE_LATENCY_HIGH_MS` / `ADAPTIVE_LATENCY_LOW_MS` — thresholds for increasing/decreasing window (default 800, 300).
