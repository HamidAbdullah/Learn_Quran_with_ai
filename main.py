"""
Quran recitation verification API — 3-layer pipeline.
Layer 1: Dual ASR (Whisper + Wav2Vec2), best transcript selection.
Layer 2: CTC forced alignment for word-level timing and confidence.
Layer 3: Tajweed verification (acoustic + phonetic).
"""
import os
import warnings

warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*", category=UserWarning)
warnings.filterwarnings("ignore", module="urllib3")
warnings.filterwarnings("ignore", message=".*PySoundFile failed.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*audioread.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*__audioread_load.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*FP16 is not supported on CPU.*", category=UserWarning)

import json
import whisper
import uvicorn
import shutil
import io
import asyncio
import traceback
import uuid
import threading
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# Config (env + dotenv)
try:
    import config as _config
    get_quran_path = _config.get_quran_path
    get_cors_origins = _config.get_cors_origins
    resolve_device = _config.resolve_device
    PORT = _config.PORT
    HOST = _config.HOST
    WHISPER_MODEL = _config.WHISPER_MODEL
    WAV2VEC2_MODEL = _config.WAV2VEC2_MODEL
    ASR_USE_WHISPER = getattr(_config, "ASR_USE_WHISPER", True)
    WAV2VEC2_QUANTIZE_8BIT = _config.WAV2VEC2_QUANTIZE_8BIT
    TORCH_COMPILE = _config.TORCH_COMPILE
    BEAM_WIDTH = getattr(_config, "BEAM_WIDTH", 1)
    WHISPER_BEAM_SIZE = getattr(_config, "WHISPER_BEAM_SIZE", 1)
    WS_CHUNK_THRESHOLD_BYTES = _config.WS_CHUNK_THRESHOLD_BYTES
    WS_BUFFER_THRESHOLD = _config.WS_BUFFER_THRESHOLD
    WS_PARTIAL_RESULT_INTERVAL = _config.WS_PARTIAL_RESULT_INTERVAL
    VERIFY_TIMEOUT_SEC = getattr(_config, "VERIFY_TIMEOUT_SEC", 10)
    VERIFY_MAX_AUDIO_DURATION_SEC = getattr(_config, "VERIFY_MAX_AUDIO_DURATION_SEC", 30)
except ImportError:
    get_quran_path = lambda: ""
    get_cors_origins = lambda: ["*"]
    resolve_device = lambda: ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    PORT, HOST = 8001, "0.0.0.0"
    WHISPER_MODEL = "base"
    WAV2VEC2_MODEL = "rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v2"
    ASR_USE_WHISPER = False
    WAV2VEC2_QUANTIZE_8BIT = False
    TORCH_COMPILE = False
    BEAM_WIDTH = 1
    WHISPER_BEAM_SIZE = 1
    WS_CHUNK_THRESHOLD_BYTES = 150000
    WS_BUFFER_THRESHOLD = 150000
    WS_PARTIAL_RESULT_INTERVAL = 2.5
    VERIFY_TIMEOUT_SEC = 10.0
    VERIFY_MAX_AUDIO_DURATION_SEC = 30.0

from core.normalization import normalize_arabic
from core.scoring import score_recitation, segment_transcript_by_reference, count_matching_words
from core.phonetics import PhoneticAnalyzer, load_audio_for_alignment
from core.tajweed import TajweedRulesEngine
from core.asr import run_dual_asr
from core.metrics import wer, cer
from alignment.ctc_alignment import align_reference_to_audio

app = FastAPI(title="Quran AI Production API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_quran():
    """Load Quran: config path or quran_with_audio.json / quran.json in cwd."""
    path = get_quran_path() if callable(get_quran_path) else ""
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            quran_map = {item["verse_key"]: item for item in data}
            return list(quran_map.values()), quran_map
    for name in ("quran_with_audio.json", "quran.json"):
        p = os.path.join(os.getcwd(), name)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                quran_map = {item["verse_key"]: item for item in data}
                return list(quran_map.values()), quran_map
            if isinstance(data, dict) and not data.get("verse_key"):
                dataset, quran_map = [], {}
                for surah_num, ayahs in data.items():
                    for ayah_num, item in ayahs.items():
                        verse_key = f"{surah_num}:{ayah_num}"
                        entry = {"verse_key": verse_key, **item}
                        dataset.append(entry)
                        quran_map[verse_key] = entry
                return dataset, quran_map
    return [], {}


try:
    quran_dataset, quran_map = _load_quran()
    if not quran_map:
        print("Warning: No Quran data found. Add quran.json or quran_with_audio.json or set QURAN_DATA_PATH")
except Exception as e:
    print(f"Error loading Quran: {e}")
    quran_dataset = []
    quran_map = {}

inference_lock = threading.Lock()

# Model selection from env (Whisper only loaded when ASR_USE_WHISPER=true for 2–3s target)
_device = resolve_device() if callable(resolve_device) else "cpu"
print("Loading Intelligence Engines...")
whisper_model = None
if ASR_USE_WHISPER:
    whisper_model = whisper.load_model(
        os.environ.get("WHISPER_MODEL", WHISPER_MODEL),
        device=_device,
    )
    print("Whisper loaded (dual-ASR mode).")
phonetic_analyzer = PhoneticAnalyzer(
    model_name=os.environ.get("WAV2VEC2_MODEL", WAV2VEC2_MODEL),
    device=_device,
    quantize_8bit=WAV2VEC2_QUANTIZE_8BIT,
    use_torch_compile=TORCH_COMPILE,
    beam_width=BEAM_WIDTH,
)
tajweed_engine = TajweedRulesEngine()
print("All Engines Loaded." + (" (Wav2Vec2-only: target 2–3s)" if not ASR_USE_WHISPER else ""))

os.makedirs("static", exist_ok=True)
app.mount("/demo-static", StaticFiles(directory="static"), name="static")


@app.get("/demo", include_in_schema=False)
@app.get("/demo/", include_in_schema=False)
async def serve_demo():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except Exception:
        raise HTTPException(status_code=404, detail="Demo UI not found.")


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Quran AI Production API is running",
        "verses_loaded": len(quran_dataset),
    }


@app.get("/verses/{surah}/{ayah}", response_model=None)
def get_verse(surah: int, ayah: int):
    verse_key = f"{surah}:{ayah}"
    verse_data = quran_map.get(verse_key)
    if not verse_data:
        raise HTTPException(status_code=404, detail=f"Ayah {verse_key} not found")
    text = verse_data.get("text_uthmani", "")
    words = [w for w in text.split() if normalize_arabic(w)]
    return {
        "verse_key": verse_key,
        "text_uthmani": text,
        "words": [{"word": w, "start_time": None, "end_time": None} for w in words],
        "translation_en": verse_data.get("translation_en", ""),
        "transliteration": verse_data.get("transliteration", ""),
    }


def _run_verification_pipeline(temp_filename: str, verse_key: str, original_text: str):
    """
    3-layer verification: ASR → CTC alignment → Tajweed → scoring.
    Never raises; returns result with word_analysis and teacher_feedback always.
    Adds timing_ms when available; falls back to current behavior on any failure.
    """
    import time
    _start = time.perf_counter()
    timing_ms = {}

    default_result = {
        "transcribed_text": "",
        "accuracy_score": 0.0,
        "word_analysis": [],
        "extra_words": [],
        "tajweed_score": 100.0,
        "confidence_level": "low",
        "confidence_score": 0.0,
        "teacher_feedback_text": "Could not complete verification. Please check audio length and try again.",
        "tajweed_feedback": [],
        "timing_data": [],
        "asr_result": None,
        "wer": None,
        "cer": None,
    }
    import librosa
    import torch.nn.functional as F
    y, sr = None, 16000
    duration_sec = 0.0
    try:
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("ignore", FutureWarning)
            # Cap duration so pipeline stays ~2–3s (config: VERIFY_MAX_AUDIO_DURATION_SEC)
            y, sr = librosa.load(temp_filename, sr=16000, duration=VERIFY_MAX_AUDIO_DURATION_SEC)
        duration_sec = len(y) / sr
        timing_ms["load_audio"] = round((time.perf_counter() - t0) * 1000)
        if duration_sec < 0.3:
            print(f"[Verify] verse_key={verse_key} skipped: audio too short ({round(duration_sec, 2)}s)")
            default_result["teacher_feedback_text"] = "Audio is too short. Please recite at least part of the verse."
            return default_result
    except Exception:
        pass

    # Layer 1 — Dual ASR (single entry point; reuse logits for alignment and Tajweed)
    asr_result = None
    try:
        t1 = time.perf_counter()
        asr_result = run_dual_asr(
            whisper_model=whisper_model,
            phonetic_analyzer=phonetic_analyzer,
            reference_text=original_text,
            audio_path=temp_filename,
            audio_y_sr=(y, sr) if y is not None else None,
            whisper_beam_size=WHISPER_BEAM_SIZE,
            use_whisper=ASR_USE_WHISPER,
        )
        timing_ms["asr"] = round((time.perf_counter() - t1) * 1000)
    except Exception as e:
        print(f"ASR error: {e}")
        asr_result = {
            "selected_transcript": "",
            "selected_source": "wav2vec",
            "selection_reason": "error",
            "logits": None,
        }
    if asr_result is None:
        asr_result = {
            "selected_transcript": "",
            "selected_source": "wav2vec",
            "selection_reason": "error",
            "logits": None,
        }
    transcribed_text = asr_result["selected_transcript"]
    logits = asr_result.get("logits")
    # Strip logits from asr_result for JSON response (tensors not serializable)
    asr_result_serializable = {k: v for k, v in asr_result.items() if k != "logits"}

    # Layer 2 — CTC forced alignment (reuse emission from single forward when available)
    alignment_words = []
    ref_words = [w for w in original_text.split() if normalize_arabic(w)]
    ref_norm = normalize_arabic(original_text)
    emission = None
    if logits is not None:
        emission = F.log_softmax(logits, dim=-1)[0].cpu()
    try:
        t2 = time.perf_counter()
        align_result = align_reference_to_audio(
            processor=phonetic_analyzer.processor,
            model=phonetic_analyzer.model,
            audio_path=temp_filename if emission is None else None,
            reference_text=ref_norm,
            reference_words=ref_words,
            device=phonetic_analyzer.device,
            load_audio_fn=load_audio_for_alignment,
            emission=emission,
        )
        timing_ms["alignment"] = round((time.perf_counter() - t2) * 1000)
        if align_result.get("alignment_success") and align_result.get("words"):
            alignment_words = align_result["words"]
    except Exception:
        pass

    # Layer 3 — Tajweed light only: Madd, Qalqalah, Ghunnah (segment-level CTC when alignment available)
    tajweed_feedback = []
    try:
        t3 = time.perf_counter()
        if y is not None and len(y) > 0:
            if alignment_words:
                frames_per_word = None
                if logits is not None:
                    phonetic_results = phonetic_analyzer.analyze_alignment_from_logits(logits)
                    frames = phonetic_results.get("frames") or []
                    if frames and alignment_words:
                        import numpy as np
                        n_frames, n_words = len(frames), len(alignment_words)
                        frames_per_word = []
                        for i in range(n_words):
                            s = alignment_words[i].get("start_time") or 0.0
                            e = alignment_words[i].get("end_time") or s + 0.1
                            f_start = int(s * 50)
                            f_end = min(int(e * 50), n_frames)
                            frames_per_word.append(frames[f_start:f_end] if f_end > f_start else [])
                fb, _ = tajweed_engine.get_teacher_feedback_segment_level(
                    alignment_words, y, sr, frames_per_word=frames_per_word
                )
                tajweed_feedback = fb
            else:
                # Single segment = full audio for light rules only
                fake_alignment = [{"start_time": 0.0, "end_time": duration_sec}]
                tajweed_feedback = tajweed_engine.get_teacher_feedback_light(fake_alignment, y, sr)
        timing_ms["tajweed"] = round((time.perf_counter() - t3) * 1000)
    except Exception:
        pass

    try:
        t4 = time.perf_counter()
        final_result = score_recitation(
            original_text,
            transcribed_text,
            tajweed_feedback=tajweed_feedback,
            alignment_words=alignment_words if alignment_words else None,
            audio_duration=duration_sec,
        )
        timing_ms["scoring"] = round((time.perf_counter() - t4) * 1000)
    except Exception:
        final_result = score_recitation(
            original_text,
            transcribed_text,
            tajweed_feedback=tajweed_feedback,
            alignment_words=alignment_words if alignment_words else None,
            audio_duration=duration_sec,
        )
    timing_ms["total"] = round((time.perf_counter() - _start) * 1000)
    if timing_ms:
        final_result["timing_ms"] = timing_ms
        # Log pipeline timing for observability
        parts = [f"{k}={v}ms" for k, v in timing_ms.items()]
        print(f"[Verify] verse_key={verse_key} audio_sec={round(duration_sec, 1)}s " + " ".join(parts))
    final_result["asr_result"] = asr_result_serializable
    final_result["wer"] = round(wer(original_text, transcribed_text), 4) if transcribed_text else None
    final_result["cer"] = round(cer(original_text, transcribed_text), 4) if transcribed_text else None
    return final_result


@app.post("/verify")
async def verify_recitation(
    surah: int = Query(..., description="Surah number (1-114)"),
    ayah: int = Query(..., description="Ayah number"),
    audio: UploadFile = File(...),
):
    verse_key = f"{surah}:{ayah}"
    verse_data = quran_map.get(verse_key)
    if not verse_data:
        raise HTTPException(status_code=404, detail=f"Ayah {verse_key} not found")
    original_text = verse_data.get("text_uthmani", "")

    temp_filename = f"temp_verify_{uuid.uuid4().hex}_{audio.filename or 'audio'}"

    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)

        def _run_with_lock():
            with inference_lock:
                return _run_verification_pipeline(temp_filename, verse_key, original_text)

        loop = asyncio.get_event_loop()
        executor_task = loop.run_in_executor(None, _run_with_lock)
        try:
            final_result = await asyncio.wait_for(executor_task, timeout=VERIFY_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"Verification timed out after {VERIFY_TIMEOUT_SEC}s. Try a shorter recording.",
            )
        return final_result
    except asyncio.CancelledError:
        raise
    except HTTPException:
        raise
    except Exception as e:
        print(f"Verify Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except Exception:
                pass


@app.websocket("/ws/verify")
async def websocket_verify(websocket: WebSocket):
    """
    Real-time streaming: accept audio chunks, sliding buffer, 2–3 s overlapping windows.
    Send partial_results at WS_PARTIAL_RESULT_INTERVAL; do NOT re-run full verse inference on every chunk.
    """
    params = websocket.query_params
    surah, ayah = params.get("surah"), params.get("ayah")
    if not surah or not ayah:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    verse_key = f"{surah}:{ayah}"
    audio_buffer = io.BytesIO()
    analysis_task = None
    threshold = WS_BUFFER_THRESHOLD
    partial_interval = WS_PARTIAL_RESULT_INTERVAL
    last_partial_sent_at: float = 0.0

    async def process_incremental(buffer_data: bytes):
        """Run ASR on current buffer only (no full-verse CTC/Tajweed). Send partial_result to UI."""
        nonlocal last_partial_sent_at
        temp_partial = f"temp_ws_{verse_key.replace(':', '_')}_{uuid.uuid4().hex}.webm"
        try:
            with open(temp_partial, "wb") as f:
                f.write(buffer_data)
            loop = asyncio.get_event_loop()

            def run_ws_inference():
                with inference_lock:
                    try:
                        result = whisper_model.transcribe(temp_partial, language="ar")
                        return result.get("text", "").strip()
                    except Exception:
                        return ""

            partial_text = await loop.run_in_executor(None, run_ws_inference)
            try:
                await websocket.send_json({
                    "type": "partial_result",
                    "transcribed": partial_text,
                    "status": "analyzing",
                })
                last_partial_sent_at = loop.time()
            except (WebSocketDisconnect, RuntimeError):
                pass
        except Exception as e:
            print(f"WS Worker Error: {e}")
        finally:
            if os.path.exists(temp_partial):
                try:
                    os.remove(temp_partial)
                except Exception:
                    pass

    try:
        loop = asyncio.get_event_loop()
        while True:
            data = await websocket.receive_bytes()
            audio_buffer.write(data)
            now = loop.time()
            # Only trigger partial inference when buffer exceeds threshold and interval elapsed (or first time)
            if audio_buffer.tell() > threshold and (analysis_task is None or analysis_task.done()):
                if now - last_partial_sent_at >= partial_interval or last_partial_sent_at == 0.0:
                    analysis_task = asyncio.create_task(process_incremental(audio_buffer.getvalue()))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WS Error: {e}")
        traceback.print_exc()


# ----- Phase 4: Streaming /ws/recite (new endpoint; does not replace /ws/verify or /verify) -----
def _get_reference_text(verse_key: str) -> str:
    return (quran_map.get(verse_key) or {}).get("text_uthmani", "")


def _run_batch_verification_from_bytes(verse_key: str, reference_text: str, audio_bytes: bytes):
    """Fail-safe: run full batch pipeline on buffered audio when streaming session ends."""
    if not audio_bytes or len(audio_bytes) < 3200:
        return {"error": "Audio too short for verification", "accuracy_score": 0.0}
    from streaming.streaming_asr import _raw_to_wav
    wav_bytes = _raw_to_wav(audio_bytes)
    path = f"temp_ws_final_{verse_key.replace(':', '_')}_{uuid.uuid4().hex}.wav"
    try:
        with open(path, "wb") as f:
            f.write(wav_bytes)
        return _run_verification_pipeline(path, verse_key, reference_text)
    finally:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass


try:
    from streaming.websocket_server import build_ws_recite_handler
    try:
        _ws_window_ms = float(os.environ.get("WS_WINDOW_SECONDS", "2.5")) * 1000
        _ws_overlap_ms = float(os.environ.get("WS_OVERLAP_SECONDS", "0.5")) * 1000
        _ws_max_queue = int(os.environ.get("WS_MAX_QUEUE", "3"))
    except Exception:
        _ws_window_ms, _ws_overlap_ms, _ws_max_queue = 2500.0, 500.0, 3
    try:
        import metrics.streaming_metrics as _streaming_metrics
        _get_adaptive_window_ms = lambda: _streaming_metrics.get_adaptive_window_seconds() * 1000
        _metrics_module = _streaming_metrics
    except ImportError:
        _get_adaptive_window_ms = None
        _metrics_module = None
    _ws_recite_handler = build_ws_recite_handler(
        get_whisper_model=lambda: whisper_model,
        get_reference_text=lambda k: _get_reference_text(k),
        get_inference_lock=lambda: inference_lock,
        run_batch_verification=_run_batch_verification_from_bytes,
        window_duration_ms=_ws_window_ms,
        overlap_duration_ms=_ws_overlap_ms,
        max_queue_depth=_ws_max_queue,
        get_adaptive_window_ms=_get_adaptive_window_ms,
        get_metrics=_metrics_module,
    )
    app.websocket("/ws/recite")(_ws_recite_handler)
except ImportError as e:
    print(f"Phase 4 streaming not registered: {e}")


# ----- Phase 4.1: Observability GET /metrics/streaming (JSON snapshot) -----
try:
    from metrics.streaming_metrics import get_snapshot
    @app.get("/metrics/streaming", include_in_schema=False)
    def metrics_streaming():
        """Return JSON snapshot of streaming metrics: active_connections, avg_latency_ms, p95_latency_ms, asr_failure_count, dropped_segments."""
        return get_snapshot()
except ImportError:
    pass


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("HOST", HOST), port=int(os.environ.get("PORT", PORT)))
