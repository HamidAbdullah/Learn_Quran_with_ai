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
    WAV2VEC2_QUANTIZE_8BIT = _config.WAV2VEC2_QUANTIZE_8BIT
    TORCH_COMPILE = _config.TORCH_COMPILE
    WS_CHUNK_THRESHOLD_BYTES = _config.WS_CHUNK_THRESHOLD_BYTES
    WS_BUFFER_THRESHOLD = _config.WS_BUFFER_THRESHOLD
    WS_PARTIAL_RESULT_INTERVAL = _config.WS_PARTIAL_RESULT_INTERVAL
except ImportError:
    get_quran_path = lambda: ""
    get_cors_origins = lambda: ["*"]
    resolve_device = lambda: ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    PORT, HOST = 8001, "0.0.0.0"
    WHISPER_MODEL = "base"
    WAV2VEC2_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
    WAV2VEC2_QUANTIZE_8BIT = False
    TORCH_COMPILE = False
    WS_CHUNK_THRESHOLD_BYTES = 150000
    WS_BUFFER_THRESHOLD = 150000
    WS_PARTIAL_RESULT_INTERVAL = 2.5

from core.normalization import normalize_arabic
from core.scoring import score_recitation, segment_transcript_by_reference, count_matching_words
from core.phonetics import PhoneticAnalyzer, load_audio_for_alignment
from core.tajweed import TajweedRulesEngine
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

# Model selection from env
_device = resolve_device() if callable(resolve_device) else "cpu"
print("Loading Intelligence Engines...")
whisper_model = whisper.load_model(
    os.environ.get("WHISPER_MODEL", WHISPER_MODEL),
    device=_device,
)
phonetic_analyzer = PhoneticAnalyzer(
    model_name=os.environ.get("WAV2VEC2_MODEL", WAV2VEC2_MODEL),
    device=_device,
    quantize_8bit=WAV2VEC2_QUANTIZE_8BIT,
    use_torch_compile=TORCH_COMPILE,
)
tajweed_engine = TajweedRulesEngine()
print("All Engines Loaded.")

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
    """
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
    }
    import librosa
    import torch.nn.functional as F
    y, sr = None, 16000
    duration_sec = 0.0
    try:
        y, sr = librosa.load(temp_filename, sr=16000, duration=300)
        duration_sec = len(y) / sr
        if duration_sec < 0.3:
            default_result["teacher_feedback_text"] = "Audio is too short. Please recite at least part of the verse."
            return default_result
    except Exception:
        pass

    # Layer 1 — Dual ASR (single Wav2Vec2 forward; reuse logits for alignment and Tajweed)
    whisper_text = ""
    try:
        trans = whisper_model.transcribe(temp_filename, language="ar")
        whisper_text = (trans.get("text") or "").strip()
    except Exception:
        pass
    wav2vec_text = ""
    logits = None
    if y is not None:
        wav2vec_text, logits = phonetic_analyzer.run_forward(y, sr)
    if not wav2vec_text:
        wav2vec_text = phonetic_analyzer.transcribe(temp_filename) or ""
    if wav2vec_text and wav2vec_text.count(" ") < 2:
        wav2vec_text = segment_transcript_by_reference(wav2vec_text, original_text)
    if whisper_text and wav2vec_text:
        w_match = count_matching_words(original_text, wav2vec_text)
        wh_match = count_matching_words(original_text, whisper_text)
        transcribed_text = wav2vec_text if w_match >= wh_match else whisper_text
    else:
        transcribed_text = whisper_text or wav2vec_text

    # Layer 2 — CTC forced alignment (reuse emission from single forward when available)
    alignment_words = []
    ref_words = [w for w in original_text.split() if normalize_arabic(w)]
    ref_norm = normalize_arabic(original_text)
    emission = None
    if logits is not None:
        emission = F.log_softmax(logits, dim=-1)[0].cpu()
    try:
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
        if align_result.get("alignment_success") and align_result.get("words"):
            alignment_words = align_result["words"]
    except Exception:
        pass

    # Layer 3 — Tajweed (reuse logits and preloaded audio)
    tajweed_feedback = []
    try:
        if logits is not None:
            phonetic_results = phonetic_analyzer.analyze_alignment_from_logits(logits)
        else:
            phonetic_results = phonetic_analyzer.analyze_alignment(temp_filename, original_text)
        acoustic_features = (
            phonetic_analyzer.get_phonetic_features_from_audio(y, sr)
            if y is not None
            else phonetic_analyzer.get_phonetic_features(temp_filename)
        )
        if phonetic_results.get("frames") or acoustic_features.get("intensity"):
            tajweed_feedback = tajweed_engine.get_teacher_feedback(phonetic_results, acoustic_features)
    except Exception:
        pass

    # Scoring with time-alignment when available
    final_result = score_recitation(
        original_text,
        transcribed_text,
        tajweed_feedback=tajweed_feedback,
        alignment_words=alignment_words if alignment_words else None,
        audio_duration=duration_sec,
    )
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
        final_result = await loop.run_in_executor(None, _run_with_lock)
        return final_result
    except asyncio.CancelledError:
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


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("HOST", HOST), port=int(os.environ.get("PORT", PORT)))
