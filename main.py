import os
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
from core.normalization import normalize_arabic
from core.scoring import score_recitation
from core.phonetics import PhoneticAnalyzer
from core.tajweed import TajweedRulesEngine

app = FastAPI(title="Quran AI Production API")

# Enable CORS safely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _load_quran():
    """Load Quran data: list format (verse_key) or nested (surah -> ayah)."""
    # Try flat list first (quran_with_audio.json)
    path1 = "quran_with_audio.json"
    if os.path.exists(path1):
        with open(path1, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            quran_map = {item["verse_key"]: item for item in data}
            return list(quran_map.values()), quran_map
    # Fallback: nested quran.json
    path2 = "quran.json"
    if os.path.exists(path2):
        with open(path2, "r", encoding="utf-8") as f:
            nested = json.load(f)
        dataset = []
        quran_map = {}
        for surah_num, ayahs in nested.items():
            for ayah_num, item in ayahs.items():
                verse_key = f"{surah_num}:{ayah_num}"
                entry = {"verse_key": verse_key, **item}
                dataset.append(entry)
                quran_map[verse_key] = entry
        print(f"Loaded {len(dataset)} verses from {path2} (nested format)")
        return dataset, quran_map
    return [], {}

try:
    quran_dataset, quran_map = _load_quran()
    if not quran_map:
        print("Warning: No Quran data found. Add quran.json or quran_with_audio.json")
except Exception as e:
    print(f"Error loading Quran: {e}")
    quran_dataset = []
    quran_map = {}

# Single Global Lock for thread-safe inference
inference_lock = threading.Lock()

# Initialize Intelligence Engines
print("Loading Intelligence Engines...")
whisper_model = whisper.load_model("base")
phonetic_analyzer = PhoneticAnalyzer()
tajweed_engine = TajweedRulesEngine()
print("All Engines Loaded.")

# Mount Static Files for Demo
os.makedirs("static", exist_ok=True)
app.mount("/demo-static", StaticFiles(directory="static"), name="static")

@app.get("/demo", include_in_schema=False)
@app.get("/demo/", include_in_schema=False)
async def serve_demo():
    """Explicit endpoint to serve the demo page accurately, circumventing trailing slash issues."""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        raise HTTPException(status_code=404, detail="Demo UI not found.")

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Quran AI Production API is running safely",
        "verses_loaded": len(quran_dataset),
    }


@app.get("/verses/{surah}/{ayah}", response_model=None)
def get_verse(surah: int, ayah: int):
    """Get one verse by surah and ayah number (for UI verse preview)."""
    verse_key = f"{surah}:{ayah}"
    verse_data = quran_map.get(verse_key)
    if not verse_data:
        raise HTTPException(status_code=404, detail=f"Ayah {verse_key} not found")
    return {
        "verse_key": verse_key,
        "text_uthmani": verse_data.get("text_uthmani", ""),
        "translation_en": verse_data.get("translation_en", ""),
        "transliteration": verse_data.get("transliteration", ""),
    }


@app.post("/verify")
async def verify_recitation(
    surah: int = Query(..., description="Surah number (1-114)"),
    ayah: int = Query(..., description="Ayah number"),
    audio: UploadFile = File(...)
):
    verse_key = f"{surah}:{ayah}"
    verse_data = quran_map.get(verse_key)
    
    if not verse_data:
        raise HTTPException(status_code=404, detail=f"Ayah {verse_key} not found")
    
    original_text = verse_data.get("text_uthmani", "")
    
    # Use UUID to prevent file lock crashes from multiple concurrent users
    temp_filename = f"temp_verify_{uuid.uuid4().hex}_{audio.filename}"
    
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        
        # Move blocking tasks to executor to avoid hanging event loop (WebSocket drop/CORS fetch error fix)
        print(f"Pipeline processing: {verse_key} - Async Executor")
        
        loop = asyncio.get_event_loop()
        
        def run_full_inference():
            # Synchronize single-gpu/cpu AI loading
            with inference_lock:
                transcription = whisper_model.transcribe(temp_filename, language="ar")
                transcribed_text = transcription["text"]
                
                phonetic_results = phonetic_analyzer.analyze_alignment(temp_filename, original_text)
                acoustic_features = phonetic_analyzer.get_phonetic_features(temp_filename)
                tajweed_feedback = tajweed_engine.get_teacher_feedback(phonetic_results, acoustic_features)
                
                final_result = score_recitation(original_text, transcribed_text, tajweed_feedback)
                return final_result

        final_result = await loop.run_in_executor(None, run_full_inference)
        return final_result

    except Exception as e:
        print(f"Verify Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.websocket("/ws/verify")
async def websocket_verify(websocket: WebSocket):
    # Manually parse query parameters for better stability and error logging
    params = websocket.query_params
    surah = params.get("surah")
    ayah = params.get("ayah")

    print(f"WS Attempt: surah={surah}, ayah={ayah}")

    if not surah or not ayah:
        print("WS Rejection: Missing surah or ayah parameters")
        await websocket.close(code=1008)
        return

    await websocket.accept()
    print(f"WS Connection Established for {surah}:{ayah} ✅")
    
    verse_key = f"{surah}:{ayah}"
    verse_data = quran_map.get(verse_key)
    
    audio_buffer = io.BytesIO()
    analysis_task = None
    
    async def process_chunks(buffer_data):
        # Use UUID to avoid multi-client partial temp file collisions
        temp_partial = f"temp_ws_{verse_key}_{uuid.uuid4().hex}.webm"
        try:
            with open(temp_partial, "wb") as f:
                f.write(buffer_data)
            
            loop = asyncio.get_event_loop()
            
            def run_ws_inference():
                with inference_lock:
                    result = whisper_model.transcribe(temp_partial, language="ar")
                    return result["text"]
                    
            partial_text = await loop.run_in_executor(None, run_ws_inference)
            
            # Send result only if WS is still alive
            try:
                await websocket.send_json({
                    "type": "partial_result",
                    "transcribed": partial_text,
                    "status": "analyzing"
                })
            except (WebSocketDisconnect, RuntimeError):
                pass
        except Exception as e:
            print(f"WS Worker Error: {e}")
        finally:
            if os.path.exists(temp_partial):
                os.remove(temp_partial)

    try:
        while True:
            data = await websocket.receive_bytes()
            audio_buffer.write(data)
            
            # Threshold: ~150KB
            if audio_buffer.tell() > 150000 and (analysis_task is None or analysis_task.done()):
                print(f"Triggering partial analysis for {verse_key} buffer size: {audio_buffer.tell()}")
                analysis_task = asyncio.create_task(process_chunks(audio_buffer.getvalue()))

    except WebSocketDisconnect:
        print(f"WS Disconnected: {verse_key} ✅ normal close")
    except Exception as e:
        print(f"WS Runtime Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)