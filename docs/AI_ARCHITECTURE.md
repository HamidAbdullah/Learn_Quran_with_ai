# Quran AI Backend - Architecture & Model Review

This document provides a comprehensive technical breakdown of how the Quran AI Recitation Backend was built, detailing its machine learning architecture, training provisions, inference pipeline, and systemic stability measures.

---

## 1. How the AI Was Built (Architecture)

The backend employs a **Hybrid Intelligence System**, combining State-of-the-Art Deep Learning models with a deterministic rule-based linguistic engine to score and analyze Quranic recitation.

### Core Dependencies & Packages
The key packages powering the intelligence and web layers include:
- **FastAPI & Uvicorn**: Provide high-performance asynchronous HTTP and WebSocket streaming.
- **OpenAI Whisper (`openai-whisper`)**: Used for the primary Speech-to-Text transcription (loading the `"base"` model).
- **Transformers (`transformers`)**: Hugging Face pipeline handling Wav2Vec2 models.
- **PyTorch (`torch`, `torchaudio`)**: The fundamental deep learning framework for tensor calculations and executing models on CPU/CUDA.
- **Librosa (`librosa`)**: Essential for acoustic feature extraction (pitch, RMS energy, duration).
- **RapidFuzz (`rapidfuzz`)**: Ultra-fast fuzzy string matching used for word-by-word sequence alignment and deviation detection.

### The Intelligence Pipeline 
When a user submits an audio recitation, it moves sequentially through multiple layers:

1. **Transcription (Speech-to-Text)**: 
   `whisper.load_model("base")` parses the Arabic audio and generates a raw string of what the AI "heard".
2. **Acoustic & Phonetic Extraction (Wav2Vec2)**: 
   The `PhoneticAnalyzer` takes the same audio and passes it through `jonatasgrosman/wav2vec2-large-xlsr-53-arabic`. This performs forced alignment, returning Character Error approximations and frame-by-frame logit probabilities.
   Simultaneously, *Librosa* extracts pure acoustic features—such as RMS intensity and pitch track.
3. **Tajweed Rule Validation (Rule Engine)**: 
   The `TajweedRulesEngine` uses the numeric outputs from the acoustic and phonetic layers (like sudden leaps in RMS intensity indicating "bursts" of energy) to evaluate specific Tajweed rules mathematically. For instance, **Qalqalah** (bouncing sound) is detected by taking `np.diff(intensity_data)` and looking for bounds > 0.05.
4. **Scoring & Normalization**: 
   The `scoring.py` script normalizes both the Quranic ground truth (stripping Uthmani diacritics via regex) and the transcribed text. It then performs exact sequence matching and RapidFuzz confidence scores to give fine-grained feedback like "Missing Word" or "Minor Phoneme Slip", merging this with Tajweed accuracy for a final graded score out of 100.

---

## 2. How Model Training is Implemented

Currently, the production backend acts strictly as an **Inference Engine** (running pre-trained state-of-the-art weights in direct evaluation mode rather than actively shifting weights/fine-tuning dynamically). 

However, the foundation for **Fine-Tuning / Post-Training** is fully scaffolding out in the `scripts/generate_training_dataset.py` file. 

*   **Dataset Generation Pipeline**: 
    The script prepares rigorous training sets by iterating over the local `quran_with_audio.json`. 
*   **Normalization Strategy**: 
    It intentionally scrubs diacritics, Alif/Ya variations, and Quranic stop-marks structurally to generate simplified, normalized text targets alongside raw `.mp3`/`.wav` references.
*   **Manifest Creation**: 
    It builds a unified JSON (`training_dataset.json`) and a CSV Manifest (`manifest.csv`). In standard Deep Learning (e.g. HuggingFace Trainer, Kaldi, or Whisper's Fine-Tuning paradigm), producing this audio-transcription manifest is the mandatory Step 1. 

**Conclusion for Training**: The code is primed perfectly to pipe the generated `manifest.csv` into a PyTorch fine-tuning loop in the future (perhaps to adapt Whisper's exact Uthmani diacritic output or to train specific reciter intonations). 

---

## 3. Stability & Infrastructure Security

The repository contains several highly robust, production-level guards to ensure stability under load and handle asynchronous network conditions:

### 1. Thread-Safe Global Locking
Since Whisper and Wav2Vec2 are immensely heavy processes computationally, running concurrent FastAPI web requests blindly would crash the server with "Out Of Memory" (OOM) or conflicting CUDA contexts. 
*   **The Fix**: A global `inference_lock = threading.Lock()` limits massive GPU/CPU tensor shifts to one concurrent process. Requests effectively form a safe queue mathematically rather than imploding the RAM.

### 2. Event-Loop Relief (Asynchronous Escaping)
Heavy ML operations naturally *block* event loops built for websockets, causing WebSockets to forcefully disconnect or CORS Fetch requests to timeout. 
*   **The Fix**: The codebase uses `asyncio.get_event_loop().run_in_executor(None, run_full_inference)`—shunting the blocking ML models to a separate background thread precisely so the main web networking loop can continue ticking over safely. 

### 3. I/O Isolation & Garbage Collection
*   Files are dynamically saved for inference using `uuid.uuid4().hex`. This ensures that if 10 users upload recitations at exactly the same millisecond, the temp files do not collide on the filesystem.
*   Try-except-finally blocks consistently intercept crashes. Crucially, the temp `.webm`/`.wav` files are manually pruned out (`os.remove`) inside the `finally` conditions. This is fundamental server stability to prevent local disk space from eventually filling up.
*   The WebSocket endpoint correctly limits buffer sizes to safe transmission chunks (~150KB loops) ensuring bandwidth streams incrementally without RAM spiking.
