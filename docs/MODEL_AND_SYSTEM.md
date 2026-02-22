# Quran AI: Model, Stability, Accuracy & Libraries

This document describes **how the recitation verification model works**, **stability and robustness**, **accuracy**, **GPU/CPU usage and model sizes**, and **which libraries are used and why**.

---

## 0. Three-layer verification pipeline (current architecture)

| Layer | Role |
|-------|------|
| **Layer 1 — ASR** | Dual ASR: Whisper (or Quran Whisper) + Wav2Vec2 Arabic/Quran. Best transcript chosen by word-match score vs reference and confidence. Arabic normalized (ي/ی, ك/ک, Hamza, Tatweel) before comparison. |
| **Layer 2 — Forced alignment** | CTC forced alignment (Viterbi on Wav2Vec2 emissions) yields **word-level** `start_time`, `end_time`, `confidence`, `phonetic_similarity`. Frame-level probability alignment; no greedy word splitting only. |
| **Layer 3 — Tajweed** | Acoustic + phonetic: Qalqalah (intensity gradient), Madd (duration), Ghunnah (nasal band / frame confidence), Idgham (transition smoothness). Librosa RMS, spectral rolloff; fallback heuristics. |

**Scoring**: When alignment is available, per-word score = **0.6 × text_similarity + 0.3 × phonetic_similarity + 0.1 × timing_similarity**. Thresholds: ≥92% → correct; 78–92% → minor_mistake; &lt;78% → wrong. Response always includes `word_analysis`, `teacher_feedback_text`, confidence scores; pipeline never crashes on short/noise/accents.

**Configuration**: `python-dotenv` + env (e.g. `WHISPER_MODEL`, `WAV2VEC2_MODEL`, `DEVICE`, `WAV2VEC2_QUANTIZE_8BIT`, `TORCH_COMPILE`, `QURAN_DATA_PATH`, `PORT`, `CORS_ORIGINS`). See `.env.example`.

---

## 1. How the model works (end-to-end pipeline)

### 1.1 High-level flow

```
User records audio → Upload (POST /verify) or WebSocket chunks
       ↓
Audio saved to temp file (UUID-named to avoid collisions)
       ↓
┌──────────────────────────────────────────────────────────────────┐
│  LAYER 1 — ASR (single lock)                                      │
│  Whisper + Wav2Vec2-Arabic transcribe; segment by reference if    │
│  needed; pick best transcript by count_matching_words.           │
│  LAYER 2 — CTC forced alignment                                   │
│  Wav2Vec2 emissions → trellis + Viterbi → word start/end/confidence│
│  LAYER 3 — Tajweed                                                 │
│  RMS, spectral rolloff, frame confidence → Qalqalah, Madd,       │
│  Ghunnah, Idgham, Ikhfa, Meem Sakina.                            │
│  SCORING                                                          │
│  0.6×text + 0.3×phonetic + 0.1×timing per word; word_analysis     │
│  with optional start_time/end_time.                              │
└──────────────────────────────────────────────────────────────────┘
       ↓
Temp file deleted; JSON response with word_analysis, scores, feedback
```

### 1.2 Components

| Component | Role |
|-----------|------|
| **Whisper (base)** | Primary ASR; outputs Arabic text **with spaces** (good word boundaries). |
| **Wav2Vec2-Arabic (XLS-R 53)** | Second ASR; strong on Arabic; output often has **no spaces** (CTC). Used for transcript choice and for Tajweed (frame-level confidence). |
| **Segment-by-reference** | When Wav2Vec2 returns one long string, we split it into words by aligning to the reference verse (greedy match per word). |
| **Best-transcript choice** | For each of Whisper and Wav2Vec2 we count how many reference words have a match; we keep the transcript with the **higher count** for final scoring. |
| **Normalization** | Reference and transcript are normalized (same script, no diacritics, hamza/alif unified) so comparison is fair. |
| **Word alignment** | `SequenceMatcher` (diff) + **optimal pairing** in “replace” blocks: best ref–user word pairs by similarity first, so the right spoken word is matched to the right verse word. |
| **Scoring thresholds** | ≥92% similarity → **correct**; 78–92% → **minor_mistake**; &lt;78% → **wrong**. Similarity is ratio-first (strict), with a small partial_ratio boost for near-matches only. |
| **Tajweed engine** | Uses Wav2Vec2 frame-level confidence and librosa acoustic features (RMS, duration). Qalqalah uses intensity gradient; most other rules are placeholder feedback. |

### 1.3 Data flow

- **Quran text**: Loaded at startup from `quran.json` (nested) or `quran_with_audio.json` (flat list). Keys: `verse_key`, `text_uthmani`, `translation_en`, etc.
- **Verse lookup**: `GET /verses/{surah}/{ayah}` returns the verse for the UI. `POST /verify` and WebSocket use the same `quran_map` to get `text_uthmani` for the chosen verse.
- **Response**: `accuracy_score` (word-level), `tajweed_score`, `word_analysis` (per-word status and confidence), `extra_words`, `teacher_feedback_text`, `tajweed_feedback`.

---

## 2. Stability

### 2.1 Concurrency and resources

- **Single `inference_lock`**: All model runs (Whisper, Wav2Vec2, Tajweed) for a given request run inside one lock. Only one verification runs at a time, avoiding GPU/CPU contention and OOM.
- **Async + executor**: Heavy work runs in `run_in_executor(None, run_full_inference)` so the FastAPI event loop is not blocked; other requests (e.g. GET /verses, static) can still be served.
- **Temp files**: UUID in filenames (`temp_verify_{uuid}_{filename}`) to avoid collisions between concurrent users; files are removed in a `finally` block.

### 2.2 Error handling

- **CancelledError**: Re-raised so server shutdown (e.g. Ctrl+C) does not turn into 500.
- **Tajweed pipeline**: Wrapped in try/except; on failure we still return word-level scoring with empty or default Tajweed feedback.
- **Phonetics**: Short audio (&lt;0.5 s) or load errors return `None` or safe defaults; no crash.
- **Audio loading**: Librosa warnings (PySoundFile fallback, deprecations) are suppressed in `_load_audio()` so logs stay clean.

### 2.3 WebSocket

- **Partial analysis**: When buffer &gt; ~150KB, a task runs Whisper on the current buffer and sends `partial_result`; buffer is **not** cleared between runs (known limitation: same audio can be re-processed).
- **Reconnect**: Client can reconnect with backoff; server does not persist session state.

### 2.4 Configuration and security

- **Config**: `config.py` + `python-dotenv`; env vars: `PORT`, `HOST`, `QURAN_DATA_PATH`, `WHISPER_MODEL`, `WAV2VEC2_MODEL`, `DEVICE`, `WAV2VEC2_QUANTIZE_8BIT`, `TORCH_COMPILE`, `WS_CHUNK_THRESHOLD_BYTES`, `CORS_ORIGINS`. See `.env.example`.
- **CORS**: Set `CORS_ORIGINS` to specific origins in production.

---

## 3. Accuracy

### 3.1 What drives accuracy

- **ASR quality**: Whisper base is decent for Arabic; Wav2Vec2-Arabic (XLS-R 53) is strong. Choosing the transcript that **matches the reference better** (by word count) improves consistency.
- **Normalization**: Same script and rules for reference and transcript (e.g. ي/ی, ك/ک, hamza/alif) so words are comparable.
- **Word alignment**: Optimal pairing in “replace” blocks reduces correct words being marked wrong by wrong pairing.
- **Thresholds**: 92% for “correct” and 78% for “minor” keep clearly wrong words from being accepted while allowing small ASR slips.

### 3.2 Current behavior and limitations

- **Forced alignment**: CTC alignment (see `alignment/ctc_alignment.py`) provides word-level timing and confidence when Wav2Vec2 vocab supports the reference text; otherwise scoring falls back to text-only.
- **Tajweed**: Qalqalah uses intensity gradient; Ghunnah uses sustained intensity + frame confidence; Idgham uses transition smoothness; Madd/Ikhfa/Meem Sakina use heuristics and acoustic cues where available.
- **Single lock**: One verification at a time; under load, latency increases and no parallel GPU utilization.

### 3.3 Tuning

- **Similarity**: `core/scoring.py`: `SIMILARITY_CORRECT`, `SIMILARITY_MINOR`, and `_word_similarity()` (ratio vs partial_ratio).
- **Best transcript**: `main.py`: logic that compares `count_matching_words(original_text, whisper_text)` vs `wav2vec_text` and picks the better one.
- **ASR model**: Env `ASR_MODEL` (e.g. `rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v2`) to use a Quran-finetuned model.

---

## 4. GPU / CPU and model sizes

### 4.1 Models loaded at startup

| Model | Library | Typical size (approx) | Device | Purpose |
|-------|---------|------------------------|--------|---------|
| **Whisper base** | `openai-whisper` | ~140 MB | CPU or GPU (if available) | Primary transcript; word boundaries. |
| **Wav2Vec2-XLS-R 53 Arabic** | `transformers` | ~1.2 GB | `cuda` if available else `cpu` | Second transcript; frame-level features for Tajweed. |
| **Tajweed engine** | — | 0 (rules only) | — | Combines Wav2Vec2 outputs + librosa features. |

- **Whisper**: `whisper.load_model("base")` — runs on CPU with FP32 if no GPU; can use GPU if PyTorch sees CUDA.
- **Wav2Vec2**: `PhoneticAnalyzer` uses `torch.device("cuda" if torch.cuda.is_available() else "cpu")`; model and processor are loaded from Hugging Face (e.g. `jonatasgrosman/wav2vec2-large-xlsr-53-arabic`).

### 4.2 Memory (rough)

- **CPU-only**: Expect ~2–3 GB RAM for both models and runtime (Python, FastAPI, librosa, etc.).
- **GPU**: Same models on GPU need ~2–4 GB VRAM (Whisper base + Wav2Vec2-large). A 4 GB card is typically enough; 6 GB+ is comfortable.

### 4.3 Speed (typical, single request)

- **Whisper base (CPU)**: ~5–30 s for a short verse (depends on CPU and audio length).
- **Wav2Vec2 (CPU)**: ~2–10 s per call (transcribe + analyze_alignment + get_phonetic_features).
- **Total /verify**: Often 10–45 s per request on CPU; significantly faster on a decent GPU.

### 4.4 Changing model size

- **Whisper**: In `main.py`, replace `whisper.load_model("base")` with `"small"`, `"medium"`, or `"large-v3"` for better accuracy and higher memory use.
- **Wav2Vec2**: Change `ASR_MODEL` or default in `core/phonetics.py` (e.g. Quran-finetuned or smaller XLS-R).

---

## 5. Libraries and why

| Library | Version (min) | Why used |
|---------|----------------|----------|
| **FastAPI** | 0.104+ | Async API, OpenAPI, type hints, file upload, WebSocket. |
| **uvicorn** | 0.24+ | ASGI server; runs the FastAPI app. |
| **openai-whisper** | 20231117+ | Strong multilingual ASR; good Arabic and natural word boundaries. |
| **torch** | 2.0+ | Backend for Whisper and Hugging Face Wav2Vec2. |
| **transformers** | 4.35+ | Loads Wav2Vec2ForCTC and Wav2Vec2Processor from Hugging Face. |
| **librosa** | 0.10+ | Load audio (any format), resample to 16 kHz, extract RMS and duration for Tajweed. |
| **soundfile** | 0.12+ | Preferred backend for librosa to avoid PySoundFile/audioread warnings. |
| **numpy** | 1.24+ | Used by librosa, Whisper, and Tajweed (e.g. intensity diff for Qalqalah). |
| **rapidfuzz** | 3.0+ | Fast fuzzy string matching (ratio, partial_ratio) for word similarity and alignment. |
| **python-multipart** | 0.0.6+ | Required by FastAPI for `File()` uploads. |

### 5.1 Standard library / built-in

- **difflib.SequenceMatcher**: Aligns reference and user word sequences (equal/replace/delete/insert).
- **re**: Arabic normalization (diacritics, script variants).
- **threading.Lock**: Single inference lock.
- **asyncio**: Non-blocking run via `run_in_executor` and WebSocket handling.
- **uuid**: Unique temp filenames.
- **warnings**: Suppress urllib3 and librosa warnings.

### 5.2 Optional / future

- **python-dotenv**: Not in current requirements; would allow `PORT`, `QURAN_DATA_PATH`, `ASR_MODEL`, `CORS_ORIGINS` from env.
- **CTC segmentation / aeneas**: Would enable word-level timings and stricter alignment; not used today.

---

## 6. File layout (reference)

```
quran-ai-backend/
├── main.py              # FastAPI app, 3-layer pipeline, /verify, /verses, WebSocket
├── config.py            # Env config (get_quran_path, resolve_device, PORT, models, etc.)
├── .env.example         # Example env vars
├── requirements.txt    # Dependencies (incl. python-dotenv)
├── quran.json           # Verse data (nested by surah/ayah)
├── core/
│   ├── normalization.py # normalize_arabic() — ي/ی, ك/ک, Hamza, Tatweel
│   ├── scoring.py       # score_recitation(..., alignment_words, audio_duration), segment_transcript_by_reference
│   ├── phonetics.py     # PhoneticAnalyzer (transcribe, analyze_alignment, get_phonetic_features, optional 8bit/compile)
│   └── tajweed.py       # TajweedRulesEngine (acoustic Qalqalah, Madd, Ghunnah, Idgham, etc.)
├── alignment/
│   ├── __init__.py
│   └── ctc_alignment.py  # CTC forced alignment (trellis + Viterbi), word-level timings
├── tests/
│   └── test_scoring.py  # Unit tests for scoring, alignment scoring, segment_by_reference
├── static/
│   └── index.html       # Demo UI
└── docs/
    └── MODEL_AND_SYSTEM.md  # This document
```

---

## 7. Summary table

| Topic | Summary |
|-------|---------|
| **How it works** | Two ASRs (Whisper + Wav2Vec2) → pick best match to verse → normalize → align words (optimal pairing) → score with 92% / 78% thresholds; Tajweed from Wav2Vec2 + librosa. |
| **Stability** | Single inference lock, executor for heavy work, temp file cleanup, CancelledError re-raised, Tajweed/audio errors handled so scoring still returns. |
| **Accuracy** | Driven by ASR choice, normalization, optimal word pairing, and strict similarity; no word-level timing; Tajweed mostly heuristic. |
| **GPU/CPU** | Whisper base ~140 MB; Wav2Vec2-large ~1.2 GB; both use GPU if available; CPU-only needs ~2–3 GB RAM; GPU ~2–4 GB VRAM. |
| **Libraries** | FastAPI/uvicorn (API), Whisper (ASR), transformers (Wav2Vec2), librosa/soundfile (audio), rapidfuzz (matching), torch/numpy (numerics). |

This document reflects the codebase as of the last review. For run instructions and API summary, see the main [README](../README.md).
