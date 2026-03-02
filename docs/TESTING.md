# How to Test the Quran AI Backend

## 1. Start the server

```bash
cd /Users/hamid/Desktop/KivyxProjects/quran-ai-backend
source venv/bin/activate   # or: . venv/bin/activate
python main.py
```

Server runs at **http://localhost:8001** (or `PORT` from config).

- Health: **GET http://localhost:8001/**
- Demo UI: **GET http://localhost:8001/demo**

---

## 2. Test `/verify` with a real audio file

### Option A: curl

```bash
# Replace with your audio path (e.g. a verse from 000_versebyverse)
curl -X POST "http://localhost:8001/verify?surah=1&ayah=1" \
  -H "Accept: application/json" \
  -F "audio=@/path/to/your/audio.mp3"
```

Example with a local file:

```bash
curl -X POST "http://localhost:8001/verify?surah=1&ayah=1" \
  -F "audio=@/Users/hamid/Downloads/000_versebyverse/001001.mp3"
```

### Option B: Python script (already in repo)

```bash
# Edit scripts/test_api.py if needed: set audio_path and url port (8001)
python scripts/test_api.py
```

Make sure `audio_path` in the script points to an existing MP3/WAV (e.g. `001001.mp3` for Surah 1, Ayah 1).

### What you get in the response

- `transcribed_text`, `accuracy_score`, `word_analysis` (each word has `status`, `color`: green/yellow/red/grey)
- `wer`, `cer` (normalized Arabic), `tajweed_feedback` (light: Madd, Qalqalah, Ghunnah)
- `timing_ms` (load_audio, asr, alignment, tajweed, scoring, total)

---

## 3. Test via Demo UI (browser)

1. Start the server: `python main.py`
2. Open: **http://localhost:8001/demo**
3. Choose Surah and Ayah, record or upload audio, then click verify.

The demo uses the same `/verify` API and shows word-level feedback and scores.

---

## 4. Run unit tests (pytest)

```bash
cd /Users/hamid/Desktop/KivyxProjects/quran-ai-backend
source venv/bin/activate
pytest tests/ -v
```

To run a single test file:

```bash
pytest tests/test_scoring.py -v
pytest tests/test_asr.py -v
```

---

## 5. Benchmark WER/CER (with dataset)

If you have a dataset (e.g. `dataset/training_dataset.json` with `audio` paths and `text_uthmani`):

```bash
python scripts/benchmark_wer_cer.py dataset/training_dataset.json --run-asr --limit 5 -o report.json
```

This loads models, runs ASR on each item, and prints WER/CER plus worst/best 10. Use `--limit 5` for a quick test.

---

## 6. Quick smoke test (no audio file)

Check that the server and Quran data load:

```bash
# Health + verse count
curl -s http://localhost:8001/ | python -m json.tool

# Get verse 1:1 text
curl -s "http://localhost:8001/verses/1/1" | python -m json.tool
```

---

## Troubleshooting

| Issue | What to do |
|-------|------------|
| `Connection refused` | Start server with `python main.py` and confirm port 8001. |
| `404 Ayah not found` | Ensure `quran.json` or `quran_with_audio.json` is in the project root (or set `QURAN_DATA_PATH`). |
| `Audio too short` | Use at least ~0.3 s of audio; a full verse clip is best. |
| Slow first request | Models load on startup; first `/verify` may be slower. |
| Out of memory | Use smaller Whisper (e.g. `WHISPER_MODEL=small`) or enable `WAV2VEC2_QUANTIZE_8BIT=true`. |
