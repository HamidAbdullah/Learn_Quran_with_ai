# Quran AI Backend

Recitation verification and Tajweed feedback API with a Tarteel-style web UI.

## Run the project

1. **Create a virtual environment and install dependencies**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Add Quran data**  
   Place `quran.json` (or `quran_with_audio.json`) in the project root. The app supports:
   - **Nested format** (`quran.json`): `{ "1": { "1": { "text_uthmani": "...", "translation_en": "..." }, ... } }`
   - **Flat list**: `[ { "verse_key": "1:1", "text_uthmani": "...", ... } ]`

3. **Start the server**
   ```bash
   python main.py
   ```
   Or: `uvicorn main:app --host 0.0.0.0 --port 8001`

4. **Open the demo**  
   In your browser: **http://localhost:8001/demo/**

## API

- `GET /` — Health check
- `GET /verses/{surah}/{ayah}` — Get verse text and translation
- `POST /verify?surah=1&ayah=1` — Verify recitation (multipart audio file)
- `WS /ws/verify?surah=1&ayah=1` — Live partial transcription (binary audio chunks)

## Implementation plan

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the full Tarteel-like roadmap (accuracy, real-time follow, model work).
