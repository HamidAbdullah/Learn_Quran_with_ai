# Implementation Plan: Tarteel-like Quran AI

Goal: Build a **high-accuracy**, **Tarteel-style** recitation app with real-time follow-along, clear mistake detection, optional Tashkeel, and a **proper, premium web UI**.

---

## Part 1: Target Features (Tarteel Parity)

| Feature | Tarteel | Our Target | Priority |
|--------|---------|------------|----------|
| Real-time recitation follow | Word-by-word highlight as you recite | Same | P0 |
| Mistake detection | Missed / Wrong / Skipped, color-coded | Same + clear legend | P0 |
| Tashkeel (diacritics) mode | On/off recognition | On/off matching & display | P1 |
| High accuracy | ~98% claim, multiple accents | Improve ASR + alignment to >95% | P0 |
| Memorization mode | Hide words until recited | Optional “reveal as you go” | P1 |
| Multiple scripts | Uthmani, IndoPak, Adaptive | Uthmani first, then extend | P2 |
| Translations | Multiple EN translations | One EN translation per verse | P1 |
| Progress / session | Sessions, delete mistake | Save session, per-verse history | P2 |
| Premium web design | Clean, modern, RTL | Polished UI, responsive, accessible | P0 |

---

## Part 2: Phased Implementation

### Phase 1: Foundation (Accuracy + Data + API)

**1.1 Quran data layer**

- [x] **Single source of truth**  
  - Support both list format (`quran_with_audio.json`) and nested format (`quran.json`) in `main.py` (`_load_quran()`).  
  - Normalizes to flat `verse_key -> { text_uthmani, translation_en?, ... }`.  
  - Document required JSON shape in README.

- [ ] **Full Quran text**  
  - Ensure all 6,236 verses (or your target subset) are loadable; use an existing dataset (e.g. Quran.com API or static JSON) if current file is partial.

**1.2 Accuracy: ASR & alignment**

- [ ] **Arabic-first ASR**  
  - Keep Whisper as option; add **Arabic-only** path:  
    - Option A: Whisper `medium` or `large-v3` with `language="ar"` for better accuracy.  
    - Option B: Dedicated Arabic ASR (e.g. wav2vec2 XLSR Arabic already in use — ensure it’s used for primary transcript when possible).  
  - Decide single “source of truth” transcript for scoring (e.g. Whisper for robustness, wav2vec for latency/alignment).

- [ ] **Word-level alignment**  
  - Use **forced alignment** so each word has start/end time:  
    - e.g. `ctc-segmentation` with wav2vec2 CTC, or `aeneas`/`gentle`-style aligners.  
  - Output: `[{ "word": "...", "start": 0.0, "end": 0.5, "confidence": 0.98 }, ...]`.  
  - Enables real-time “current word” highlighting and precise mistake positions.

- [ ] **Tashkeel-aware matching (optional)**  
  - When “Tashkeel on”: compare with diacritics (normalize only Tatweel and variants).  
  - When “Tashkeel off”: strip diacritics (current behavior).  
  - Add query/body param `tashkeel: bool` to `/verify` and WebSocket.

**1.3 API contract**

- [x] **REST**  
  - `POST /verify`: Keep; add optional `tashkeel: bool` (later).  
  - Response: include **word-level timing** when alignment is available (e.g. `word_analysis[].start`, `word_analysis[].end`).  
  - [x] `GET /verses/{surah}/{ayah}` for fetching verse text + translation (for UI).

- [ ] **WebSocket**  
  - `ws/verify` or `ws/recite`: Send audio chunks; stream back:  
    - `partial_transcript` (interim).  
    - `word_events`: `{ "type": "word_recited", "index": 0, "word": "...", "status": "correct" | "wrong" | "skipped" }` so the UI can highlight in real time.  
  - Fix buffer reuse: after creating `process_chunks(buffer)`, clear or replace buffer so the next chunk is fresh (avoid re-processing same audio).

- [ ] **Mistake types**  
  - Standardize: `correct` | `wrong` | `skipped` | `missing` (and optionally `minor_mistake`).  
  - Map to **Tarteel-style colors** in API response (e.g. `status` + optional `display_color`).

**1.4 Config & ops**

- [x] `requirements.txt` with core deps.  
- [ ] Env: `CORS_ORIGINS`, `QURAN_DATA_PATH`, `ASR_MODEL`, `PORT`.  
- [ ] CORS: restrict origins when not dev; avoid `allow_origins=["*"]` with `allow_credentials=True`.

---

### Phase 2: Real-time follow & mistake UX (backend + protocol)

**2.1 Streaming pipeline**

- [ ] **Chunked processing**  
  - WebSocket receives audio chunks (e.g. 1–2 s).  
  - Run ASR + alignment on each chunk (or on sliding window).  
  - Emit **word_events** as soon as a word is confidently identified (with index in verse).

- [ ] **State per session**  
  - Track “last recited word index” so:  
    - “Current word” = next expected word.  
    - Emit `word_recited` (correct/wrong/skipped) and “current” pointer for UI.

**2.2 Mistake detection logic**

- [ ] **Clear semantics**  
  - **Wrong**: user said something else for this word.  
  - **Skipped**: user moved on without saying this word.  
  - **Missing**: same as skipped (or merge).  
  - **Extra**: user said a word not in the verse (optional; show as “insertion” or ignore).

- [ ] **Stable ordering**  
  - Response and WebSocket events use the same word indices as the displayed verse (0-based or 1-based, documented).

**2.3 Memorization mode (backend)**

- [ ] **Hide-until-recited**  
  - Optional param: `memorization_mode: bool`.  
  - Response can include “revealed” word indices; UI hides unrevealed words (e.g. blank or dots).  
  - Or: stream “reveal” events as words are correctly recited.

---

### Phase 3: Web app — proper design (Tarteel-like)

**3.1 Design system**

- [x] **Tokens**  
  - Colors: primary, secondary, background, surface, text, text-muted.  
  - Mistake colors: **correct** (e.g. green), **wrong** (e.g. red), **skipped/missing** (e.g. orange/amber), **current** (e.g. subtle highlight).  
  - Typography: Arabic font (e.g. Amiri / Noto Naskh) for verse; sans for UI.  
  - Spacing, radius, shadows: consistent variables (CSS or Tailwind).

- [x] **RTL & layout**  
  - Root or verse container: `dir="rtl"`, `lang="ar"` for Arabic.  
  - Layout works in both RTL and LTR (controls, nav, settings in logical order).  
  - Responsive: mobile-first; large font for verse on desktop.

**3.2 Core pages/views**

- [ ] **Home / Surah list** (optional: list all surahs)  
  - List surahs with names (Arabic + transliteration); click → verse picker or first verse.

- [x] **Recitation screen**  
  - Verse text: one line or wrapped, **word-by-word** as clickable/highlightable spans.  
  - **Current word** clearly highlighted (e.g. underline or background).  
  - **Mistake colors** on each word (correct / wrong / skipped) after verification or in real time.  
  - Record button; optional “Listen to correct recitation” (play reference audio if you add it later).  
  - Toggle: **Tashkeel on/off** (affects both display and API).  
  - Toggle: **Memorization mode** (hide words until recited).

- [x] **Results / summary**  
  - Accuracy %, Tajweed score, list of mistakes with verse context.  
  - “Try again” for same verse.

- [ ] **Settings**  
  - Tashkeel, memorization mode, script (future), language (future).

**3.3 Real-time behavior**

- [ ] **Live transcript**  
  - Show partial transcript below verse while recording (optional).  
  - **Follow-along**: as backend sends `word_recited`, highlight that word and advance “current” to next.

- [ ] **WebSocket lifecycle**  
  - Connect when user starts recording; disconnect on stop.  
  - Show connection status (connecting / live / error).  
  - Reconnect with backoff on disconnect (you already have some of this).

**3.4 Accessibility & polish**

- [ ] **A11y**  
  - Focus states, ARIA labels for record button and controls.  
  - Sufficient contrast for mistake colors (WCAG AA).  
  - Optional: announce “correct” / “wrong” via live region for screen readers.

- [ ] **Loading & errors**  
  - Skeleton or spinner while loading verse/list.  
  - Clear error messages (e.g. “Verse not found”, “Microphone denied”, “Connection lost”).  
  - Empty states (e.g. no sessions yet).

**3.5 Tech stack (suggestion)**

- **Option A**: Single HTML/JS/CSS (like now) — improve structure: separate CSS file, minimal JS framework (or vanilla).  
- **Option B**: React/Vue/Svelte + Tailwind for components and theming.  
- **Option C**: Next.js/Nuxt for SSR + i18n if you plan multiple languages.  

Start with **Option A** if you want to ship quickly; refactor to **Option B** when the UI grows.

---

### Phase 4: Tajweed & extras

**4.1 Tajweed**

- [ ] **Real feedback**  
  - Use intensity/pitch/duration from `get_phonetic_features` and alignment to attach feedback to **specific words** (e.g. “Madd on this word too short”).  
  - Surface in UI as tooltips or a sidebar per word.

**4.2 Translations**

- [ ] **Verse API**  
  - Include `translation_en` (or similar) in verse payload.  
  - UI: toggle to show translation below Arabic.

**4.3 Progress (optional)**

- [ ] **Sessions**  
  - Save session in backend (e.g. verse_key, timestamp, accuracy, mistakes).  
  - Simple “History” or “Last sessions” in UI.  
  - Optional: goals, streaks (backend + UI).

---

## Part 3: Suggested folder structure

```
quran-ai-backend/
├── main.py                 # FastAPI app, routes, WebSocket
├── config.py               # Env load (CORS, paths, model names)
├── requirements.txt
├── .env.example
├── core/
│   ├── quran_loader.py     # Load & normalize Quran data
│   ├── normalization.py   # Arabic normalizing (with tashkeel option)
│   ├── scoring.py         # Word diff + accuracy + tajweed rollup
│   ├── alignment.py        # Forced alignment (word timings)
│   ├── asr.py             # Whisper + wav2vec wrapper, single interface
│   ├── phonetics.py
│   └── tajweed.py
├── api/
│   ├── routes_verify.py
│   ├── routes_quran.py    # GET verse(s)
│   └── ws_recite.py       # WebSocket handler
├── static/                 # Or separate frontend repo
│   ├── index.html
│   ├── css/
│   │   └── theme.css      # Design tokens + layout
│   └── js/
│       ├── app.js
│       ├── recorder.js
│       └── ws-client.js
└── IMPLEMENTATION_PLAN.md  # This file
```

---

## Part 4: Order of work (checklist)

1. **Week 1 – Foundation**  
   - [x] Quran loader (support current JSON + flat list).  
   - [x] `requirements.txt` + README run instructions.  
   - [ ] Fix CORS and WebSocket buffer.  
   - [x] Add `GET /verses/{surah}/{ayah}` for verse text + translation.

2. **Week 2 – Accuracy**  
   - [ ] Word-level alignment (CTC or aligner).  
   - [ ] Optional: upgrade Whisper size or lock to wav2vec for transcript.  
   - [ ] Tashkeel-aware normalization and API flag.

3. **Week 3 – Real-time**  
   - [ ] WebSocket: word_events (word index + status).  
   - [ ] Fix chunking/buffer so each chunk processed once.  
   - [ ] Document protocol (JSON message types).

4. **Week 4 – Web UI**  
   - [ ] Design tokens (colors, fonts, mistake colors).  
   - [ ] Recitation screen: verse with word spans, current word, mistake colors.  
   - [ ] Real-time follow from WebSocket.  
   - [ ] Tashkeel toggle; memorization toggle (if backend ready).

5. **Week 5+ – Polish**  
   - [ ] Tajweed word-level feedback.  
   - [ ] Translations in UI.  
   - [ ] Sessions/history (optional).  
   - [ ] A11y pass and error handling.

---

## Part 5: Success criteria (Tarteel-like)

- **Accuracy**: >95% word-level match on clean recitation; clear distinction of wrong vs skipped.  
- **Real-time**: Word-by-word follow with &lt; ~2 s delay.  
- **Design**: Clear mistake colors, readable Arabic, responsive, RTL-correct.  
- **Options**: Tashkeel on/off, memorization mode.  
- **Stability**: No duplicate processing, clean WebSocket lifecycle, safe CORS.

Use this plan as a living doc: tick items as you go and adjust priorities (e.g. memorization or translations earlier if needed).
