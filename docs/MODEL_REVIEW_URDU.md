# Model Review — Aap ke system mein kya use ho raha hai aur kaise kaam karta hai

**Yeh document batata hai: kaun kaun se models use ho rahe hain, system ab kya karta hai, aur step-by-step kaise karta hai.**

---

## 1. Kaun kaun se models use ho rahe hain (What is used)

| # | Model / Component | Kya hai | Config / Source |
|---|-------------------|---------|------------------|
| 1 | **Whisper** | OpenAI ka speech-to-text model (Arabic). Audio → text. | `WHISPER_MODEL=medium` (config / env) |
| 2 | **Wav2Vec2 (Quran)** | Hugging Face pe Quran Arabic ke liye fine-tuned ASR. Audio → text + logits (frame-level). | `WAV2VEC2_MODEL=rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v2` |
| 3 | **CTC Alignment** | Koi alag model nahi — Wav2Vec2 ke **logits** use hote hain. Reference text + logits → Viterbi → har word ka start/end time. | `alignment/ctc_alignment.py` |
| 4 | **Tajweed Engine** | ML model nahi — **rule-based + acoustic**. Librosa se RMS, spectral rolloff; har CTC word segment pe Madd, Qalqalah, Ghunnah check. | `core/tajweed.py` |
| 5 | **Scoring / Matching** | Koi model nahi — **normalized Arabic** + **WER/CER** (edit distance) + **rapidfuzz** word similarity. | `core/scoring.py`, `core/metrics.py` |
| 6 | **Normalization** | Simple Python (regex) — Arabic script normalize (ي ك ا ة etc.), diacritics strip, Lam-Alif. | `core/normalization.py` |

**Summary:**  
- **2 ASR models:** Whisper (medium), Wav2Vec2 Quran.  
- **1 alignment step:** CTC (Wav2Vec2 logits + reference text).  
- **Tajweed:** Light rules (Madd, Qalqalah, Ghunnah) — acoustic features, koi extra ML model nahi.  
- **Scoring:** Formula + WER/CER + word similarity — koi naya model nahi.

---

## 2. Aap ka model ab kya karta hai (What your system does now)

1. **User audio leta hai** — verse (surah + ayah) ke mutabiq.
2. **Dual ASR chalata hai** — Whisper + Wav2Vec2 dono se transcript nikalta hai.
3. **Best transcript choose karta hai** — WER kam → phir CER kam → phir confidence zyada.
4. **Word-level timing nikalta hai** — CTC alignment se har reference word ko start_time, end_time, confidence milti hai.
5. **Light Tajweed check karta hai** — har word segment pe Madd (duration), Qalqalah (burst), Ghunnah (nasal).
6. **Word-by-word color deta hai** — Green (≥92%), Yellow (78–92%), Red (<78%), Grey (skip).
7. **WER, CER, timing_ms, tajweed_feedback** response mein deta hai.

**Kya nahi karta:**  
- Naya phoneme-from-audio model nahi chalata.  
- Diacritization pipeline mein optional hai (CAMeL/AraT5), abhi verify flow mein use nahi.  
- Research-level heavy Tajweed (Idgham, Ikhfa, Meem Sakina, etc.) ab light mode mein nahi chal rahe.

---

## 3. Kaise karta hai — step by step (How it works)

### Step 1: Audio load

- **Kya:** `librosa.load(temp_file, sr=16000)` — 16 kHz mono.
- **Kyun:** Wav2Vec2 aur alignment 16 kHz expect karte hain.
- **Agar audio < 0.3 s:** "Audio too short" return; aage kuch nahi.

---

### Step 2: Dual ASR (Layer 1)

**Whisper:**

- **Kya:** `whisper_model.transcribe(audio_path, language="ar", beam_size=5)`.
- **Output:** Text + `no_speech_prob` (confidence ke liye).
- **Source:** OpenAI Whisper **medium** (config se).

**Wav2Vec2:**

- **Kya:** Audio (y, sr) → `PhoneticAnalyzer.run_forward(y, sr)` → **ek hi forward**.
- **Andar:** Processor se input → model → logits → CTC **beam search** (beam_width=20) → transcript.
- **Output:** Transcript + **logits** (alignment ke liye reuse).
- **Agar transcript mein spaces kam hon:** `segment_transcript_by_reference()` se reference ke hisaab se word boundaries lagate hain.

**Selection (best transcript):**

1. Dono transcripts pe reference ke saath **WER** aur **CER** (normalized Arabic).
2. **Pehle:** jis ki WER **kam** hai wo choose.
3. **Agar WER equal:** jis ki CER **kam** hai.
4. **Agar dono equal:** jiska **confidence zyada** hai (Whisper: 1 - no_speech_prob; Wav2Vec2: frame confidence avg).

---

### Step 3: CTC Alignment (Layer 2)

- **Input:** Reference verse (normalized), reference words, aur Wav2Vec2 ke **logits** (pehle step se — dobara forward nahi).
- **Kaam:** Logits → log-probs → **trellis** → **Viterbi path** → har token (character/space) ko frames map → **word boundaries** (start_frame, end_frame) → time (20 ms per frame).
- **Output:** `alignment_words`: har word ke liye `start_time`, `end_time`, `confidence`, `phonetic_similarity`.
- **Agar emission na ho:** Audio dobara load karke Wav2Vec2 forward (fallback).

---

### Step 4: Tajweed Light (Layer 3)

- **Input:** `alignment_words`, audio (y, sr), optional `frames_per_word` (Wav2Vec2 frame confidence per word).
- **Kaam:** Har word ke liye us segment ka audio slice → **librosa** se RMS (intensity), spectral_rolloff → sirf **3 rules**:
  - **Madd:** Segment duration (chhota segment → thoda kam score, normal → 0.92).
  - **Qalqalah:** Intensity gradient + peak ratio (bounce jaisa burst).
  - **Ghunnah:** Intensity + nasal band (sustained, frame confidence).
- **Output:** `tajweed_feedback`: list of `{ "rule": "Madd"|"Qalqalah"|"Ghunnah", "score", "feedback" }`.
- **Agar alignment na ho:** Pura audio ek segment maan ke same 3 rules chalate hain.

---

### Step 5: Scoring aur word color

- **Input:** Reference text, selected transcript, (optional) alignment_words, audio_duration, tajweed_feedback.
- **Word alignment:** Reference words vs user words — **SequenceMatcher** + **rapidfuzz** similarity.
- **Har word ka status:**  
  - **correct** → similarity ≥ 92%  
  - **minor_mistake** → 78–92%  
  - **wrong** → < 78%  
  - **missing** → user ne bola hi nahi  
- **Color:** correct → **green**, minor_mistake → **yellow**, wrong → **red**, missing → **grey**.
- **accuracy_score:** (sahi words / total reference words) × 100.
- **WER / CER:** `core/metrics` — normalized Arabic, edit distance (word-level WER, char-level CER).

---

### Step 6: Response

- **Jo return hota hai:**  
  `word_analysis` (har word: word, status, **color**, confidence, feedback),  
  `accuracy_score`, `wer`, `cer`, `timing_ms`, `tajweed_feedback`,  
  `transcribed_text`, `teacher_feedback_text`, `timing_data`, `asr_result` (bina logits ke).

---

## 4. Flow diagram (short)

```
Audio (MP3/WAV)
    → Load 16 kHz
    → Whisper (medium)     → transcript_whisper
    → Wav2Vec2 (Quran)     → transcript_wav2vec + logits
    → Select best (WER ↓, CER ↓, confidence ↑)
    → CTC alignment (logits + reference) → alignment_words
    → Tajweed light (per word segment)   → tajweed_feedback
    → Score (word match + color)         → word_analysis, accuracy_score, wer, cer
    → Return JSON
```

---

## 5. Config summary (jo models/behavior control karta hai)

| Env / Config | Default | Kaam |
|--------------|--------|------|
| `WHISPER_MODEL` | medium | Whisper size (small/medium/large-v3). |
| `WAV2VEC2_MODEL` | rabah2026/...-quran-v2 | Quran Wav2Vec2 (must Quran-finetuned). |
| `BEAM_WIDTH` | 20 | Wav2Vec2 CTC beam (zyada = slow, thoda zyada accurate). |
| `WHISPER_BEAM_SIZE` | 5 | Whisper decode beam. |
| `DEVICE` | auto | cuda / cpu. |
| `WAV2VEC2_QUANTIZE_8BIT` | false | 8-bit quantize (kam memory, thoda accuracy trade-off). |

---

**Short:**  
- **Models:** Whisper medium + Wav2Vec2 Quran (2 ASR).  
- **Alignment:** CTC (Wav2Vec2 logits, koi alag model nahi).  
- **Tajweed:** Light rules (Madd, Qalqalah, Ghunnah) — acoustic only.  
- **Scoring:** WER, CER, normalized Arabic, 92%/78% + color.  
- **Flow:** Audio → Dual ASR → best transcript → CTC alignment → Tajweed light → scoring → word_analysis + wer + cer + timing_ms + tajweed_feedback.
