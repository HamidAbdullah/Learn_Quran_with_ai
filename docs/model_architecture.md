# Quran AI: Model Architecture & Verification Pipeline

Our system employs a **Multi-Layer Intelligence Pipeline** designed specifically for high-accuracy Quranic recitation verification. It is optimized for both real-time streaming (WebSocket) and full-sequence analysis.

## 🏗️ The 4-Layer Analysis Flow

### Layer 1: Speech-to-Text (ASR)
- **Engine**: OpenAI Whisper (`base` or `tiny` for speed).
- **Optimization**: We use a specialized Arabic-first decoding approach. It provides the global "ground truth" of which words were uttered.
- **Role**: Transcription and basic word-order verification.

### Layer 2: Phonetic Alignment & Scoring
- **Engine**: Wav2Vec2-Arabic (XLS-R 53 Cross-Lingual).
- **Process**: Perform **Forced Alignment** between the ASR output and the Uthmani script.
- **Accuracy**: Calculates character-level timing. This allows us to identify exactly where a letter was suppressed, skipped, or pronounced incorrectly.

### Layer 3: Advanced Normalization & Fuzzy Matching
- **Constraint**: Islamic text comparison must handle Uthmani script variations.
- **Normalization**: We strip diacritics (Harakat), normalize Alif/Ya varieties, and remove special Uthmani marks to ensure comparisons focus on the core root letters (Rasm).
- **Similarity**: Uses a combination of Levenshtein distance and semantic phoneme mapping.

### Layer 4: Acoustic Intelligence (Tajweed)
- **Engine**: Librosa + Custom Rules Engine.
- **Verification**:
    - **Intensity Flux**: Detects the "bounce" of **Qalqalah**.
    - **Duration Analysis**: Measures Harakat levels for **Madd**.
    - **Spectral Energy**: Identifies nasalization peaks for **Ghunnah**.

## 📊 Scoring & Trust Thresholds

We implement a strict confidence-based threshold system to ensure the "Teacher" is always reliable:

| Confidence Level | Score Range | UI Color | Meaning |
| :--- | :--- | :--- | :--- |
| **High** | > 95% | 🟢 Green | Perfect Pronunciation & Tajweed |
| **Medium** | 80% - 95% | 🟡 Yellow | Minor phoneme slip or acoustic uncertainty |
| **Low** | < 80% | 🔴 Red | Incorrect word or significant Tajweed error |

---

## ⚡ Production Real-Time Performance
- **Streaming**: Audio is chunked into 1-2 second windows and processed via WebSockets.
- **Latency**: Sub-second partial feedback while the user is still speaking.
- **Inference**: Optimized for CPU using dynamic quantization.
