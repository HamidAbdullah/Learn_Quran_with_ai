# Quran AI Backend — Deep Technical Review & Research Roadmap

**Reviewer lens:** Senior ML/AI Engineer (Speech AI, NLP, Tajweed).  
**Scope:** Architecture, ML pipeline, performance, scalability, accuracy, and path to Tarteel/Seerat-level systems.

---

## 1. Current System Analysis

### 1.1 Architecture Overview

The system implements a **3-layer verification pipeline**:

| Layer | Implementation | Role |
|-------|----------------|------|
| **Layer 1 — ASR** | Whisper (base) + Wav2Vec2-XLSR-53-Arabic | Dual ASR; best transcript by `count_matching_words` vs reference; optional `segment_transcript_by_reference` for CTC output without spaces |
| **Layer 2 — Alignment** | Custom CTC forced alignment (Viterbi on Wav2Vec2 emissions) | Word-level `start_time`, `end_time`, `confidence`, `phonetic_similarity`; single forward pass reused for transcript + alignment |
| **Layer 3 — Tajweed** | `TajweedRulesEngine` + Librosa RMS/spectral_rolloff + Wav2Vec2 frame confidence | Qalqalah (intensity gradient), Madd (duration heuristic), Ghunnah (sustained intensity + frame conf), Idgham (transition smoothness), Ikhfa, Meem Sakina |

**Scoring:** Per-word combined score when alignment exists: **0.6×text + 0.3×phonetic + 0.1×timing**. Thresholds: ≥92% → correct; 78–92% → minor_mistake; <78% → wrong. Final rollup: `combined_score = 0.7×accuracy_score + 0.3×tajweed_score`.

**Concurrency:** Single global `inference_lock`; heavy work offloaded via `run_in_executor(None, _run_with_lock)` so the event loop is not blocked.

**Data flow:** Quran loaded from `quran.json` / `quran_with_audio.json`; `POST /verify` and `GET /verses/{surah}/{ayah}` use the same verse map. Temp files are UUID-named and removed in `finally`.

### 1.2 Implementation Level Assessment

| Dimension | Level | Evidence |
|-----------|--------|----------|
| **ASR** | **Intermediate** | Dual ASR (Whisper + Wav2Vec2), best-transcript selection, Arabic normalization; no streaming ASR, no diarization, no domain-specific fine-tuning in codebase |
| **Alignment** | **Intermediate** | CTC Viterbi alignment with word boundaries; no phoneme-level or subword alignment; no external aligner (e.g. MFA, aeneas) |
| **Tajweed** | **Basic–Intermediate** | Rule-based acoustic heuristics (intensity, duration, spectral rolloff) + Wav2Vec2 frame confidence; no learned Tajweed classifier; rules not tied to specific word/segment |
| **Scoring** | **Intermediate** | Weighted text + phonetic + timing; fuzzy matching (RapidFuzz); clear status taxonomy (correct / wrong / minor_mistake / missing / extra) |
| **Backend** | **Intermediate** | FastAPI, async + executor, WebSocket partial results; single-thread inference, no queue, no model serving abstraction |
| **Dataset/Training** | **Basic** | Manifest generation (`generate_training_dataset.py`, `training_dataset.json`) for future fine-tuning; no training loop, no evaluation metrics in repo |

**Overall:** **Intermediate** — Solid production-oriented pipeline with dual ASR and CTC alignment, but Tajweed is largely heuristic and there is no learned Tajweed model, no phoneme-level error localization, and no real-time word-level streaming feedback.

---

## 2. Weaknesses

### 2.1 Audio Processing

- **Noise reduction:** Raw `librosa.load(..., sr=16000)` only. No spectral gating, no Wiener filtering, no deep learning denoising (e.g. Demucs, FullSubNet). Noisy environments will hurt ASR and Tajweed.
- **Voice activity detection (VAD):** No explicit VAD. Silence at start/end or long pauses are not trimmed; alignment and scoring assume contiguous speech. WebSocket uses a byte threshold, not speech boundaries.
- **Preprocessing:** No AGC, no high-pass filter for plosives, no emphasis on formant region for Arabic. Single resample to 16 kHz is correct for Wav2Vec2 but not tuned for Tajweed-specific bands.
- **Feature extraction:** Only RMS and spectral rolloff at default hop. No MFCCs, no mel spectrograms, no pitch (F0) track, no formant-based features. Tajweed rules (e.g. Ghunnah nasal band) use a coarse intensity window, not a proper nasal formant detector.

### 2.2 Speech Recognition

- **No streaming ASR:** WebSocket runs full Whisper on accumulated buffer; no chunked streaming decoder (e.g. Whisper streaming, or streaming Wav2Vec2 with limited context).
- **Single best transcript:** No ROVER/consensus; no confidence-weighted combination of Whisper and Wav2Vec2.
- **Reference-based segmentation:** Greedy segmentation by reference words is good for CTC, but one insertion/deletion can shift all following word boundaries.
- **No diacritic-aware ASR:** All comparison is after stripping diacritics; no Tashkeel-on path or diacritic-specific model.
- **Model choice:** Whisper base is relatively weak for Arabic; Wav2Vec2 is strong but not Quran-finetuned in default config (config mentions `rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v2` but default is `jonatasgrosman/...`).

### 2.3 Tajweed Rule Detection

- **Heuristic-only:** Qalqalah, Madd, Ghunnah, Idgham, Ikhfa, Meem Sakina use fixed thresholds (e.g. `QALQALAH_GRADIENT_THRESHOLD = 0.05`) and global intensity/rolloff. No per-rule classifier trained on labeled Tajweed data.
- **No word/segment binding:** Tajweed feedback is a flat list (e.g. “Qalqalah”, “Madd”, “Ghunnah”) not attached to specific words or time spans. UI cannot show “Madd error on word 3”.
- **No phoneme-level Tajweed:** Rules like Ikhfa (noon saakin + specific following letters) require knowing which phoneme is at which time; current pipeline does not expose phoneme boundaries.
- **Madd/Ikhfa/Meem placeholders:** Madd uses “reasonable duration” heuristic; Meem Sakina returns high default score. No actual duration model (e.g. 2–6 harakat) or articulation verification.

### 2.4 Model Evaluation Metrics

- **No standard metrics in codebase:** No WER, CER, PER computed on a held-out set. No Tajweed rule-level precision/recall/F1. Tests validate scoring logic with mock data, not model accuracy.
- **No benchmark script:** No script that runs on `fatiha_dataset.json` or a test set and reports WER/CER/alignment error.
- **Thresholds not data-driven:** 92% / 78% and 0.6/0.3/0.1 weights are fixed; no ablation or grid search documented.

### 2.5 Backend Architecture

- **Single inference lock:** Only one verification at a time; no horizontal scaling of inference, no batch processing.
- **In-process models:** Whisper and Wav2Vec2 loaded in the same process; no TorchServe, Triton, or ONNX runtime separation. Memory footprint ~2–4 GB (GPU) or ~2–3 GB (CPU).
- **No request queue:** Under load, requests block behind the lock; no priority, no timeout per stage.
- **WebSocket buffer semantics:** Partial results run on full buffer without clearing; same audio can be re-processed; no “word_events” or “current word index” in the protocol yet (per IMPLEMENTATION_PLAN).

### 2.6 Latency Optimization

- **No Faster-Whisper:** Still using `openai-whisper`; CTranslate2/Faster-Whisper would give ~4× speedup on CPU.
- **No ONNX/TensorRT:** Models run in PyTorch only; no export path for ONNX or TensorRT for production GPUs.
- **Double load of audio:** In `_run_verification_pipeline`, audio is loaded once with librosa then passed to phonetics; alignment can reuse emission—good. But Whisper transcribes from file again (separate load inside `whisper_model.transcribe`).
- **No caching:** Same verse transcribed multiple times is not cached (e.g. for reference audio or repeated users).

---

## 3. Advanced Architecture Recommendation

### 3.1 Tajweed Rule Classification (Research-Level)

- **Per-rule classifiers:** For each major rule (Qalqalah, Madd, Ghunnah, Idgham, Ikhfa, Idgham Shafawi, etc.), train a small classifier on top of:
  - **Acoustic:** Mel spectrograms (e.g. 80 mel bins, 10 ms hop), or wav2vec2/hubert frame embeddings.
  - **Phonetic:** CTC alignments + character/phoneme labels (from Wav2Vec2 vocab or a phoneme set).
  - **Context:** Reference text and word boundaries so the model knows “this segment should be Qalqalah”.
- **Multi-task head:** One encoder (e.g. Wav2Vec2 frozen or fine-tuned), multiple heads: one for ASR, one for Tajweed rule presence/quality per segment.
- **Labeled data:** Curate or crowdsource “correct vs incorrect” examples per rule (with timestamps) to train and validate.

### 3.2 Phoneme-Level Alignment and Error Localization

- **Forced alignment:** Keep CTC alignment for word-level; add **phoneme-level** alignment via:
  - **Montreal Forced Aligner (MFA)** with an Arabic G2P and acoustic model, or
  - **Phonemizer + CTC** with a phoneme vocabulary (e.g. IPA or Arabic-specific) so each phoneme has start/end.
- **Error localization:** Compare reference phoneme sequence (from reference text + G2P) to aligned user phoneme sequence; report substitutions, insertions, deletions at phoneme level, then map back to word and character for UI (e.g. “wrong letter in word 2”).
- **Use of Wav2Vec2:** Wav2Vec2 is character/subword; for true phoneme alignment you can either (a) map CTC tokens to a phoneme inventory via a lexicon, or (b) use a phoneme-based ASR (e.g. trained with phoneme targets) for alignment only.

### 3.3 Suggested Tech Stack for Alignment and Tajweed

| Component | Suggestion |
|-----------|------------|
| **Forced alignment** | CTC (current) + MFA or **aeneas** for alternative; or **ctc-segmentation** with Wav2Vec2 for long-form. |
| **Speech recognition tokenization** | Keep character-level for Arabic script; add optional **phoneme vocabulary** (e.g. CMU Arabic or custom) for Tajweed-aware training. |
| **Phonetic embeddings** | **Wav2Vec2/HuBERT** frame embeddings as input to Tajweed heads; optional **XLS-R** or **MMS** for multilingual robustness. |
| **Tajweed-specific model** | Fine-tune Wav2Vec2 on Quran + Tajweed labels (segment-level or frame-level); or add a **Tajweed adapter** (small MLP/Transformer) on top of frozen encoder. |

---

## 4. Best Models To Use

### 4.1 ASR

| Model | Use case | Notes |
|-------|----------|--------|
| **jonatasgrosman/wav2vec2-large-xlsr-53-arabic** | Current default | Strong Arabic; no spaces in output; good for alignment. |
| **rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v2** | Quran recitation | Fine-tuned on Quran; better for verse-length audio; use via `WAV2VEC2_MODEL`. |
| **Whisper base/small/medium** | Robustness, word boundaries | Prefer **faster-whisper** (CTranslate2) for production; consider **large-v3** for best Arabic. |
| **OpenAI Whisper large-v3** | Highest accuracy | Heavy; use with GPU; good for offline batch. |
| **SpeechT5 / MMS (Massively Multilingual)** | Alternative | MMS supports Arabic; can be used for ASR or representation. |

**Recommendation:** Primary path: **Wav2Vec2 Quran-finetuned** for transcript + alignment; optional **Faster-Whisper** for second transcript and streaming. Fine-tune Wav2Vec2 further on your own verse-level data if you have it.

### 4.2 Fine-Tuning Strategies

- **Whisper:** Use HuggingFace `datasets` + `transformers` (or official Whisper fine-tuning); train on `(audio, text)` with normalized or Uthmani text; low learning rate, freeze encoder for first phase.
- **Wav2Vec2:** Standard CTC fine-tuning with your `training_dataset.json` / manifest; add word-level or segment-level Tajweed labels as auxiliary targets (multi-task) if you have labels.
- **Adapter/LoRA:** Add small adapters on top of frozen Wav2Vec2 for Tajweed-only; keeps ASR stable and reduces data need for Tajweed.

### 4.3 Transfer Learning

- Start from **XLS-R 53** or **Quran Wav2Vec2**; fine-tune on your verses (with or without diacritics).
- Use **Arabic G2P** (e.g. from MFA or phonemizer) to produce phoneme sequences for reference; align user audio to reference phonemes for Tajweed-aware alignment.
- For Tajweed: pre-train a small model on “correct vs incorrect” clips per rule (from existing apps or experts), then use as a teacher or direct classifier.

---

## 5. Feature Roadmap (Beginner → Research Level)

### Must-Have (Tarteel/Seerat Parity)

| Feature | Current | Target | Effort |
|---------|---------|--------|--------|
| **Real-time recitation correction** | Partial (Whisper on buffer; no word_events) | Emit `word_recited` with index + status; clear buffer/chunk semantics | Medium |
| **Word-level mistake highlighting** | Yes (word_analysis with status) | Keep; add `start_time`/`end_time` consistently from alignment | Low |
| **Tajweed rule explanation generation** | Static feedback strings | Per-rule explanations (short text or links); optional link to verse/word | Medium |
| **Adaptive learning per user** | None | Store per-user history; highlight recurring mistakes; optional difficulty/suggestions | High |

### Research-Level

| Feature | Description |
|---------|-------------|
| **Self-supervised learning** | Use unlabeled Quran recitations (e.g. from apps) to pre-train or contrastively learn representations; then fine-tune with limited labeled Tajweed data. |
| **Multimodal Quran learning AI** | Combine audio + text (verse) + optional video (mouth/lip) for articulation feedback; use vision for hijab/articulation hints. |
| **Generative feedback explanation** | LLM (e.g. small Arabic-capable model) that takes ASR output + Tajweed flags + verse and generates natural-language feedback (“You pronounced the Qalqalah on ب too softly; try a clearer bounce.”). |

---

## 6. Evaluation Metrics

### Recommended Metrics

| Metric | Definition | Use |
|--------|------------|-----|
| **WER** | Word Error Rate (ref vs ASR) | Overall ASR quality on test set |
| **CER** | Character Error Rate | Fine-grained text accuracy |
| **PER** | Phoneme Error Rate | If you have phoneme alignment; best for Tajweed relevance |
| **Tajweed rule accuracy** | Per-rule TP/FP/FN; F1 or accuracy | For each rule (Qalqalah, Madd, etc.) on labeled segments |
| **Alignment quality** | Boundary error (start/end vs human labels), or R-value | Validate CTC (or MFA) timings |
| **User experience** | Task success, perceived usefulness, latency (e.g. P95) | Surveys + A/B; track “retry same verse” and completion rate |

### Implementation Sketch

- **Benchmark script:** Input: list of `(audio_path, reference_text)` (e.g. from `fatiha_dataset.json`). Run pipeline; compute WER/CER vs reference; optionally compare alignment to forced-alignment gold if available.
- **Tajweed eval:** Require labeled data (e.g. “this segment has correct/incorrect Qalqalah”); run Tajweed engine and compute precision/recall per rule.
- **Threshold tuning:** Grid search over SIMILARITY_CORRECT, SIMILARITY_MINOR and weight (0.6/0.3/0.1) on a dev set to maximize agreement with human judgment or WER.

---

## 7. 6-Month Development Roadmap

### Month 1: Foundation and Metrics

- **Week 1–2:** Add benchmark script: WER/CER on a fixed test set (e.g. Fatiha + 50 verses); document in README.
- **Week 2–3:** Introduce audio preprocessing: VAD (e.g. webrtc-vad or silero-vad), optional light noise reduction (e.g. noisereduce or spectral gating).
- **Week 3–4:** Switch to **Faster-Whisper**; measure latency and accuracy vs current Whisper; tune chunk size for WebSocket.

### Month 2: Alignment and Real-Time Protocol

- **Week 1–2:** Emit **word_events** from WebSocket: after each chunk (or sliding window), run alignment and send `{ type: "word_recited", index, word, status }`; fix buffer so each chunk is processed once.
- **Week 2–3:** Optional: integrate **MFA** or **aeneas** for an alternative alignment path; compare word boundaries with CTC.
- **Week 3–4:** Bind Tajweed feedback to **word index** or time range; API returns `word_analysis[i].tajweed_rules[]` with rule name + score + feedback.

### Month 3: Tajweed Data and Models

- **Week 1–2:** Curate or crowdsource 100–500 labeled segments (correct/incorrect per rule) for at least 2 rules (e.g. Qalqalah, Ghunnah).
- **Week 2–4:** Train a small Tajweed classifier (e.g. MLP on Wav2Vec2 frame means per segment); evaluate F1; plug into pipeline as optional layer.

### Month 4: Production and Scale

- **Week 1–2:** Export Wav2Vec2 (and optionally Whisper) to **ONNX**; run benchmark; optional TensorRT for GPU.
- **Week 2–3:** Separate **model server** (e.g. TorchServe or Triton) or at least a worker process with a queue (Redis + Celery or in-process queue) so API stays responsive.
- **Week 3–4:** Add **caching** for repeated same-verse verification (e.g. hash of audio + verse_key); tune TTL.

### Month 5: User Experience and Product

- **Week 1–2:** **Tajweed explanations:** Map each rule to 1–2 sentence explanation (or fetch from a small DB); return in API and show in UI.
- **Week 2–3:** **Adaptive learning:** Persist per-user (or per-session) results; endpoint or response field “frequent mistakes” / “suggest next verse”.
- **Week 3–4:** **Tashkeel mode:** Optional `tashkeel: true`; compare with diacritics where applicable; document in API.

### Month 6: Research and Polish

- **Week 1–2:** Experiment with **phoneme-level** alignment (MFA + Arabic G2P or phoneme CTC); report PER on a subset.
- **Week 2–3:** **Generative feedback:** Prototype with an LLM (local or API) for natural-language feedback from structured result.
- **Week 3–4:** Paper-style write-up: system architecture, metrics, ablation (dual ASR, alignment, Tajweed); open-source checklist and deployment guide.

### Research Papers to Study

- **ASR and alignment:** “wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations” (Baevski et al.); “CTC-Segmentation of Large Corpora for German End-to-End Speech Recognition” (CTC segmentation).
- **Arabic/Quran:** “Fine-Tuning wav2vec2 for Arabic Dialect Identification” and any Quran-specific ASR papers (e.g. rabah2026’s work).
- **Forced alignment:** “Montreal Forced Aligner” (MFA); “Aeneas” for text-audio sync.
- **Tajweed:** Search for “Tajweed detection”, “Quran recitation assessment”, “Arabic pronunciation assessment” in IEEE/ACM/Springer.

### Tools and Frameworks to Learn

- **Faster-Whisper** (CTranslate2), **ONNX Runtime**, **TorchServe** or **Triton**.
- **Montreal Forced Aligner**, **Phonemizer** (Arabic G2P).
- **webrtc-vad** or **Silero VAD**; **noisereduce** or **demucs** for denoising.
- **Redis + Celery** (or **RQ**) for task queue if you move to async jobs.

---

## 8. Code Review Summary

### What’s Solid

- **Clear 3-layer pipeline** in `main.py` with a single entry point `_run_verification_pipeline`; no pipeline logic in route handlers.
- **Reuse of Wav2Vec2:** Single forward pass; logits reused for transcript, alignment, and Tajweed frame confidence—avoids redundant loads and forward passes.
- **Scoring logic:** `_align_words` with optimal pairing in “replace” blocks; consistent use of normalization; clear status taxonomy.
- **CTC alignment:** Self-contained module; trellis + Viterbi + word boundaries; supports precomputed emission.
- **Config:** Env-based config in `config.py`; no hardcoded secrets.
- **Tests:** `test_scoring.py` covers alignment scoring, segment_by_reference, count_matching_words, edge cases.

### Refactoring Suggestions

- **Extract audio loading once:** In `_run_verification_pipeline`, load audio once (e.g. with librosa); pass `(y, sr)` into a helper that runs Whisper from a buffer or temp file written from `y` (to avoid Whisper’s internal reload). Alternatively, use Whisper’s API with waveform if supported.
- **Split pipeline into stages:** E.g. `asr_stage(audio) -> transcript, logits`; `alignment_stage(logits, reference) -> alignment_words`; `tajweed_stage(audio, logits, reference) -> tajweed_feedback`; `scoring_stage(...)`. Improves testability and allows swapping implementations (e.g. different ASR).
- **Tajweed engine:** Inject reference text and word boundaries into `get_teacher_feedback` so each rule can be computed per segment; return structure like `[{ "rule": "Qalqalah", "word_index": 0, "score": 0.9, "feedback": "..." }]`.
- **Memory:** Ensure large tensors (logits, emission) are moved to CPU or deleted after use when running on GPU to avoid OOM under repeated requests.
- **API latency:** Return as soon as scoring is done; if you add Tajweed explanations from a DB or LLM, consider returning basic result first and enriching asynchronously (e.g. WebSocket or polling).

### Performance and Latency

- **Biggest wins:** (1) Faster-Whisper, (2) ONNX/TensorRT for Wav2Vec2, (3) single audio load and reuse for Whisper.
- **Concurrency:** Keep lock for correctness; for scale, move inference to a separate worker pool or service and have API post jobs and poll or stream results.
- **WebSocket:** Sending `word_events` with index and status will improve perceived real-time behavior more than raw partial transcript alone.

---

## 9. Final Expert Rating of Project Level

| Criterion | Score (1–5) | Note |
|-----------|-------------|------|
| **Architecture clarity** | 4 | Clear layers; single lock is a known trade-off. |
| **ASR and alignment** | 4 | Dual ASR + CTC alignment is strong; no streaming, no phoneme-level yet. |
| **Tajweed** | 2 | Heuristic-only; not learned; not word-bound. |
| **Scoring and UX logic** | 4 | Good thresholds and word-level feedback. |
| **Backend and ops** | 3 | Solid FastAPI and async; no scale-out or model server. |
| **Evaluation and data** | 2 | No WER/CER/PER or Tajweed metrics in repo; training data prep only. |
| **Documentation** | 4 | MODEL_AND_SYSTEM, AI_ARCHITECTURE, IMPLEMENTATION_PLAN are detailed. |

**Overall:** **Intermediate (3.5/5)** — Production-capable foundation with strong ASR and alignment and clear scoring. To reach **advanced/research level**, focus on: (1) learned Tajweed models tied to segments, (2) phoneme-level alignment and error localization, (3) standard evaluation metrics and benchmark suite, (4) scalable inference (queue, ONNX/TensorRT, optional model server), and (5) real-time word_events and adaptive/Tajweed-explanation features.

---

*Document generated as a one-time technical review. Use with IMPLEMENTATION_PLAN.md and MODEL_AND_SYSTEM.md for ongoing development.*
