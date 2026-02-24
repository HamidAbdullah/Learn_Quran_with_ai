# Quran AI Backend — Test Cases & Implementation Guide

This document describes **what was implemented**, **how tests are structured**, and **how to run and use** them.

---

## 1. What Was Done (Summary)

### Phase 1 & 2 (Foundation + ASR)

- **Dual ASR**: Whisper + Wav2Vec2 Arabic/Quran; single entry point `run_dual_asr()` in `core/asr.py`.
- **Best-transcript selection**: First by **weighted similarity** (0.6×edit_distance + 0.3×phonetic + 0.1×confidence); fallback to match count then confidence tie-break.
- **WER/CER metrics**: `core/metrics.py` — `wer()`, `cer()`, `wer_cer()` for evaluation.
- **ASR quality score**: `core/confidence.py` — `get_asr_quality_score(transcript, reference, logits?, whisper_result?)` returns `final_score`, `token_confidence`, `edit_distance_score`, `match_score`.

### Optimization Layer (Evaluation & Production Readiness)

- **Evaluation module** (`evaluation/`):
  - **asr_metrics.py**: WER/CER distribution (mean, median, p95), worst 10 cases, failure verses, high-error verses.
  - **benchmark_runner.py**: Run benchmark on a dataset JSON, optional ASR callback, write report.
  - **dataset_validation.py**: Validate items have reference text; if audio path present, check file exists; optional corruption check via librosa.
- **Performance timing**: In `main.py`, pipeline records `timing_ms` (load_audio, asr, alignment, tajweed, scoring, total) per request.
- **Backward compatibility**: Existing API unchanged; new fields (`timing_ms`, optional `weighted_similarity` in `selection_reason`) are additive. Fallbacks if confidence or scoring fails.

---

## 2. Test Framework & How to Run

| Item | Detail |
|------|--------|
| **Framework** | Python `unittest` (standard library). |
| **Discovery** | Tests live under `tests/`; module names `test_*.py` and classes/methods named `test_*` are collected. |
| **Run all** | `python3 -m unittest discover tests -v` |
| **Run one file** | `python3 -m unittest tests.test_metrics -v` |
| **Run one class** | `python3 -m unittest tests.test_asr.TestASRSelection -v` |
| **Run one test** | `python3 -m unittest tests.test_metrics.TestWER.test_perfect_match -v` |
| **Optional** | With pytest: `pip install pytest` then `pytest tests/ -v` |

**Requirements:** Project dependencies (e.g. `rapidfuzz`, `core.normalization`, `core.metrics`, etc.). Install with `pip install -r requirements.txt`. No GPU or model download needed for unit tests (mocks are used where models would be required).

---

## 3. Test Modules & Cases (Detailed)

---

### 3.1 `tests/test_metrics.py` — WER & CER

**Purpose:** Validate that Word Error Rate and Character Error Rate are computed correctly on normalized Arabic text.

**Code under test:** `core/metrics.py` (`wer`, `cer`, `wer_cer`), `core/normalization.py` (used inside metrics).

| Test | What it does | Pass condition |
|------|----------------|----------------|
| **TestWER.test_perfect_match** | `wer(ref, hyp)` when ref and hyp are identical. | WER = 0.0 |
| **TestWER.test_normalized_match** | Reference has diacritics (Uthmani), hypothesis normalized; same base text. | WER = 0.0 (normalization makes them match) |
| **TestWER.test_one_word_wrong** | One word substituted (رب → العالم). | WER > 0 and ≤ 1.0 |
| **TestWER.test_one_word_deletion** | Hypothesis missing one word. | WER > 0 and ≤ 1.0 |
| **TestWER.test_empty_reference** | Reference empty. | WER = 0.0 (no reference words to err on) |
| **TestWER.test_empty_hypothesis** | Hypothesis empty, reference non-empty. | WER > 0 and ≤ 1.0 |
| **TestCER.test_perfect_match** | Identical character sequences. | CER = 0.0 |
| **TestCER.test_one_char_diff** | One character different (الله vs اللاه); ref chosen so normalization doesn’t hide the error. | CER > 0 and < 1.0 |
| **TestCER.test_spaces_ignored** | Same text with spaces; CER with `remove_spaces=True`. | CER = 0.0 |
| **TestWERCER.test_wer_cer_both** | `wer_cer(ref, hyp)` for perfect match. | Both WER and CER 0.0 |
| **TestWERCER.test_wer_cer_mismatch** | Mismatch (الحمد لله vs الحمد كله). | Both WER and CER > 0 |

**Using:** Run after any change to `core/metrics.py` or normalization used by metrics. No audio or models.

---

### 3.2 `tests/test_confidence.py` — ASR Quality Score

**Purpose:** Ensure `get_asr_quality_score()` returns the expected structure and values (edit-distance score, match score, final score).

**Code under test:** `core/confidence.py` (`get_asr_quality_score`, internal `_match_score`), `core/metrics.wer`, `core/scoring.count_matching_words` (via `_match_score`).

| Test | What it does | Pass condition |
|------|----------------|----------------|
| **test_perfect_match** | ref = hyp = "بسم الله الرحمن الرحيم", no logits/whisper. | `edit_distance_score` = 1.0, `match_score` = 1.0, `final_score` ≥ 0.9 |
| **test_no_logits_uses_edit_distance** | Mismatch (الحمد لله vs الحمد كله), no logits. | `edit_distance_score` < 1, all scores in [0, 1] |
| **test_empty_hypothesis** | hypothesis = "". | `edit_distance_score` = 0.0, `match_score` = 0.0 |
| **test_keys_present** | Any ref/hyp; check response shape. | Keys `final_score`, `token_confidence`, `edit_distance_score`, `match_score` present and numeric |

**Using:** Run after changes to `core/confidence.py` or to WER/count_matching_words behavior.

---

### 3.3 `tests/test_asr.py` — Dual ASR Selection

**Purpose:** Check that when both Whisper and Wav2Vec2 are (mock) used, the pipeline selects the correct transcript and returns the expected keys. No real models; mocks simulate transcript and confidence.

**Code under test:** `core/asr.py` (`run_dual_asr`, `_weighted_similarity`), and indirectly `core/confidence.get_asr_quality_score` when weighted path is used.

**Mocks:**

- `_make_whisper_model(transcribe_return)`: returns an object whose `transcribe()` returns the given dict (e.g. `{"text": "...", "no_speech_prob": 0.0}`).
- `_make_phonetic_analyzer(transcript, logits, confidence)`: returns an object whose `run_forward()` returns `(transcript, logits)`, and `analyze_alignment_from_logits` / `analyze_alignment` return a dict with `confidence_avg`.

| Test | What it does | Pass condition |
|------|----------------|----------------|
| **test_whisper_only** | Wav2Vec2 returns nothing (no transcript); Whisper returns full verse. | `selected_source` = "whisper", `selected_transcript` = full verse |
| **test_wav2vec_only** | Whisper returns empty (high no_speech_prob); Wav2Vec2 returns full verse. | `selected_source` = "wav2vec", transcript contains "الرحمن" |
| **test_selection_by_match_count** | Whisper misses one word; Wav2Vec2 has all four. | `selected_source` = "wav2vec"; `selection_reason` in ("match_count", "weighted_similarity") |
| **test_selection_by_confidence_on_tie** | Both return same text; Wav2Vec2 confidence 0.95, Whisper 0.8. | `selected_source` = "wav2vec"; `selection_reason` in ("confidence_tie", "weighted_similarity") |
| **test_asr_result_keys** | Both return text; check output shape. | All of `selected_transcript`, `selected_source`, `selection_reason`, `transcript_whisper`, `transcript_wav2vec`, `confidence_*`, `match_count_*` present in result |

**Using:** Run after changes to selection logic, weighted similarity, or ASR result structure. Explains “what” and “how” of selection without loading real models.

---

### 3.4 `tests/test_dataset_validation.py` — Dataset & Evaluation Reports

**Purpose:** Validate dataset validation (required fields, missing files) and evaluation metrics (WER/CER distribution, worst cases).

**Code under test:** `evaluation/dataset_validation.py` (`validate_dataset`, `ValidationReport`), `evaluation/asr_metrics.py` (`compute_wer_cer_distribution`, `get_worst_cases`).

#### 3.4.1 Dataset validation

| Test | What it does | Pass condition |
|------|----------------|----------------|
| **test_valid_item_has_reference_and_audio_path** | One item with `text_uthmani` and `audio` path under `/base` (file does not exist). | `n_total` = 1; `missing_files` has one entry; path contains "001001" |
| **test_missing_reference** | Item with only `verse_key` and `audio`, no text. | `missing_reference` length 1 |
| **test_reference_from_text_uthmani** | Item with only `text_uthmani`, no audio. | `n_valid` = 1, `n_total` = 1 |
| **test_missing_file** | Item with reference and audio path in a temp dir where file doesn’t exist. | `missing_files` length 1 |
| **test_report_to_dict** | Build a `ValidationReport` with n_total, n_valid, and one missing_reference entry. | `to_dict()` has correct n_total, n_valid, and one missing_reference |

#### 3.4.2 Evaluation metrics (distribution & worst cases)

| Test | What it does | Pass condition |
|------|----------------|----------------|
| **test_wer_cer_distribution** | Two samples: one perfect, one with one word wrong. | `n_samples` = 2, `wer_mean` = 0.25, `worst_wer` length 2 |
| **test_get_worst_cases** | Two samples; request worst 1 by WER. | One result; `sample_id` of worst is "1:2" |

**Using:** Run after changes to dataset validation or evaluation report format. Ensures evaluation layer behaves as specified.

---

### 3.5 `tests/test_scoring.py` — Recitation Scoring

**Purpose:** Test the scoring pipeline: word-level status (correct / wrong / minor_mistake / missing / extra), alignment scoring (0.6×text + 0.3×phonetic + 0.1×timing), segment-by-reference, and count_matching_words.

**Code under test:** `core/scoring.py` (`score_recitation`, `segment_transcript_by_reference`, `count_matching_words`), `core/normalization.normalize_arabic`.

| Test class / test | What it does | Pass condition |
|--------------------|----------------|----------------|
| **TestVerseRecitationAccuracy.test_perfect_match** | Reference (Uthmani) vs user (normalized) identical. | accuracy 100, all words "correct", confidence_level high/medium |
| **TestVerseRecitationAccuracy.test_normalized_script_match** | Persian/Arabic script variants (e.g. ی vs ي). | accuracy ≥ 0, 4 words in word_analysis |
| **TestVerseRecitationAccuracy.test_missing_word** | User omits one word. | accuracy < 100, at least one "missing" |
| **TestVerseRecitationAccuracy.test_extra_word** | User adds a word. | at least one entry in extra_words |
| **TestVerseRecitationAccuracy.test_wrong_word** | User says wrong word. | accuracy < 100, at least one "wrong" or "minor_mistake" |
| **TestAlignmentScoring.test_with_alignment_words** | Perfect text + alignment with timings and confidence. | accuracy 100; word_analysis has timing when provided |
| **TestAlignmentScoring.test_empty_alignment_fallback** | No alignment words. | accuracy 100 (text-only scoring) |
| **TestSegmentByReference.test_already_segmented** | Transcript already has spaces. | Output unchanged |
| **TestSegmentByReference.test_no_spaces** | CTC-style no spaces; segment using reference. | At least two words in output |
| **TestCountMatchingWords.test_more_matches_wins** | Two transcripts; one matches reference fully, one misses a word. | Full match count 4; other ≤ 3 |
| **TestFastSlowPauses** (several) | Alignment with fast/slow/pause patterns. | Scores and structure as expected (accuracy, timing_data, etc.) |

**Using:** Run after any change to scoring weights, thresholds, or word alignment. No audio; uses mock transcripts and alignment data.

---

## 4. How You Use This in Practice

1. **After code changes**  
   Run the full suite from project root:
   ```bash
   cd /Users/hamid/Desktop/KivyxProjects/quran-ai-backend
   python3 -m unittest discover tests -v
   ```
   You should see **42 tests, OK**.

2. **After changing only metrics or confidence**  
   ```bash
   python3 -m unittest tests.test_metrics tests.test_confidence -v
   ```

3. **After changing ASR or selection**  
   ```bash
   python3 -m unittest tests.test_asr -v
   ```

4. **After changing evaluation or validation**  
   ```bash
   python3 -m unittest tests.test_dataset_validation -v
   ```

5. **After changing scoring**  
   ```bash
   python3 -m unittest tests.test_scoring -v
   ```

6. **Scripts (no tests, but related)**  
   - Dataset validation: `python3 scripts/validate_dataset.py dataset/fatiha_dataset.json`  
   - WER/CER self-test: `python3 scripts/benchmark_wer_cer.py dataset/fatiha_dataset.json --self-test`

---

## 5. Dependencies (for tests)

- **Python 3** with `unittest` (standard library).
- Project packages: `core` (normalization, metrics, scoring, asr, confidence), `evaluation` (asr_metrics, dataset_validation, benchmark_runner). These pull in `rapidfuzz`, `numpy`, etc. from `requirements.txt`.
- **No** Whisper/Wav2Vec2 download or GPU required for the unit tests above; ASR tests use mocks.

---

## 6. Summary Table

| Module | File | Tests | Purpose |
|--------|------|-------|--------|
| Metrics | test_metrics.py | 11 | WER, CER, wer_cer on Arabic text |
| Confidence | test_confidence.py | 4 | get_asr_quality_score shape and values |
| ASR | test_asr.py | 5 | Dual ASR selection (mocked), result keys |
| Dataset & evaluation | test_dataset_validation.py | 7 | validate_dataset, EvaluationReport, get_worst_cases |
| Scoring | test_scoring.py | 15 | score_recitation, alignment, segment_by_reference, count_matching_words, timing edge cases |
| **Phase 3** | test_phoneme.py | 14 | arabic_to_phonemes, verse_to_phoneme_sequence, Madd/Qalqalah/Ghunnah/heavy, boundaries |
| **Phase 3** | test_phoneme_alignment.py | 8 | align_phoneme_sequences, Levenshtein, substitution/deletion/insertion, tajweed_error_positions |
| **Phase 3** | test_tajweed_rules.py | 9 | detect_tajweed_errors (Madd, Ghunnah, Qalqalah, Heavy_Light), tajweed_rule_accuracy |

**Total: 42 + 29 = 71 tests** (Phase 1/2 + Phase 3). All can be run with:

```bash
python3 -m unittest discover tests -v
```

---

## 7. How to Test Phase 3 (Phoneme & Tajweed)

### 7.1 Run Phase 3 unit tests only

No audio or models; uses mock phoneme sequences and alignment results.

```bash
# All Phase 3 tests (phoneme extraction, alignment, tajweed rules)
python3 -m unittest tests.test_phoneme tests.test_phoneme_alignment tests.test_tajweed_rules -v

# Single module
python3 -m unittest tests.test_phoneme -v
python3 -m unittest tests.test_phoneme_alignment -v
python3 -m unittest tests.test_tajweed_rules -v
```

### 7.2 Test the scoring pipeline with Phase 3 (text only)

Run the helper script that builds reference and hypothesis phoneme sequences, aligns them, detects tajweed errors, and calls `score_recitation` with Phase 3 arguments:

```bash
python3 scripts/test_phase3_pipeline.py
```

You will see printed output including `phoneme_accuracy`, `phoneme_aware_score`, and `tajweed_errors`.

### 7.3 Test the full API with audio (/verify)

1. Start the server: `python3 main.py`
2. Send a POST request with an audio file (e.g. recitation of 1:1):

   ```bash
   curl -X POST "http://localhost:8000/verify?surah=1&ayah=1" -F "audio=@path/to/your_audio.wav"
   ```

3. In the JSON response, when the phoneme pipeline runs successfully you will see:
   - `phoneme_accuracy` (0–1)
   - `tajweed_rule_accuracy` (0–1)
   - `phoneme_aware_score` (0–100): 0.5×word + 0.3×phoneme + 0.2×tajweed_rule
   - `tajweed_errors`: list of `{ rule_name, word, expected, detected, severity }`

If the phoneme step fails, the API still returns only Phase 1/2 fields (backward compatible).

### 7.4 Quick sanity check (Python one-liner)

From project root:

```bash
python3 -c "
from core.phoneme import verse_to_phoneme_sequence
from alignment.phoneme_alignment import align_phoneme_sequences
from tajweed.rules import detect_tajweed_errors, tajweed_rule_accuracy
from core.scoring import score_recitation

ref = 'بِسْمِ اللّٰهِ الرَّحْمٰنِ'
hyp = 'بسم الله الرحمن'
r = verse_to_phoneme_sequence(ref)
h = verse_to_phoneme_sequence(hyp)
align = align_phoneme_sequences(r['phoneme_sequence'], h['phoneme_sequence'], r['word_boundaries'], r['words'])
errs = detect_tajweed_errors(align, r['words'])
tacc = tajweed_rule_accuracy(align)
result = score_recitation(ref, hyp, phoneme_accuracy=align['phoneme_accuracy'], tajweed_rule_accuracy=tacc, tajweed_errors=errs)
print('phoneme_accuracy', result.get('phoneme_accuracy'))
print('phoneme_aware_score', result.get('phoneme_aware_score'))
print('tajweed_errors count', len(result.get('tajweed_errors', [])))
"
```
