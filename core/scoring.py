"""
Tarteel-style recitation scoring: word-level alignment with clear mistake types.
- correct: word recited correctly
- wrong: incorrect pronunciation
- minor_mistake: small phoneme slip (e.g. 78–92% similarity)
- missing / skipped: word omitted
- extra: user said a word not in the verse (insertion)

Final word score = 0.6 * text_similarity + 0.3 * phonetic_similarity + 0.1 * timing_similarity
when alignment (CTC) data is provided; otherwise text-only with 92% / 78% thresholds.
"""
from difflib import SequenceMatcher
from rapidfuzz import fuzz
from .normalization import normalize_arabic
from typing import Dict, Any, List, Optional, Tuple

# Tarteel-like thresholds
SIMILARITY_CORRECT = 0.92   # >= 92% → correct
SIMILARITY_MINOR = 0.78     # 78–92% → minor_mistake; < 78% → wrong

# Combined score weights (when alignment is available)
WEIGHT_TEXT = 0.6
WEIGHT_PHONETIC = 0.3
WEIGHT_TIMING = 0.1

def _word_similarity(ref_word: str, user_word: str) -> float:
    """Strict similarity for scoring: ratio is primary; partial_ratio only slight boost for 1-char ASR slips."""
    ratio = fuzz.ratio(ref_word, user_word) / 100.0
    partial = fuzz.partial_ratio(ref_word, user_word) / 100.0
    # Prefer ratio so wrong words don't pass; allow small boost from partial for near-matches only
    if ratio >= 0.85:
        return max(ratio, min(partial, ratio + 0.06))
    return ratio


def segment_transcript_by_reference(raw_transcript: str, reference_text: str) -> str:
    """
    When ASR returns text without spaces (e.g. wav2vec CTC), segment it using the
    reference verse so we get word boundaries and can compute accuracy properly.
    """
    raw_norm = normalize_arabic(raw_transcript or "").replace(" ", "")
    ref_words = normalize_arabic(reference_text or "").split()
    if not ref_words or not raw_norm:
        return (raw_transcript or "").strip()

    # If transcript already has several spaces, assume it's already word-segmented
    if (raw_transcript or "").count(" ") >= max(1, len(ref_words) - 2):
        return (raw_transcript or "").strip()

    segments: List[str] = []
    remaining = raw_norm
    # Greedy: for each ref word, find best-matching prefix of remaining (min length 1)
    for ref_w in ref_words:
        if not remaining:
            break
        best_len = 1
        best_sim = 0.0
        max_len = min(len(remaining), len(ref_w) + 10)
        for L in range(1, max_len + 1):
            cand = remaining[:L]
            sim = max(fuzz.ratio(ref_w, cand), fuzz.partial_ratio(ref_w, cand)) / 100.0
            if sim > best_sim:
                best_sim = sim
                best_len = L
        segments.append(remaining[:best_len])
        remaining = remaining[best_len:]
    if remaining:
        segments.append(remaining)
    return " ".join(segments)


def _timing_similarity(
    word_index: int,
    alignment_words: List[Dict[str, Any]],
    total_duration: float,
) -> float:
    """
    Compute timing similarity for one word: order and duration reasonability.
    Returns 0.0–1.0. Uses alignment word start_time/end_time when available.
    """
    if not alignment_words or word_index >= len(alignment_words):
        return 0.5  # neutral when no timing data
    w = alignment_words[word_index]
    start = w.get("start_time") or 0.0
    end = w.get("end_time") or 0.0
    if end <= start or total_duration <= 0:
        return 0.5
    duration = end - start
    # Reasonable word duration: 0.2–3.0 s typical; penalize too short or too long
    if duration < 0.05:
        return 0.3
    if duration > 4.0:
        return 0.7
    # Order: word should start after previous word ended
    if word_index > 0:
        prev_end = alignment_words[word_index - 1].get("end_time") or 0.0
        if start < prev_end - 0.15:  # large overlap → order issue
            return 0.4
    return 1.0


def _align_words(
    orig_norm: List[str],
    orig_display: List[str],
    user_norm: List[str],
    alignment_words: Optional[List[Dict[str, Any]]] = None,
    audio_duration: float = 0.0,
) -> Tuple[List[Dict[str, Any]], List[str], int]:
    """
    Align reference words to user words. When alignment exists: 0.6*text + 0.3*phonetic + 0.1*timing.
    When alignment does NOT exist: timing_weight = 0.0. Returns (word_analysis, extra_user_words, correct_count).
    """
    matcher = SequenceMatcher(None, orig_norm, user_norm)
    word_analysis: List[Dict[str, Any]] = []
    extra_words: List[str] = []
    correct_count = 0
    use_alignment = alignment_words and len(alignment_words) >= len(orig_display)

    def _status_from_combined(text_sim: float, phonetic_sim: float, timing_sim: float) -> str:
        if use_alignment:
            combined = WEIGHT_TEXT * text_sim + WEIGHT_PHONETIC * phonetic_sim + WEIGHT_TIMING * timing_sim
            if combined >= SIMILARITY_CORRECT:
                return "correct"
            if combined >= SIMILARITY_MINOR:
                return "minor_mistake"
            return "wrong"
        if text_sim >= SIMILARITY_CORRECT:
            return "correct"
        if text_sim >= SIMILARITY_MINOR:
            return "minor_mistake"
        return "wrong"

    def _feedback_from_status(status: str) -> str:
        return (
            "Perfectly recited." if status == "correct"
            else "Minor phoneme slip." if status == "minor_mistake"
            else "Check pronunciation."
        )

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for i in range(i1, i2):
                text_sim = 1.0
                phonetic_sim = 1.0
                if use_alignment and i < len(alignment_words):
                    phonetic_sim = float(alignment_words[i].get("confidence") or alignment_words[i].get("phonetic_similarity") or 1.0)
                    timing_sim = _timing_similarity(i, alignment_words, audio_duration)
                else:
                    timing_sim = 0.0  # timing_weight = 0 when no alignment
                status = _status_from_combined(text_sim, phonetic_sim, timing_sim)
                if status == "correct":
                    correct_count += 1
                combined_conf = (
                    WEIGHT_TEXT * text_sim + WEIGHT_PHONETIC * phonetic_sim + (WEIGHT_TIMING * timing_sim if use_alignment else 0.0)
                )
                if not use_alignment:
                    combined_conf = text_sim  # text-only
                entry = {
                    "word": orig_display[i],
                    "status": status,
                    "confidence": combined_conf,
                    "feedback": _feedback_from_status(status),
                }
                if use_alignment and i < len(alignment_words):
                    aw = alignment_words[i]
                    if aw.get("start_time") is not None:
                        entry["start_time"] = aw["start_time"]
                    if aw.get("end_time") is not None:
                        entry["end_time"] = aw["end_time"]
                word_analysis.append(entry)

        elif tag == "replace":
            orig_range = orig_norm[i1:i2]
            orig_display_range = orig_display[i1:i2]
            user_range = user_norm[j1:j2]
            pairs: List[Tuple[int, int, float]] = []
            for ii, o_norm in enumerate(orig_range):
                for jj, u in enumerate(user_range):
                    sim = _word_similarity(o_norm, u)
                    pairs.append((ii, jj, sim))
            pairs.sort(key=lambda x: -x[2])
            used_ref, used_user = set(), set()
            assigned: Dict[int, Tuple[int, int, float]] = {}  # ref_idx -> (user_idx, text_sim)
            for ii, jj, sim in pairs:
                if ii not in used_ref and jj not in used_user:
                    used_ref.add(ii)
                    used_user.add(jj)
                    assigned[ii] = (jj, sim)
            for ii, o_disp in enumerate(orig_display_range):
                o_norm = orig_range[ii]
                global_idx = i1 + ii
                if ii in assigned:
                    jj, text_sim = assigned[ii][0], assigned[ii][1]
                    phonetic_sim = 0.5
                    if use_alignment and global_idx < len(alignment_words):
                        phonetic_sim = float(alignment_words[global_idx].get("confidence") or alignment_words[global_idx].get("phonetic_similarity") or 0.5)
                    timing_sim = _timing_similarity(global_idx, alignment_words or [], audio_duration)
                    status = _status_from_combined(text_sim, phonetic_sim, timing_sim)
                    if status == "correct":
                        correct_count += 1
                    combined = (
                        WEIGHT_TEXT * text_sim + WEIGHT_PHONETIC * phonetic_sim + (WEIGHT_TIMING * timing_sim if use_alignment else 0.0)
                    )
                    if not use_alignment:
                        combined = text_sim
                    entry = {
                        "word": o_disp,
                        "status": status,
                        "confidence": combined,
                        "feedback": _feedback_from_status(status),
                    }
                    if use_alignment and global_idx < len(alignment_words):
                        aw = alignment_words[global_idx]
                        if aw.get("start_time") is not None:
                            entry["start_time"] = aw["start_time"]
                        if aw.get("end_time") is not None:
                            entry["end_time"] = aw["end_time"]
                    word_analysis.append(entry)
                else:
                    word_analysis.append({
                        "word": o_disp,
                        "status": "missing",
                        "confidence": 0.0,
                        "feedback": "Word omitted.",
                    })
            for jj, u in enumerate(user_range):
                if jj not in used_user:
                    extra_words.append(u)

        elif tag == "delete":
            for i in range(i1, i2):
                word_analysis.append({
                    "word": orig_display[i],
                    "status": "missing",
                    "confidence": 0.0,
                    "feedback": "Word omitted.",
                })

        elif tag == "insert":
            for j in range(j1, j2):
                extra_words.append(user_norm[j])

    return word_analysis, extra_words, correct_count


def score_recitation(
    original_text: str,
    user_text: str,
    tajweed_feedback: Optional[List[Dict[str, Any]]] = None,
    alignment_words: Optional[List[Dict[str, Any]]] = None,
    audio_duration: float = 0.0,
    phoneme_accuracy: Optional[float] = None,
    tajweed_rule_accuracy: Optional[float] = None,
    tajweed_errors: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Tarteel-style multi-layer verification with optional time-alignment scoring.
    When alignment_words is provided, per-word score = 0.6*text + 0.3*phonetic + 0.1*timing.
    Returns word_analysis with status: correct | wrong | minor_mistake | missing | extra.
    Phase 3: When phoneme_accuracy and tajweed_rule_accuracy are provided, adds
    phoneme_aware_score = 0.5*word + 0.3*phoneme + 0.2*tajweed_rule. Backward compatible.
    """
    orig_norm = normalize_arabic(original_text)
    user_norm = normalize_arabic(user_text or "")

    orig_words = orig_norm.split()
    user_words = user_norm.split()

    orig_display_words = [w for w in original_text.split() if normalize_arabic(w)]
    if len(orig_display_words) != len(orig_words):
        orig_display_words = orig_words

    word_analysis, extra_words, correct_count = _align_words(
        orig_words,
        orig_display_words,
        user_words,
        alignment_words=alignment_words,
        audio_duration=audio_duration,
    )

    accuracy_score = (correct_count / len(orig_words) * 100) if orig_words else 0.0

    tajweed_score = 100.0
    if tajweed_feedback:
        tajweed_score = sum(item.get("score", 0) for item in tajweed_feedback) / len(tajweed_feedback) * 100

    combined_score = (accuracy_score * 0.7) + (tajweed_score * 0.3)
    confidence_score = round(combined_score, 2)  # 0–100 numeric
    if combined_score >= 95:
        confidence_level = "high"
        teacher_feedback = "Excellent recitation! Your pronunciation and Tajweed are near perfect."
    elif combined_score >= 80:
        confidence_level = "medium"
        teacher_feedback = "Good effort. Focus a bit more on the highlighted words and rules."
    else:
        confidence_level = "low"
        teacher_feedback = "Please review the correct pronunciation and Tajweed rules for this verse."

    # timing_data: word-level start/end for UI (from alignment or word_analysis)
    timing_data = [
        {"word": w["word"], "start_time": w.get("start_time"), "end_time": w.get("end_time")}
        for w in word_analysis
    ]

    result: Dict[str, Any] = {
        "transcribed_text": user_text,
        "accuracy_score": round(accuracy_score, 2),
        "word_analysis": word_analysis,
        "extra_words": extra_words,
        "tajweed_score": round(tajweed_score, 2),
        "confidence_level": confidence_level,
        "confidence_score": confidence_score,
        "teacher_feedback_text": teacher_feedback,
        "tajweed_feedback": tajweed_feedback or [],
        "timing_data": timing_data,
    }

    # Phase 3: phoneme-aware combined score when phoneme alignment data is provided
    if phoneme_accuracy is not None and tajweed_rule_accuracy is not None:
        word_acc = accuracy_score / 100.0
        phoneme_aware = 0.5 * word_acc + 0.3 * phoneme_accuracy + 0.2 * tajweed_rule_accuracy
        result["phoneme_accuracy"] = round(phoneme_accuracy, 4)
        result["tajweed_rule_accuracy"] = round(tajweed_rule_accuracy, 4)
        result["phoneme_aware_score"] = round(phoneme_aware * 100.0, 2)
        if tajweed_errors is not None:
            result["tajweed_errors"] = tajweed_errors

    return result


def count_matching_words(reference_text: str, transcript: str) -> int:
    """Return how many reference words have a matching word in the transcript (for choosing best ASR)."""
    orig_norm = normalize_arabic(reference_text).split()
    user_norm = normalize_arabic(transcript or "").split()
    if not orig_norm:
        return 0
    word_analysis, _, correct_count = _align_words(orig_norm, orig_norm, user_norm)
    return correct_count


def get_lightweight_word_feedback(reference_text: str, partial_transcript: str) -> List[Dict[str, Any]]:
    """
    Lightweight word-level feedback for streaming: no alignment/timing, text-only.
    Returns one entry per reference word: {"word": str, "status": "correct"|"wrong"|"minor_mistake"|"missing"|"pending"}.
    Used by Phase 4 streaming layer for incremental feedback; does NOT run full tajweed/scoring.
    """
    orig_norm = normalize_arabic(reference_text or "").split()
    user_norm = normalize_arabic(partial_transcript or "").split()
    if not orig_norm:
        return []
    word_analysis, _, _ = _align_words(orig_norm, orig_norm, user_norm)
    return [{"word": wa.get("word", ""), "status": wa.get("status", "pending")} for wa in word_analysis]
