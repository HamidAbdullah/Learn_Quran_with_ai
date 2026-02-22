from difflib import SequenceMatcher
from rapidfuzz import fuzz
from .normalization import normalize_arabic
from typing import Dict, Any, List

def score_recitation(original_text: str, user_text: str, tajweed_feedback: List[Dict[str, Any]] = None):
    """
    High-accuracy multi-layer verification.
    Color Thresholds:
      - Green (>95%): High Confidence
      - Yellow (80-95%): Medium Confidence
      - Red (<80%): Low Confidence
    """
    orig_norm = normalize_arabic(original_text)
    user_norm = normalize_arabic(user_text)
    
    orig_words = orig_norm.split()
    user_words = user_norm.split()
    orig_display_words = original_text.split()
    
    matcher = SequenceMatcher(None, orig_words, user_words)
    word_analysis = []
    correct_count = 0
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for i in range(i1, i2):
                word_analysis.append({
                    "word": orig_display_words[i],
                    "status": "correct",
                    "confidence": 1.0,
                    "feedback": "Perfectly recited."
                })
                correct_count += 1
        elif tag == 'replace':
            for i in range(i1, i2):
                orig_word = orig_words[i]
                best_sim = 0
                for j in range(j1, j2):
                    sim = fuzz.ratio(orig_word, user_words[j]) / 100.0
                    if sim > best_sim: best_sim = sim
                
                status = "wrong"
                if best_sim >= 0.95:
                    status = "correct"
                    correct_count += 1
                elif best_sim >= 0.80:
                    status = "minor_mistake"
                
                word_analysis.append({
                    "word": orig_display_words[i],
                    "status": status,
                    "confidence": best_sim,
                    "feedback": "Minor phoneme slip." if status == "minor_mistake" else "Check pronunciation."
                })
        elif tag == 'delete':
            for i in range(i1, i2):
                word_analysis.append({
                    "word": orig_display_words[i],
                    "status": "missing",
                    "confidence": 0,
                    "feedback": "Word omitted."
                })
        elif tag == 'insert':
            pass # We focus on verifying the original text

    # Final scores
    accuracy_score = (correct_count / len(orig_words) * 100) if orig_words else 0
    
    tajweed_score = 100
    if tajweed_feedback:
        tajweed_score = sum(item["score"] for item in tajweed_feedback) / len(tajweed_feedback) * 100
    
    # Global confidence level
    combined_score = (accuracy_score * 0.7) + (tajweed_score * 0.3)
    if combined_score >= 95:
        confidence_level = "high"
        teacher_feedback = "Excellent recitation! Your pronunciation and Tajweed are near perfect."
    elif combined_score >= 80:
        confidence_level = "medium"
        teacher_feedback = "Good effort. Just focus a bit more on the highlighted rules."
    else:
        confidence_level = "low"
        teacher_feedback = "Please review the correct pronunciation and Tajweed rules for this Ayah."

    return {
        "transcribed_text": user_text,
        "accuracy_score": round(accuracy_score, 2),
        "word_analysis": word_analysis,
        "tajweed_score": round(tajweed_score, 2),
        "confidence_level": confidence_level,
        "teacher_feedback_text": teacher_feedback,
        "tajweed_feedback": tajweed_feedback or []
    }
