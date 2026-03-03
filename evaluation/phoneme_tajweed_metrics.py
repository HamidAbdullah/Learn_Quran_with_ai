"""
Research-level evaluation: Phoneme Error Rate, Tajweed Rule Precision/Recall, Expert Agreement.
"""
from typing import Dict, Any, List, Optional, Tuple


def phoneme_error_rate(
    reference_phonemes: List[str],
    hypothesis_phonemes: List[str],
) -> float:
    """
    Phoneme Error Rate = 1 - phoneme_accuracy (edit distance normalized by reference length).
    Returns PER in [0, +inf); 0 = perfect.
    """
    if not reference_phonemes:
        return 0.0 if not hypothesis_phonemes else 1.0
    try:
        from alignment.phoneme_alignment import align_phoneme_sequences
        align = align_phoneme_sequences(reference_phonemes, hypothesis_phonemes or [])
        acc = align.get("phoneme_accuracy", 0.0)
        return round(1.0 - acc, 4)
    except Exception:
        return 1.0


def tajweed_rule_precision_recall(
    predicted_tajweed_errors: List[Dict[str, Any]],
    reference_tajweed_labels: List[Dict[str, Any]],
    rule_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Tajweed rule-level precision and recall vs reference labels.
    reference_tajweed_labels: list of {"rule_name", "word_index" or "word", "correct": bool} or similar.
    predicted_tajweed_errors: list of {"rule_name", "word", ...} from detect_tajweed_errors.
    Returns {"precision": float, "recall": float, "f1": float, "per_rule": {...}}.
    """
    # Placeholder: when no reference labels, return Nones
    if not reference_tajweed_labels:
        return {
            "precision": None,
            "recall": None,
            "f1": None,
            "message": "Tajweed P/R require reference_tajweed_labels (expert or gold).",
        }
    # Build set of (rule_name, word) for ref errors (where correct=False)
    ref_errors = set()
    ref_total = 0
    for item in reference_tajweed_labels:
        rule = item.get("rule_name", "")
        word = item.get("word") or item.get("word_index")
        correct = item.get("correct", True)
        ref_total += 1
        if not correct:
            ref_errors.add((rule, str(word)))
    pred_errors = set()
    for e in predicted_tajweed_errors or []:
        rule = e.get("rule_name", "")
        word = e.get("word") or ""
        pred_errors.add((rule, word))
    tp = len(ref_errors & pred_errors)
    pred_n = len(pred_errors)
    ref_n = len(ref_errors)
    precision = tp / pred_n if pred_n else 0.0
    recall = tp / ref_n if ref_n else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "predicted_errors": pred_n,
        "reference_errors": ref_n,
    }


def expert_agreement_score(
    system_scores: List[float],
    expert_scores: List[float],
) -> Optional[float]:
    """
    Correlation or agreement between system quality scores and expert ratings.
    Returns Pearson correlation when both lists have same length and variance; else None.
    """
    if not system_scores or not expert_scores or len(system_scores) != len(expert_scores):
        return None
    try:
        import math
        n = len(system_scores)
        mx = sum(system_scores) / n
        my = sum(expert_scores) / n
        sx = math.sqrt(sum((x - mx) ** 2 for x in system_scores) / n) if n else 0
        sy = math.sqrt(sum((y - my) ** 2 for y in expert_scores) / n) if n else 0
        if sx * sy == 0:
            return None
        r = sum((system_scores[i] - mx) * (expert_scores[i] - my) for i in range(n)) / (n * sx * sy)
        return round(r, 4)
    except Exception:
        return None
