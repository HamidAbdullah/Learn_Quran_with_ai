"""
Tajweed rule detection engine for Phase 3.

Consumes phoneme alignment output (tajweed_error_positions, phoneme_errors) and
classifies violations into structured errors:
- Madd length mismatch (expected aa/ii/uu, got short or wrong)
- Missing Ghunnah (expected n_G/m_G, got n/m or deletion)
- Qalqalah mispronunciation (expected *_Q, got plain or substitution)
- Heavy/Light misarticulation (expected *_H heavy, got light or vice versa)

Returns: rule_name, word, expected, detected, severity.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Madd phonemes (long vowels)
MADD_PHONEMES = {"aa", "ii", "uu"}
# Short vowels
SHORT_VOWELS = {"a", "i", "u"}
# Qalqalah suffix
QALQALAH_SUFFIX = "_Q"
# Ghunnah suffix
GHUNNAH_SUFFIX = "_G"
# Heavy letter suffix
HEAVY_SUFFIX = "_H"
# Base consonants that have heavy counterparts (for heavy/light rules)
HEAVY_LIGHT_PAIRS = [
    ("s", "s_H"),   # sin / sad
    ("d", "d_H"),   # dal / dad
    ("t", "t_H"),   # ta / ta heavy
    ("z", "z_H"),   # zay / za
    ("q", "q"),     # qaf is always heavy in classical
]


@dataclass
class TajweedError:
    """Single Tajweed rule violation."""
    rule_name: str
    word: Optional[str]
    expected: str
    detected: str
    severity: str  # "high", "medium", "low"


def _severity_from_rule(rule_name: str, expected: str, detected: str) -> str:
    """Assign severity based on rule and how wrong the detection is."""
    if not detected or detected == "deletion":
        return "high"
    if rule_name in ("Madd", "Ghunnah"):
        return "high"  # Madd and Ghunnah are critical
    if rule_name == "Qalqalah":
        return "medium"
    if rule_name == "Heavy_Light":
        return "medium"
    return "low"


def detect_tajweed_errors(
    phoneme_alignment_result: Dict[str, Any],
    reference_words: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Map phoneme alignment errors to Tajweed rule violations.

    Args:
        phoneme_alignment_result: Output from align_phoneme_sequences (phoneme_errors, tajweed_error_positions).
        reference_words: Optional list of reference words for per-word reporting.

    Returns:
        List of {
            "rule_name": str,
            "word": str | None,
            "expected": str,
            "detected": str,
            "severity": "high" | "medium" | "low",
        }
    """
    reference_words = reference_words or []
    errors: List[Dict[str, Any]] = []
    seen_key: set = set()  # (rule_name, word_index, ref_phoneme) to dedupe

    tajweed_positions = phoneme_alignment_result.get("tajweed_error_positions") or []
    phoneme_errors = phoneme_alignment_result.get("phoneme_errors") or []

    for pos in tajweed_positions:
        ref_p = pos.get("ref_phoneme") or ""
        hyp_p = pos.get("hyp_phoneme")
        word_idx = pos.get("word_index", -1)
        word = pos.get("word")
        if word is None and 0 <= word_idx < len(reference_words):
            word = reference_words[word_idx]
        op_type = pos.get("type", "substitution")

        # Madd length mismatch: reference had aa/ii/uu but hypothesis has short or wrong
        if ref_p in MADD_PHONEMES:
            detected = str(hyp_p) if hyp_p else "deletion"
            key = ("Madd", word_idx, ref_p)
            if key not in seen_key:
                seen_key.add(key)
                errors.append({
                    "rule_name": "Madd",
                    "word": word,
                    "expected": f"long vowel ({ref_p})",
                    "detected": detected if detected else "missing",
                    "severity": _severity_from_rule("Madd", ref_p, detected),
                })

        # Missing Ghunnah: reference had n_G or m_G, hypothesis has plain n/m or deletion
        elif ref_p.endswith(GHUNNAH_SUFFIX):
            detected = str(hyp_p) if hyp_p else "deletion"
            key = ("Ghunnah", word_idx, ref_p)
            if key not in seen_key:
                seen_key.add(key)
                errors.append({
                    "rule_name": "Ghunnah",
                    "word": word,
                    "expected": f"nasalization ({ref_p})",
                    "detected": detected if detected else "missing",
                    "severity": _severity_from_rule("Ghunnah", ref_p, detected),
                })

        # Qalqalah mispronunciation: reference had *_Q, hypothesis has plain or wrong
        elif ref_p.endswith(QALQALAH_SUFFIX):
            base = ref_p.replace(QALQALAH_SUFFIX, "")
            detected = str(hyp_p) if hyp_p else "deletion"
            key = ("Qalqalah", word_idx, ref_p)
            if key not in seen_key:
                seen_key.add(key)
                errors.append({
                    "rule_name": "Qalqalah",
                    "word": word,
                    "expected": f"bounce on {base} ({ref_p})",
                    "detected": detected if detected else "missing",
                    "severity": _severity_from_rule("Qalqalah", ref_p, detected),
                })

        # Heavy/Light: reference had *_H (heavy), hypothesis has light or wrong
        elif ref_p.endswith(HEAVY_SUFFIX):
            detected = str(hyp_p) if hyp_p else "deletion"
            key = ("Heavy_Light", word_idx, ref_p)
            if key not in seen_key:
                seen_key.add(key)
                errors.append({
                    "rule_name": "Heavy_Light",
                    "word": word,
                    "expected": f"heavy letter ({ref_p})",
                    "detected": detected if detected else "missing",
                    "severity": _severity_from_rule("Heavy_Light", ref_p, detected),
                })

    # Optionally scan raw phoneme_errors for substitutions that look like heavy/light or Madd
    for err in phoneme_errors:
        if err.get("type") != "substitution":
            continue
        ref_p = err.get("ref_phoneme") or ""
        hyp_p = err.get("hyp_phoneme") or ""
        ref_idx = err.get("ref_idx", -1)
        word_idx = -1
        for wi, (start, end) in enumerate(phoneme_alignment_result.get("word_boundaries") or []):
            if start <= ref_idx < end:
                word_idx = wi
                break
        word = reference_words[word_idx] if 0 <= word_idx < len(reference_words) else None
        key = ("Heavy_Light_subst", word_idx, ref_p)
        if ref_p != hyp_p and ref_p.endswith(HEAVY_SUFFIX) and not hyp_p.endswith(HEAVY_SUFFIX) and key not in seen_key:
            seen_key.add(key)
            errors.append({
                "rule_name": "Heavy_Light",
                "word": word,
                "expected": f"heavy ({ref_p})",
                "detected": hyp_p,
                "severity": "medium",
            })

    return errors


def tajweed_rule_accuracy(phoneme_alignment_result: Dict[str, Any]) -> float:
    """
    Compute Tajweed rule accuracy: 1 - (num_tajweed_errors / num_tajweed_relevant_phonemes).
    If no Tajweed-relevant phonemes in reference, returns 1.0.
    """
    positions = phoneme_alignment_result.get("tajweed_error_positions") or []
    ref_len = phoneme_alignment_result.get("reference_length") or 0
    if ref_len == 0:
        return 1.0
    # Count unique ref_idx that are Tajweed errors (deletions + substitutions at Tajweed positions)
    n_errors = len(positions)
    # Approximate denominator: assume ~10–20% of phonemes are Tajweed-relevant for typical verse
    n_relevant = max(1, ref_len // 8)
    return max(0.0, min(1.0, 1.0 - n_errors / n_relevant))
