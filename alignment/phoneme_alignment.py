"""
Phoneme-level alignment for Phase 3 — Phoneme Alignment Engine.

Aligns reference phoneme sequence with hypothesis (user) phoneme sequence using
Levenshtein edit distance. Tags each error as substitution, deletion, or insertion
and maps error positions to word indices for Tajweed rule detection.

Does not require audio or ASR; operates on phoneme sequences produced by core/phoneme.py.
"""
from typing import List, Dict, Any, Optional, Tuple

# Phoneme symbols that indicate Tajweed-relevant segments (for error positioning)
TAJWEED_PHONEME_SUFFIXES = ("_Q", "_G", "_H")  # Qalqalah, Ghunnah, Heavy
MADD_PHONEMES = ("aa", "ii", "uu")


def _levenshtein_alignment(
    ref: List[str],
    hyp: List[str],
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Compute Levenshtein edit distance and return alignment with operation tags.
    Returns (num_edits, list of operations).
    Each operation: {"type": "match"|"substitution"|"deletion"|"insertion", "ref_idx", "hyp_idx", "ref_phoneme", "hyp_phoneme"}.
    """
    R, H = len(ref), len(hyp)
    INF = 10 ** 9
    dp = [[INF] * (H + 1) for _ in range(R + 1)]
    dp[0][0] = 0
    for i in range(R + 1):
        for j in range(H + 1):
            if i == 0 and j == 0:
                continue
            if i > 0:
                dp[i][j] = min(dp[i][j], dp[i - 1][j] + 1)
            if j > 0:
                dp[i][j] = min(dp[i][j], dp[i][j - 1] + 1)
            if i > 0 and j > 0:
                cost = 0 if ref[i - 1] == hyp[j - 1] else 1
                dp[i][j] = min(dp[i][j], dp[i - 1][j - 1] + cost)

    # Backtrack to build operations
    ops: List[Dict[str, Any]] = []
    i, j = R, H
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            ops.append({
                "type": "match",
                "ref_idx": i - 1,
                "hyp_idx": j - 1,
                "ref_phoneme": ref[i - 1],
                "hyp_phoneme": hyp[j - 1],
            })
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append({
                "type": "deletion",
                "ref_idx": i - 1,
                "hyp_idx": -1,
                "ref_phoneme": ref[i - 1],
                "hyp_phoneme": None,
            })
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append({
                "type": "insertion",
                "ref_idx": -1,
                "hyp_idx": j - 1,
                "ref_phoneme": None,
                "hyp_phoneme": hyp[j - 1],
            })
            j -= 1
        else:
            ops.append({
                "type": "substitution",
                "ref_idx": i - 1,
                "hyp_idx": j - 1,
                "ref_phoneme": ref[i - 1],
                "hyp_phoneme": hyp[j - 1],
            })
            i -= 1
            j -= 1
    ops.reverse()
    return dp[R][H], ops


def _is_tajweed_relevant(phoneme: str) -> bool:
    """True if phoneme is Madd, Qalqalah, Ghunnah, or Heavy (for error positioning)."""
    if not phoneme:
        return False
    if phoneme in MADD_PHONEMES:
        return True
    return any(phoneme.endswith(s) for s in TAJWEED_PHONEME_SUFFIXES)


def align_phoneme_sequences(
    reference_phonemes: List[str],
    hypothesis_phonemes: List[str],
    word_boundaries: Optional[List[Tuple[int, int]]] = None,
    reference_words: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Align reference and hypothesis phoneme sequences; tag errors; compute accuracy and Tajweed error positions.

    Args:
        reference_phonemes: Flat list of reference phonemes (e.g. from verse_to_phoneme_sequence).
        hypothesis_phonemes: Flat list of user/hypothesis phonemes (e.g. from ASR text → verse_to_phoneme_sequence).
        word_boundaries: Optional [(start, end), ...] for each word into reference_phonemes.
        reference_words: Optional list of reference words (for structured errors).

    Returns:
        {
            "phoneme_accuracy": float,        # 1 - (S+D+I)/len(reference_phonemes)
            "phoneme_errors": List[Dict],    # each: type, ref_phoneme, hyp_phoneme, ref_idx, hyp_idx
            "tajweed_error_positions": List[Dict],  # errors where ref or hyp phoneme is Tajweed-relevant: word_index, rule_hint, ...
            "num_substitutions": int,
            "num_deletions": int,
            "num_insertions": int,
            "reference_length": int,
            "hypothesis_length": int,
        }
    """
    ref = reference_phonemes or []
    hyp = hypothesis_phonemes or []
    word_boundaries = word_boundaries or []
    reference_words = reference_words or []

    out: Dict[str, Any] = {
        "phoneme_accuracy": 1.0,
        "phoneme_errors": [],
        "tajweed_error_positions": [],
        "num_substitutions": 0,
        "num_deletions": 0,
        "num_insertions": 0,
        "reference_length": len(ref),
        "hypothesis_length": len(hyp),
    }
    if not ref:
        out["phoneme_accuracy"] = 1.0 if not hyp else 0.0
        return out

    num_edits, ops = _levenshtein_alignment(ref, hyp)
    out["phoneme_accuracy"] = max(0.0, min(1.0, 1.0 - num_edits / len(ref)))

    errors: List[Dict[str, Any]] = []
    tajweed_positions: List[Dict[str, Any]] = []
    s, d, i = 0, 0, 0

    for op in ops:
        t = op["type"]
        if t == "substitution":
            s += 1
            errors.append({
                "type": "substitution",
                "ref_phoneme": op["ref_phoneme"],
                "hyp_phoneme": op["hyp_phoneme"],
                "ref_idx": op["ref_idx"],
                "hyp_idx": op["hyp_idx"],
            })
            if _is_tajweed_relevant(op["ref_phoneme"] or "") or _is_tajweed_relevant(op["hyp_phoneme"] or ""):
                word_idx = -1
                for wi, (start, end) in enumerate(word_boundaries):
                    if start <= op["ref_idx"] < end:
                        word_idx = wi
                        break
                tajweed_positions.append({
                    "ref_idx": op["ref_idx"],
                    "word_index": word_idx,
                    "type": "substitution",
                    "ref_phoneme": op["ref_phoneme"],
                    "hyp_phoneme": op["hyp_phoneme"],
                    "word": reference_words[word_idx] if 0 <= word_idx < len(reference_words) else None,
                })
        elif t == "deletion":
            d += 1
            errors.append({
                "type": "deletion",
                "ref_phoneme": op["ref_phoneme"],
                "hyp_phoneme": None,
                "ref_idx": op["ref_idx"],
                "hyp_idx": -1,
            })
            if _is_tajweed_relevant(op["ref_phoneme"] or ""):
                word_idx = -1
                for wi, (start, end) in enumerate(word_boundaries):
                    if start <= op["ref_idx"] < end:
                        word_idx = wi
                        break
                tajweed_positions.append({
                    "ref_idx": op["ref_idx"],
                    "word_index": word_idx,
                    "type": "deletion",
                    "ref_phoneme": op["ref_phoneme"],
                    "hyp_phoneme": None,
                    "word": reference_words[word_idx] if 0 <= word_idx < len(reference_words) else None,
                })
        elif t == "insertion":
            i += 1
            errors.append({
                "type": "insertion",
                "ref_phoneme": None,
                "hyp_phoneme": op["hyp_phoneme"],
                "ref_idx": -1,
                "hyp_idx": op["hyp_idx"],
            })

    out["phoneme_errors"] = errors
    out["tajweed_error_positions"] = tajweed_positions
    out["num_substitutions"] = s
    out["num_deletions"] = d
    out["num_insertions"] = i
    out["word_boundaries"] = word_boundaries
    return out
