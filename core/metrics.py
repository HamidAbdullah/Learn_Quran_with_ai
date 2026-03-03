"""
ASR evaluation metrics: Word Error Rate (WER) and Character Error Rate (CER).
Uses normalized Arabic text; edit distance (S+D+I) / reference length.
"""
from typing import Tuple, List
from core.normalization import normalize_arabic


def _edit_distance_word_level(ref_words: List[str], hyp_words: List[str]) -> Tuple[int, int, int, int]:
    """
    Levenshtein edit distance at word level. Returns (substitutions, deletions, insertions, ref_len).
    """
    R, H = len(ref_words), len(hyp_words)
    # dp[i][j] = (min edits to match ref[:i] to hyp[:j], s, d, i)
    # We only need the counts: S, D, I. Standard approach: one DP for distance, then backtrack for S,D,I.
    INF = 10 ** 9
    # dp[i][j] = minimum edit distance to transform ref_words[:i] into hyp_words[:j]
    dp = [[INF] * (H + 1) for _ in range(R + 1)]
    dp[0][0] = 0
    for i in range(R + 1):
        for j in range(H + 1):
            if i == 0 and j == 0:
                continue
            if i > 0:
                dp[i][j] = min(dp[i][j], dp[i - 1][j] + 1)  # deletion
            if j > 0:
                dp[i][j] = min(dp[i][j], dp[i][j - 1] + 1)  # insertion
            if i > 0 and j > 0:
                cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
                dp[i][j] = min(dp[i][j], dp[i - 1][j - 1] + cost)

    # Backtrack to get S, D, I
    s, d, ins = 0, 0, 0
    i, j = R, H
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            d += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ins += 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            s += 1
            i -= 1
            j -= 1
        else:
            i -= 1
            j -= 1
    return s, d, ins, R


def _edit_distance_char_level(ref_chars: List[str], hyp_chars: List[str]) -> Tuple[int, int, int, int]:
    """Levenshtein at character level. Returns (S, D, I, ref_len)."""
    R, H = len(ref_chars), len(hyp_chars)
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
                cost = 0 if ref_chars[i - 1] == hyp_chars[j - 1] else 1
                dp[i][j] = min(dp[i][j], dp[i - 1][j - 1] + cost)

    s, d, ins = 0, 0, 0
    i, j = R, H
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_chars[i - 1] == hyp_chars[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            d += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ins += 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            s += 1
            i -= 1
            j -= 1
        else:
            i -= 1
            j -= 1
    return s, d, ins, R


def wer(reference: str, hypothesis: str, normalize: bool = True) -> float:
    """
    Word Error Rate: (S + D + I) / N where N = number of reference words.
    reference: ground-truth text (e.g. verse).
    hypothesis: ASR output.
    normalize: apply Arabic normalization to both (default True).
    Returns value in [0, +inf); 0 = perfect match.
    """
    ref_norm = normalize_arabic(reference) if normalize else reference
    hyp_norm = normalize_arabic(hypothesis or "") if normalize else (hypothesis or "")
    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    s, d, ins, R = _edit_distance_word_level(ref_words, hyp_words)
    edits = s + d + ins
    return edits / R


def cer(reference: str, hypothesis: str, normalize: bool = True, remove_spaces: bool = True) -> float:
    """
    Character Error Rate: (S + D + I) / N where N = number of reference characters.
    remove_spaces: if True, compare without spaces (standard for Arabic).
    Returns value in [0, +inf); 0 = perfect match.
    """
    ref_norm = normalize_arabic(reference) if normalize else reference
    hyp_norm = normalize_arabic(hypothesis or "") if normalize else (hypothesis or "")
    if remove_spaces:
        ref_norm = ref_norm.replace(" ", "")
        hyp_norm = hyp_norm.replace(" ", "")
    ref_chars = list(ref_norm)
    hyp_chars = list(hyp_norm)
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    s, d, ins, R = _edit_distance_char_level(ref_chars, hyp_chars)
    edits = s + d + ins
    return edits / R


def wer_cer(
    reference: str,
    hypothesis: str,
    normalize: bool = True,
) -> Tuple[float, float]:
    """
    Compute both WER and CER for reference vs hypothesis.
    Returns (wer, cer).
    """
    return wer(reference, hypothesis, normalize=normalize), cer(
        reference, hypothesis, normalize=normalize
    )
