"""
Phoneme-from-audio interface for Seerat AI mode.
CRITICAL: Do not use ASR text for phonemes in pronunciation scoring.

This module defines:
- Phoneme vocabulary (Madd, Ghunnah, Qalqalah, Heavy/Light)
- Interface: audio → predicted phoneme sequence, frame confidence, pronunciation deviation score
- Stub implementation until a trained phoneme CTC / pronunciation model is available.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ----- Phoneme vocabulary (research-level) -----
# Madd classes with duration modeling (2, 4, 6 harakat)
MADD_PHONEMES = {"aa", "ii", "uu"}
MADD_DURATION_CLASSES = ["aa_2", "aa_4", "aa_6", "ii_2", "ii_4", "ii_6", "uu_2", "uu_4", "uu_6"]

# Ghunnah resonance
GHUNNAH_PHONEMES = {"n_G", "m_G"}

# Qalqalah burst energy
QALQALAH_PHONEMES = {"b_Q", "j_Q", "d_Q", "t_Q", "q_Q"}  # ب ج د ط ق with Qalqalah

# Heavy/Light consonants
HEAVY_PHONEMES = {"s_H", "d_H", "t_H", "z_H"}  # ص ض ط ظ; q is always heavy in classical

PHONEME_VOCABULARY: List[str] = (
    list(MADD_PHONEMES)
    + MADD_DURATION_CLASSES
    + list(GHUNNAH_PHONEMES)
    + list(QALQALAH_PHONEMES)
    + list(HEAVY_PHONEMES)
    + ["a", "i", "u", "b", "t", "j", "d", "r", "z", "s", "sh", "c", "gh", "f", "q", "k", "l", "m", "n", "h", "w", "y", "?"]
)


def predict_phonemes_from_audio(
    audio: np.ndarray,
    sr: int,
    reference_phonemes: Optional[List[str]] = None,
    reference_word_boundaries: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    """
    Predict phoneme sequence directly from audio (no ASR text).
    Stub: returns empty/None until a phoneme CTC or pronunciation model is trained/integrated.

    Returns:
        - phoneme_sequence: List[str] (empty if model not available)
        - frame_confidence: List[float] per-frame confidence (empty if N/A)
        - pronunciation_deviation_score: float 0–1 (1 = no deviation; 0 = max deviation). None if N/A.
        - available: bool (False when using stub)
    """
    # Stub: no trained model yet
    return {
        "phoneme_sequence": [],
        "frame_confidence": [],
        "pronunciation_deviation_score": None,
        "available": False,
        "message": "Phoneme-from-audio model not yet trained. Pronunciation scoring uses fallback (diacritized text or unavailable).",
    }


def pronunciation_score_from_audio(
    audio: np.ndarray,
    sr: int,
    reference_phonemes: List[str],
    reference_word_boundaries: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[Optional[float], Optional[float], bool]:
    """
    Compute pronunciation accuracy and deviation from audio using phoneme-from-audio model.
    Returns (phoneme_accuracy 0–1, pronunciation_deviation 0–1, model_available).
    When model not available, returns (None, None, False).
    """
    out = predict_phonemes_from_audio(
        audio, sr,
        reference_phonemes=reference_phonemes,
        reference_word_boundaries=reference_word_boundaries,
    )
    if not out.get("available"):
        return None, None, False
    # When available: compare out["phoneme_sequence"] to reference_phonemes (e.g. Levenshtein)
    # and use out["pronunciation_deviation_score"]
    per = None
    if out.get("phoneme_sequence") and reference_phonemes:
        from alignment.phoneme_alignment import align_phoneme_sequences
        align = align_phoneme_sequences(reference_phonemes, out["phoneme_sequence"])
        per = align.get("phoneme_accuracy")
    dev = out.get("pronunciation_deviation_score")
    return per, dev, True


def get_phoneme_vocabulary() -> List[str]:
    """Return full phoneme vocabulary for training or decoding."""
    return list(PHONEME_VOCABULARY)
