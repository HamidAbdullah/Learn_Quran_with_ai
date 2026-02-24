"""
Phoneme extraction layer for Phase 3 — Phoneme Alignment Engine.

Rule-based mapping from Arabic script to phoneme sequences. Supports:
- Madd detection (long vowels: aa, ii, uu)
- Qalqalah letters (ب ج د ط ق when saakin)
- Ghunnah detection (nasalization on ن and م in specific contexts)
- Heavy vs light letters (ص ض ط ظ ق vs their light counterparts)

No deep learning; used for phoneme-level alignment and Tajweed rule detection.
"""
import re
from typing import List, Dict, Any, Optional, Tuple

# Unicode ranges for Arabic diacritics (harakat)
FATHA = "\u064B"  # َ
KASRA = "\u064D"  # ِ
DAMMA = "\u064C"  # ُ
SUKUN = "\u0652"  # ْ
SHADDA = "\u0651"  # ّ
# Combined short vowels (with tanween etc.)
SHORT_A = "\u064E"  # fatha
SHORT_I = "\u0650"  # kasra
SHORT_U = "\u064F"  # damma

# Single-letter to phoneme (consonants). Heavy letters get _H suffix for rule detection.
_ARABIC_CONSONANT: Dict[str, str] = {
    "\u0627": "?",   # alif
    "\u0623": "?",   # alif hamza above
    "\u0625": "?",   # alif hamza below
    "\u0622": "?",   # alif madda
    "\u0621": "?",   # hamza
    "\u0628": "b",
    "\u062A": "t",
    "\u062B": "th",
    "\u062C": "j",
    "\u062D": "h",   # ha heavy
    "\u062E": "kh",
    "\u062F": "d",
    "\u0630": "dh",
    "\u0631": "r",
    "\u0632": "z",
    "\u0633": "s",
    "\u0634": "sh",
    "\u0635": "s_H",  # sad heavy
    "\u0636": "d_H",  # dad heavy
    "\u0637": "t_H",  # ta heavy
    "\u0638": "z_H",  # za heavy
    "\u0639": "c",   # ayn
    "\u063A": "gh",
    "\u0641": "f",
    "\u0642": "q",
    "\u0643": "k",
    "\u0644": "l",
    "\u0645": "m",
    "\u0646": "n",
    "\u0647": "h",
    "\u0648": "w",
    "\u064A": "y",
    "\u0629": "h",   # ta marbuta
    "\u0640": "",    # tatweel (elongation) — skip
}

# Qalqalah letters (when saakin): ب ج د ط ق — we tag as _Q for rule detection
_QALQALAH = {"b", "j", "d", "t_H", "q"}
# Ghunnah: ن and م in idghaam/ikhfa contexts — tag as _G
# Heavy letters (for heavy/light rules)
_HEAVY = {"s_H", "d_H", "t_H", "z_H", "q"}

# Normalize common variants to base letter for lookup
_NORMALIZE_LETTER = str.maketrans(
    "إأآءئؤىیكکة",
    "ااااييويككه"
)


def _get_diacritic_type(char: str) -> Optional[str]:
    """Return 'fatha', 'kasra', 'damma', 'sukun', 'shadda', or None."""
    if char in ("\u064E", "\u064B"):  # fatha, tanwin fath
        return "fatha"
    if char in ("\u0650", "\u064D"):  # kasra, tanwin kasr
        return "kasra"
    if char in ("\u064F", "\u064C"):  # damma, tanwin damm
        return "damma"
    if char == "\u0652":
        return "sukun"
    if char == "\u0651":
        return "shadda"
    return None


def _is_letter(c: str) -> bool:
    """True if c is an Arabic letter (not diacritic)."""
    return "\u0621" <= c <= "\u064A" or c in "\u0622\u0623\u0625\u0671"


def arabic_to_phonemes(word: str) -> List[str]:
    """
    Convert a single Arabic word to a list of phoneme symbols (rule-based).

    Handles:
    - Consonants with heavy/light (e.g. ص → s_H)
    - Short vowels from diacritics (fatha→a, kasra→i, damma→u)
    - Madd: long vowels (alif after fatha → aa, ya after kasra → ii, waw after damma → uu)
    - Sukun: no vowel after that letter
    - Shadda: gemination (consonant doubled in output)
    - Qalqalah: ب ج د ط ق with sukun get _Q suffix for rule layer
    - Ghunnah: ن/م in nasal context get _G suffix

    Args:
        word: Arabic word, optionally with diacritics (e.g. بِسْمِ).

    Returns:
        List of phoneme strings, e.g. ["b", "i", "s", "m", "i"].
    """
    if not word or not word.strip():
        return []
    word = word.strip()
    out: List[str] = []
    i = 0
    last_vowel: Optional[str] = None  # "a", "i", "u" for Madd

    while i < len(word):
        c = word[i]
        # Tatweel: skip
        if c == "\u0640":
            i += 1
            continue
        # Diacritic
        diac = _get_diacritic_type(c)
        if diac:
            if diac == "shadda" and out:
                # Double previous consonant
                out.append(out[-1])
            elif diac == "sukun":
                last_vowel = None
            elif diac == "fatha":
                last_vowel = "a"
            elif diac == "kasra":
                last_vowel = "i"
            elif diac == "damma":
                last_vowel = "u"
            i += 1
            continue

        if not _is_letter(c):
            i += 1
            continue

        norm = c.translate(_NORMALIZE_LETTER)
        phon = _ARABIC_CONSONANT.get(norm) or _ARABIC_CONSONANT.get(c)
        if not phon:
            i += 1
            continue

        # Look ahead for vowel (next char might be diacritic)
        next_vowel: Optional[str] = None
        has_sukun = False
        j = i + 1
        while j < len(word) and not _is_letter(word[j]) and word[j] != "\u0640":
            d = _get_diacritic_type(word[j])
            if d == "sukun":
                has_sukun = True
            elif d == "fatha":
                next_vowel = "a"
            elif d == "kasra":
                next_vowel = "i"
            elif d == "damma":
                next_vowel = "u"
            j += 1

        # Madd: long vowel when letter is alif/ya/waw and carries elongation
        if norm == "\u0627" and last_vowel == "a":  # alif after fatha
            out.append("aa")
            last_vowel = None
            i += 1
            continue
        if norm == "\u064A" and last_vowel == "i":  # ya after kasra
            out.append("ii")
            last_vowel = None
            i += 1
            continue
        if norm == "\u0648" and last_vowel == "u":  # waw after damma
            out.append("uu")
            last_vowel = None
            i += 1
            continue

        # Consonant
        if phon != "?":  # skip standalone hamza as phoneme for simplicity, or add "?"
            # Qalqalah tag when saakin
            if has_sukun and phon in _QALQALAH:
                out.append(phon + "_Q")
            elif phon in ("n", "m") and has_sukun:
                # Ghunnah context (simplified: any nun/mim saakin)
                out.append(phon + "_G")
            else:
                out.append(phon)
        if last_vowel and phon != "?":
            out.append(last_vowel)
            last_vowel = None
        i += 1
    if last_vowel:
        out.append(last_vowel)
    return out


def verse_to_phoneme_sequence(text: str) -> Dict[str, Any]:
    """
    Convert a full verse (multiple words) to a flat phoneme sequence and word boundaries.

    Used by the phoneme alignment module to align reference vs hypothesis at phoneme level,
    and by the Tajweed rules engine to map errors back to words.

    Args:
        text: Arabic verse text (e.g. Uthmani with diacritics), space-separated words.

    Returns:
        {
            "phoneme_sequence": List[str],   # flat list of phonemes
            "word_boundaries": List[Tuple[int, int]],  # (start_idx, end_idx) per word into phoneme_sequence
            "words": List[str],              # original words
            "phonemes_per_word": List[List[str]],  # phonemes per word
        }
    """
    if not text or not text.strip():
        return {
            "phoneme_sequence": [],
            "word_boundaries": [],
            "words": [],
            "phonemes_per_word": [],
        }
    words = text.strip().split()
    phonemes_per_word: List[List[str]] = []
    for w in words:
        p = arabic_to_phonemes(w)
        phonemes_per_word.append(p)
    flat: List[str] = []
    boundaries: List[Tuple[int, int]] = []
    start = 0
    for pw in phonemes_per_word:
        end = start + len(pw)
        flat.extend(pw)
        boundaries.append((start, end))
        start = end
    return {
        "phoneme_sequence": flat,
        "word_boundaries": boundaries,
        "words": words,
        "phonemes_per_word": phonemes_per_word,
    }


def get_qalqalah_phonemes() -> List[str]:
    """Return phoneme symbols that denote Qalqalah (for rule detection)."""
    return [p + "_Q" for p in _QALQALAH]


def get_heavy_phonemes() -> List[str]:
    """Return phoneme symbols for heavy letters (for rule detection)."""
    return list(_HEAVY)


def get_ghunnah_phonemes() -> List[str]:
    """Return phoneme symbols that denote Ghunnah (for rule detection)."""
    return ["n_G", "m_G"]
