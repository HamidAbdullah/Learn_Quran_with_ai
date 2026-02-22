import re
from typing import Optional

# Canonical script: Arabic (ي، ك). ASR often outputs Persian (ی، ک) — normalize both to one.
USE_ARABIC_SCRIPT = True  # True = ي، ك (matches most Uthmani); False = ی، ک (Indo-Pak style)


def normalize_arabic(text: Optional[str], strip_diacritics: bool = True) -> str:
    """
    Normalize Arabic/Quranic text for matching: one script, no diacritics (unless disabled).
    Handles Uthmani, Indo-Pak, and ASR output (e.g. wav2vec often outputs ی، ک).
    """
    if not text:
        return ""

    # Remove Tatweel (ـ)
    text = re.sub(r'\u0640', '', text)

    if strip_diacritics:
        # Diacritics and Quranic marks (harakat, sukun, shadda, etc.)
        text = re.sub(r'[\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]', '', text)

    # Alif / Hamza varieties → ا (ASR often uses ا for ء and vice versa)
    text = re.sub(r'[إأآاٱٰء]', 'ا', text)
    # Ya / Alef Maksura → ي (or ی if not USE_ARABIC_SCRIPT)
    text = re.sub(r'[يىیئ]', 'ي', text)
    # Teh Marbuta → ه
    text = re.sub(r'ة', 'ه', text)
    # Waw varieties
    text = re.sub(r'[ؤو]', 'و', text)
    # Kaf: Arabic ك and Persian ک → single form for matching
    text = re.sub(r'[كک]', 'ك', text)

    # Remove digits (Western and Arabic-Indic)
    text = re.sub(r'[0-9\u0660-\u0669\u06F0-\u06F9]', '', text)
    # Remove any remaining non-Arabic letters (Quranic symbols, etc.)
    text = re.sub(r'[^\u0621-\u064A\s]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()
