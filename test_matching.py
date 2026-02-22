import re
from difflib import SequenceMatcher
from rapidfuzz import fuzz

def normalize_arabic(text):
    if not text:
        return ""
    # Remove all diacritics
    text = re.sub(r'[\u064B-\u065F\u0670\u06D6-\u06ED]', '', text)
    # Normalize Alif varieties
    text = re.sub(r'[إأآا]', 'ا', text)
    # Normalize Ya and Alef Maksura to a single form
    text = re.sub(r'[يىیئ]', 'ی', text)
    # Normalize Kaf
    text = re.sub(r'ك', 'ک', text)
    # Remove digits
    text = re.sub(r'[0-9۰-۹]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def align_words(original_text_raw, user_text_norm):
    original_text_norm = normalize_arabic(original_text_raw)
    original_words = original_text_norm.split()
    original_words_raw = original_text_raw.split()
    user_words = user_text_norm.split()

    matcher = SequenceMatcher(None, original_words, user_words)
    words_result = []
    correct_count = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for i in range(i1, i2):
                words_result.append({
                    "original": original_words_raw[i],
                    "user": user_words[j1 + (i - i1)],
                    "status": "correct",
                    "similarity": 100.0
                })
                correct_count += 1
        elif tag == 'replace':
            orig_range = original_words[i1:i2]
            user_range = user_words[j1:j2]
            for i in range(max(len(orig_range), len(user_range))):
                if i < len(orig_range) and i < len(user_range):
                    sim = fuzz.ratio(orig_range[i], user_range[i])
                    status = "correct" if sim > 90 else "minor_mistake" if sim > 70 else "wrong"
                    if status == "correct": correct_count += 1
                    words_result.append({
                        "original": original_words_raw[i1 + i],
                        "user": user_range[i],
                        "status": status,
                        "similarity": round(sim, 2)
                    })
                elif i < len(orig_range):
                    words_result.append({
                        "original": original_words_raw[i1 + i],
                        "user": "",
                        "status": "missing",
                        "similarity": 0
                    })
                elif i < len(user_range):
                    words_result.append({
                        "original": "",
                        "user": user_range[i],
                        "status": "extra",
                        "similarity": 0
                    })
        elif tag == 'delete':
            for i in range(i1, i2):
                words_result.append({
                    "original": original_words_raw[i],
                    "user": "",
                    "status": "missing",
                    "similarity": 0
                })
        elif tag == 'insert':
            for j in range(j1, j2):
                words_result.append({
                    "original": "",
                    "user": user_words[j],
                    "status": "extra",
                    "similarity": 0
                })
    
    accuracy = (correct_count / len(original_words)) * 100 if original_words else 0
    return words_result, accuracy

# Test Case 1: Perfect Match
print("--- Test Case 1: Perfect Match ---")
orig = "بِسْمِ اللّٰهِ الرَّحْمٰنِ الرَّحِیْمِ"
user = normalize_arabic("باسم الله الرحمن الرحيم") # Simplified spelling often from STT
res, acc = align_words(orig, user)
for r in res: print(r)
print(f"Accuracy: {acc}%")

# Test Case 2: Missing Word
print("\n--- Test Case 2: Missing Word ---")
user_missing = normalize_arabic("بسم الله الرحيم") # "الرحمن" missing
res, acc = align_words(orig, user_missing)
for r in res: print(r)
print(f"Accuracy: {acc}%")

# Test Case 3: Minor Mistake & Extra Word
print("\n--- Test Case 3: Minor Mistake & Extra Word ---")
user_mix = normalize_arabic("بسم الله الرحمان الرحيم ماشاءالله") # "الرحمن" -> "الرحمان" (minor/correct depending on fuzz), plus extra
res, acc = align_words(orig, user_mix)
for r in res: print(r)
print(f"Accuracy: {acc}%")
