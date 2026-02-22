import re

def normalize_arabic(text):
    if not text:
        return ""
    
    # Remove Tatweel (ـ)
    text = re.sub(r'\u0640', '', text)
    
    # Remove all diacritics / harakat and Uthmani marks
    # \u064B-\u065F: Standard Arabic diacritics
    # \u0670: Superscript Alif
    # \u06D6-\u06DC: Small Quranic marks
    # \u06DF-\u06E8: Small Quranic marks (Round zero, open zero, etc)
    # \u06EA-\u06ED: Empty center letters/marks
    text = re.sub(r'[\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]', '', text)
    
    # Normalize Alif varieties
    text = re.sub(r'[إأآاٱٰ]', 'ا', text)
    
    # Normalize Ya and Alef Maksura
    text = re.sub(r'[يىیئ]', 'ي', text)
    
    # Normalize Teh Marbuta
    text = re.sub(r'ة', 'ه', text)
    
    # Normalize Waw varieties
    text = re.sub(r'[ؤو]', 'و', text)
    
    # Remove digits (Western and Arabic-Indic)
    text = re.sub(r'[0-9\u0660-\u0669\u06F0-\u06F9]', '', text)
    
    # Remove punctuation and special characters
    text = re.sub(r'[^\u0621-\u064A\s]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()
