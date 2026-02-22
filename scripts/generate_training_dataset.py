import json
import os
import re

def normalize_arabic(text):
    if not text:
        return ""
    # Remove all diacritics
    text = re.sub(r'[\u064B-\u065F\u0670\u06D6-\u06ED]', '', text)
    # Normalize Alif varieties
    text = re.sub(r'[إأآا]', 'ا', text)
    # Normalize Ya and Alef Maksura
    text = re.sub(r'[يىیئ]', 'ی', text)
    # Normalize Kaf
    text = re.sub(r'ك', 'ک', text)
    # Remove Quranic special marks and digits
    text = re.sub(r'[0-9۰-۹]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def generate_training_data():
    dataset_path = "/Users/hamid/Desktop/KivyxProjects/quran-ai-backend/quran_with_audio.json"
    audio_base_dir = "/Users/hamid/Downloads/000_versebyverse/"
    output_dir = "/Users/hamid/Desktop/KivyxProjects/quran-ai-backend/dataset/"
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        return
        
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    training_data = []
    
    for entry in data:
        audio_file = entry.get("audio_file")
        if not audio_file:
            continue
            
        audio_path = os.path.join(audio_base_dir, audio_file)
        # We use Uthmani text for the ground truth, but normalized for training stability
        # Depending on the model, we might want diacritics, but for core StT, 
        # normalized is often better for initial alignment.
        
        # However, for high-quality Quran StT, we might want a version WITH diacritics
        # and a version WITHOUT. Let's provide both in the JSON.
        
        normalized_text = normalize_arabic(entry.get("text_uthmani", ""))
        
        training_data.append({
            "audio": audio_path,
            "text": entry.get("text_uthmani", ""),
            "normalized_text": normalized_text,
            "verse_key": entry.get("verse_key")
        })
        
    # Save as JSON (standard for many training scripts)
    output_json = os.path.join(output_dir, "training_dataset.json")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=4)
        
    # Also save as a manifest file (CSV example)
    output_csv = os.path.join(output_dir, "manifest.csv")
    with open(output_csv, 'w', encoding='utf-8') as f:
        f.write("audio_path,text\n")
        for item in training_data:
            # Escape quotes for CSV
            text = item["normalized_text"].replace('"', '""')
            f.write(f'"{item["audio"]}","{text}"\n')
            
    print(f"Generated training dataset with {len(training_data)} samples.")
    print(f"JSON: {output_json}")
    print(f"CSV: {output_csv}")

if __name__ == "__main__":
    generate_training_data()
