import os
import json

def index_audio():
    audio_dir = "/Users/hamid/Downloads/000_versebyverse/"
    quran_json_path = "/Users/hamid/Desktop/KivyxProjects/quran-ai-backend/quran_full.json"
    
    if not os.path.exists(quran_json_path):
        print(f"Error: {quran_json_path} not found.")
        return
        
    with open(quran_json_path, 'r', encoding='utf-8') as f:
        quran_data = json.load(f)
        
    audio_files = os.listdir(audio_dir)
    audio_mapping = {}
    
    # Pre-map available files for efficiency
    # Format: 001001.mp3 -> (1, 1)
    available_audio = {}
    for filename in audio_files:
        if filename.endswith(".mp3"):
            name_part = filename.replace(".mp3", "")
            if len(name_part) == 6:
                try:
                    surah_num = int(name_part[:3])
                    ayah_num = int(name_part[3:])
                    available_audio[f"{surah_num}:{ayah_num}"] = filename
                except ValueError:
                    continue
                    
    results = []
    missing_count = 0
    
    for entry in quran_data:
        verse_key = entry.get("verse_key")
        if verse_key in available_audio:
            entry["audio_file"] = available_audio[verse_key]
        else:
            entry["audio_file"] = None
            missing_count += 1
            
    output_path = "/Users/hamid/Desktop/KivyxProjects/quran-ai-backend/quran_with_audio.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(quran_data, f, ensure_ascii=False, indent=4)
        
    print(f"Indexed {len(quran_data)} verses.")
    print(f"Mapped: {len(quran_data) - missing_count}")
    print(f"Missing audio: {missing_count}")
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    index_audio()
