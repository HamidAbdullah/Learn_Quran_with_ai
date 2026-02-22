import requests
import json
import os

def final_verification():
    url = "http://localhost:8000/verify"
    params = {"surah": 108, "ayah": 1}
    audio_path = "/Users/hamid/Downloads/000_versebyverse/108001.mp3"
    
    if not os.path.exists(audio_path):
        print(f"Error: {audio_path} not found.")
        return

    print("--- Starting Production Verification ---")
    print(f"Testing Surah {params['surah']}, Ayah {params['ayah']}...")
    
    with open(audio_path, "rb") as f:
        files = {"audio": (os.path.basename(audio_path), f, "audio/mpeg")}
        try:
            response = requests.post(url, params=params, files=files, timeout=120)
            if response.status_code == 200:
                print("✅ API Success!")
                data = response.json()
                print(f"Confidence Level: {data['confidence_level'].upper()}")
                print(f"Accuracy Score: {data['accuracy_score']}%")
                print(f"Tajweed Score: {data['tajweed_score']}%")
                print(f"\nFeedback: {data['teacher_feedback_text']}")
                
                print("\n💡 Teacher Tips (Word-level):")
                for item in data['word_analysis'][:3]: # Show first 3 words
                    print(f"  {item['word']} ({item['status']}): {item['feedback']}")
                
                print("\n📖 Tajweed Specifics:")
                for fb in data['tajweed_feedback']:
                    print(f"  [{fb['rule']}] Score: {int(fb['score']*100)}% - {fb['feedback']}")
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"❌ Verification failed: {e}")

if __name__ == "__main__":
    final_verification()
