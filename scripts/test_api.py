import requests
import os

def test_verification():
    url = "http://localhost:8000/verify"
    params = {
        "surah": 108,
        "ayah": 1
    }
    audio_path = "/Users/hamid/Downloads/000_versebyverse/108001.mp3"
    
    if not os.path.exists(audio_path):
        print(f"Error: {audio_path} not found.")
        return
        
    print(f"Sending request for Surah {params['surah']}, Ayah {params['ayah']}...")
    with open(audio_path, "rb") as f:
        files = {"audio": (os.path.basename(audio_path), f, "audio/mpeg")}
        try:
            response = requests.post(url, params=params, files=files, timeout=60)
            if response.status_code == 200:
                print("Success!")
                print(json.dumps(response.json(), indent=4, ensure_ascii=False))
            else:
                print(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    import json
    test_verification()
