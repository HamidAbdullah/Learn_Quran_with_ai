#!/usr/bin/env python3
"""Quick test for POST /verify. Run server first: python main.py"""
import json
import os
import sys

import requests

# Default port from config
PORT = int(os.environ.get("PORT", "8001"))
URL = f"http://localhost:{PORT}/verify"

# Default: Surah 1 Ayah 1; override with env or edit
AUDIO_PATH = os.environ.get("TEST_AUDIO_PATH", "/Users/hamid/Downloads/000_versebyverse/001001.mp3")
SURAH = int(os.environ.get("TEST_SURAH", "1"))
AYAH = int(os.environ.get("TEST_AYAH", "1"))


def test_verification():
    if not os.path.exists(AUDIO_PATH):
        print(f"Error: Audio file not found: {AUDIO_PATH}")
        print("Set TEST_AUDIO_PATH to a valid path, or use e.g.:")
        print("  TEST_AUDIO_PATH=/path/to/001001.mp3 python scripts/test_api.py")
        return 1
    print(f"POST {URL}  surah={SURAH} ayah={AYAH}  audio={AUDIO_PATH}")
    with open(AUDIO_PATH, "rb") as f:
        files = {"audio": (os.path.basename(AUDIO_PATH), f, "audio/mpeg")}
        try:
            response = requests.post(URL, params={"surah": SURAH, "ayah": AYAH}, files=files, timeout=120)
            if response.status_code == 200:
                data = response.json()
                print("Success!")
                print("quality_score:", data.get("quality_score"))
                print("accuracy_score:", data.get("accuracy_score"))
                print("explanation:", data.get("explanation"))
                print("model_limitations:", data.get("model_limitations"))
                print("\nFull JSON:")
                print(json.dumps(data, indent=2, ensure_ascii=False))
            else:
                print(f"Error {response.status_code}: {response.text}")
            return 0 if response.status_code == 200 else 1
        except Exception as e:
            print(f"Request failed: {e}")
            return 1


if __name__ == "__main__":
    sys.exit(test_verification())
