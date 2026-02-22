# Production & Integration Guide: Quran AI

## 🛠️ Tech Stack: What we used and Why

We selected a "Heavyweight AI + Lightweight API" stack to ensure maximum accuracy with real-time performance.

| Package | Purpose | Why? |
| :--- | :--- | :--- |
| **FastAPI** | Backend Framework | High-performance, native support for WebSockets (essential for real-time). |
| **OpenAI Whisper** | ASR (Speech-to-Text) | Best-in-class Arabic transcription. We use `base` for a balance of speed/accuracy. |
| **Transformers** | AI Model Loader | Standard for loading Wav2Vec2 and other pre-trained HuggingFace models. |
| **Wav2Vec2** | Phonemic Alignment | Essential for 100% accuracy matching. It checks pronunciation at the letter level. |
| **Librosa / Soundfile** | Audio Processing | Used to extract acoustic features (intensity/pitch) for Tajweed verification. |
| **RapidFuzz / Difflib** | Fuzzy Matching | Compares the user's speech against the Quranic script, ignoring diacritics. |
| **Uvicorn** | Server Engine | An ASGI server that allows the app to handle many simultaneous WebSocket connections. |

---

## 📱 Integration with "Islam Encyclo"

This system is designed as a **Microservice Architecture**. You can integrate it into your React Native app (Islam Encyclo) easily:

### 1. Real-Time Feedback (Recommended)
Use a WebSocket library in React Native (like `react-native-websocket` or native `WebSocket`).
- **Endpoint**: `ws://your-server-ip:8000/ws/verify?surah=1&ayah=6`
- **Logic**: Send audio chunks every 1 second. The backend will send back partial transcription JSONs instantly.

### 2. Final Analysis (Post-Recitation)
Use a standard `POST` request.
- **Endpoint**: `http://your-server-ip:8000/verify?surah=1&ayah=6`
- **Body**: `FormData` containing the audio file.

---

## 🚀 Is it ready for Production?

It is currently **"Launch Candidate 1"**. The core intelligence and stability are production-ready, but for a high-traffic app, you should:

1.  **Authentication**: Add JWT tokens to the endpoints so only your app users can access the AI.
2.  **GPU Acceleration**: Currently running on CPU. For many users, move it to an NVIDIA GPU (change `.to("cpu")` to `.to("cuda")` in the code).
3.  **Caching**: Cache common Surah transcriptions to save processing time.

---

## 📜 Git History Note: "The Refactoring"
You may see many deletions in the git history. This is because we moved from **Phase 1 (Basic Script)** to **Phase 4 (Production Engine)**.
- We replaced simple synchronous code with complex asynchronous WebSocket logic.
- We added a layer-based intelligence system which required restructuring the `core/` folder.
- **Result**: The code is now much smaller, faster, and more modular than the early experimental versions.
