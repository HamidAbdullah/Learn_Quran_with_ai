"""
Production configuration via environment variables.
Load with python-dotenv; no hardcoded model paths or secrets.
"""
import os
from pathlib import Path

# Load .env if present (optional in production where env is set by orchestrator)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ----- Server -----
PORT = int(os.environ.get("PORT", "8001"))
HOST = os.environ.get("HOST", "0.0.0.0")

# ----- Quran data -----
QURAN_DATA_PATH = os.environ.get("QURAN_DATA_PATH", "")
# If empty, we try quran_with_audio.json then quran.json in cwd
def get_quran_path() -> str:
    if QURAN_DATA_PATH and os.path.isfile(QURAN_DATA_PATH):
        return QURAN_DATA_PATH
    for name in ("quran_with_audio.json", "quran.json"):
        p = Path.cwd() / name
        if p.exists():
            return str(p)
    return ""

# ----- ASR models (target 2–3s verify: Wav2Vec2-only; set ASR_USE_WHISPER=true for dual) -----
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")  # base = fast; small/medium = slower, more accurate

# Wav2Vec2: MUST be Quran-finetuned (no generic Arabic)
WAV2VEC2_MODEL = os.environ.get(
    "WAV2VEC2_MODEL",
    "rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v2",
)

# false = Wav2Vec2 only (~2–3s); true = Whisper + Wav2Vec2 (slower, best WER)
ASR_USE_WHISPER = os.environ.get("ASR_USE_WHISPER", "false").lower() in ("1", "true", "yes")

# ----- ASR decoding (1 = greedy, 2–3s; 5–10 = slower, better accuracy) -----
BEAM_WIDTH = int(os.environ.get("BEAM_WIDTH", "1"))  # 1 = greedy decode for 2–3s target
WHISPER_BEAM_SIZE = int(os.environ.get("WHISPER_BEAM_SIZE", "1"))  # 1 = greedy when Whisper used

# ----- Device & optimization -----
# auto | cuda | cpu
DEVICE = os.environ.get("DEVICE", "auto")
def resolve_device() -> str:
    if DEVICE != "auto":
        return DEVICE
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

# Enable 8-bit quantization for Wav2Vec2 (reduces VRAM, slight accuracy trade-off)
WAV2VEC2_QUANTIZE_8BIT = os.environ.get("WAV2VEC2_QUANTIZE_8BIT", "false").lower() in ("1", "true", "yes")

# Torch compile for inference (PyTorch 2.0+)
TORCH_COMPILE = os.environ.get("TORCH_COMPILE", "false").lower() in ("1", "true", "yes")

# ----- Streaming (2–3 s overlapping windows; do not re-run full verse on every chunk) -----
WS_BUFFER_THRESHOLD = int(os.environ.get("WS_BUFFER_THRESHOLD", os.environ.get("WS_CHUNK_THRESHOLD_BYTES", "150000")))
WS_PARTIAL_RESULT_INTERVAL = float(os.environ.get("WS_PARTIAL_RESULT_INTERVAL", "2.5"))  # seconds between partial results
WS_CHUNK_THRESHOLD_BYTES = WS_BUFFER_THRESHOLD  # backward compat
WS_WINDOW_SECONDS = float(os.environ.get("WS_WINDOW_SECONDS", "2.5"))
WS_OVERLAP_SECONDS = float(os.environ.get("WS_OVERLAP_SECONDS", "0.5"))
# Phase 4.1: max pending ASR segments per connection; if exceeded, send server_busy and skip chunk
WS_MAX_QUEUE = int(os.environ.get("WS_MAX_QUEUE", "3"))

# ----- Verify endpoint (target 2–3s response) -----
VERIFY_TIMEOUT_SEC = float(os.environ.get("VERIFY_TIMEOUT_SEC", "10"))
# Max audio seconds to process; lower = faster (30s keeps pipeline ~2–3s)
VERIFY_MAX_AUDIO_DURATION_SEC = float(os.environ.get("VERIFY_MAX_AUDIO_DURATION_SEC", "30"))

# ----- CORS (production: set to specific origins) -----
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")
def get_cors_origins() -> list:
    """Return list of allowed CORS origins from env."""
    if CORS_ORIGINS == "*":
        return ["*"]
    return [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]
