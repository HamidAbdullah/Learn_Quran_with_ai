"""
Arabic-optimized ASR (Wav2Vec2) for Quran recitation.
Supports Quran-finetuned models, optional 8-bit quantization, and phoneme energy analysis for Tajweed.
"""
import os
import warnings
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from typing import Dict, Any, Optional, Callable, Tuple

MIN_AUDIO_DURATION = 0.5

# Default: strong Arabic ASR. Quran-finetuned: rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v2
DEFAULT_ASR_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"


def _load_audio(path: str, sr: Optional[int] = None):
    """Load audio with warnings suppressed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", FutureWarning)
        if sr is not None:
            return librosa.load(path, sr=sr)
        return librosa.load(path)


def load_audio_for_alignment(path: str, sr: int = 16000):
    """Load 16 kHz mono for Wav2Vec2/alignment. Returns (y, sr)."""
    return _load_audio(path, sr=sr)


class PhoneticAnalyzer:
    """
    Arabic-optimized ASR (Wav2Vec2 XLSR) for Quran recitation.
    Used for dual-ASR (secondary) and CTC alignment; supports quantization and torch.compile.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        quantize_8bit: bool = False,
        use_torch_compile: bool = False,
    ):
        self.model_name = (
            model_name
            or os.environ.get("WAV2VEC2_MODEL")
            or os.environ.get("ASR_MODEL")
            or DEFAULT_ASR_MODEL
        )
        self.device = device or (os.environ.get("DEVICE") or "auto")
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.quantize_8bit = quantize_8bit or (
            os.environ.get("WAV2VEC2_QUANTIZE_8BIT", "").lower() in ("1", "true", "yes")
        )
        self.use_torch_compile = use_torch_compile or (
            os.environ.get("TORCH_COMPILE", "").lower() in ("1", "true", "yes")
        )
        print(f"Initializing PhoneticAnalyzer ({self.model_name}) on {self.device} (8bit={self.quantize_8bit}, compile={self.use_torch_compile})...")
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name, use_safetensors=True)
        if self.quantize_8bit:
            try:
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
            except Exception:
                pass
        self.model = self.model.to(self.device)
        self.model.eval()
        if self.use_torch_compile and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception:
                pass

    def transcribe(self, audio_path: str) -> Optional[str]:
        """Transcribe audio to Arabic text. Returns None if too short or on error."""
        try:
            speech, sr = _load_audio(audio_path, sr=16000)
            if len(speech) / sr < MIN_AUDIO_DURATION:
                return None
            input_values = self.processor(
                speech, return_tensors="pt", sampling_rate=16000
            ).input_values.to(self.device)
            with torch.no_grad():
                logits = self.model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)[0]
            return self.processor.decode(predicted_ids)
        except Exception:
            return None

    def run_forward(self, speech: np.ndarray, sr: int = 16000) -> Tuple[Optional[str], Optional[torch.Tensor]]:
        """
        Single forward pass: return (transcription, logits). Reuse logits for alignment and Tajweed.
        Returns (None, None) if audio too short or on error.
        """
        try:
            if len(speech) / sr < MIN_AUDIO_DURATION:
                return None, None
            input_values = self.processor(
                speech, return_tensors="pt", sampling_rate=sr
            ).input_values.to(self.device)
            with torch.no_grad():
                logits = self.model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)[0]
            transcription = self.processor.decode(predicted_ids)
            return transcription, logits
        except Exception:
            return None, None

    def analyze_alignment_from_logits(self, logits: torch.Tensor) -> Dict[str, Any]:
        """Same structure as analyze_alignment but from precomputed logits (avoids extra forward pass)."""
        out = {"transcription": "", "confidence_avg": 0.0, "frames": []}
        try:
            if logits is None or logits.dim() < 2:
                return out
            logits = logits.to(self.device)
            predicted_ids = torch.argmax(logits, dim=-1)[0]
            out["transcription"] = self.processor.decode(predicted_ids)
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
            max_probs, _ = torch.max(probs, dim=-1)
            out["confidence_avg"] = float(torch.mean(max_probs))
            out["frames"] = max_probs.cpu().numpy().tolist()
        except Exception:
            pass
        return out

    def analyze_alignment(self, audio_path: str, transcript: str) -> Dict[str, Any]:
        """Returns transcription + frame-level confidence for Tajweed. Safe defaults on error."""
        out = {"transcription": "", "confidence_avg": 0.0, "frames": []}
        try:
            speech, sr = _load_audio(audio_path, sr=16000)
            if len(speech) / sr < MIN_AUDIO_DURATION:
                return out
            input_values = self.processor(
                speech, return_tensors="pt", sampling_rate=16000
            ).input_values.to(self.device)
            with torch.no_grad():
                logits = self.model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)[0]
            out["transcription"] = self.processor.decode(predicted_ids)
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
            max_probs, _ = torch.max(probs, dim=-1)
            out["confidence_avg"] = float(torch.mean(max_probs))
            out["frames"] = max_probs.cpu().numpy().tolist()
        except Exception:
            pass
        return out

    def get_phonetic_features_from_audio(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Acoustic features from preloaded (y, sr). Avoids reloading the same file."""
        default = {
            "intensity": [],
            "spectral_rolloff": [],
            "duration": 0.0,
            "sample_rate": 22050,
            "frame_times": [],
        }
        try:
            if len(y) == 0:
                return default
            duration = float(librosa.get_duration(y=y, sr=sr))
            rms = librosa.feature.rms(y=y)[0]
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            hop = 512
            n_frames = rms.shape[0]
            frame_times = (np.arange(n_frames) * hop / sr).tolist()
            return {
                "intensity": rms.tolist(),
                "spectral_rolloff": rolloff.tolist(),
                "duration": duration,
                "sample_rate": sr,
                "frame_times": frame_times,
            }
        except Exception:
            return default

    def get_phonetic_features(self, audio_path: str) -> Dict[str, Any]:
        """
        Acoustic features for Tajweed: RMS intensity, spectral rolloff, duration.
        Used for Qalqalah (intensity gradient), Madd (duration), Ghunnah (spectral band).
        """
        default = {
            "intensity": [],
            "spectral_rolloff": [],
            "duration": 0.0,
            "sample_rate": 22050,
            "frame_times": [],
        }
        try:
            y, sr = _load_audio(audio_path)
            return self.get_phonetic_features_from_audio(y, sr)
        except Exception:
            return default

    def get_phoneme_energy_analysis(
        self, audio_path: str, frame_times: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Phoneme-level energy analysis: intensity gradient (for Qalqalah), duration ratios.
        frame_times: optional list of segment boundaries (seconds) to compute per-segment energy.
        """
        out = {
            "intensity_gradient": [],
            "duration_seconds": 0.0,
            "segment_energies": [],
        }
        try:
            feats = self.get_phonetic_features(audio_path)
            intensity = np.array(feats["intensity"])
            out["duration_seconds"] = feats["duration"]
            if len(intensity) > 1:
                out["intensity_gradient"] = np.diff(intensity).tolist()
            if frame_times and len(frame_times) >= 2:
                ft = np.array(frame_times)
                sr_feat = len(intensity) / (feats["duration"] or 1e-6)
                seg_energies = []
                for i in range(len(frame_times) - 1):
                    t0, t1 = frame_times[i], frame_times[i + 1]
                    i0 = int(t0 * sr_feat)
                    i1 = min(int(t1 * sr_feat), len(intensity))
                    if i1 > i0:
                        seg_energies.append(float(np.mean(intensity[i0:i1])))
                    else:
                        seg_energies.append(0.0)
                out["segment_energies"] = seg_energies
        except Exception:
            pass
        return out
