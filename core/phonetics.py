"""
Arabic-optimized ASR (Wav2Vec2) for Quran recitation.
Supports Quran-finetuned models, beam search decoding (production), optional 8-bit quantization.
Single forward pass; emissions reused for alignment (no duplicate forward).
"""
import logging
import os
import warnings
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from typing import Dict, Any, Optional, Callable, Tuple, List

MIN_AUDIO_DURATION = 0.5
SAMPLING_RATE = 16000

# Production default: Quran-finetuned model only
DEFAULT_ASR_MODEL = "rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v2"

logger = logging.getLogger(__name__)


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


def _ctc_beam_search_decode(
    log_probs: torch.Tensor,
    processor: Any,
    vocab_size: int,
    blank_id: int = 0,
    beam_width: int = 50,
) -> str:
    """
    CTC prefix beam search. Decode log_probs (T, V) to best transcript.
    Replaces greedy argmax for lower WER in production. Uses processor.decode for correct output.
    """
    if beam_width <= 1 or log_probs.size(0) == 0:
        return ""
    T, V = log_probs.size(0), log_probs.size(1)
    V = min(V, vocab_size)
    log_probs = log_probs.cpu().numpy()
    beams = [((), blank_id, 0.0)]
    for t in range(T):
        lp_t = log_probs[t]
        next_beams: Dict[tuple, float] = {}
        for (ids, last, log_p) in beams:
            key_blank = (ids, last)
            next_beams[key_blank] = max(next_beams.get(key_blank, -1e9), log_p + float(lp_t[blank_id]))
            for v in range(V):
                if v == blank_id:
                    continue
                lp_v = float(lp_t[v])
                if v == last:
                    key = (ids, v)
                else:
                    key = (ids + (v,), v)
                next_beams[key] = max(next_beams.get(key, -1e9), log_p + lp_v)
        sorted_beams = sorted(next_beams.items(), key=lambda x: -x[1])[: beam_width]
        beams = [(k[0], k[1], p) for k, p in sorted_beams]
    if not beams:
        return ""
    best_ids, _, _ = max(beams, key=lambda x: x[2])
    if not best_ids:
        return ""
    collapsed: List[int] = []
    for i in best_ids:
        if i == blank_id:
            continue
        if not collapsed or collapsed[-1] != i:
            collapsed.append(i)
    if not collapsed:
        return ""
    return processor.decode(collapsed).strip()


def _get_decoder_vocab(processor: Any) -> Tuple[int, int]:
    """Return (blank_id, vocab_size) from Wav2Vec2 processor for CTC beam search."""
    vocab = getattr(processor.tokenizer, "get_vocab", None)
    if vocab is None:
        vocab = getattr(processor.tokenizer, "encoder", None) or {}
    if callable(vocab):
        vocab = vocab()
    blank_id = getattr(processor.tokenizer, "pad_token_id", None) or 0
    vocab_size = len(vocab)
    return blank_id, vocab_size


class PhoneticAnalyzer:
    """
    Arabic-optimized ASR (Wav2Vec2) for Quran recitation.
    Production: beam search decoding, 16 kHz, emissions reused for alignment (no duplicate forward).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        quantize_8bit: bool = False,
        use_torch_compile: bool = False,
        beam_width: Optional[int] = None,
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
        self.beam_width = beam_width if beam_width is not None else int(os.environ.get("BEAM_WIDTH", "50"))
        print(
            f"Initializing PhoneticAnalyzer ({self.model_name}) on {self.device} "
            f"(8bit={self.quantize_8bit}, compile={self.use_torch_compile}, beam_width={self.beam_width})..."
        )
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name, use_safetensors=True)
        self._blank_id, self._vocab_size = _get_decoder_vocab(self.processor)
        if self.beam_width > 1:
            logger.info("Wav2Vec2 decoding: beam_search beam_width=%d (production accuracy mode)", self.beam_width)
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
        """Transcribe audio to Arabic text (16 kHz). Beam search when beam_width > 1. Returns None if too short or on error."""
        try:
            speech, sr = _load_audio(audio_path, sr=SAMPLING_RATE)
            if len(speech) / sr < MIN_AUDIO_DURATION:
                return None
            input_values = self.processor(
                speech, return_tensors="pt", sampling_rate=SAMPLING_RATE
            ).input_values.to(self.device)
            with torch.no_grad():
                logits = self.model(input_values).logits
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)[0]
            if self.beam_width > 1:
                return _ctc_beam_search_decode(
                    log_probs, self.processor, self._vocab_size, self._blank_id, self.beam_width
                )
            predicted_ids = torch.argmax(logits, dim=-1)[0]
            return self.processor.decode(predicted_ids)
        except Exception:
            return None

    def run_forward(self, speech: np.ndarray, sr: int = SAMPLING_RATE) -> Tuple[Optional[str], Optional[torch.Tensor]]:
        """
        Single forward pass: return (transcription, logits). Reuse logits for alignment and Tajweed.
        Decoding: beam search when beam_width > 1 (production). Sampling rate must be 16 kHz for Wav2Vec2.
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
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)[0]
            if self.beam_width > 1:
                transcription = _ctc_beam_search_decode(
                    log_probs, self.processor, self._vocab_size, self._blank_id, self.beam_width
                )
            else:
                predicted_ids = torch.argmax(logits, dim=-1)[0]
                transcription = self.processor.decode(predicted_ids)
            return transcription, logits
        except Exception:
            return None, None

    def analyze_alignment_from_logits(self, logits: torch.Tensor) -> Dict[str, Any]:
        """Same structure as analyze_alignment but from precomputed logits (avoids extra forward pass). Uses same decoding as run_forward (beam when beam_width > 1)."""
        out = {"transcription": "", "confidence_avg": 0.0, "frames": []}
        try:
            if logits is None or logits.dim() < 2:
                return out
            logits = logits.to(self.device)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)[0]
            if self.beam_width > 1:
                out["transcription"] = _ctc_beam_search_decode(
                    log_probs, self.processor, self._vocab_size, self._blank_id, self.beam_width
                )
            else:
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
        """Returns transcription + frame-level confidence for Tajweed. Uses beam decode when beam_width > 1."""
        out = {"transcription": "", "confidence_avg": 0.0, "frames": []}
        try:
            speech, sr = _load_audio(audio_path, sr=SAMPLING_RATE)
            if len(speech) / sr < MIN_AUDIO_DURATION:
                return out
            input_values = self.processor(
                speech, return_tensors="pt", sampling_rate=SAMPLING_RATE
            ).input_values.to(self.device)
            with torch.no_grad():
                logits = self.model(input_values).logits
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)[0]
            if self.beam_width > 1:
                out["transcription"] = _ctc_beam_search_decode(
                    log_probs, self.processor, self._vocab_size, self._blank_id, self.beam_width
                )
            else:
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
