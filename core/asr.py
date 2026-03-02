"""
Dual ASR module: Whisper + Wav2Vec2 Arabic/Quran, best-transcript selection.
Selection: prioritize lower WER, then lower CER, then higher confidence (production accuracy).
Runs Whisper and Wav2Vec2 in parallel when both inputs are available to cut ASR latency.
"""
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np

from core.normalization import normalize_arabic
from core.scoring import segment_transcript_by_reference, count_matching_words
from core.metrics import wer, cer

logger = logging.getLogger(__name__)


def run_dual_asr(
    whisper_model: Any,
    phonetic_analyzer: Any,
    reference_text: str,
    audio_path: Optional[str] = None,
    audio_y_sr: Optional[Tuple[np.ndarray, int]] = None,
    whisper_beam_size: Optional[int] = None,
    use_whisper: bool = True,
) -> Dict[str, Any]:
    """
    Run dual ASR (Whisper + Wav2Vec2) or Wav2Vec2-only. When use_whisper=False, only Wav2Vec2 runs (~2–3s).
    Selection: lower WER then CER then confidence when both engines run.

    Args:
        whisper_model: OpenAI Whisper model (can be None if use_whisper=False).
        phonetic_analyzer: PhoneticAnalyzer (run_forward; beam_width=1 for greedy speed).
        reference_text: Verse text (Uthmani) for segmentation and selection.
        audio_path: Path to audio file.
        audio_y_sr: Optional (y, sr) preloaded audio for Wav2Vec2.
        whisper_beam_size: Optional beam size for Whisper.
        use_whisper: If False, skip Whisper and use only Wav2Vec2 (faster, 2–3s target).

    Returns:
        dict with selected_transcript, selected_source, logits, ...
    """
    ref_norm = normalize_arabic(reference_text or "")
    ref_words = [w for w in ref_norm.split() if w]

    out = {
        "selected_transcript": "",
        "selected_source": "wav2vec",
        "selection_reason": "single_engine",
        "transcript_whisper": "",
        "transcript_wav2vec": "",
        "confidence_whisper": 0.0,
        "confidence_wav2vec": 0.0,
        "match_count_whisper": 0,
        "match_count_wav2vec": 0,
        "wer_whisper": None,
        "cer_whisper": None,
        "wer_wav2vec": None,
        "cer_wav2vec": None,
        "logits": None,
    }

    whisper_text = ""
    confidence_whisper = 0.0
    whisper_result = None
    wav2vec_text = ""
    logits = None
    confidence_wav2vec = 0.0

    def _run_whisper() -> Tuple[str, float, Optional[Dict]]:
        t, c, r = "", 0.0, None
        if not audio_path or not whisper_model:
            return t, c, r
        try:
            trans_kw: Dict[str, Any] = {"language": "ar"}
            if whisper_beam_size is not None and whisper_beam_size > 0:
                trans_kw["beam_size"] = whisper_beam_size
            trans = whisper_model.transcribe(audio_path, **trans_kw)
            t = (trans.get("text") or "").strip()
            no_speech = trans.get("no_speech_prob")
            c = float(1.0 - no_speech) if no_speech is not None else (0.9 if t else 0.0)
            r = trans
        except Exception:
            pass
        return t, c, r

    def _run_wav2vec() -> Tuple[str, Any, float]:
        t, lg, c = "", None, 0.0
        if audio_y_sr is not None:
            y, sr = audio_y_sr
            result = phonetic_analyzer.run_forward(y, sr)
            if result:
                t, lg = result[0], result[1]
            if lg is not None:
                analysis = phonetic_analyzer.analyze_alignment_from_logits(lg)
                c = float(analysis.get("confidence_avg") or 0.0)
        if (not t or not t.strip()) and audio_path:
            t = phonetic_analyzer.transcribe(audio_path) or ""
            if t:
                analysis = phonetic_analyzer.analyze_alignment(audio_path, t)
                c = float(analysis.get("confidence_avg") or 0.0)
        return t or "", lg, c

    run_whisper = use_whisper and whisper_model is not None and audio_path is not None
    run_wav2vec = audio_path is not None or audio_y_sr is not None

    if run_whisper and run_wav2vec:
        with ThreadPoolExecutor(max_workers=2) as executor:
            f_whisper = executor.submit(_run_whisper)
            f_wav2vec = executor.submit(_run_wav2vec)
            whisper_text, confidence_whisper, whisper_result = f_whisper.result()
            wav2vec_text, logits, confidence_wav2vec = f_wav2vec.result()
    else:
        if run_whisper:
            whisper_text, confidence_whisper, whisper_result = _run_whisper()
        if run_wav2vec:
            wav2vec_text, logits, confidence_wav2vec = _run_wav2vec()

    out["transcript_whisper"] = whisper_text
    out["confidence_whisper"] = confidence_whisper
    out["match_count_whisper"] = count_matching_words(reference_text, whisper_text) if whisper_text else 0

    if wav2vec_text and wav2vec_text.strip().count(" ") < max(1, len(ref_words) - 2):
        wav2vec_text = segment_transcript_by_reference(wav2vec_text, reference_text)

    out["transcript_wav2vec"] = wav2vec_text or ""
    out["confidence_wav2vec"] = confidence_wav2vec
    out["match_count_wav2vec"] = count_matching_words(reference_text, wav2vec_text) if wav2vec_text else 0
    out["logits"] = logits

    # --- WER / CER vs reference (normalized Arabic) ---
    if reference_text:
        if whisper_text:
            out["wer_whisper"] = round(wer(reference_text, whisper_text), 4)
            out["cer_whisper"] = round(cer(reference_text, whisper_text), 4)
        if wav2vec_text:
            out["wer_wav2vec"] = round(wer(reference_text, wav2vec_text), 4)
            out["cer_wav2vec"] = round(cer(reference_text, wav2vec_text), 4)

    # --- Selection: (1) lower WER  (2) then lower CER  (3) then higher confidence ---
    if whisper_text and wav2vec_text:
        w_wer = out["wer_whisper"] if out["wer_whisper"] is not None else 1.0
        w_cer = out["cer_whisper"] if out["cer_whisper"] is not None else 1.0
        v_wer = out["wer_wav2vec"] if out["wer_wav2vec"] is not None else 1.0
        v_cer = out["cer_wav2vec"] if out["cer_wav2vec"] is not None else 1.0
        # Prefer lower WER, then lower CER, then higher confidence
        pick_wav2vec = (
            v_wer < w_wer
            or (v_wer == w_wer and v_cer < w_cer)
            or (v_wer == w_wer and v_cer == w_cer and confidence_wav2vec >= confidence_whisper)
        )
        if pick_wav2vec:
            out["selected_transcript"] = wav2vec_text
            out["selected_source"] = "wav2vec"
            out["selection_reason"] = "wer_cer_confidence"
        else:
            out["selected_transcript"] = whisper_text
            out["selected_source"] = "whisper"
            out["selection_reason"] = "wer_cer_confidence"
    else:
        out["selected_transcript"] = whisper_text or wav2vec_text
        if whisper_text and not wav2vec_text:
            out["selected_source"] = "whisper"
        else:
            out["selected_source"] = "wav2vec"
        out["selection_reason"] = "single_engine"

    # --- Log selection for production observability ---
    logger.info(
        "ASR selection: Whisper WER=%.4f CER=%.4f | Wav2Vec2 WER=%.4f CER=%.4f | selected=%s reason=%s",
        out["wer_whisper"] if out["wer_whisper"] is not None else -1.0,
        out["cer_whisper"] if out["cer_whisper"] is not None else -1.0,
        out["wer_wav2vec"] if out["wer_wav2vec"] is not None else -1.0,
        out["cer_wav2vec"] if out["cer_wav2vec"] is not None else -1.0,
        out["selected_source"],
        out["selection_reason"],
    )
    return out
