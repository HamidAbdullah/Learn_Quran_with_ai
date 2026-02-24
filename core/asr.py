"""
Dual ASR module: Whisper + Wav2Vec2 Arabic/Quran, best-transcript selection, confidence scoring.
Selection uses weighted similarity: 0.6*(1-WER) + 0.3*phonetic + 0.1*confidence. Fallback: match-count then confidence.
"""
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np

from core.normalization import normalize_arabic
from core.scoring import segment_transcript_by_reference, count_matching_words
from core.metrics import wer

WEIGHT_EDIT_DISTANCE = 0.6
WEIGHT_PHONETIC = 0.3
WEIGHT_CONFIDENCE = 0.1


def _weighted_similarity(
    reference: str,
    transcript: str,
    phonetic_similarity: float,
    confidence_score: float,
) -> float:
    edit_distance_score = max(0.0, min(1.0, 1.0 - wer(reference, transcript or "")))
    return (
        WEIGHT_EDIT_DISTANCE * edit_distance_score
        + WEIGHT_PHONETIC * max(0.0, min(1.0, phonetic_similarity))
        + WEIGHT_CONFIDENCE * max(0.0, min(1.0, confidence_score))
    )


def run_dual_asr(
    whisper_model: Any,
    phonetic_analyzer: Any,
    reference_text: str,
    audio_path: Optional[str] = None,
    audio_y_sr: Optional[Tuple[np.ndarray, int]] = None,
) -> Dict[str, Any]:
    """
    Run dual ASR (Whisper + Wav2Vec2), select best transcript by match count then confidence.
    Uses at most one Whisper forward and one Wav2Vec2 forward.

    Args:
        whisper_model: OpenAI Whisper model (transcribe(audio_path, language="ar")).
        phonetic_analyzer: PhoneticAnalyzer (run_forward(y, sr) and transcribe(path)).
        reference_text: Verse text (Uthmani) for segmentation and selection.
        audio_path: Path to audio file (used for Whisper and for Wav2Vec2 if audio_y_sr not set).
        audio_y_sr: Optional (y, sr) preloaded audio; when set, Wav2Vec2 uses this and audio_path is only for Whisper.

    Returns:
        {
            "selected_transcript": str,
            "selected_source": "whisper" | "wav2vec",
            "selection_reason": str,
            "transcript_whisper": str,
            "transcript_wav2vec": str,
            "confidence_whisper": float,
            "confidence_wav2vec": float,
            "match_count_whisper": int,
            "match_count_wav2vec": int,
            "logits": optional tensor (for alignment/Tajweed reuse),
        }
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
        "logits": None,
    }

    # --- Whisper ---
    whisper_text = ""
    confidence_whisper = 0.0
    whisper_result = None
    if audio_path:
        try:
            trans = whisper_model.transcribe(audio_path, language="ar")
            whisper_result = trans
            whisper_text = (trans.get("text") or "").strip()
            no_speech = trans.get("no_speech_prob")
            if no_speech is not None:
                confidence_whisper = float(1.0 - no_speech)
            else:
                confidence_whisper = 0.9 if whisper_text else 0.0
        except Exception:
            pass
    out["transcript_whisper"] = whisper_text
    out["confidence_whisper"] = confidence_whisper
    out["match_count_whisper"] = count_matching_words(reference_text, whisper_text) if whisper_text else 0

    # --- Wav2Vec2 (reuse preloaded audio if provided) ---
    wav2vec_text = ""
    logits = None
    confidence_wav2vec = 0.0
    if audio_y_sr is not None:
        y, sr = audio_y_sr
        wav2vec_text, logits = phonetic_analyzer.run_forward(y, sr) or (None, None)
        if logits is not None:
            analysis = phonetic_analyzer.analyze_alignment_from_logits(logits)
            confidence_wav2vec = float(analysis.get("confidence_avg") or 0.0)
    if (not wav2vec_text or not wav2vec_text.strip()) and audio_path:
        wav2vec_text = phonetic_analyzer.transcribe(audio_path) or ""
        if wav2vec_text:
            analysis = phonetic_analyzer.analyze_alignment(audio_path, wav2vec_text)
            confidence_wav2vec = float(analysis.get("confidence_avg") or 0.0)

    if wav2vec_text and wav2vec_text.strip().count(" ") < max(1, len(ref_words) - 2):
        wav2vec_text = segment_transcript_by_reference(wav2vec_text, reference_text)

    out["transcript_wav2vec"] = wav2vec_text or ""
    out["confidence_wav2vec"] = confidence_wav2vec
    out["match_count_wav2vec"] = count_matching_words(reference_text, wav2vec_text) if wav2vec_text else 0
    out["logits"] = logits

    # --- Selection: weighted similarity, fallback to match_count then confidence ---
    wh_match = out["match_count_whisper"]
    w_match = out["match_count_wav2vec"]

    use_weighted = False
    sim_whisper = None
    sim_wav2vec = None
    try:
        from core.confidence import get_asr_quality_score
        q_wh = get_asr_quality_score(
            whisper_text, reference_text,
            whisper_result=whisper_result,
        )
        q_wv = get_asr_quality_score(
            wav2vec_text, reference_text,
            logits=logits,
            token_confidence_from_logits=confidence_wav2vec if logits is not None else None,
        )
        sim_whisper = _weighted_similarity(
            reference_text, whisper_text,
            phonetic_similarity=q_wh["token_confidence"],
            confidence_score=confidence_whisper,
        )
        sim_wav2vec = _weighted_similarity(
            reference_text, wav2vec_text,
            phonetic_similarity=confidence_wav2vec,
            confidence_score=q_wv["token_confidence"],
        )
        use_weighted = True
    except Exception:
        pass

    if whisper_text and wav2vec_text:
        if use_weighted and sim_wav2vec is not None and sim_whisper is not None:
            if sim_wav2vec >= sim_whisper:
                out["selected_transcript"] = wav2vec_text
                out["selected_source"] = "wav2vec"
                out["selection_reason"] = "weighted_similarity"
            else:
                out["selected_transcript"] = whisper_text
                out["selected_source"] = "whisper"
                out["selection_reason"] = "weighted_similarity"
        elif w_match > wh_match:
            out["selected_transcript"] = wav2vec_text
            out["selected_source"] = "wav2vec"
            out["selection_reason"] = "match_count"
        elif wh_match > w_match:
            out["selected_transcript"] = whisper_text
            out["selected_source"] = "whisper"
            out["selection_reason"] = "match_count"
        else:
            if confidence_wav2vec >= confidence_whisper:
                out["selected_transcript"] = wav2vec_text
                out["selected_source"] = "wav2vec"
                out["selection_reason"] = "confidence_tie"
            else:
                out["selected_transcript"] = whisper_text
                out["selected_source"] = "whisper"
                out["selection_reason"] = "confidence_tie"
    else:
        out["selected_transcript"] = whisper_text or wav2vec_text
        if whisper_text and not wav2vec_text:
            out["selected_source"] = "whisper"
        else:
            out["selected_source"] = "wav2vec"
        out["selection_reason"] = "single_engine"

    return out
