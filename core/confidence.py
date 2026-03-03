"""
Research-grade ASR quality scoring: token confidence, edit-distance score, combined final score.
Combines token log-probability (from logits or Whisper segments), normalized edit distance, and match count.
"""
from typing import Dict, Any, Optional

from core.metrics import wer
from core.normalization import normalize_arabic


def _match_score(reference: str, hypothesis: str) -> float:
    """Normalized match count: matched words / ref words, in [0, 1]."""
    from core.scoring import count_matching_words
    ref_norm = [w for w in normalize_arabic(reference or "").split() if w]
    if not ref_norm:
        return 1.0 if not (hypothesis or "").strip() else 0.0
    n = count_matching_words(reference, hypothesis or "")
    return n / len(ref_norm)


def get_asr_quality_score(
    transcript: str,
    reference: str,
    logits: Optional[Any] = None,
    whisper_result: Optional[Dict[str, Any]] = None,
    token_confidence_from_logits: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute research-grade ASR quality score combining:
    - Token confidence (from Wav2Vec2 logits or Whisper segment logprobs)
    - Edit-distance score: 1 - WER (word-level similarity)
    - Match count score (optional component)

    Returns:
        {
            "final_score": float in [0, 1],
            "token_confidence": float in [0, 1],
            "edit_distance_score": float in [0, 1],
            "match_score": float in [0, 1],
        }
    """
    out = {
        "final_score": 0.0,
        "token_confidence": 0.0,
        "edit_distance_score": 0.0,
        "match_score": 0.0,
    }
    try:
        # Edit distance score: 1 - WER (1 = perfect)
        w = wer(reference, transcript or "")
        out["edit_distance_score"] = max(0.0, min(1.0, 1.0 - w))

        # Match score
        try:
            out["match_score"] = _match_score(reference, transcript or "")
        except Exception:
            out["match_score"] = out["edit_distance_score"]

        # Token confidence
        if token_confidence_from_logits is not None:
            out["token_confidence"] = max(0.0, min(1.0, float(token_confidence_from_logits)))
        elif logits is not None:
            try:
                import torch
                probs = torch.nn.functional.softmax(logits, dim=-1)
                if probs.dim() >= 2:
                    max_probs, _ = probs.max(dim=-1)
                    out["token_confidence"] = float(max_probs.mean().cpu().numpy())
                else:
                    out["token_confidence"] = out["edit_distance_score"]
            except Exception:
                out["token_confidence"] = out["edit_distance_score"]
        elif whisper_result is not None:
            # Whisper: use segment avg_logprob (negative log prob); normalize to [0,1]
            segments = whisper_result.get("segments") or []
            if segments:
                import math
                logprobs = [s.get("avg_logprob") for s in segments if s.get("avg_logprob") is not None]
                if logprobs:
                    # avg_logprob is negative; exp(avg_logprob) in (0,1]
                    mean_logprob = sum(logprobs) / len(logprobs)
                    out["token_confidence"] = max(0.0, min(1.0, math.exp(mean_logprob)))
                else:
                    no_speech = whisper_result.get("no_speech_prob")
                    out["token_confidence"] = 1.0 - no_speech if no_speech is not None else 0.5
            else:
                no_speech = whisper_result.get("no_speech_prob")
                out["token_confidence"] = 1.0 - no_speech if no_speech is not None else out["edit_distance_score"]
        else:
            out["token_confidence"] = out["edit_distance_score"]

        # Final: weighted combination (beam-score-like + edit + match)
        out["final_score"] = (
            0.4 * out["token_confidence"]
            + 0.4 * out["edit_distance_score"]
            + 0.2 * out["match_score"]
        )
        out["final_score"] = max(0.0, min(1.0, out["final_score"]))
    except Exception:
        out["final_score"] = out["edit_distance_score"]
        out["token_confidence"] = out["edit_distance_score"]
    return out
