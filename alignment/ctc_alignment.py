"""
CTC forced alignment using frame-level probability alignment and Viterbi path decoding.
Aligns reference Quran ayah text to user audio for word-level start_time, end_time, confidence.
Uses Wav2Vec2 emissions; no dependency on deprecated torchaudio.forced_align.
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import torch
from core.normalization import normalize_arabic

# Wav2Vec2: 320 samples per frame at 16 kHz → 20 ms per frame → 50 fps
FRAME_RATE_HZ = 50.0
FRAME_DURATION_SEC = 1.0 / FRAME_RATE_HZ


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


@dataclass
class Segment:
    label: str
    start_frame: int
    end_frame: int
    score: float

    @property
    def start_time(self) -> float:
        return self.start_frame * FRAME_DURATION_SEC

    @property
    def end_time(self) -> float:
        return self.end_frame * FRAME_DURATION_SEC


def _build_trellis(emission: torch.Tensor, tokens: List[int], blank_id: int = 0) -> torch.Tensor:
    """Build trellis matrix for CTC Viterbi alignment (log domain). Frame-level probability decoding."""
    num_frame = emission.size(0)
    num_tokens = len(tokens)
    trellis = torch.zeros((num_frame, num_tokens), device=emission.device, dtype=emission.dtype)
    trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
    trellis[0, 1:] = -float("inf")
    trellis[-num_tokens + 1:, 0] = float("inf")
    for t in range(num_frame - 1):
        trellis[t + 1, 1:] = torch.maximum(
            trellis[t, 1:] + emission[t, blank_id],
            trellis[t, :-1] + emission[t, tokens[1:]],
        )
    return trellis


def _backtrack_path(
    trellis: torch.Tensor,
    emission: torch.Tensor,
    tokens: List[int],
    blank_id: int = 0,
) -> List[Point]:
    """Backtrack to find the most likely alignment path (Viterbi). Frame-level decoding."""
    t, j = trellis.size(0) - 1, trellis.size(1) - 1
    path = [Point(j, t, emission[t, blank_id].exp().item())]
    while j > 0:
        assert t > 0
        p_stay = emission[t - 1, blank_id]
        p_change = emission[t - 1, tokens[j]]
        stayed = trellis[t - 1, j] + p_stay
        changed = trellis[t - 1, j - 1] + p_change
        t -= 1
        if changed > stayed:
            j -= 1
            prob = p_change.exp().item()
        else:
            prob = p_stay.exp().item()
        path.append(Point(j, t, prob))
    while t > 0:
        prob = emission[t - 1, blank_id].exp().item()
        path.append(Point(j, t - 1, prob))
        t -= 1
    return path[::-1]


def _merge_repeats(path: List[Point], token_labels: List[str]) -> List[Segment]:
    """Merge consecutive same-token frames into segments with average score."""
    if not path:
        return []
    segments = []
    i1, i2 = 0, 0
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        idx = path[i1].token_index
        label = token_labels[idx] if idx < len(token_labels) else ""
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1) if i2 > i1 else 0.0
        segments.append(
            Segment(
                label=label,
                start_frame=path[i1].time_index,
                end_frame=path[i2 - 1].time_index + 1,
                score=score,
            )
        )
        i1 = i2
    return segments


def _text_to_token_ids(
    text: str,
    vocab: Dict[str, int],
    word_delimiter: str = " ",
    unk_id: Optional[int] = None,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Convert normalized Arabic text to token IDs and word boundaries.
    Returns (token_ids, word_ranges) where word_ranges are (start_idx, end_idx) into token_ids.
    """
    tokens: List[int] = []
    word_ranges: List[Tuple[int, int]] = []
    space_id = vocab.get(word_delimiter)
    words = text.split()
    for wi, word in enumerate(words):
        start = len(tokens)
        for c in word:
            tid = vocab.get(c, unk_id)
            if tid is not None:
                tokens.append(tid)
        end = len(tokens)
        word_ranges.append((start, end))
        if wi < len(words) - 1 and space_id is not None:
            tokens.append(space_id)
    return tokens, word_ranges


def _segments_to_word_timings(
    segments: List[Segment],
    token_ids: List[int],
    word_ranges: List[Tuple[int, int]],
    id_to_label: Dict[int, str],
) -> List[Dict[str, Any]]:
    """
    Map character-level segments to word-level timings by grouping segment indices by word.
    """
    # Build segment index -> (start_frame, end_frame, score)
    seg_frames: List[Tuple[int, int, float]] = []
    for s in segments:
        seg_frames.append((s.start_frame, s.end_frame, s.score))
    # We have one segment per token in path order (after merge_repeats).
    # Segments correspond to token_ids order (with repeats merged).
    # So segment[k] corresponds to the k-th distinct token in the path.
    # Actually merge_repeats gives one segment per token *position* in the transcript (token_labels).
    # token_labels = [id_to_label.get(i, "") for i in token_ids] but we built segments from path with token_index.
    # So segments[i].label is the label at path position i; path position = token_index into transcript.
    # Our "transcript" for the path was the list of token_ids. So token_index in path is index into token_ids.
    # So segment 0 = token_ids[0], segment 1 = token_ids[1], ... but we merged repeats so we have one segment per unique token position in the path order. So segments are in path order: first segment is first token in transcript, etc. So segment index = token position in transcript (token_ids). So for word_ranges (start, end), the segments for that word are segments[start:end]. So we take min start_frame of segments[start:end] and max end_frame of segments[start:end], and average score.
    # But wait: we have one segment per token in the *path*; the path has repeated indices (same token over many frames). So after merge_repeats we have exactly len(token_ids) segments? No: merge_repeats merges consecutive path points with the same token_index. So we get one segment per distinct (token_index) in path order. So the number of segments equals the number of distinct token indices in the path, which equals len(token_ids) because the path goes through each token. So segments[i] corresponds to token_ids[i].
    word_timings = []
    for (start_idx, end_idx) in word_ranges:
        if start_idx >= len(segments) or end_idx > len(segments):
            word_timings.append({
                "start_time": 0.0,
                "end_time": 0.0,
                "confidence": 0.0,
                "phonetic_similarity": 0.0,
            })
            continue
        seg_slice = segments[start_idx:end_idx]
        start_frame = min(s.start_frame for s in seg_slice)
        end_frame = max(s.end_frame for s in seg_slice)
        avg_score = sum(s.score for s in seg_slice) / len(seg_slice) if seg_slice else 0.0
        word_timings.append({
            "start_time": start_frame * FRAME_DURATION_SEC,
            "end_time": end_frame * FRAME_DURATION_SEC,
            "confidence": round(avg_score, 4),
            "phonetic_similarity": round(avg_score, 4),
        })
    return word_timings


def align_reference_to_audio(
    processor: Any,
    model: Any,
    audio_path: str,
    reference_text: str,
    reference_words: List[str],
    device: str,
    load_audio_fn: Any,
    blank_id: int = 0,
) -> Dict[str, Any]:
    """
    Run CTC forced alignment: reference ayah text + user audio → word-level timings and confidence.

    Args:
        processor: Wav2Vec2Processor (for vocab and feature extraction).
        model: Wav2Vec2ForCTC model.
        audio_path: Path to audio file (16 kHz expected).
        reference_text: Normalized reference verse text (with spaces).
        reference_words: List of reference words (for output structure).
        device: torch device.
        load_audio_fn: function(path, sr?) -> (waveform, sr).
        blank_id: CTC blank token id (usually 0).

    Returns:
        {
            "words": [{"word", "start_time", "end_time", "confidence", "phonetic_similarity"}, ...],
            "frames": [...]
            "alignment_success": bool,
        }
    """
    result = {
        "words": [],
        "frames": [],
        "alignment_success": False,
    }
    try:
        speech, sr = load_audio_fn(audio_path, sr=16000)
        if len(speech) / sr < 0.3:
            return result
        # Get emissions (log probs)
        inputs = processor(
            speech, return_tensors="pt", sampling_rate=16000, padding=True
        )
        input_values = inputs.input_values.to(device)
        with torch.no_grad():
            logits = model(input_values).logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        emission = log_probs[0].cpu()
        # Vocab: token -> id (Wav2Vec2CTCTokenizer has get_vocab or encoder)
        tok = processor.tokenizer
        vocab = getattr(tok, "get_vocab", None) and tok.get_vocab() or getattr(tok, "encoder", None) or {}
        if not vocab:
            return result
        unk_id = vocab.get("<unk>") or vocab.get("|") or 0
        token_ids, word_ranges = _text_to_token_ids(reference_text, vocab, " ", unk_id)
        if not token_ids:
            return result
        # Blank id: CTC uses 0 as blank in most HF models
        if blank_id not in range(emission.size(-1)):
            blank_id = 0
        # Pipeline: Frame trellis → Backtrack path → Word boundaries
        trellis = _build_trellis(emission, token_ids, blank_id)
        path = _backtrack_path(trellis, emission, token_ids, blank_id)
        id_to_label = {v: k for k, v in vocab.items()}
        token_labels = [id_to_label.get(i, "") for i in token_ids]
        segments = _merge_repeats(path, token_labels)
        result["frames"] = [s.score for s in segments]
        word_timings = _segments_to_word_timings(
            segments, token_ids, word_ranges, id_to_label
        )
        # Attach reference word text to each timing
        for i, w in enumerate(reference_words):
            timing = word_timings[i] if i < len(word_timings) else {
                "start_time": 0.0, "end_time": 0.0, "confidence": 0.0, "phonetic_similarity": 0.0,
            }
            result["words"].append({
                "word": w,
                "start_time": timing["start_time"],
                "end_time": timing["end_time"],
                "confidence": timing["confidence"],
                "phonetic_similarity": timing["phonetic_similarity"],
            })
        result["alignment_success"] = len(result["words"]) == len(reference_words) and any(
            w.get("end_time", 0) > 0 for w in result["words"]
        )
    except Exception:
        pass
    return result
