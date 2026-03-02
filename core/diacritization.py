"""
Diacritization layer for Seerat AI mode.
Pipeline: ASR text → Diacritized text → phoneme reference mapping.
Never compare phonemes without diacritization.

Optional backends: CAMeL Tools or AraT5 (HuggingFace).
If no backend available, returns original text and callers should not use for phoneme comparison.
"""
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_diacritizer: Optional[object] = None
_backend_name: Optional[str] = None


def _load_camel() -> Optional[object]:
    """Try loading CAMeL Tools diacritizer."""
    try:
        from camel_tools.utils.diacritize import Diacritizer
        # Default model; can be overridden via env
        d = Diacritizer.pretrained()
        return d
    except Exception as e:
        logger.debug("CAMeL diacritizer not available: %s", e)
        return None


def _load_arat5() -> Optional[object]:
    """Try loading AraT5 diacritization model (HuggingFace)."""
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        model_name = "UBC-NLP/AraT5-base-diacritization"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.eval()
        return {"tokenizer": tokenizer, "model": model}
    except Exception as e:
        logger.debug("AraT5 diacritizer not available: %s", e)
        return None


def _diacritize_camel(text: str, diacritizer: object) -> str:
    if not text or not text.strip():
        return text
    try:
        return diacritizer.diacritize([text.strip()])[0]
    except Exception:
        return text


def _diacritize_arat5(text: str, pipeline: dict) -> str:
    if not text or not text.strip():
        return text
    try:
        import torch
        tokenizer = pipeline["tokenizer"]
        model = pipeline["model"]
        inputs = tokenizer(text.strip(), return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            out = model.generate(**inputs, max_length=512)
        return tokenizer.decode(out[0], skip_special_tokens=True)
    except Exception:
        return text


def init_diacritizer(backend: Optional[str] = None) -> bool:
    """
    Initialize diacritization backend. Prefer CAMeL if available, else AraT5.
    backend: "camel" | "arat5" | None (auto).
    Returns True if a backend was loaded.
    """
    global _diacritizer, _backend_name
    _diacritizer = None
    _backend_name = None
    if backend == "camel":
        _diacritizer = _load_camel()
        _backend_name = "camel" if _diacritizer else None
    elif backend == "arat5":
        _diacritizer = _load_arat5()
        _backend_name = "arat5" if _diacritizer else None
    else:
        _diacritizer = _load_camel()
        if _diacritizer:
            _backend_name = "camel"
        else:
            _diacritizer = _load_arat5()
            _backend_name = "arat5" if _diacritizer else None
    if _backend_name:
        logger.info("Diacritization backend loaded: %s", _backend_name)
    return _backend_name is not None


def diacritize(text: str) -> Tuple[str, bool]:
    """
    Diacritize Arabic text (add tashkeel). Never compare phonemes without diacritization.
    Returns (diacritized_text, success). If no backend or error, returns (original_text, False).
    """
    global _diacritizer, _backend_name
    if _diacritizer is None and _backend_name is None:
        init_diacritizer()
    if _diacritizer is None:
        return text, False
    try:
        if _backend_name == "camel":
            out = _diacritize_camel(text, _diacritizer)
        elif _backend_name == "arat5":
            out = _diacritize_arat5(text, _diacritizer)
        else:
            return text, False
        return out or text, True
    except Exception as e:
        logger.warning("Diacritization failed: %s", e)
        return text, False


def is_diacritization_available() -> bool:
    """Return True if a diacritization backend is loaded."""
    return _diacritizer is not None
