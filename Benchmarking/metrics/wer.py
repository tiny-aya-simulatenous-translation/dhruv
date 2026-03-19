"""
Word Error Rate via Whisper ASR.

Transcribes reconstructed audio using OpenAI Whisper and computes WER
against the reference transcription. Supports Hindi and Turkish.
"""

import functools

_whisper_model = None
_whisper_model_name = None


def _get_whisper(model_name: str = "large-v3", device: str = None):
    """Lazy-load the Whisper model (cached across calls)."""
    global _whisper_model, _whisper_model_name
    if _whisper_model is not None and _whisper_model_name == model_name:
        return _whisper_model

    import torch
    import whisper
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper model: {model_name} on {device} ...")
    _whisper_model = whisper.load_model(model_name, device=device)
    _whisper_model_name = model_name
    return _whisper_model


def _normalize_text(text: str) -> str:
    """Basic normalization: lowercase, strip punctuation, collapse whitespace."""
    import re
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _edit_distance(ref_words: list[str], hyp_words: list[str]) -> int:
    """Standard Levenshtein distance on word lists."""
    n, m = len(ref_words), len(hyp_words)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[j], prev = min(dp[j] + 1, dp[j - 1] + 1, prev + cost), dp[j]
    return dp[m]


def transcribe(audio_path: str, language: str = None, model_name: str = "large-v3") -> str:
    """Transcribe audio file using Whisper. Returns raw transcription text."""
    model = _get_whisper(model_name)
    result = model.transcribe(
        audio_path,
        language=language,
        task="transcribe",
    )
    return result["text"]


def compute_wer(
    recon_path: str,
    reference_text: str,
    language: str = None,
    whisper_model: str = "large-v3",
) -> dict:
    """
    Transcribe reconstructed audio and compute WER against reference.

    Args:
        recon_path:     Path to reconstructed audio file.
        reference_text: Ground-truth transcription.
        language:       Language hint for Whisper ("hi" for Hindi, "tr" for Turkish).
        whisper_model:  Whisper model size.

    Returns:
        dict with "wer", "ref_text" (normalized), "hyp_text" (Whisper output).
    """
    hyp_raw = transcribe(recon_path, language=language, model_name=whisper_model)

    ref_norm = _normalize_text(reference_text)
    hyp_norm = _normalize_text(hyp_raw)

    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()

    if not ref_words:
        wer_val = 0.0 if not hyp_words else 1.0
    else:
        wer_val = _edit_distance(ref_words, hyp_words) / len(ref_words)

    return {
        "wer": round(float(wer_val), 4),
        "ref_text": ref_norm,
        "hyp_text": hyp_norm,
    }
