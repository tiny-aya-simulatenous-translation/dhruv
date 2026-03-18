"""
DNSMOS P.835 quality scoring.

Reuses the ONNX-based DNSMOS scorer from data_prep. Produces three scores
per sample on a 1–5 scale: OVRL (overall), SIG (signal), BAK (background).
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

DNSMOS_MODEL_URL = (
    "https://huggingface.co/Vyvo-Research/dnsmos/resolve/main/sig_bak_ovr.onnx"
)
DNSMOS_SAMPLE_RATE = 16000
DNSMOS_INPUT_LENGTH = 9.01  # seconds


def _download_model(model_dir: str) -> Path:
    model_dir = Path(model_dir)
    model_path = model_dir / "sig_bak_ovr.onnx"
    if model_path.exists():
        return model_path
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading DNSMOS model to {model_path} ...")
    result = subprocess.run(
        ["curl", "-L", "-o", str(model_path), DNSMOS_MODEL_URL],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not model_path.exists() or model_path.stat().st_size < 1000:
        model_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download DNSMOS model: {result.stderr}")
    return model_path


class _DNSMOSScorer:
    def __init__(self, model_dir: str):
        import onnxruntime as ort

        model_path = _download_model(model_dir)
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def _prepare(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if sr != DNSMOS_SAMPLE_RATE:
            import torchaudio, torch
            waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            waveform = torchaudio.functional.resample(waveform, sr, DNSMOS_SAMPLE_RATE)
            audio = waveform.squeeze(0).numpy()
        target_len = int(DNSMOS_INPUT_LENGTH * DNSMOS_SAMPLE_RATE)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]
        return audio.astype(np.float32)

    def score(self, audio: np.ndarray, sr: int) -> dict:
        audio = self._prepare(audio, sr)
        result = self.session.run(None, {self.input_name: audio.reshape(1, -1)})
        sig, bak, ovrl = result[0][0]
        return {
            "dnsmos_ovrl": round(float(ovrl), 4),
            "dnsmos_sig": round(float(sig), 4),
            "dnsmos_bak": round(float(bak), 4),
        }


_scorer_cache: _DNSMOSScorer | None = None


def compute_dnsmos(audio_path: str, model_dir: str = "models/dnsmos") -> dict:
    """
    Compute DNSMOS scores for an audio file.

    Returns dict with keys: dnsmos_ovrl, dnsmos_sig, dnsmos_bak (each 1–5).
    """
    global _scorer_cache
    if _scorer_cache is None:
        _scorer_cache = _DNSMOSScorer(model_dir)
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return _scorer_cache.score(audio, sr)
