"""
DNSMOS P.835 local scorer using ONNX Runtime.

Downloads and caches the ONNX model from the Microsoft DNS-Challenge repo.
Produces three scores per audio sample: OVRL, SIG, BAK (all on 1-5 scale).

Reference: https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import onnxruntime as ort

DNSMOS_MODEL_URL = (
    "https://huggingface.co/Vyvo-Research/dnsmos/resolve/main/sig_bak_ovr.onnx"
)
DNSMOS_SAMPLE_RATE = 16000
DNSMOS_INPUT_LENGTH = 9.01  # seconds -- model expects this length


def download_dnsmos_model(model_dir: str = "models/dnsmos") -> Path:
    """Download the ONNX model if not cached. Safe to call before spawning workers."""
    model_dir = Path(model_dir)
    model_path = model_dir / "sig_bak_ovr.onnx"
    if model_path.exists():
        return model_path
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading DNSMOS model to {model_path}...")
    result = subprocess.run(
        ["curl", "-L", "-o", str(model_path), DNSMOS_MODEL_URL],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not model_path.exists() or model_path.stat().st_size < 1000:
        model_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download DNSMOS model.\n"
            f"  curl stderr: {result.stderr}\n"
            f"  Download manually:\n"
            f"    curl -L -o {model_path} {DNSMOS_MODEL_URL}"
        )
    print("Done.")
    return model_path


class DNSMOSScorer:
    """Score audio files using DNSMOS P.835 (OVRL, SIG, BAK)."""

    def __init__(self, model_dir: str = "models/dnsmos"):
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / "sig_bak_ovr.onnx"
        if not self.model_path.exists():
            download_dnsmos_model(model_dir)
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def _prepare_input(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Resample to 16kHz and pad/truncate to expected input length."""
        if sr != DNSMOS_SAMPLE_RATE:
            import torchaudio
            import torch

            waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=DNSMOS_SAMPLE_RATE)
            waveform = resampler(waveform)
            audio = waveform.squeeze(0).numpy()

        target_len = int(DNSMOS_INPUT_LENGTH * DNSMOS_SAMPLE_RATE)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)), mode="constant")
        else:
            audio = audio[:target_len]

        return audio.astype(np.float32)

    def score(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        """
        Score a single audio sample.

        Args:
            audio: 1D numpy array of audio samples
            sr: sample rate of the audio

        Returns:
            dict with keys: dnsmos_ovrl, dnsmos_sig, dnsmos_bak
        """
        audio = self._prepare_input(audio, sr)
        input_data = audio.reshape(1, -1)
        result = self.session.run(None, {self.input_name: input_data})
        # Model outputs: [SIG, BAK, OVRL]
        sig, bak, ovrl = result[0][0]
        return {
            "dnsmos_ovrl": round(float(ovrl), 4),
            "dnsmos_sig": round(float(sig), 4),
            "dnsmos_bak": round(float(bak), 4),
        }

    def score_file(self, path: str, sr: int = None) -> dict[str, float]:
        """Score an audio file by path."""
        import soundfile as sf

        audio, file_sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return self.score(audio, sr or file_sr)
