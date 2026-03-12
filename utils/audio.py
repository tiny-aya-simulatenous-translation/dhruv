"""Shared audio utilities: load, resample, trim, loudness normalize."""

import torch
import torchaudio
import numpy as np
import soundfile as sf


def load_audio(path: str) -> tuple[torch.Tensor, int]:
    """Load audio file, return (waveform [channels, samples], sample_rate).

    Uses soundfile (libsndfile) which handles wav/flac/ogg natively without
    requiring FFmpeg or torchcodec.
    """
    audio_np, sr = sf.read(path, dtype="float32")
    if audio_np.ndim == 1:
        audio_np = audio_np[np.newaxis, :]  # [1, T]
    else:
        audio_np = audio_np.T  # [channels, T]
    return torch.from_numpy(audio_np), sr


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Convert to mono by averaging channels. Input: [C, T], output: [1, T]."""
    if waveform.shape[0] == 1:
        return waveform
    return waveform.mean(dim=0, keepdim=True)


def resample_audio(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Resample waveform using high-quality sinc interpolation."""
    if orig_sr == target_sr:
        return waveform
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
    return resampler(waveform)


def trim_silence(
    waveform: torch.Tensor,
    sr: int,
    db_threshold: float = 30.0,
    min_silence_duration: float = 0.1,
) -> torch.Tensor:
    """
    Trim leading and trailing silence based on energy threshold.

    Args:
        waveform: [1, T] mono audio tensor
        sr: sample rate
        db_threshold: dB below peak to consider as silence
        min_silence_duration: minimum silence duration in seconds to trim
    """
    audio_np = waveform.squeeze(0).numpy()

    if np.max(np.abs(audio_np)) < 1e-8:
        return waveform

    frame_length = int(0.025 * sr)  # 25ms frames
    hop_length = int(0.010 * sr)    # 10ms hop

    energy = []
    for i in range(0, len(audio_np) - frame_length, hop_length):
        frame = audio_np[i : i + frame_length]
        energy.append(np.sqrt(np.mean(frame ** 2)))
    energy = np.array(energy)

    if len(energy) == 0:
        return waveform

    peak_energy = np.max(energy)
    if peak_energy < 1e-8:
        return waveform

    threshold = peak_energy * (10 ** (-db_threshold / 20))
    active = energy > threshold

    if not np.any(active):
        return waveform

    first_active = np.argmax(active)
    last_active = len(active) - 1 - np.argmax(active[::-1])

    start_sample = max(0, first_active * hop_length - int(0.05 * sr))  # 50ms padding
    end_sample = min(len(audio_np), (last_active + 1) * hop_length + int(0.05 * sr))

    return waveform[:, start_sample:end_sample]


def loudness_normalize(
    waveform: torch.Tensor, sr: int, target_lufs: float = -23.0
) -> torch.Tensor:
    """
    Loudness-normalize audio to target LUFS (ITU-R BS.1770-4).
    Requires pyloudnorm.
    """
    import pyloudnorm as pyln

    audio_np = waveform.squeeze(0).numpy().astype(np.float64)

    if np.max(np.abs(audio_np)) < 1e-8:
        return waveform

    meter = pyln.Meter(sr)
    try:
        current_loudness = meter.integrated_loudness(audio_np)
    except Exception:
        return waveform

    if np.isinf(current_loudness):
        return waveform

    normalized = pyln.normalize.loudness(audio_np, current_loudness, target_lufs)

    # Peak-clip to prevent clipping
    peak = np.max(np.abs(normalized))
    if peak > 1.0:
        normalized = normalized / peak * 0.99

    return torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)


def get_duration(waveform: torch.Tensor, sr: int) -> float:
    """Return duration in seconds."""
    return waveform.shape[-1] / sr
