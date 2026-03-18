"""
Segmental Signal-to-Noise Ratio (SSNR).

Computes SNR over short overlapping frames, then averages across voiced
frames only (frames where the original has energy above a threshold).
This avoids inflating the score with silent regions.
"""

import numpy as np
import soundfile as sf


def _load_and_align(orig_path: str, recon_path: str, target_sr: int = 16000):
    """Load two audio files, resample if needed, and trim to same length."""
    orig, sr_o = sf.read(orig_path, dtype="float32")
    recon, sr_r = sf.read(recon_path, dtype="float32")

    if orig.ndim > 1:
        orig = orig.mean(axis=1)
    if recon.ndim > 1:
        recon = recon.mean(axis=1)

    if sr_o != target_sr or sr_r != target_sr:
        import torchaudio, torch
        if sr_o != target_sr:
            t = torch.tensor(orig, dtype=torch.float32).unsqueeze(0)
            orig = torchaudio.functional.resample(t, sr_o, target_sr).squeeze(0).numpy()
        if sr_r != target_sr:
            t = torch.tensor(recon, dtype=torch.float32).unsqueeze(0)
            recon = torchaudio.functional.resample(t, sr_r, target_sr).squeeze(0).numpy()

    min_len = min(len(orig), len(recon))
    return orig[:min_len], recon[:min_len]


def compute_ssnr(
    orig_path: str,
    recon_path: str,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    sr: int = 16000,
    floor_db: float = -30.0,
    ceil_db: float = 35.0,
) -> dict:
    """
    Compute Segmental SNR between original and reconstructed audio.

    Args:
        orig_path:  Path to original audio file.
        recon_path: Path to reconstructed audio file.
        frame_ms:   Frame length in milliseconds.
        hop_ms:     Hop length in milliseconds.
        sr:         Analysis sample rate.
        floor_db:   Clamp per-frame SNR below this value.
        ceil_db:    Clamp per-frame SNR above this value.

    Returns:
        dict with "ssnr_db" (average segmental SNR in dB).
    """
    orig, recon = _load_and_align(orig_path, recon_path, target_sr=sr)

    frame_len = int(frame_ms * sr / 1000)
    hop_len = int(hop_ms * sr / 1000)

    noise = orig - recon
    eps = 1e-10

    # energy threshold: skip frames where original is mostly silent
    global_energy = np.mean(orig ** 2)
    energy_threshold = global_energy * 1e-4

    snr_frames = []
    for start in range(0, len(orig) - frame_len + 1, hop_len):
        orig_frame = orig[start : start + frame_len]
        noise_frame = noise[start : start + frame_len]

        sig_energy = np.sum(orig_frame ** 2)
        if sig_energy < energy_threshold:
            continue

        noise_energy = np.sum(noise_frame ** 2) + eps
        snr_db = 10.0 * np.log10(sig_energy / noise_energy)
        snr_db = np.clip(snr_db, floor_db, ceil_db)
        snr_frames.append(snr_db)

    if not snr_frames:
        return {"ssnr_db": 0.0}

    return {"ssnr_db": round(float(np.mean(snr_frames)), 4)}
