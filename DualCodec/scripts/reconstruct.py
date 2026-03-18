#!/usr/bin/env python3
"""
Audio reconstruction evaluation for finetuned DualCodec checkpoints.

Encodes audio through the codec and decodes back, saving both original
and reconstructed waveforms for qualitative comparison.

Usage:
    python scripts/reconstruct.py \
        --checkpoint output_checkpoints/dualcodec_ft_hindi_25hz/checkpoint/epoch-0009_step-0055000_loss-0.000000-dualcodec_ft_hindi_25hz \
        --hf_dataset Pranavz/hinglish-casual \
        --num_samples 10 \
        --output_dir reconstruction_outputs

    # Or with local wav files:
    python scripts/reconstruct.py \
        --checkpoint output_checkpoints/dualcodec_ft_hindi_25hz/checkpoint/epoch-0009_step-0055000_loss-0.000000-dualcodec_ft_hindi_25hz \
        --audio_dir /path/to/wavs \
        --num_samples 10
"""

import argparse
import os
import sys
import json

import torch
import torchaudio
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def build_model(model_config="dualcodec_25hz_16384_1024_12vq"):
    """Instantiate DualCodec from its hydra config (no weights loaded)."""
    import hydra
    from hydra import initialize_config_dir

    conf_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "dualcodec", "conf", "model")
    )
    with initialize_config_dir(version_base="1.3", config_dir=conf_dir):
        cfg = hydra.compose(config_name=f"{model_config}.yaml", overrides=[])
        model = hydra.utils.instantiate(cfg.model)
    return model


def load_checkpoint(model, checkpoint_path):
    """Load finetuned weights saved by accelerator's save_state."""
    import safetensors.torch

    weights_file = os.path.join(checkpoint_path, "model.safetensors")
    if not os.path.isfile(weights_file):
        raise FileNotFoundError(f"No model.safetensors found in {checkpoint_path}")

    safetensors.torch.load_model(model, weights_file, strict=False)
    print(f"Loaded weights from {weights_file}")
    return model


def load_samples_from_hf(dataset_name, num_samples, min_duration=1.0, max_duration=30.0):
    """Pull audio samples from a HuggingFace dataset."""
    from dualcodec.dataset.hindi_dataset import HuggingFaceAudioDataset

    ds = HuggingFaceAudioDataset(
        dataset_name=dataset_name,
        split="train",
        min_duration=min_duration,
        max_duration=max_duration,
    )
    samples = []
    for i, sample in enumerate(ds):
        if i >= num_samples:
            break
        waveform = sample["speech"]
        sr = sample["sample_rate"]
        if sr != 24000:
            waveform = torchaudio.functional.resample(waveform, sr, 24000)
        samples.append({
            "waveform": waveform,
            "name": f"sample_{i:04d}",
        })
    return samples


def load_samples_from_dir(audio_dir, num_samples):
    """Load wav/flac/mp3 files from a local directory."""
    exts = {".wav", ".flac", ".mp3", ".ogg"}
    files = sorted(
        f for f in os.listdir(audio_dir)
        if os.path.splitext(f)[1].lower() in exts
    )[:num_samples]

    samples = []
    for f in files:
        waveform, sr = torchaudio.load(os.path.join(audio_dir, f))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 24000:
            waveform = torchaudio.functional.resample(waveform, sr, 24000)
        samples.append({
            "waveform": waveform,
            "name": os.path.splitext(f)[0],
        })
    return samples


def compute_snr(original, reconstructed):
    """Signal-to-noise ratio in dB."""
    noise = original - reconstructed
    snr = 10 * torch.log10(
        (original ** 2).sum() / ((noise ** 2).sum() + 1e-8)
    )
    return snr.item()


@torch.no_grad()
def reconstruct(inference_engine, waveform, device, num_quantizers=8):
    """Encode → decode a single waveform, return the reconstruction."""
    audio = waveform.unsqueeze(0).to(device)  # (1, 1, T)
    semantic_codes, acoustic_codes = inference_engine.encode(audio, n_quantizers=num_quantizers)
    recon = inference_engine.decode(semantic_codes, acoustic_codes)

    min_len = min(audio.shape[-1], recon.shape[-1])
    original = audio[..., :min_len].cpu()
    recon = recon[..., :min_len].cpu()
    return original.squeeze(0), recon.squeeze(0)


def main():
    parser = argparse.ArgumentParser(description="DualCodec reconstruction evaluation")
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to the finetuned checkpoint directory (contains model.safetensors)",
    )
    parser.add_argument("--hf_dataset", default=None, help="HuggingFace dataset name")
    parser.add_argument("--audio_dir", default=None, help="Local directory of audio files")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--num_quantizers", type=int, default=8)
    parser.add_argument("--output_dir", default="reconstruction_outputs")
    parser.add_argument(
        "--model_config", default="dualcodec_25hz_16384_1024_12vq",
        help="Model config name under dualcodec/conf/model/",
    )
    parser.add_argument(
        "--w2v_path", default=None,
        help="Path to w2v-bert-2.0 (default: ./w2v-bert-2.0)",
    )
    parser.add_argument(
        "--dualcodec_ckpts", default=None,
        help="Path to pretrained dualcodec_ckpts dir (for mean/var stats). Default: ./dualcodec_ckpts",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.hf_dataset is None and args.audio_dir is None:
        parser.error("Provide either --hf_dataset or --audio_dir")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    w2v_path = args.w2v_path or os.path.join(repo_root, "w2v-bert-2.0")
    dualcodec_ckpts = args.dualcodec_ckpts or os.path.join(repo_root, "dualcodec_ckpts")

    # --- Build model and load finetuned weights ---
    print(f"Building model ({args.model_config}) ...")
    model = build_model(args.model_config)
    model = load_checkpoint(model, args.checkpoint)
    model.eval()

    # --- Build the Inference engine (handles semantic feature extraction) ---
    from dualcodec.infer.dualcodec.inference_with_semantic import Inference

    print(f"Loading semantic model from {w2v_path} ...")
    engine = Inference(
        dualcodec_model=model,
        dualcodec_path=dualcodec_ckpts,
        w2v_path=w2v_path,
        device=device,
        autocast=True,
    )

    # --- Load audio samples ---
    if args.hf_dataset:
        print(f"Loading {args.num_samples} samples from HF dataset: {args.hf_dataset}")
        samples = load_samples_from_hf(args.hf_dataset, args.num_samples)
    else:
        print(f"Loading samples from {args.audio_dir}")
        samples = load_samples_from_dir(args.audio_dir, args.num_samples)

    if not samples:
        print("No samples found!")
        return

    # --- Run reconstruction ---
    os.makedirs(args.output_dir, exist_ok=True)
    metrics = []

    print(f"\nReconstructing {len(samples)} samples (quantizers={args.num_quantizers}) ...\n")
    for sample in samples:
        name = sample["name"]
        waveform = sample["waveform"]  # (1, T)

        original, recon = reconstruct(engine, waveform, device, args.num_quantizers)
        snr = compute_snr(original, recon)

        orig_path = os.path.join(args.output_dir, f"{name}_original.wav")
        recon_path = os.path.join(args.output_dir, f"{name}_reconstructed.wav")
        torchaudio.save(orig_path, original, 24000)
        torchaudio.save(recon_path, recon, 24000)

        duration = original.shape[-1] / 24000
        metrics.append({"name": name, "snr_db": round(snr, 2), "duration_s": round(duration, 2)})
        print(f"  {name}: SNR = {snr:.2f} dB  ({duration:.1f}s)")

    avg_snr = np.mean([m["snr_db"] for m in metrics])
    print(f"\nAverage SNR: {avg_snr:.2f} dB")
    print(f"Outputs saved to {args.output_dir}/")

    summary = {"avg_snr_db": round(avg_snr, 2), "num_samples": len(metrics), "samples": metrics}
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
