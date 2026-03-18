#!/usr/bin/env python3
"""
Audio reconstruction evaluation for finetuned DualCodec checkpoints.

Encodes audio through the codec and decodes back, saving both original
and reconstructed waveforms for qualitative comparison.

Usage:
    # Finetuned model only:
    python scripts/reconstruct.py \
        --checkpoint output_checkpoints/.../epoch-0009_step-0055000_... \
        --hf_dataset tiny-aya-translate/hinglish-casual \
        --num_samples 50 --benchmark \
        --output_dir ../Benchmarking/data/hindi

    # Finetuned + base pretrained model (for comparison):
    python scripts/reconstruct.py \
        --checkpoint output_checkpoints/.../epoch-0009_step-0055000_... \
        --hf_dataset tiny-aya-translate/hinglish-casual \
        --num_samples 50 --benchmark --also_base \
        --output_dir ../Benchmarking/data/hindi

    # Local wav files:
    python scripts/reconstruct.py \
        --checkpoint output_checkpoints/.../epoch-0009_step-0055000_... \
        --audio_dir /path/to/wavs --num_samples 10
"""

import argparse
import os
import sys
import json

import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
import soundfile as sf

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
    """Pull audio samples from a HuggingFace dataset, including transcripts."""
    import io
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    api = HfApi()
    repo_files = api.list_repo_files(dataset_name, repo_type="dataset")
    parquet_files = sorted(f for f in repo_files if f.endswith(".parquet"))

    print(f"[load_samples_from_hf] Found {len(parquet_files)} parquet shards")

    samples = []
    for fname in parquet_files:
        if len(samples) >= num_samples:
            break
        local_path = hf_hub_download(
            repo_id=dataset_name, filename=fname, repo_type="dataset",
        )
        table = pq.read_table(local_path)
        text_col = next(
            (c for c in ["text", "transcript", "utterance"] if c in table.column_names),
            None,
        )

        for row_idx in range(table.num_rows):
            if len(samples) >= num_samples:
                break
            try:
                audio_entry = table.column("audio")[row_idx].as_py()
                audio_bytes = audio_entry["bytes"]
                data, sr = sf.read(io.BytesIO(audio_bytes))
                waveform = torch.from_numpy(data).float()
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                else:
                    waveform = waveform.T
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                duration = waveform.shape[-1] / sr
                if duration < min_duration or duration > max_duration:
                    continue

                if sr != 24000:
                    waveform = torchaudio.functional.resample(waveform, sr, 24000)

                sample = {
                    "waveform": waveform,
                    "name": f"sample_{len(samples):04d}",
                }
                if text_col:
                    sample["text"] = table.column(text_col)[row_idx].as_py()

                samples.append(sample)
            except Exception as e:
                print(f"  Skipping row {row_idx}: {e}")
                continue

    print(f"[load_samples_from_hf] Loaded {len(samples)} samples"
          f" ({'with' if samples and 'text' in samples[0] else 'without'} transcripts)")
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
        "--benchmark", action="store_true",
        help="Output in benchmark-ready layout: output_dir/originals/ and output_dir/codecs/dualcodec/",
    )
    parser.add_argument(
        "--also_base", action="store_true",
        help="Also reconstruct with the pretrained base model (from dualcodec_ckpts/) for comparison",
    )
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
    # Build directly instead of using Inference() to avoid cached_path dependency
    from dualcodec.infer.dualcodec.inference_with_semantic import (
        Inference, _build_semantic_model,
    )

    print(f"Loading semantic model from {w2v_path} ...")
    if not os.path.isdir(dualcodec_ckpts):
        print(f"ERROR: {dualcodec_ckpts} not found.")
        print("Download it with:")
        print('  python3 -c "from huggingface_hub import snapshot_download; '
              "snapshot_download('amphion/dualcodec', local_dir='dualcodec_ckpts')\"")
        return

    engine = object.__new__(Inference)
    engine.semantic_cfg = _build_semantic_model(
        dualcodec_path=dualcodec_ckpts,
        semantic_model_path=w2v_path,
        device=device,
    )
    engine.model = model
    engine.model.to(device)
    engine.model.eval()
    for key in engine.semantic_cfg:
        if isinstance(engine.semantic_cfg[key], (torch.nn.Module, torch.Tensor)):
            engine.semantic_cfg[key] = engine.semantic_cfg[key].to(device)
    engine.device = device
    engine.autocast = True

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

    # --- Run reconstruction for a given engine, save outputs ---
    def run_reconstruction(eng, samples, codec_name, save_originals=True):
        if args.benchmark:
            orig_dir = os.path.join(args.output_dir, "originals")
            recon_dir = os.path.join(args.output_dir, "codecs", codec_name)
            os.makedirs(orig_dir, exist_ok=True)
            os.makedirs(recon_dir, exist_ok=True)
        else:
            os.makedirs(args.output_dir, exist_ok=True)

        metrics = []
        print(f"\nReconstructing {len(samples)} samples [{codec_name}] (quantizers={args.num_quantizers}) ...\n")

        for sample in samples:
            name = sample["name"]
            waveform = sample["waveform"]

            original, recon = reconstruct(eng, waveform, device, args.num_quantizers)
            snr = compute_snr(original, recon)

            wav_name = f"{name}.wav"
            if args.benchmark:
                if save_originals:
                    sf.write(os.path.join(orig_dir, wav_name), original.numpy().T, 24000)
                sf.write(os.path.join(recon_dir, wav_name), recon.numpy().T, 24000)
            else:
                sf.write(os.path.join(args.output_dir, f"{name}_{codec_name}_original.wav"), original.numpy().T, 24000)
                sf.write(os.path.join(args.output_dir, f"{name}_{codec_name}_recon.wav"), recon.numpy().T, 24000)

            duration = original.shape[-1] / 24000
            metrics.append({"name": name, "snr_db": round(snr, 2), "duration_s": round(duration, 2)})
            print(f"  {name}: SNR = {snr:.2f} dB  ({duration:.1f}s)")

        avg_snr = np.mean([m["snr_db"] for m in metrics])
        print(f"\n[{codec_name}] Average SNR: {avg_snr:.2f} dB")

        summary = {"codec": codec_name, "avg_snr_db": round(avg_snr, 2), "num_samples": len(metrics), "samples": metrics}
        metrics_fname = f"metrics_{codec_name}.json" if args.benchmark else "metrics.json"
        with open(os.path.join(args.output_dir, metrics_fname), "w") as f:
            json.dump(summary, f, indent=2)

        return avg_snr

    # --- Finetuned model ---
    ft_snr = run_reconstruction(engine, samples, codec_name="dualcodec", save_originals=True)

    # --- Base pretrained model (optional) ---
    if args.also_base:
        import safetensors.torch

        base_weights = os.path.join(dualcodec_ckpts, "dualcodec_25hz_16384_1024.safetensors")
        if not os.path.isfile(base_weights):
            print(f"\nWARNING: Base model weights not found at {base_weights}")
            print("Download full dualcodec_ckpts with:")
            print('  python3 -c "from huggingface_hub import snapshot_download; '
                  "snapshot_download('amphion/dualcodec', local_dir='dualcodec_ckpts')\"")
        else:
            print(f"\nLoading base pretrained model from {base_weights} ...")
            base_model = build_model(args.model_config)
            safetensors.torch.load_model(base_model, base_weights, strict=False)
            base_model.eval()

            base_engine = object.__new__(Inference)
            base_engine.semantic_cfg = engine.semantic_cfg  # reuse same w2v-bert
            base_engine.model = base_model
            base_engine.model.to(device)
            base_engine.model.eval()
            base_engine.device = device
            base_engine.autocast = True

            base_snr = run_reconstruction(base_engine, samples, codec_name="dualcodec_base", save_originals=False)

            print(f"\n{'='*50}")
            print(f"  COMPARISON")
            print(f"  Finetuned:  {ft_snr:.2f} dB avg SNR")
            print(f"  Base:       {base_snr:.2f} dB avg SNR")
            print(f"  Delta:      {ft_snr - base_snr:+.2f} dB")
            print(f"{'='*50}")

    # Save transcripts.jsonl for benchmark WER evaluation
    has_transcripts = any("text" in s and s["text"] for s in samples)
    if args.benchmark and has_transcripts:
        transcript_path = os.path.join(args.output_dir, "transcripts.jsonl")
        with open(transcript_path, "w", encoding="utf-8") as f:
            for sample in samples:
                text = sample.get("text", "")
                if text:
                    entry = {"file": f"{sample['name']}.wav", "text": text}
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Transcripts saved to {transcript_path}")

    print(f"\nOutputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
