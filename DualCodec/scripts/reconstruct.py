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
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def build_model(model_config="dualcodec_25hz_16384_1024_12vq"):
    """Instantiate DualCodec from its hydra config (no weights loaded)."""
    import hydra
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    conf_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "dualcodec", "conf", "model")
    )
    GlobalHydra.instance().clear()
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


def _load_from_parquet(dataset_name, num_samples, min_duration, max_duration):
    """Fast path: read audio bytes directly from parquet (works when audio is embedded)."""
    import io
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    api = HfApi()
    repo_files = api.list_repo_files(dataset_name, repo_type="dataset")
    parquet_files = sorted(f for f in repo_files if f.endswith(".parquet"))

    if not parquet_files:
        return None

    # Check if any audio column has embedded bytes
    first_local = hf_hub_download(repo_id=dataset_name, filename=parquet_files[0], repo_type="dataset")
    first_table = pq.read_table(first_local)
    audio_col_name = next(
        (c for c in ["audio", "audio_filepath", "audio_path"]
         if c in first_table.column_names),
        None,
    )
    if audio_col_name is None:
        return None
    first_audio = first_table.column(audio_col_name)[0].as_py()
    if not isinstance(first_audio, dict) or "bytes" not in first_audio or first_audio["bytes"] is None:
        return None

    print(f"[load_from_parquet] Found {len(parquet_files)} shards with embedded audio (column: {audio_col_name})")

    # Collect all candidate rows across shards (metadata only, no decoding yet)
    all_rows = []
    for fname in parquet_files:
        local_path = hf_hub_download(repo_id=dataset_name, filename=fname, repo_type="dataset")
        table = pq.read_table(local_path)
        text_col = next(
            (c for c in ["text", "transcript", "transcription", "utterance", "normalized", "verbatim"]
             if c in table.column_names),
            None,
        )
        has_duration = "duration" in table.column_names
        for row_idx in range(table.num_rows):
            if has_duration:
                dur = table.column("duration")[row_idx].as_py()
                if dur < min_duration or dur > max_duration:
                    continue
            audio_entry = table.column(audio_col_name)[row_idx].as_py()
            if not isinstance(audio_entry, dict) or not audio_entry.get("bytes"):
                continue
            row_data = {"audio_bytes": audio_entry["bytes"]}
            if text_col:
                row_data["text"] = table.column(text_col)[row_idx].as_py()
            all_rows.append(row_data)
            if len(all_rows) >= num_samples:
                break
        if len(all_rows) >= num_samples:
            break
    print(f"[load_from_parquet] Collected {len(all_rows)} rows, decoding audio in parallel ...")

    # Parallel decode with thread pool
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _decode_one(idx_and_row):
        idx, row = idx_and_row
        try:
            data, sr = sf.read(io.BytesIO(row["audio_bytes"]))
            waveform = torch.from_numpy(data).float()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.T
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            duration = waveform.shape[-1] / sr
            if duration < min_duration or duration > max_duration:
                return None
            if sr != 24000:
                waveform = torchaudio.functional.resample(waveform, sr, 24000)
            sample = {"waveform": waveform, "name": f"sample_{idx:04d}"}
            if "text" in row:
                sample["text"] = row["text"]
            return sample
        except Exception:
            return None

    samples = []
    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_decode_one, (i, r)): i for i, r in enumerate(all_rows)}
        for future in as_completed(futures):
            result = future.result()
            done += 1
            if result is not None:
                samples.append(result)
            if done % 200 == 0:
                print(f"  [load_from_parquet] decoded {done}/{len(all_rows)} ...")

    # Sort by name to maintain order
    samples.sort(key=lambda s: s["name"])
    # Re-number
    for i, s in enumerate(samples):
        s["name"] = f"sample_{i:04d}"
    print(f"[load_from_parquet] Done: {len(samples)} samples ready")
    return samples


def _load_from_datasets_lib(dataset_name, num_samples, min_duration, max_duration):
    """Fallback: use HuggingFace datasets library (handles file-reference audio)."""
    import os
    os.environ.setdefault("HF_AUDIO_DECODER", "soundfile")
    from datasets import load_dataset, Audio

    print(f"[load_from_datasets] Streaming {dataset_name} via datasets library ...")
    splits = ["test", "train", "validation"]
    ds = None
    for split in splits:
        try:
            ds = load_dataset(dataset_name, split=split, streaming=True)
            print(f"[load_from_datasets] Using split: {split}")
            break
        except Exception as e:
            print(f"[load_from_datasets] Failed to load split '{split}': {e}")
            continue
    if ds is None:
        print(f"[load_from_datasets] Could not load any split from {dataset_name}")
        return []

    # Disable audio decoding to avoid torchcodec/FFmpeg dependency;
    # we decode manually with soundfile below.
    try:
        audio_cols = [c for c in ds.column_names if "audio" in c.lower()]
        if audio_cols:
            ds = ds.cast_column(audio_cols[0], Audio(decode=False))
            print(f"[load_from_datasets] Disabled auto-decode for '{audio_cols[0]}', will decode with soundfile")
    except Exception as e:
        print(f"[load_from_datasets] Could not cast audio column: {e}")

    text_col = None
    audio_col = None
    samples = []
    for row in tqdm(ds):
        if len(samples) >= num_samples:
            break
        try:
            # Find the audio column on first iteration
            if audio_col is None:
                for candidate in ["audio", "audio_filepath", "path", "file"]:
                    if candidate in row and row[candidate] is not None:
                        audio_col = candidate
                        break
                if audio_col is None:
                    print(f"[load_from_datasets] Could not find audio column. Keys: {list(row.keys())}")
                    return []
                print(f"[load_from_datasets] Using audio column: {audio_col}")

            audio = row.get(audio_col)
            if audio is None:
                continue

            # Decode audio manually with soundfile
            import io as _io
            if isinstance(audio, dict) and "array" in audio:
                waveform = torch.from_numpy(audio["array"]).float()
                sr = audio["sampling_rate"]
            elif isinstance(audio, dict) and "bytes" in audio and audio["bytes"] is not None:
                data, sr = sf.read(_io.BytesIO(audio["bytes"]))
                waveform = torch.from_numpy(data).float()
            elif isinstance(audio, dict) and "path" in audio and audio.get("bytes") is None:
                # File-reference audio: download and read
                from huggingface_hub import hf_hub_download
                local = hf_hub_download(
                    repo_id=dataset_name, filename=audio["path"], repo_type="dataset"
                )
                data, sr = sf.read(local)
                waveform = torch.from_numpy(data).float()
            elif hasattr(audio, "get_all_samples"):
                decoded = audio.get_all_samples()
                waveform = decoded.data.float()
                sr = decoded.sample_rate
            else:
                print(f"  Skipping: unsupported audio type (got {type(audio)})")
                continue

            # Normalize to (1, T) mono
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            duration = waveform.shape[-1] / sr
            if duration < min_duration or duration > max_duration:
                continue
            if sr != 24000:
                waveform = torchaudio.functional.resample(waveform, sr, 24000)

            # Find text column on first iteration
            if text_col is None:
                text_col = next(
                    (c for c in ["text", "transcript", "utterance", "normalized", "verbatim"]
                     if c in row and row[c]),
                    None,
                )

            sample = {"waveform": waveform, "name": f"sample_{len(samples):04d}"}
            if text_col and row.get(text_col):
                sample["text"] = row[text_col]
            samples.append(sample)
        except Exception as e:
            print(f"  Skipping: {e}")
            continue
    return samples


def load_samples_from_hf(dataset_name, num_samples, min_duration=1.0, max_duration=30.0):
    """Pull audio samples from a HuggingFace dataset, including transcripts.

    Tries fast parquet path first (embedded audio bytes), then falls back to
    the datasets library (for datasets with file-reference audio like Lahaja).
    """
    samples = _load_from_parquet(dataset_name, num_samples, min_duration, max_duration)
    if samples is None:
        print(f"[load_samples_from_hf] No embedded audio in parquets, falling back to datasets library")
        samples = _load_from_datasets_lib(dataset_name, num_samples, min_duration, max_duration)

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


@torch.no_grad()
def reconstruct_batch(inference_engine, waveforms, device, num_quantizers=8):
    """Encode → decode a batch of waveforms. Returns list of (original, recon) pairs.

    Pads to the longest waveform in the batch, then trims back after decoding.
    """
    lengths = [w.shape[-1] for w in waveforms]
    max_len = max(lengths)

    # Pad all waveforms to max_len and stack → (B, 1, max_len)
    padded = torch.stack([
        F.pad(w, (0, max_len - w.shape[-1])) for w in waveforms
    ]).to(device)

    semantic_codes, acoustic_codes = inference_engine.encode(padded, n_quantizers=num_quantizers)
    recon_batch = inference_engine.decode(semantic_codes, acoustic_codes)

    results = []
    for i, orig_len in enumerate(lengths):
        min_len = min(orig_len, recon_batch.shape[-1])
        orig = padded[i:i+1, :, :min_len].cpu()
        recon = recon_batch[i:i+1, :, :min_len].cpu()
        results.append((orig.squeeze(0), recon.squeeze(0)))
    return results


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
        batch_size = 8
        print(f"\nReconstructing {len(samples)} samples [{codec_name}] (quantizers={args.num_quantizers}, batch={batch_size}) ...\n")

        for batch_start in range(0, len(samples), batch_size):
            batch_samples = samples[batch_start:batch_start + batch_size]
            waveforms = [s["waveform"] for s in batch_samples]

            try:
                results = reconstruct_batch(eng, waveforms, device, args.num_quantizers)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    print(f"  OOM at batch {batch_start}, falling back to single-sample ...")
                    results = [reconstruct(eng, w, device, args.num_quantizers) for w in waveforms]
                else:
                    raise

            for sample, (original, recon) in zip(batch_samples, results):
                name = sample["name"]
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

            if (batch_start + batch_size) % 200 < batch_size or batch_start + batch_size >= len(samples):
                print(f"  [{codec_name}] {min(batch_start + batch_size, len(samples))}/{len(samples)} done ...")

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
