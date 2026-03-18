#!/usr/bin/env python3
"""
Codec Benchmarking Pipeline for TinyAya Speech-to-Speech Translation.

Evaluates neural speech codecs on audio reconstruction quality across
Hindi and Turkish using four metrics:
  - DNSMOS (perceptual audio quality, 1–5 scale)
  - SSNR  (segmental signal-to-noise ratio, dB)
  - WER   (word error rate via Whisper ASR)
  - TTFAT (time to first audio token, ms)

Expected directory layout:
  data_root/
  ├── hindi/
  │   ├── originals/           # original wav files
  │   ├── transcripts.jsonl    # {"file": "001.wav", "text": "..."}
  │   └── codecs/
  │       ├── mimi/            # reconstructed wavs (same filenames)
  │       ├── dualcodec/
  │       ├── kanade/
  │       └── bicodec/
  └── turkish/
      ├── originals/
      ├── transcripts.jsonl
      └── codecs/
          ├── mimi/
          ├── dualcodec/
          ├── kanade/
          └── bicodec/

Usage:
  python benchmark.py --config config.yaml
  python benchmark.py --config config.yaml --metrics dnsmos ssnr
  python benchmark.py --config config.yaml --codecs dualcodec mimi
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from metrics.dnsmos import compute_dnsmos
from metrics.ssnr import compute_ssnr
from metrics.wer import compute_wer


LANG_TO_WHISPER = {"hindi": "hi", "turkish": "tr"}


def load_transcripts(transcript_path: str) -> dict[str, str]:
    """Load transcripts.jsonl → {filename: text}."""
    transcripts = {}
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            fname = entry.get("file", entry.get("filename", entry.get("utt_id", "")))
            text = entry.get("text", entry.get("transcript", ""))
            if fname and text:
                transcripts[fname] = text
    return transcripts


def find_audio_pairs(originals_dir: str, recon_dir: str) -> list[dict]:
    """Match original files with reconstructed files by filename."""
    exts = {".wav", ".flac", ".mp3", ".ogg"}
    orig_files = {
        f.name: f
        for f in Path(originals_dir).iterdir()
        if f.suffix.lower() in exts
    }
    pairs = []
    for fname, orig_path in sorted(orig_files.items()):
        recon_path = Path(recon_dir) / fname
        if recon_path.exists():
            pairs.append({
                "filename": fname,
                "original": str(orig_path),
                "reconstructed": str(recon_path),
            })
    return pairs


def run_benchmark(config: dict, metric_names: list[str], codec_filter: list[str] | None):
    """Run the full benchmark and return structured results."""
    data_root = config["data_root"]
    languages = config["languages"]
    codecs = config["codecs"]
    dnsmos_model_dir = config.get("dnsmos_model_dir", "models/dnsmos")
    whisper_model = config.get("whisper_model", "large-v3")
    output_dir = Path(config.get("output_dir", "benchmark_results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if codec_filter:
        codecs = [c for c in codecs if c in codec_filter]

    results = defaultdict(lambda: defaultdict(dict))
    all_sample_results = []

    for lang in languages:
        lang_dir = os.path.join(data_root, lang)
        originals_dir = os.path.join(lang_dir, "originals")
        transcript_path = os.path.join(lang_dir, "transcripts.jsonl")
        whisper_lang = LANG_TO_WHISPER.get(lang, lang[:2])

        transcripts = {}
        if os.path.exists(transcript_path):
            transcripts = load_transcripts(transcript_path)
            print(f"[{lang}] Loaded {len(transcripts)} transcripts")
        elif "wer" in metric_names:
            print(f"[{lang}] WARNING: transcripts.jsonl not found, skipping WER")

        for codec in codecs:
            recon_dir = os.path.join(lang_dir, "codecs", codec)
            if not os.path.isdir(recon_dir):
                print(f"[{lang}/{codec}] Reconstructed directory not found, skipping")
                continue

            pairs = find_audio_pairs(originals_dir, recon_dir)
            if not pairs:
                print(f"[{lang}/{codec}] No matching audio pairs found, skipping")
                continue

            print(f"\n{'='*60}")
            print(f"  Evaluating: {codec} / {lang}  ({len(pairs)} samples)")
            print(f"{'='*60}")

            sample_metrics = []

            for pair in tqdm(pairs, desc=f"{codec}/{lang}"):
                fname = pair["filename"]
                orig = pair["original"]
                recon = pair["reconstructed"]
                m = {"filename": fname, "codec": codec, "language": lang}

                if "dnsmos" in metric_names:
                    try:
                        dnsmos_orig = compute_dnsmos(orig, model_dir=dnsmos_model_dir)
                        dnsmos_recon = compute_dnsmos(recon, model_dir=dnsmos_model_dir)
                        m["dnsmos_orig_ovrl"] = dnsmos_orig["dnsmos_ovrl"]
                        m["dnsmos_recon_ovrl"] = dnsmos_recon["dnsmos_ovrl"]
                        m["dnsmos_recon_sig"] = dnsmos_recon["dnsmos_sig"]
                        m["dnsmos_recon_bak"] = dnsmos_recon["dnsmos_bak"]
                    except Exception as e:
                        print(f"  DNSMOS error on {fname}: {e}")

                if "ssnr" in metric_names:
                    try:
                        ssnr = compute_ssnr(orig, recon)
                        m["ssnr_db"] = ssnr["ssnr_db"]
                    except Exception as e:
                        print(f"  SSNR error on {fname}: {e}")

                if "wer" in metric_names and fname in transcripts:
                    try:
                        wer_result = compute_wer(
                            recon, transcripts[fname],
                            language=whisper_lang,
                            whisper_model=whisper_model,
                        )
                        m["wer"] = wer_result["wer"]
                        m["hyp_text"] = wer_result["hyp_text"]
                    except Exception as e:
                        print(f"  WER error on {fname}: {e}")

                sample_metrics.append(m)

            # aggregate
            agg = _aggregate(sample_metrics, metric_names)
            results[lang][codec] = agg
            all_sample_results.extend(sample_metrics)

            _print_codec_summary(codec, lang, agg)

    # save per-sample results
    samples_path = output_dir / "per_sample_results.jsonl"
    with open(samples_path, "w", encoding="utf-8") as f:
        for m in all_sample_results:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # save aggregated results
    agg_path = output_dir / "aggregated_results.json"
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(dict(results), f, indent=2, ensure_ascii=False)

    return dict(results)


def _aggregate(sample_metrics: list[dict], metric_names: list[str]) -> dict:
    """Compute mean/std for each metric across samples."""
    agg = {}
    keys_to_agg = []
    if "dnsmos" in metric_names:
        keys_to_agg += ["dnsmos_orig_ovrl", "dnsmos_recon_ovrl", "dnsmos_recon_sig", "dnsmos_recon_bak"]
    if "ssnr" in metric_names:
        keys_to_agg.append("ssnr_db")
    if "wer" in metric_names:
        keys_to_agg.append("wer")

    for key in keys_to_agg:
        vals = [m[key] for m in sample_metrics if key in m]
        if vals:
            arr = np.array(vals)
            agg[key] = {"mean": round(float(arr.mean()), 4), "std": round(float(arr.std()), 4)}

    agg["num_samples"] = len(sample_metrics)
    return agg


def _print_codec_summary(codec: str, lang: str, agg: dict):
    parts = [f"  {codec}/{lang} ({agg['num_samples']} samples):"]
    if "dnsmos_recon_ovrl" in agg:
        parts.append(f"  DNSMOS OVRL: {agg['dnsmos_recon_ovrl']['mean']:.2f} +/- {agg['dnsmos_recon_ovrl']['std']:.2f}")
    if "ssnr_db" in agg:
        parts.append(f"  SSNR: {agg['ssnr_db']['mean']:.2f} +/- {agg['ssnr_db']['std']:.2f} dB")
    if "wer" in agg:
        parts.append(f"  WER: {agg['wer']['mean']:.4f} +/- {agg['wer']['std']:.4f}")
    print("\n".join(parts))


def print_results_table(results: dict, metric_names: list[str]):
    """Print a formatted Codec x Language comparison table."""
    if not results:
        return

    languages = sorted(results.keys())
    all_codecs = sorted({c for lang_data in results.values() for c in lang_data})

    # build column specs based on enabled metrics
    col_specs = []
    if "dnsmos" in metric_names:
        col_specs.append(("DNSMOS", "dnsmos_recon_ovrl", ""))
    if "ssnr" in metric_names:
        col_specs.append(("SSNR (dB)", "ssnr_db", ""))
    if "wer" in metric_names:
        col_specs.append(("WER", "wer", ""))

    print(f"\n{'='*80}")
    print("  CODEC BENCHMARK RESULTS")
    print(f"{'='*80}\n")

    for lang in languages:
        print(f"  Language: {lang.upper()}")
        print(f"  {'─'*60}")

        header = f"  {'Codec':<20}"
        for name, _, _ in col_specs:
            header += f"{name:>18}"
        print(header)
        print(f"  {'─'*60}")

        for codec in all_codecs:
            if codec not in results[lang]:
                continue
            agg = results[lang][codec]
            row = f"  {codec:<20}"
            for _, key, _ in col_specs:
                if key in agg:
                    val = agg[key]["mean"]
                    std = agg[key]["std"]
                    row += f"{val:>10.3f} ±{std:<6.3f}"
                else:
                    row += f"{'N/A':>18}"
            print(row)

        print(f"  {'─'*60}\n")

    print(f"{'='*80}")


def run_ttfat(config: dict, codec_wrappers: dict[str, object] | None = None):
    """
    Run TTFAT measurement for codecs that provide a wrapper.

    codec_wrappers: {"dualcodec": <DualCodecWrapper instance>, ...}
    Returns dict[codec_name] -> ttfat results.
    """
    if not codec_wrappers:
        print("No codec wrappers provided — skipping TTFAT measurement.")
        print("To measure TTFAT, import and instantiate codec wrappers in run_ttfat_standalone.py")
        return {}

    from metrics.latency import measure_ttfat

    data_root = config["data_root"]
    languages = config["languages"]
    ttfat_samples = config.get("ttfat_num_samples", 5)

    results = {}
    for codec_name, wrapper in codec_wrappers.items():
        codec_latencies = {}
        for lang in languages:
            originals_dir = os.path.join(data_root, lang, "originals")
            if not os.path.isdir(originals_dir):
                continue
            exts = {".wav", ".flac", ".mp3", ".ogg"}
            audio_files = sorted(
                f for f in Path(originals_dir).iterdir() if f.suffix.lower() in exts
            )[:ttfat_samples]

            if not audio_files:
                continue

            all_ttfat = []
            for audio_file in audio_files:
                ttfat = measure_ttfat(wrapper, str(audio_file), num_runs=3, warmup_runs=1)
                all_ttfat.append(ttfat["ttfat_mean_ms"])

            arr = np.array(all_ttfat)
            codec_latencies[lang] = {
                "ttfat_mean_ms": round(float(arr.mean()), 2),
                "ttfat_std_ms": round(float(arr.std()), 2),
            }
            print(f"  TTFAT {codec_name}/{lang}: {arr.mean():.1f} ± {arr.std():.1f} ms")

        results[codec_name] = codec_latencies
    return results


def main():
    parser = argparse.ArgumentParser(description="Codec Benchmarking Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--metrics", nargs="+", default=["dnsmos", "ssnr", "wer"],
        choices=["dnsmos", "ssnr", "wer", "ttfat"],
        help="Which metrics to compute",
    )
    parser.add_argument(
        "--codecs", nargs="+", default=None,
        help="Subset of codecs to evaluate (default: all in config)",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # quality + WER metrics (work from pre-computed reconstructed audio)
    quality_metrics = [m for m in args.metrics if m != "ttfat"]
    if quality_metrics:
        results = run_benchmark(config, quality_metrics, args.codecs)
        print_results_table(results, quality_metrics)

    # TTFAT requires live codec wrappers — print instructions if requested
    if "ttfat" in args.metrics:
        print("\n[TTFAT] To measure latency, use run_ttfat_standalone.py with codec wrappers.")
        print("        See codec_wrappers/ for examples.\n")


if __name__ == "__main__":
    main()
