# DualCodec: Training → Reconstruction → Benchmarking Report

## Overview

This report documents the complete pipeline for finetuning DualCodec on Hindi/Hinglish speech data and evaluating reconstruction quality as part of the TinyAya codec benchmarking effort.

**Codec:** DualCodec (25Hz, 16384 semantic + 1024 acoustic codebook)
**Owner:** Dhruv
**Dataset:** `tiny-aya-translate/hinglish-casual` (~100 hours Hindi/Hinglish)
**Hardware:** Single H100 GPU

---

## 1. Environment Setup

```bash
# Clone and install DualCodec
cd /home/ubuntu/dhruv
git clone https://github.com/tiny-aya-simulatenous-translation/dhruv.git
cd DualCodec
pip install -e ".[tts]"

# Additional dependencies
pip install onnxruntime openai-whisper soundfile

# System dependency for Whisper
sudo apt install ffmpeg -y

# Login to HuggingFace (if dataset is private)
huggingface-cli login
```

---

## 2. Training

### 2.1 Pretrained Model Downloads (automatic)

The training script auto-downloads these on first run:

| Model | Source | Local Path |
|---|---|---|
| DualCodec pretrained | `amphion/dualcodec` | `dualcodec_ckpts/` |
| Wav2Vec2-BERT 2.0 | `facebook/w2v-bert-2.0` | `w2v-bert-2.0/` |

### 2.2 Training Configuration

Config: `dualcodec/conf/dualcodec_ft_hindi.yaml`

| Parameter | Value |
|---|---|
| Base model | `dualcodec_25hz_16384_1024_12vq` |
| Resume type | `finetune` (loads pretrained weights, resets optimizer) |
| Batch size | 6 |
| Segment length | 76800 samples (3.2s at 24kHz) |
| Learning rate | 1e-5 (AdamW) |
| Max steps | 50000 |
| Checkpoint stride | Every 5000 steps |
| Semantic VQ | Enabled |
| Distillation loss weight | 15.0 |
| Mixed precision | Disabled |

### 2.3 Training Command

```bash
cd /home/ubuntu/dhruv/DualCodec

bash scripts/train_hindi.sh tiny-aya-translate/hinglish-casual
```

Or equivalently:

```bash
accelerate launch --num_processes 1 train.py \
    --config-name=dualcodec_ft_hindi \
    machine.hf_dataset_name="tiny-aya-translate/hinglish-casual" \
    trainer.batch_size=6 \
    data.segment_speech.segment_length=76800
```

### 2.4 Training Output

Checkpoints saved to:
```
output_checkpoints/dualcodec_ft_hindi_25hz/checkpoint/
├── epoch-0009_step-0055000_loss-0.000000-dualcodec_ft_hindi_25hz/
│   ├── model.safetensors          # DualCodec generator (~390MB)
│   ├── model_1.safetensors        # Discriminator (~187MB)
│   ├── optimizer.bin              # Generator optimizer
│   ├── optimizer_1.bin            # Discriminator optimizer
│   └── random_states_0.pkl
```

---

## 3. Reconstruction (Inference)

### 3.1 What it does

Encodes audio through the finetuned codec (audio → semantic + acoustic tokens) and decodes back (tokens → waveform). Compares original vs reconstructed audio.

### 3.2 Command: Finetuned model only

```bash
cd /home/ubuntu/dhruv/DualCodec

python3 scripts/reconstruct.py \
    --checkpoint output_checkpoints/dualcodec_ft_hindi_25hz/checkpoint/epoch-0009_step-0055000_loss-0.000000-dualcodec_ft_hindi_25hz \
    --hf_dataset tiny-aya-translate/hinglish-casual \
    --num_samples 50 \
    --benchmark \
    --output_dir /home/ubuntu/dhruv/Benchmarking/data/hindi
```

### 3.3 Command: Finetuned + base pretrained (for comparison)

```bash
python3 scripts/reconstruct.py \
    --checkpoint output_checkpoints/dualcodec_ft_hindi_25hz/checkpoint/epoch-0009_step-0055000_loss-0.000000-dualcodec_ft_hindi_25hz \
    --hf_dataset tiny-aya-translate/hinglish-casual \
    --num_samples 50 \
    --benchmark \
    --also_base \
    --output_dir /home/ubuntu/dhruv/Benchmarking/data/hindi
```

### 3.4 Reconstruction Output

```
Benchmarking/data/hindi/
├── originals/                     # Original audio (24kHz wav)
│   ├── sample_0000.wav
│   └── ...
├── codecs/
│   ├── dualcodec/                 # Finetuned model reconstruction
│   │   ├── sample_0000.wav
│   │   └── ...
│   └── dualcodec_base/            # Pretrained model reconstruction (if --also_base)
│       ├── sample_0000.wav
│       └── ...
├── transcripts.jsonl              # Auto-extracted from HF dataset
├── metrics_dualcodec.json         # Per-sample SNR for finetuned
└── metrics_dualcodec_base.json    # Per-sample SNR for base
```

### 3.5 Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | (required) | Path to finetuned checkpoint directory |
| `--hf_dataset` | None | HuggingFace dataset name |
| `--audio_dir` | None | Local directory of wav files (alternative to HF) |
| `--num_samples` | 10 | Number of samples to reconstruct |
| `--num_quantizers` | 8 | Number of RVQ quantizers (1 semantic + 7 acoustic) |
| `--benchmark` | False | Output in benchmark-ready directory layout |
| `--also_base` | False | Also reconstruct with pretrained base model |
| `--model_config` | `dualcodec_25hz_16384_1024_12vq` | Model architecture config |

---

## 4. Benchmarking

### 4.1 Metrics

| Metric | What it measures | Tool |
|---|---|---|
| **DNSMOS** (1–5) | Perceptual audio quality (overall, signal, background) | ONNX DNSMOS P.835 model |
| **SSNR** (dB) | Segmental signal-to-noise ratio across voiced frames | Custom implementation |
| **WER** | Word error rate — linguistic content preservation | Whisper ASR (`large-v3`) |
| **TTFAT** (ms) | Time to first audio token — encoding latency | Codec wrapper interface |

### 4.2 Benchmark Config

File: `Benchmarking/config.yaml`

```yaml
data_root: ./data
languages:
  - hindi
codecs:
  - dualcodec
  - dualcodec_base
whisper_model: large-v3
```

### 4.3 Running the Benchmark

```bash
cd /home/ubuntu/dhruv/Benchmarking

# All metrics (DNSMOS + SSNR + WER)
python benchmark.py --config config.yaml --codecs dualcodec dualcodec_base

# Speech quality only (faster, no Whisper)
python benchmark.py --config config.yaml --metrics dnsmos ssnr --codecs dualcodec dualcodec_base

# WER only
python benchmark.py --config config.yaml --metrics wer --codecs dualcodec dualcodec_base
```

### 4.4 Benchmark Output

Terminal prints a formatted comparison table:

```
================================================================================
  CODEC BENCHMARK RESULTS
================================================================================

  Language: HINDI
  ────────────────────────────────────────────────────────────
  Codec                           DNSMOS         SSNR (dB)               WER
  ────────────────────────────────────────────────────────────
  dualcodec                3.909 ±0.178      8.425 ±1.015           (pending)
  dualcodec_base               (pending)       (pending)            (pending)
  ────────────────────────────────────────────────────────────
```

Results saved to:
- `benchmark_results/aggregated_results.json` — mean/std per codec per language
- `benchmark_results/per_sample_results.jsonl` — every metric for every sample

---

## 5. Results

### 5.1 DualCodec Finetuned (55k steps) — Hindi

| Metric | Value |
|---|---|
| **DNSMOS OVRL** | 3.91 ± 0.18 |
| **SSNR** | 8.43 ± 1.01 dB |
| **WER** | (pending — requires ffmpeg + Whisper) |

### 5.2 DualCodec Base Pretrained — Hindi

| Metric | Value |
|---|---|
| **DNSMOS OVRL** | (pending) |
| **SSNR** | (pending) |
| **WER** | (pending) |

### 5.3 Comparison (to be filled after running --also_base)

| | Finetuned | Base | Delta |
|---|---|---|---|
| DNSMOS OVRL | 3.91 | — | — |
| SSNR (dB) | 8.43 | — | — |
| WER | — | — | — |

---

## 6. Quick Reference: Full Pipeline Commands

```bash
# === SETUP ===
cd /home/ubuntu/dhruv/DualCodec
pip install -e ".[tts]"
pip install onnxruntime openai-whisper soundfile
sudo apt install ffmpeg -y

# === TRAIN ===
bash scripts/train_hindi.sh tiny-aya-translate/hinglish-casual

# === RECONSTRUCT (finetuned + base) ===
python3 scripts/reconstruct.py \
    --checkpoint output_checkpoints/dualcodec_ft_hindi_25hz/checkpoint/epoch-0009_step-0055000_loss-0.000000-dualcodec_ft_hindi_25hz \
    --hf_dataset tiny-aya-translate/hinglish-casual \
    --num_samples 50 \
    --benchmark --also_base \
    --output_dir /home/ubuntu/dhruv/Benchmarking/data/hindi

# === BENCHMARK ===
cd /home/ubuntu/dhruv/Benchmarking
python benchmark.py --config config.yaml --codecs dualcodec dualcodec_base
```

---

## 7. Repository Structure

```
Tiny_Aya_Speech/
├── DualCodec/
│   ├── train.py                           # Hydra training entry point
│   ├── scripts/
│   │   ├── train_hindi.sh                 # One-command training launcher
│   │   └── reconstruct.py                 # Inference + benchmark output
│   ├── dualcodec/
│   │   ├── conf/                          # Hydra configs (model, data, trainer)
│   │   ├── model_codec/                   # DualCodec model + trainer
│   │   ├── dataset/                       # HF + JSONL dataset loaders
│   │   └── infer/dualcodec/               # Inference engine
│   ├── dualcodec_ckpts/                   # Pretrained weights (auto-downloaded)
│   ├── w2v-bert-2.0/                      # Semantic model (auto-downloaded)
│   └── output_checkpoints/                # Training outputs
│
└── Benchmarking/
    ├── benchmark.py                       # Main benchmarking script
    ├── config.yaml                        # Benchmark configuration
    ├── metrics/
    │   ├── dnsmos.py                      # DNSMOS P.835 scorer
    │   ├── ssnr.py                        # Segmental SNR
    │   ├── wer.py                         # WER via Whisper
    │   └── latency.py                     # TTFAT measurement
    ├── codec_wrappers/
    │   ├── base.py                        # CodecWrapper interface
    │   └── dualcodec_wrapper.py           # DualCodec implementation
    └── data/
        └── hindi/
            ├── originals/                 # Test audio
            ├── codecs/{dualcodec,dualcodec_base}/
            └── transcripts.jsonl
```
