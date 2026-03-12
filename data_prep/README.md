# Hindi Audio Data Preparation Pipeline

Codec-agnostic pipeline to curate **~200-500 hours of clean monolingual Hindi speech at 24kHz** for:

1. Benchmarking audio codecs (Mimi, BiCodec, DualCodec, Spark, etc.)
2. Stage 1 audio pretraining with whichever codec wins

The pipeline is **completely independent** of any codec or model code. It produces clean WAV files and structured manifests -- codec tokenization happens downstream.

## Pipeline Overview

```
Download → Standardize → Score (DNSMOS) → Analyze (notebook) → Filter + Dedup → Package
```

Each step reads/writes JSONL manifests. Steps are **idempotent** -- if a step crashes halfway, re-run it and it skips already-processed samples.

## Directory Structure

```
data_prep/
├── config.yaml                  # All paths, thresholds, dataset configs
├── requirements.txt             # Python dependencies
├── run_trial.sh                 # One-command trial run on FLEURS Hindi
├── download.py                  # Step 1: Download datasets → raw manifest
├── standardize.py               # Step 2: Resample, normalize, trim, filter
├── score_quality.py             # Step 3: Compute DNSMOS scores (ONNX, CPU)
├── filter_dedup.py              # Step 4: Apply thresholds + dedup by transcript
├── package.py                   # Step 5: Train/val split, export Parquet + JSONL
├── notebooks/
│   └── quality_analysis.ipynb   # Step 3b: Score distributions + threshold tuning
├── utils/
│   ├── __init__.py
│   ├── audio.py                 # Load, resample, trim, loudness normalize
│   ├── dnsmos.py                # DNSMOS ONNX model download + scoring
│   └── manifest.py              # JSONL read/write/append helpers
├── models/                      # Auto-downloaded DNSMOS ONNX model
└── data/                        # All generated data (raw, standardized, scored, final)
```

## Prerequisites

- **Python 3.10+** (tested on 3.14)
- **CPU only** -- no GPU required for any step

```bash
cd data_prep
pip install -r requirements.txt
```

> **Note:** `datasets<4.0.0` is pinned because FLEURS Hindi uses a legacy HuggingFace dataset script. If you only use Common Voice / IndicVoices-R / OpenSLR, this pin can be relaxed.

## Quick Start (Trial Run)

Run the full pipeline on FLEURS Hindi (~10 hours, ~15-25 min on a Mac):

```bash
bash run_trial.sh
```

This executes all 5 steps automatically. Afterwards, open `notebooks/quality_analysis.ipynb` to inspect score distributions and threshold choices.

## Full Pipeline Usage

### Step 1: Download Datasets

```bash
python download.py --config config.yaml --datasets common_voice_hi indicvoices_r openslr_hindi
```

Downloads raw audio from each source and writes a unified manifest to `data/raw/manifest.jsonl`.

Available datasets: `common_voice_hi`, `indicvoices_r`, `openslr_hindi`, `fleurs_hi`

Use `--max_samples N` to cap samples per dataset (useful for testing).

### Step 2: Standardize

```bash
python standardize.py --config config.yaml --num_workers 8
```

For each sample:
1. Load audio (supports WAV, FLAC, OGG)
2. Convert to mono
3. Resample to **24kHz**
4. Trim leading/trailing silence (energy-based, configurable threshold)
5. Loudness normalize to **-23 LUFS**
6. Skip if duration < 3s or > 45s after trimming
7. Peak-clip to [-1.0, 1.0]
8. Save as 24kHz 16-bit mono WAV

Output: `data/standardized/manifest.jsonl`

### Step 3: Score Quality (DNSMOS)

```bash
python score_quality.py --config config.yaml --num_workers 8
```

Runs Microsoft's DNSMOS P.835 model (ONNX, CPU) on each sample. The model is auto-downloaded on first run from HuggingFace. Produces three scores per sample (all on a 1-5 scale):

| Score | Meaning | Role |
|-------|---------|------|
| **OVRL** | Overall quality | Primary filtering gate |
| **SIG** | Signal/speech quality | Detects distorted speech |
| **BAK** | Background noise quality | Replaces need for separate SNR metric |

Output: `data/scored/manifest.jsonl`

### Step 3b: Analyze & Pick Thresholds

Open `notebooks/quality_analysis.ipynb` and run all cells. It provides:

- DNSMOS score distribution histograms per dataset
- Per-dataset summary statistics (mean, std, percentiles)
- **"Surviving hours vs. threshold"** curve -- the key plot for picking your cutoff
- Joint OVRL + BAK heatmap
- Audio playback for samples near the threshold boundary

Update `config.yaml` with your chosen thresholds before proceeding.

**Default thresholds** (validated in the trial run):
- `dnsmos_ovrl_threshold: 3.0`
- `dnsmos_bak_threshold: 2.5`

### Step 4: Filter + Deduplicate

```bash
python filter_dedup.py --config config.yaml
```

1. **Quality gate:** Drop samples below DNSMOS OVRL and BAK thresholds
2. **Transcript dedup:** Group by normalized transcript, keep the highest-DNSMOS sample per unique sentence (prevents the model from memorizing text patterns)

Override thresholds on the fly: `--ovrl_threshold 3.2 --bak_threshold 3.0`

Skip dedup if desired: `--skip_dedup`

Output: `data/final/manifest.jsonl`

### Step 5: Package

```bash
python package.py --config config.yaml
```

1. **Split:** 95% train / 5% validation, stratified by source dataset
2. **Export:** Parquet files + JSONL manifests for both splits

Output directory: `data/final/hf_dataset/`

```
hf_dataset/
├── train.parquet
├── train.jsonl
├── validation.parquet
└── validation.jsonl
```

To push to HuggingFace Hub:

```bash
python package.py --config config.yaml --push --hub_repo "your-org/hindi-speech-24k"
```

> **Note:** The HuggingFace Arrow format (`save_to_disk`) may fail on Python 3.14 due to a `dill`/`pickle` compatibility bug. Parquet + JSONL outputs are always saved successfully regardless.

## Output Format

Each sample in the final manifest:

```json
{
  "utt_id": "cv_hi_train_000001",
  "audio_path": "data/standardized/cv_hi_train_000001.wav",
  "duration_s": 7.2,
  "transcript": "...",
  "source_dataset": "common_voice_hi",
  "dnsmos_ovrl": 3.82,
  "dnsmos_sig": 3.91,
  "dnsmos_bak": 4.10
}
```

Audio: 24kHz, mono, 16-bit WAV, loudness-normalized to -23 LUFS. Codecs needing 16kHz can trivially downsample at benchmark time.

## Dataset Sources

### Tier 1 -- Start here (target: 200-400 hrs clean)

| Dataset | Est. Hindi Hours | Quality | Access |
|---------|-----------------|---------|--------|
| IndicVoices-R | 50-150 hrs | High (TTS-processed) | HuggingFace |
| Common Voice Hindi | 150-200 hrs (validated) | Medium (crowdsourced, variable) | HuggingFace |
| OpenSLR Hindi | ~100 hrs | High (clean read speech) | Direct download |
| FLEURS Hindi | ~10 hrs | Medium (read speech, 16kHz) | HuggingFace |

### Tier 2 -- Add only if Tier 1 falls short

| Dataset | Notes |
|---------|-------|
| SPRING-INX | Large but may require registration |
| IndicVoices (full) | Much larger but less curated |
| Gram Vaani | 1111 hrs but very noisy telephone speech; requires restoration |

## Configuration

All parameters are centralized in `config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_rate` | 24000 | Output sample rate (Hz) |
| `loudness_lufs` | -23.0 | Target loudness |
| `min_duration_s` | 3.0 | Minimum sample duration after trimming |
| `max_duration_s` | 45.0 | Maximum sample duration |
| `trim_silence` | true | Enable silence trimming |
| `trim_db_threshold` | 30 | dB threshold for silence detection |
| `num_workers` | 8 | Parallel workers for standardize/scoring |
| `dnsmos_ovrl_threshold` | 3.0 | DNSMOS OVRL quality gate |
| `dnsmos_bak_threshold` | 2.5 | DNSMOS BAK noise gate |

## Trial Run Results (FLEURS Hindi)

| Stage | Samples | Hours |
|-------|---------|-------|
| Raw (downloaded) | 3,092 | ~10 |
| Standardized (after duration filter) | 2,775 | 8.2 |
| After quality filter (OVRL >= 3.0, BAK >= 2.5) | 1,316 | 3.9 |
| After transcript dedup | 1,056 | 3.1 |

DNSMOS score summary:

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| OVRL | 2.95 | 0.57 | 0.97 | 4.28 |
| SIG | 3.77 | -- | -- | -- |
| BAK | 3.02 | -- | -- | -- |

~48% survival rate at OVRL >= 3.0 is expected (threshold sits at the median). If this rate holds across larger Tier 1 datasets, the full pipeline should yield 200-300+ clean hours.

## Escape Hatch: Speech Restoration

If Tier 1 datasets don't yield enough clean hours, borderline-quality samples (DNSMOS OVRL 2.5-3.0) can be restored using [Sidon](https://github.com/sarulab-speech/Sidon) before re-scoring. This is not a default pipeline step -- it requires a separate GPU environment with w2v-BERT 2.0 dependencies.

