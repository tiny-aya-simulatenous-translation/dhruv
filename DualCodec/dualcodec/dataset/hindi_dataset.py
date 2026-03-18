"""
Dataset classes for DualCodec training on custom data.

Supports two modes:
  1. HuggingFaceAudioDataset  — stream from any HF dataset with an `audio` column
  2. ManifestAudioDataset     — load from a local JSONL manifest
"""

import json
import os
import random
import io
import torch
import torchaudio
from torch.utils.data import IterableDataset


class HuggingFaceAudioDataset(IterableDataset):
    """Streams audio from a HuggingFace dataset with the standard schema:
        - audio: {bytes: binary, path: string}
        - text: string  (ignored for codec training)

    Yields dicts compatible with the DualCodec gluster_filter pipeline
    when is_emilia=False and load_from_tar=False:
        - "wav": placeholder path (audio loaded in-memory by this class)
        - "speech": torch.Tensor (1, T) — the decoded waveform
        - "sample_rate": int
        - "duration": float (seconds)
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        audio_column: str = "audio",
        streaming: bool = True,
        min_duration: float = 1.0,
        max_duration: float = 45.0,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.audio_column = audio_column
        self.streaming = streaming
        self.min_duration = min_duration
        self.max_duration = max_duration

        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq

        print(f"[HuggingFaceAudioDataset] dataset={dataset_name}, split={split}")

    def _decode_audio(self, audio_entry):
        """Decode audio bytes from the HF audio struct into a waveform tensor."""
        import soundfile as sf

        audio_bytes = audio_entry["bytes"]
        buf = io.BytesIO(audio_bytes)

        try:
            # soundfile handles WAV, FLAC, OGG natively — no FFmpeg needed
            data, sr = sf.read(buf)
            waveform = torch.from_numpy(data).float()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.T
        except Exception:
            # fallback for formats soundfile can't handle (e.g. MP3):
            # write to temp file and let torchaudio try all its backends
            import tempfile
            path = audio_entry.get("path", "")
            ext = os.path.splitext(path)[1].lower() if path else ".wav"
            with tempfile.NamedTemporaryFile(suffix=ext, delete=True) as tmp:
                tmp.write(audio_bytes)
                tmp.flush()
                waveform, sr = torchaudio.load(tmp.name)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform, sr

    def __iter__(self):
        from huggingface_hub import HfApi
        import pyarrow.parquet as pq

        api = HfApi()
        repo_files = api.list_repo_files(self.dataset_name, repo_type="dataset")
        parquet_files = sorted(
            [f for f in repo_files if f.endswith(".parquet") and self.split in f]
        )

        if not parquet_files:
            parquet_files = sorted([f for f in repo_files if f.endswith(".parquet")])

        shard_indices = list(range(len(parquet_files)))
        random.shuffle(shard_indices)

        print(
            f"[HuggingFaceAudioDataset] Found {len(parquet_files)} parquet shards"
        )

        from huggingface_hub import hf_hub_download

        for shard_idx in shard_indices:
            fname = parquet_files[shard_idx]
            local_path = hf_hub_download(
                repo_id=self.dataset_name,
                filename=fname,
                repo_type="dataset",
            )
            table = pq.read_table(local_path)

            row_indices = list(range(table.num_rows))
            random.shuffle(row_indices)

            has_duration_col = "duration" in table.column_names

            for row_idx in row_indices:
                try:
                    if has_duration_col:
                        dur = table.column("duration")[row_idx].as_py()
                        if dur < self.min_duration or dur > self.max_duration:
                            continue

                    audio_entry = table.column(self.audio_column)[row_idx].as_py()
                    waveform, sr = self._decode_audio(audio_entry)

                    duration = waveform.shape[-1] / sr
                    if not has_duration_col:
                        if duration < self.min_duration or duration > self.max_duration:
                            continue

                    yield {
                        "wav": audio_entry.get("path", ""),
                        "speech": waveform,
                        "sample_rate": sr,
                        "duration": duration,
                    }
                except Exception as e:
                    print(f"[HuggingFaceAudioDataset] Skipping row {row_idx}: {e}")
                    continue

    def __len__(self):
        return 0  # unknown for streaming


class ManifestAudioDataset(IterableDataset):
    """Reads a JSONL manifest and yields sample dicts compatible with the
    DualCodec data pipeline (gluster_opener -> gluster_filter -> ...).

    Expected manifest format (one JSON object per line):
    {"audio_path": "/path/to/audio.wav", "duration_s": 7.2, ...}
    """

    def __init__(
        self,
        manifest_path: str,
        data_root: str = "",
        min_duration: float = 1.0,
        max_duration: float = 45.0,
        shuffle: bool = True,
    ):
        self.shuffle = shuffle
        self.samples = []

        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)

                duration = entry.get("duration_s", entry.get("duration", 0))
                if duration < min_duration or duration > max_duration:
                    continue

                audio_path = entry.get("audio_path", entry.get("wav", ""))
                if data_root and not os.path.isabs(audio_path):
                    audio_path = os.path.join(data_root, audio_path)

                self.samples.append(
                    {
                        "wav": audio_path,
                        "duration": float(duration),
                        "utt_id": entry.get("utt_id", ""),
                    }
                )

        total_hours = sum(s["duration"] for s in self.samples) / 3600
        print(
            f"[ManifestAudioDataset] Loaded {len(self.samples)} samples "
            f"({total_hours:.1f} hours) from {manifest_path}"
        )

    def __iter__(self):
        indices = list(range(len(self.samples)))
        if self.shuffle:
            random.shuffle(indices)
        for idx in indices:
            yield self.samples[idx]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
