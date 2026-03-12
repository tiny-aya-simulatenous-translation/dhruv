"""JSONL manifest read/write utilities."""

import json
from pathlib import Path
from typing import Iterator


def read_manifest(path: str | Path) -> list[dict]:
    """Read a JSONL manifest file into a list of dicts."""
    path = Path(path)
    if not path.exists():
        return []
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def iter_manifest(path: str | Path) -> Iterator[dict]:
    """Iterate over a JSONL manifest without loading all into memory."""
    path = Path(path)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_manifest(path: str | Path, entries: list[dict]) -> None:
    """Write a list of dicts to a JSONL manifest file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def append_manifest(path: str | Path, entry: dict) -> None:
    """Append a single entry to a JSONL manifest file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def get_processed_ids(path: str | Path) -> set[str]:
    """Get set of utt_ids already in a manifest. Useful for resuming."""
    return {entry["utt_id"] for entry in iter_manifest(path)}
