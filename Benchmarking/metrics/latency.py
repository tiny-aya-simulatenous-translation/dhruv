"""
Time to First Audio Token (TTFAT) measurement.

Measures how quickly a codec produces its first discrete token from raw audio.
Uses the CodecWrapper interface so each codec team just implements encode().
"""

import time
import numpy as np
import soundfile as sf


def measure_ttfat(
    codec_wrapper,
    audio_path: str,
    num_runs: int = 5,
    warmup_runs: int = 2,
) -> dict:
    """
    Measure Time to First Audio Token for a codec.

    The codec_wrapper must implement encode_first_token(audio_path) -> token,
    which returns as soon as the first token is available. If not available,
    falls back to full encode() and measures that instead.

    Args:
        codec_wrapper: Instance of CodecWrapper (or subclass).
        audio_path:    Path to input audio file.
        num_runs:      Number of timed runs (after warmup).
        warmup_runs:   Number of warmup runs (not counted).

    Returns:
        dict with "ttfat_mean_ms", "ttfat_std_ms", "ttfat_min_ms".
    """
    has_streaming = hasattr(codec_wrapper, "encode_first_token")
    fn = codec_wrapper.encode_first_token if has_streaming else codec_wrapper.encode

    # warmup (GPU JIT, model caching, etc.)
    for _ in range(warmup_runs):
        fn(audio_path)

    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        fn(audio_path)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    arr = np.array(latencies)
    return {
        "ttfat_mean_ms": round(float(arr.mean()), 2),
        "ttfat_std_ms": round(float(arr.std()), 2),
        "ttfat_min_ms": round(float(arr.min()), 2),
    }
