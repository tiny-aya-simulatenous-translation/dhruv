"""
Base class for codec wrappers.

Each team member implements a subclass that loads their codec and exposes
a uniform encode/decode interface for the benchmarking pipeline.
"""

from abc import ABC, abstractmethod


class CodecWrapper(ABC):
    """
    Minimal interface every codec must implement for benchmarking.

    Subclasses should:
      1. Load the model + weights in __init__
      2. Implement encode(audio_path) -> tokens
      3. Implement decode(tokens) -> audio_path (writes to disk, returns path)
      4. Optionally implement encode_first_token(audio_path) for streaming TTFAT
    """

    @abstractmethod
    def encode(self, audio_path: str):
        """Encode an audio file into discrete tokens. Returns token representation."""
        ...

    @abstractmethod
    def decode(self, tokens, output_path: str) -> str:
        """Decode tokens back to audio, save to output_path. Returns output_path."""
        ...

    @abstractmethod
    def reconstruct(self, audio_path: str, output_path: str) -> str:
        """Full encode-decode round-trip. Returns path to reconstructed audio."""
        ...
