"""
DualCodec wrapper for the benchmarking pipeline.

Example usage:
    wrapper = DualCodecWrapper(
        checkpoint_path="path/to/checkpoint",
        dualcodec_repo="path/to/DualCodec",
    )
    wrapper.reconstruct("input.wav", "output.wav")
"""

import sys
import os
import torch
import torchaudio

from .base import CodecWrapper


class DualCodecWrapper(CodecWrapper):
    def __init__(
        self,
        checkpoint_path: str,
        dualcodec_repo: str,
        model_config: str = "dualcodec_25hz_16384_1024_12vq",
        w2v_path: str = None,
        dualcodec_ckpts: str = None,
        device: str = None,
        num_quantizers: int = 8,
    ):
        # ensure DualCodec is importable
        if dualcodec_repo not in sys.path:
            sys.path.insert(0, dualcodec_repo)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_quantizers = num_quantizers

        w2v_path = w2v_path or os.path.join(dualcodec_repo, "w2v-bert-2.0")
        dualcodec_ckpts = dualcodec_ckpts or os.path.join(dualcodec_repo, "dualcodec_ckpts")

        # build model
        import hydra
        from hydra import initialize_config_dir
        import safetensors.torch

        conf_dir = os.path.join(dualcodec_repo, "dualcodec", "conf", "model")
        with initialize_config_dir(version_base="1.3", config_dir=os.path.abspath(conf_dir)):
            cfg = hydra.compose(config_name=f"{model_config}.yaml", overrides=[])
            model = hydra.utils.instantiate(cfg.model)

        weights_file = os.path.join(checkpoint_path, "model.safetensors")
        safetensors.torch.load_model(model, weights_file, strict=False)
        model.eval()

        from dualcodec.infer.dualcodec.inference_with_semantic import Inference

        self.engine = Inference(
            dualcodec_model=model,
            dualcodec_path=dualcodec_ckpts,
            w2v_path=w2v_path,
            device=self.device,
            autocast=True,
        )

    @torch.no_grad()
    def encode(self, audio_path: str):
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 24000:
            waveform = torchaudio.functional.resample(waveform, sr, 24000)
        audio = waveform.unsqueeze(0).to(self.device)
        semantic_codes, acoustic_codes = self.engine.encode(audio, n_quantizers=self.num_quantizers)
        return {"semantic_codes": semantic_codes, "acoustic_codes": acoustic_codes}

    @torch.no_grad()
    def decode(self, tokens, output_path: str) -> str:
        recon = self.engine.decode(tokens["semantic_codes"], tokens["acoustic_codes"])
        recon = recon.squeeze(0).cpu()
        torchaudio.save(output_path, recon, 24000)
        return output_path

    @torch.no_grad()
    def reconstruct(self, audio_path: str, output_path: str) -> str:
        tokens = self.encode(audio_path)
        return self.decode(tokens, output_path)

    @torch.no_grad()
    def encode_first_token(self, audio_path: str):
        """For TTFAT: measures time to produce first token (= full encode for non-streaming)."""
        return self.encode(audio_path)
