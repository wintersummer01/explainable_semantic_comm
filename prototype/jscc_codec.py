"""Split a trained stage-compatible JSCC model into encoder and decoder calls."""

import torch
from einops import rearrange


ENCODER_STAGES = ("enc_1", "enc_2", "enc_3_0", "enc_3_1", "enc_3_2", "enc_4", "enc_proj")
DECODER_STAGES = ("dec_proj", "dec_1", "dec_2_0", "dec_2_1", "dec_2_2", "dec_3", "dec_4")


class JSCCCodec:
    def __init__(self, network):
        missing = [name for name in ENCODER_STAGES + DECODER_STAGES if not hasattr(network, name)]
        if missing:
            raise TypeError(f"Model is not stage-compatible; missing: {', '.join(missing)}")
        self.network = network

    @torch.inference_mode()
    def encode(self, image):
        latent = image
        for name in ENCODER_STAGES:
            latent = getattr(self.network, name)(latent)
        symbols = rearrange(latent, "b (c iq) h w -> b (c h w) iq", iq=2)
        symbols = torch.complex(symbols[..., 0], symbols[..., 1])
        return symbols, latent.shape

    @torch.inference_mode()
    def decode(self, symbols, latent_shape):
        _, channels, height, width = latent_shape
        real = torch.stack((symbols.real, symbols.imag), dim=-1)
        latent = rearrange(real, "b (c h w) iq -> b (c iq) h w", h=height, w=width)
        if latent.shape[1] != channels:
            raise ValueError(f"Expected {channels} latent channels, got {latent.shape[1]}")
        image = latent
        for name in DECODER_STAGES:
            image = getattr(self.network, name)(image)
        return image
