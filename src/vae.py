from typing import Optional
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from simple_vae.vae_decoder import VaeDecoder
from simple_vae.vae_encoder import VaeEncoder
from simple_vae.outputs import BaseVaeOutput


class VAE(nn.Module):
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
    ):
        super().__init__()

        if encoder is None:
            encoder = VaeEncoder(config)

        if decoder is None:
            decoder = VaeDecoder(config)

        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        input_encoding=None,
        latent=None,
        attention_mask=None,
    ):
        if input_encoding is None and latent is None:
            raise ValueError("Both `input_encoding` and `latent` sent to VAE are None.")

        if latent is None:
            latent = self.encoder(input_encoding, attention_mask=attention_mask)

        recon_encoding = self.decoder(latent)

        return BaseVaeOutput(latent=latent, reconstructed_encoding=recon_encoding)
