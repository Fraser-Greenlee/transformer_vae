from transformers.utils import logging
import torch
from torch import nn

logger = logging.get_logger(__name__)


class VaeDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # TODO allow stacks of linear layers with dropout here to better autoencode
        self.latent_to_token = nn.Linear(config.latent_size, config.d_output)

        if config.vae_decoder_use_layer_norm:
            self.norm = nn.LayerNorm(config.d_output, config.layer_norm_eps)
        else:
            self.norm = lambda x: x

    def forward(self, latent) -> torch.Tensor:
        return self.norm(self.latent_to_token(latent))
