from transformers.utils import logging
import torch
from torch import nn

logger = logging.get_logger(__name__)


class VaeEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # TODO allow stacks of linear layers with dropout here to better autoencode
        self.token_to_latent = nn.Linear(config.d_input, config.latent_size)
        self.tanh = nn.Tanh()

    @staticmethod
    def mean_pooling(token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, token_embeddings, attention_mask=None) -> torch.Tensor:
        return self.tanh(self.token_to_latent(self.mean_pooling(token_embeddings, attention_mask)))
