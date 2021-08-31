from typing import Optional
import torch
from dataclasses import dataclass
from transformers.file_utils import ModelOutput

from transformers.modeling_outputs import Seq2SeqLMOutput


@dataclass
class Seq2SeqLMVaeOutput(Seq2SeqLMOutput):
    latent: torch.FloatTensor = None
    reconstructed_encoding: torch.FloatTensor = None


@dataclass
class BaseVaeOutput(ModelOutput):
    """
    Base class for a VAE's outputs.

    Args:
        reconstructed_encoding (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Reconstructed hidden states originally from the last layer of the encoder.
        latent (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, latent_size)`):
            Latent codes representing encoded sequences.
        reg_loss (:obj:`torch.FloatTensor` of shape :obj:`(batch_size)`):
            MMD-VAE regularisation loss for this step.
    """

    latent: torch.FloatTensor = None
    reconstructed_encoding: torch.FloatTensor = None
    reg_loss: Optional[torch.FloatTensor] = None
