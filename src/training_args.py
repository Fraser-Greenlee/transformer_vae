from dataclasses import dataclass, field

from transformers.training_args import TrainingArguments

from simple_vae.reg_loss import REG_LOSSES


@dataclass
class VaeTrainingArguments(TrainingArguments):
    reg_loss_type: str = field(
        default='MMD', metadata={"help": f"The reg loss type to use, choose one of {', '.join(REG_LOSSES.keys())}."},
    )
    reg_schedule_k: float = field(default=0.0025,   metadata={"help": "Multiplied by global_step in a sigmoid, more gradually increase regulariser loss weight."})
    reg_schedule_b: float = field(default=6.25,     metadata={"help": "Added to global step in sigmoid, further delays increase in regulariser loss weight."})
    no_reg_loss: bool = field(default=False, metadata={"help": "Disable the VAE reg loss."})
    recon_loss: bool = field(default=False, metadata={"help": "Use a reconstruction loss on input encodings."})

    def __post_init__(self):
        super().__post_init__()
        if self.reg_loss_type not in REG_LOSSES:
            raise Exception(f"Invalid self.reg_loss_type=`{self.reg_loss_type}` must be one of {', '.join(REG_LOSSES.keys())}.")
