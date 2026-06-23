from dataclasses import dataclass, field
from pathlib import Path



#-------------- GLOBAL VARIABLES -----------------#

REPO_ROOT = Path(__file__).resolve().parents[0]

BATCH_SIZE = 256
MAIN_SEED = 42



#----------------- MODEL CONFIG ------------------#

@dataclass
class VisionEncoderConfig:
    d_model:        int = 192
    image_size:     int = 224
    patch_size:     int = 16
    n_layers:       int = 12
    n_heads:        int = 3
    d_mlp:          int = None # default = 4 * d_model
    d_embedding:    int = 256
    n_patches:      int = field(init=False) # avoid n_patches from being passed in during initialization

    def __post_init__(self):
        if self.d_mlp is None:
            self.d_mlp = 4 * self.d_model

        self.n_patches = (self.image_size // self.patch_size)**2


@dataclass
class TextEncoderConfig:
    d_model:        int = 192
    max_seq_len:    int = 70
    vocab_size:     int = 38
    n_layers:       int = 4
    n_heads:        int = 4
    d_mlp:          int = None # default = 4 * d_model
    d_embedding:    int = 256
    pad_token_id:   int = 0
    eos_token_id:   int = 2

    def __post_init__(self):
        if self.d_mlp is None:
            self.d_mlp = 4 * self.d_model


@dataclass(frozen=True)
class TrainingConfig:
    train_seed:             int = MAIN_SEED

    # Training
    num_epochs:             int = 50
    batch_size:             int = BATCH_SIZE
    learning_rate:          float = 5e-4
    weight_decay:           float = 0.2
    warmup_epochs:          int = 5
    grad_clip:              float = 1.0

    # Checkpointing
    checkpoint_dir:         Path = REPO_ROOT / "data" / "checkpoints"
    checkpoint_every:       int = 1