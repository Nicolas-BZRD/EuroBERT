from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class TrainConfig:
    project_name: str = "training"
    reload_checkpoint: Optional[str] = None
    output_dir: str = "output"

    lr: float = 1e-4
    num_epochs: int = 1
    clip_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    optimizer: str = "AdamW"
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-5
    fused: bool | None = False

    lr_scheduler: str = "WarmupStableDecayLR"
    pct_start: float = 0.01
    div_factor: int = 0
    end_start: float = 0
    final_div_factor: int = 0

    # Compilation configurations. These configurations are described in the
    # torch.compile documentation.
    # https://pytorch.org/docs/stable/generated/torch.compile.html
    compile_model: bool = False  # Compile model
    compile_mode: str | None = None  # Compilation mode
    compile_options: dict | None = None  # Compilation options

    # Validation configurations
    run_validation: bool = True
    validation_step: int = 5000

    # Save configurations
    save_step: int = 5000
    save_model: bool = True
    save_optimizer: bool = True
    save_scheduler: bool = True
    save_data_loader: bool = True
    save_config: bool = True

    # Masking configurations
    mlm_probability: float = 0.5
    mask_probability: float = 1.0
    random_probability: float = 0.0
    original_probability: float = 0.0

    # Reloading configurations
    skip_reload_scheduler: bool = False
    skip_reload_dataloader: bool = False
    skip_reload_tensorboard: bool = False

    # Other configurations
    fsdp: bool = False
    ddp: bool = False
    mixed_bfloat16: bool = True
    seed: int = 42
    tensorboard: bool = True
    profile: bool = False
    exit_end_profiling: bool = True
    profiler_output: Literal["chrome", "tensorboard"] = "chrome"
    log_every_n_steps: int = 10
