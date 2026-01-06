from dataclasses import dataclass
from typing import Optional
import yaml
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainingConfig:
    # Model / tokenizer
    base_model: str = "codellama/CodeLlama-7b-Instruct-hf"
    max_seq_length: int = 1024  # 2048 possible, but 1024 is safer on 4GB

    # LoRA / QLoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")

    # Dataset
    dataset_name: str = "spider"
    train_split: str = "train"
    eval_split: str = "validation"

    # Training
    output_dir: str = "models/codellama_spider_lora"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1   # 4GB GPU -> keep at 1
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8   # effective batch ~8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    max_grad_norm: float = 1.0
    seed: int = 42

    # Hardware / performance
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"

    # Dataloader
    dataloader_num_workers: int = 2
    dataloader_pin_memory: bool = True

    # Misc
    report_to: Optional[str] = None  # "wandb" if you want

    # Inference / generation
    gen_max_new_tokens: int = 256
    gen_temperature: float = 0.2
    gen_top_p: float = 0.9


def load_config(config_path: Optional[str] = None) -> TrainingConfig:
    """Optionally load overrides from a YAML file."""
    cfg = TrainingConfig()

    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    else:
        config_path = Path(config_path)

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    return cfg
