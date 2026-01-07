from dataclasses import dataclass
from typing import Optional
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

@dataclass
class TrainingConfig:

    # Model / tokenizer
    base_model: str = "codellama/CodeLlama-7b-Instruct-hf"
    max_seq_length: int = 1024  # Keep at 1024 for 4GB VRAM

    # LoRA / QLoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")

    # Dataset
    dataset_name: str = "spider"
    train_split: str = "train"
    eval_split: str = "validation"

    # Training (tuned for RTX 3050 4GB with CPU offload)
    output_dir: str = "models/codellama_spider_lora"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
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
    gradient_checkpointing: bool = False
    optim: str = "adamw_torch"

    # Dataloader
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = False

    # Misc
    report_to: Optional[str] = None

    # Inference / generation
    gen_max_new_tokens: int = 256
    gen_temperature: float = 0.2
    gen_top_p: float = 0.9


def load_config(config_path: Optional[str] = None) -> TrainingConfig:
    """Load config from YAML file if exists, otherwise use defaults."""
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
