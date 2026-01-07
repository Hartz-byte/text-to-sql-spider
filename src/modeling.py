import torch
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model

from .config import TrainingConfig


def load_tokenizer(cfg: TrainingConfig) -> PreTrainedTokenizerBase:
    """Load CodeLlama tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model_and_tokenizer(cfg: TrainingConfig):
    """
    Load CodeLlama 7B for training with device_map="auto".
    
    For RTX 3050 (4GB) + 16GB RAM:
    - GPU: ~3.5 GB (first 6 layers)
    - CPU: ~9 GB (remaining 26 layers + norm + head)
    - No gradient checkpointing (causes backward pass issues)
    - Use FP16 for memory efficiency
    """
    
    tokenizer = load_tokenizer(cfg)

    print("Loading CodeLlama model with device_map='auto'...")
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    print(f"✓ Model loaded")
    print(f"  GPU: ~3.5 GB (early layers on GPU:0)")
    print(f"  CPU: ~9 GB (later layers + norm + head on CPU)")

    # LoRA
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=list(cfg.target_modules),
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_cfg)
    print(f"\n{'='*60}")
    model.print_trainable_parameters()
    print(f"{'='*60}\n")

    # Only enable input_require_grads, NOT gradient checkpointing
    model.enable_input_require_grads()

    return model, tokenizer
