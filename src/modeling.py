import torch
from typing import Tuple

from transformers import (
    AutoTokenizer, 
    PreTrainedTokenizerBase,
    BitsAndBytesConfig,
    AutoModelForCausalLM
)
from peft import LoraConfig, get_peft_model

from .config import TrainingConfig


def load_tokenizer(cfg: TrainingConfig) -> PreTrainedTokenizerBase:
    """Load tokenizer for CodeLlama."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model_and_tokenizer(cfg: TrainingConfig):
    """
    Load CodeLlama 7B with 4-bit QLoRA (no Unsloth).
    
    Optimized for 4GB VRAM:
    - 4-bit NF4 quantization (reduces 16 GB model to ~4 GB)
    - LoRA adapters (only trainable params)
    - Gradient checkpointing (reduces activation memory)
    - device_map auto (manages memory across CPU/GPU)
    
    Returns (model, tokenizer).
    """
    
    # 4-bit quantization config optimized for RTX 3050 4GB
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,      # Nested quantization
        bnb_4bit_quant_type="nf4",           # NF4 quantization (better quality)
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Load model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=bnb_config,
        device_map="auto",                    # Auto manage GPU/CPU memory
        trust_remote_code=True,
    )

    # Load tokenizer
    tokenizer = load_tokenizer(cfg)

    # LoRA configuration: only fine-tune 0.2% of parameters
    lora_cfg = LoraConfig(
        r=cfg.lora_r,                        # LoRA rank
        lora_alpha=cfg.lora_alpha,           # LoRA scaling
        target_modules=list(cfg.target_modules),
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA to model
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()       # Show param counts

    # Enable gradient checkpointing to save memory during training
    # This trades compute for memory (safe for 4GB)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    return model, tokenizer
