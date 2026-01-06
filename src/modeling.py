from typing import Tuple
import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from peft import LoraConfig, get_peft_model
from .config import TrainingConfig


def load_tokenizer(cfg: TrainingConfig) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model_and_tokenizer(cfg: TrainingConfig):
    """
    Load CodeLlama 7B with Unsloth (4-bit) and apply LoRA.
    Returns (model, tokenizer).
    """
    dtype = torch.float16

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq_length,
        dtype=dtype,
        load_in_4bit=True,  # QLoRA style
    )

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # LoRA configuration
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=list(cfg.target_modules),
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Enable gradient checkpointing for memory
    model.gradient_checkpointing_enable()

    return model, tokenizer
