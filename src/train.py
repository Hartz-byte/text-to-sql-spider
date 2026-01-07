import os
import logging
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

from .config import load_config, PROJECT_ROOT
from .modeling import load_model_and_tokenizer
from .data import prepare_datasets


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class FixedTrainer(Trainer):
    """
    Trainer subclass that prevents moving model to device.
    
    When model is loaded with device_map="auto", some params may be on
    meta device. Trainer's _move_model_to_device() tries to call .to(device)
    which fails on meta tensors. This patch skips that step since the model
    is already properly placed by accelerate.
    """
    def _move_model_to_device(self, model, device):
        """Skip moving model - already placed by device_map='auto'."""
        # Do nothing - model is already on correct devices
        return


def main():
    cfg = load_config()
    logger.info("Config loaded")

    output_dir = PROJECT_ROOT / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    logger.info("Loading model...")
    model, tokenizer = load_model_and_tokenizer(cfg)

    logger.info("Preparing datasets...")
    datasets = prepare_datasets(tokenizer, cfg)
    train_ds = datasets["train"]
    eval_ds = datasets["validation"]

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        evaluation_strategy=cfg.eval_strategy,
        save_strategy=cfg.save_strategy,
        load_best_model_at_end=True,
        max_grad_norm=cfg.max_grad_norm,
        seed=cfg.seed,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        optim=cfg.optim,
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=cfg.dataloader_pin_memory,
        report_to=cfg.report_to,
        remove_unused_columns=False,
    )

    # Use FixedTrainer instead of Trainer
    trainer = FixedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info("=" * 80)
    logger.info("STARTING TRAINING (RTX 3050 + 16GB RAM)")
    logger.info("=" * 80)

    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_ds)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info("=" * 80)
    logger.info(f"✓ Training complete! Model: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
