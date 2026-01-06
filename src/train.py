import os
import logging
from pathlib import Path

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


def main():
    cfg = load_config()
    logger.info("Loaded config: %s", cfg)

    # Ensure output dir exists
    output_dir = PROJECT_ROOT / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanity: device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Ensure you installed torch with CUDA and have an NVIDIA GPU.")

    logger.info("CUDA device: %s", torch.cuda.get_device_name(0))

    # 1. Model & tokenizer
    logger.info("Loading model + tokenizer with Unsloth...")
    model, tokenizer = load_model_and_tokenizer(cfg)

    # 2. Data
    logger.info("Preparing datasets...")
    datasets = prepare_datasets(tokenizer, cfg)
    train_ds = datasets["train"]
    eval_ds = datasets["validation"]

    logger.info("Train size: %d, Eval size: %d", len(train_ds), len(eval_ds))

    # 3. Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 4. TrainingArguments
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
        report_to=cfg.report_to,  # set to "wandb" if you want
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 6. Train
    logger.info("Starting training...")
    train_result = trainer.train()
    trainer.save_model(str(output_dir))  # saves adapter (LoRA) weights
    tokenizer.save_pretrained(str(output_dir))

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_ds)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("Training complete. Best model saved to %s", output_dir)


if __name__ == "__main__":
    main()
