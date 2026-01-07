from typing import Dict, Any
from datasets import load_dataset, DatasetDict
from transformers import PreTrainedTokenizerBase
from .config import TrainingConfig


SYSTEM_PROMPT = (
    "You are an expert text-to-SQL model. "
    "Given a database schema and a natural language question, "
    "generate a correct SQL query that answers the question.\n"
)


def build_schema_string(example: Dict[str, Any]) -> str:
    """Build schema string from example."""
    # Simple fallback: just use db_id
    # For full schema, download original Spider files from Yale repo
    db_id = example.get("db_id", "unknown")
    return f"Database: {db_id}\n"


def format_example(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Format Spider example into instruction-tuning format.
    
    Spider HF dataset fields:
    - "question": str (natural language query)
    - "query": str (SQL query) ← KEY: NOT "sql"
    - "db_id": str (database identifier)
    """
    schema_str = build_schema_string(example)
    question = example["question"]
    query = example["query"]  # ← FIX: Spider uses "query", not "sql"

    prompt = (
        f"<s>[INST] {SYSTEM_PROMPT}\n"
        f"Schema:\n{schema_str}\n\n"
        f"Question: {question}\n"
        f"[/INST]\n"
        f"SQL:"
    )

    full_text = f"{prompt} {query}</s>"

    return {"text": full_text}


def tokenize_function(examples: Dict[str, Any], tokenizer: PreTrainedTokenizerBase, cfg: TrainingConfig):
    """Tokenize batch."""
    return tokenizer(
        examples["text"],
        max_length=cfg.max_seq_length,
        truncation=True,
        padding="max_length",
    )


def prepare_datasets(tokenizer: PreTrainedTokenizerBase, cfg: TrainingConfig) -> DatasetDict:
    """Load Spider, format, tokenize."""
    print("Loading Spider dataset...")
    raw = load_dataset(cfg.dataset_name)

    print("Formatting examples...")
    train_ds = raw[cfg.train_split].map(format_example)
    eval_ds = raw[cfg.eval_split].map(format_example)

    print("Tokenizing...")
    train_ds = train_ds.map(
        lambda batch: tokenize_function(batch, tokenizer, cfg),
        batched=True,
        batch_size=32,
        remove_columns=train_ds.column_names,
    )
    eval_ds = eval_ds.map(
        lambda batch: tokenize_function(batch, tokenizer, cfg),
        batched=True,
        batch_size=32,
        remove_columns=eval_ds.column_names,
    )

    train_ds = train_ds.with_format("torch")
    eval_ds = eval_ds.with_format("torch")

    train_ds = train_ds.map(lambda ex: {"labels": ex["input_ids"]})
    eval_ds = eval_ds.map(lambda ex: {"labels": ex["input_ids"]})

    print(f"✓ Train: {len(train_ds)} examples")
    print(f"✓ Eval: {len(eval_ds)} examples")

    return DatasetDict(train=train_ds, validation=eval_ds)
