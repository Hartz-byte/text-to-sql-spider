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
    """
    Very simple schema representation.

    If your HF Spider variant has rich schema info, adapt this accordingly.
    We assume a field `db_schema` with:
    {table_name: [{"name": col_name, "type": col_type}, ...], ...}
    """
    if "db_schema" not in example:
        # Fallback: just use db_id; for best performance you should load real schema.
        return f"Database: {example.get('db_id', 'unknown')}\n"

    schema_lines = []
    schema = example["db_schema"]
    for table_name, cols in schema.items():
        col_parts = []
        for col in cols:
            col_name = col.get("name", "col")
            col_type = col.get("type", "TEXT")
            col_parts.append(f"{col_name} {col_type}")
        schema_lines.append(f"CREATE TABLE {table_name} ({', '.join(col_parts)});")
    return "\n".join(schema_lines)


def format_example(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Build a single training text string.

    Pattern:
    <s>[INST] SYSTEM_PROMPT
    Schema:
    ...
    Question: ...
    [/INST]
    SQL: ...
    </s>
    """
    schema_str = build_schema_string(example)
    question = example["question"]
    sql = example["sql"]

    prompt = (
        f"<s>[INST] {SYSTEM_PROMPT}\n"
        f"Schema:\n{schema_str}\n\n"
        f"Question: {question}\n"
        f"[/INST]\n"
        f"SQL:"
    )

    # For causal LM, we train to predict `SQL: ...` + sql
    full_text = f"{prompt} {sql}</s>"

    return {"text": full_text}


def tokenize_function(examples: Dict[str, Any], tokenizer: PreTrainedTokenizerBase, cfg: TrainingConfig):
    # Tokenize batch of examples
    return tokenizer(
        examples["text"],
        max_length=cfg.max_seq_length,
        truncation=True,
        padding="max_length",
    )


def prepare_datasets(tokenizer: PreTrainedTokenizerBase, cfg: TrainingConfig) -> DatasetDict:
    """
    Load Spider, format prompts, tokenize.

    Returns a DatasetDict with columns: input_ids, attention_mask, labels
    suitable for causal LM training.
    """
    raw = load_dataset(cfg.dataset_name)

    # Map to prompt format
    def add_text_column(example):
        return format_example(example)

    train_ds = raw[cfg.train_split].map(add_text_column)
    eval_ds = raw[cfg.eval_split].map(add_text_column)

    # Tokenize
    train_ds = train_ds.map(
        lambda batch: tokenize_function(batch, tokenizer, cfg),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    eval_ds = eval_ds.map(
        lambda batch: tokenize_function(batch, tokenizer, cfg),
        batched=True,
        remove_columns=eval_ds.column_names,
    )

    # Trainer expects labels
    train_ds = train_ds.rename_column("input_ids", "input_ids")
    eval_ds = eval_ds.rename_column("input_ids", "input_ids")

    train_ds = train_ds.with_format("torch")
    eval_ds = eval_ds.with_format("torch")

    # Causal LM: labels = input_ids
    train_ds = train_ds.map(lambda ex: {"labels": ex["input_ids"]})
    eval_ds = eval_ds.map(lambda ex: {"labels": ex["input_ids"]})

    return DatasetDict(train=train_ds, validation=eval_ds)
