import os
from dotenv import load_dotenv
load_dotenv()
import torch
from pathlib import Path

from transformers import GenerationConfig, BitsAndBytesConfig, AutoModelForCausalLM
from peft import PeftModel

from .config import load_config, PROJECT_ROOT
from .modeling import load_tokenizer
from .data import SYSTEM_PROMPT, build_schema_string


EXAMPLE_QUESTIONS = [
    {
        "db_id": "concert_singer",
        "question": "How many singers are there?",
        "db_schema": {},
    },
    {
        "db_id": "flight_2",
        "question": "List the names of all airlines.",
        "db_schema": {},
    },
]


def make_prompt(example, cfg):
    schema_str = build_schema_string(example)
    question = example["question"]

    prompt = (
        f"<s>[INST] {SYSTEM_PROMPT}\n"
        f"Schema:\n{schema_str}\n\n"
        f"Question: {question}\n"
        f"[/INST]\n"
        f"SQL:"
    )

    return prompt


def main():
    cfg = load_config()
    output_dir = PROJECT_ROOT / cfg.output_dir

    print("Loading model...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = load_tokenizer(cfg)

    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(model, output_dir)
    model = model.merge_and_unload()
    model.eval()

    gen_cfg = GenerationConfig(
        max_new_tokens=cfg.gen_max_new_tokens,
        temperature=cfg.gen_temperature,
        top_p=cfg.gen_top_p,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    print("\n" + "=" * 80)
    print("TEXT-TO-SQL INFERENCE")
    print("=" * 80)

    for ex in EXAMPLE_QUESTIONS:
        prompt = make_prompt(ex, cfg)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=gen_cfg)

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        sql_start = decoded.find("SQL:") + len("SQL:")
        generated_sql = decoded[sql_start:].strip()

        print(f"\nDatabase: {ex['db_id']}")
        print(f"Question: {ex['question']}")
        print(f"Generated SQL: {generated_sql}")
        print("-" * 80)


if __name__ == "__main__":
    main()
