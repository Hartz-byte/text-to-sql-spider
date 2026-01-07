import torch
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from transformers import GenerationConfig, AutoModelForCausalLM
from peft import PeftModel

from .config import load_config, PROJECT_ROOT
from .modeling import load_tokenizer
from .data import SYSTEM_PROMPT, build_schema_string


EXAMPLE_QUESTIONS = [
    {
        "db_id": "concert_singer",
        "question": "How many singers are there?",
    },
    {
        "db_id": "flight_2",
        "question": "List the names of all airlines.",
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

    print("Loading model (GPU-only)...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.float16,
        device_map="cuda:0",  # GPU-only
        trust_remote_code=True,
    )

    tokenizer = load_tokenizer(cfg)

    print("Loading LoRA...")
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
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=gen_cfg)

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        sql_start = decoded.find("SQL:") + len("SQL:")
        sql = decoded[sql_start:].strip()

        print(f"\nDB: {ex['db_id']}")
        print(f"Q:  {ex['question']}")
        print(f"SQL: {sql}")
        print("-" * 80)


if __name__ == "__main__":
    main()
