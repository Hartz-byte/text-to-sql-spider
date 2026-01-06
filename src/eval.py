import torch
from pathlib import Path
from transformers import GenerationConfig
from unsloth import FastLanguageModel
from peft import PeftModel

from .config import load_config, PROJECT_ROOT
from .modeling import load_tokenizer
from .data import SYSTEM_PROMPT, build_schema_string


EXAMPLE_QUESTIONS = [
    {
        "db_id": "concert_singer",
        "question": "How many singers are there?",
        "db_schema": {},  # if you have schema, populate here
    },
    {
        "db_id": "flight_2",
        "question": "List the names of all airlines.",
        "db_schema": {},
    },
]


def make_prompt(example, tokenizer, cfg):
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

    # Load base model in 4-bit
    base_model_name = cfg.base_model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=cfg.max_seq_length,
        dtype=torch.float16,
        load_in_4bit=True,
    )

    tokenizer = load_tokenizer(cfg)

    # Load LoRA adapters
    model = PeftModel.from_pretrained(model, output_dir)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    gen_cfg = GenerationConfig(
        max_new_tokens=cfg.gen_max_new_tokens,
        temperature=cfg.gen_temperature,
        top_p=cfg.gen_top_p,
        do_sample=False,
    )

    for ex in EXAMPLE_QUESTIONS:
        prompt = make_prompt(ex, tokenizer, cfg)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=gen_cfg,
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("=" * 80)
        print("QUESTION:", ex["question"])
        print("---")
        print(decoded)
        print("=" * 80)


if __name__ == "__main__":
    main()
