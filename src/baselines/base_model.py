"""
Qwen2.5-Math-7B-Instruct baseline: zero-shot and few-shot WITHOUT LoRA. Runs locally on consumer GPU with 4-bit quantization.
"""
import argparse
import json
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from data.dataset import auto_load, format_chat_messages
from src.baselines.prompts import (
    FEW_SHOT_EXAMPLE_TEMPLATE,
    FEW_SHOT_HEADER,
    FEW_SHOT_QUERY_TEMPLATE,
)
from src.hf_utils import resolve_model_path


def run_base_model_inference(
    model,
    tokenizer,
    data: list[dict],
    system_prompt: str,
    user_template: str,
    mode: str = "zero_shot",
    few_shot_examples: list[dict] | None = None,
    n_shots: int = 3,
    max_new_tokens: int = 512,
) -> list[str]:
    results = []

    for item in tqdm(data, desc=f"Base model {mode}"):
        if mode == "few_shot" and few_shot_examples:
            few_shot_text = FEW_SHOT_HEADER
            for ex in few_shot_examples[:n_shots]:
                few_shot_text += FEW_SHOT_EXAMPLE_TEMPLATE.format(
                    source=ex["source"], target=ex["target"]
                )
            few_shot_text += FEW_SHOT_QUERY_TEMPLATE.format(source=item["source"])
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": few_shot_text},
            ]
        else:
            messages = format_chat_messages(item["source"], system_prompt, user_template)

        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        results.append(generated.strip())

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--mode", choices=["zero_shot", "few_shot"], default="zero_shot")
    parser.add_argument("--few_shot_file", default=None)
    parser.add_argument("--n_shots", type=int, default=3)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    mcfg = cfg["model"]
    pcfg = cfg["prompt"]
    model_dtype = getattr(torch, mcfg.get("torch_dtype", "float16"))
    model_path = resolve_model_path(mcfg["name_or_path"])

    if model_path != mcfg["name_or_path"]:
        print(f"Using cached model snapshot: {model_path}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=model_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=model_dtype,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = auto_load(args.input_file)

    few_shot_examples = None
    if args.mode == "few_shot" and args.few_shot_file:
        few_shot_examples = auto_load(args.few_shot_file)

    predictions = run_base_model_inference(
        model=model,
        tokenizer=tokenizer,
        data=data,
        system_prompt=pcfg["system"],
        user_template=pcfg["user_template"],
        mode=args.mode,
        few_shot_examples=few_shot_examples,
        n_shots=args.n_shots,
    )

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for item, pred in zip(data, predictions):
            record = {"source": item["source"], "target": item["target"], "prediction": pred}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(predictions)} predictions to {args.output_file}")


if __name__ == "__main__":
    main()
