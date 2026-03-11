import argparse
import json
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from data.dataset import auto_load, format_chat_messages
from src.hf_utils import resolve_model_path


def load_model(
    base_model_path: str,
    adapter_path: str,
    quantize_4bit: bool = True,
    torch_dtype: str = "float16",
):
    dtype = getattr(torch, torch_dtype)
    resolved_model_path = resolve_model_path(base_model_path)

    if resolved_model_path != base_model_path:
        print(f"Using cached model snapshot: {resolved_model_path}")

    bnb_config = None
    if quantize_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        resolved_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def run_inference(
    model,
    tokenizer,
    sources: list[str],
    system_prompt: str,
    user_template: str,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    do_sample: bool = False,
    batch_size: int = 1,
) -> list[str]:
    results = []

    for i in tqdm(range(0, len(sources), batch_size), desc="Inference"):
        batch_sources = sources[i : i + batch_size]
        batch_messages = [
            format_chat_messages(s, system_prompt, user_template)
            for s in batch_sources
        ]
        batch_texts = [
            tokenizer.apply_chat_template(
                msgs,
                add_generation_prompt=True,
                tokenize=False,
            )
            for msgs in batch_messages
        ]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, output in enumerate(outputs):
            prompt_len = inputs["input_ids"][j].shape[0]
            generated = tokenizer.decode(
                output[prompt_len:],
                skip_special_tokens=True,
            )
            results.append(generated.strip())

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--adapter_path",
        required=True,
        help="Path to LoRA adapter directory",
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Input JSONL file with 'source' field",
    )
    parser.add_argument("--output_file", required=True, help="Output JSONL file")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--no_quantize",
        action="store_true",
        help="Disable 4-bit quantization",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    mcfg = cfg["model"]
    pcfg = cfg["prompt"]
    gcfg = cfg["generation"]

    model, tokenizer = load_model(
        base_model_path=mcfg["name_or_path"],
        adapter_path=args.adapter_path,
        quantize_4bit=not args.no_quantize,
        torch_dtype=mcfg.get("torch_dtype", "float16"),
    )

    data = auto_load(args.input_file)
    sources = [d["source"] for d in data]
    targets = [d.get("target", "") for d in data]

    predictions = run_inference(
        model=model,
        tokenizer=tokenizer,
        sources=sources,
        system_prompt=pcfg["system"],
        user_template=pcfg["user_template"],
        max_new_tokens=gcfg["max_new_tokens"],
        temperature=gcfg.get("temperature", 0.1),
        do_sample=gcfg.get("do_sample", False),
        batch_size=args.batch_size,
    )

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for src, tgt, pred in zip(sources, targets, predictions):
            record = {"source": src, "target": tgt, "prediction": pred}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Predictions saved to {args.output_file} ({len(predictions)} samples)")


if __name__ == "__main__":
    main()
