import argparse
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from data.dataset import DataCollatorForMathCorrection, MathCorrectionDataset
from src.hf_utils import resolve_model_path


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_quantization_config(cfg: dict) -> BitsAndBytesConfig:
    qcfg = cfg["quantization"]
    return BitsAndBytesConfig(
        load_in_4bit=qcfg["load_in_4bit"],
        bnb_4bit_quant_type=qcfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, qcfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=qcfg["bnb_4bit_use_double_quant"],
    )


def build_lora_config(cfg: dict) -> LoraConfig:
    lcfg = cfg["lora"]
    return LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["lora_alpha"],
        lora_dropout=lcfg["lora_dropout"],
        target_modules=lcfg["target_modules"],
        task_type=lcfg["task_type"],
        bias=lcfg["bias"],
    )


def build_training_args(
    cfg: dict,
    output_dir_override: str = None,
) -> TrainingArguments:
    tcfg = cfg["training"]
    output_dir = output_dir_override or tcfg["output_dir"]
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=tcfg["num_train_epochs"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        per_device_eval_batch_size=tcfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        gradient_checkpointing=tcfg["gradient_checkpointing"],
        learning_rate=tcfg["learning_rate"],
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        warmup_ratio=tcfg["warmup_ratio"],
        weight_decay=tcfg["weight_decay"],
        max_grad_norm=tcfg["max_grad_norm"],
        logging_steps=tcfg["logging_steps"],
        eval_strategy=tcfg["eval_strategy"],
        eval_steps=tcfg["eval_steps"],
        save_strategy=tcfg["save_strategy"],
        save_steps=tcfg["save_steps"],
        save_total_limit=tcfg["save_total_limit"],
        load_best_model_at_end=tcfg["load_best_model_at_end"],
        metric_for_best_model=tcfg["metric_for_best_model"],
        greater_is_better=tcfg["greater_is_better"],
        bf16=tcfg.get("bf16", False),
        fp16=tcfg.get("fp16", False),
        dataloader_num_workers=tcfg["dataloader_num_workers"],
        report_to=tcfg["report_to"],
        seed=tcfg["seed"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--lora_r",
        type=int,
        default=None,
        help="Override LoRA rank for ablation",
    )
    parser.add_argument(
        "--data_fraction",
        type=float,
        default=1.0,
        help="Fraction of training data to use",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        default=None,
        help="Override LoRA target modules (comma-separated, e.g. 'q_proj,k_proj,v_proj,o_proj')",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.lora_r is not None:
        cfg["lora"]["r"] = args.lora_r
        cfg["lora"]["lora_alpha"] = args.lora_r * 2
    if args.learning_rate is not None:
        cfg["training"]["learning_rate"] = args.learning_rate
    if args.target_modules is not None:
        cfg["lora"]["target_modules"] = [
            m.strip() for m in args.target_modules.split(",")
        ]

    mcfg = cfg["model"]
    model_path = resolve_model_path(mcfg["name_or_path"])
    if model_path != mcfg["name_or_path"]:
        print(f"Using cached model snapshot: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=mcfg.get("trust_remote_code", True),
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = build_quantization_config(cfg)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=mcfg.get("trust_remote_code", True),
        torch_dtype=getattr(torch, mcfg.get("torch_dtype", "float16")),
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    lora_config = build_lora_config(cfg)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    pcfg = cfg["prompt"]
    dcfg = cfg["data"]

    train_dataset = MathCorrectionDataset(
        data_path=dcfg["train_file"],
        tokenizer=tokenizer,
        max_length=dcfg["max_length"],
        system_prompt=pcfg["system"],
        user_template=pcfg["user_template"],
    )

    if args.data_fraction < 1.0:
        import random

        random.seed(cfg["training"]["seed"])
        total = len(train_dataset.raw_data)
        n = int(total * args.data_fraction)
        train_dataset.raw_data = random.sample(train_dataset.raw_data, n)
        print(f"Using {n}/{total} training samples ({args.data_fraction:.0%})")

    val_dataset = MathCorrectionDataset(
        data_path=dcfg["val_file"],
        tokenizer=tokenizer,
        max_length=dcfg["max_length"],
        system_prompt=pcfg["system"],
        user_template=pcfg["user_template"],
    )

    collator = DataCollatorForMathCorrection(
        tokenizer,
        max_length=dcfg["max_length"],
    )
    training_args = build_training_args(cfg, args.output_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )
    trainer.train()

    final_dir = Path(training_args.output_dir) / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Model saved to {final_dir}")


if __name__ == "__main__":
    main()
