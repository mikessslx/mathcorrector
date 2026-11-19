import json
import random
from pathlib import Path
from typing import Callable, Optional

from torch.utils.data import Dataset


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: str) -> list[dict]:
    import csv

    records = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(dict(row))
    return records


LOADERS: dict[str, Callable] = {
    ".jsonl": load_jsonl,
    ".json": load_json,
    ".csv": load_csv,
}


def auto_load(path: str) -> list[dict]:
    suffix = Path(path).suffix.lower()
    loader = LOADERS.get(suffix)
    if loader is None:
        raise ValueError(f"Unsupported file format: {suffix}. Supported: {list(LOADERS.keys())}")
    return loader(path)


def format_chat_messages(
    source: str,
    system_prompt: str,
    user_template: str,
) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_template.format(source=source)},
    ]


def _normalize_input_ids(tokenized) -> list[int]:
    """Normalize tokenizer outputs from different transformers versions into list[int]."""
    if isinstance(tokenized, list):
        if tokenized and isinstance(tokenized[0], list):
            tokenized = tokenized[0]
        return [int(x) for x in tokenized]

    if hasattr(tokenized, "ids"):  # tokenizers.Encoding
        return [int(x) for x in tokenized.ids]

    if isinstance(tokenized, dict):
        if "input_ids" not in tokenized:
            raise TypeError("Unsupported dict-like tokenizer output without 'input_ids'.")
        return _normalize_input_ids(tokenized["input_ids"])

    if hasattr(tokenized, "input_ids"):  # BatchEncoding-like
        return _normalize_input_ids(tokenized.input_ids)

    if hasattr(tokenized, "tolist"):  # tensor/ndarray-like
        return _normalize_input_ids(tokenized.tolist())

    raise TypeError(f"Unsupported tokenizer output type: {type(tokenized)}")


class MathCorrectionDataset(Dataset):
    """Dataset that tokenizes (source, target) pairs into Qwen2.5 chat format."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 1024,
        system_prompt: str = "",
        user_template: str = "{source}",
        load_fn: Optional[Callable] = None,
        is_test: bool = False,
    ):
        loader = load_fn or auto_load
        self.raw_data = loader(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.is_test = is_test

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        item = self.raw_data[idx]
        source = item["source"]
        target = item["target"]

        messages = format_chat_messages(
            source,
            self.system_prompt,
            self.user_template,
        )

        if self.is_test:
            prompt_ids = _normalize_input_ids(
                self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors=None,
                )
            )
            return {
                "input_ids": prompt_ids,
                "source": source,
                "target": target,
            }

        messages_with_response = messages + [{"role": "assistant", "content": target}]
        full_ids = _normalize_input_ids(
            self.tokenizer.apply_chat_template(
                messages_with_response,
                add_generation_prompt=False,
                return_tensors=None,
            )
        )

        prompt_ids = _normalize_input_ids(
            self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors=None,
            )
        )
        prompt_len = len(prompt_ids)
        labels = [-100] * prompt_len + full_ids[prompt_len:]

        if len(full_ids) > self.max_length:
            full_ids = full_ids[: self.max_length]
            labels = labels[: self.max_length]

        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": [1] * len(full_ids),
        }


class DataCollatorForMathCorrection:
    """Pads batch to uniform length with left-padding for input_ids, right-padding for labels."""

    def __init__(self, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.max_length = max_length

    def __call__(self, features: list[dict]) -> dict:
        import torch

        max_len = min(max(len(f["input_ids"]) for f in features), self.max_length)

        input_ids_batch = []
        labels_batch = []
        attention_mask_batch = []

        for f in features:
            ids = f["input_ids"][:max_len]
            pad_len = max_len - len(ids)

            input_ids_batch.append([self.pad_token_id] * pad_len + ids)
            attention_mask_batch.append([0] * pad_len + [1] * len(ids))

            if "labels" in f:
                lab = f["labels"][:max_len]
                labels_batch.append([-100] * pad_len + lab)

        result = {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
        }
        if labels_batch:
            result["labels"] = torch.tensor(labels_batch, dtype=torch.long)

        return result


def split_data(
    data_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """Split a single data file into train/val/test JSONL files."""
    records = auto_load(data_path)
    random.seed(seed)
    random.shuffle(records)

    n = len(records)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train.jsonl": records[:n_train],
        "val.jsonl": records[n_train : n_train + n_val],
        "test.jsonl": records[n_train + n_val :],
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for filename, data in splits.items():
        with open(out / filename, "w", encoding="utf-8") as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  {filename}: {len(data)} samples")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument("--input", required=True, help="Path to input data file")
    parser.add_argument("--output_dir", default="data/", help="Output directory")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_data(args.input, args.output_dir, args.train_ratio, args.val_ratio, args.seed)
