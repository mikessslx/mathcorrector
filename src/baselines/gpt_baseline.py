import argparse
import asyncio
import json
import os
from pathlib import Path

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from data.dataset import auto_load
from src.baselines.prompts import (
    FEW_SHOT_EXAMPLE_TEMPLATE,
    FEW_SHOT_HEADER,
    FEW_SHOT_QUERY_TEMPLATE,
    SYSTEM_PROMPT,
    ZERO_SHOT_TEMPLATE,
)


def build_zero_shot_messages(source: str) -> list[dict]:
    """Build a zero-shot request."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ZERO_SHOT_TEMPLATE.format(source=source)},
    ]


def build_few_shot_messages(source: str, examples: list[dict], n_shots: int = 3) -> list[dict]:
    """Build a few-shot request."""
    few_shot_text = FEW_SHOT_HEADER
    for ex in examples[:n_shots]:
        few_shot_text += FEW_SHOT_EXAMPLE_TEMPLATE.format(source=ex["source"], target=ex["target"])
    few_shot_text += FEW_SHOT_QUERY_TEMPLATE.format(source=source)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": few_shot_text},
    ]


async def call_api_with_retry(
    client: AsyncOpenAI,
    messages: list[dict],
    model: str,
    max_retries: int = 3,
    temperature: float = 0.0,
) -> str:
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} retries: {e}")
                return ""
            wait = 2 ** attempt
            await asyncio.sleep(wait)
    return ""


async def run_baseline(
    data: list[dict],
    client: AsyncOpenAI,
    model: str,
    mode: str = "zero_shot",
    few_shot_examples: list[dict] | None = None,
    n_shots: int = 3,
    concurrency: int = 5,
) -> list[str]:
    semaphore = asyncio.Semaphore(concurrency)

    async def process_one(item):
        async with semaphore:
            if mode == "few_shot" and few_shot_examples:
                messages = build_few_shot_messages(item["source"], few_shot_examples, n_shots)
            else:
                messages = build_zero_shot_messages(item["source"])
            return await call_api_with_retry(client, messages, model)

    tasks = [process_one(item) for item in data]
    results = await tqdm_asyncio.gather(*tasks, desc=f"GPT {mode}")
    return list(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--model", default="gpt-5.4", help="Model name for the API")
    parser.add_argument("--base_url", default=None, help="Custom API base URL")
    parser.add_argument("--api_key", default=None, help="API key (or set OPENAI_API_KEY env)")
    parser.add_argument("--mode", choices=["zero_shot", "few_shot"], default="zero_shot")
    parser.add_argument("--few_shot_file", default=None, help="JSONL file for few-shot examples")
    parser.add_argument("--n_shots", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL", None)

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    data = auto_load(args.input_file)

    few_shot_examples = None
    if args.mode == "few_shot" and args.few_shot_file:
        few_shot_examples = auto_load(args.few_shot_file)

    predictions = asyncio.run(
        run_baseline(
            data=data,
            client=client,
            model=args.model,
            mode=args.mode,
            few_shot_examples=few_shot_examples,
            n_shots=args.n_shots,
            concurrency=args.concurrency,
        )
    )

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for item, pred in zip(data, predictions):
            record = {"source": item["source"], "target": item["target"], "prediction": pred}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(predictions)} predictions to {args.output_file}")


if __name__ == "__main__":
    main()
