import argparse
import yaml

from src.inference import load_model, run_inference


def main():
    parser = argparse.ArgumentParser(description="Interactive chat for Math Error Correction model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--adapter_path", required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--no_quantize", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
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

    max_new_tokens = args.max_new_tokens or gcfg.get("max_new_tokens", 512)

    print("MathCorrector interactive mode")
    print("Type your text and press Enter. Type 'exit' to quit.")

    while True:
        user_text = input("\nYou: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit", "q"}:
            print("Bye.")
            break

        prediction = run_inference(
            model=model,
            tokenizer=tokenizer,
            sources=[user_text],
            system_prompt=pcfg["system"],
            user_template=pcfg["user_template"],
            max_new_tokens=max_new_tokens,
            temperature=gcfg.get("temperature", 0.1),
            do_sample=gcfg.get("do_sample", False),
            batch_size=1,
        )[0]

        print(f"Model: {prediction}")


if __name__ == "__main__":
    main()
