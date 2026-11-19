#!/bin/bash
set -euo pipefail

# --- Configuration ---
CONFIG="configs/default.yaml"
TEST_FILE="data/test.jsonl"
FEW_SHOT_FILE="data/train.jsonl"
PRED_DIR="predictions"
RESULTS_DIR="results"

# GPT API settings (override via environment)
GPT_MODEL="${GPT_MODEL:-gpt-5.4}"
GPT_BASE_URL="${GPT_BASE_URL:-}"
GPT_API_KEY="${GPT_API_KEY:-}"

# LoRA adapter path (set after training)
ADAPTER_PATH="${ADAPTER_PATH:-outputs/qlora_r16/final}"

mkdir -p "$PRED_DIR" "$RESULTS_DIR"

echo "============================================"
echo "  Running All Baselines & Evaluation"
echo "============================================"

# --- 1. Base model (zero-shot) ---
echo -e "\n[1/5] Qwen2.5-Math-7B zero-shot..."
python3 -m src.baselines.base_model \
    --config "$CONFIG" \
    --input_file "$TEST_FILE" \
    --output_file "$PRED_DIR/base_zero_shot.jsonl" \
    --mode zero_shot

python3 -m src.evaluate \
    --predictions_file "$PRED_DIR/base_zero_shot.jsonl" \
    --output_file "$RESULTS_DIR/base_zero_shot.json"

# --- 2. Base model (few-shot) ---
echo -e "\n[2/5] Qwen2.5-Math-7B few-shot..."
python3 -m src.baselines.base_model \
    --config "$CONFIG" \
    --input_file "$TEST_FILE" \
    --output_file "$PRED_DIR/base_few_shot.jsonl" \
    --mode few_shot \
    --few_shot_file "$FEW_SHOT_FILE" \
    --n_shots 3

python3 -m src.evaluate \
    --predictions_file "$PRED_DIR/base_few_shot.jsonl" \
    --output_file "$RESULTS_DIR/base_few_shot.json"

# --- 3. GPT zero-shot ---
echo -e "\n[3/5] GPT zero-shot..."
python3 -m src.baselines.gpt_baseline \
    --input_file "$TEST_FILE" \
    --output_file "$PRED_DIR/gpt_zero_shot.jsonl" \
    --model "$GPT_MODEL" \
    ${GPT_BASE_URL:+--base_url "$GPT_BASE_URL"} \
    ${GPT_API_KEY:+--api_key "$GPT_API_KEY"} \
    --mode zero_shot

python3 -m src.evaluate \
    --predictions_file "$PRED_DIR/gpt_zero_shot.jsonl" \
    --output_file "$RESULTS_DIR/gpt_zero_shot.json"

# --- 4. GPT few-shot ---
echo -e "\n[4/5] GPT few-shot..."
python3 -m src.baselines.gpt_baseline \
    --input_file "$TEST_FILE" \
    --output_file "$PRED_DIR/gpt_few_shot.jsonl" \
    --model "$GPT_MODEL" \
    ${GPT_BASE_URL:+--base_url "$GPT_BASE_URL"} \
    ${GPT_API_KEY:+--api_key "$GPT_API_KEY"} \
    --mode few_shot \
    --few_shot_file "$FEW_SHOT_FILE" \
    --n_shots 3

python3 -m src.evaluate \
    --predictions_file "$PRED_DIR/gpt_few_shot.jsonl" \
    --output_file "$RESULTS_DIR/gpt_few_shot.json"

# --- 5. Our model (QLoRA fine-tuned) ---
echo -e "\n[5/5] QLoRA fine-tuned model..."
python3 -m src.inference \
    --config "$CONFIG" \
    --adapter_path "$ADAPTER_PATH" \
    --input_file "$TEST_FILE" \
    --output_file "$PRED_DIR/qlora_finetuned.jsonl"

python3 -m src.evaluate \
    --predictions_file "$PRED_DIR/qlora_finetuned.jsonl" \
    --output_file "$RESULTS_DIR/qlora_finetuned.json"

echo -e "\n============================================"
echo "  All evaluations complete!"
echo "  Results: $RESULTS_DIR/"
echo "============================================"
