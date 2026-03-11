#!/bin/bash
set -euo pipefail

CONFIG="configs/default.yaml"
TEST_FILE="data/test.jsonl"
PRED_DIR="predictions/ablation"
RESULTS_DIR="results"
MAIN_RESULT="$RESULTS_DIR/qlora_finetuned.json"

mkdir -p "$PRED_DIR" "$RESULTS_DIR"

if [ ! -f "$MAIN_RESULT" ]; then
    echo "Missing main experiment result: $MAIN_RESULT" >&2
    echo "Run the main experiment first so rank=16 and target_modules=all can be reused." >&2
    exit 1
fi

get_target_modules() {
    case "$1" in
        attention_only)
            echo "q_proj,k_proj,v_proj,o_proj"
            ;;
        mlp_only)
            echo "gate_proj,up_proj,down_proj"
            ;;
        *)
            echo "Unknown target module group: $1" >&2
            exit 1
            ;;
    esac
}

echo "============================================"
echo "  Ablation 1 — LoRA Rank"
echo "============================================"

for RANK in 8 32 64; do
    echo -e "\n--- LoRA rank = $RANK ---"
    OUTPUT="outputs/ablation_rank_${RANK}"

    python3 -m src.train \
        --config "$CONFIG" \
        --output_dir "$OUTPUT" \
        --lora_r "$RANK"

    python3 -m src.inference \
        --config "$CONFIG" \
        --adapter_path "$OUTPUT/final" \
        --input_file "$TEST_FILE" \
        --output_file "$PRED_DIR/rank_${RANK}.jsonl"

    python3 -m src.evaluate \
        --predictions_file "$PRED_DIR/rank_${RANK}.jsonl" \
        --output_file "$RESULTS_DIR/ablation_rank_${RANK}.json"
done

# Copy main experiment result as rank=16
cp "$MAIN_RESULT" "$RESULTS_DIR/ablation_rank_16.json"
cp "$PRED_DIR/../qlora_finetuned.jsonl" "$PRED_DIR/rank_16.jsonl" 2>/dev/null || true
echo -e "\n--- LoRA rank = 16 (copied from main experiment) ---"

echo -e "\n============================================"
echo "  Ablation 2 — Target Modules"
echo "============================================"

for GROUP in attention_only mlp_only; do
    MODULES="$(get_target_modules "$GROUP")"
    echo -e "\n--- target_modules = $GROUP ($MODULES) ---"
    OUTPUT="outputs/ablation_modules_${GROUP}"

    python3 -m src.train \
        --config "$CONFIG" \
        --output_dir "$OUTPUT" \
        --target_modules "$MODULES"

    python3 -m src.inference \
        --config "$CONFIG" \
        --adapter_path "$OUTPUT/final" \
        --input_file "$TEST_FILE" \
        --output_file "$PRED_DIR/modules_${GROUP}.jsonl"

    python3 -m src.evaluate \
        --predictions_file "$PRED_DIR/modules_${GROUP}.jsonl" \
        --output_file "$RESULTS_DIR/ablation_modules_${GROUP}.json"
done

# Copy main experiment result as modules=all
cp "$MAIN_RESULT" "$RESULTS_DIR/ablation_modules_all.json"
cp "$PRED_DIR/../qlora_finetuned.jsonl" "$PRED_DIR/modules_all.jsonl" 2>/dev/null || true
echo -e "\n--- target_modules = all (copied from main experiment) ---"

echo -e "\n============================================"
echo "  Ablation studies complete!"
echo "  Results: $RESULTS_DIR/"
echo "============================================"
