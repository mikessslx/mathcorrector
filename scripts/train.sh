#!/bin/bash
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"
OUTPUT_DIR="${2:-outputs/qlora_r16}"

echo "=== Training QLoRA Math Error Correction ==="
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"

python3 -m src.train \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR"

echo "=== Training complete ==="
echo "Adapter saved to: $OUTPUT_DIR/final/"
