# MathCorrector

> A QLoRA-based specialist model for mathematical text error correction. It fine-tunes `Qwen/Qwen2.5-Math-7B-Instruct` to minimally repair terminology, notation, formula and logical errors in English mathematical text interleaved with LaTeX. All data reported in this paper were from MathTEC.

## What Is Included

```text
mathcorrector/
├── configs/
│   └── default.yaml        # Default QLoRA, data, and generation config
├── data/
│   ├── mathtec_bench_converted.jsonl
│   │                       # Full 500-example benchmark
│   ├── train.jsonl         # 400-example training split
│   ├── val.json            # 50-example validation split
│   ├── test.jsonl          # 50-example test split
│   └── dataset.py          # Data loading, chat formatting, split utility
├── src/
│   ├── train.py            # QLoRA training entry point
│   ├── inference.py        # Batch inference with LoRA adapters
│   ├── evaluate.py         # EM, token PRF, GLEU, CER, detection metrics
│   ├── chat.py             # Interactive CLI for correction
│   ├── hf_utils.py         # Local path / HF cache resolution
│   └── baselines/
│       ├── base_model.py   # Local Qwen zero-shot / few-shot baselines
│       ├── gpt_baseline.py # OpenAI-compatible GPT baselines
│       └── prompts.py      # Baseline prompt templates
├── scripts/
│   ├── train.sh            # Train the default QLoRA model
│   ├── run_baselines.sh    # Reproduce main baseline comparisons
│   ├── run_ablations.sh    # Reproduce LoRA rank / module ablations
│   ├── train.ps1           # PowerShell training script
│   ├── run_baselines.ps1   # PowerShell baseline script
│   └── run_ablations.ps1   # PowerShell ablation script
├── analysis/
│   ├── error_analysis.py   # Heuristic edit-type analysis
├── predictions/            # Released prediction files
├── results/                # Released evaluation JSON results
├── paper/
│   └── paper.tex           # ACL-style paper draft
├── requirements.txt
└── README.md
```

## Reproduce Experiments

1. Training

```bash
pip install -r requirements.txt
bash scripts/train.sh
```

2. Baselines

```bash
# Set GPT API
export GPT_MODEL="gpt-5.4"
export GPT_BASE_URL="https://your-relay.com/v1"
export GPT_API_KEY="sk-..."

bash scripts/run_baselines.sh
```

3. Ablations

```bash
bash scripts/run_ablations.sh
```

4. Evaluation

```bash
python3 -m src.evaluate \
  --predictions_file predictions/qlora_finetuned.jsonl \
  --output_file results/qlora_finetuned.json
```

5. Interactive Demo

```bash
python3 -m src.chat \
  --config configs/default.yaml \
  --adapter_path outputs/qlora_r16/final
```

```text
Input:  For all $x \in \mathbb{R}$, the funciton $f(x) = x^{2}$ is continuos.
Output: For all $x \in \mathbb{R}$, the function $f(x) = x^{2}$ is continuous.
```

## Results

### Main Results

| Model | Precision | Recall | F0.5 | Exact Match | CER | Detection Acc. |
|--------|-----------|--------|------|-------------|-----|----------------|
| GPT zero-shot | 0.937 | 0.961 | 0.941 | 0.900 | 0.0046 | 0.980 |
| GPT 3-shot | 0.947 | 0.935 | 0.945 | 0.880 | 0.0049 | 0.980 |
| Qwen zero-shot | 0.028 | 0.104 | 0.032 | 0.000 | 7.7447 | 0.960 |
| Qwen 3-shot | 0.057 | 0.156 | 0.065 | 0.000 | 10.3341 | 0.960 |
| **MathCorrector (QLoRA, rank 16)** | **0.961** | 0.948 | **0.958** | **0.900** | **0.0014** | 0.980 |

### Ablation Results

LoRA rank:

| Rank | Precision | Recall | F0.5 | Exact Match | CER |
|------|-----------|--------|------|-------------|-----|
| 8 | 0.961 | 0.948 | 0.958 | 0.900 | 0.0014 |
| 16 | 0.961 | 0.948 | 0.958 | 0.900 | 0.0014 |
| 32 | 0.974 | 0.961 | 0.971 | 0.920 | 0.0006 |
| 64 | 0.961 | 0.948 | 0.958 | 0.900 | 0.0012 |

Target modules:

| Target Modules | Precision | Recall | F0.5 | Exact Match | CER |
|----------------|-----------|--------|------|-------------|-----|
| Attention only | 0.947 | 0.935 | 0.945 | 0.880 | 0.0008 |
| MLP only | 0.961 | 0.948 | 0.958 | 0.900 | 0.0014 |
| All modules | 0.961 | 0.948 | 0.958 | 0.900 | 0.0014 |
