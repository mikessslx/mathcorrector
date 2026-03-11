$ErrorActionPreference = 'Stop'

# --- Configuration ---
$CONFIG = 'configs/default.yaml'
$TEST_FILE = 'data/test.jsonl'
$FEW_SHOT_FILE = 'data/train.jsonl'
$PRED_DIR = 'predictions'
$RESULTS_DIR = 'results'

# GPT API settings (override via environment)
$GPT_MODEL = if ($env:GPT_MODEL) { $env:GPT_MODEL } else { 'gpt-5.4' }
$GPT_BASE_URL = $env:GPT_BASE_URL
$GPT_API_KEY = $env:GPT_API_KEY

# LoRA adapter path (set after training)
$ADAPTER_PATH = if ($env:ADAPTER_PATH) { $env:ADAPTER_PATH } else { 'outputs/qlora_r16/final' }

New-Item -ItemType Directory -Force -Path $PRED_DIR, $RESULTS_DIR | Out-Null

function Run-Python {
    param([string[]]$CommandArgs)

    & python @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: python $($CommandArgs -join ' ')"
    }
}

Write-Host '============================================'
Write-Host '  Running All Baselines & Evaluation'
Write-Host '============================================'

# --- 1. Base model (zero-shot) ---
Write-Host "`n[1/5] Qwen2.5-Math-7B zero-shot..."
Run-Python -CommandArgs @(
    '-m', 'src.baselines.base_model',
    '--config', $CONFIG,
    '--input_file', $TEST_FILE,
    '--output_file', "$PRED_DIR/base_zero_shot.jsonl",
    '--mode', 'zero_shot'
)

Run-Python -CommandArgs @(
    '-m', 'src.evaluate',
    '--predictions_file', "$PRED_DIR/base_zero_shot.jsonl",
    '--output_file', "$RESULTS_DIR/base_zero_shot.json"
)

# --- 2. Base model (few-shot) ---
Write-Host "`n[2/5] Qwen2.5-Math-7B few-shot..."
Run-Python -CommandArgs @(
    '-m', 'src.baselines.base_model',
    '--config', $CONFIG,
    '--input_file', $TEST_FILE,
    '--output_file', "$PRED_DIR/base_few_shot.jsonl",
    '--mode', 'few_shot',
    '--few_shot_file', $FEW_SHOT_FILE,
    '--n_shots', '3'
)

Run-Python -CommandArgs @(
    '-m', 'src.evaluate',
    '--predictions_file', "$PRED_DIR/base_few_shot.jsonl",
    '--output_file', "$RESULTS_DIR/base_few_shot.json"
)

# --- 3. GPT zero-shot ---
Write-Host "`n[3/5] GPT zero-shot..."
$gptZeroShotArgs = @(
    '-m', 'src.baselines.gpt_baseline',
    '--input_file', $TEST_FILE,
    '--output_file', "$PRED_DIR/gpt_zero_shot.jsonl",
    '--model', $GPT_MODEL
)
if ($GPT_BASE_URL) { $gptZeroShotArgs += @('--base_url', $GPT_BASE_URL) }
if ($GPT_API_KEY) { $gptZeroShotArgs += @('--api_key', $GPT_API_KEY) }
$gptZeroShotArgs += @('--mode', 'zero_shot')
Run-Python -CommandArgs $gptZeroShotArgs

Run-Python -CommandArgs @(
    '-m', 'src.evaluate',
    '--predictions_file', "$PRED_DIR/gpt_zero_shot.jsonl",
    '--output_file', "$RESULTS_DIR/gpt_zero_shot.json"
)

# --- 4. GPT few-shot ---
Write-Host "`n[4/5] GPT few-shot..."
$gptFewShotArgs = @(
    '-m', 'src.baselines.gpt_baseline',
    '--input_file', $TEST_FILE,
    '--output_file', "$PRED_DIR/gpt_few_shot.jsonl",
    '--model', $GPT_MODEL
)
if ($GPT_BASE_URL) { $gptFewShotArgs += @('--base_url', $GPT_BASE_URL) }
if ($GPT_API_KEY) { $gptFewShotArgs += @('--api_key', $GPT_API_KEY) }
$gptFewShotArgs += @(
    '--mode', 'few_shot',
    '--few_shot_file', $FEW_SHOT_FILE,
    '--n_shots', '3'
)
Run-Python -CommandArgs $gptFewShotArgs

Run-Python -CommandArgs @(
    '-m', 'src.evaluate',
    '--predictions_file', "$PRED_DIR/gpt_few_shot.jsonl",
    '--output_file', "$RESULTS_DIR/gpt_few_shot.json"
)

# --- 5. Our model (QLoRA fine-tuned) ---
Write-Host "`n[5/5] QLoRA fine-tuned model..."
Run-Python -CommandArgs @(
    '-m', 'src.inference',
    '--config', $CONFIG,
    '--adapter_path', $ADAPTER_PATH,
    '--input_file', $TEST_FILE,
    '--output_file', "$PRED_DIR/qlora_finetuned.jsonl"
)

Run-Python -CommandArgs @(
    '-m', 'src.evaluate',
    '--predictions_file', "$PRED_DIR/qlora_finetuned.jsonl",
    '--output_file', "$RESULTS_DIR/qlora_finetuned.json"
)

Write-Host "`n============================================"
Write-Host '  All evaluations complete!'
Write-Host "  Results: $RESULTS_DIR/"
Write-Host '============================================'
