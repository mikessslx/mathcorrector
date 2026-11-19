$ErrorActionPreference = 'Stop'

$CONFIG = 'configs/default.yaml'
$TEST_FILE = 'data/test.jsonl'
$PRED_DIR = 'predictions/ablation'
$RESULTS_DIR = 'results'
$MAIN_RESULT = Join-Path $RESULTS_DIR 'qlora_finetuned.json'

New-Item -ItemType Directory -Force -Path $PRED_DIR, $RESULTS_DIR | Out-Null

if (-not (Test-Path $MAIN_RESULT)) {
    throw "Missing main experiment result: $MAIN_RESULT. Run the main experiment first so rank=16 and target_modules=all can be reused."
}

function Run-Python {
    param([string[]]$CommandArgs)

    & python @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: python $($CommandArgs -join ' ')"
    }
}

function Get-TargetModules {
    param([string]$Group)

    switch ($Group) {
        'attention_only' { return 'q_proj,k_proj,v_proj,o_proj' }
        'mlp_only' { return 'gate_proj,up_proj,down_proj' }
        default { throw "Unknown target module group: $Group" }
    }
}

Write-Host '============================================'
Write-Host '  Ablation 1 - LoRA Rank'
Write-Host '============================================'

foreach ($RANK in @('8', '32', '64')) {
    Write-Host "`n--- LoRA rank = $RANK ---"
    $OUTPUT = "outputs/ablation_rank_$RANK"

    Run-Python -CommandArgs @(
        '-m', 'src.train',
        '--config', $CONFIG,
        '--output_dir', $OUTPUT,
        '--lora_r', $RANK
    )

    Run-Python -CommandArgs @(
        '-m', 'src.inference',
        '--config', $CONFIG,
        '--adapter_path', "$OUTPUT/final",
        '--input_file', $TEST_FILE,
        '--output_file', "$PRED_DIR/rank_$RANK.jsonl"
    )

    Run-Python -CommandArgs @(
        '-m', 'src.evaluate',
        '--predictions_file', "$PRED_DIR/rank_$RANK.jsonl",
        '--output_file', "$RESULTS_DIR/ablation_rank_$RANK.json"
    )
}

Copy-Item -Force $MAIN_RESULT "$RESULTS_DIR/ablation_rank_16.json"
$mainPredPath = Join-Path (Join-Path $PRED_DIR '..') 'qlora_finetuned.jsonl'
$mainPred = Resolve-Path $mainPredPath -ErrorAction SilentlyContinue
if ($mainPred) { Copy-Item -Force $mainPred "$PRED_DIR/rank_16.jsonl" }
Write-Host "`n--- LoRA rank = 16 (copied from main experiment) ---"

Write-Host '============================================'
Write-Host '  Ablation 2 - Target Modules'
Write-Host '============================================'

foreach ($GROUP in @('attention_only', 'mlp_only')) {
    $MODULES = Get-TargetModules -Group $GROUP
    Write-Host "`n--- target_modules = $GROUP ($MODULES) ---"
    $OUTPUT = "outputs/ablation_modules_$GROUP"

    Run-Python -CommandArgs @(
        '-m', 'src.train',
        '--config', $CONFIG,
        '--output_dir', $OUTPUT,
        '--target_modules', $MODULES
    )

    Run-Python -CommandArgs @(
        '-m', 'src.inference',
        '--config', $CONFIG,
        '--adapter_path', "$OUTPUT/final",
        '--input_file', $TEST_FILE,
        '--output_file', "$PRED_DIR/modules_$GROUP.jsonl"
    )

    Run-Python -CommandArgs @(
        '-m', 'src.evaluate',
        '--predictions_file', "$PRED_DIR/modules_$GROUP.jsonl",
        '--output_file', "$RESULTS_DIR/ablation_modules_$GROUP.json"
    )
}

Copy-Item -Force $MAIN_RESULT "$RESULTS_DIR/ablation_modules_all.json"
if ($mainPred) { Copy-Item -Force $mainPred "$PRED_DIR/modules_all.jsonl" }
Write-Host "`n--- target_modules = all (copied from main experiment) ---"

Write-Host "`n============================================"
Write-Host '  Ablation studies complete!'
Write-Host "  Results: $RESULTS_DIR/"
Write-Host '============================================'
