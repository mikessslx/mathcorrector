param(
    [string]$Config = 'configs/default.yaml',
    [string]$OutputDir = 'outputs/qlora_r16'
)

$ErrorActionPreference = 'Stop'

function Run-Python {
    param([string[]]$CommandArgs)

    & python @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: python $($CommandArgs -join ' ')"
    }
}

Write-Host '=== Training QLoRA Math Error Correction ==='
Write-Host "Config: $Config"
Write-Host "Output: $OutputDir"

Run-Python -CommandArgs @(
    '-m', 'src.train',
    '--config', $Config,
    '--output_dir', $OutputDir
)

Write-Host '=== Training complete ==='
Write-Host "Adapter saved to: $OutputDir/final/"
