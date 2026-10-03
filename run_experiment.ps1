param([ValidateSet('sanity','baseline')][string]$Mode='baseline')
$ErrorActionPreference='Stop'
Push-Location $PSScriptRoot
try {
    $python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python)) { throw 'Create .venv first; see README.md.' }
    & $python scripts/experiment.py prepare
    if ($LASTEXITCODE -ne 0) { throw 'Dataset validation failed.' }
    $output = "results/table1_$Mode"
    $arguments = @('scripts/experiment.py','run','--output',$output)
    if ($Mode -eq 'sanity') { $arguments += @('--limit','20') }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw 'Inference failed; inspect failures.json.' }
    & $python scripts/make_figures.py --output $output
    if ($LASTEXITCODE -ne 0) { throw 'Evaluation/report failed.' }
    Write-Host "Report: $output/report.md"
} finally { Pop-Location }
