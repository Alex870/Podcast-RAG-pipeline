param(
    [string]$Config,
    [int]$MaxParallelModelRequests
)

function Wait-ForExitPrompt {
    if (-not $env:PODCAST_RAG_SUPPRESS_PAUSE -and $Host.Name -eq "ConsoleHost") {
        [void](Read-Host "Press Enter to continue")
    }
}

function Exit-Script {
    param([int]$Code = 0)
    Wait-ForExitPrompt
    exit $Code
}

trap {
    Write-Error $_
    Exit-Script 1
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ConfigPath = Join-Path $ProjectRoot "podcast_rag_config.json"
$ConfigExamplePath = Join-Path $ProjectRoot "examples\podcast_rag_config.example.json"

if (-not $Config) {
    $Config = if (Test-Path -LiteralPath $ConfigPath) { $ConfigPath } else { $ConfigExamplePath }
}

if (-not (Test-Path -LiteralPath $Config)) {
    throw "Missing config: $Config"
}

if (-not $MaxParallelModelRequests -or $MaxParallelModelRequests -lt 1) {
    throw "Provide -MaxParallelModelRequests with a value of 1 or higher."
}

$configObject = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
$controlPath = if ($configObject.control_file) { [string]$configObject.control_file } else { "state/pipeline_control.json" }
if (-not [System.IO.Path]::IsPathRooted($controlPath)) {
    $controlPath = Join-Path $ProjectRoot $controlPath
}

$controlDir = Split-Path -Parent $controlPath
if ($controlDir -and -not (Test-Path -LiteralPath $controlDir)) {
    New-Item -ItemType Directory -Path $controlDir -Force | Out-Null
}

$payload = [ordered]@{
    max_parallel_model_requests = $MaxParallelModelRequests
    updated_at = (Get-Date -Format o)
    note = "The running pipeline reloads this file before launching new model requests."
}

$payload | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $controlPath -Encoding UTF8
Write-Host "Set max_parallel_model_requests=$MaxParallelModelRequests in $controlPath"
Exit-Script 0
