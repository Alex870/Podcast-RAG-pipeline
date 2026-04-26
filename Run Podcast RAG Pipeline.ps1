param(
    [string]$Config,
    [string]$InputDir,
    [string]$FileGlob,
    [string]$Model,
    [string]$BaseUrl,
    [switch]$OneFile,
    [switch]$CreateStopFile,
    [switch]$ClearStopFile,
    [switch]$SkipDependencyCheck
)

$PythonScript = Join-Path $PSScriptRoot "podcast_rag_pipeline.py"
$ConfigPath = Join-Path $PSScriptRoot "podcast_rag_config.json"
$ConfigExamplePath = Join-Path $PSScriptRoot "podcast_rag_config.example.json"

if (-not $Config) {
    $Config = $ConfigPath
}

function Test-PythonDependencies {
    $dependencyCheck = @"
import importlib
import sys

required = [
    'chromadb',
    'hdbscan',
    'langchain_chroma',
    'langchain_community',
    'langchain_huggingface',
    'langchain_openai',
    'numpy',
    'openai',
    'sklearn',
    'umap',
]
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f'{name}: {type(exc).__name__}: {exc}')

if missing:
    print('MISSING:' + '|'.join(missing))
    sys.exit(1)
"@

    & python -c $dependencyCheck
    return $LASTEXITCODE
}

if (-not (Test-Path -LiteralPath $Config)) {
    if (-not (Test-Path -LiteralPath $ConfigExamplePath)) {
        throw "Missing config file and example config: $ConfigExamplePath"
    }

    Copy-Item -LiteralPath $ConfigExamplePath -Destination $Config
    Write-Host "Created config: $Config"
}

if ($CreateStopFile) {
    & python $PythonScript --config $Config --create-stop-file
    exit $LASTEXITCODE
}

if ($ClearStopFile) {
    $configObject = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
    $stopPath = if ($configObject.stop_file) { [string]$configObject.stop_file } else { "state/stop_after_current.txt" }
    if (-not [System.IO.Path]::IsPathRooted($stopPath)) {
        $stopPath = Join-Path $PSScriptRoot $stopPath
    }
    if (Test-Path -LiteralPath $stopPath) {
        Remove-Item -LiteralPath $stopPath
        Write-Host "Removed stop file: $stopPath"
    }
}

if (-not $SkipDependencyCheck) {
    $dependencyExitCode = Test-PythonDependencies
    if ($dependencyExitCode -ne 0) {
        Write-Host ""
        Write-Host "Python dependencies are missing. Install them with:"
        Write-Host "  python -m pip install -r `"$PSScriptRoot\podcast_rag_requirements.txt`""
        exit $dependencyExitCode
    }
}

$argsList = @("--config", $Config)

if ($InputDir) {
    $argsList += @("--input-dir", $InputDir)
}

if ($FileGlob) {
    $argsList += @("--file-glob", $FileGlob)
}

if ($Model) {
    $argsList += @("--model", $Model)
}

if ($BaseUrl) {
    $argsList += @("--base-url", $BaseUrl)
}

if ($OneFile) {
    $argsList += "--one-file"
}

& python $PythonScript @argsList
exit $LASTEXITCODE
