param(
    [string]$Config,
    [string]$CondaEnvName = "podcast-rag-pipeline",
    [string]$InputDir,
    [string]$FileGlob,
    [string]$Model,
    [string]$BaseUrl,
    [int]$MaxParallelModelRequests,
    [switch]$OneFile,
    [switch]$CreateStopFile,
    [switch]$ClearStopFile,
    [switch]$InspectCache,
    [switch]$ConfigDoctor,
    [switch]$ModelEval,
    [int]$ModelEvalLimit = 3,
    [switch]$BuildTopicIndex,
    [switch]$FakeLlm,
    [switch]$CreateCondaEnv,
    [switch]$SkipDependencyCheck
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
$PythonScript = Join-Path $ProjectRoot "podcast_rag_pipeline.py"
$ConfigPath = Join-Path $ProjectRoot "podcast_rag_config.json"
$ConfigExamplePath = Join-Path $ProjectRoot "examples\podcast_rag_config.example.json"
$RequirementsPath = Join-Path $ProjectRoot "podcast_rag_requirements.txt"

if (-not $Config) {
    $Config = $ConfigPath
}

function Invoke-ProjectPython {
    param(
        [string[]]$Arguments
    )

    & conda run --no-capture-output -n $CondaEnvName python @Arguments
}

function Test-CondaEnv {
    $envListJson = & conda env list --json | ConvertFrom-Json
    foreach ($envPath in $envListJson.envs) {
        if ((Split-Path -Leaf $envPath) -eq $CondaEnvName) {
            return $true
        }
    }
    return $false
}

function New-ProjectCondaEnv {
    if (Test-CondaEnv) {
        Write-Host "Conda environment already exists: $CondaEnvName"
        return
    }

    & conda create -y -n $CondaEnvName python=3.11 pip
    if ($LASTEXITCODE -eq 0) {
        & conda run --no-capture-output -n $CondaEnvName python -m pip install --upgrade pip
    }
    if ($LASTEXITCODE -eq 0) {
        & conda run --no-capture-output -n $CondaEnvName python -m pip install -r $RequirementsPath
    }

    if ($LASTEXITCODE -ne 0) {
        Exit-Script $LASTEXITCODE
    }
}

function Test-PythonDependencies {
    $dependencyCheckPath = Join-Path ([System.IO.Path]::GetTempPath()) ("podcast_rag_dependency_check_{0}.py" -f [guid]::NewGuid().ToString("N"))
    @"
import importlib
import sys

required = [
    'hdbscan',
    'langchain_community',
    'langchain_huggingface',
    'langchain_openai',
    'numpy',
    'openai',
    'sklearn',
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
"@ | Set-Content -LiteralPath $dependencyCheckPath -Encoding UTF8

    try {
        Invoke-ProjectPython -Arguments @($dependencyCheckPath)
        return $LASTEXITCODE
    } finally {
        Remove-Item -LiteralPath $dependencyCheckPath -Force -ErrorAction SilentlyContinue
    }
}

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Error "Conda was not found on PATH. Open a Miniconda/Anaconda PowerShell prompt, or add Conda to PATH."
    Exit-Script 1
}

if (-not (Test-Path -LiteralPath $Config)) {
    if (-not (Test-Path -LiteralPath $ConfigExamplePath)) {
        Write-Error "Missing config file and example config: $ConfigExamplePath"
        Exit-Script 1
    }

    Copy-Item -LiteralPath $ConfigExamplePath -Destination $Config
    Write-Host "Created config: $Config"
}

if ($CreateStopFile) {
    $configObject = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
    $stopPath = if ($configObject.stop_file) { [string]$configObject.stop_file } else { "state/stop_after_current.txt" }
    if (-not [System.IO.Path]::IsPathRooted($stopPath)) {
        $stopPath = Join-Path $ProjectRoot $stopPath
    }
    $stopDir = Split-Path -Parent $stopPath
    if ($stopDir -and -not (Test-Path -LiteralPath $stopDir)) {
        New-Item -ItemType Directory -Path $stopDir | Out-Null
    }
    "Stop requested at $(Get-Date -Format o)" | Set-Content -LiteralPath $stopPath -Encoding UTF8
    Write-Host "Created stop file: $stopPath"
    Exit-Script 0
}

if ($ClearStopFile) {
    $configObject = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
    $stopPath = if ($configObject.stop_file) { [string]$configObject.stop_file } else { "state/stop_after_current.txt" }
    if (-not [System.IO.Path]::IsPathRooted($stopPath)) {
        $stopPath = Join-Path $ProjectRoot $stopPath
    }
    if (Test-Path -LiteralPath $stopPath) {
        Remove-Item -LiteralPath $stopPath
        Write-Host "Removed stop file: $stopPath"
    }
}

if ($CreateCondaEnv) {
    New-ProjectCondaEnv
    Exit-Script 0
}

if (-not (Test-CondaEnv)) {
    Write-Host "Conda environment '$CondaEnvName' was not found."
    Write-Host "Create it with:"
    Write-Host "  .\scripts\Run-PodcastRagPipeline.ps1 -CreateCondaEnv"
    Exit-Script 1
}

if (-not $SkipDependencyCheck) {
    $dependencyExitCode = Test-PythonDependencies
    if ($dependencyExitCode -ne 0) {
        Write-Host ""
        Write-Host "Python dependencies are missing. Install them with:"
        Write-Host "  conda run -n $CondaEnvName python -m pip install -r `"$RequirementsPath`""
        Exit-Script $dependencyExitCode
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

if ($MaxParallelModelRequests) {
    $argsList += @("--max-parallel-model-requests", $MaxParallelModelRequests)
}

if ($OneFile) {
    $argsList += "--one-file"
}

if ($InspectCache) {
    $argsList += "--inspect-cache"
}

if ($ConfigDoctor) {
    $argsList += "--config-doctor"
}

if ($ModelEval) {
    $argsList += @("--model-eval", "--model-eval-limit", $ModelEvalLimit)
}

if ($BuildTopicIndex) {
    $argsList += "--build-topic-index"
}

if ($FakeLlm) {
    $argsList += "--fake-llm"
}

$pythonArgs = @($PythonScript) + $argsList
Invoke-ProjectPython -Arguments $pythonArgs
Exit-Script $LASTEXITCODE
