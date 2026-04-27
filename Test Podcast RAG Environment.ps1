param(
    [string]$Config,
    [string]$CondaEnvName = "podcast-rag-pipeline",
    [string]$BaseUrl,
    [string]$Model,
    [string]$ApiKey,
    [switch]$SkipInferenceTest
)

$ErrorActionPreference = "Continue"

$ProjectRoot = $PSScriptRoot
$PythonScript = Join-Path $ProjectRoot "podcast_rag_pipeline.py"
$ConfigPath = Join-Path $ProjectRoot "podcast_rag_config.json"
$ConfigExamplePath = Join-Path $ProjectRoot "podcast_rag_config.example.json"
$RequirementsPath = Join-Path $ProjectRoot "podcast_rag_requirements.txt"

if (-not $Config) {
    $Config = if (Test-Path -LiteralPath $ConfigPath) { $ConfigPath } else { $ConfigExamplePath }
}

$script:Failed = 0
$script:Warned = 0

function Write-Check {
    param(
        [ValidateSet("PASS", "WARN", "FAIL")]
        [string]$Status,
        [string]$Name,
        [string]$Detail
    )

    $color = switch ($Status) {
        "PASS" { "Green" }
        "WARN" { "Yellow" }
        "FAIL" { "Red" }
    }

    if ($Status -eq "FAIL") {
        $script:Failed += 1
    } elseif ($Status -eq "WARN") {
        $script:Warned += 1
    }

    Write-Host ("[{0}] {1}" -f $Status, $Name) -ForegroundColor $color
    if ($Detail) {
        Write-Host ("      {0}" -f $Detail)
    }
}

function Resolve-ProjectPath {
    param([string]$Value)

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $null
    }

    if ([System.IO.Path]::IsPathRooted($Value)) {
        return $Value
    }

    return Join-Path $ProjectRoot $Value
}

function Test-WritableDirectory {
    param(
        [string]$Name,
        [string]$Path
    )

    try {
        if (-not (Test-Path -LiteralPath $Path)) {
            New-Item -ItemType Directory -Path $Path -Force | Out-Null
        }

        $testPath = Join-Path $Path (".write_test_{0}.tmp" -f [guid]::NewGuid().ToString("N"))
        "ok" | Set-Content -LiteralPath $testPath -Encoding UTF8
        Remove-Item -LiteralPath $testPath -Force
        Write-Check -Status PASS -Name $Name -Detail $Path
    } catch {
        Write-Check -Status FAIL -Name $Name -Detail ("Not writable: {0}. {1}" -f $Path, $_.Exception.Message)
    }
}

function Invoke-CondaPython {
    param([string[]]$Arguments)

    & conda run --no-capture-output -n $CondaEnvName python @Arguments
    return $LASTEXITCODE
}

Write-Host ""
Write-Host "Podcast RAG environment diagnostic"
Write-Host "Project: $ProjectRoot"
Write-Host ""

if (Test-Path -LiteralPath $PythonScript) {
    Write-Check -Status PASS -Name "Pipeline script" -Detail $PythonScript
} else {
    Write-Check -Status FAIL -Name "Pipeline script" -Detail "Missing: $PythonScript"
}

if (Test-Path -LiteralPath $RequirementsPath) {
    Write-Check -Status PASS -Name "Requirements file" -Detail $RequirementsPath
} else {
    Write-Check -Status FAIL -Name "Requirements file" -Detail "Missing: $RequirementsPath"
}

if (Test-Path -LiteralPath $Config) {
    Write-Check -Status PASS -Name "Config file" -Detail $Config
} else {
    Write-Check -Status FAIL -Name "Config file" -Detail "Missing: $Config"
}

$configObject = $null
if (Test-Path -LiteralPath $Config) {
    try {
        $configObject = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
        Write-Check -Status PASS -Name "Config JSON" -Detail "Parsed successfully."
    } catch {
        Write-Check -Status FAIL -Name "Config JSON" -Detail $_.Exception.Message
    }
}

if ((Split-Path -Leaf $Config) -eq "podcast_rag_config.example.json") {
    Write-Check -Status WARN -Name "Runtime config" -Detail "Using the example config because podcast_rag_config.json does not exist yet."
}

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Check -Status FAIL -Name "Conda command" -Detail "Conda was not found on PATH."
} else {
    try {
        $condaVersion = (& conda --version) -join " "
        Write-Check -Status PASS -Name "Conda command" -Detail $condaVersion
    } catch {
        Write-Check -Status FAIL -Name "Conda command" -Detail $_.Exception.Message
    }
}

$condaEnvFound = $false
if (Get-Command conda -ErrorAction SilentlyContinue) {
    try {
        $envListJson = & conda env list --json | ConvertFrom-Json
        foreach ($envPath in $envListJson.envs) {
            if ((Split-Path -Leaf $envPath) -eq $CondaEnvName) {
                $condaEnvFound = $true
                Write-Check -Status PASS -Name "Conda environment" -Detail ("{0} at {1}" -f $CondaEnvName, $envPath)
                break
            }
        }

        if (-not $condaEnvFound) {
            Write-Check -Status FAIL -Name "Conda environment" -Detail "Missing '$CondaEnvName'. Run: .\Run Podcast RAG Pipeline.ps1 -CreateCondaEnv"
        }
    } catch {
        Write-Check -Status FAIL -Name "Conda environment list" -Detail $_.Exception.Message
    }
}

if ($condaEnvFound) {
    $pythonVersion = & conda run --no-capture-output -n $CondaEnvName python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Check -Status PASS -Name "Environment Python" -Detail (($pythonVersion | Out-String).Trim())
    } else {
        Write-Check -Status FAIL -Name "Environment Python" -Detail (($pythonVersion | Out-String).Trim())
    }

    if (Test-Path -LiteralPath $PythonScript) {
        $compileOutput = & conda run --no-capture-output -n $CondaEnvName python -m py_compile $PythonScript 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Check -Status PASS -Name "Pipeline syntax" -Detail "py_compile completed."
        } else {
            Write-Check -Status FAIL -Name "Pipeline syntax" -Detail (($compileOutput | Out-String).Trim())
        }
    }

    $dependencyCheckPath = Join-Path ([System.IO.Path]::GetTempPath()) ("podcast_rag_dependency_check_{0}.py" -f [guid]::NewGuid().ToString("N"))
    @"
import importlib
import importlib.metadata as metadata
import sys

modules = [
    ("chromadb", "chromadb"),
    ("hdbscan", "hdbscan"),
    ("langchain", "langchain"),
    ("langchain_chroma", "langchain-chroma"),
    ("langchain_community", "langchain-community"),
    ("langchain_core", "langchain-core"),
    ("langchain_huggingface", "langchain-huggingface"),
    ("langchain_openai", "langchain-openai"),
    ("langchain_text_splitters", "langchain-text-splitters"),
    ("numpy", "numpy"),
    ("openai", "openai"),
    ("sklearn", "scikit-learn"),
    ("sentence_transformers", "sentence-transformers"),
]

failures = []
versions = []
for module_name, package_name in modules:
    try:
        importlib.import_module(module_name)
        try:
            version = metadata.version(package_name)
        except Exception:
            version = "unknown"
        versions.append(f"{package_name}={version}")
    except Exception as exc:
        failures.append(f"{module_name}: {type(exc).__name__}: {exc}")

print("VERSIONS:" + "|".join(versions))
if failures:
    print("FAILURES:" + "|".join(failures))
    sys.exit(1)
"@ | Set-Content -LiteralPath $dependencyCheckPath -Encoding UTF8

    try {
        $dependencyOutput = & conda run --no-capture-output -n $CondaEnvName python $dependencyCheckPath 2>&1
        if ($LASTEXITCODE -eq 0) {
            $versionsLine = ($dependencyOutput | Where-Object { $_ -like "VERSIONS:*" } | Select-Object -First 1)
            $detail = if ($versionsLine) { $versionsLine.Substring("VERSIONS:".Length).Replace("|", ", ") } else { "Imports completed." }
            Write-Check -Status PASS -Name "Python libraries" -Detail $detail
        } else {
            Write-Check -Status FAIL -Name "Python libraries" -Detail (($dependencyOutput | Out-String).Trim())
        }
    } finally {
        Remove-Item -LiteralPath $dependencyCheckPath -Force -ErrorAction SilentlyContinue
    }
}

if ($configObject) {
    $inputDir = Resolve-ProjectPath ([string]$configObject.input_dir)
    $fileGlob = if ($configObject.file_glob) { [string]$configObject.file_glob } else { "**/*_speaker_transcript.json" }
    $processedDir = Resolve-ProjectPath ([string]$configObject.processed_dir)
    $processedDataDir = Resolve-ProjectPath ([string]$configObject.processed_data_dir)
    $debugOutputDir = Resolve-ProjectPath ([string]$configObject.debug_output_dir)
    $statePath = Resolve-ProjectPath ([string]$configObject.state_path)
    $persistDir = Resolve-ProjectPath ([string]$configObject.persist_dir)
    $stopFile = Resolve-ProjectPath ([string]$configObject.stop_file)
    $controlFile = Resolve-ProjectPath ([string]$configObject.control_file)

    if ($inputDir) {
        if (Test-Path -LiteralPath $inputDir) {
            Write-Check -Status PASS -Name "Input directory" -Detail $inputDir
            $pattern = Join-Path $inputDir $fileGlob
            $matches = @(Get-ChildItem -Path $pattern -File -Recurse -ErrorAction SilentlyContinue)
            if ($matches.Count -gt 0) {
                Write-Check -Status PASS -Name "Transcript discovery" -Detail ("Found {0} matching file(s)." -f $matches.Count)
            } else {
                Write-Check -Status WARN -Name "Transcript discovery" -Detail ("No files matched '{0}' yet." -f $fileGlob)
            }
        } else {
            Write-Check -Status WARN -Name "Input directory" -Detail "Missing now, but the pipeline can create it: $inputDir"
        }
    }

    if ($processedDir) {
        Test-WritableDirectory -Name "Processed directory" -Path $processedDir
    }

    if ($processedDataDir) {
        Test-WritableDirectory -Name "Processed data cache directory" -Path $processedDataDir
    }

    if ($debugOutputDir) {
        Test-WritableDirectory -Name "Debug output directory" -Path $debugOutputDir
    }

    if ($statePath) {
        Test-WritableDirectory -Name "State directory" -Path (Split-Path -Parent $statePath)
    }

    if ($persistDir) {
        Test-WritableDirectory -Name "Chroma persist directory" -Path $persistDir
    }

    if ($stopFile -and (Test-Path -LiteralPath $stopFile)) {
        Write-Check -Status WARN -Name "Stop file" -Detail "A stop request is currently present: $stopFile"
    } elseif ($stopFile) {
        Write-Check -Status PASS -Name "Stop file" -Detail "No pending stop request at $stopFile"
    }

    if ($controlFile) {
        $controlDir = Split-Path -Parent $controlFile
        Test-WritableDirectory -Name "Control directory" -Path $controlDir
        if (Test-Path -LiteralPath $controlFile) {
            try {
                $controlObject = Get-Content -LiteralPath $controlFile -Raw | ConvertFrom-Json
                $parallel = if ($controlObject.max_parallel_model_requests) { [int]$controlObject.max_parallel_model_requests } else { 0 }
                if ($parallel -ge 1) {
                    Write-Check -Status PASS -Name "Live control file" -Detail ("{0}; max_parallel_model_requests={1}" -f $controlFile, $parallel)
                } else {
                    Write-Check -Status WARN -Name "Live control file" -Detail "File exists but max_parallel_model_requests is missing or invalid."
                }
            } catch {
                Write-Check -Status WARN -Name "Live control file" -Detail ("Could not parse {0}: {1}" -f $controlFile, $_.Exception.Message)
            }
        } else {
            Write-Check -Status WARN -Name "Live control file" -Detail "Will be created on the next pipeline run: $controlFile"
        }
    }
}

$resolvedBaseUrl = if ($BaseUrl) { $BaseUrl } elseif ($env:LM_STUDIO_BASE_URL) { $env:LM_STUDIO_BASE_URL } elseif ($configObject -and $configObject.lm_studio_base_url) { [string]$configObject.lm_studio_base_url } else { "http://127.0.0.1:1234/v1" }
$resolvedModel = if ($Model) { $Model } elseif ($env:LM_STUDIO_MODEL) { $env:LM_STUDIO_MODEL } elseif ($configObject -and $configObject.lm_studio_model) { [string]$configObject.lm_studio_model } else { "" }
$resolvedApiKey = if ($ApiKey) { $ApiKey } elseif ($env:LM_STUDIO_API_KEY) { $env:LM_STUDIO_API_KEY } elseif ($configObject -and $configObject.lm_studio_api_key) { [string]$configObject.lm_studio_api_key } else { "lm-studio" }

try {
    $modelsUri = ($resolvedBaseUrl.TrimEnd("/") + "/models")
    $headers = @{ Authorization = "Bearer $resolvedApiKey" }
    $modelsResponse = Invoke-RestMethod -Uri $modelsUri -Headers $headers -Method Get -TimeoutSec 15
    $availableModels = @($modelsResponse.data | ForEach-Object { $_.id })
    Write-Check -Status PASS -Name "LM Studio API" -Detail "Connected to $modelsUri"

    if ($resolvedModel) {
        if ($availableModels -contains $resolvedModel) {
            Write-Check -Status PASS -Name "LM Studio model" -Detail $resolvedModel
        } else {
            Write-Check -Status FAIL -Name "LM Studio model" -Detail ("'{0}' not found. Available: {1}" -f $resolvedModel, ($availableModels -join ", "))
        }
    } else {
        Write-Check -Status WARN -Name "LM Studio model" -Detail "No model name configured."
    }

    if (-not $SkipInferenceTest -and $resolvedModel -and ($availableModels -contains $resolvedModel)) {
        $chatUri = ($resolvedBaseUrl.TrimEnd("/") + "/chat/completions")
        $body = @{
            model = $resolvedModel
            messages = @(@{ role = "user"; content = "ping" })
            max_tokens = 1
            temperature = 0
        } | ConvertTo-Json -Depth 5

        $null = Invoke-RestMethod -Uri $chatUri -Headers $headers -Method Post -Body $body -ContentType "application/json" -TimeoutSec 60
        Write-Check -Status PASS -Name "LM Studio inference" -Detail "One-token chat completion succeeded."
    } elseif ($SkipInferenceTest) {
        Write-Check -Status WARN -Name "LM Studio inference" -Detail "Skipped by request."
    }
} catch {
    Write-Check -Status FAIL -Name "LM Studio API" -Detail ("Could not reach {0}. {1}" -f $resolvedBaseUrl, $_.Exception.Message)
}

Write-Host ""
if ($script:Failed -gt 0) {
    Write-Host ("Diagnostic failed with {0} failure(s) and {1} warning(s)." -f $script:Failed, $script:Warned) -ForegroundColor Red
    exit 1
}

if ($script:Warned -gt 0) {
    Write-Host ("Diagnostic passed with {0} warning(s)." -f $script:Warned) -ForegroundColor Yellow
    exit 0
}

Write-Host "Diagnostic passed with no warnings." -ForegroundColor Green
exit 0
