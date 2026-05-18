param(
    [ValidateSet("Prompt", "Run", "Debug", "CacheCheck", "SetControl", "CreateStopFile", "ClearStopFile", "CreateCondaEnv")]
    [string]$Action = "Prompt",
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

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RunScript = Join-Path $ScriptRoot "scripts\Run-PodcastRagPipeline.ps1"
$DebugScript = Join-Path $ScriptRoot "scripts\Test-PodcastRagEnvironment.ps1"
$CacheScript = Join-Path $ScriptRoot "scripts\Test-ProcessedDataCache.ps1"
$ControlScript = Join-Path $ScriptRoot "scripts\Set-PodcastRagControl.ps1"

function Invoke-LauncherScript {
    param(
        [string]$Path,
        [hashtable]$Parameters = @{}
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing launcher script: $Path"
    }

    $previousSuppressPause = $env:PODCAST_RAG_SUPPRESS_PAUSE
    $env:PODCAST_RAG_SUPPRESS_PAUSE = "1"
    try {
        & $Path @Parameters
        $childExitCode = $LASTEXITCODE
    } finally {
        if ($null -eq $previousSuppressPause) {
            Remove-Item Env:PODCAST_RAG_SUPPRESS_PAUSE -ErrorAction SilentlyContinue
        } else {
            $env:PODCAST_RAG_SUPPRESS_PAUSE = $previousSuppressPause
        }
    }

    Exit-Script $childExitCode
}

function Read-PositiveInteger {
    param(
        [string]$Prompt
    )

    while ($true) {
        $inputValue = (Read-Host $Prompt).Trim()
        if ([int]::TryParse($inputValue, [ref]$parsedValue) -and $parsedValue -ge 1) {
            return $parsedValue
        }
        Write-Host "Please enter an integer value of 1 or higher." -ForegroundColor Yellow
    }
}

if ($Action -eq "Prompt") {
    Write-Host ""
    Write-Host "Podcast RAG Pipeline"
    Write-Host "Choose what to run:"
    Write-Host "  1. Run environment validation"
    Write-Host "  2. Run the main RAG pipeline"
    Write-Host "  3. Check processed-data cache health"
    Write-Host "  4. Set live max_parallel_model_requests"
    Write-Host "  5. Create stop-after-current-file request"
    Write-Host "  6. Clear stop-after-current-file request"
    Write-Host "  7. Create or refresh the Conda environment"
    Write-Host "  Q. Quit"
    $selection = (Read-Host "Enter 1, 2, 3, 4, 5, 6, 7, or Q").Trim()

    switch ($selection.ToUpperInvariant()) {
        "1" { $Action = "Debug" }
        "2" { $Action = "Run" }
        "3" { $Action = "CacheCheck" }
        "4" { $Action = "SetControl" }
        "5" { $Action = "CreateStopFile" }
        "6" { $Action = "ClearStopFile" }
        "7" { $Action = "CreateCondaEnv" }
        "Q" { Exit-Script 0 }
        default {
            Write-Host "Unrecognized selection. Exiting."
            Exit-Script 1
        }
    }
}

switch ($Action) {
    "Debug" {
        Invoke-LauncherScript -Path $DebugScript
    }
    "Run" {
        Invoke-LauncherScript -Path $RunScript
    }
    "CacheCheck" {
        Invoke-LauncherScript -Path $CacheScript
    }
    "SetControl" {
        if (-not $MaxParallelModelRequests -or $MaxParallelModelRequests -lt 1) {
            $MaxParallelModelRequests = Read-PositiveInteger -Prompt "Enter max_parallel_model_requests"
        }
        Invoke-LauncherScript -Path $ControlScript -Parameters @{ MaxParallelModelRequests = $MaxParallelModelRequests }
    }
    "CreateStopFile" {
        Invoke-LauncherScript -Path $RunScript -Parameters @{ CreateStopFile = $true }
    }
    "ClearStopFile" {
        Invoke-LauncherScript -Path $RunScript -Parameters @{ ClearStopFile = $true }
    }
    "CreateCondaEnv" {
        Invoke-LauncherScript -Path $RunScript -Parameters @{ CreateCondaEnv = $true }
    }
}

Exit-Script 0
