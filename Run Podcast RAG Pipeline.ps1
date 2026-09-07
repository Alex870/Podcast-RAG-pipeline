param(
    [ValidateSet("Prompt", "Run", "Debug", "CacheCheck", "SetControl", "CreateStopFile", "ClearStopFile", "CreateCondaEnv", "BuildTopicIndex", "Migrate", "Partitions")]
    [string]$Action = "Prompt",
    [int]$MaxParallelModelRequests,
    [string]$Config = "",
    [string]$CondaEnvName = "podcast-rag-pipeline"
)

function Exit-Script {
    param([int]$Code = 0)
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
$MigrationScript = Join-Path $ScriptRoot "scripts\Migrate-LegacyPodcastRagState.ps1"
$PartitionScript = Join-Path $ScriptRoot "scripts\Manage-PodcastRagPartitions.ps1"

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

function Invoke-InteractiveChild {
    param(
        [string]$Path,
        [hashtable]$Parameters = @{}
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        Write-Error "Missing launcher script: $Path"
        return 1
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
    if ($null -eq $childExitCode) { return 0 }
    return $childExitCode
}

function Pause-InteractiveMenu {
    [void](Read-Host "Press Enter to return to the main menu")
}

function Invoke-MaintenanceMenu {
    while ($true) {
        Write-Host ""
        Write-Host "Maintenance" -ForegroundColor Cyan
        Write-Host "  1. Check legacy/default processed-cache health"
        Write-Host "  2. Build or refresh the legacy/default topic index"
        Write-Host "  Q. Back"
        $selection = (Read-Host "Choose an action").Trim().ToUpperInvariant()
        switch ($selection) {
            "1" {
                [void](Invoke-InteractiveChild -Path $CacheScript -Parameters @{ Config = $Config })
                Pause-InteractiveMenu
            }
            "2" {
                [void](Invoke-InteractiveChild -Path $RunScript -Parameters @{ Config = $Config; CondaEnvName = $CondaEnvName; BuildTopicIndex = $true })
                Pause-InteractiveMenu
            }
            "Q" { return }
            default { Write-Host "Please choose one of the displayed actions." -ForegroundColor Yellow }
        }
    }
}

function Start-InteractiveMenu {
    while ($true) {
        Write-Host ""
        Write-Host "Podcast RAG Pipeline" -ForegroundColor Cyan
        Write-Host "Partition-first interactive menu"
        Write-Host "  1. Process / resume pending partition work"
        Write-Host "  2. Manage partitions"
        Write-Host "  3. View partition status"
        Write-Host "  4. Control a running partition"
        Write-Host "  5. Validate the environment"
        Write-Host "  6. Create or refresh the Conda environment"
        Write-Host "  7. Cache and topic maintenance"
        Write-Host "  8. Advanced legacy flat-input processing"
        Write-Host "  9. Migrate legacy settings and state"
        Write-Host "  Q. Quit"
        $selection = (Read-Host "Choose an action").Trim().ToUpperInvariant()
        switch ($selection) {
            "1" {
                $childExitCode = Invoke-InteractiveChild -Path $PartitionScript -Parameters @{ Config = $Config; CondaEnvName = $CondaEnvName; Mode = "Process" }
                if ($childExitCode -eq 130) { return }
                Pause-InteractiveMenu
            }
            "2" {
                [void](Invoke-InteractiveChild -Path $PartitionScript -Parameters @{ Config = $Config; CondaEnvName = $CondaEnvName; Mode = "Menu" })
            }
            "3" {
                [void](Invoke-InteractiveChild -Path $PartitionScript -Parameters @{ Config = $Config; CondaEnvName = $CondaEnvName; Mode = "Status" })
                Pause-InteractiveMenu
            }
            "4" {
                [void](Invoke-InteractiveChild -Path $PartitionScript -Parameters @{ Config = $Config; CondaEnvName = $CondaEnvName; Mode = "Control" })
                Pause-InteractiveMenu
            }
            "5" {
                [void](Invoke-InteractiveChild -Path $DebugScript -Parameters @{ Config = $Config; CondaEnvName = $CondaEnvName })
                Pause-InteractiveMenu
            }
            "6" {
                [void](Invoke-InteractiveChild -Path $RunScript -Parameters @{ Config = $Config; CondaEnvName = $CondaEnvName; CreateCondaEnv = $true })
                Pause-InteractiveMenu
            }
            "7" { Invoke-MaintenanceMenu }
            "8" {
                Write-Host ""
                Write-Host "Legacy flat-input processing uses the configured input_dir and state paths." -ForegroundColor Yellow
                $childExitCode = Invoke-InteractiveChild -Path $RunScript -Parameters @{ Config = $Config; CondaEnvName = $CondaEnvName }
                if ($childExitCode -eq 130) { return }
                Pause-InteractiveMenu
            }
            "9" {
                [void](Invoke-InteractiveChild -Path $MigrationScript -Parameters @{})
                Pause-InteractiveMenu
            }
            "Q" { return }
            default { Write-Host "Please choose one of the displayed actions." -ForegroundColor Yellow }
        }
    }
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
    Start-InteractiveMenu
    Exit-Script 0
}

switch ($Action) {
    "Debug" {
        Invoke-LauncherScript -Path $DebugScript -Parameters @{ Config = $Config; CondaEnvName = $CondaEnvName }
    }
    "Run" {
        Invoke-LauncherScript -Path $RunScript -Parameters @{ Config = $Config; CondaEnvName = $CondaEnvName }
    }
    "CacheCheck" {
        Invoke-LauncherScript -Path $CacheScript -Parameters @{ Config = $Config }
    }
    "SetControl" {
        if (-not $MaxParallelModelRequests -or $MaxParallelModelRequests -lt 1) {
            $MaxParallelModelRequests = Read-PositiveInteger -Prompt "Enter max_parallel_model_requests"
        }
        Invoke-LauncherScript -Path $ControlScript -Parameters @{ Config = $Config; MaxParallelModelRequests = $MaxParallelModelRequests }
    }
    "CreateStopFile" {
        Invoke-LauncherScript -Path $RunScript -Parameters @{ Config = $Config; CondaEnvName = $CondaEnvName; CreateStopFile = $true }
    }
    "ClearStopFile" {
        Invoke-LauncherScript -Path $RunScript -Parameters @{ Config = $Config; CondaEnvName = $CondaEnvName; ClearStopFile = $true }
    }
    "CreateCondaEnv" {
        Invoke-LauncherScript -Path $RunScript -Parameters @{ Config = $Config; CondaEnvName = $CondaEnvName; CreateCondaEnv = $true }
    }
    "BuildTopicIndex" {
        Invoke-LauncherScript -Path $RunScript -Parameters @{ Config = $Config; CondaEnvName = $CondaEnvName; BuildTopicIndex = $true }
    }
    "Migrate" {
        Invoke-LauncherScript -Path $MigrationScript
    }
    "Partitions" {
        Invoke-LauncherScript -Path $PartitionScript -Parameters @{ Config = $Config; CondaEnvName = $CondaEnvName; Mode = "Menu" }
    }
}

Exit-Script 0
