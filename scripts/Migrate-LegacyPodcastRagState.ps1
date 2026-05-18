Add-Type -AssemblyName System.Windows.Forms

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$script:Results = New-Object System.Collections.Generic.List[object]

function Wait-ForExitPrompt {
    if (-not $env:PODCAST_RAG_SUPPRESS_PAUSE -and $Host.Name -eq "ConsoleHost") {
        [void](Read-Host "Press Enter to continue")
    }
}

function Add-Result {
    param(
        [string]$Status,
        [string]$Label,
        [string]$Detail
    )

    $script:Results.Add([pscustomobject]@{
        Status = $Status
        Label  = $Label
        Detail = $Detail
    })
}

function Write-ResultSummary {
    Write-Host ""
    Write-Host ("=" * 78)
    Write-Host "Migration Checklist"
    Write-Host ("=" * 78)

    foreach ($result in $script:Results) {
        switch ($result.Status) {
            "PASS" { $color = "Green" }
            "WARN" { $color = "Yellow" }
            "FAIL" { $color = "Red" }
            default { $color = "Gray" }
        }

        Write-Host -NoNewline ("[{0}]" -f $result.Status) -ForegroundColor $color
        Write-Host (" {0}: {1}" -f $result.Label, $result.Detail)
    }
}

function Select-LegacyFolder {
    param(
        [string]$InitialFolder
    )

    $dialog = New-Object System.Windows.Forms.FolderBrowserDialog -Property @{
        RootFolder  = "MyComputer"
        Description = "Select the legacy Podcast-RAG-pipeline directory to migrate from."
    }

    if ($InitialFolder -and (Test-Path -LiteralPath $InitialFolder)) {
        $dialog.SelectedPath = $InitialFolder
    }

    $selection = $dialog.ShowDialog()
    if ($selection -ne [System.Windows.Forms.DialogResult]::OK) {
        return $null
    }

    return $dialog.SelectedPath
}

function Backup-IfExists {
    param(
        [string]$TargetPath
    )

    if (-not (Test-Path -LiteralPath $TargetPath)) {
        return $null
    }

    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $backupPath = "$TargetPath.migration-backup-$timestamp"
    Move-Item -LiteralPath $TargetPath -Destination $backupPath -Force
    return $backupPath
}

function Copy-FileWithBackup {
    param(
        [string]$SourcePath,
        [string]$TargetPath,
        [string]$Label
    )

    if (-not (Test-Path -LiteralPath $SourcePath -PathType Leaf)) {
        Add-Result "WARN" $Label ("Not found in legacy directory: {0}" -f $SourcePath)
        return
    }

    $targetDir = Split-Path -Parent $TargetPath
    if ($targetDir) {
        New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
    }

    $backupPath = Backup-IfExists -TargetPath $TargetPath
    Copy-Item -LiteralPath $SourcePath -Destination $TargetPath -Force

    if ($backupPath) {
        Add-Result "PASS" $Label ("Copied to {0}; previous file backed up to {1}" -f $TargetPath, $backupPath)
    } else {
        Add-Result "PASS" $Label ("Copied to {0}" -f $TargetPath)
    }
}

function Copy-DirectoryMerge {
    param(
        [string]$SourcePath,
        [string]$TargetPath,
        [string]$Label
    )

    if (-not (Test-Path -LiteralPath $SourcePath -PathType Container)) {
        Add-Result "WARN" $Label ("Not found in legacy directory: {0}" -f $SourcePath)
        return
    }

    New-Item -ItemType Directory -Path $TargetPath -Force | Out-Null
    $items = @(Get-ChildItem -LiteralPath $SourcePath -Force)
    if ($items.Count -eq 0) {
        Add-Result "WARN" $Label ("Directory exists but is empty: {0}" -f $SourcePath)
        return
    }

    foreach ($item in $items) {
        Copy-Item -LiteralPath $item.FullName -Destination $TargetPath -Recurse -Force
    }

    Add-Result "PASS" $Label ("Merged {0} item(s) from {1} into {2}" -f $items.Count, $SourcePath, $TargetPath)
}

function Copy-DirectoryWithProgress {
    param(
        [string]$SourcePath,
        [string]$TargetPath,
        [string]$Label
    )

    if (-not (Test-Path -LiteralPath $SourcePath -PathType Container)) {
        Add-Result "WARN" $Label ("Not found in legacy directory: {0}" -f $SourcePath)
        return
    }

    New-Item -ItemType Directory -Path $TargetPath -Force | Out-Null

    $sourceRoot = [System.IO.Path]::GetFullPath($SourcePath).TrimEnd('\', '/')
    $files = @(Get-ChildItem -LiteralPath $SourcePath -Recurse -File -Force)
    $directories = @(Get-ChildItem -LiteralPath $SourcePath -Recurse -Directory -Force)

    foreach ($directory in $directories) {
        $relativeDirectory = $directory.FullName.Substring($sourceRoot.Length).TrimStart('\', '/')
        $targetDirectory = Join-Path $TargetPath $relativeDirectory
        New-Item -ItemType Directory -Path $targetDirectory -Force | Out-Null
    }

    if ($files.Count -eq 0) {
        Add-Result "WARN" $Label ("Directory exists but contains no files: {0}" -f $SourcePath)
        return
    }

    $activity = "{0}: copying files" -f $Label
    for ($index = 0; $index -lt $files.Count; $index++) {
        $file = $files[$index]
        $relativeFile = $file.FullName.Substring($sourceRoot.Length).TrimStart('\', '/')
        $targetFile = Join-Path $TargetPath $relativeFile
        $targetDirectory = Split-Path -Parent $targetFile
        if ($targetDirectory) {
            New-Item -ItemType Directory -Path $targetDirectory -Force | Out-Null
        }

        $percentComplete = [int](($index / $files.Count) * 100)
        Write-Progress -Id 1 -Activity $activity -Status ("{0} of {1}: {2}" -f ($index + 1), $files.Count, $relativeFile) -PercentComplete $percentComplete
        Copy-Item -LiteralPath $file.FullName -Destination $targetFile -Force
    }

    Write-Progress -Id 1 -Activity $activity -Completed
    Add-Result "PASS" $Label ("Copied {0} file(s) from {1} into {2}" -f $files.Count, $SourcePath, $TargetPath)
}

function Write-Utf8NoBomFile {
    param(
        [string]$Path,
        [string]$Content
    )

    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Content, $encoding)
}

function Test-PathContainedBy {
    param(
        [string]$Path,
        [string]$Container
    )

    if ([string]::IsNullOrWhiteSpace($Path) -or [string]::IsNullOrWhiteSpace($Container)) {
        return $false
    }

    try {
        $resolvedPath = [System.IO.Path]::GetFullPath($Path)
        $resolvedContainer = [System.IO.Path]::GetFullPath($Container)
    } catch {
        return $false
    }

    $resolvedContainer = $resolvedContainer.TrimEnd('\', '/')
    return $resolvedPath -eq $resolvedContainer -or $resolvedPath.StartsWith("$resolvedContainer\", [System.StringComparison]::OrdinalIgnoreCase)
}

function Convert-ToProjectRelativePath {
    param(
        [string]$Path,
        [string]$BasePath
    )

    $resolvedPath = [System.IO.Path]::GetFullPath($Path)
    $resolvedBase = [System.IO.Path]::GetFullPath($BasePath)
    $uriPath = New-Object System.Uri(($resolvedPath.TrimEnd('\') + '\'))
    $uriBase = New-Object System.Uri(($resolvedBase.TrimEnd('\') + '\'))
    $relative = $uriBase.MakeRelativeUri($uriPath).ToString().TrimEnd('/')
    return [System.Uri]::UnescapeDataString($relative).Replace('/', '\')
}

function Test-PathTypeMatch {
    param(
        [string]$Path,
        [string]$PathType
    )

    switch ($PathType) {
        "File" { return (Test-Path -LiteralPath $Path -PathType Leaf) }
        "Directory" { return (Test-Path -LiteralPath $Path -PathType Container) }
        default { return (Test-Path -LiteralPath $Path) }
    }
}

function Get-ConfigValueOrDefault {
    param(
        [psobject]$ConfigObject,
        [string]$Key,
        [string]$DefaultValue
    )

    if ($null -ne $ConfigObject -and $ConfigObject.PSObject.Properties[$Key]) {
        $value = [string]$ConfigObject.$Key
        if (-not [string]::IsNullOrWhiteSpace($value)) {
            return $value
        }
    }
    return $DefaultValue
}

function Resolve-ConfiguredPath {
    param(
        [string]$ConfiguredPath,
        [string]$LegacyRoot,
        [string]$CurrentProjectRoot,
        [string]$PathType
    )

    if ([string]::IsNullOrWhiteSpace($ConfiguredPath)) {
        return [pscustomobject]@{
            ConfigValue   = $ConfiguredPath
            RelativePath  = $null
            PortablePath  = $null
            LegacyPath    = $null
            Resolution    = "empty"
            ShouldRemap   = $false
        }
    }

    $normalizedConfigured = $ConfiguredPath.Replace('/', '\')
    if ([System.IO.Path]::IsPathRooted($normalizedConfigured)) {
        if (Test-PathContainedBy -Path $normalizedConfigured -Container $LegacyRoot) {
            $relativePath = Convert-ToProjectRelativePath -Path $normalizedConfigured -BasePath $LegacyRoot
            $legacyPath = if (Test-PathTypeMatch -Path $normalizedConfigured -PathType $PathType) { $normalizedConfigured } else { $null }
            return [pscustomobject]@{
                ConfigValue   = $ConfiguredPath
                RelativePath  = $relativePath
                PortablePath  = $relativePath.Replace('\', '/')
                LegacyPath    = $legacyPath
                Resolution    = "legacy_absolute"
                ShouldRemap   = $true
            }
        }

        if (Test-PathContainedBy -Path $normalizedConfigured -Container $CurrentProjectRoot) {
            $relativePath = Convert-ToProjectRelativePath -Path $normalizedConfigured -BasePath $CurrentProjectRoot
            $legacyCandidate = Join-Path $LegacyRoot $relativePath
            return [pscustomobject]@{
                ConfigValue   = $ConfiguredPath
                RelativePath  = $relativePath
                PortablePath  = $relativePath.Replace('\', '/')
                LegacyPath    = if (Test-PathTypeMatch -Path $legacyCandidate -PathType $PathType) { $legacyCandidate } else { $null }
                Resolution    = if (Test-PathTypeMatch -Path $legacyCandidate -PathType $PathType) { "current_repo_relative_fallback" } else { "current_repo_relative_only" }
                ShouldRemap   = $true
            }
        }

        return [pscustomobject]@{
            ConfigValue   = $ConfiguredPath
            RelativePath  = $null
            PortablePath  = $null
            LegacyPath    = $null
            Resolution    = "external_absolute"
            ShouldRemap   = $false
        }
    }

    $legacyCandidate = Join-Path $LegacyRoot $normalizedConfigured
    return [pscustomobject]@{
        ConfigValue   = $ConfiguredPath
        RelativePath  = $normalizedConfigured
        PortablePath  = $normalizedConfigured.Replace('\', '/')
        LegacyPath    = if (Test-PathTypeMatch -Path $legacyCandidate -PathType $PathType) { $legacyCandidate } else { $null }
        Resolution    = if (Test-PathTypeMatch -Path $legacyCandidate -PathType $PathType) { "legacy_relative" } else { "relative_missing" }
        ShouldRemap   = $false
    }
}

function Set-ConfigValue {
    param(
        [psobject]$ConfigObject,
        [string]$Key,
        [string]$Value
    )

    if ($ConfigObject.PSObject.Properties[$Key]) {
        $ConfigObject.$Key = $Value
    } else {
        $ConfigObject | Add-Member -NotePropertyName $Key -NotePropertyValue $Value
    }
}

function Update-MigratedConfigPaths {
    param(
        [string]$ConfigPath,
        [System.Collections.Generic.List[object]]$PathMappings
    )

    if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
        return
    }

    try {
        $configObject = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
        foreach ($mapping in $PathMappings) {
            if (-not $mapping.ShouldRemap -or [string]::IsNullOrWhiteSpace($mapping.NewValue)) {
                continue
            }
            Set-ConfigValue -ConfigObject $configObject -Key $mapping.Key -Value $mapping.NewValue
            Add-Result "PASS" ("Config path remap: {0}" -f $mapping.Key) ("Updated to {0}" -f $mapping.NewValue)
        }
        $json = $configObject | ConvertTo-Json -Depth 10
        Write-Utf8NoBomFile -Path $ConfigPath -Content ($json + [Environment]::NewLine)
    } catch {
        Add-Result "WARN" "Config path remapping" ("Could not update migrated config paths: {0}" -f $_.Exception.Message)
    }
}

function Confirm-OverwriteIfNeeded {
    param(
        [System.Collections.Generic.List[object]]$PlannedTargets
    )

    $existingTargets = @($PlannedTargets | ForEach-Object { $_.TargetPath } | Where-Object {
        -not [string]::IsNullOrWhiteSpace($_) -and (Test-Path -LiteralPath $_)
    } | Select-Object -Unique)

    if ($existingTargets.Count -eq 0) {
        return $true
    }

    Write-Host ""
    Write-Host "Warning: the migration will overwrite files or merge into existing target directories." -ForegroundColor Yellow
    Write-Host "Existing targets detected:"
    foreach ($path in $existingTargets | Select-Object -First 8) {
        Write-Host (" - {0}" -f $path)
    }
    if ($existingTargets.Count -gt 8) {
        Write-Host (" - ... and {0} more" -f ($existingTargets.Count - 8))
    }

    $response = (Read-Host "Continue with migration? (Y/N)").Trim().ToUpperInvariant()
    return $response -eq "Y"
}

$legacyRoot = Select-LegacyFolder -InitialFolder $ProjectRoot
if ([string]::IsNullOrWhiteSpace($legacyRoot)) {
    Write-Host "No legacy directory selected. Exiting."
    Wait-ForExitPrompt
    return
}

$resolvedLegacyRoot = (Resolve-Path -LiteralPath $legacyRoot).Path
$resolvedProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path

if ($resolvedLegacyRoot -eq $resolvedProjectRoot) {
    Add-Result "WARN" "Legacy directory selection" "Selected directory is the current repository root. No migration was performed."
    Write-ResultSummary
    Wait-ForExitPrompt
    return
}

Add-Result "PASS" "Legacy directory selection" ("Migrating from {0}" -f $resolvedLegacyRoot)
Add-Result "PASS" "Target repository" ("Writing into {0}" -f $resolvedProjectRoot)

$legacyConfigPath = Join-Path $resolvedLegacyRoot "podcast_rag_config.json"
$legacyConfig = $null
if (Test-Path -LiteralPath $legacyConfigPath -PathType Leaf) {
    try {
        $legacyConfig = Get-Content -LiteralPath $legacyConfigPath -Raw | ConvertFrom-Json
    } catch {
        Add-Result "WARN" "Legacy config parsing" ("Could not parse legacy config at {0}: {1}" -f $legacyConfigPath, $_.Exception.Message)
    }
}

$configSpecs = @(
    [pscustomobject]@{ Key = "input_dir"; Label = "Input transcript directory"; Type = "Directory"; Default = "data"; CopyMode = "progress"; Optional = $true },
    [pscustomobject]@{ Key = "processed_dir"; Label = "Processed transcript archive"; Type = "Directory"; Default = "processed"; CopyMode = "merge"; Optional = $true },
    [pscustomobject]@{ Key = "processed_data_dir"; Label = "Processed cache directory"; Type = "Directory"; Default = "processed_data"; CopyMode = "merge"; Optional = $false },
    [pscustomobject]@{ Key = "state_path"; Label = "Pipeline state file"; Type = "File"; Default = "state/podcast_rag_state.json"; CopyMode = "file"; Optional = $false },
    [pscustomobject]@{ Key = "stop_file"; Label = "Stop-after-current-file request"; Type = "File"; Default = "state/stop_after_current.txt"; CopyMode = "skip"; Optional = $true },
    [pscustomobject]@{ Key = "control_file"; Label = "Pipeline control file"; Type = "File"; Default = "state/pipeline_control.json"; CopyMode = "file"; Optional = $true },
    [pscustomobject]@{ Key = "debug_output_dir"; Label = "Debug output directory"; Type = "Directory"; Default = "debug_output"; CopyMode = "merge"; Optional = $true },
    [pscustomobject]@{ Key = "checkpoint_dir"; Label = "File checkpoint directory"; Type = "Directory"; Default = "state/file_checkpoints"; CopyMode = "merge"; Optional = $true },
    [pscustomobject]@{ Key = "run_report_dir"; Label = "Run report directory"; Type = "Directory"; Default = "state/run_reports"; CopyMode = "merge"; Optional = $true },
    [pscustomobject]@{ Key = "run_snapshot_path"; Label = "Current run snapshot"; Type = "File"; Default = "state/current_run_snapshot.json"; CopyMode = "file"; Optional = $true },
    [pscustomobject]@{ Key = "topic_contribution_dir"; Label = "Topic contribution directory"; Type = "Directory"; Default = "state/topic_contributions"; CopyMode = "merge"; Optional = $true },
    [pscustomobject]@{ Key = "topic_index_path"; Label = "Topic index file"; Type = "File"; Default = "state/topic_index.json"; CopyMode = "file"; Optional = $true },
    [pscustomobject]@{ Key = "topic_index_manifest_path"; Label = "Topic index manifest"; Type = "File"; Default = "state/topic_index_manifest.json"; CopyMode = "file"; Optional = $true },
    [pscustomobject]@{ Key = "model_eval_output_dir"; Label = "Model evaluation output directory"; Type = "Directory"; Default = "model_eval"; CopyMode = "merge"; Optional = $true }
)

$migrationPlans = New-Object System.Collections.Generic.List[object]
$configPathMappings = New-Object System.Collections.Generic.List[object]
$plannedTargets = New-Object System.Collections.Generic.List[object]
$plannedTargets.Add([pscustomobject]@{ TargetPath = (Join-Path $resolvedProjectRoot "podcast_rag_config.json") }) | Out-Null

foreach ($spec in $configSpecs) {
    $configuredValue = Get-ConfigValueOrDefault -ConfigObject $legacyConfig -Key $spec.Key -DefaultValue $spec.Default
    $resolved = Resolve-ConfiguredPath -ConfiguredPath $configuredValue -LegacyRoot $resolvedLegacyRoot -CurrentProjectRoot $resolvedProjectRoot -PathType $spec.Type
    $targetPath = if (-not [string]::IsNullOrWhiteSpace($resolved.RelativePath)) {
        Join-Path $resolvedProjectRoot $resolved.RelativePath
    } else {
        $null
    }

    $plan = [pscustomobject]@{
        Key        = $spec.Key
        Label      = $spec.Label
        Type       = $spec.Type
        CopyMode   = $spec.CopyMode
        Optional   = $spec.Optional
        Configured = $configuredValue
        Resolved   = $resolved
        TargetPath = $targetPath
    }
    $migrationPlans.Add($plan) | Out-Null

    if ($targetPath) {
        $plannedTargets.Add([pscustomobject]@{ TargetPath = $targetPath }) | Out-Null
    }

    if ($resolved.ShouldRemap -and -not [string]::IsNullOrWhiteSpace($resolved.PortablePath)) {
        $configPathMappings.Add([pscustomobject]@{
            Key        = $spec.Key
            NewValue   = $resolved.PortablePath
            ShouldRemap = $true
        }) | Out-Null
    }
}

if (-not (Confirm-OverwriteIfNeeded -PlannedTargets $plannedTargets)) {
    Add-Result "WARN" "Migration confirmation" "User cancelled migration after overwrite warning."
    Write-ResultSummary
    Wait-ForExitPrompt
    return
}

if (Test-Path -LiteralPath $legacyConfigPath -PathType Leaf) {
    Copy-FileWithBackup -SourcePath $legacyConfigPath -TargetPath (Join-Path $resolvedProjectRoot "podcast_rag_config.json") -Label "Runtime config"
} else {
    Add-Result "WARN" "Runtime config" "Legacy runtime config was not found."
}

foreach ($plan in $migrationPlans) {
    if ($plan.CopyMode -eq "skip") {
        if ($plan.Resolved.LegacyPath) {
            Add-Result "WARN" $plan.Label ("Skipped stale stop-request file from legacy directory: {0}" -f $plan.Resolved.LegacyPath)
        }
        continue
    }

    if (-not $plan.Resolved.LegacyPath) {
        switch ($plan.Resolved.Resolution) {
            "external_absolute" {
                Add-Result "WARN" $plan.Label ("Configured path is outside the legacy directory and was not copied: {0}" -f $plan.Configured)
            }
            "relative_missing" {
                Add-Result "WARN" $plan.Label ("Configured path was not found under the legacy directory: {0}" -f $plan.Configured)
            }
            "current_repo_relative_only" {
                Add-Result "WARN" $plan.Label ("Path was remapped to the new repo layout, but the matching legacy content was not found: {0}" -f $plan.Configured)
            }
            default {
                if (-not $plan.Optional) {
                    Add-Result "WARN" $plan.Label ("Legacy content was not found for configured path: {0}" -f $plan.Configured)
                }
            }
        }
        continue
    }

    switch ($plan.CopyMode) {
        "file" {
            Copy-FileWithBackup -SourcePath $plan.Resolved.LegacyPath -TargetPath $plan.TargetPath -Label $plan.Label
        }
        "merge" {
            Copy-DirectoryMerge -SourcePath $plan.Resolved.LegacyPath -TargetPath $plan.TargetPath -Label $plan.Label
        }
        "progress" {
            if ($plan.Resolved.Resolution -eq "external_absolute") {
                Add-Result "WARN" $plan.Label ("Configured input directory is outside the legacy directory and was not copied: {0}" -f $plan.Configured)
            } else {
                Copy-DirectoryWithProgress -SourcePath $plan.Resolved.LegacyPath -TargetPath $plan.TargetPath -Label $plan.Label
            }
        }
    }
}

Update-MigratedConfigPaths -ConfigPath (Join-Path $resolvedProjectRoot "podcast_rag_config.json") -PathMappings $configPathMappings

Write-ResultSummary
Wait-ForExitPrompt
