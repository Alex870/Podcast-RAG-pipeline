param(
    [string]$Config = "",
    [string]$CondaEnvName = "podcast-rag-pipeline",
    [ValidateSet("Menu", "Process", "Status", "Control", "Create")]
    [string]$Mode = "Menu"
)

function Exit-Script { param([int]$Code = 0) exit $Code }
trap { Write-Error $_; Exit-Script 1 }

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonScript = Join-Path $ProjectRoot "podcast_rag_pipeline.py"
$RunScript = Join-Path $ProjectRoot "scripts\Run-PodcastRagPipeline.ps1"
if (-not $Config) { $Config = Join-Path $ProjectRoot "podcast_rag_config.json" }

function Invoke-PodcastRagCli {
    param([string[]]$Arguments)

    $output = @(& conda run --no-capture-output -n $CondaEnvName python -u $PythonScript --config $Config @Arguments 2>&1)
    $exitCode = $LASTEXITCODE
    [pscustomobject]@{
        ExitCode = $exitCode
        Output = @($output | ForEach-Object { [string]$_ })
    }
}

function Invoke-PodcastRagJson {
    param([string[]]$Arguments)

    $result = Invoke-PodcastRagCli -Arguments $Arguments
    if ($result.ExitCode -ne 0) {
        $result.Output | ForEach-Object { Write-Host $_ -ForegroundColor Red }
        return $null
    }

    $text = $result.Output -join [Environment]::NewLine
    try {
        return ($text | ConvertFrom-Json)
    } catch {
        Write-Host "The pipeline returned unexpected status output." -ForegroundColor Red
        $result.Output | ForEach-Object { Write-Host $_ }
        return $null
    }
}

function Invoke-PodcastRagChild {
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
        & $Path @Parameters | ForEach-Object { Write-Host ([string]$_) }
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

function Invoke-PartitionCommand {
    param([string[]]$Arguments)

    $result = Invoke-PodcastRagCli -Arguments (@("partitions") + $Arguments)
    $result.Output | ForEach-Object { Write-Host $_ }
    return $result.ExitCode
}

function Get-PartitionCatalog {
    return Invoke-PodcastRagJson -Arguments @("partitions", "list")
}

function Get-PartitionStatus {
    param([string]$PartitionId)
    return Invoke-PodcastRagJson -Arguments @("status", "--partition", $PartitionId)
}

function Get-PartitionRoot {
    param([string]$PartitionId)
    return Join-Path $ProjectRoot (Join-Path "partitions" $PartitionId)
}

function Get-PartitionEpisodes {
    param([object]$Status)

    $episodesByKey = @{}
    foreach ($handoff in @($Status.handoffs)) {
        if (-not $handoff.valid) { continue }
        $manifestPath = [string]$handoff.manifest
        if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { continue }
        try {
            $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
        } catch {
            continue
        }
        foreach ($episode in @($manifest.episodes)) {
            $key = [string]$episode.episode_uid
            if (-not $key) { $key = [string]$episode.episode_id }
            if (-not $key -or $episodesByKey.ContainsKey($key)) { continue }
            $episodesByKey[$key] = [pscustomobject]@{
                EpisodeId = [string]$episode.episode_id
                EpisodeUid = [string]$episode.episode_uid
                Title = [string]$episode.episode_title
                Date = [string]$episode.episode_date
                HandoffId = [string]$manifest.handoff_id
            }
        }
    }
    return @($episodesByKey.Values | Sort-Object Date, EpisodeId)
}

function Get-PartitionStateEntries {
    param([object]$Status)

    $entries = @()
    if ($null -eq $Status.files) { return $entries }
    foreach ($property in $Status.files.PSObject.Properties) {
        if ($null -ne $property.Value) { $entries += $property.Value }
    }
    return $entries
}

function Get-PartitionSummary {
    param([object]$Status)

    $episodes = @(Get-PartitionEpisodes -Status $Status)
    $stateByEpisode = @{}
    foreach ($entry in @(Get-PartitionStateEntries -Status $Status)) {
        $key = [string]$entry.episode_uid
        if (-not $key) { $key = [string]$entry.episode_id }
        if (-not $key) { continue }
        $previous = $stateByEpisode[$key]
        if ($null -eq $previous -or [string]$entry.updated_at -gt [string]$previous.updated_at) {
            $stateByEpisode[$key] = $entry
        }
    }

    $completed = 0
    $failed = 0
    $quarantined = 0
    foreach ($episode in $episodes) {
        $entry = $stateByEpisode[[string]$episode.EpisodeUid]
        if ($null -eq $entry) { $entry = $stateByEpisode[[string]$episode.EpisodeId] }
        switch ([string]$entry.status) {
            { $_ -in @("completed", "skipped") } { $completed++; break }
            { $_ -in @("failed", "interrupted") } { $failed++; break }
            "quarantined" { $quarantined++; break }
        }
    }

    $invalidHandoffs = @($Status.handoffs | Where-Object { -not $_.valid }).Count
    $pending = [Math]::Max(0, $episodes.Count - $completed - $failed - $quarantined)
    [pscustomobject]@{
        Episodes = $episodes.Count
        Completed = $completed
        Pending = $pending
        Failed = $failed
        Quarantined = $quarantined
        InvalidHandoffs = $invalidHandoffs
        Handoffs = @($Status.handoffs).Count
        StateEntries = @($Status.files.PSObject.Properties).Count
    }
}

function Get-PartitionRow {
    param(
        [object]$Catalog,
        [string]$PartitionId
    )
    return @($Catalog.partitions | Where-Object { [string]$_.partition_id -eq $PartitionId })[0]
}

function Show-PartitionSummary {
    param(
        [object]$Row,
        [object]$Status,
        [switch]$Detailed
    )

    $summary = Get-PartitionSummary -Status $Status
    $active = if ($Row.active) { "active" } else { "not active" }
    Write-Host ""
    Write-Host ("Partition: {0} ({1})" -f $Row.display_name, $active) -ForegroundColor Cyan
    Write-Host ("  ID: {0}" -f $Row.partition_id)
    Write-Host ("  Context: {0} / {1}" -f $Row.context_type, $Row.workflow_profile)
    Write-Host ("  Lifecycle: {0}" -f $Row.status)
    Write-Host ("  Handoff packages: {0}; declared episodes: {1}" -f $summary.Handoffs, $summary.Episodes)
    Write-Host ("  Completed/cached: {0}; pending: {1}; failed/interrupted: {2}; quarantined: {3}" -f $summary.Completed, $summary.Pending, $summary.Failed, $summary.Quarantined)
    if ($summary.InvalidHandoffs -gt 0) {
        Write-Host ("  Invalid handoff packages: {0}" -f $summary.InvalidHandoffs) -ForegroundColor Yellow
    }
    if ($Detailed) {
        Write-Host ("  Handoff inbox: {0}" -f $Row.handoff_inbox)
        Write-Host ("  Processed data: {0}" -f $Row.processed_data)
        Write-Host ("  State file: {0}" -f $Status.state_path)
        foreach ($handoff in @($Status.handoffs)) {
            $handoffStatus = if ($handoff.valid) { "valid" } else { "invalid" }
            Write-Host ("    {0}: {1}, episodes={2}" -f $handoff.handoff_id, $handoffStatus, $handoff.episode_count)
            if (-not $handoff.valid) {
                foreach ($error in @($handoff.errors | Select-Object -First 3)) {
                    Write-Host ("      {0}" -f $error.message) -ForegroundColor Yellow
                }
            }
        }
    }
    return $summary
}

function Show-PartitionDashboard {
    param([switch]$IncludeArchived)

    $catalog = Get-PartitionCatalog
    if ($null -eq $catalog) { return $null }
    $rows = @($catalog.partitions | Where-Object { $IncludeArchived -or $_.status -ne "archived" } | Sort-Object display_name)
    if ($rows.Count -eq 0) {
        Write-Host "No partitions are available." -ForegroundColor Yellow
        return $catalog
    }

    Write-Host ""
    Write-Host "Partitions" -ForegroundColor Cyan
    Write-Host "  #  Name                         Context    Status    Episodes  Done  Pending  Failed  Bad handoffs"
    Write-Host "  -- ---------------------------- ---------- --------- --------- ----- -------- ------- ------------"
    $number = 1
    foreach ($row in $rows) {
        $status = Get-PartitionStatus -PartitionId $row.partition_id
        if ($null -eq $status) { continue }
        $summary = Get-PartitionSummary -Status $status
        $marker = if ($row.active) { "*" } else { " " }
        $displayName = [string]$row.display_name
        if ($displayName.Length -gt 28) { $displayName = $displayName.Substring(0, 28) }
        Write-Host ("  {0,2}{1} {2,-28} {3,-10} {4,-9} {5,9} {6,5} {7,8} {8,7} {9,12}" -f $number, $marker, $displayName, $row.context_type, $row.status, $summary.Episodes, $summary.Completed, $summary.Pending, $summary.Failed, $summary.InvalidHandoffs)
        $number++
    }
    Write-Host "  * active partition"
    return $catalog
}

function Select-Partition {
    param(
        [switch]$IncludeArchived,
        [switch]$ProcessableOnly
    )

    $catalog = Get-PartitionCatalog
    if ($null -eq $catalog) { return $null }
    $choices = @($catalog.partitions | Where-Object {
        ($IncludeArchived -or $_.status -ne "archived") -and
        (-not $ProcessableOnly -or ($_.status -ne "archived" -and -not $_.legacy))
    } | Sort-Object display_name)
    if ($choices.Count -eq 0) {
        Write-Host "No suitable partitions are available." -ForegroundColor Yellow
        return $null
    }

    Write-Host ""
    for ($index = 0; $index -lt $choices.Count; $index++) {
        $row = $choices[$index]
        $marker = if ($row.active) { " (active)" } else { "" }
        $archived = if ($row.status -eq "archived") { " [archived]" } else { "" }
        Write-Host ("  {0}. {1}{2}{3} - {4}" -f ($index + 1), $row.display_name, $marker, $archived, $row.context_type)
    }
    Write-Host "  Q. Cancel"
    $selectionInput = Read-Host "Choose a partition number, or press Enter for the active partition"
    $selection = if ($null -eq $selectionInput) { "" } else { $selectionInput.Trim() }
    if ($selection.ToUpperInvariant() -eq "Q") { return $null }
    if (-not $selection) {
        $active = @($choices | Where-Object { $_.active })
        if ($active.Count -eq 1) { return $active[0] }
        Write-Host "There is no active partition. Choose a number." -ForegroundColor Yellow
        return $null
    }
    $number = 0
    if (-not [int]::TryParse($selection, [ref]$number) -or $number -lt 1 -or $number -gt $choices.Count) {
        Write-Host "Please choose one of the displayed partition numbers." -ForegroundColor Yellow
        return $null
    }
    return $choices[$number - 1]
}

function Read-RequiredValue {
    param([string]$Prompt)
    while ($true) {
        $valueInput = Read-Host $Prompt
        $value = if ($null -eq $valueInput) { "" } else { $valueInput.Trim() }
        if ($value) { return $value }
        Write-Host "A value is required." -ForegroundColor Yellow
    }
}

function Read-YesNo {
    param(
        [string]$Prompt,
        [bool]$Default = $true
    )
    $suffix = if ($Default) { "[Y/n]" } else { "[y/N]" }
    while ($true) {
        $valueInput = Read-Host "$Prompt $suffix"
        $value = if ($null -eq $valueInput) { "" } else { $valueInput.Trim().ToLowerInvariant() }
        if (-not $value) { return $Default }
        if ($value -in @("y", "yes")) { return $true }
        if ($value -in @("n", "no")) { return $false }
        Write-Host "Please answer Y or N." -ForegroundColor Yellow
    }
}

function Get-SafeGeneratedId {
    param([string]$DisplayName)
    $id = $DisplayName.ToLowerInvariant() -replace "[^a-z0-9]+", "-"
    $id = $id.Trim("-")
    if (-not $id) { $id = "partition" }
    if ($id.Length -gt 80) { $id = $id.Substring(0, 80).TrimEnd("-") }
    return $id
}

function Get-UniqueGeneratedId {
    param(
        [string]$DisplayName,
        [object]$Catalog
    )
    $base = Get-SafeGeneratedId -DisplayName $DisplayName
    $existing = @($Catalog.partitions | ForEach-Object { [string]$_.partition_id })
    $candidate = $base
    $suffix = 2
    while ($existing -contains $candidate) {
        $tail = "-$suffix"
        $prefixLength = [Math]::Max(1, 80 - $tail.Length)
        $candidate = $base.Substring(0, [Math]::Min($prefixLength, $base.Length)).TrimEnd("-") + $tail
        $suffix++
    }
    return $candidate
}

function Read-ListValue {
    param([string]$Prompt)
    $rawInput = Read-Host $Prompt
    $raw = if ($null -eq $rawInput) { "" } else { $rawInput.Trim() }
    if (-not $raw) { return @() }
    return @($raw.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}

function Read-KeyValueList {
    param([string]$Prompt)
    $values = Read-ListValue -Prompt $Prompt
    foreach ($value in $values) {
        if ($value -notmatch "^[^=]+=.*$") {
            Write-Host "Ignoring '$value' because it is not in key=value form." -ForegroundColor Yellow
            continue
        }
        $value
    }
}

function Invoke-CreatePartitionFlow {
    $name = Read-RequiredValue -Prompt "Friendly partition name"
    $context = ""
    while ($context -notin @("podcast", "meeting", "custom")) {
        $context = (Read-Host "Context type (podcast, meeting, or custom)").Trim().ToLowerInvariant()
        if ($context -notin @("podcast", "meeting", "custom")) {
            Write-Host "Choose podcast, meeting, or custom." -ForegroundColor Yellow
        }
    }

    $profile = switch ($context) {
        "podcast" { "podcast" }
        "meeting" { "anonymous_meeting" }
        default { Read-RequiredValue -Prompt "Workflow profile" }
    }

    $catalog = Get-PartitionCatalog
    if ($null -eq $catalog) { return }
    $generatedId = Get-UniqueGeneratedId -DisplayName $name -Catalog $catalog
    Write-Host ""
    Write-Host "Generated partition ID: $generatedId" -ForegroundColor Cyan
    $id = $generatedId
    if (Read-YesNo -Prompt "Does the upstream handoff already define a different partition ID?" -Default $false) {
        $id = Read-RequiredValue -Prompt "Upstream partition ID"
    }

    $corpus = $id
    $inbox = ""
    $description = ""
    $owner = ""
    $tags = @()
    $privacy = @()
    $retention = @()
    $overrides = @()
    if (Read-YesNo -Prompt "Enter advanced metadata or path settings" -Default $false) {
        $corpusInput = (Read-Host "Corpus ID (press Enter to use $id)").Trim()
        if ($corpusInput) { $corpus = $corpusInput }
        $inbox = (Read-Host "Handoff inbox relative path (press Enter for the generated default)").Trim()
        $description = (Read-Host "Description (optional)").Trim()
        $owner = (Read-Host "Owner (optional)").Trim()
        $tags = Read-ListValue -Prompt "Tags, comma-separated (optional)"
        $privacy = Read-KeyValueList -Prompt "Privacy metadata, key=value, comma-separated (optional)"
        $retention = Read-KeyValueList -Prompt "Retention metadata, key=value, comma-separated (optional)"
        $overrides = Read-KeyValueList -Prompt "Approved processing overrides, key=value, comma-separated (optional)"
    }

    Write-Host ""
    Write-Host "Create partition with:" -ForegroundColor Cyan
    Write-Host "  Name: $name"
    Write-Host "  ID: $id"
    Write-Host "  Context/profile: $context / $profile"
    Write-Host "  Corpus ID: $corpus"
    if ($inbox) { Write-Host "  Handoff inbox: $inbox" }
    if (-not (Read-YesNo -Prompt "Create this partition" -Default $true)) { return }

    $arguments = @("partitions", "create", "--id", $id, "--name", $name, "--context-type", $context, "--workflow-profile", $profile, "--corpus-id", $corpus)
    if ($inbox) { $arguments += @("--handoff-inbox", $inbox) }
    if ($description) { $arguments += @("--description", $description) }
    if ($owner) { $arguments += @("--owner", $owner) }
    foreach ($tag in $tags) { $arguments += @("--tag", $tag) }
    foreach ($item in $privacy) { $arguments += @("--privacy", $item) }
    foreach ($item in $retention) { $arguments += @("--retention", $item) }
    foreach ($item in $overrides) { $arguments += @("--override", $item) }
    if (Read-YesNo -Prompt "Make this the active partition" -Default $true) { $arguments += "--make-active" }

    $result = Invoke-PodcastRagCli -Arguments $arguments
    $result.Output | ForEach-Object { Write-Host $_ }
    if ($result.ExitCode -ne 0) { return }

    Write-Host "Partition created successfully." -ForegroundColor Green
    if (Read-YesNo -Prompt "Process or resume this partition now" -Default $false) {
        Invoke-ProcessFlow -SelectedPartitionId $id
    }
}

function Show-PartitionDetailsFlow {
    $row = Select-Partition -IncludeArchived
    if ($null -eq $row) { return }
    $result = Invoke-PodcastRagCli -Arguments @("partitions", "show", "--id", $row.partition_id)
    $result.Output | ForEach-Object { Write-Host $_ }
    $status = Get-PartitionStatus -PartitionId $row.partition_id
    if ($null -ne $status) { [void](Show-PartitionSummary -Row $row -Status $status -Detailed) }
}

function Show-PartitionStatusFlow {
    $row = Select-Partition -IncludeArchived
    if ($null -eq $row) { return }
    $status = Get-PartitionStatus -PartitionId $row.partition_id
    if ($null -eq $status) { return }
    [void](Show-PartitionSummary -Row $row -Status $status -Detailed)
    Write-Host ""
    Write-Host "Recent episode state:" -ForegroundColor Cyan
    foreach ($entry in @(Get-PartitionStateEntries -Status $status | Sort-Object updated_at -Descending | Select-Object -First 12)) {
        Write-Host ("  {0}: {1}" -f ([IO.Path]::GetFileName([string]$entry.path)), $entry.status)
    }
}

function Show-LatestRunReportFlow {
    $row = Select-Partition -IncludeArchived
    if ($null -eq $row) { return }
    $reportDirectory = Join-Path (Get-PartitionRoot -PartitionId $row.partition_id) "state\run_reports"
    $report = Get-ChildItem -LiteralPath $reportDirectory -Filter "*.run_report.md" -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $report) {
        Write-Host "No run reports have been written for '$($row.display_name)'." -ForegroundColor Yellow
        return
    }
    Write-Host ""
    Write-Host ("Latest run report for {0}: {1}" -f $row.display_name, $report.Name) -ForegroundColor Cyan
    Get-Content -LiteralPath $report.FullName | ForEach-Object { Write-Host $_ }
}

function Select-Episode {
    param([object]$Status)
    $episodes = @(Get-PartitionEpisodes -Status $Status)
    if ($episodes.Count -eq 0) {
        Write-Host "No declared episodes are available." -ForegroundColor Yellow
        return $null
    }
    $stateByEpisode = @{}
    foreach ($entry in @(Get-PartitionStateEntries -Status $Status)) {
        $key = [string]$entry.episode_uid
        if (-not $key) { $key = [string]$entry.episode_id }
        if ($key) { $stateByEpisode[$key] = [string]$entry.status }
    }
    Write-Host ""
    for ($index = 0; $index -lt $episodes.Count; $index++) {
        $episode = $episodes[$index]
        $state = $stateByEpisode[[string]$episode.EpisodeUid]
        if (-not $state) { $state = $stateByEpisode[[string]$episode.EpisodeId] }
        if (-not $state) { $state = "pending" }
        $label = if ($episode.Title) { $episode.Title } else { $episode.EpisodeId }
        Write-Host ("  {0}. {1} [{2}]" -f ($index + 1), $label, $state)
    }
    Write-Host "  Q. Cancel"
    $selectionInput = Read-Host "Choose an episode number"
    $selection = if ($null -eq $selectionInput) { "" } else { $selectionInput.Trim() }
    if ($selection.ToUpperInvariant() -eq "Q") { return $null }
    $number = 0
    if (-not [int]::TryParse($selection, [ref]$number) -or $number -lt 1 -or $number -gt $episodes.Count) {
        Write-Host "Please choose one of the displayed episode numbers." -ForegroundColor Yellow
        return $null
    }
    return $episodes[$number - 1]
}

function Invoke-ProcessFlow {
    param([string]$SelectedPartitionId)

    $row = if ($SelectedPartitionId) {
        $catalog = Get-PartitionCatalog
        if ($null -eq $catalog) { return }
        Get-PartitionRow -Catalog $catalog -PartitionId $SelectedPartitionId
    } else {
        Select-Partition -ProcessableOnly
    }
    if ($null -eq $row) { return }
    if ($row.status -eq "archived" -or $row.legacy) {
        Write-Host "Archived or legacy partitions cannot use managed processing." -ForegroundColor Yellow
        return
    }

    $status = Get-PartitionStatus -PartitionId $row.partition_id
    if ($null -eq $status) { return }
    $summary = Show-PartitionSummary -Row $row -Status $status
    $lockPath = Join-Path (Get-PartitionRoot -PartitionId $row.partition_id) "state\active_run.lock"
    if (Test-Path -LiteralPath $lockPath -PathType Leaf) {
        Write-Host "This partition already has an active processing run:" -ForegroundColor Yellow
        Get-Content -LiteralPath $lockPath | ForEach-Object { Write-Host "  $_" }
        return
    }
    if ($summary.Episodes -eq 0) {
        Write-Host "No valid handoff episodes are ready for processing." -ForegroundColor Yellow
        return
    }

    Write-Host ""
    Write-Host "  A. Process / resume all pending work"
    Write-Host "  N. Process only the next pending episode"
    Write-Host "  O. Process one selected episode"
    Write-Host "  F. Force-rebuild one selected episode"
    Write-Host "  Q. Cancel"
    $operation = (Read-Host "Choose an operation").Trim().ToUpperInvariant()
    if ($operation -eq "Q") { return }

    $episode = $null
    $force = $false
    $oneFile = $operation -eq "N"
    if ($operation -in @("O", "F")) {
        $episode = Select-Episode -Status $status
        if ($null -eq $episode) { return }
        $force = $operation -eq "F"
        if ($force -and -not (Read-YesNo -Prompt ("Force-rebuild '{0}'? Existing cache and errata will be backed up first." -f $episode.EpisodeId) -Default $false)) {
            return
        }
    } elseif ($operation -notin @("A", "N")) {
        Write-Host "Please choose A, N, O, F, or Q." -ForegroundColor Yellow
        return
    }

    if (-not (Read-YesNo -Prompt ("Start processing for '{0}'" -f $row.display_name) -Default $true)) { return }
    $parameters = @{
        Config = $Config
        CondaEnvName = $CondaEnvName
        Managed = $true
        Partition = $row.partition_id
    }
    if ($null -ne $episode) { $parameters.Episode = $episode.EpisodeId }
    if ($oneFile) { $parameters.OneFile = $true }
    if ($force) { $parameters.ForceReprocess = $true }
    Write-Host ""
    Write-Host ("Starting managed processing for {0}. Completed caches and valid checkpoints will be reused." -f $row.display_name) -ForegroundColor Cyan
    $exitCode = Invoke-PodcastRagChild -Path $RunScript -Parameters $parameters
    if ($exitCode -eq 130) {
        Write-Host "Ctrl+C stop complete. State and reports were saved; exiting without confirmation." -ForegroundColor Green
        Exit-Script 130
    }
    if ($exitCode -ne 0) {
        Write-Host ("Processing exited with code {0}. Review the status and run report before retrying." -f $exitCode) -ForegroundColor Yellow
    } else {
        Write-Host "Processing finished." -ForegroundColor Green
    }
    $updatedStatus = Get-PartitionStatus -PartitionId $row.partition_id
    if ($null -ne $updatedStatus) { [void](Show-PartitionSummary -Row $row -Status $updatedStatus) }
}

function Write-Utf8NoBomFile {
    param([string]$Path, [string]$Content)
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Content, $encoding)
}

function Get-PartitionStateDirectory {
    param([string]$PartitionId)
    $path = Join-Path (Get-PartitionRoot -PartitionId $PartitionId) "state"
    if (-not (Test-Path -LiteralPath $path -PathType Container)) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
    }
    return $path
}

function Invoke-ControlFlow {
    $catalog = Get-PartitionCatalog
    if ($null -eq $catalog) { return }
    $running = @($catalog.partitions | Where-Object {
        $_.status -ne "archived" -and -not $_.legacy -and
        (Test-Path -LiteralPath (Join-Path (Get-PartitionRoot -PartitionId $_.partition_id) "state\active_run.lock") -PathType Leaf)
    })
    if ($running.Count -gt 0) {
        Write-Host "Running partition(s):" -ForegroundColor Cyan
        foreach ($row in $running) { Write-Host ("  {0} - {1}" -f $row.display_name, $row.partition_id) }
    }
    $row = Select-Partition -ProcessableOnly
    if ($null -eq $row) { return }
    $stateDirectory = Get-PartitionStateDirectory -PartitionId $row.partition_id
    $stopPath = Join-Path $stateDirectory "stop_after_current.txt"
    $controlPath = Join-Path $stateDirectory "pipeline_control.json"
    Write-Host ""
    Write-Host "  1. Request stop after the current episode"
    Write-Host "  2. Clear stop request"
    Write-Host "  3. Set max parallel model requests"
    Write-Host "  Q. Cancel"
    $choice = (Read-Host "Choose a control action").Trim().ToUpperInvariant()
    switch ($choice) {
        "1" {
            Write-Utf8NoBomFile -Path $stopPath -Content ("Stop requested at {0}{1}" -f (Get-Date -Format o), [Environment]::NewLine)
            Write-Host "Stop requested for $($row.display_name): $stopPath" -ForegroundColor Green
        }
        "2" {
            if (Test-Path -LiteralPath $stopPath) { Remove-Item -LiteralPath $stopPath -Force }
            Write-Host "Stop request cleared for $($row.display_name)." -ForegroundColor Green
        }
        "3" {
            $value = 0
            while ($value -lt 1) {
                $raw = (Read-Host "New max_parallel_model_requests").Trim()
                if (-not [int]::TryParse($raw, [ref]$value) -or $value -lt 1) {
                    Write-Host "Please enter an integer value of 1 or higher." -ForegroundColor Yellow
                    $value = 0
                }
            }
            $payload = [ordered]@{
                max_parallel_model_requests = $value
                updated_at = (Get-Date -Format o)
                note = "The running pipeline reloads this file before launching new model requests."
            }
            Write-Utf8NoBomFile -Path $controlPath -Content (($payload | ConvertTo-Json -Depth 5) + [Environment]::NewLine)
            Write-Host "Set max_parallel_model_requests=$value for $($row.display_name)." -ForegroundColor Green
        }
        "Q" { return }
        default { Write-Host "Please choose one of the displayed actions." -ForegroundColor Yellow }
    }
}

function Invoke-ArchiveFlow {
    $row = Select-Partition -IncludeArchived
    if ($null -eq $row) { return }
    if ($row.status -eq "archived") {
        if (-not (Read-YesNo -Prompt "Restore '$($row.display_name)'" -Default $true)) { return }
        [void](Invoke-PartitionCommand -Arguments @("archive", "--id", $row.partition_id, "--restore"))
    } else {
        if ($row.active) {
            Write-Host "Select another active partition before archiving this one." -ForegroundColor Yellow
            return
        }
        if (-not (Read-YesNo -Prompt "Archive '$($row.display_name)'" -Default $false)) { return }
        [void](Invoke-PartitionCommand -Arguments @("archive", "--id", $row.partition_id))
    }
}

function Invoke-SelectActiveFlow {
    $row = Select-Partition
    if ($null -eq $row) { return }
    [void](Invoke-PartitionCommand -Arguments @("use", "--id", $row.partition_id))
}

function Invoke-PartitionMenu {
    while ($true) {
        Write-Host ""
        Write-Host "Partition management" -ForegroundColor Cyan
        Write-Host "  1. View partition dashboard"
        Write-Host "  2. Create a partition"
        Write-Host "  3. Process / resume a partition"
        Write-Host "  4. View partition status"
        Write-Host "  5. Inspect partition details"
        Write-Host "  6. Select active/default partition"
        Write-Host "  7. Archive or restore a partition"
        Write-Host "  8. Validate partition setup"
        Write-Host "  9. Control a running partition"
        Write-Host "  10. View latest run report"
        Write-Host "  Q. Back"
        $selection = (Read-Host "Choose an action").Trim().ToUpperInvariant()
        switch ($selection) {
            "1" { [void](Show-PartitionDashboard -IncludeArchived) }
            "2" { Invoke-CreatePartitionFlow }
            "3" { Invoke-ProcessFlow }
            "4" { Show-PartitionStatusFlow }
            "5" { Show-PartitionDetailsFlow }
            "6" { Invoke-SelectActiveFlow }
            "7" { Invoke-ArchiveFlow }
            "8" { [void](Invoke-PartitionCommand -Arguments @("doctor")) }
            "9" { Invoke-ControlFlow }
            "10" { Show-LatestRunReportFlow }
            "Q" { return }
            default { Write-Host "Please choose one of the displayed actions." -ForegroundColor Yellow }
        }
    }
}

switch ($Mode) {
    "Menu" { Invoke-PartitionMenu }
    "Process" { Invoke-ProcessFlow }
    "Status" { Show-PartitionStatusFlow }
    "Control" { Invoke-ControlFlow }
    "Create" { Invoke-CreatePartitionFlow }
}

Exit-Script 0
