param(
    [string]$Config = "",
    [string]$CondaEnvName = "podcast-rag-pipeline"
)

function Exit-Script { param([int]$Code = 0) exit $Code }
trap { Write-Error $_; Exit-Script 1 }

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonScript = Join-Path $ProjectRoot "podcast_rag_pipeline.py"
if (-not $Config) { $Config = Join-Path $ProjectRoot "podcast_rag_config.json" }

function Invoke-PartitionCli {
    param([string[]]$Arguments)
    & conda run --no-capture-output -n $CondaEnvName python $PythonScript --config $Config partitions @Arguments
    return $LASTEXITCODE
}

while ($true) {
    Write-Host ""
    Write-Host "Processing partitions"
    Write-Host "  1. List partitions"
    Write-Host "  2. Create a partition"
    Write-Host "  3. Show a partition"
    Write-Host "  4. Select active partition"
    Write-Host "  5. Archive a partition"
    Write-Host "  6. Validate partition setup"
    Write-Host "  Q. Done"
    $selection = (Read-Host "Choose an action").Trim().ToUpperInvariant()
    if ($selection -eq "Q") { Exit-Script 0 }
    switch ($selection) {
        "1" { [void](Invoke-PartitionCli @("list")) }
        "2" {
            $id = (Read-Host "Short ID (lowercase letters, digits, '-' or '_')").Trim()
            $name = (Read-Host "Display name").Trim()
            $context = (Read-Host "Context type (podcast, meeting, or custom)").Trim()
            $profile = (Read-Host "Workflow profile").Trim()
            $corpus = (Read-Host "Corpus ID (press Enter to use the partition ID)").Trim()
            $inbox = (Read-Host "Handoff inbox relative path (press Enter for the generated default)").Trim()
            $description = (Read-Host "Description (optional)").Trim()
            $owner = (Read-Host "Owner (optional)").Trim()
            $tags = (Read-Host "Tags, comma-separated (optional)").Trim()
            $privacy = (Read-Host "Privacy metadata, key=value (optional)").Trim()
            $retention = (Read-Host "Retention metadata, key=value (optional)").Trim()
            $arguments = @("create", "--id", $id, "--name", $name, "--context-type", $context, "--workflow-profile", $profile)
            if ($corpus) { $arguments += @("--corpus-id", $corpus) }
            if ($inbox) { $arguments += @("--handoff-inbox", $inbox) }
            if ($description) { $arguments += @("--description", $description) }
            if ($owner) { $arguments += @("--owner", $owner) }
            if ($tags) {
                foreach ($tag in $tags.Split(",")) {
                    if ($tag.Trim()) { $arguments += @("--tag", $tag.Trim()) }
                }
            }
            if ($privacy) { $arguments += @("--privacy", $privacy) }
            if ($retention) { $arguments += @("--retention", $retention) }
            [void](Invoke-PartitionCli $arguments)
        }
        "3" {
            $id = (Read-Host "Partition ID").Trim()
            [void](Invoke-PartitionCli @("show", "--id", $id))
        }
        "4" {
            $id = (Read-Host "Partition ID to make active").Trim()
            [void](Invoke-PartitionCli @("use", "--id", $id))
        }
        "5" {
            $id = (Read-Host "Partition ID to archive").Trim()
            [void](Invoke-PartitionCli @("archive", "--id", $id))
        }
        "6" { [void](Invoke-PartitionCli @("doctor")) }
        default { Write-Host "Please choose one of the displayed options." -ForegroundColor Yellow }
    }
}
