param(
    [string]$ProcessedDataDir
)

$ProjectRoot = $PSScriptRoot
if (-not $ProcessedDataDir) {
    $configPath = Join-Path $ProjectRoot "podcast_rag_config.json"
    $examplePath = Join-Path $ProjectRoot "podcast_rag_config.example.json"
    $configToRead = if (Test-Path -LiteralPath $configPath) { $configPath } else { $examplePath }
    $config = Get-Content -LiteralPath $configToRead -Raw | ConvertFrom-Json
    $ProcessedDataDir = if ($config.processed_data_dir) { [string]$config.processed_data_dir } else { "processed_data" }
}

if (-not [System.IO.Path]::IsPathRooted($ProcessedDataDir)) {
    $ProcessedDataDir = Join-Path $ProjectRoot $ProcessedDataDir
}

if (-not (Test-Path -LiteralPath $ProcessedDataDir)) {
    Write-Host "Processed data directory does not exist: $ProcessedDataDir"
    exit 0
}

$patterns = @(
    "please provide the podcast transcript",
    "please provide the transcript",
    "please provide the source text",
    "once shared",
    "once you share",
    "i'll generate",
    "i can summarize"
)

$badFiles = New-Object System.Collections.Generic.HashSet[string]
foreach ($file in Get-ChildItem -LiteralPath $ProcessedDataDir -Filter "*.json" -File) {
    foreach ($pattern in $patterns) {
        $matches = Select-String -LiteralPath $file.FullName -Pattern $pattern -SimpleMatch -CaseSensitive:$false -ErrorAction SilentlyContinue
        if ($matches) {
            [void]$badFiles.Add($file.FullName)
            break
        }
    }
}

if ($badFiles.Count -eq 0) {
    Write-Host "No missing-context summaries found in $ProcessedDataDir" -ForegroundColor Green
    exit 0
}

Write-Host "Found missing-context summaries in $($badFiles.Count) processed data cache file(s):" -ForegroundColor Yellow
$badFiles | Sort-Object | ForEach-Object { Write-Host "  $_" }
Write-Host ""
Write-Host "Delete those cache file(s) and rerun the pipeline to rebuild them with the new validation."
exit 1

