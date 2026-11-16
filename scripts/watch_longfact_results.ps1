param(
    [string[]]$Runs = @('tiny-baseline', '8b-baseline')
)

# Event-driven continuation, not remote polling. One result wakes the assistant;
# it can re-arm this command after reviewing the available work.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$watchers = @()
$subscriptions = @()
$runByDirectory = @{}

try {
    foreach ($run in $Runs) {
        $directory = [System.IO.Path]::GetFullPath((Join-Path $root "scoring\$run"))
        $runByDirectory[$directory] = $run
        $watcher = New-Object System.IO.FileSystemWatcher
        $watcher.Path = $directory
        $watcher.Filter = 'longfact-*.json'
        $watcher.IncludeSubdirectories = $false
        $watcher.NotifyFilter = [System.IO.NotifyFilters]::LastWrite -bor [System.IO.NotifyFilters]::FileName
        foreach ($eventName in @('Created', 'Changed')) {
            $identifier = "longfact-$run-$eventName-$([guid]::NewGuid())"
            $null = Register-ObjectEvent -InputObject $watcher -EventName $eventName -SourceIdentifier $identifier
            $subscriptions += $identifier
        }
        $watcher.EnableRaisingEvents = $true
        $watchers += $watcher
    }
    Write-Output 'LongFact result notification armed; waiting for a filesystem event, not polling.'
    while ($true) {
        $event = Wait-Event
        $path = $event.SourceEventArgs.FullPath
        Remove-Event -EventIdentifier $event.EventIdentifier
        if (-not $path -or -not [System.IO.File]::Exists($path)) { continue }
        try {
            $record = [System.IO.File]::ReadAllText($path) | ConvertFrom-Json
        } catch {
            # Writes can produce an early event before the complete JSON exists.
            # Wait for the next filesystem event; do not repeatedly read/sleep.
            continue
        }
        if ($record.scorer_version -ne 'longfact-evidence-claims-v4-reviewed') { continue }
        if ($record.status -notin @('ok', 'needs_evidence_review', 'grading_error', 'retrieval_incomplete')) { continue }
        # Applying an assistant receipt rewrites an already handled result.
        # Do not wake again for our own completed-review write.
        if ($record.status -eq 'ok' -and $null -ne $record.evidence_review) { continue }
        $directory = [System.IO.Path]::GetDirectoryName($path)
        if (-not $runByDirectory.ContainsKey($directory)) { continue }
        @{
            continuation = 'CONTINUE_BASELINE_WORK_NOW'
            run = $runByDirectory[$directory]
            id = $record.id
            status = $record.status
            claims = $record.claim_counts
            path = $path
            message = 'A result landed; review pending evidence, recover failures, update saved results, and re-arm notifications.'
        } | ConvertTo-Json -Depth 6
        break
    }
} finally {
    foreach ($identifier in $subscriptions) {
        Unregister-Event -SourceIdentifier $identifier -ErrorAction SilentlyContinue
    }
    foreach ($watcher in $watchers) { $watcher.Dispose() }
}
