<#  Sync manifests + metrics from GCS into results\ for local inspection.
    The paper's tables are generated from these files, never hand-typed.  #>
param([string]$Bucket = "innate-portal-501710-n1-spark")

$root = Split-Path -Parent $PSScriptRoot
$dest = Join-Path $root "results"

foreach ($prefix in @("manifests", "metrics")) {
    $exists = gcloud storage ls "gs://$Bucket/$prefix/" 2>$null
    if ($exists) {
        gcloud storage rsync -r "gs://$Bucket/$prefix" (Join-Path $dest $prefix)
    } else {
        Write-Host "skip: gs://$Bucket/$prefix (not created yet)" -ForegroundColor DarkGray
    }
}

Write-Host "`nManifests:" -ForegroundColor Cyan
Get-ChildItem (Join-Path $dest "manifests\*.json") | Sort-Object Name |
    Select-Object Name, Length | Format-Table -AutoSize