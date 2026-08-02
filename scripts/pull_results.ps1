<#  Sync manifests + metrics from GCS into results\ for local inspection.
    The paper's tables are generated from these files, never hand-typed.  #>
param([string]$Bucket = "innate-portal-501710-n1-spark")

$root = Split-Path -Parent $PSScriptRoot
$dest = Join-Path $root "results"

gcloud storage rsync -r "gs://$Bucket/manifests" "$dest\manifests"
gcloud storage rsync -r "gs://$Bucket/metrics"   "$dest\metrics"

Write-Host "`nManifests:" -ForegroundColor Cyan
Get-ChildItem "$dest\manifests\*.json" | Sort-Object Name |
    Select-Object Name, Length | Format-Table -AutoSize