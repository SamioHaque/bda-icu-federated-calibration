param(
  [Parameter(Mandatory=$true)][string]$Script,
  [string]  $Cluster = "bda-dev",
  [string[]]$JobArgs = @()
)

$root = Split-Path -Parent $PSScriptRoot
Push-Location $root

$sha = (git rev-parse --short HEAD 2>$null)
if (-not $sha) { $sha = "nogit" }
Write-Host "git_sha = $sha" -ForegroundColor Cyan

$gcloudArgs = @(
  "dataproc", "jobs", "submit", "pyspark", $Script,
  "--cluster=$Cluster",
  "--region=us-east1",
  "--py-files=config.py,src/manifest.py",
  "--",
  "--git-sha=$sha"
) + $JobArgs

& gcloud @gcloudArgs

Pop-Location