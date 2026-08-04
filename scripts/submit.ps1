param(
  [Parameter(Mandatory=$true)][string]$Script,
  [string]  $Cluster    = "bda-dev",
  [string]  $Properties = "",
  [string[]]$JobArgs    = @()
)

$root = Split-Path -Parent $PSScriptRoot
Push-Location $root

$sha = (git rev-parse --short HEAD 2>$null)
if (-not $sha) { $sha = "nogit" }
Write-Host "git_sha = $sha" -ForegroundColor Cyan

# s5_scalability imports the real S2/S3 logic, so those modules ship too.
$gcloudArgs = @(
  "dataproc", "jobs", "submit", "pyspark", $Script,
  "--cluster=$Cluster",
  "--region=us-east1",
  "--py-files=config.py,src/manifest.py,src/s2_features.py,src/s3_train.py"
)
if ($Properties -ne "") { $gcloudArgs += "--properties=$Properties" }
$gcloudArgs += @("--", "--git-sha=$sha") + $JobArgs

& gcloud @gcloudArgs

Pop-Location