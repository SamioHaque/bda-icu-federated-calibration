<#  S5 sweep driver.

    Every point runs in ONE cluster session: identical work measured on two
    freshly created clusters differed by 28% in earlier runs, so cluster
    churn would swamp the effect.

    Executor count is requested per job. The cluster must have capacity for
    the largest value -- a 2-worker cluster provides 4 executors, so 1/2/4
    are valid and 8 would stall waiting for containers that never arrive.
#>
param(
  [int[]]    $Executors = @(1, 2, 4),
  [double[]] $Fractions = @(0.25, 1.0),
  [string]   $Cluster   = "bda-dev",
  [switch]   $SkipS2,
  [switch]   $SkipS3
)

$root = Split-Path -Parent $PSScriptRoot
$submit = Join-Path $PSScriptRoot "submit.ps1"

function Run-Point($stage, $execs, $frac, $model) {
  $props = "spark.executor.instances=$execs"
  $jobArgs = @("--stage=$stage", "--fraction=$frac",
               "--expect-executors=$execs", "--exec-label=$execs")
  if ($stage -eq "s3") { $jobArgs += "--model=$model" }

  Write-Host ""
  Write-Host ("=== {0} | executors {1} | fraction {2} | {3} ===" -f `
              $stage, $execs, $frac, $model) -ForegroundColor Yellow
  & $submit -Script "src\s5_scalability.py" -Cluster $Cluster `
            -Properties $props -JobArgs $jobArgs
}

if (-not $SkipS2) {
  foreach ($f in $Fractions) { foreach ($e in $Executors) {
    Run-Point "s2" $e $f "na"
  } }
}

if (-not $SkipS3) {
  foreach ($f in $Fractions) { foreach ($e in $Executors) {
    foreach ($m in @("lr", "gbt")) { Run-Point "s3" $e $f $m }
  } }
}

Write-Host ""
Write-Host "Sweep complete. Results appended to gs://.../metrics/scalability/" -ForegroundColor Green