param(
  [string]$Name        = "bda-dev",
  [int]   $Workers     = 2,
  [string]$MachineType = "n2-standard-4",
  [int]   $ExecCores   = 2,
  [string]$ExecMemory  = "4g",
  [int]   $Shuffle     = 64,
  [string]$MaxIdle     = "30m"
)

$execs = $Workers * 2
$props = "spark:spark.dynamicAllocation.enabled=false," +
         "spark:spark.executor.instances=$execs," +
         "spark:spark.executor.cores=$ExecCores," +
         "spark:spark.executor.memory=$ExecMemory," +
         "spark:spark.sql.shuffle.partitions=$Shuffle"

gcloud dataproc clusters create $Name `
  --region=us-east1 --zone=us-east1-b `
  --image-version=2.2-debian12 `
  --master-machine-type=$MachineType --master-boot-disk-size=100 `
  --num-workers=$Workers --worker-machine-type=$MachineType --worker-boot-disk-size=100 `
  --bucket=innate-portal-501710-n1-spark `
  --max-idle=$MaxIdle `
  --enable-component-gateway `
  --properties=$props

Write-Host "Cluster '$Name': $Workers workers, $execs executors x $ExecCores cores, max-idle $MaxIdle." -ForegroundColor Green