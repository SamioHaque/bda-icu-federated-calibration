param([string]$Name = "bda-dev")
gcloud dataproc clusters delete $Name --region=us-east1 --quiet