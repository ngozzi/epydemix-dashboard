# Resilient launcher for the EpyScenario dashboard (Windows PowerShell).
#
# - Forces single-threaded native math (numpy/BLAS/numexpr) to avoid a class of
#   intermittent native segfaults when heavy math runs inside Streamlit's worker
#   threads.
# - Supervises the Streamlit process and restarts it automatically if it exits.
#
# Usage:  .\run_dashboard.ps1 [-Port 8501]

param([int]$Port = 8501)

Set-Location -Path $PSScriptRoot

$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:NUMEXPR_MAX_THREADS = "1"
$env:VECLIB_MAXIMUM_THREADS = "1"

$py = ".\venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

while ($true) {
    & $py -m streamlit run Dashboard.py --server.port $Port --server.headless true
    Write-Warning "[supervisor] streamlit exited (code $LASTEXITCODE) - restarting in 2s..."
    Start-Sleep -Seconds 2
}
