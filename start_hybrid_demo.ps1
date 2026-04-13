Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "Virtual environment not found at .venv. Create it first."
    exit 1
}

Write-Host "Starting hybrid demo backend on http://127.0.0.1:5050 ..."
& $python ".\run_hci_server.py"
