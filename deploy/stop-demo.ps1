# Stops the proxy and the tunnel started by start-demo.ps1.
#
# Does not stop LM Studio (quit it from its own window) and does not touch the
# Streamlit Cloud deployment, which keeps running — it will simply have no model
# to talk to until the tunnel is back up.
#
#   .\deploy\stop-demo.ps1

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$pidFile = Join-Path $root "deploy\.run\pids.txt"

if (Test-Path $pidFile) {
    foreach ($line in Get-Content $pidFile) {
        $procId = $line.Trim()
        if (-not $procId) { continue }
        $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
        if ($proc) {
            Stop-Process -Id $procId -Force
            Write-Host "[stopped] $($proc.ProcessName) (pid $procId)"
        }
    }
    Remove-Item $pidFile -Force
}

# Belt and braces: catch anything left holding the ports, e.g. a run that was
# started by hand rather than by start-demo.ps1.
foreach ($port in 1235) {
    Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
        ForEach-Object {
            Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
            Write-Host "[stopped] process holding port $port (pid $($_.OwningProcess))"
        }
}
Get-Process cloudflared -ErrorAction SilentlyContinue | ForEach-Object {
    Stop-Process -Id $_.Id -Force
    Write-Host "[stopped] cloudflared (pid $($_.Id))"
}

Write-Host ""
Write-Host "Proxy and tunnel are down. LM Studio and the Cloud app are untouched."
Write-Host "The deployed app will fail until you run start-demo.ps1 and update the"
Write-Host "LLM_BASE_URL secret with the new hostname."
