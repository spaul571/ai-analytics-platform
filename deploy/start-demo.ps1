# Starts the two local processes the deployed app depends on:
#
#   1. deploy/llm_proxy.py  - bearer-token gate in front of LM Studio (port 1235)
#   2. cloudflared          - outbound tunnel giving that gate a public hostname
#
# It does NOT start LM Studio. Do that yourself first: load google/gemma-4-e4b
# and start the server on port 1234 (Developer tab -> Status: Running).
#
# Run from the repo root:
#   .\deploy\start-demo.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
$envFile = Join-Path $root ".env"
$stateDir = Join-Path $root "deploy\.run"
$tunnelLog = Join-Path $stateDir "tunnel.log"
$pidFile = Join-Path $stateDir "pids.txt"

New-Item -ItemType Directory -Force -Path $stateDir | Out-Null

# --- cloudflared ------------------------------------------------------------
# winget installs it here but does not add it to already-open shells' PATH.
$cloudflared = "C:\Program Files (x86)\cloudflared\cloudflared.exe"
if (-not (Test-Path $cloudflared)) {
    $onPath = Get-Command cloudflared -ErrorAction SilentlyContinue
    if ($onPath) { $cloudflared = $onPath.Source }
    else { throw "cloudflared not found. Install it: winget install --id Cloudflare.cloudflared" }
}

# --- token ------------------------------------------------------------------
# Reused across runs so the LLM_API_KEY already sitting in the Streamlit Cloud
# secrets keeps working. Only the hostname changes between runs, not the token.
$token = $null
if (Test-Path $envFile) {
    $match = Select-String -Path $envFile -Pattern '^PROXY_TOKEN=(.+)$'
    if ($match) { $token = $match.Matches[0].Groups[1].Value.Trim() }
}
if (-not $token) {
    $token = & $python -c "import secrets; print(secrets.token_urlsafe(32))"
    Add-Content -Path $envFile -Encoding utf8 -Value "`n# Bearer token required by deploy/llm_proxy.py. Must equal LLM_API_KEY in the`n# Streamlit Cloud secrets.`nPROXY_TOKEN=$token"
    Write-Host "Generated a new PROXY_TOKEN and wrote it to .env" -ForegroundColor Yellow
    Write-Host "You must update LLM_API_KEY in the Streamlit Cloud secrets to match." -ForegroundColor Yellow
}

# --- 1. LM Studio must already be up ---------------------------------------
try {
    Invoke-RestMethod -Uri "http://localhost:1234/v1/models" -TimeoutSec 5 | Out-Null
    Write-Host "[ok]   LM Studio is answering on :1234"
} catch {
    throw "LM Studio is not answering on http://localhost:1234. Start its server and load google/gemma-4-e4b first."
}

# --- 2. the auth proxy ------------------------------------------------------
$env:PROXY_TOKEN = $token   # inherited by the child process
$proxy = Start-Process -FilePath $python `
    -ArgumentList "-m", "uvicorn", "deploy.llm_proxy:app", "--port", "1235", "--host", "127.0.0.1" `
    -WorkingDirectory $root -WindowStyle Minimized -PassThru

Start-Sleep -Seconds 3
try {
    Invoke-RestMethod -Uri "http://127.0.0.1:1235/healthz" -TimeoutSec 5 | Out-Null
    Write-Host "[ok]   auth proxy is up on :1235"
} catch {
    throw "The proxy did not come up. Is port 1235 already in use?"
}

# --- 3. the tunnel ----------------------------------------------------------
Remove-Item $tunnelLog -ErrorAction SilentlyContinue
$tunnel = Start-Process -FilePath $cloudflared `
    -ArgumentList "tunnel", "--url", "http://localhost:1235", "--no-autoupdate" `
    -RedirectStandardError $tunnelLog -RedirectStandardOutput "$tunnelLog.out" `
    -WindowStyle Minimized -PassThru

"$($proxy.Id)`n$($tunnel.Id)" | Set-Content -Path $pidFile -Encoding utf8

# cloudflared prints the hostname to stderr a few seconds after connecting.
$hostname = $null
foreach ($i in 1..30) {
    Start-Sleep -Seconds 1
    if (Test-Path $tunnelLog) {
        $found = Select-String -Path $tunnelLog -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com'
        if ($found) { $hostname = $found.Matches[0].Value; break }
    }
}
if (-not $hostname) { throw "The tunnel did not report a hostname. See $tunnelLog" }

Write-Host "[ok]   tunnel is up"
Write-Host ""

# --- verify the whole chain from the public side ---------------------------
# The hostname exists before its DNS record has propagated, so the first few
# lookups legitimately fail with "remote name could not be resolved". Retry
# rather than reporting a working tunnel as broken.
$reachable = $false
foreach ($i in 1..20) {
    try {
        Invoke-RestMethod -Uri "$hostname/healthz" -TimeoutSec 10 | Out-Null
        $reachable = $true
        break
    } catch {
        Start-Sleep -Seconds 3
    }
}
if ($reachable) {
    Write-Host "[ok]   the public hostname resolves and answers"
} else {
    Write-Host "WARNING: $hostname did not answer. DNS may still be propagating; retry in a minute." -ForegroundColor Yellow
}

# An unauthenticated call must be refused. Windows PowerShell 5.1 raises on any
# non-2xx, so the 401 arrives as an exception and is read off the response.
# (-SkipHttpErrorCheck would be cleaner but is PowerShell 7 only.)
if ($reachable) {
    $unauth = $null
    try {
        $unauth = (Invoke-WebRequest -Uri "$hostname/v1/models" -TimeoutSec 20 -UseBasicParsing).StatusCode
    } catch {
        $unauth = [int]$_.Exception.Response.StatusCode
    }
    if ($unauth -eq 401) {
        Write-Host "[ok]   unauthenticated requests are refused (401)"
    } else {
        Write-Host "WARNING: an unauthenticated request returned $unauth, expected 401. The endpoint may be open." -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "=======================================================================" -ForegroundColor Green
Write-Host " Paste this into Streamlit Cloud -> App settings -> Secrets" -ForegroundColor Green
Write-Host " (the hostname is new every run; the token is not)" -ForegroundColor Green
Write-Host "=======================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "LLM_BASE_URL = `"$hostname/v1`""
Write-Host "LLM_MODEL    = `"google/gemma-4-e4b`""
Write-Host "LLM_API_KEY  = `"$token`""
Write-Host ""
Write-Host "Saving the secrets reboots the app. Then ask it a question."
Write-Host "Stop everything with:  .\deploy\stop-demo.ps1"
