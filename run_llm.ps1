<#
.SYNOPSIS
  Check requirements, start Ollama, and run the LLM layer (ZEIT3101 Group 9).

.DESCRIPTION
  1. Python 3.12 launcher present
  2. Virtual environment at %USERPROFILE%\.venvs\zeit3101 (created if missing;
     kept outside OneDrive so package files do not sync)
  3. Python packages (installed from requirements.txt + llm\requirements.txt if missing)
  4. Ollama installed, server running (started hidden if not), model pulled
  5. Full check: data/results present, models load and match results\
  6. Run the requested command

  Never writes to the pipeline's data or results.
  Layout note: this launcher now lives inside the pipeline folder (previously it sat
  beside a separate V0.0.3 folder). $Workspace is therefore the pipeline root.

.EXAMPLE
  .\run_llm.ps1                                   # demo brief (road 53-1006-3)
  .\run_llm.ps1 brief --road 1-4126-1 --date 1993-03-30 --show-hindsight
  .\run_llm.ps1 -Model llama3.1:8b brief --road 1-4126-1 --date 1993-03-30
  .\run_llm.ps1 -NoLLM brief --road 1-4126-1 --date 1993-03-30   # template only, no Ollama
  .\run_llm.ps1 list --split test
  .\run_llm.ps1 test                              # pytest
  .\run_llm.ps1 eval                              # 20-road evaluation with -Model
  .\run_llm.ps1 check                             # checks only
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Model = "qwen3:8b",
    [switch]$NoLLM,
    [switch]$SkipInstall,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs
)

$ErrorActionPreference = "Stop"
$Workspace = $PSScriptRoot
$VenvDir = Join-Path $HOME ".venvs\zeit3101"
$Py = Join-Path $VenvDir "Scripts\python.exe"
$OllamaUrl = "http://127.0.0.1:11434"

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore::UserWarning"
$env:LLM_MODEL = $Model

function Step($msg) { Write-Host "`n== $msg" -ForegroundColor Cyan }
function Die($msg)  { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

# Run a native command; return only its exit code (output goes to the console, and
# PowerShell 5.1 is stopped from treating stderr lines as terminating errors).
function Invoke-Native {
    param([string]$Exe, [string[]]$Arguments)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Exe @Arguments 2>&1 | ForEach-Object { "$_" } | Out-Host
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prev
    return $code
}

Push-Location $Workspace
try {
    if (-not $CliArgs -or $CliArgs.Count -eq 0) {
        $CliArgs = @("brief", "--road", "53-1006-3", "--date", "2003-04-16", "--show-hindsight")
    }
    $Command = $CliArgs[0]
    $NeedOllama = (-not $NoLLM) -and ($Command -in @("brief", "eval", "check"))

    # 1. Python ---------------------------------------------------------------
    Step "Python"
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        Die "Python launcher 'py' not found. Install Python 3.12 from python.org."
    }
    if ((Invoke-Native py @("-3.12", "--version")) -ne 0) {
        Die "Python 3.12 not installed (py -3.12 failed). Install it from python.org."
    }

    # 2. Virtual environment ------------------------------------------------------
    Step "Virtual environment ($VenvDir)"
    if (-not (Test-Path $Py)) {
        Write-Host "Creating venv..."
        if ((Invoke-Native py @("-3.12", "-m", "venv", $VenvDir)) -ne 0) { Die "venv creation failed" }
    } else {
        Write-Host "Found."
    }

    # 3. Packages -----------------------------------------------------------------
    Step "Python packages"
    $pkgCode = Invoke-Native $Py @("-m", "llm.check_env", "--packages-only")
    if ($pkgCode -ne 0) {
        if ($SkipInstall) { Die "Packages missing and -SkipInstall given." }
        Write-Host "Installing requirements..."
        $null = Invoke-Native $Py @("-m", "pip", "install", "-q", "--upgrade", "pip")
        $code = Invoke-Native $Py @("-m", "pip", "install", "-q",
            "-r", (Join-Path $Workspace "requirements.txt"),
            "-r", (Join-Path $Workspace "llm\requirements.txt"))
        if ($code -ne 0) { Die "pip install failed" }
        if ((Invoke-Native $Py @("-m", "llm.check_env", "--packages-only")) -ne 0) {
            Die "Packages still missing after install."
        }
    }

    # 4. Ollama -------------------------------------------------------------------
    if ($NeedOllama) {
        Step "Ollama"
        $ollama = (Get-Command ollama -ErrorAction SilentlyContinue).Source
        if (-not $ollama) {
            $candidate = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
            if (Test-Path $candidate) { $ollama = $candidate }
        }
        if (-not $ollama) {
            Die "Ollama not installed. Get it from https://ollama.com/download, or use -NoLLM."
        }

        $running = $false
        try { $null = Invoke-RestMethod "$OllamaUrl/api/version" -TimeoutSec 2; $running = $true } catch {}
        if (-not $running) {
            Write-Host "Starting Ollama server..."
            Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden
            for ($i = 0; $i -lt 30 -and -not $running; $i++) {
                Start-Sleep -Seconds 1
                try { $null = Invoke-RestMethod "$OllamaUrl/api/version" -TimeoutSec 2; $running = $true } catch {}
            }
            if (-not $running) { Die "Ollama server did not start within 30 s." }
        }
        $ver = (Invoke-RestMethod "$OllamaUrl/api/version").version
        Write-Host "Server running (version $ver)."

        $tags = (Invoke-RestMethod "$OllamaUrl/api/tags").models | ForEach-Object { $_.name }
        if ($tags -notcontains $Model) {
            Write-Host "Pulling $Model (one-off download)..."
            & $ollama pull $Model   # direct, so the progress bar renders
            if ($LASTEXITCODE -ne 0) { Die "ollama pull $Model failed (check the model name)" }
        } else {
            Write-Host "$Model available."
        }
    }

    # 5. Full check ---------------------------------------------------------------
    Step "Requirement check"
    $checkArgs = @("-m", "llm.check_env", "--model", $Model)
    if (-not $NeedOllama) { $checkArgs += "--no-ollama" }
    if ((Invoke-Native $Py $checkArgs) -ne 0) { Die "Requirement check failed (see [FAIL] lines above)." }

    # 6. Run ----------------------------------------------------------------------
    $rest = @()
    if ($CliArgs.Count -gt 1) { $rest = $CliArgs[1..($CliArgs.Count - 1)] }
    switch ($Command) {
        "check" { Write-Host "`nAll checks passed." -ForegroundColor Green; exit 0 }
        "test" {
            Step "Tests"
            exit (Invoke-Native $Py (@("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider") + $rest))
        }
        "eval" {
            Step "Evaluation ($Model)"
            if (-not (Test-Path (Join-Path $Workspace "llm\eval\sample.csv"))) {
                $null = Invoke-Native $Py @("-m", "llm.eval.run_eval", "sample")
            }
            $evalArgs = @("-m", "llm.eval.run_eval", "run", "--models", $Model)
            exit (Invoke-Native $Py ($evalArgs + $rest))
        }
        default {
            Step "llm.cli $($CliArgs -join ' ')"
            $cli = @("-m", "llm.cli") + $CliArgs
            if ($Command -eq "brief") {
                if ($NoLLM) { $cli += "--no-llm" } else { $cli += @("--model", $Model) }
            }
            exit (Invoke-Native $Py $cli)
        }
    }
}
finally {
    Pop-Location
}
