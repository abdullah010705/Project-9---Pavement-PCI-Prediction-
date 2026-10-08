@echo off
rem Double-click or run from cmd: bypasses the PowerShell script execution policy for this script only.
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_llm.ps1" %*
if "%~1"=="" pause
