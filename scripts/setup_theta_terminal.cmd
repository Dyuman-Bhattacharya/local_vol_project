@echo off
setlocal

set "ROOT=%~dp0.."
set "CREDS=%ROOT%\config\theta_creds.txt"
set "TEMPLATE=%ROOT%\config\theta_creds.example.txt"

if not exist "%CREDS%" (
  copy /Y "%TEMPLATE%" "%CREDS%" >nul
)

echo Opened %CREDS%
echo Put your Theta email on line 1 and your Theta password on line 2, then save the file.
start "" notepad "%CREDS%"

endlocal
