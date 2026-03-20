@echo off
setlocal

set "ROOT=%~dp0.."
set "CREDS=%ROOT%\config\theta_creds.txt"

if not exist "%CREDS%" (
  echo Missing credentials file: %CREDS%
  echo Run scripts\setup_theta_terminal.cmd first.
  exit /b 1
)

findstr /C:"YOUR_THETA_EMAIL" "%CREDS%" >nul 2>nul
if %ERRORLEVEL% EQU 0 (
  echo Credentials file still contains template placeholders.
  echo Edit %CREDS% and replace the two lines with your Theta email and password.
  exit /b 1
)

where java >nul 2>nul
if errorlevel 1 (
  echo Java is not available on PATH.
  exit /b 1
)

set "THETA_JAR="
for %%F in ("%USERPROFILE%\Downloads\ThetaTerminalv3.jar" "%USERPROFILE%\Downloads\ThetaTerminalv3 (1).jar") do (
  if exist "%%~fF" set "THETA_JAR=%%~fF"
)

if not defined THETA_JAR (
  for /f "delims=" %%F in ('powershell -NoProfile -Command "Get-ChildItem \"$env:USERPROFILE\Downloads\" -Filter \"ThetaTerminal*.jar\" ^| Sort-Object LastWriteTime -Descending ^| Select-Object -First 1 -ExpandProperty FullName"') do (
    set "THETA_JAR=%%F"
  )
)

if not defined THETA_JAR (
  echo Could not find ThetaTerminal JAR in %USERPROFILE%\Downloads
  exit /b 1
)

echo Launching Theta Terminal with %THETA_JAR%
java -jar "%THETA_JAR%" --creds-file "%CREDS%"
set "CODE=%ERRORLEVEL%"
echo Theta Terminal exited with code %CODE%

endlocal & exit /b %CODE%
