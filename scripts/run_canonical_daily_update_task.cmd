@echo off
setlocal
cd /d "C:\Users\dyuma\Projects\local_vol_project"
if not exist "C:\Users\dyuma\Projects\local_vol_project\output\canonical_daily" mkdir "C:\Users\dyuma\Projects\local_vol_project\output\canonical_daily"
set "TASK_LOG=C:\Users\dyuma\Projects\local_vol_project\output\canonical_daily\daily_task.log"
set "PYTHON_EXE="
if exist "C:\Users\dyuma\Projects\local_vol_project\.venv\Scripts\python.exe" set "PYTHON_EXE=C:\Users\dyuma\Projects\local_vol_project\.venv\Scripts\python.exe"
if not defined PYTHON_EXE (
  for /f "usebackq delims=" %%I in (`poetry env info --path 2^>nul`) do set "POETRY_ENV=%%I"
  if defined POETRY_ENV if exist "%POETRY_ENV%\Scripts\python.exe" set "PYTHON_EXE=%POETRY_ENV%\Scripts\python.exe"
)
if not defined PYTHON_EXE if exist "C:\Users\dyuma\AppData\Local\pypoetry\Cache\virtualenvs\local-vol-project-tu6VQQ12-py3.13\Scripts\python.exe" set "PYTHON_EXE=C:\Users\dyuma\AppData\Local\pypoetry\Cache\virtualenvs\local-vol-project-tu6VQQ12-py3.13\Scripts\python.exe"
echo ==== %date% %time% task start ====>> "%TASK_LOG%"
if defined PYTHON_EXE (
  "%PYTHON_EXE%" "C:\Users\dyuma\Projects\local_vol_project\scripts\run_canonical_daily_update.py" --config "config\daily_collection.yaml" >> "%TASK_LOG%" 2>&1
) else (
  py -3 "C:\Users\dyuma\Projects\local_vol_project\scripts\run_canonical_daily_update.py" --config "config\daily_collection.yaml" >> "%TASK_LOG%" 2>&1
)
set "EXITCODE=%ERRORLEVEL%"
echo ==== %date% %time% task end code %EXITCODE% ====>> "%TASK_LOG%"
exit /b %EXITCODE%
