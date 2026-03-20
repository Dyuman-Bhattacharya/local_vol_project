#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import getpass
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAUNCHER = PROJECT_ROOT / "scripts" / "run_canonical_daily_update_task.cmd"
DEFAULT_LOG = PROJECT_ROOT / "output" / "canonical_daily" / "daily_task.log"


def load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Config must be a mapping: {path}")
    return data


def quote_windows_arg(value: str) -> str:
    return '"' + value.replace('"', '\\"') + '"'


def write_launcher(config_path: Path, launcher_path: Path = DEFAULT_LAUNCHER, log_path: Path = DEFAULT_LOG) -> Path:
    launcher_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rel_cfg = config_path.relative_to(PROJECT_ROOT)
    lines = [
        "@echo off",
        "setlocal",
        f'cd /d "{PROJECT_ROOT}"',
        f'if not exist "{log_path.parent}" mkdir "{log_path.parent}"',
        f'set "TASK_LOG={log_path}"',
        'set "PYTHON_EXE="',
        f'if exist "{PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"}" set "PYTHON_EXE={PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"}"',
        'if not defined PYTHON_EXE (',
        '  for /f "usebackq delims=" %%I in (`poetry env info --path 2^>nul`) do set "POETRY_ENV=%%I"',
        '  if defined POETRY_ENV if exist "%POETRY_ENV%\\Scripts\\python.exe" set "PYTHON_EXE=%POETRY_ENV%\\Scripts\\python.exe"',
        ')',
        f'if not defined PYTHON_EXE if exist "{Path(sys.executable).resolve()}" set "PYTHON_EXE={Path(sys.executable).resolve()}"',
        'echo ==== %date% %time% task start ====>> "%TASK_LOG%"',
        'if defined PYTHON_EXE (',
        f'  "%PYTHON_EXE%" {quote_windows_arg(str(PROJECT_ROOT / "scripts" / "run_canonical_daily_update.py"))} --config {quote_windows_arg(str(rel_cfg))} >> "%TASK_LOG%" 2>&1',
        ') else (',
        f'  py -3 {quote_windows_arg(str(PROJECT_ROOT / "scripts" / "run_canonical_daily_update.py"))} --config {quote_windows_arg(str(rel_cfg))} >> "%TASK_LOG%" 2>&1',
        ')',
        'set "EXITCODE=%ERRORLEVEL%"',
        'echo ==== %date% %time% task end code %EXITCODE% ====>> "%TASK_LOG%"',
        "exit /b %EXITCODE%",
    ]
    launcher_path.write_text("\r\n".join(lines) + "\r\n", encoding="ascii")
    return launcher_path


def build_task_command(config_path: Path, launcher_path: Path = DEFAULT_LAUNCHER) -> str:
    launcher = write_launcher(config_path, launcher_path=launcher_path)
    return str(launcher)


def build_base_task_xml(task_name: str, launcher_command: str, snapshot_time: str) -> str:
    username = getpass.getuser()
    domain = os.environ.get("USERDOMAIN", "")
    full_user = f"{domain}\\{username}" if domain else username
    start_boundary = dt.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT") + f"{snapshot_time}:00"
    cmd = r"C:\Windows\System32\cmd.exe"
    args = f'/c "{launcher_command}"'
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Author>{full_user}</Author>
    <URI>\\{task_name}</URI>
  </RegistrationInfo>
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>{start_boundary}</StartBoundary>
      <Enabled>true</Enabled>
      <ScheduleByDay>
        <DaysInterval>1</DaysInterval>
      </ScheduleByDay>
    </CalendarTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>{full_user}</UserId>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>true</WakeToRun>
    <ExecutionTimeLimit>PT4H</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{cmd}</Command>
      <Arguments>{args}</Arguments>
      <WorkingDirectory>{PROJECT_ROOT}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""


def build_powershell_install_script(task_name: str, command: str, snapshot_time: str, force: bool) -> str:
    return f"""
$ErrorActionPreference = 'Stop'
$action = New-ScheduledTaskAction -Execute '{command}'
$trigger = New-ScheduledTaskTrigger -Daily -At '{snapshot_time}'
$settings = New-ScheduledTaskSettingsSet `
  -StartWhenAvailable `
  -WakeToRun `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries `
  -MultipleInstances IgnoreNew `
  -ExecutionTimeLimit (New-TimeSpan -Hours 4) `
  -RestartCount 3 `
  -RestartInterval (New-TimeSpan -Minutes 15)
$principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
$task = New-ScheduledTask -Action $action -Trigger $trigger -Principal $principal -Settings $settings
Register-ScheduledTask -TaskName '{task_name}' -InputObject $task -Force | Out-Null
"""


def build_powershell_install_script_s4u(task_name: str, command: str, snapshot_time: str, force: bool) -> str:
    username = getpass.getuser()
    domain = os.environ.get("USERDOMAIN", "")
    full_user = f"{domain}\\{username}" if domain else username
    return f"""
$ErrorActionPreference = 'Stop'
$action = New-ScheduledTaskAction -Execute '{command}'
$trigger = New-ScheduledTaskTrigger -Daily -At '{snapshot_time}'
$settings = New-ScheduledTaskSettingsSet `
  -StartWhenAvailable `
  -WakeToRun `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries `
  -MultipleInstances IgnoreNew `
  -ExecutionTimeLimit (New-TimeSpan -Hours 4) `
  -RestartCount 3 `
  -RestartInterval (New-TimeSpan -Minutes 15)
$principal = New-ScheduledTaskPrincipal -UserId '{full_user}' -LogonType S4U -RunLevel Limited
$task = New-ScheduledTask -Action $action -Trigger $trigger -Principal $principal -Settings $settings
Register-ScheduledTask -TaskName '{task_name}' -InputObject $task -Force | Out-Null
"""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Install a robust Windows Task Scheduler job for the 15:45 ET SPY snapshot update.")
    ap.add_argument("--config", default="config/daily_collection.yaml")
    ap.add_argument("--task_name", default="LocalVolProject-SPY-1545")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--run_now", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def main() -> None:
    if sys.platform != "win32":
        raise RuntimeError("This installer only supports Windows Task Scheduler.")

    args = parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    cfg = load_config(config_path)
    snapshot_time = str(cfg.get("collection", {}).get("snapshot_time", "15:45"))

    task_cmd = build_task_command(config_path)
    base_xml = build_base_task_xml(args.task_name, task_cmd, snapshot_time)

    print("Task command:")
    print(quote_windows_arg(task_cmd))
    print("")
    print("Task settings:")
    print("- preferred account: SYSTEM")
    print("- fallback account: current user via S4U")
    print("- start when available: true")
    print("- wake to run: true")
    print("- allow on battery: true")
    print("- restart on failure: 3 times every 15 minutes")

    if args.dry_run:
        with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, encoding="utf-16") as tmp:
            tmp.write(base_xml)
            xml_path = Path(tmp.name)
        print("")
        print("Base schtasks command (XML-backed):")
        print(f'schtasks /Create /TN "{args.task_name}" /XML "{xml_path}" /F')
        print("")
        print(base_xml)
        print("")
        print(build_powershell_install_script(args.task_name, task_cmd, snapshot_time, args.force))
        return

    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, encoding="utf-16") as tmp:
        tmp.write(base_xml)
        xml_path = Path(tmp.name)
    try:
        subprocess.run(
            [
                "schtasks",
                "/Create",
                "/TN",
                args.task_name,
                "/XML",
                str(xml_path),
                "/F",
            ],
            check=True,
        )
        print("Installed base daily task from XML.")
    finally:
        try:
            xml_path.unlink(missing_ok=True)
        except Exception:
            pass

    hardened = False
    for script_builder, label in [
        (build_powershell_install_script, "SYSTEM"),
        (build_powershell_install_script_s4u, "S4U"),
    ]:
        ps_script = script_builder(args.task_name, task_cmd, snapshot_time, args.force)
        with tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=False, encoding="utf-8") as tmp:
            tmp.write(ps_script)
            tmp_path = Path(tmp.name)
        try:
            subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(tmp_path),
                ],
                check=True,
            )
            print(f"Installed scheduled task using {label} principal.")
            hardened = True
            if args.run_now:
                subprocess.run(["schtasks", "/Run", "/TN", args.task_name], check=True)
            break
        except subprocess.CalledProcessError as e:
            print(f"{label} hardening attempt failed; leaving base task in place.", file=sys.stderr)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    if not hardened:
        print("Task remains installed with the base current-user scheduler settings.", file=sys.stderr)
    if args.run_now:
        subprocess.run(["schtasks", "/Run", "/TN", args.task_name], check=True)


if __name__ == "__main__":
    main()
