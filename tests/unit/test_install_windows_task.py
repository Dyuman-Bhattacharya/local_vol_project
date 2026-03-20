from __future__ import annotations

import importlib.util
from pathlib import Path


def test_build_task_command_includes_runner_and_config() -> None:
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "install_windows_daily_snapshot_task.py"
    spec = importlib.util.spec_from_file_location("install_task", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    cfg_path = root / "config" / "daily_collection.yaml"
    cmd = module.build_task_command(cfg_path)
    assert "run_canonical_daily_update_task.cmd" in cmd
    launcher = root / "scripts" / "run_canonical_daily_update_task.cmd"
    assert launcher.exists()
    text = launcher.read_text(encoding="ascii")
    assert "run_canonical_daily_update.py" in text
    assert "config\\daily_collection.yaml" in text
    assert "TASK_LOG" in text
    assert "mkdir" in text
    assert "poetry env info --path" in text
    assert ".venv\\Scripts\\python.exe" in text
    assert "py -3" in text


def test_build_base_task_xml_uses_resilient_battery_settings() -> None:
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "install_windows_daily_snapshot_task.py"
    spec = importlib.util.spec_from_file_location("install_task", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    xml = module.build_base_task_xml("LocalVolProject-SPY-1545", r"C:\task.cmd", "15:45")
    assert "<DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>" in xml
    assert "<StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>" in xml
    assert "<StartWhenAvailable>true</StartWhenAvailable>" in xml
    assert "<WakeToRun>true</WakeToRun>" in xml
    assert "<Command>C:\\Windows\\System32\\cmd.exe</Command>" in xml


def test_build_powershell_script_uses_system_and_resilient_settings() -> None:
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "install_windows_daily_snapshot_task.py"
    spec = importlib.util.spec_from_file_location("install_task", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    ps = module.build_powershell_install_script("LocalVolProject-SPY-1545", r"C:\task.cmd", "15:45", True)
    assert "Unregister-ScheduledTask" not in ps
    assert "New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest" in ps
    assert "-StartWhenAvailable" in ps
    assert "-WakeToRun" in ps
    assert "-AllowStartIfOnBatteries" in ps
    assert "-DontStopIfGoingOnBatteries" in ps


def test_build_powershell_script_s4u_uses_current_user() -> None:
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "install_windows_daily_snapshot_task.py"
    spec = importlib.util.spec_from_file_location("install_task", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    ps = module.build_powershell_install_script_s4u("LocalVolProject-SPY-1545", r"C:\task.cmd", "15:45", True)
    assert "Unregister-ScheduledTask" not in ps
    assert "-LogonType S4U" in ps
    assert "-RunLevel Limited" in ps
    assert "Register-ScheduledTask" in ps
