from __future__ import annotations

import os
import platform


_PATCHED = False


def apply_windows_platform_fastpath() -> None:
    """
    Avoid slow or hanging WMI-based platform probes on Windows during SciPy import.

    Python 3.13 on Windows can route platform.machine()/uname() through WMI.
    Some subprocess contexts in this repo's integration tests stall there while
    importing NumPy/SciPy. The pricing and calibration code does not need the
    full dynamic platform probe, so we provide a stable environment-based answer.
    """
    global _PATCHED
    if _PATCHED or os.name != "nt":
        return

    arch = (
        os.environ.get("PROCESSOR_ARCHITEW6432")
        or os.environ.get("PROCESSOR_ARCHITECTURE")
        or "AMD64"
    )
    system = "Windows"
    node = os.environ.get("COMPUTERNAME", "")

    def _machine() -> str:
        return str(arch)

    def _processor() -> str:
        return str(arch)

    def _uname():
        return platform.uname_result(
            system=system,
            node=node,
            release="",
            version="",
            machine=str(arch),
            processor=str(arch),
        )

    platform.machine = _machine
    platform.processor = _processor
    platform.uname = _uname
    _PATCHED = True
